#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""dep_gen capture + replay sim test.

Re-runs the ``vector_example`` orchestration with ``--enable-dep-gen``.
Verifies the end-to-end dep_gen pipeline on a2a3sim:

  ``<output_prefix>/deps.json`` is produced by the host replay
  (PTO2TensorMap replay → JSON edge list), and contains exactly the
  6 edges documented in example_orchestration.cpp. The capture path
  (host collector drains the device ring buffer into memory and feeds
  the replay directly — no submit_trace.bin on disk) is exercised
  implicitly: if it broke, deps.json would be empty or wrong.

deps.json is now the sole source of truth for fanout edges — the device
hot path no longer records L2SwimlaneAicpuTaskRecord::fanout[], so there is no
"fanout ⊆ deps" cross-check to run. swimlane_converter.py joins
deps.json into the Perfetto trace at post-process time.

Compute correctness is delegated to the upstream ``vector_example`` test —
this case re-uses the same orchestration to keep coverage focused on the
capture+replay+validation pipeline.
"""

import json
import shutil
import subprocess
import sys
import time

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename

KERNELS_BASE = "../../../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"


def _task_id(ring: int, local: int) -> int:
    """Encode (ring_id, local_id) → 64-bit raw matching ``PTO2TaskId::raw`` —
    keeps the bit layout (``(ring << 32) | local``) in one place rather than
    repeating ``1 << 32`` arithmetic at every call site.
    """
    return (ring << 32) | local


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestDepGen(SceneTestCase):
    """Vector example, run with dep_gen enabled, then verify generated deps."""

    CALLABLE = {
        "orchestration": {
            "source": f"{KERNELS_BASE}/orchestration/example_orchestration.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": f"{KERNELS_BASE}/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": f"{KERNELS_BASE}/aiv/kernel_add_scalar.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "source": f"{KERNELS_BASE}/aiv/kernel_mul.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "default",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 0},
            "params": {},
        },
    ]

    def generate_args(self, params):
        SIZE = 128 * 128
        return TaskArgsBuilder(
            Tensor("a", torch.full((SIZE,), 2.0, dtype=torch.float32)),
            Tensor("b", torch.full((SIZE,), 3.0, dtype=torch.float32)),
            Tensor("f", torch.zeros(SIZE, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        args.f[:] = (args.a + args.b + 1) * (args.a + args.b + 2) + (args.a + args.b)

    def test_run(self, st_platform, st_worker, request):
        # Run the standard scene-test loop, then assert dep_gen output for the
        # cases that actually ran on this platform. Without this override, the
        # pytest path silently passes when dep_gen is disabled in the AICPU
        # build (the trace ring stays empty and deps.json is just `{"edges":[]}`)
        # — the bug that prompted #742. Use the framework helper so the
        # rounds-guard stays consistent with SceneTestCase.test_run (super()
        # already warned, so warn=False here).
        # Marker taken before the run so _post_validate binds to this
        # invocation's output dir rather than a stale same-label leftover.
        run_marker = int(time.time())  # floor to whole seconds: safe if outputs/ ever lands on a coarse-mtime fs
        super().test_run(st_platform, st_worker, request)
        if not self._effective_enable_dep_gen(request):
            return
        for case in self.CASES:
            if st_platform in case.get("platforms", []):
                self._post_validate(case, run_marker)

    def _post_validate(self, case, run_marker):
        """Assert deps.json for this invocation contains the 6 edges documented
        in example_orchestration.cpp. Reached only with dep_gen effectively on
        and the case's platform matching, so the run must have produced an
        output dir; a missing one is a capture regression, not a skip.
        """
        case_name = case["name"]
        safe_label = _sanitize_for_filename(f"TestDepGen_{case_name}")
        outputs = _outputs_dir()
        # Only dirs created by this run (mtime past the pre-run marker); a stale
        # leftover must not stand in for a missing current-run output.
        matches = [p for p in outputs.glob(f"{safe_label}_*") if p.stat().st_mtime >= run_marker]
        assert matches, (
            f"--enable-dep-gen is on and case {case_name!r} ran, but no output dir "
            f"was created this run — capture/replay pipeline regression"
        )
        out_dir = max(matches, key=lambda p: p.stat().st_mtime)

        # ---- deps.json (host replay output — sole dep_gen artifact on disk) ----
        # We only reach here with --enable-dep-gen on and rounds<=1 (the
        # test_run gate via _effective_enable_dep_gen) AND an output dir present
        # (the case actually ran). deps.json MUST therefore have been produced;
        # its absence means the capture->reconcile->replay pipeline silently
        # produced nothing (reconcile drops or replay failure) — exactly the
        # regression this test exists to catch (#742). Fail loudly, don't skip.
        deps_path = out_dir / "deps.json"
        assert deps_path.exists(), (
            f"--enable-dep-gen is on and {out_dir} exists, but deps.json was not produced "
            f"— capture/reconcile/replay pipeline regression"
        )
        with deps_path.open() as f:
            deps = json.load(f)
        # Strided-Tensor schema: annotated edges with tasks[] / tensors[]
        # sidecars carrying strided slice descriptors (start_offset +
        # stride[]). Project annotated edges down to a (pred, succ) set for
        # the existing structural checks; the annotation sanity check below
        # verifies the tensor metadata path.
        raw_edges = deps.get("edges", [])
        deps_edges = set()
        for e in raw_edges:
            assert isinstance(e, dict), f"deps.json edge must be an object, got {type(e).__name__}: {e!r}"
            pred, succ = e.get("pred"), e.get("succ")
            if pred is None or succ is None:
                continue
            deps_edges.add((int(pred), int(succ)))

        # example_orchestration.cpp comment block (verified by tracing the source):
        #   t0: ring 0, local 0
        #   t1..t4: ring 1, local 0..3  (inner manual scope → ring 1)
        # Edges: t0->t1, t0->t2, t1->t3, t2->t3, t0->t4, t3->t4
        t0 = _task_id(0, 0)
        t1 = _task_id(1, 0)
        t2 = _task_id(1, 1)
        t3 = _task_id(1, 2)
        t4 = _task_id(1, 3)
        expected_edges = {(t0, t1), (t0, t2), (t1, t3), (t2, t3), (t0, t4), (t3, t4)}
        missing = expected_edges - deps_edges
        assert not missing, f"deps.json missing expected edges: {missing} (got {deps_edges})"
        # Allow extra edges (creator-retention may add owner edges that don't appear
        # in the comment's logical-dep view), but flag anything outside the task set.
        valid_ids = {t0, t1, t2, t3, t4}
        bad = {e for e in deps_edges if e[0] not in valid_ids or e[1] not in valid_ids}
        assert not bad, f"deps.json contains edges referencing unknown task ids: {bad}"

        # ---- Annotated-edge sanity ----
        # Replay always emits the tensor-info sidecar; the differential check
        # inside the replay would have failed the run before we got here if
        # the annotated pass disagreed with compute_task_fanin. These
        # assertions just confirm the schema actually carries the expected
        # blocks (so e.g. a future "always write empty arrays" bug would
        # surface here, not silently in a downstream viewer).
        tasks = deps.get("tasks", [])
        tensors = deps.get("tensors", [])
        task_ids = {int(t["task_id"]) for t in tasks if "task_id" in t}
        assert valid_ids <= task_ids, f"tasks[] missing expected ids: {valid_ids - task_ids}"
        # Every non-explicit edge should reference a tensor_id present in
        # tensors[]. EXPLICIT edges legitimately omit it.
        tensor_ids = {int(t["tensor_id"]) for t in tensors if "tensor_id" in t}
        for e in raw_edges:
            if not isinstance(e, dict):
                continue
            source = e.get("source")
            if source == "explicit":
                continue
            tid = e.get("tensor_id")
            assert tid is not None and int(tid) in tensor_ids, (
                f"edge {e.get('pred')}->{e.get('succ')} (source={source}) "
                f"references tensor_id {tid} absent from tensors[]"
            )
            # Annotated edges must carry consumer-side strided slice info.
            assert "consumer_shape" in e and "consumer_start_offset" in e and "consumer_strides" in e, (
                f"edge {e.get('pred')}->{e.get('succ')} (source={source}) missing consumer_shape/start_offset/strides"
            )

        # ---- Tool smoke: deps_viewer (text) ----
        # scene_test auto-generates deps_viewer.txt via _graph_case_dep_gen;
        # smoke verifies it was produced and has the expected sections.
        out_txt = out_dir / "deps_viewer.txt"
        assert out_txt.exists(), f"scene_test auto-hook did not produce {out_txt}"
        text = out_txt.read_text()
        assert "SUMMARY" in text and "TASK INDEX" in text, "text deps graph missing expected sections"

        for extra in (["--direction", "LR"], ["--engine", "dot"]):
            bad = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "simpler_setup.tools.deps_viewer",
                    str(deps_path),
                    "--format",
                    "text",
                    *extra,
                ],
                check=False,
                timeout=60,
                capture_output=True,
                text=True,
            )
            assert bad.returncode != 0, f"text mode should reject {' '.join(extra)}"
            assert "only valid with --format html" in bad.stderr

        if shutil.which("dot"):
            out_html = out_dir / "_smoke_deps.html"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "simpler_setup.tools.deps_viewer",
                    str(deps_path),
                    "--format",
                    "html",
                    "-o",
                    str(out_html),
                ],
                check=True,
                timeout=60,
            )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
