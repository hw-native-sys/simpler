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

Re-runs the ``vector_example`` orchestration with ``--enable-dep-gen`` (and,
in standalone mode, auto-adds ``--enable-l2-swimlane`` for the fanout ⊆ deps
gate). Verifies the end-to-end dep_gen pipeline on a2a3sim:

  1. ``<output_prefix>/deps.json`` is produced by the host replay
     (PTO2TensorMap replay → JSON edge list), and contains exactly the
     6 edges documented in example_orchestration.cpp. The capture path
     (host collector drains the device ring buffer into memory and feeds
     the replay directly — no submit_trace.bin on disk) is exercised
     implicitly: if it broke, deps.json would be empty or wrong.
  2. **Validation gate** (when l2_perf_records.json is present, i.e.
     ``--enable-l2-swimlane`` was also enabled): every edge in
     L2PerfRecord::fanout[] also appears in deps.json. deps may have
     MORE edges than fanout (race-window edges fanout missed); we never
     assert symmetry — that's the entire reason dep_gen exists.

Pytest entry: needs ``--enable-dep-gen`` (capture+replay assertions) and
``--enable-l2-swimlane`` (fanout ⊆ deps gate). Standalone entry: pass
``--enable-dep-gen`` and the swimlane flag is added automatically so a
plain ``python test_dep_gen_capture.py -p a2a3sim --enable-dep-gen``
exercises the full gate.

Compute correctness is delegated to the upstream ``vector_example`` test —
this case re-uses the same orchestration to keep coverage focused on the
capture+replay+validation pipeline.
"""

import json
import sys

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename

KERNELS_BASE = "../../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestDepGenCapture(SceneTestCase):
    """Vector example, run with dep_gen enabled, then verify submit_trace.bin."""

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
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
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
        # — the bug that prompted this PR. Standalone keeps its own validator.
        # Use the framework helper so the rounds-guard stays consistent with
        # SceneTestCase.test_run (super() already warned, so warn=False here).
        super().test_run(st_platform, st_worker, request)
        if not self._effective_enable_dep_gen(request):
            return
        for case in self.CASES:
            if st_platform in case.get("platforms", []):
                self._post_validate(case)

    def _post_validate(self, case):
        """Hook invoked after the case ran when --enable-dep-gen is in effect.

        Locates the per-case output_prefix directory and asserts:
          - deps.json exists (host collector → replay pipeline produced it)
            and contains the 6 edges documented in example_orchestration.cpp
          - if l2_perf_records.json is also present, every fanout edge it
            records is a subset of the deps.json edge set
        """
        case_name = case["name"]
        safe_label = _sanitize_for_filename(f"TestDepGenCapture_{case_name}")
        outputs = _outputs_dir()
        matches = sorted(outputs.glob(f"{safe_label}_*"), key=lambda p: p.stat().st_mtime)
        assert matches, f"no output prefix under {outputs} matching {safe_label}_*"
        out_dir = matches[-1]

        # ---- deps.json (host replay output — sole dep_gen artifact on disk) ----
        deps_path = out_dir / "deps.json"
        assert deps_path.exists(), f"deps.json not produced under {out_dir} — capture or replay failed?"
        with deps_path.open() as f:
            deps = json.load(f)
        assert deps.get("version") == 1, f"deps.json version {deps.get('version')} != 1"
        deps_edges = {(int(e[0]), int(e[1])) for e in deps.get("edges", []) if isinstance(e, list) and len(e) == 2}

        # example_orchestration.cpp comment block (verified by tracing the source):
        #   t0: ring 0, local 0
        #   t1..t4: ring 1, local 0..3  (inner manual scope → ring 1)
        # Edges: t0->t1, t0->t2, t1->t3, t2->t3, t0->t4, t3->t4
        t0 = 0
        t1 = 1 << 32
        t2 = (1 << 32) | 1
        t3 = (1 << 32) | 2
        t4 = (1 << 32) | 3
        expected_edges = {(t0, t1), (t0, t2), (t1, t3), (t2, t3), (t0, t4), (t3, t4)}
        missing = expected_edges - deps_edges
        assert not missing, f"deps.json missing expected edges: {missing} (got {deps_edges})"
        # Allow extra edges (creator-retention may add owner edges that don't appear
        # in the comment's logical-dep view), but flag anything outside the task set.
        valid_ids = {t0, t1, t2, t3, t4}
        bad = {e for e in deps_edges if e[0] not in valid_ids or e[1] not in valid_ids}
        assert not bad, f"deps.json contains edges referencing unknown task ids: {bad}"

        # ---- fanout ⊆ deps validation gate ----
        perf = out_dir / "l2_perf_records.json"
        if perf.exists():
            with perf.open() as f:
                pdata = json.load(f)
            fanout_edges = set()
            for task in pdata.get("tasks", []):
                src = int(task["task_id"])
                for succ in task.get("fanout", []):
                    fanout_edges.add((src, int(succ)))
            missing_in_deps = fanout_edges - deps_edges
            assert not missing_in_deps, (
                f"fanout ⊆ deps gate FAILED: edges present in l2_perf_records.json "
                f"fanout[] but absent from deps.json: {missing_in_deps}. "
                f"This is a replay-side regression — the replay should be a "
                f"superset of the runtime's fanout view."
            )


def _post_run_verify():
    """Iterate all cases and run _post_validate. Used by standalone main."""
    inst = TestDepGenCapture()
    for case in TestDepGenCapture.CASES:
        inst._post_validate(case)


if __name__ == "__main__":
    # The standalone entry exits with 0/1 from inside SceneTestCase.run_module,
    # so the verification has to happen *before* the exit, or we need to catch
    # the SystemExit and only do verification on success. Easier: catch.
    enable_dep_gen = "--enable-dep-gen" in sys.argv
    # Auto-add --enable-l2-swimlane when --enable-dep-gen is on so the fanout ⊆ deps
    # gate runs by default in standalone. Both flags compose cleanly; the gate is
    # the most informative thing the test produces, so don't make the user remember
    # to ask for it.
    if enable_dep_gen and "--enable-l2-swimlane" not in sys.argv:
        sys.argv.append("--enable-l2-swimlane")
    try:
        SceneTestCase.run_module(__name__)
    except SystemExit as e:
        if e.code not in (0, None):
            raise
        if enable_dep_gen:
            _post_run_verify()
            print("[dep_gen_capture] post-run verification PASSED")
        sys.exit(0)
