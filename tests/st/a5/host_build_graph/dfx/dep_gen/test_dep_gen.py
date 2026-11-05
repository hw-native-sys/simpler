#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""dep_gen host-direct capture test for a5 host_build_graph.

Runs the ``vector_example`` orchestration with ``--enable-dep-gen`` and asserts
the 4 creator edges it declares come out. host_build_graph orchestrates on the
host, so its graph is captured from the orchestrator's own dependency path
rather than replayed from a device ring; the device-orch answer for the same
orchestration is covered by the tensormap_and_ringbuffer dep_gen test, which is
what makes a divergence visible.

Compute correctness is delegated to the upstream ``vector_example`` test; this
case re-uses the orchestration purely to keep coverage on the capture pipeline.
"""

import json

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename

KERNELS_BASE = "../../vector_example/kernels"


def _output_dirs(test_cls_name, case_name):
    """Every output dir this case has ever produced, newest run included."""
    safe_label = _sanitize_for_filename(f"{test_cls_name}_{case_name}")
    return set(_outputs_dir().glob(f"{safe_label}_*"))


def _load_deps(test_cls_name, case_name, dirs_before_run):
    """Parse the deps.json under the output dir this invocation created.

    Identified by set difference against a pre-run snapshot rather than by
    timestamp: an mtime comparison floors to whole seconds, so a leftover dir
    from an earlier run in the same second also matches, and a run that emitted
    nothing would silently validate that stale deps.json.
    """
    fresh = _output_dirs(test_cls_name, case_name) - dirs_before_run
    assert fresh, (
        f"--enable-dep-gen is on and case {case_name!r} ran, but no output dir "
        f"was created this run — capture pipeline regression"
    )
    out_dir = max(fresh, key=lambda p: p.stat().st_mtime)
    deps_path = out_dir / "deps.json"
    assert deps_path.exists(), (
        f"--enable-dep-gen is on and {out_dir} exists, but deps.json was not produced — host-direct capture regression"
    )
    with deps_path.open() as f:
        return json.load(f)


def _edges_by_position(deps):
    """Project edges onto (submit_index(pred), submit_index(succ), source).

    tasks[] is in submit order, so position identifies a task independently of
    which ring the runtime placed it on.
    """
    position = {int(t["task_id"]): i for i, t in enumerate(deps["tasks"])}
    unknown = {
        (e["pred"], e["succ"])
        for e in deps["edges"]
        if int(e["pred"]) not in position or int(e["succ"]) not in position
    }
    assert not unknown, f"deps.json contains edges referencing unknown task ids: {unknown}"
    return {(position[int(e["pred"])], position[int(e["succ"])], e["source"]) for e in deps["edges"]}


def _assert_annotations(deps):
    """Every non-explicit edge names a registered tensor and its consumer slice."""
    tensor_ids = {int(t["tensor_id"]) for t in deps.get("tensors", []) if "tensor_id" in t}
    for e in deps["edges"]:
        assert isinstance(e, dict), f"deps.json edge must be an object, got {type(e).__name__}: {e!r}"
        if e.get("source") == "explicit":
            continue
        tid = e.get("tensor_id")
        assert tid is not None and int(tid) in tensor_ids, (
            f"edge {e.get('pred')}->{e.get('succ')} (source={e.get('source')}) "
            f"references tensor_id {tid} absent from tensors[]"
        )
        assert "consumer_shape" in e and "consumer_start_offset" in e and "consumer_strides" in e, (
            f"edge {e.get('pred')}->{e.get('succ')} (source={e.get('source')}) "
            f"missing consumer_shape/start_offset/strides"
        )


@scene_test(level=2, runtime="host_build_graph")
class TestDepGenHostBuildGraph(SceneTestCase):
    """Vector example on host_build_graph, run with dep_gen enabled."""

    CALLABLE = {
        "orchestration": {
            "source": f"{KERNELS_BASE}/orchestration/example_orch.cpp",
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
            "platforms": ["a5sim", "a5"],
            "manual": True,
            "params": {},
        },
    ]

    def generate_args(self, params):
        SIZE = 128 * 128
        return TaskArgsBuilder(
            TensorArg("a", torch.full((SIZE,), 2.0, dtype=torch.float32)),
            TensorArg("b", torch.full((SIZE,), 3.0, dtype=torch.float32)),
            TensorArg("f", torch.zeros(SIZE, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        args.f[:] = (args.a + args.b + 1) * (args.a + args.b + 2)

    def test_run(self, st_platform, st_worker, request):
        # Run the standard scene-test loop, then assert the captured graph for
        # the cases that ran on this platform. Without the override the pytest
        # path would pass while capture produced nothing. The snapshot is taken
        # before the run so _post_validate binds to a directory this invocation
        # created rather than a leftover from an earlier one.
        dirs_before_run = {
            case["name"]: _output_dirs("TestDepGenHostBuildGraph", case["name"])
            for case in self._matching_cases(st_platform, request)
        }
        super().test_run(st_platform, st_worker, request)
        if not self._effective_enable_dep_gen(request):
            return
        for case in self._matching_cases(st_platform, request):
            self._post_validate(case, dirs_before_run[case["name"]])

    def _post_validate(self, case, dirs_before_run):
        """Assert deps.json holds the 4 edges of example_orch.cpp."""
        deps = _load_deps("TestDepGenHostBuildGraph", case["name"], dirs_before_run)

        tasks = deps.get("tasks", [])
        assert len(tasks) == 4, f"expected 4 submitted tasks, got {len(tasks)}: {[t.get('task_id') for t in tasks]}"

        # example_orch.cpp, in submit order:
        #   t0: c = a + b   t1: d = c + 1   t2: e = c + 2   t3: f = d * e
        # Every edge is born in creator retention: the intermediates are
        # runtime-allocated OUTPUT tensors, which register_task_outputs does not
        # put in the tensormap, so Step B has nothing to match here.
        got = _edges_by_position(deps)
        expected = {
            (0, 1, "creator"),
            (0, 2, "creator"),
            (1, 3, "creator"),
            (2, 3, "creator"),
        }
        assert got == expected, (
            f"captured graph differs from the orchestration's dependencies: "
            f"missing={expected - got}, extra={got - expected}"
        )

        _assert_annotations(deps)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
