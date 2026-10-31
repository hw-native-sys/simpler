#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Graph Execution replays dynamic tensor and scalar bindings across config-keyed definitions."""

import json

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename


@scene_test(level=2, runtime="host_build_graph")
class TestGraphExecutionHostBuildGraph(SceneTestCase):
    RTOL = 1e-5
    ATOL = 1e-5

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/graph_execution_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT, D.OUT, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "../vector_example/kernels/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": "../vector_example/kernels/aiv/kernel_add_scalar.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "source": "../vector_example/kernels/aiv/kernel_mul.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "record_then_replay_1d",
            "platforms": ["a2a3sim", "a2a3"],
            "manual": ["a2a3sim"],
            "params": {"shape": (128 * 128,)},
        },
        {
            "name": "record_then_replay_2d",
            "platforms": ["a2a3sim", "a2a3"],
            "params": {"shape": (128 * 128, 1)},
        },
    ]

    def generate_args(self, params):
        shape = params["shape"]
        return TaskArgsBuilder(
            TensorArg("a", torch.full(shape, 2.0, dtype=torch.float32)),
            TensorArg("b", torch.full(shape, 3.0, dtype=torch.float32)),
            TensorArg("output_1", torch.zeros(shape, dtype=torch.float32)),
            TensorArg("output_3", torch.zeros(shape, dtype=torch.float32)),
            TensorArg("output_5", torch.zeros(shape, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        base = args.a + args.b
        args.output_1[:] = (base + 1.0) * (base + 2.0)
        args.output_3[:] = (base + 100.0) * (base + 4.0)
        args.output_5[:] = (base + 5.0) * (base + 6.0)

    def test_run(self, st_platform, st_worker, request):
        matched_cases = self._matching_cases(st_platform, request)
        prior_mtimes = {case["name"]: self._matching_output_mtimes(case) for case in matched_cases}
        super().test_run(st_platform, st_worker, request)
        if not self._effective_enable_dep_gen(request):
            return
        for case in matched_cases:
            self._validate_outer_graph_capture(case, prior_mtimes[case["name"]])

    def test_repeated_aicore_so_load(self, st_platform, st_worker):
        if st_platform != "a2a3sim":
            pytest.skip("AICore shared objects are reloaded only by the simulator")
        self._run_and_validate_l2(st_worker, self.build_callable(st_platform), self.CASES[1], rounds=180)

    @staticmethod
    def _matching_output_mtimes(case):
        safe_label = _sanitize_for_filename(f"TestGraphExecutionHostBuildGraph_{case['name']}")
        return {path: path.stat().st_mtime_ns for path in _outputs_dir().glob(f"{safe_label}_*")}

    @classmethod
    def _validate_outer_graph_capture(cls, case, prior_mtimes):
        matches = [
            path
            for path, mtime_ns in cls._matching_output_mtimes(case).items()
            if path not in prior_mtimes or mtime_ns > prior_mtimes[path]
        ]
        assert matches, f"no dep-gen output directory created for {case['name']}"
        deps_path = max(matches, key=lambda path: path.stat().st_mtime_ns) / "deps.json"
        assert deps_path.exists(), f"deps.json missing for {case['name']}"
        with deps_path.open() as f:
            deps = json.load(f)

        tasks = deps.get("tasks", [])
        assert [int(task["task_id"]) for task in tasks] == list(range(8)), (
            "dep-gen must capture the seed, two outer GRAPH tasks, and the five-task fallback Graph body"
        )
        outer_graph_ids = {
            int(task["task_id"])
            for task in tasks
            if task.get("kernel_ids") == [-1, -1, -1] and len(task.get("args", [])) == 3
        }
        assert outer_graph_ids == {1, 7}, f"outer GRAPH tasks missing from deps.json: {outer_graph_ids}"

        outer_edges = {
            (int(edge["pred"]), int(edge["succ"]), edge.get("source"))
            for edge in deps.get("edges", [])
            if int(edge["succ"]) in outer_graph_ids
        }
        assert outer_edges == {(0, 1, "creator"), (0, 7, "creator")}, (
            f"outer GRAPH boundary edges differ from runtime dependencies: {outer_edges}"
        )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
