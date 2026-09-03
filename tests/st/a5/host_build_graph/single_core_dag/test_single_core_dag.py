#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Correctness gates for homogeneous and mixed AIC/AIV dependency graphs."""

import ctypes
import json

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename

GRAPH_CASES = {
    "chain_64": (0, 64),
    "diamond_8": (1, 25),
    "fanout_63": (2, 64),
    "fanin_32x32": (3, 64),
    "multi_root_64": (4, 64),
    "random_64": (5, 64),
}

AUTOMATIC_CASES = {"aic_chain_64", "aiv_chain_64", "mixed_diamond_8"}


@scene_test(level=2, runtime="host_build_graph")
class TestHbgSingleCoreDag(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/single_core_dag_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/check_dag.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "source": "kernels/check_dag.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": f"{core_type}_{name}",
            "platforms": ["a5sim", "a5"],
            "params": {
                "graph_case": graph_case,
                "task_count": task_count,
                "core_type": core_type,
            },
            **({} if f"{core_type}_{name}" in AUTOMATIC_CASES else {"manual": True}),
        }
        for core_type, names in {
            "aic": ("chain_64",),
            "aiv": ("chain_64",),
            "mixed": tuple(GRAPH_CASES),
        }.items()
        for name in names
        for graph_case, task_count in (GRAPH_CASES[name],)
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("task_state", torch.zeros(64 * 8, dtype=torch.int64)),
            Scalar("graph_case", ctypes.c_int64(params["graph_case"])),
            Scalar("core_type", ctypes.c_int64({"aic": 0, "aiv": 1, "mixed": 2}[params["core_type"]])),
        )

    def compute_golden(self, args, params):
        count = params["task_count"]
        args.task_state[: count * 8 : 8] = torch.arange(1, count + 1, dtype=torch.int64)

    def test_run(self, st_platform, st_worker, request):
        outputs_dir = _outputs_dir()
        previous_outputs = (
            {path: path.stat().st_mtime_ns for path in outputs_dir.iterdir()} if outputs_dir.exists() else {}
        )
        super().test_run(st_platform, st_worker, request)
        level = self._effective_enable_chip_swimlane(request)
        if level == 0:
            return

        record_fields = {"start_cycles", "end_cycles", "loop_iter", "kind", "tasks_processed", "task_id"}
        for case in self._matching_cases(st_platform, request):
            case_label = _sanitize_for_filename(f"TestHbgSingleCoreDag_{case['name']}")
            matches = [
                path
                for path in outputs_dir.glob(f"{case_label}_*")
                if path not in previous_outputs or path.stat().st_mtime_ns > previous_outputs[path]
            ]
            assert matches, f"no output directory created for {case_label}"
            output_prefix = max(matches, key=lambda path: path.stat().st_mtime_ns)
            perf_path = output_prefix / "chip_swimlane_records.json"
            raw = json.loads(perf_path.read_text())
            aicore_rows = raw["aicore_tasks"]
            assert aicore_rows, "AICore task records are missing"
            if level >= 2:
                scheduler_tasks = raw["scheduler_tasks"]
                assert scheduler_tasks["schema_version"] == 1
                assert scheduler_tasks["producer"] == "aicore"
                scheduler_rows = scheduler_tasks["records"]
                aicore_by_key = {(int(row[0]), int(row[2])): row for row in aicore_rows}
                assert {(int(row[0]), int(row[1])) for row in scheduler_rows} == set(aicore_by_key)
                for core_id, reg_task_id, dispatch_cycles, finish_cycles in scheduler_rows:
                    aicore_row = aicore_by_key[(int(core_id), int(reg_task_id))]
                    assert 0 < dispatch_cycles <= aicore_row[3] <= aicore_row[4] <= finish_cycles
                assert raw["aicpu_lifecycle_records"], "AICPU lifecycle records are missing"
            if level < 3:
                continue
            streams = raw["scheduler_records"]["streams"]
            assert streams, "A5 HBG AICore Scheduler records are missing"
            assert all(stream["producer"] == "aicore" for stream in streams)
            assert all(stream["platform"] == "a5" and stream["runtime"] == "host_build_graph" for stream in streams)
            records = [record for stream in streams for record in stream["records"]]
            assert all(set(record) == record_fields for record in records)
            assert all(0 < record["start_cycles"] <= record["end_cycles"] for record in records)
            required_kinds = {"bootstrap", "dispatch", "complete", "resolve", "idle"}
            if case["params"]["graph_case"] != GRAPH_CASES["multi_root_64"][0]:
                required_kinds.add("fanin")
            emitted_kinds = {record["kind"] for record in records}
            assert required_kinds <= emitted_kinds, f"missing Scheduler kinds: {sorted(required_kinds - emitted_kinds)}"
            profiled_task_ids = {int(row[1]) for row in raw["aicore_tasks"]}
            for kind in ("dispatch", "complete"):
                recorded_task_ids = {int(record["task_id"]) for record in records if record["kind"] == kind}
                assert recorded_task_ids == profiled_task_ids, (
                    f"{kind} records do not cover every profiled task: "
                    f"missing={sorted(profiled_task_ids - recorded_task_ids)} "
                    f"unexpected={sorted(recorded_task_ids - profiled_task_ids)}"
                )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
