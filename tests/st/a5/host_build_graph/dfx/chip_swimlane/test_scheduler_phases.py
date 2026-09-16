#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from collections import Counter

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename

FANOUT_WIDTH = 32
EXPECTED_PROFILED_TASKS = 1 + FANOUT_WIDTH + 1 + 1  # root + children + final + terminal dummy


@scene_test(level=2, runtime="host_build_graph")
class TestSchedulerPhases(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/scheduler_phases_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aiv/kernel_noop.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "source": "kernels/aiv/kernel_empty.cpp",
                "core_type": "aiv",
                "signature": [],
            },
        ],
    }

    CASES = [
        {
            "name": "resolve_dummy",
            "platforms": ["a5sim", "a5"],
            "manual": ["a5sim"],
            "params": {},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("input", torch.zeros(1, dtype=torch.int32)),
        )

    def compute_golden(self, args, params):
        args.input[0] = 1

    def test_run(self, st_platform, st_worker, request):
        outputs_dir = _outputs_dir()
        previous_outputs = (
            {path: path.stat().st_mtime_ns for path in outputs_dir.iterdir()} if outputs_dir.exists() else {}
        )
        super().test_run(st_platform, st_worker, request)
        level = self._effective_enable_chip_swimlane(request)
        if level == 0:
            return

        for case in self._matching_cases(st_platform, request):
            case_label = _sanitize_for_filename(f"TestSchedulerPhases_{case['name']}")
            matches = [
                path
                for path in outputs_dir.glob(f"{case_label}_*")
                if path not in previous_outputs or path.stat().st_mtime_ns > previous_outputs[path]
            ]
            assert matches, f"no output directory created for {case_label}"
            output_prefix = max(matches, key=lambda path: path.stat().st_mtime_ns)
            raw = json.loads((output_prefix / "chip_swimlane_records.json").read_text())
            aicore_rows = raw["aicore_tasks"]
            assert len(aicore_rows) == EXPECTED_PROFILED_TASKS, (
                f"task timing covers {len(aicore_rows)} tasks, expected {EXPECTED_PROFILED_TASKS} "
                "for the fanout/fanin DAG and terminal dummy"
            )

            if level >= 2:
                scheduler_tasks = raw["scheduler_tasks"]
                assert scheduler_tasks["schema_version"] == 1
                assert scheduler_tasks["producer"] == "aicore"
                scheduler_rows = scheduler_tasks["records"]
                assert len(scheduler_rows) == len(aicore_rows)
                aicore_by_key = {(int(row[0]), int(row[2])): row for row in aicore_rows}
                assert {(int(row[0]), int(row[1])) for row in scheduler_rows} == set(aicore_by_key)
                for core_id, reg_task_id, dispatch_cycles, finish_cycles in scheduler_rows:
                    aicore_row = aicore_by_key[(int(core_id), int(reg_task_id))]
                    assert 0 < dispatch_cycles <= aicore_row[3] <= aicore_row[4] <= finish_cycles
                lifecycle_records = raw["aicpu_lifecycle_records"]
                assert lifecycle_records, "AICPU lifecycle records are missing"
                thread_ids = [int(record["aicpu_thread_id"]) for record in lifecycle_records]
                assert thread_ids == list(range(len(thread_ids)))
                assert all("worker_id" not in record for record in lifecycle_records)
                for record in lifecycle_records:
                    assert 0 < record["handshake_start_cycles"] <= record["handshake_complete_cycles"]
                    assert 0 < record["context_publish_start_cycles"] <= record["context_publish_complete_cycles"]
                    assert 0 < record["bootstrap_wait_start_cycles"] <= record["bootstrap_complete_cycles"]
                    assert 0 < record["register_release_start_cycles"] <= record["register_release_end_cycles"]
                    assert 0 < record["exit_signal_start_cycles"] <= record["exit_signal_end_cycles"]
                    assert 0 < record["exit_wait_start_cycles"] <= record["exit_wait_end_cycles"]
                leader = lifecycle_records[0]
                assert 0 < leader["config_start_cycles"] <= leader["topology_complete_cycles"]
                assert all(
                    record["config_start_cycles"] == record["topology_complete_cycles"] == 0
                    for record in lifecycle_records[1:]
                )
            else:
                assert "scheduler_tasks" not in raw
                assert "aicpu_lifecycle_records" not in raw

            if level >= 3:
                streams = raw["scheduler_records"]["streams"]
                assert streams, "A5 HBG AICore Scheduler records are missing"
                assert all(stream["producer"] == "aicore" for stream in streams)
                assert all(stream["capture"]["dropped"] == 0 for stream in streams)
                emitted_kinds = {record["kind"] for stream in streams for record in stream["records"]}
                required_kinds = {"bootstrap", "state_probe", "dispatch", "complete", "resolve", "refill", "idle"}
                assert required_kinds <= emitted_kinds, (
                    f"missing Scheduler kinds: {sorted(required_kinds - emitted_kinds)}"
                )
                assert not ({"fanin", "ready_claim", "ready_steal", "direct_refill"} & emitted_kinds)
                records = [record for stream in streams for record in stream["records"]]
                launch_kinds = {"dispatch", "worksteal", "refill"}
                launches_by_task = Counter(
                    int(record["task_id"])
                    for record in records
                    if record["kind"] in launch_kinds and record["task_id"] is not None
                )
                assert set(launches_by_task) == {int(row[1]) for row in aicore_rows}
                assert set(launches_by_task.values()) == {1}
                probes_by_task = Counter(
                    int(record["task_id"])
                    for record in records
                    if record["kind"] == "state_probe" and record["task_id"] is not None
                )
                assert probes_by_task == launches_by_task
                for stream in streams:
                    ordered = sorted(
                        stream["records"],
                        key=lambda record: (record["start_cycles"], record["end_cycles"]),
                    )
                    assert all(
                        previous["end_cycles"] <= current["start_cycles"]
                        for previous, current in zip(ordered, ordered[1:])
                    )
                probe_by_task = {
                    int(record["task_id"]): record
                    for record in records
                    if record["kind"] == "state_probe" and record["task_id"] is not None
                }
                launch_by_task = {
                    int(record["task_id"]): record
                    for record in records
                    if record["kind"] in launch_kinds and record["task_id"] is not None
                }
                assert all(
                    probe_by_task[task_id]["end_cycles"] <= launch["start_cycles"]
                    for task_id, launch in launch_by_task.items()
                )
            else:
                assert "scheduler_records" not in raw


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
