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

import time

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename
from simpler_setup.tools.swimlane_converter import read_perf_data


@scene_test(level=2, runtime="host_build_graph")
class TestSchedulerPhases(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/scheduler_phases_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aiv/kernel_noop.cpp",
                "core_type": "aiv",
                "signature": [D.IN],
            },
        ],
    }

    CASES = [
        {
            "name": "resolve_dummy",
            "platforms": ["a2a3sim", "a2a3"],
            "manual": ["a2a3sim"],
            "params": {},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("input", torch.zeros(1, dtype=torch.int32)),
        )

    def compute_golden(self, args, params):
        pass

    def test_run(self, st_platform, st_worker, request):
        run_marker = int(time.time())
        super().test_run(st_platform, st_worker, request)
        if self._effective_enable_chip_swimlane(request) < 3:
            return

        for case in self._matching_cases(st_platform, request):
            case_label = _sanitize_for_filename(f"TestSchedulerPhases_{case['name']}")
            matches = [p for p in _outputs_dir().glob(f"{case_label}_*") if p.stat().st_mtime >= run_marker]
            assert matches, f"no output directory created for {case_label}"
            perf_path = max(matches, key=lambda p: p.stat().st_mtime) / "chip_swimlane_records.json"
            assert perf_path.exists(), f"missing chip swimlane artifact: {perf_path}"

            data = read_perf_data(perf_path)
            phase_threads = data.get("aicpu_scheduler_phases")
            assert phase_threads, "scheduler phase records are missing"
            assigned_threads = {thread_idx for thread_idx in data.get("core_to_thread", []) if thread_idx >= 0}
            recording = [(thread_idx, records) for thread_idx, records in enumerate(phase_threads) if records]
            assert recording, "no scheduler thread emitted phase records"

            # Scheduler threads are symmetric: each one owns cores and resolves
            # the completions it observes, so no thread records phases without
            # holding a core slice.
            core_less = [thread_idx for thread_idx, _ in recording if thread_idx not in assigned_threads]
            assert not core_less, f"scheduler threads own cores; core-less threads recorded phases: {core_less}"

            # The dummy retirement is not one thread's job any more -- whichever
            # thread wins the queue does it -- but the graph holds exactly one
            # dummy task, so the chip-wide count is still one.
            dummy = [record for _, records in recording for record in records if record.get("phase") == "dummy"]
            assert dummy, "no dummy phase recorded on any scheduler thread"
            assert sum(record["tasks_processed"] for record in dummy) == 1
            assert all(record["loop_iter"] > 0 for record in dummy)
            assert all(record["end_time_us"] >= record["start_time_us"] for record in dummy)

            for thread_idx, records in recording:
                assert len(records) < 64, f"thread {thread_idx} phase aggregation produced excessive records"
                # Compare only same-kind bars: the Dummy and AsyncPoll spans sit
                # inside the same iteration's Complete span, so cross-kind overlap
                # is expected.
                ordered = sorted(
                    (record for record in records if record.get("phase") == "dummy"),
                    key=lambda record: (record["start_time_us"], record["end_time_us"]),
                )
                assert all(left["end_time_us"] <= right["start_time_us"] for left, right in zip(ordered, ordered[1:]))


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
