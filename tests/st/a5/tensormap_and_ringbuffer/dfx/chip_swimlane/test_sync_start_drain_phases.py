#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Positive coverage for A5 global sync-start drain phase records."""

from __future__ import annotations

from pathlib import Path

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.tools.swimlane_converter import read_perf_data

FLOATS_PER_CACHE_LINE = 16
NUM_TASKS = 4
MAX_AIV = 72
MAX_TOTAL_CL = MAX_AIV // 12 + MAX_AIV // 3 + MAX_AIV // 12 + MAX_AIV // 2


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSyncStartDrainPhases(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "../../spmd_sync_start_aiv/kernels/orchestration/spmd_sync_start_aiv_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "SPMD_WRITE_AIV",
                "source": "../../spmd_multiblock_aiv/kernels/aiv/kernel_spmd_write.cpp",
                "core_type": "aiv",
            },
        ],
    }

    CASES = [
        {
            "name": "global_aiv",
            "platforms": ["a5sim", "a5"],
            "manual": ["a5sim"],
            "params": {},
        }
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("output", torch.zeros(MAX_TOTAL_CL * FLOATS_PER_CACHE_LINE, dtype=torch.float32)),
            TensorArg("layout", torch.zeros(2 * NUM_TASKS, dtype=torch.int32)),
        )

    def compute_golden(self, args, params):
        pass

    def _build_config(self, config_dict, *args, **kwargs):
        config = super()._build_config(config_dict, *args, **kwargs)
        self._trace_perf_level = int(kwargs.get("enable_chip_swimlane", args[0] if args else 0))
        output_prefix = kwargs.get("output_prefix", "")
        self._trace_perf_path = Path(output_prefix) / "chip_swimlane_records.json" if output_prefix else None
        return config

    def compare_outputs(self, test_args, golden_args, output_names, params):
        layout = [int(value) for value in test_args.layout]
        expected = torch.zeros(MAX_TOTAL_CL, dtype=torch.float32)
        for task_idx in range(NUM_TASKS):
            block_num, base_cl = layout[2 * task_idx], layout[2 * task_idx + 1]
            assert block_num >= 1, f"task {task_idx} reported block_num {block_num}"
            for block_idx in range(block_num):
                expected[base_cl + block_idx] = float(block_idx)
        actual = test_args.output.reshape(MAX_TOTAL_CL, FLOATS_PER_CACHE_LINE)[:, 0]
        assert torch.equal(actual, expected), f"output disagrees with reported layout {layout}"

        if getattr(self, "_trace_perf_level", 0) < 3:
            return
        perf_path = self._trace_perf_path
        assert perf_path is not None and perf_path.exists(), "chip swimlane scheduler capture is missing"
        perf = read_perf_data(perf_path)
        phase_records = [record for thread in perf.get("scheduler_records", []) for record in thread]
        sync_start_blocks = layout[0] + layout[2] + layout[2 * (NUM_TASKS - 1)]
        phase_work = {}
        for phase in ("drain", "drain_prepare", "drain_publish"):
            records = [record for record in phase_records if record.get("phase") == phase]
            phase_work[phase] = sum(int(record.get("tasks_processed", 0)) for record in records)
            assert 0 < phase_work[phase] <= sync_start_blocks, (
                f"{phase} reported {phase_work[phase]} blocks/subtasks for {sync_start_blocks} sync-start blocks; "
                f"artifact={perf_path}"
            )
        assert len(set(phase_work.values())) == 1, f"drain phase workloads disagree: {phase_work}; artifact={perf_path}"


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
