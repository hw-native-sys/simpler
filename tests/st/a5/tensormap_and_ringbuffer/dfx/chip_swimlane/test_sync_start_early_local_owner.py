#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Regression for #1548: an a5 early sync-start cohort fits one scheduler.

Idle-capacity cases run with one and three schedulers, using a cohort sized to
fit any local owner in that topology. A pending-capacity case fills every AIV
running slot first, forcing the local owner to stage the eight-block consumer
entirely into free pending slots. A scheduler-loop fence makes the setup
deterministic; a second fence keeps the blockers running through the consumer's
early-dispatch opportunity during their measured one-second hold. Phase capture
then verifies all eight local stages.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.tools.swimlane_converter import read_perf_data

FLOATS_PER_CACHE_LINE = 16
PRODUCER_BLOCKS = 1
MAX_CONSUMER_BLOCKS = 8
TOTAL_BLOCKS = PRODUCER_BLOCKS + MAX_CONSUMER_BLOCKS
BLOCKER_STATUS_CAPACITY = 72
OUTPUT_CACHE_LINES = TOTAL_BLOCKS + BLOCKER_STATUS_CAPACITY


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSyncStartEarlyLocalOwner(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/sync_start_early_local_owner_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "SLOW_FLAGGED_PRODUCER_AIC",
                "source": "../../spmd_sync_start_early_dispatch/kernels/aiv/kernel_spmd_write_slow.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "name": "SYNC_START_CONSUMER_AIV",
                "source": "../../spmd_multiblock_aiv/kernels/aiv/kernel_spmd_write.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 2,
                "name": "SPIN_BLOCKER_AIV",
                "source": "kernels/aiv/kernel_spmd_spin.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "single_scheduler_idle",
            "platforms": ["a5sim", "a5"],
            "manual": ["a5sim"],
            # One orchestrator plus one scheduler. The scheduler owns every AIV
            # core, so this eight-block cohort fits its local idle capacity.
            "config": {"aicpu_thread_num": 2},
            "params": {"mode": 0, "consumer_blocks": 8},
        },
        {
            "name": "three_schedulers_idle",
            "platforms": ["a5sim", "a5"],
            "manual": ["a5sim"],
            # One orchestrator plus three schedulers. The popping scheduler can
            # stage all four blocks without stopping either peer.
            "config": {"aicpu_thread_num": 4},
            "params": {"mode": 0, "consumer_blocks": 4},
        },
        {
            "name": "single_scheduler_pending_only",
            "platforms": ["a5sim", "a5"],
            "manual": ["a5sim"],
            # Every blocker reports started before an ACK-sweep fence. A second
            # scheduler-loop fence completes during their measured one-second
            # hold, proving the consumer used pending rather than idle slots.
            "config": {"aicpu_thread_num": 2},
            "params": {"mode": 1, "consumer_blocks": 8},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("output", torch.zeros(OUTPUT_CACHE_LINES * FLOATS_PER_CACHE_LINE, dtype=torch.float32)),
            Scalar("mode", int(params["mode"])),
            Scalar("consumer_blocks", int(params["consumer_blocks"])),
        )

    def compute_golden(self, args, params):
        for block_idx in range(PRODUCER_BLOCKS):
            args.output[block_idx * FLOATS_PER_CACHE_LINE] = float(block_idx)
        for block_idx in range(int(params["consumer_blocks"])):
            args.output[(PRODUCER_BLOCKS + block_idx) * FLOATS_PER_CACHE_LINE] = float(block_idx)

    def _build_config(self, config_dict, *args, **kwargs):
        config = super()._build_config(config_dict, *args, **kwargs)
        self._trace_perf_level = int(kwargs.get("enable_chip_swimlane", args[0] if args else 0))
        output_prefix = kwargs.get("output_prefix", "")
        self._trace_perf_path = Path(output_prefix) / "chip_swimlane_records.json" if output_prefix else None
        return config

    def compare_outputs(self, test_args, golden_args, output_names, params):
        super().compare_outputs(test_args, golden_args, output_names, params)
        if getattr(self, "_trace_perf_level", 0) < 3:
            return

        perf_path = self._trace_perf_path
        assert perf_path is not None, "chip swimlane enabled without an output prefix"
        assert perf_path.exists(), f"chip_swimlane_records.json missing under {perf_path.parent}"

        perf = read_perf_data(perf_path)
        assert int(perf.get("chip_swimlane_level", 0)) >= 3, f"scheduler phases missing from {perf_path}"
        phase_records = [record for thread in perf.get("scheduler_records", []) for record in thread]
        assert phase_records, f"scheduler phase capture is empty under {perf_path}"
        phase_counts = Counter(record.get("phase") for record in phase_records)
        expected_blocks = int(params["consumer_blocks"])
        drain_phase_names = {"drain", "drain_prepare", "drain_publish"}
        drain_records = [record for record in phase_records if record.get("phase") in drain_phase_names]
        drain_published = sum(
            int(record.get("tasks_processed", 0)) for record in phase_records if record.get("phase") == "drain_publish"
        )

        assert not drain_records, (
            "#1548: the a5 early sync-start consumer fits one scheduler's local "
            "AIV running/pending capacity but entered global drain; "
            f"drain_publish staged {drain_published}/{expected_blocks} blocks, "
            f"mode={params['mode']}, phase_counts={dict(phase_counts)}, artifact={perf_path}"
        )

        early_staged = sum(
            int(record.get("tasks_processed", 0)) for record in phase_records if record.get("phase") == "early_dispatch"
        )
        assert early_staged == expected_blocks, (
            "the a5 local-owner fast path was not observed: "
            f"early_dispatch staged {early_staged}, expected {expected_blocks}; "
            f"mode={params['mode']}, phase_counts={dict(phase_counts)}, artifact={perf_path}"
        )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
