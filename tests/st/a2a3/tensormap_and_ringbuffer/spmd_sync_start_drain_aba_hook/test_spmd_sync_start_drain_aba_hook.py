#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Deterministic test-hook proof for the sync_start drain ABA window."""

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

FLOATS_PER_CACHE_LINE = 16
MIX_SLOTS = 3
SYNC_BLOCKS = 24
HOLDER_BLOCKS = 1


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSpmdSyncStartDrainAbaHook(SceneTestCase):
    RTOL = 0
    ATOL = 0

    @pytest.fixture(autouse=True)
    def _force_drain_aba_hook(self, monkeypatch):
        monkeypatch.setenv("SIMPLER_DRAIN_ABA_TEST", "1")

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_sync_start_drain_aba_hook_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "SPMD_MIX_SLOW_AIC",
                "source": "kernels/aic/kernel_spmd_mix_pressure.cpp",
                "core_type": "aic",
                "signature": [D.INOUT, D.IN],
            },
            {
                "func_id": 1,
                "name": "SPMD_MIX_SLOW_AIV0",
                "source": "kernels/aiv/kernel_spmd_mix_pressure.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.IN],
            },
            {
                "func_id": 2,
                "name": "SPMD_MIX_SLOW_AIV1",
                "source": "kernels/aiv/kernel_spmd_mix_pressure.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.IN],
            },
        ],
    }

    CASES = [
        {
            "name": "Case1",
            "platforms": ["a2a3"],
            "config": {
                "aicpu_thread_num": 4,
                "block_dim": 24,
                "runtime_env": {"ring_heap": 32 * 1024 * 1024, "ring_task_window": 1024},
            },
            "params": {},
        }
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            Tensor("output", torch.zeros(SYNC_BLOCKS * MIX_SLOTS * FLOATS_PER_CACHE_LINE, dtype=torch.float32)),
            Tensor("holder", torch.zeros(HOLDER_BLOCKS * MIX_SLOTS * FLOATS_PER_CACHE_LINE, dtype=torch.float32)),
            Tensor("scratch", torch.zeros(1024, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        out = args.output
        for block_idx in range(SYNC_BLOCKS):
            for slot in range(MIX_SLOTS):
                out[(block_idx * MIX_SLOTS + slot) * FLOATS_PER_CACHE_LINE] = float(block_idx)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
