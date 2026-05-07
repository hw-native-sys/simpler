#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SPMD sync_start stress: 54 tasks over 6 rounds with mixed shapes (MIX + AIV).

Grand total: 840 CL = 13440 float32.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

FLOATS_PER_CACHE_LINE = 16
ROUNDS = 6
SHAPE_MIX, SHAPE_AIV = "MIX", "AIV"
MIX_SLOTS, AIV_SLOTS = 3, 1
NORMAL_MIX_BN, SYNC_MIX_BN, SYNC_AIV_BN, NORMAL_AIV_BN = 4, 12, 8, 4


def _build_tasks():
    tasks, cl = [], 0
    for _ in range(ROUNDS):
        for _ in range(4):
            tasks.append((NORMAL_MIX_BN, cl, SHAPE_MIX))
            cl += NORMAL_MIX_BN * MIX_SLOTS
        for _ in range(2):
            tasks.append((SYNC_MIX_BN, cl, SHAPE_MIX))
            cl += SYNC_MIX_BN * MIX_SLOTS
        for _ in range(2):
            tasks.append((SYNC_AIV_BN, cl, SHAPE_AIV))
            cl += SYNC_AIV_BN * AIV_SLOTS
        tasks.append((NORMAL_AIV_BN, cl, SHAPE_AIV))
        cl += NORMAL_AIV_BN * AIV_SLOTS
    return tasks


TASKS = _build_tasks()
TOTAL_CL = sum(bn * (MIX_SLOTS if s == SHAPE_MIX else AIV_SLOTS) for bn, _, s in TASKS)


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSpmdSyncStartStress(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_sync_start_stress_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "SPMD_MIX_AIC",
                "source": "../spmd_multiblock_mix/kernels/aic/kernel_spmd_mix.cpp",
                "core_type": "aic",
            },
            {
                "func_id": 1,
                "name": "SPMD_MIX_AIV0",
                "source": "../spmd_multiblock_mix/kernels/aiv/kernel_spmd_mix.cpp",
                "core_type": "aiv",
            },
            {
                "func_id": 2,
                "name": "SPMD_MIX_AIV1",
                "source": "../spmd_multiblock_mix/kernels/aiv/kernel_spmd_mix.cpp",
                "core_type": "aiv",
            },
            {
                "func_id": 3,
                "name": "SPMD_WRITE_AIV",
                "source": "../spmd_multiblock_aiv/kernels/aiv/kernel_spmd_write.cpp",
                "core_type": "aiv",
            },
        ],
    }

    CASES = [
        {
            "name": "Case1",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {},
        }
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(Tensor("output", torch.zeros(TOTAL_CL * FLOATS_PER_CACHE_LINE, dtype=torch.float32)))

    def compute_golden(self, args, params):
        out = args.output
        for block_num, base_cl, shape in TASKS:
            for block_idx in range(block_num):
                if shape == SHAPE_MIX:
                    for slot in range(MIX_SLOTS):
                        out[(base_cl + block_idx * MIX_SLOTS + slot) * FLOATS_PER_CACHE_LINE] = float(block_idx)
                else:
                    out[(base_cl + block_idx) * FLOATS_PER_CACHE_LINE] = float(block_idx)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
