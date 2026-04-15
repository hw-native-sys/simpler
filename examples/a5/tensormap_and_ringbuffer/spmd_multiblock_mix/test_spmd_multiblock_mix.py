#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SPMD multi-block MIX: five MIX tasks with varying block_num.

  T0 (block_num=2):  basic multi-block MIX
  T1 (block_num=8):  saturates one sched thread
  T2 (block_num=12): forces cross-thread dispatch
  T3 (block_num=24): occupies all clusters across all 3 sched threads
  T4 (block_num=48): two full rounds of all clusters

Each block occupies 3 cache lines (AIC, AIV0, AIV1). All three cores
in the same block write float(block_idx) to their respective CL.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

FLOATS_PER_CACHE_LINE = 16
SLOTS_PER_BLOCK = 3

TASKS = [
    (2, 0),
    (8, 6),
    (12, 30),
    (24, 66),
    (48, 138),
]

TOTAL_CL = sum(block_num * SLOTS_PER_BLOCK for block_num, _ in TASKS)


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSpmdMultiblockMix(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_multiblock_mix_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {"func_id": 0, "source": "kernels/aic/kernel_spmd_mix.cpp", "core_type": "aic"},
            {"func_id": 1, "source": "kernels/aiv/kernel_spmd_mix.cpp", "core_type": "aiv"},
            {"func_id": 2, "source": "kernels/aiv/kernel_spmd_mix.cpp", "core_type": "aiv"},
        ],
    }

    CASES = [
        {
            "name": "Case1",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {},
        },
    ]

    def generate_args(self, params):
        output = torch.zeros(TOTAL_CL * FLOATS_PER_CACHE_LINE, dtype=torch.float32)
        return TaskArgsBuilder(Tensor("output", output))

    def compute_golden(self, args, params):
        out = args.output
        for block_num, base_cl in TASKS:
            for block_idx in range(block_num):
                for slot in range(SLOTS_PER_BLOCK):
                    cl = base_cl + block_idx * SLOTS_PER_BLOCK + slot
                    out[cl * FLOATS_PER_CACHE_LINE] = float(block_idx)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
