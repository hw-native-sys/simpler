#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SPMD starvation prevention: 18 normal MIX + 2 sync_start MIX tasks.

Validates that the drain mechanism prevents sync_start tasks from being starved.

Layout:
  Wave 1: 6 x normal(block_num=4)      -> CL 0..71
  Sync 0: 1 x sync_start(block_num=6)  -> CL 72..89
  Wave 2: 6 x normal(block_num=4)      -> CL 90..161
  Sync 1: 1 x sync_start(block_num=6)  -> CL 162..179
  Wave 3: 6 x normal(block_num=4)      -> CL 180..251

Total: 252 CL = 4032 float32.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

FLOATS_PER_CACHE_LINE = 16
SLOTS_PER_BLOCK = 3
NORMAL_BLOCK_NUM = 4
SYNC_BLOCK_NUM = 6
NORMAL_CL = NORMAL_BLOCK_NUM * SLOTS_PER_BLOCK
SYNC_CL = SYNC_BLOCK_NUM * SLOTS_PER_BLOCK


def _build_tasks():
    tasks = []
    cl = 0
    for _ in range(6):
        tasks.append((NORMAL_BLOCK_NUM, cl))
        cl += NORMAL_CL
    tasks.append((SYNC_BLOCK_NUM, cl))
    cl += SYNC_CL
    for _ in range(6):
        tasks.append((NORMAL_BLOCK_NUM, cl))
        cl += NORMAL_CL
    tasks.append((SYNC_BLOCK_NUM, cl))
    cl += SYNC_CL
    for _ in range(6):
        tasks.append((NORMAL_BLOCK_NUM, cl))
        cl += NORMAL_CL
    return tasks


TASKS = _build_tasks()
TOTAL_CL = sum(bn * SLOTS_PER_BLOCK for bn, _ in TASKS)


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSpmdStarvation(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_starvation_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {"func_id": 0, "source": "../spmd_multiblock_mix/kernels/aic/kernel_spmd_mix.cpp", "core_type": "aic"},
            {"func_id": 1, "source": "../spmd_multiblock_mix/kernels/aiv/kernel_spmd_mix.cpp", "core_type": "aiv"},
            {"func_id": 2, "source": "../spmd_multiblock_mix/kernels/aiv/kernel_spmd_mix.cpp", "core_type": "aiv"},
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
