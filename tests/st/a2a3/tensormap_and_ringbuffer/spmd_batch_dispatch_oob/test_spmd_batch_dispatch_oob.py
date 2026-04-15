#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Regression test for batch dispatch OOB (issue #565).

Submits two back-to-back MIX tasks each with block_num=48 (>> 24 clusters).
When both tasks enter the ready queue simultaneously, pop_ready_tasks_batch
returns got=2.  Without the fix, the first task's do-while drains all idle
clusters, and the second task's do-while calls pop_first() on an empty mask,
returning -1 as cluster_offset — an out-of-bounds array index.

Each block writes float(block_idx) at 3 cache lines (AIC, AIV0, AIV1).
Output tensor: 2 * 48 * 3 = 288 cache lines = 4608 float32.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

FLOATS_PER_CACHE_LINE = 16
SLOTS_PER_BLOCK = 3
TASKS = [(48, 0), (48, 144)]
TOTAL_CL = sum(bn * SLOTS_PER_BLOCK for bn, _ in TASKS)


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSpmdBatchDispatchOob(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_batch_dispatch_oob_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {"func_id": 0, "source": "kernels/aic/kernel_write.cpp", "core_type": "aic"},
            {"func_id": 1, "source": "kernels/aiv/kernel_write.cpp", "core_type": "aiv"},
            {"func_id": 2, "source": "kernels/aiv/kernel_write.cpp", "core_type": "aiv"},
        ],
    }

    CASES = [
        {
            "name": "Case1",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {},
        }
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(Tensor("output", torch.zeros(TOTAL_CL * FLOATS_PER_CACHE_LINE, dtype=torch.float32)))

    def compute_golden(self, args, params):
        out = args.output
        for block_num, base_cl in TASKS:
            for block_idx in range(block_num):
                for slot in range(SLOTS_PER_BLOCK):
                    cl = base_cl + block_idx * SLOTS_PER_BLOCK + slot
                    out[cl * FLOATS_PER_CACHE_LINE] = float(block_idx)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
