# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Golden test for SPMD starvation prevention.

Submits 18 normal MIX tasks interleaved with 2 sync_start MIX tasks and
verifies all 20 tasks complete with correct output.  The test validates that
the drain mechanism prevents sync_start tasks from being starved.

Layout:
  Wave 1: 6 × normal(block_num=4)  -> CL 0..71
  Sync 0: 1 × sync_start(block_num=6) -> CL 72..89
  Wave 2: 6 × normal(block_num=4)  -> CL 90..161
  Sync 1: 1 × sync_start(block_num=6) -> CL 162..179
  Wave 3: 6 × normal(block_num=4)  -> CL 180..251

Total: 252 CL = 4032 float32.

Args layout: [output]
"""

import torch

__outputs__ = ["output"]
RTOL = 0
ATOL = 0

ALL_CASES = {
    "Case1": {},
}

DEFAULT_CASE = "Case1"

FLOATS_PER_CACHE_LINE = 16
SLOTS_PER_BLOCK = 3  # AIC, AIV0, AIV1
NORMAL_BLOCK_NUM = 4
SYNC_BLOCK_NUM = 6
NORMAL_CL = NORMAL_BLOCK_NUM * SLOTS_PER_BLOCK  # 12
SYNC_CL = SYNC_BLOCK_NUM * SLOTS_PER_BLOCK  # 18


# Build flat task list as (block_num, base_cl)
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
TOTAL_CL = sum(bn * SLOTS_PER_BLOCK for bn, _ in TASKS)  # 252


def generate_inputs(params: dict) -> list:
    output = torch.zeros(TOTAL_CL * FLOATS_PER_CACHE_LINE, dtype=torch.float32)
    return [("output", output)]


def compute_golden(tensors: dict, params: dict) -> None:
    out = torch.as_tensor(tensors["output"])
    for block_num, base_cl in TASKS:
        for block_idx in range(block_num):
            for slot in range(SLOTS_PER_BLOCK):
                cl = base_cl + block_idx * SLOTS_PER_BLOCK + slot
                out[cl * FLOATS_PER_CACHE_LINE] = float(block_idx)
    tensors["output"][:] = out
