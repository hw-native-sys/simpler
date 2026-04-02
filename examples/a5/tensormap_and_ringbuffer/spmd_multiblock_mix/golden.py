# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Golden test for SPMD multi-block MIX.

Submits five MIX tasks (AIC + AIV0 + AIV1) with block_num = 2, 8, 12, 24, 48 to verify:
  T0 (block_num=2):  basic multi-block MIX
  T1 (block_num=8):  saturates one sched thread (8 clusters)
  T2 (block_num=12): forces cross-thread dispatch via ready_queue re-push
  T3 (block_num=24): occupies all clusters across all 3 sched threads
  T4 (block_num=48): two full rounds of all clusters

Each block occupies 3 cache lines (AIC, AIV0, AIV1).  All three cores
in the same block write the same float(block_idx) to their respective CL.

Output tensor: 282 cache lines = 4512 float32.

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

# (block_num, base_cl) for each submitted task
TASKS = [
    (2, 0),  # T0: basic MIX (6 CL)
    (8, 6),  # T1: saturate single thread (24 CL)
    (12, 30),  # T2: cross-thread (36 CL)
    (24, 66),  # T3: all clusters (72 CL)
    (48, 138),  # T4: two full rounds (144 CL)
]

TOTAL_CL = sum(block_num * SLOTS_PER_BLOCK for block_num, _ in TASKS)  # 66


def generate_inputs(params: dict) -> list:
    output = torch.zeros(TOTAL_CL * FLOATS_PER_CACHE_LINE, dtype=torch.float32)
    return [
        ("output", output),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    out = torch.as_tensor(tensors["output"])
    for block_num, base_cl in TASKS:
        for block_idx in range(block_num):
            for slot in range(SLOTS_PER_BLOCK):
                cl = base_cl + block_idx * SLOTS_PER_BLOCK + slot
                out[cl * FLOATS_PER_CACHE_LINE] = float(block_idx)
    tensors["output"][:] = out
