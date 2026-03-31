# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Golden test for SPMD multi-block AIV.

Submits five AIV tasks with block_num = 4, 16, 24, 48, 96 to verify:
  T0 (block_num=4):  basic multi-block — fits within one sched thread
  T1 (block_num=16): saturates one sched thread (8 clusters × 2 AIV)
  T2 (block_num=24): forces cross-thread dispatch via ready_queue re-push
  T3 (block_num=48): occupies all AIV cores across all 3 sched threads
  T4 (block_num=96): two full rounds of all AIV cores

Each block writes float(block_idx) at cache line (base_cl + block_idx).
Output tensor: 188 cache lines = 3008 float32.

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

# (block_num, base_cl) for each submitted task
TASKS = [
    (4, 0),  # T0: basic
    (16, 4),  # T1: saturate single thread
    (24, 20),  # T2: cross-thread
    (48, 44),  # T3: all AIV cores
    (96, 92),  # T4: two full rounds
]

TOTAL_CL = sum(block_num for block_num, _ in TASKS)  # 44


def generate_inputs(params: dict) -> list:
    output = torch.zeros(TOTAL_CL * FLOATS_PER_CACHE_LINE, dtype=torch.float32)
    return [
        ("output", output),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    out = torch.as_tensor(tensors["output"])
    for block_num, base_cl in TASKS:
        for block_idx in range(block_num):
            out[(base_cl + block_idx) * FLOATS_PER_CACHE_LINE] = float(block_idx)
    tensors["output"][:] = out
