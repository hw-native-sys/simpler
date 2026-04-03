# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Golden test for SPMD sync_start.

Submits 4 MIX tasks (3 with require_sync_start=true, 1 baseline) and verifies
all blocks of every task write the correct float(block_idx) to their cache line.

Tasks (AIC=slot0, AIV0=slot1, AIV1=slot2):
  T0: block_num=2,  sync_start=True  -> CL 0..5
  T1: block_num=8,  sync_start=True  -> CL 6..29
  T2: block_num=2,  sync_start=False -> CL 30..35  (baseline)
  T3: block_num=12, sync_start=True  -> CL 36..71

Output tensor: 72 cache lines = 1152 float32.

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
    (2, 0),  # T0: sync_start=True
    (8, 6),  # T1: sync_start=True
    (2, 30),  # T2: sync_start=False (baseline)
    (12, 36),  # T3: sync_start=True
]

TOTAL_CL = sum(block_num * SLOTS_PER_BLOCK for block_num, _ in TASKS)  # 72


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
