# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Golden test for SPMD sync_start boundary conditions.

Tests edge-case block_num values relative to per-thread cluster capacity (8 clusters
with 3 sched threads = 24 total clusters, 48 total AIV cores).

MIX tasks (SLOTS_PER_BLOCK=3):
  T0: block_num=1,  sync_start=True  -> CL 0..2     (degenerate: always fast path)
  T1: block_num=8,  sync_start=True  -> CL 3..26    (exactly one thread's capacity)
  T2: block_num=9,  sync_start=True  -> CL 27..53   (one over: must enter drain)
  T3: block_num=23, sync_start=True  -> CL 54..122  (max valid: total_clusters - 1)
  T4: block_num=1,  sync_start=False -> CL 123..125  (baseline)

Output tensor: 126 cache lines = 2016 float32.

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
    (1, 0),  # T0: sync=True, degenerate
    (8, 3),  # T1: sync=True, exactly one thread's clusters
    (9, 27),  # T2: sync=True, one over → drain
    (23, 54),  # T3: sync=True, max valid (total_clusters - 1)
    (1, 123),  # T4: sync=False, baseline
]

TOTAL_CL = sum(block_num * SLOTS_PER_BLOCK for block_num, _ in TASKS)  # 126


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
