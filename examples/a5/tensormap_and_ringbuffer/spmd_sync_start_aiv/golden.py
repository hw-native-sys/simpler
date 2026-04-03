# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Golden test for SPMD sync_start with AIV-only tasks.

Submits 4 AIV tasks (3 with require_sync_start=true, 1 baseline) to exercise
the AIV-specific fast path (count_idle_aiv_cores) and drain slow path.

Tasks:
  T0: block_num=4,  sync_start=True  -> CL 0..3    (fast path)
  T1: block_num=16, sync_start=True  -> CL 4..19   (saturate one thread)
  T2: block_num=4,  sync_start=False -> CL 20..23  (baseline)
  T3: block_num=24, sync_start=True  -> CL 24..47  (cross-thread drain)

Output tensor: 48 cache lines = 768 float32.

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
    (4, 0),  # T0: sync_start=True, fast path
    (16, 4),  # T1: sync_start=True, saturate single thread
    (4, 20),  # T2: sync_start=False, baseline
    (24, 24),  # T3: sync_start=True, cross-thread drain
]

TOTAL_CL = sum(block_num for block_num, _ in TASKS)  # 48


def generate_inputs(params: dict) -> list:
    output = torch.zeros(TOTAL_CL * FLOATS_PER_CACHE_LINE, dtype=torch.float32)
    return [("output", output)]


def compute_golden(tensors: dict, params: dict) -> None:
    out = torch.as_tensor(tensors["output"])
    for block_num, base_cl in TASKS:
        for block_idx in range(block_num):
            cl = base_cl + block_idx
            out[cl * FLOATS_PER_CACHE_LINE] = float(block_idx)
    tensors["output"][:] = out
