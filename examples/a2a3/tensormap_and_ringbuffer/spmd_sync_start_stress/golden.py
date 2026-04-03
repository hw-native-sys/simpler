# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Golden test for SPMD sync_start stress / CAS contention with mixed shapes.

Submits 6 rounds of mixed-shape tasks to stress drain CAS contention, ack
barrier, and state cleanup across drain cycles.  All three resource shapes
(MIX, AIV, AIC) are exercised with both sync and non-sync modes.

Each round (9 tasks):
  4 × normal MIX  (block_num=4, sync=false) → 4 × 4 × 3 = 48 CL
  2 × sync MIX    (block_num=12, sync=true) → 2 × 12 × 3 = 72 CL
  2 × sync AIV    (block_num=8, sync=true)  → 2 × 8 × 1 = 16 CL
  1 × normal AIV  (block_num=4, sync=false) → 1 × 4 × 1 = 4 CL
  Round total: 140 CL

6 rounds → 54 tasks (24 normal MIX + 12 sync MIX + 12 sync AIV + 6 normal AIV)
Grand total: 840 CL = 13440 float32

Stress coverage:
  - 24 drain cycles (12 MIX + 12 AIV) → validates state cleanup
  - 2 sync MIX + 2 sync AIV per round → CAS contention across shapes
  - Normal tasks occupy clusters → forces drain slow path
  - 54 tasks total → no task loss under sustained load

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
ROUNDS = 6

# shape constants: (slots_per_block, written_slots)
# MIX: kernel writes at base_cl + block_idx * 3 + {0,1,2}, 3 CL per block, all written
# AIV: kernel writes at base_cl + block_idx, 1 CL per block
SHAPE_MIX = "MIX"
SHAPE_AIV = "AIV"

MIX_SLOTS = 3
AIV_SLOTS = 1

NORMAL_MIX_BN = 4
SYNC_MIX_BN = 12
SYNC_AIV_BN = 8
NORMAL_AIV_BN = 4


def _build_tasks():
    """Returns list of (block_num, base_cl, shape_str)."""
    tasks = []
    cl = 0
    for _ in range(ROUNDS):
        # 4 × normal MIX
        for _ in range(4):
            tasks.append((NORMAL_MIX_BN, cl, SHAPE_MIX))
            cl += NORMAL_MIX_BN * MIX_SLOTS
        # 2 × sync MIX
        for _ in range(2):
            tasks.append((SYNC_MIX_BN, cl, SHAPE_MIX))
            cl += SYNC_MIX_BN * MIX_SLOTS
        # 2 × sync AIV
        for _ in range(2):
            tasks.append((SYNC_AIV_BN, cl, SHAPE_AIV))
            cl += SYNC_AIV_BN * AIV_SLOTS
        # 1 × normal AIV
        tasks.append((NORMAL_AIV_BN, cl, SHAPE_AIV))
        cl += NORMAL_AIV_BN * AIV_SLOTS
    return tasks


TASKS = _build_tasks()
TOTAL_CL = sum(bn * (MIX_SLOTS if shape == SHAPE_MIX else AIV_SLOTS) for bn, _, shape in TASKS)  # 840


def generate_inputs(params: dict) -> list:
    output = torch.zeros(TOTAL_CL * FLOATS_PER_CACHE_LINE, dtype=torch.float32)
    return [("output", output)]


def compute_golden(tensors: dict, params: dict) -> None:
    out = torch.as_tensor(tensors["output"])
    for block_num, base_cl, shape in TASKS:
        for block_idx in range(block_num):
            if shape == SHAPE_MIX:
                # MIX kernel writes float(block_idx) at all 3 slots
                for slot in range(MIX_SLOTS):
                    cl = base_cl + block_idx * MIX_SLOTS + slot
                    out[cl * FLOATS_PER_CACHE_LINE] = float(block_idx)
            else:
                # AIV kernel writes float(block_idx) at 1 slot
                cl = base_cl + block_idx
                out[cl * FLOATS_PER_CACHE_LINE] = float(block_idx)
    tensors["output"][:] = out
