# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Golden test for SPMD context accessors (Phase 2: block_dim=1).

Verifies that get_block_idx and get_block_num return correct values for all
three subtask slots (AIC, AIV0, AIV1) in a MIX task, and that AIV
kernels read the correct sub_block_id from GlobalContext.

Phase 2 invariants: block_idx=0, block_num=1.
GlobalContext: sub_block_id 0 (AIV0/left), 1 (AIV1/right).

Output layout (float32[48], 3 cache lines):
  [0..15]  = AIC  slot: [block_idx, block_num, pad x14]
  [16..31] = AIV0 slot: [block_idx, block_num, sub_block_id=0, pad x13]
  [32..47] = AIV1 slot: [block_idx, block_num, sub_block_id=1, pad x13]

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

# 16 floats per slot = 64 bytes = 1 cache line
FLOATS_PER_CACHE_LINE = 16


def generate_inputs(params: dict) -> list:
    output = torch.zeros(3 * FLOATS_PER_CACHE_LINE, dtype=torch.float32)
    return [
        ("output", output),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    out = torch.as_tensor(tensors["output"])
    # Cache line 0: AIC (no sub_block_id)
    out[0] = 0.0  # block_idx
    out[1] = 1.0  # block_num
    # Cache line 1: AIV0 (sub_block_id=0)
    base = 1 * FLOATS_PER_CACHE_LINE
    out[base + 0] = 0.0  # block_idx
    out[base + 1] = 1.0  # block_num
    out[base + 2] = 0.0  # sub_block_id
    # Cache line 2: AIV1 (sub_block_id=1)
    base = 2 * FLOATS_PER_CACHE_LINE
    out[base + 0] = 0.0  # block_idx
    out[base + 1] = 1.0  # block_num
    out[base + 2] = 1.0  # sub_block_id
    tensors["output"][:] = out
