#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SPMD basic context accessors: single MIX task verifying block_idx, block_num, sub_block_id.

Submits one MIX task (AIC + AIV0 + AIV1) with block_dim=1.
Each subtask writes its SPMD context at a sub_block_id-based offset.

Output layout (float32[48], 3 cache lines):
  [0..15]  = AIC  slot: [block_idx, block_num, pad x14]
  [16..31] = AIV0 slot: [block_idx, block_num, sub_block_id=0, pad x13]
  [32..47] = AIV1 slot: [block_idx, block_num, sub_block_id=1, pad x13]
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

FLOATS_PER_CACHE_LINE = 16


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSpmdBasic(SceneTestCase):
    """SPMD context accessors with a single MIX task."""

    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_basic_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {"func_id": 0, "name": "SPMD_READ_AIC", "source": "kernels/aic/kernel_spmd_read.cpp", "core_type": "aic"},
            {"func_id": 1, "name": "SPMD_READ_AIV0", "source": "kernels/aiv/kernel_spmd_read.cpp", "core_type": "aiv"},
            {"func_id": 2, "name": "SPMD_READ_AIV1", "source": "kernels/aiv/kernel_spmd_read.cpp", "core_type": "aiv"},
        ],
    }

    CASES = [
        {
            "name": "Case1",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {},
        },
    ]

    def generate_args(self, params):
        output = torch.zeros(3 * FLOATS_PER_CACHE_LINE, dtype=torch.float32)
        return TaskArgsBuilder(Tensor("output", output))

    def compute_golden(self, args, params):
        out = args.output
        out[0] = 0.0
        out[1] = 1.0
        base = 1 * FLOATS_PER_CACHE_LINE
        out[base + 0] = 0.0
        out[base + 1] = 1.0
        out[base + 2] = 0.0
        base = 2 * FLOATS_PER_CACHE_LINE
        out[base + 0] = 0.0
        out[base + 1] = 1.0
        out[base + 2] = 1.0


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
