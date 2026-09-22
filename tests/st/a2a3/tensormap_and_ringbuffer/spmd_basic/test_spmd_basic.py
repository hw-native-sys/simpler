#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SPMD context accessors: one MIX task reading block_idx, block_num, sub_block_id, L2 offset.

Submits one MIX task (AIC + AIV0 + AIV1) with block_num=1.
Each subtask writes its SPMD context at a sub_block_id-based offset.

Output layout (float32[48], 3 cache lines):
  [0..15]  = AIC  slot: [block_idx, block_num, pad, l2_alias, pad x12]
  [16..31] = AIV0 slot: [block_idx, block_num, sub_block_id=0, l2_alias, pad x12]
  [32..47] = AIV1 slot: [block_idx, block_num, sub_block_id=1, l2_alias, pad x12]

`l2_alias` is 1.0 when that core sees a nonzero `get_l2_cache_offset(args)`. The
offset itself is whatever the driver reports for the device, so only its presence
is predictable here — onboard asserts all three cores see one, which is what
catches the offset being seeded for the AIV pair alone rather than every core.
Sim reaches no driver and asserts zero, which pins the args index and the
GlobalContext layout agreeing between the AICore-side and AICPU-side headers.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

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
            {
                "func_id": 0,
                "name": "SPMD_READ_AIC",
                "source": "kernels/aic/kernel_spmd_read.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "name": "SPMD_READ_AIV0",
                "source": "kernels/aiv/kernel_spmd_read.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 2,
                "name": "SPMD_READ_AIV1",
                "source": "kernels/aiv/kernel_spmd_read.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }

    CASES = [
        # Split by platform only because `l2_alias` differs: sim reaches no driver
        # and so exposes no nocache alias, while a device that has one reports it.
        {
            "name": "Case1",
            "platforms": ["a2a3sim"],
            "params": {"l2_alias": 0.0},
        },
        {
            "name": "Case1Onboard",
            "platforms": ["a2a3"],
            "params": {"l2_alias": 1.0},
        },
    ]

    def generate_args(self, params):
        output = torch.zeros(3 * FLOATS_PER_CACHE_LINE, dtype=torch.float32)
        return TaskArgsBuilder(TensorArg("output", output))

    def compute_golden(self, args, params):
        out = args.output
        l2_alias = params["l2_alias"]
        out[0] = 0.0
        out[1] = 1.0
        out[3] = l2_alias
        base = 1 * FLOATS_PER_CACHE_LINE
        out[base + 0] = 0.0
        out[base + 1] = 1.0
        out[base + 2] = 0.0
        out[base + 3] = l2_alias
        base = 2 * FLOATS_PER_CACHE_LINE
        out[base + 0] = 0.0
        out[base + 1] = 1.0
        out[base + 2] = 1.0
        out[base + 3] = l2_alias


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
