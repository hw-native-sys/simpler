#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SIMT basic element-scatter: minimal AIV scatter kernel that exercises the SIMT launch path.

Config (block_dim=24, aicpu_thread_num=4, sequential identity indices)
mirrors the ptoas-validated mscatter reference at
mscatter_fp32_8x32_seq_20260513_140539/test_mscatter.py. Identity
indices keep the golden trivially src-equals-out so a failure here
points at the SIMT launch path itself (TLV injection, localMemorySize
budget, sync) rather than at the scatter index semantics.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

TILE_ROWS = 8
TILE_COLS = 32
SRC_ELEMS = TILE_ROWS * TILE_COLS  # 256
DST_LEN = SRC_ELEMS  # 256


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSimtBasic(SceneTestCase):
    RTOL = 1e-5
    ATOL = 1e-5

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/simt_basic_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "SIMT_SCATTER",
                "source": "kernels/aiv/kernel_simt_scatter.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "Case1",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {},
        }
    ]

    def generate_args(self, params):
        torch.manual_seed(0)
        src = torch.randn(SRC_ELEMS, dtype=torch.float32)
        # Identity indices (0..DST_LEN-1) — matches the ptoas reference and
        # makes the golden trivially `out == src`. Switch to torch.randperm
        # later once the baseline launch path is confirmed green.
        indices = torch.arange(DST_LEN, dtype=torch.int32)
        out = torch.zeros(DST_LEN, dtype=torch.float32)
        return TaskArgsBuilder(
            Tensor("src", src),
            Tensor("indices", indices),
            Tensor("out", out),
        )

    def compute_golden(self, args, params):
        args.out.zero_()
        args.out[args.indices.to(torch.int64)] = args.src


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
