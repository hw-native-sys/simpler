#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Mixed AIC+AIV example covering all 5 resource shapes.

  1. AIC_AIV_X2: C = A@B, F = D+E, I = G*H
  2. AIC_ONLY:   J = A@B
  3. AIV_X1:     K = D+E
  4. AIV_X2:     L = D+E, M = G*H
  5. AIC_AIV_X1: N = A@B, O = D+E

All use 128x128 float32 tiles, repeated over num_iters slices.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

MATMUL_SIZE = 128
TILE_ELEMS = 128 * 128


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestMixedExample(SceneTestCase):
    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/mixed_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [
                D.IN,
                D.IN,
                D.OUT,
                D.IN,
                D.IN,
                D.OUT,
                D.IN,
                D.IN,
                D.OUT,
                D.OUT,
                D.OUT,
                D.OUT,
                D.OUT,
                D.OUT,
                D.OUT,
            ],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "MATMUL",
                "source": "kernels/aic/kernel_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "name": "ADD",
                "source": "kernels/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "name": "MUL",
                "source": "kernels/aiv/kernel_mul.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 3,
                "name": "ADD_STANDALONE",
                "source": "kernels/aiv/kernel_add_standalone.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 4,
                "name": "MUL_STANDALONE",
                "source": "kernels/aiv/kernel_mul_standalone.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "case1",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {"num_iters": 4},
        },
        {
            "name": "case2",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "manual": True,
            "params": {"num_iters": 1},
        },
    ]

    def generate_args(self, params):
        num_iters = params["num_iters"]
        torch.manual_seed(42)

        A = (torch.randn(MATMUL_SIZE, MATMUL_SIZE, dtype=torch.float32) * 0.01).flatten()
        B = (torch.randn(MATMUL_SIZE, MATMUL_SIZE, dtype=torch.float32) * 0.01).flatten()
        D = torch.randn(TILE_ELEMS, dtype=torch.float32) * 0.01
        E = torch.randn(TILE_ELEMS, dtype=torch.float32) * 0.01
        G = torch.randn(TILE_ELEMS, dtype=torch.float32) * 0.01
        H = torch.randn(TILE_ELEMS, dtype=torch.float32) * 0.01

        def zeros():
            return torch.zeros(num_iters * TILE_ELEMS, dtype=torch.float32)

        return TaskArgsBuilder(
            Tensor("A", A),
            Tensor("B", B),
            Tensor("C", zeros()),
            Tensor("D", D),
            Tensor("E", E),
            Tensor("F", zeros()),
            Tensor("G", G),
            Tensor("H", H),
            Tensor("I", zeros()),
            Tensor("J", zeros()),
            Tensor("K", zeros()),
            Tensor("L", zeros()),
            Tensor("M", zeros()),
            Tensor("N", zeros()),
            Tensor("O", zeros()),
        )

    def compute_golden(self, args, params):
        num_iters = params["num_iters"]

        A = args.A.reshape(MATMUL_SIZE, MATMUL_SIZE)
        B = args.B.reshape(MATMUL_SIZE, MATMUL_SIZE)

        golden_matmul = torch.matmul(A, B).flatten()
        golden_add = args.D + args.E
        golden_mul = args.G * args.H

        for name in ["C", "J", "N"]:
            out = getattr(args, name).reshape(num_iters, TILE_ELEMS)
            for i in range(num_iters):
                out[i] = golden_matmul

        for name in ["F", "K", "L", "O"]:
            out = getattr(args, name).reshape(num_iters, TILE_ELEMS)
            for i in range(num_iters):
                out[i] = golden_add

        for name in ["I", "M"]:
            out = getattr(args, name).reshape(num_iters, TILE_ELEMS)
            for i in range(num_iters):
                out[i] = golden_mul


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
