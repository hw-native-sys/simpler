#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Mixed AIC+AIV example: all 5 resource shapes per iteration.

Args layout (15 tensors): [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O]
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
                "source": "kernels/aic/kernel_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": "kernels/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "source": "kernels/aiv/kernel_mul.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 3,
                "source": "kernels/aiv/kernel_add_standalone.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 4,
                "source": "kernels/aiv/kernel_mul_standalone.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "case1",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {"num_iters": 4},
        },
        {
            "name": "case2",
            "manual": True,
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {"num_iters": 1},
        },
    ]

    def generate_args(self, params):
        num_iters = params["num_iters"]
        torch.manual_seed(42)
        A = torch.randn(MATMUL_SIZE, MATMUL_SIZE, dtype=torch.float32) * 0.01
        B = torch.randn(MATMUL_SIZE, MATMUL_SIZE, dtype=torch.float32) * 0.01
        D_t = torch.randn(TILE_ELEMS, dtype=torch.float32) * 0.01
        E = torch.randn(TILE_ELEMS, dtype=torch.float32) * 0.01
        G = torch.randn(TILE_ELEMS, dtype=torch.float32) * 0.01
        H = torch.randn(TILE_ELEMS, dtype=torch.float32) * 0.01

        def z():
            return torch.zeros(num_iters * TILE_ELEMS, dtype=torch.float32)

        return TaskArgsBuilder(
            Tensor("A", A.flatten()),
            Tensor("B", B.flatten()),
            Tensor("C", z()),
            Tensor("D", D_t),
            Tensor("E", E),
            Tensor("F", z()),
            Tensor("G", G),
            Tensor("H", H),
            Tensor("I", z()),
            Tensor("J", z()),
            Tensor("K", z()),
            Tensor("L", z()),
            Tensor("M", z()),
            Tensor("N", z()),
            Tensor("O", z()),
        )

    def compute_golden(self, args, params):
        num_iters = params["num_iters"]
        golden_matmul = torch.matmul(
            args.A.reshape(MATMUL_SIZE, MATMUL_SIZE), args.B.reshape(MATMUL_SIZE, MATMUL_SIZE)
        ).flatten()
        golden_add = args.D + args.E
        golden_mul = args.G * args.H
        for t in [args.C, args.J, args.N]:
            out = t.reshape(num_iters, TILE_ELEMS)
            for i in range(num_iters):
                out[i] = golden_matmul
        for t in [args.F, args.K, args.L, args.O]:
            out = t.reshape(num_iters, TILE_ELEMS)
            for i in range(num_iters):
                out[i] = golden_add
        for t in [args.I, args.M]:
            out = t.reshape(num_iters, TILE_ELEMS)
            for i in range(num_iters):
                out[i] = golden_mul


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
