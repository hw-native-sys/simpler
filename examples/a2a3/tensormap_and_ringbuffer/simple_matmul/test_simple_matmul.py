#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Test simple matrix multiplication kernel  C = torch.matmul(A, B)."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSimpleMatmul(SceneTestCase):
    RTOL = 1e-5
    ATOL = 1e-5

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/simple_matmul_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "pto_simple_matmul",
                "source": "kernels/aic/kernel_simple_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            }
        ],
    }

    CASES = [
         {
            "name": "MatMul_4_32_32",
            "platforms": ["a2a3sim", "a2a3", "a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 16},
            "params": {"batch_dim": 4, "matrix_size": 32},
        },
        {
            "name": "MatMul_16_32_32",
            "platforms": ["a2a3sim", "a2a3", "a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 16},
            "params": {"batch_dim": 4, "matrix_size": 128},
        },
        {
            "name": "MatMul_16_128_128",
            "platforms": ["a2a3sim", "a2a3", "a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 16},
            "params": {"batch_dim": 16, "matrix_size": 128},
        },
        {
            "name": "MatMul_8_128_128",
            "platforms": ["a2a3sim", "a2a3", "a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 16},
            "params": {"batch_dim": 8, "matrix_size": 128},
        },
    ]

    def generate_args(self, params):
        matrix_size = params["matrix_size"]
        batch_dim = params["batch_dim"]
        A = torch.randn(batch_dim, matrix_size, matrix_size, dtype=torch.float32)
        B = torch.randn(batch_dim, matrix_size, matrix_size, dtype=torch.float32)
        C = torch.zeros(batch_dim, matrix_size, matrix_size, dtype=torch.float32)
        config = torch.tensor([matrix_size, batch_dim], dtype=torch.int64)
        return TaskArgsBuilder(
            Tensor("A", A.flatten()),
            Tensor("B", B.flatten()),
            Tensor("C", C.flatten()),
            Tensor("config", config),
        )

    def compute_golden(self, args, params):
        matrix_size = params["matrix_size"]
        batch_dim = params["batch_dim"]
        A = args.A.reshape(batch_dim, matrix_size, matrix_size)
        B = args.B.reshape(batch_dim, matrix_size, matrix_size)
        C = A @ B
        return C


if __name__ == "__main__":
    TestSimpleMatmul.run_module(__name__)
