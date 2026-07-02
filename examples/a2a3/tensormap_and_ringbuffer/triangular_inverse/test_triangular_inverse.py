#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Triangular inverse (recursive unrolled): M_inv = inv(M) for upper/lower-triangular unit diagonal matrices."""

import numpy as np
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test


def random_tri_matrix(n, block_dim_x, block_dim_y, scale=0.1, is_lower=False):
    if is_lower:
        return scale * torch.tril(torch.rand((block_dim_x, block_dim_y, n, n)), diagonal=-1)
    else:
        return scale * torch.triu(torch.rand((block_dim_x, block_dim_y, n, n)), diagonal=1)


def linalg_inv(A: torch.tensor) -> torch.tensor:
    assert A.ndim == 4, "Expected 4D tensor"
    assert A.shape[-2] == A.shape[-1], "Expected square matrices on last two dimensions"
    in_dtype = A.dtype
    n = A.shape[-1]
    Identity = np.eye(n, dtype=np.double)
    golden_numpy = np.zeros(A.shape)
    for x in range(A.shape[0]):
        for y in range(A.shape[1]):
            golden_numpy[x, y] = np.linalg.inv(A[x, y].double().numpy().astype(np.double) + Identity)
    return torch.from_numpy(golden_numpy).to(in_dtype)


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestTriangularInverse(SceneTestCase):
    # fp16 arithmetic — use tolerances appropriate for half-precision results
    RTOL = 1e-5
    ATOL = 1e-2

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/triangular_inverse_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "TRI_INV",
                "source": "kernels/aic/kernel_tri_inv_rec_unroll.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT, D.IN],
            }
        ],
    }

    CASES = [
        {
            "name": "Case_upper_tri_matrix_size_16",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"num_matrices": 1, "matrix_size": 16, "is_lower": 0},
        },
        {
            "name": "Case_upper_tri_matrix_size_32",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"num_matrices": 1, "matrix_size": 32, "is_lower": 0},
        },
        {
            "name": "Case_upper_tri_matrix_size_64",
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"num_matrices": 1, "matrix_size": 64, "is_lower": 0},
        },
        {
            "name": "Case_upper_tri_matrix_size_128",
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"num_matrices": 1, "matrix_size": 128, "is_lower": 0},
        },
        {
            "name": "Case_lower_tri_matrix_size_128",
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"num_matrices": 1, "matrix_size": 128, "is_lower": 1},
        },
    ]

    def generate_args(self, params):
        matrix_size = params["matrix_size"]
        num_matrices = params["num_matrices"]
        block_dim = min(num_matrices, 24)
        is_lower = params["is_lower"]

        # Build well-conditioned triangular matrices in fp16.
        # Start with random values and zero out the off-triangle, then set
        # the diagonal to a value in [0.5, 1.5] to ensure invertibility.
        M_fp16 = random_tri_matrix(matrix_size, 1, num_matrices, is_lower=is_lower).to(torch.float16)
        I_neg = -torch.eye(matrix_size, dtype=torch.float16)
        M_inv = torch.randn((num_matrices, 1, matrix_size, matrix_size), dtype=torch.float16)
        config = torch.tensor([matrix_size, num_matrices, is_lower, block_dim], dtype=torch.int64)

        return TaskArgsBuilder(
            Tensor("M", M_fp16.flatten()),
            Tensor("I_neg", I_neg.flatten()),
            Tensor("M_inv", M_inv.flatten()),
            Tensor("config", config),
        )

    def compute_golden(self, args, params):
        n = params["matrix_size"]
        num_matrices = params["num_matrices"]
        M = args.M.reshape(1, num_matrices, n, n)
        M_inv = args.M_inv.reshape(1, num_matrices, n, n)
        M_inv[:] = linalg_inv(M)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
