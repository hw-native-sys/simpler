#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""BGEMM — aicpu_build_graph runtime with tiled matrix multiplication.

Computation: C = A @ B (4x4x4 grid, 64x64 tiles).
Tests AIC (Cube) + AIV (Vector) cooperation with tile-first memory layout.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

TILE_M = 64
TILE_K = 64
TILE_N = 64
GRID_M = 4
GRID_K = 4
GRID_N = 4
BATCH = 1


@scene_test(level=2, runtime="aicpu_build_graph")
class TestBgemm(SceneTestCase):
    """BGEMM: tiled C = A @ B with AIC gemm + AIV tile add."""

    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/bgemm_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aic/kernel_gemm_tile.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": "kernels/aiv/kernel_tile_add.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.IN],
            },
        ],
    }

    CASES = [
        {
            "name": "default",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {},
        },
    ]

    def generate_args(self, params):
        A = torch.randn(BATCH, GRID_M, GRID_K, TILE_M, TILE_K, dtype=torch.float32) * 0.01
        B = torch.randn(BATCH, GRID_K, GRID_N, TILE_K, TILE_N, dtype=torch.float32) * 0.01
        C = torch.zeros(BATCH, GRID_M, GRID_N, TILE_M, TILE_N, dtype=torch.float32)

        return TaskArgsBuilder(
            Tensor("A", A.flatten()),
            Tensor("B", B.flatten()),
            Tensor("C", C.flatten()),
        )

    def compute_golden(self, args, params):
        A = args.A.reshape(BATCH, GRID_M, GRID_K, TILE_M, TILE_K)
        B = args.B.reshape(BATCH, GRID_K, GRID_N, TILE_K, TILE_N)
        C = args.C.reshape(BATCH, GRID_M, GRID_N, TILE_M, TILE_N)

        C[:] = 0.0
        for batch in range(BATCH):
            for m_idx in range(GRID_M):
                for n_idx in range(GRID_N):
                    for k_idx in range(GRID_K):
                        C[batch, m_idx, n_idx] += torch.matmul(A[batch, m_idx, k_idx], B[batch, k_idx, n_idx])


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
