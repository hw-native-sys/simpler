# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Golden test specification for BGEMM (tensormap_and_ringbuffer Runtime).

Computation: C = A @ B (tiled matrix multiplication)
Configuration: 4x4x4 grid, 64x64 tiles

Args layout: [A, B, C]  — shape/dtype/size in ContinuousTensor metadata
"""

import torch

__outputs__ = ["C"]
RTOL = 1e-3
ATOL = 1e-3

TILE_M = 64
TILE_K = 64
TILE_N = 64

GRID_M = 4
GRID_K = 4
GRID_N = 4
BATCH = 2

M = TILE_M * GRID_M
K = TILE_K * GRID_K
N = TILE_N * GRID_N


def generate_inputs(params: dict) -> list:
    """Generate input tensors with tile-first memory layout."""
    A = torch.randn(BATCH, GRID_M, GRID_K, TILE_M, TILE_K, dtype=torch.float32) * 0.01
    B = torch.randn(BATCH, GRID_K, GRID_N, TILE_K, TILE_N, dtype=torch.float32) * 0.01
    C = torch.zeros(BATCH, GRID_M, GRID_N, TILE_M, TILE_N, dtype=torch.float32)

    A_flat = A.flatten()
    B_flat = B.flatten()
    C_flat = C.flatten()

    return [
        ("A", A_flat),
        ("B", B_flat),
        ("C", C_flat),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    """Compute golden result: C[m,n] = sum(k) A[m,k] @ B[k,n]."""
    A = torch.as_tensor(tensors["A"]).reshape(BATCH, GRID_M, GRID_K, TILE_M, TILE_K)
    B = torch.as_tensor(tensors["B"]).reshape(BATCH, GRID_K, GRID_N, TILE_K, TILE_N)
    C = torch.as_tensor(tensors["C"]).reshape(BATCH, GRID_M, GRID_N, TILE_M, TILE_N)

    C[:] = 0.0

    for batch in range(BATCH):
        for m_idx in range(GRID_M):
            for n_idx in range(GRID_N):
                for k_idx in range(GRID_K):
                    C[batch, m_idx, n_idx] += torch.matmul(A[batch, m_idx, k_idx], B[batch, k_idx, n_idx])

    tensors["C"][:] = C.flatten()
