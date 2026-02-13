"""
Golden test specification for BGEMM (Host Build Graph Runtime).

Computation: C = A @ B (tiled matrix multiplication)
Configuration: 4x4x4 grid, 64x64 tiles
"""

import numpy as np

__outputs__ = ["C"]
TENSOR_ORDER = ["A", "B", "C"]
RTOL = 1e-3
ATOL = 1e-3

TILE_M = 64
TILE_K = 64
TILE_N = 64

GRID_M = 4
GRID_K = 4
GRID_N = 4
BATCH = 1

M = TILE_M * GRID_M
K = TILE_K * GRID_K
N = TILE_N * GRID_N


def generate_inputs(params: dict) -> dict:
    """Generate input tensors with tile-first memory layout."""
    A = np.random.randn(BATCH, GRID_M, GRID_K, TILE_M, TILE_K).astype(np.float32) * 0.01
    B = np.random.randn(BATCH, GRID_K, GRID_N, TILE_K, TILE_N).astype(np.float32) * 0.01
    C = np.zeros((BATCH, GRID_M, GRID_N, TILE_M, TILE_N), dtype=np.float32)

    return {
        "A": A.flatten(),
        "B": B.flatten(),
        "C": C.flatten(),
    }


def compute_golden(tensors: dict, params: dict) -> None:
    """Compute golden result: C[m,n] = sum(k) A[m,k] @ B[k,n]."""
    A = tensors["A"].reshape(BATCH, GRID_M, GRID_K, TILE_M, TILE_K)
    B = tensors["B"].reshape(BATCH, GRID_K, GRID_N, TILE_K, TILE_N)
    C = tensors["C"].reshape(BATCH, GRID_M, GRID_N, TILE_M, TILE_N)

    C[:] = 0.0

    for batch in range(BATCH):
        for m_idx in range(GRID_M):
            for n_idx in range(GRID_N):
                for k_idx in range(GRID_K):
                    C[batch, m_idx, n_idx] += np.matmul(
                        A[batch, m_idx, k_idx],
                        B[batch, k_idx, n_idx]
                    )

    tensors["C"][:] = C.flatten()
