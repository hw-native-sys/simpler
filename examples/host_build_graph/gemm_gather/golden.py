"""
Golden test for gemm_gather (a2a3).

GEMM: C = A @ B, 64x64 float.
Gather: out = src0[src1] (linear index), src0 (32, 1024), src1 (16, 64) int32, out (16, 64) float.
Args: [A, B, C, src0, src1, out, size_A, size_B, size_C, size_src0, size_src1, size_out]
code_runner passes torch tensors to compute_golden; use .numel() and .item() for compatibility.
"""

import ctypes
import numpy as np
import torch

__outputs__ = ["C", "out"]
RTOL = 1e-3
ATOL = 1e-3

# GEMM: single 64x64 tile
GEMM_TILE = 64

# Gather: comm-isa shape
GATHER_SRC0_ROWS = 32
GATHER_SRC0_COLS = 1024
GATHER_SRC1_ROWS = 16
GATHER_SRC1_COLS = 64


def generate_inputs(params: dict) -> list:
    np.random.seed(42)

    # GEMM: A, B, C (64, 64) float
    A = np.random.randn(GEMM_TILE, GEMM_TILE).astype(np.float32) * 0.01
    B = np.random.randn(GEMM_TILE, GEMM_TILE).astype(np.float32) * 0.01
    C = np.zeros((GEMM_TILE, GEMM_TILE), dtype=np.float32)

    # Gather: src0 (32, 1024), src1 (16, 64) int32 indices, out (16, 64)
    src0 = np.random.randn(GATHER_SRC0_ROWS, GATHER_SRC0_COLS).astype(np.float32) * 0.01
    src0_flat = src0.flatten()
    max_idx = src0_flat.size - 1
    src1 = np.random.randint(0, max_idx + 1, size=(GATHER_SRC1_ROWS, GATHER_SRC1_COLS), dtype=np.int32)
    out = np.zeros((GATHER_SRC1_ROWS, GATHER_SRC1_COLS), dtype=np.float32)

    A_flat = A.flatten()
    B_flat = B.flatten()
    C_flat = C.flatten()
    src0_flat = src0.flatten()
    src1_flat = src1.flatten()
    out_flat = out.flatten()

    return [
        ("A", A_flat),
        ("B", B_flat),
        ("C", C_flat),
        ("src0", src0_flat),
        ("src1", src1_flat),
        ("out", out_flat),
        ("size_A", ctypes.c_int64(A_flat.nbytes)),
        ("size_B", ctypes.c_int64(B_flat.nbytes)),
        ("size_C", ctypes.c_int64(C_flat.nbytes)),
        ("size_src0", ctypes.c_int64(src0_flat.nbytes)),
        ("size_src1", ctypes.c_int64(src1_flat.nbytes)),
        ("size_out", ctypes.c_int64(out_flat.nbytes)),
    ]


def _numel(x):
    """Element count for both numpy and torch (code_runner passes torch tensors)."""
    if hasattr(x, "numel") and callable(x.numel):
        return x.numel()
    return int(x.size)


def compute_golden(tensors: dict, params: dict) -> None:
    A = tensors["A"].reshape(GEMM_TILE, GEMM_TILE)
    B = tensors["B"].reshape(GEMM_TILE, GEMM_TILE)
    C = tensors["C"].reshape(GEMM_TILE, GEMM_TILE)
    src0 = tensors["src0"].reshape(GATHER_SRC0_ROWS, GATHER_SRC0_COLS)
    src1 = tensors["src1"].reshape(GATHER_SRC1_ROWS, GATHER_SRC1_COLS)
    out = tensors["out"].reshape(GATHER_SRC1_ROWS, GATHER_SRC1_COLS)

    # GEMM: C = A @ B (tensors from code_runner are torch)
    C.copy_(torch.matmul(A.float(), B.float()))

    # Gather: out[i,j] = src0.flatten()[src1[i,j]]
    src0_flat = src0.flatten()
    n_src0 = _numel(src0_flat)
    for i in range(GATHER_SRC1_ROWS):
        for j in range(GATHER_SRC1_COLS):
            idx = int(src1[i, j].item() if hasattr(src1[i, j], "item") else src1[i, j])
            if idx < 0:
                idx = 0
            if idx >= n_src0:
                idx = n_src0 - 1
            out[i, j] = src0_flat[idx]

    tensors["C"][:] = C.flatten()
    tensors["out"][:] = out.flatten()
