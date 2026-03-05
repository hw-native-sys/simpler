"""
Golden reference for cpt_and_comm: GEMM then TGATHER.

Each rank: C = A @ B (64x64), gather first 64 elements to root.
Root output: [rank0_first64, rank1_first64, ...]
"""

import ctypes
import numpy as np

# GEMM 64x64, gather 64 elements per rank
TILE = 64
GATHER_COUNT = 64

ALL_CASES = {"Default": {}}
DEFAULT_CASE = "Default"
__outputs__ = ["out"]
RTOL = 1e-4
ATOL = 1e-4


def generate_inputs(params: dict) -> list:
    """Return flat argument list. For requires_comm, params includes device_ctx_ptr, win_base, n_ranks, root, rank_id."""
    rank_id = params.get("rank_id", 0)
    n_ranks = params.get("n_ranks", 2)
    root = params.get("root", 0)

    # A, B: 64x64 per rank (different data per rank)
    np.random.seed(42 + rank_id)
    a = np.random.randn(TILE, TILE).astype(np.float32) * 0.1
    b = np.random.randn(TILE, TILE).astype(np.float32) * 0.1
    c = np.zeros((TILE, TILE), dtype=np.float32)  # GEMM output
    out = np.zeros((n_ranks * GATHER_COUNT,), dtype=np.float32)  # root only

    result = [
        ("a", a),
        ("b", b),
        ("c", c),
        ("out", out),
        ("size_a", ctypes.c_int64(a.nbytes)),
        ("size_b", ctypes.c_int64(b.nbytes)),
        ("size_c", ctypes.c_int64(c.nbytes)),
        ("size_out", ctypes.c_int64(out.nbytes)),
    ]

    if "device_ctx_ptr" in params and "win_base" in params:
        result.extend([
            ("device_ctx_ptr", ctypes.c_uint64(params["device_ctx_ptr"])),
            ("win_base", ctypes.c_uint64(params["win_base"])),
            ("n_ranks", ctypes.c_int32(n_ranks)),
            ("root", ctypes.c_int32(root)),
            ("rank_id", ctypes.c_int32(rank_id)),
        ])

    return result


def compute_golden(tensors: dict, params: dict) -> None:
    """Compute expected: BGEMM then gather first GATHER_COUNT from each rank."""
    rank_id = params.get("rank_id", 0)
    n_ranks = params.get("n_ranks", 2)
    root = params.get("root", 0)

    a = tensors["a"]
    b = tensors["b"]
    c = tensors["c"]
    out = tensors["out"]

    # GEMM: C = A @ B
    c[:] = a @ b

    # Gather: root collects first GATHER_COUNT from each rank
    if rank_id == root:
        for r in range(n_ranks):
            # Simulate rank r's GEMM output (we only have our own, so for golden we compute all)
            np.random.seed(42 + r)
            ar = np.random.randn(TILE, TILE).astype(np.float32) * 0.1
            br = np.random.randn(TILE, TILE).astype(np.float32) * 0.1
            cr = ar @ br
            flat = cr.flatten()
            out[r * GATHER_COUNT : (r + 1) * GATHER_COUNT] = flat[:GATHER_COUNT]
