"""
Golden reference for AllGather (TGATHER variant, no compute).

Each rank contributes GATHER_COUNT float32 elements.
After AllGather, EVERY rank holds the concatenation of all ranks' data.
"""

import ctypes
import numpy as np

GATHER_COUNT = 64

ALL_CASES = {"Default": {}}
DEFAULT_CASE = "Default"
__outputs__ = ["out"]
RTOL = 1e-4
ATOL = 1e-4


def generate_inputs(params: dict) -> list:
    """Return flat argument list. For requires_comm, params includes device_ctx_ptr, win_in_base, win_out_base, n_ranks, root, rank_id."""
    rank_id = params.get("rank_id", 0)
    n_ranks = params.get("n_ranks", 2)
    root = params.get("root", 0)

    np.random.seed(42 + rank_id)
    src = np.random.randn(GATHER_COUNT).astype(np.float32) * 0.1
    out = np.zeros((n_ranks * GATHER_COUNT,), dtype=np.float32)

    result = [
        ("src", src),
        ("out", out),
        ("size_src", ctypes.c_int64(src.nbytes)),
        ("size_out", ctypes.c_int64(out.nbytes)),
    ]

    if "device_ctx_ptr" in params and "win_in_base" in params and "win_out_base" in params:
        result.extend([
            ("device_ctx_ptr", ctypes.c_uint64(params["device_ctx_ptr"])),
            ("win_in_base", ctypes.c_uint64(params["win_in_base"])),
            ("win_out_base", ctypes.c_uint64(params["win_out_base"])),
            ("n_ranks", ctypes.c_int32(n_ranks)),
            ("root", ctypes.c_int32(root)),
            ("rank_id", ctypes.c_int32(rank_id)),
        ])

    return result


def compute_golden(tensors: dict, params: dict) -> None:
    """AllGather: every rank gets the full concatenation of all ranks' data."""
    n_ranks = params.get("n_ranks", 2)
    out = tensors["out"]

    out_np = out.cpu().numpy() if hasattr(out, 'cpu') else np.asarray(out)
    for r in range(n_ranks):
        np.random.seed(42 + r)
        src_r = np.random.randn(GATHER_COUNT).astype(np.float32) * 0.1
        out_np[r * GATHER_COUNT : (r + 1) * GATHER_COUNT] = src_r[:GATHER_COUNT]
