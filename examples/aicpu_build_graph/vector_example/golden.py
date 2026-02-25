"""
Golden script for aicpu_build_graph example.

Computation:
    f = (a + b + 1) * (a + b + 2)
    where a=2.0, b=3.0, so f=42.0

Args layout: [ptr_a, ptr_b, ptr_f, SIZE]
"""

import ctypes
import torch

__outputs__ = ["f"]

RTOL = 1e-5
ATOL = 1e-5


def generate_inputs(params: dict) -> list:
    ROWS = 128
    COLS = 128
    SIZE = ROWS * COLS

    a = torch.full((SIZE,), 2.0, dtype=torch.float32)
    b = torch.full((SIZE,), 3.0, dtype=torch.float32)
    f = torch.zeros(SIZE, dtype=torch.float32)

    return [
        ("a", a),
        ("b", b),
        ("f", f),
        ("SIZE", ctypes.c_int64(SIZE)),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    a = tensors["a"]
    b = tensors["b"]
    tensors["f"][:] = (a + b + 1) * (a + b + 2)
