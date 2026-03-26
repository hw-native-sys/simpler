"""
Golden script for tensormap_and_ringbuffer example.

Computation:
    f = (a + b + 1) * (a + b + 2) + (a + b)
    where a=2.0, b=3.0, so f=47.0

Args layout: [a, b, f]  — shape/dtype/size in TaskArg metadata
"""

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
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    a = torch.as_tensor(tensors["a"])
    b = torch.as_tensor(tensors["b"])
    tensors["f"][:] = (a + b + 1) * (a + b + 2) + (a + b)
