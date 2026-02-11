"""
Golden script for aicpu_build_graph example.

Computation:
    f = (a + b + 1) * (a + b + 2)
    where a=2.0, b=3.0, so f=42.0
"""

import torch

__outputs__ = ["f"]

# Args layout (host orchestration): [ptr_a, ptr_b, ptr_f, size_a, size_b, size_f, SIZE]
TENSOR_ORDER = ["a", "b", "f"]

RTOL = 1e-5
ATOL = 1e-5


def generate_inputs(params: dict) -> dict:
    ROWS = 128
    COLS = 128
    SIZE = ROWS * COLS

    return {
        "a": torch.full((SIZE,), 2.0, dtype=torch.float32),
        "b": torch.full((SIZE,), 3.0, dtype=torch.float32),
        "f": torch.zeros(SIZE, dtype=torch.float32),
    }


def compute_golden(tensors: dict, params: dict) -> None:
    a = tensors["a"]
    b = tensors["b"]
    tensors["f"][:] = (a + b + 1) * (a + b + 2)

