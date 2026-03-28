"""
Golden script for AscendC vector example.

Computation:
    z = x + y          (AscendC AddCustom)
    w = z * z          (PTO kernel_mul)

With x=2.0, y=3.0:
    z = 5.0
    w = 25.0

Args layout: [ptr_x, ptr_y, ptr_z, ptr_w, size_x, size_y, size_z, size_w, SIZE]
"""

import ctypes
import torch

__outputs__ = ["z", "w"]

RTOL = 1e-5
ATOL = 1e-5


def generate_inputs(params: dict) -> list:
    ROWS = 128
    COLS = 128
    SIZE = ROWS * COLS

    x = torch.full((SIZE,), 2.0, dtype=torch.float32)
    y = torch.full((SIZE,), 3.0, dtype=torch.float32)
    z = torch.zeros(SIZE, dtype=torch.float32)
    w = torch.zeros(SIZE, dtype=torch.float32)

    return [
        ("x", x),
        ("y", y),
        ("z", z),
        ("w", w),
        ("size_x", ctypes.c_int64(x.nbytes)),
        ("size_y", ctypes.c_int64(y.nbytes)),
        ("size_z", ctypes.c_int64(z.nbytes)),
        ("size_w", ctypes.c_int64(w.nbytes)),
        ("SIZE", ctypes.c_int64(SIZE)),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    x = torch.as_tensor(tensors["x"])
    y = torch.as_tensor(tensors["y"])
    z_val = x + y
    tensors["z"][:] = z_val
    tensors["w"][:] = z_val * z_val
