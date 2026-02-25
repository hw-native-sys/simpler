"""
Golden script for matmul example.

Computation:
    F = exp(sqrt(log(A)) @ W1 + sqrt(log(A)) @ W2)

    Task graph (diamond topology):
          t0: B = sqrt(log(A))      [AIV, half->half]
         /  \
       t1     t2                    [AIC, half*half->float]
       |       |
    C=B@W1  D=B@W2
         \  /
          t3: F = exp(C + D)        [AIV, float->float]

    where A = e^4 (128x128, float16), W1 = W2 = 1/256 (128x128, float16)
    Result: F = exp(2) ~ 7.389

Args layout: [ptr_a, ptr_w1, ptr_w2, ptr_f, size_a, size_w1, size_w2, size_f, SIZE]
"""

import ctypes
import torch

__outputs__ = ["f"]

RTOL = 1e-2
ATOL = 1e-2


def generate_inputs(params: dict) -> list:
    ROWS = 128
    COLS = 128
    SIZE = ROWS * COLS

    input_value = torch.exp(torch.tensor(4.0)).item()
    weight_value = 1.0 / (2 * COLS)

    a = torch.full((SIZE,), input_value, dtype=torch.float16)
    w1 = torch.full((SIZE,), weight_value, dtype=torch.float16)
    w2 = torch.full((SIZE,), weight_value, dtype=torch.float16)
    f = torch.zeros(SIZE, dtype=torch.float32)

    return [
        ("a", a),
        ("w1", w1),
        ("w2", w2),
        ("f", f),
        ("size_a", ctypes.c_int64(a.nbytes)),
        ("size_w1", ctypes.c_int64(w1.nbytes)),
        ("size_w2", ctypes.c_int64(w2.nbytes)),
        ("size_f", ctypes.c_int64(f.nbytes)),
        ("SIZE", ctypes.c_int64(SIZE)),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    ROWS = 128
    COLS = 128

    a = torch.as_tensor(tensors["a"]).reshape(ROWS, COLS).to(torch.float32)
    w1 = torch.as_tensor(tensors["w1"]).reshape(ROWS, COLS).to(torch.float32)
    w2 = torch.as_tensor(tensors["w2"]).reshape(ROWS, COLS).to(torch.float32)

    b = torch.sqrt(torch.log(a))
    c = torch.matmul(b, w1)
    d = torch.matmul(b, w2)
    f = torch.exp(c + d)

    tensors["f"][:] = f.flatten().to(torch.float32)
