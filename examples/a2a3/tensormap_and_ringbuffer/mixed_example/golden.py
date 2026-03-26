"""
Golden test specification for mixed AIC+AIV example.

Covers all 5 resource shapes per iteration:
  1. AIC_AIV_X2: C = A@B, F = D+E, I = G*H
  2. AIC_ONLY:   J = A@B
  3. AIV_X1:     K = D+E
  4. AIV_X2:     L = D+E, M = G*H
  5. AIC_AIV_X1: N = A@B, O = D+E

All use 128x128 float32 tiles, repeated over num_iters slices.

Args layout (15 args): [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O]
  Shape/dtype/size in TaskArg metadata.
"""

import torch

__outputs__ = ["C", "F", "I", "J", "K", "L", "M", "N", "O"]
RTOL = 1e-3
ATOL = 1e-3

ALL_CASES = {
    "case1": {"num_iters": 4},
    "case2": {"num_iters": 1},
}

DEFAULT_CASE = "case1"

MATMUL_SIZE = 128
TILE_ELEMS = 128 * 128


def generate_inputs(params: dict) -> list:
    num_iters = params["num_iters"]

    torch.manual_seed(42)

    # Matmul inputs (shared by AIC tasks)
    A = torch.randn(MATMUL_SIZE, MATMUL_SIZE, dtype=torch.float32) * 0.01
    B = torch.randn(MATMUL_SIZE, MATMUL_SIZE, dtype=torch.float32) * 0.01

    # Add inputs (shared by AIV add tasks)
    D = torch.randn(TILE_ELEMS, dtype=torch.float32) * 0.01
    E = torch.randn(TILE_ELEMS, dtype=torch.float32) * 0.01

    # Mul inputs (shared by AIV mul tasks)
    G = torch.randn(TILE_ELEMS, dtype=torch.float32) * 0.01
    H = torch.randn(TILE_ELEMS, dtype=torch.float32) * 0.01

    # Output buffers (num_iters slices each)
    C = torch.zeros(num_iters * TILE_ELEMS, dtype=torch.float32)
    F = torch.zeros(num_iters * TILE_ELEMS, dtype=torch.float32)
    I = torch.zeros(num_iters * TILE_ELEMS, dtype=torch.float32)
    J = torch.zeros(num_iters * TILE_ELEMS, dtype=torch.float32)
    K = torch.zeros(num_iters * TILE_ELEMS, dtype=torch.float32)
    L = torch.zeros(num_iters * TILE_ELEMS, dtype=torch.float32)
    M = torch.zeros(num_iters * TILE_ELEMS, dtype=torch.float32)
    N = torch.zeros(num_iters * TILE_ELEMS, dtype=torch.float32)
    O = torch.zeros(num_iters * TILE_ELEMS, dtype=torch.float32)

    A_flat = A.flatten()
    B_flat = B.flatten()

    return [
        ("A", A_flat),
        ("B", B_flat),
        ("C", C),
        ("D", D),
        ("E", E),
        ("F", F),
        ("G", G),
        ("H", H),
        ("I", I),
        ("J", J),
        ("K", K),
        ("L", L),
        ("M", M),
        ("N", N),
        ("O", O),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    num_iters = params["num_iters"]

    A = torch.as_tensor(tensors["A"]).reshape(MATMUL_SIZE, MATMUL_SIZE)
    B = torch.as_tensor(tensors["B"]).reshape(MATMUL_SIZE, MATMUL_SIZE)
    D = torch.as_tensor(tensors["D"])
    E = torch.as_tensor(tensors["E"])
    G = torch.as_tensor(tensors["G"])
    H = torch.as_tensor(tensors["H"])

    golden_matmul = torch.matmul(A, B).flatten()
    golden_add = D + E
    golden_mul = G * H

    for name in ["C", "J", "N"]:
        out = torch.as_tensor(tensors[name]).reshape(num_iters, TILE_ELEMS)
        for i in range(num_iters):
            out[i] = golden_matmul
        tensors[name][:] = out.flatten()

    for name in ["F", "K", "L", "O"]:
        out = torch.as_tensor(tensors[name]).reshape(num_iters, TILE_ELEMS)
        for i in range(num_iters):
            out[i] = golden_add
        tensors[name][:] = out.flatten()

    for name in ["I", "M"]:
        out = torch.as_tensor(tensors[name]).reshape(num_iters, TILE_ELEMS)
        for i in range(num_iters):
            out[i] = golden_mul
        tensors[name][:] = out.flatten()
