"""
Golden test specification for alternating matmul-add test.

Computation:
- M independent matmul tasks per batch: C[b,m] = A[b,m] @ B[b,m] (128x128x128)
- N independent add tasks per batch: Z[b,n] = X[b,n] + Y[b,n] (64x128)

Args layout: [ptr_A, ptr_B, ptr_C, ptr_X, ptr_Y, ptr_Z,
              size_A, size_B, size_C, size_X, size_Y, size_Z, ptr_config]
"""

import ctypes
import torch
import time

__outputs__ = ["C", "Z"]
RTOL = 1e-3
ATOL = 1e-3

ALL_CASES = {
    "case1": {
        "batch": 32,
        "M": 1,  # Number of matmul tasks per batch
        "N": 1,  # Number of add tasks per batch
        "random_seed": False,  # False = use fixed seed (42), True = random seed
    },
    "case2": {
        "batch": 64,
        "M": 1,
        "N": 1,
        "random_seed": True,
    },
}

DEFAULT_CASE = "case1"


def generate_inputs(params: dict) -> list:
    """Generate input tensors with configurable batch and task counts."""
    batch = params["batch"]
    M = params["M"]
    N = params["N"]
    random_seed = params.get("random_seed", False)

    # Validate parameters
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}")
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")

    # Prevent integer overflow in orchestration (task_idx = b * M + m or b * N + n)
    INT32_MAX = 2**31 - 1
    if batch * M > INT32_MAX:
        raise ValueError(f"batch * M = {batch * M} exceeds INT32_MAX ({INT32_MAX}), risk of overflow")
    if batch * N > INT32_MAX:
        raise ValueError(f"batch * N = {batch * N} exceeds INT32_MAX ({INT32_MAX}), risk of overflow")

    # Fixed sizes: matmul 128x128x128, add 64x128
    matmul_size = 128
    add_rows = 64
    add_cols = 128

    # Prevent excessive memory allocation
    total_matmul_elements = batch * M * matmul_size * matmul_size
    total_add_elements = batch * N * add_rows * add_cols

    # Limit single tensor to 10GB (adjustable based on system)
    MAX_TENSOR_GB = 10
    MAX_ELEMENTS = MAX_TENSOR_GB * 1024**3 // 4  # 4 bytes per float32

    matmul_gb = total_matmul_elements * 4 / 1024**3
    add_gb = total_add_elements * 4 / 1024**3

    if total_matmul_elements > MAX_ELEMENTS:
        raise ValueError(f"Matmul tensor too large: {matmul_gb:.2f} GB (max {MAX_TENSOR_GB} GB)")
    if total_add_elements > MAX_ELEMENTS:
        raise ValueError(f"Add tensor too large: {add_gb:.2f} GB (max {MAX_TENSOR_GB} GB)")

    # If random_seed is False, use fixed seed (42); if True, use random seed
    if not random_seed:
        seed = 42
    else:
        seed = int(time.time() * 1000) % (2**31)
    torch.manual_seed(seed)

    # Matmul tensors: 128x128x128
    A = torch.randn(batch, M, matmul_size, matmul_size, dtype=torch.float32) * 0.01
    B = torch.randn(batch, M, matmul_size, matmul_size, dtype=torch.float32) * 0.01
    C = torch.zeros(batch, M, matmul_size, matmul_size, dtype=torch.float32)

    # Add tensors: 64x128
    X = torch.randn(batch, N, add_rows, add_cols, dtype=torch.float32) * 0.01
    Y = torch.randn(batch, N, add_rows, add_cols, dtype=torch.float32) * 0.01
    Z = torch.zeros(batch, N, add_rows, add_cols, dtype=torch.float32)

    A_flat = A.flatten()
    B_flat = B.flatten()
    C_flat = C.flatten()
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()

    config = torch.tensor([batch, M, N], dtype=torch.int64)

    return [
        ("A", A_flat),
        ("B", B_flat),
        ("C", C_flat),
        ("X", X_flat),
        ("Y", Y_flat),
        ("Z", Z_flat),
        ("size_A", ctypes.c_int64(A_flat.nbytes)),
        ("size_B", ctypes.c_int64(B_flat.nbytes)),
        ("size_C", ctypes.c_int64(C_flat.nbytes)),
        ("size_X", ctypes.c_int64(X_flat.nbytes)),
        ("size_Y", ctypes.c_int64(Y_flat.nbytes)),
        ("size_Z", ctypes.c_int64(Z_flat.nbytes)),
        ("config", config),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    """Compute golden results for matmul and add operations."""
    batch = params["batch"]
    M = params["M"]
    N = params["N"]

    # Fixed sizes: matmul 128x128x128, add 64x128
    matmul_size = 128
    add_rows = 64
    add_cols = 128

    A = torch.as_tensor(tensors["A"]).reshape(batch, M, matmul_size, matmul_size)
    B = torch.as_tensor(tensors["B"]).reshape(batch, M, matmul_size, matmul_size)
    C = torch.as_tensor(tensors["C"]).reshape(batch, M, matmul_size, matmul_size)

    X = torch.as_tensor(tensors["X"]).reshape(batch, N, add_rows, add_cols)
    Y = torch.as_tensor(tensors["Y"]).reshape(batch, N, add_rows, add_cols)
    Z = torch.as_tensor(tensors["Z"]).reshape(batch, N, add_rows, add_cols)

    for b in range(batch):
        for m in range(M):
            C[b, m] = torch.matmul(A[b, m], B[b, m])

    for b in range(batch):
        for n in range(N):
            Z[b, n] = X[b, n] + Y[b, n]

    tensors["C"][:] = C.flatten()
    tensors["Z"][:] = Z.flatten()
