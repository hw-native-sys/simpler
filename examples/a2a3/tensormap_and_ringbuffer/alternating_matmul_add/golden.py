# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Golden test specification for alternating matmul-add test.

Computation:
- M independent matmul tasks per batch: C[b,m] = A[b,m] @ B[b,m] (128x128x128)
- N independent add tasks per batch: Z[b,n] = X[b,n] + Y[b,n] (128x128)

Args layout: [A, B, C, X, Y, Z, batch, M_val, N_val, matmul_batch, add_batch]
  Tensors retain original shapes; config values as scalars.
"""

import ctypes
import time

import torch

__outputs__ = ["C", "Z"]
RTOL = 1e-3
ATOL = 1e-3

ALL_CASES = {
    "Case1": {
        "batch": 500,
        "M": 4,
        "N": 4,
        "random_seed": False,
        "matmul_batch": 4,
        "add_batch": 4,
    },
    "Case2": {
        "batch": 512,
        "M": 2,  # Number of matmul tasks per batch
        "N": 5,  # Number of add tasks per batch
        "random_seed": True,  # False = use fixed seed (42), True = random seed
        "matmul_batch": 4,  # Number of matmul tiles per task
        "add_batch": 5,  # Number of add tiles per task
    },
}

DEFAULT_CASE = "Case1"


def generate_inputs(params: dict) -> list:
    """Generate input tensors with configurable batch and task counts."""
    batch = params["batch"]
    M = params["M"]
    N = params["N"]
    random_seed = params.get("random_seed", False)
    matmul_batch = params.get("matmul_batch", 1)
    add_batch = params.get("add_batch", 1)

    # Validate parameters
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}")
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    if matmul_batch <= 0:
        raise ValueError(f"matmul_batch must be positive, got {matmul_batch}")
    if add_batch <= 0:
        raise ValueError(f"add_batch must be positive, got {add_batch}")

    # Validate divisibility for task grouping
    total_matmul_tasks = batch * M
    total_add_tasks = batch * N

    if total_matmul_tasks % matmul_batch != 0:
        raise ValueError(
            f"total_matmul_tasks ({total_matmul_tasks}) must be divisible by matmul_batch ({matmul_batch})"
        )
    if total_add_tasks % add_batch != 0:
        raise ValueError(f"total_add_tasks ({total_add_tasks}) must be divisible by add_batch ({add_batch})")

    # Prevent integer overflow in orchestration (task_idx = b * M + m or b * N + n)
    INT32_MAX = 2**31 - 1
    if total_matmul_tasks > INT32_MAX:
        raise ValueError(f"total_matmul_tasks ({total_matmul_tasks}) exceeds INT32_MAX ({INT32_MAX}), risk of overflow")
    if total_add_tasks > INT32_MAX:
        raise ValueError(f"total_add_tasks ({total_add_tasks}) exceeds INT32_MAX ({INT32_MAX}), risk of overflow")

    # Fixed sizes: matmul 128x128x128, add 128x128
    matmul_size = 128
    add_rows = 128
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

    # Add tensors: 128x128
    X = torch.randn(batch, N, add_rows, add_cols, dtype=torch.float32) * 0.01
    Y = torch.randn(batch, N, add_rows, add_cols, dtype=torch.float32) * 0.01
    Z = torch.zeros(batch, N, add_rows, add_cols, dtype=torch.float32)

    A_flat = A.flatten()
    B_flat = B.flatten()
    C_flat = C.flatten()
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()

    return [
        ("A", A_flat),
        ("B", B_flat),
        ("C", C_flat),
        ("X", X_flat),
        ("Y", Y_flat),
        ("Z", Z_flat),
        ("batch", ctypes.c_int64(batch)),
        ("M_val", ctypes.c_int64(M)),
        ("N_val", ctypes.c_int64(N)),
        ("matmul_batch", ctypes.c_int64(matmul_batch)),
        ("add_batch", ctypes.c_int64(add_batch)),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    """Compute golden results for matmul and add operations."""
    batch = params["batch"]
    M = params["M"]
    N = params["N"]

    # Fixed sizes: matmul 128x128x128, add 128x128
    matmul_size = 128
    add_rows = 128
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
