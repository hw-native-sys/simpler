"""
Golden test specification for BGEMM (tensormap_and_ringbuffer Runtime).

Computation: C = A @ B (tiled matrix multiplication)
Configuration controlled by ALL_CASES parameters:
  - incore_task_num: total number of incore tasks
  - incore_data_size: tile size (tile_size = incore_data_size)
  - grid_k: number of K-dimension partitions (batch_size = incore_task_num / grid_k)

Args layout: [ptr_A, ptr_B, ptr_C, ptr_config, size_A, size_B, size_C]
"""

import ctypes
import torch

__outputs__ = ["C"]
RTOL = 1e-3
ATOL = 1e-3

# Supported tile sizes must match the switch-case in kernel_gemm_tile.cpp (AIC)
# and kernel_tile_add.cpp (AIV), which only instantiate templates for these values.
SUPPORTED_TILE_SIZES = {16, 32, 64, 128}

ALL_CASES = {
    "Case1": {
        "incore_task_num": 64,
        "incore_data_size": 128,
        "grid_k": 2,
    },
}

DEFAULT_CASE = "Case1"


def generate_inputs(params: dict) -> list:
    """Generate input tensors with tile-first memory layout."""
    tile_size = params["incore_data_size"]
    if tile_size not in SUPPORTED_TILE_SIZES:
        raise ValueError(
            f"incore_data_size={tile_size} is not supported. "
            f"Must be one of {sorted(SUPPORTED_TILE_SIZES)}."
        )
    grid_k = params["grid_k"]
    batch_size = params["incore_task_num"] // grid_k

    A = torch.randn(batch_size, grid_k, tile_size, tile_size, dtype=torch.float32) * 0.01
    B = torch.randn(batch_size, grid_k, tile_size, tile_size, dtype=torch.float32) * 0.01
    C = torch.zeros(batch_size, tile_size, tile_size, dtype=torch.float32)

    config = torch.tensor([tile_size, grid_k, batch_size], dtype=torch.int64)

    A_flat = A.flatten()
    B_flat = B.flatten()
    C_flat = C.flatten()

    return [
        ("A", A_flat),
        ("B", B_flat),
        ("C", C_flat),
        ("config", config),
        ("size_A", ctypes.c_int64(A_flat.nbytes)),
        ("size_B", ctypes.c_int64(B_flat.nbytes)),
        ("size_C", ctypes.c_int64(C_flat.nbytes)),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    """Compute golden result: C[batch] = sum(k) A[batch,k] @ B[batch,k]."""
    tile_size = params["incore_data_size"]
    grid_k = params["grid_k"]
    batch_size = params["incore_task_num"] // grid_k

    A = torch.as_tensor(tensors["A"]).reshape(batch_size, grid_k, tile_size, tile_size)
    B = torch.as_tensor(tensors["B"]).reshape(batch_size, grid_k, tile_size, tile_size)
    C = torch.as_tensor(tensors["C"]).reshape(batch_size, tile_size, tile_size)

    C[:] = 0.0

    for batch in range(batch_size):
        for k_idx in range(grid_k):
            C[batch] += torch.matmul(A[batch, k_idx], B[batch, k_idx])

    tensors["C"][:] = C.flatten()


if __name__ == "__main__":
    params = {"name": DEFAULT_CASE, **ALL_CASES[DEFAULT_CASE]}
    result = generate_inputs(params)
    tensors = {name: tensor for name, tensor in result if isinstance(tensor, torch.Tensor)}
    compute_golden(tensors, params)

    tile_size = params["incore_data_size"]
    grid_k = params["grid_k"]
    batch_size = params["incore_task_num"] // grid_k

    print(f"=== BGEMM Golden Test ({params['name']}) ===")
    print(f"incore_task_num={params['incore_task_num']}, incore_data_size={tile_size}, grid_k={grid_k}")
    print(f"Derived: tile_size={tile_size}, batch_size={batch_size}")

    C = tensors["C"].reshape(batch_size, tile_size, tile_size)
    print(f"Output shape: {C.shape}")
    print(f"Output range: [{C.min():.4f}, {C.max():.4f}]")
    print(f"Output mean: {C.mean():.4f}")
    print("Golden test passed!")
