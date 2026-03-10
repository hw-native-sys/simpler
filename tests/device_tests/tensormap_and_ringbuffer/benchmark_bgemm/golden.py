"""
Golden test specification for BGEMM (tensormap_and_ringbuffer Runtime).

Computation: C = A @ B (tiled matrix multiplication)
Configuration controlled by ALL_CASES parameters:
  - matmul_add_task_num: number of matmul/add tasks (each matmul has a corresponding add)
  - incore_task_granularity: dict with incore_data_size (tile size) and incore_loop
  - grid_k: number of K-dimension partitions
  - num_groups = matmul_add_task_num / grid_k

Args layout: [ptr_A, ptr_B, ptr_C, ptr_config, size_A, size_B, size_C]
"""

import ctypes
import torch

__outputs__ = ["C"]
RTOL = 1e-3
ATOL = 1e-3

# Supported tile sizes must match the switch-case in kernel_gemm_tile.cpp (AIC)
# and kernel_tile_add.cpp (AIV), which only instantiate templates for these values.
SUPPORTED_INCORE_DATA_SIZES = {16, 32, 64, 128}

ALL_CASES = {
    "Case0": {
        "matmul_add_task_num": 500,
        "incore_task_granularity": {
            "incore_data_size": 128,
            "incore_loop": 4,
        },
        "grid_k": 2,
    },
    "Case1": {
        "matmul_add_task_num": 64,
        "incore_task_granularity": {
            "incore_data_size": 128,
            "incore_loop": 4,
        },
        "grid_k": 2,
    },
    "Case2": {
        "matmul_add_task_num": 256,
        "incore_task_granularity": {
            "incore_data_size": 128,
            "incore_loop": 4,
        },
        "grid_k": 2,
    },
    "Case3": {
        "matmul_add_task_num": 64,
        "incore_task_granularity": {
            "incore_data_size": 128,
            "incore_loop": 16,
        },
        "grid_k": 2,
    },
    "Case4": {
        "matmul_add_task_num": 64,
        "incore_task_granularity": {
            "incore_data_size": 128,
            "incore_loop": 4,
        },
        "grid_k": 4,
    },
}

DEFAULT_CASE = "Case0"


def generate_inputs(params: dict) -> list:
    """Generate input tensors with tile-first memory layout."""
    granularity = params["incore_task_granularity"]
    tile_size = granularity["incore_data_size"]
    incore_loop = granularity["incore_loop"]
    matmul_add_task_num = params["matmul_add_task_num"]
    grid_k = params["grid_k"]

    # --- constraint checks ---
    if tile_size not in SUPPORTED_INCORE_DATA_SIZES:
        raise ValueError(
            f"incore_data_size={tile_size} is not supported. "
            f"Must be one of {sorted(SUPPORTED_INCORE_DATA_SIZES)}."
        )
    if incore_loop <= 0:
        raise ValueError(f"incore_loop must be positive, got {incore_loop}")
    if grid_k <= 0:
        raise ValueError(f"grid_k must be positive, got {grid_k}")
    if matmul_add_task_num % grid_k != 0:
        raise ValueError(
            f"matmul_add_task_num ({matmul_add_task_num}) must be "
            f"divisible by grid_k ({grid_k})."
        )

    num_groups = matmul_add_task_num // grid_k

    A = torch.randn(incore_loop * num_groups, grid_k, tile_size, tile_size, dtype=torch.float32) * 0.01
    B = torch.randn(incore_loop * num_groups, grid_k, tile_size, tile_size, dtype=torch.float32) * 0.01
    C = torch.zeros(incore_loop * num_groups, tile_size, tile_size, dtype=torch.float32)

    # Reshape A/B to [num_groups, grid_k, incore_loop, tile_size, tile_size]
    # so that incore_loop tiles are contiguous for each (group, k_idx)
    A = A.reshape(num_groups, incore_loop, grid_k, tile_size, tile_size)
    A = A.permute(0, 2, 1, 3, 4).contiguous()  # [num_groups, grid_k, incore_loop, tile_size, tile_size]
    B = B.reshape(num_groups, incore_loop, grid_k, tile_size, tile_size)
    B = B.permute(0, 2, 1, 3, 4).contiguous()

    config = torch.tensor([tile_size, grid_k, num_groups, incore_loop], dtype=torch.int64)

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
    """Compute golden result: C[i] = sum(k) A[i,k] @ B[i,k]."""
    granularity = params["incore_task_granularity"]
    tile_size = granularity["incore_data_size"]
    incore_loop = granularity["incore_loop"]
    grid_k = params["grid_k"]
    num_groups = params["matmul_add_task_num"] // grid_k

    # A/B layout: [num_groups, grid_k, incore_loop, tile_size, tile_size]
    A = torch.as_tensor(tensors["A"]).reshape(num_groups, grid_k, incore_loop, tile_size, tile_size)
    B = torch.as_tensor(tensors["B"]).reshape(num_groups, grid_k, incore_loop, tile_size, tile_size)
    C = torch.as_tensor(tensors["C"]).reshape(incore_loop * num_groups, tile_size, tile_size)

    C[:] = 0.0

    for group in range(num_groups):
        for k_idx in range(grid_k):
            for i in range(incore_loop):
                batch = group * incore_loop + i
                C[batch] += torch.matmul(A[group, k_idx, i], B[group, k_idx, i])

    tensors["C"][:] = C.flatten()


if __name__ == "__main__":
    params = {"name": DEFAULT_CASE, **ALL_CASES[DEFAULT_CASE]}
    result = generate_inputs(params)
    tensors = {name: tensor for name, tensor in result if isinstance(tensor, torch.Tensor)}
    compute_golden(tensors, params)

    granularity = params["incore_task_granularity"]
    tile_size = granularity["incore_data_size"]
    incore_loop = granularity["incore_loop"]
    grid_k = params["grid_k"]
    num_groups = params["matmul_add_task_num"] // grid_k

    print(f"=== BGEMM Golden Test ({params['name']}) ===")
    print(f"matmul_add_task_num={params['matmul_add_task_num']}")
    print(f"incore_task_granularity={granularity}")
    print(f"grid_k={grid_k}, num_groups={num_groups}")

    C = tensors["C"].reshape(incore_loop * num_groups, tile_size, tile_size)
    print(f"Output shape: {C.shape}")
    print(f"Output range: [{C.min():.4f}, {C.max():.4f}]")
    print(f"Output mean: {C.mean():.4f}")
    print("Golden test passed!")
