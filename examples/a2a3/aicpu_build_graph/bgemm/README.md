# BGEMM Example (AICPU Build Graph Runtime)

Tiled matrix multiplication example demonstrating Cube (AIC) and Vector (AIV) core cooperation.

## Computation

```
C = A @ B
```

Tiled computation with 4x4x4 grid:
- Tile size: 64 x 64
- Matrix A: 256 x 256 (4x4 tiles)
- Matrix B: 256 x 256 (4x4 tiles)
- Matrix C: 256 x 256 (4x4 tiles)

## Task Graph

For each output tile C[m,n]:
```
for k in [0, GRID_K):
    P = A[m,k] @ B[k,n]    (gemm_tile on Cube core)
    C[m,n] = C[m,n] + P    (tile_add on Vector core)
```

Dependencies:
- gemm_tile → tile_add: P must be computed before accumulation
- tile_add[k] → gemm_tile[k+1]: K-dimension accumulation is sequential

Total tasks: 128 (64 gemm + 64 add)

## Kernels

| Kernel | Core Type | Function |
|--------|-----------|----------|
| kernel_gemm_tile | AIC (Cube) | 64x64 matrix multiplication |
| kernel_tile_add | AIV (Vector) | 64x64 element-wise addition |

## File Structure

```
bgemm/
├── golden.py                          # Test specification
├── README.md                          # This file
└── kernels/
    ├── kernel_config.py               # Kernel configuration
    ├── orchestration/
    │   └── bgemm_orch.cpp             # Task graph builder
    ├── aic/
    │   └── kernel_gemm_tile.cpp       # Cube core matmul kernel
    └── aiv/
        └── kernel_tile_add.cpp        # Vector core add kernel
```

## Technical Details

### Memory Layout (Tile-First)

```
A: [BATCH, GRID_M, GRID_K, TILE_M, TILE_K]
B: [BATCH, GRID_K, GRID_N, TILE_K, TILE_N]
C: [BATCH, GRID_M, GRID_N, TILE_M, TILE_N]
```

### Runtime Characteristics

- Task graph is built on AICPU
- Framework automatically manages I/O tensor device memory
- Orchestration function allocates intermediate buffers via AicpuBuildApi

### Kernel Implementation

Both kernels use PTO ISA tile operations:

- **kernel_gemm_tile**: Uses `TileLeft`, `TileRight`, `TileAcc` types with `TLOAD`, `TMOV`, `TMATMUL`, `TSTORE` instructions
- **kernel_tile_add**: Uses `TileVec` type with `TLOAD`, `TADD`, `TSTORE` instructions

### Pipeline Synchronization

Kernels include proper pipeline synchronization:
- `PIPE_MTE2` → `PIPE_M`/`PIPE_V`: After loads, before compute
- `PIPE_M`/`PIPE_V` → `PIPE_MTE3`: After compute, before store
