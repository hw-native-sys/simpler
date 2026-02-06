# Paged Attention CCE Example

This example demonstrates Paged Attention implementation using CCE (Cube Core Engine) code generation, with AIC matmul kernels and AIV vector kernels using PTO Tile API.

## Overview

Paged Attention is an efficient attention mechanism that processes KV cache in fixed-size blocks, enabling memory-efficient inference for long sequences. This implementation uses:

- **CCE-style codegen** for AIC kernels (Cube unit matmul)
- **PTO Tile API** for AIV kernels (Vector unit operations)
- **Online Softmax** algorithm for numerically stable incremental computation

### Supported Platforms

| Platform | Description |
|----------|-------------|
| a2a3sim | Thread-based simulator |
| a2a3 | Ascend hardware (requires device ID) |

### Algorithm

For each query token, the attention is computed incrementally across KV cache blocks:

```
For each block j:
    sij = Qi @ Kj^T                    # QK MatMul (AIC)
    mij, lij, pij = softmax_prepare(sij)  # Softmax (AIV)
    oi_new = pij @ Vj                  # PV MatMul (AIC)
    oi = online_update(oi, oi_new, mij, lij)  # Accumulate (AIV)
```

### Kernel Design (AIC/AIV Split)

| Kernel | Core Type | Operation | Key Instructions |
|--------|-----------|-----------|------------------|
| aic_qk_matmul | AIC (Cube) | Q @ K^T | TLOAD/TMOV/TMATMUL/TSTORE |
| aiv_softmax_prepare | AIV (Vector) | scale, rowmax, exp, rowsum | TMULS/TROWMAX/TROWEXPANDSUB/TEXP/TROWSUM |
| aic_pv_matmul | AIC (Cube) | P @ V | TLOAD/TMOV/TMATMUL/TSTORE |
| aiv_online_update | AIV (Vector) | Online Softmax + normalize | TMAX/TSUB/TEXP/TROWEXPANDMUL/TROWEXPANDDIV |

### Memory Hierarchy (AIC Matmul)

```
GM -> L1 (Mat tiles) -> L0A/L0B -> L0C (Accumulator) -> GM
```

### Task Graph Structure

For each batch, the task dependency pattern is:

```
Block 0: QK -> SF -> PV --+
Block 1: QK -> SF -> PV --+--> UP[0] -> UP[1] -> ... -> UP[n]
Block n: QK -> SF -> PV --+
```

- **QK/SF/PV chains**: Run in parallel across blocks
- **UP (Online Update)**: Serialized within batch due to accumulator dependency

## Quick Start

```bash
# Run on simulator
python examples/scripts/run_example.py \
  -k examples/paged_attention_cce/kernels \
  -g examples/paged_attention_cce/golden.py \
  -p a2a3sim

# Run on hardware (specify device ID)
python examples/scripts/run_example.py \
  -k examples/paged_attention_cce/kernels \
  -g examples/paged_attention_cce/golden.py \
  -p a2a3 -d=13

# Run multi-block test case
PA_CASE=Case2 python examples/scripts/run_example.py \
  -k examples/paged_attention_cce/kernels \
  -g examples/paged_attention_cce/golden.py \
  -p a2a3sim
```

## Directory Structure

```
paged_attention_cce/
├── README.md                    # This file
├── golden.py                    # Input generation and expected output
└── kernels/
    ├── kernel_config.py         # Kernel registration config
    ├── aic/                      # AIC kernels (CCE codegen style)
    │   ├── aic_qk_matmul.cpp     # Q @ K^T matmul
    │   └── aic_pv_matmul.cpp     # P @ V matmul
    ├── aiv/                      # AIV kernels (PTO Tile API)
    │   ├── aiv_softmax_prepare.cpp  # Softmax preparation
    │   └── aiv_online_update.cpp    # Online Softmax update + normalize
    └── orchestration/
        └── paged_attention_orch.cpp # Task graph builder
```

## Test Cases

| Case | batch | num_heads | head_dim | block_size | block_num | Description |
|------|-------|-----------|----------|------------|-----------|-------------|
| Case1 | 1 | 16 | 16 | 16 | 1 | Single block (default) |
| Case2 | 1 | 16 | 16 | 16 | 4 | Multi-block (4 blocks) |

All test cases use **16x16 tile dimensions** (256 elements, 1024 bytes per tile).

## Key Technical Details

### AIC Kernels (CCE Codegen)

```cpp
// L1 tiles: ColMajor + SLayout::RowMajor (required for matmul)
using TileMatA = Tile<TileType::Mat, float, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
using TileMatB = Tile<TileType::Mat, float, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

// L0 tiles: Use standard TileLeft/TileRight/TileAcc aliases
using LeftTile = TileLeft<float, M, K, M, K>;
using RightTile = TileRight<float, K, N, K, N>;
using AccTile = TileAcc<float, M, N, M, N>;

// Pipeline: MTE2 -> MTE1 -> M -> FIX -> MTE3
TLOAD(aMatTile, qiGlobal);           // GM -> L1
TMOV(aTile, aMatTile);               // L1 -> L0A
TMATMUL(cTile, aTile, bTile);        // L0A x L0B -> L0C
TSTORE(sijGlobal, cTile);            // L0C -> GM
```

### AIV Kernels (PTO Tile API)

**softmax_prepare**: Uses DN layout (ColMajor, 16x1) for row reduction results

```cpp
using TileScalarDN = Tile<TileType::Vec, float, kAlignedRows, 1, BLayout::ColMajor, kRows, 1>;

TMULS(sijTile, sijTile, scale_value);      // Scale
TROWMAX(maxTile, sijTile, tmpTile);        // Row max
TROWEXPANDSUB(pijTile, sijTile, maxTile);  // Subtract max (broadcast)
TEXP(pijTile, pijTile);                    // Exp
TROWSUM(sumTile, pijTile, tmpTile);        // Row sum
```

**online_update**: Uses ND/DN layout conversion for hardware compatibility

```cpp
// ND (1x16, RowMajor) for scalar arithmetic - TSUB/TMUL/TADD require RowMajor
using TileScalarND = Tile<TileType::Vec, float, 1, kNumHeads, BLayout::RowMajor, 1, kNumHeads>;
// DN (16x1, ColMajor) for row broadcast - TROWEXPANDMUL/TROWEXPANDDIV require this
using TileScalarDN = Tile<TileType::Vec, float, kNumHeads, 1, BLayout::ColMajor, kNumHeads, 1>;

// Arithmetic in ND layout
TMAX(miNewTileND, miTileND, mijTileND);
TSUB(alphaTileND, miTileND, miNewTileND);
TEXP(alphaTileND, alphaTileND);

// Reshape ND -> DN for broadcast operations
TRESHAPE(alphaTileDN, alphaTileND);
TROWEXPANDMUL(oiTile, oiTile, alphaTileDN);
```

### Data Layout

- **K stored as K^T**: (head_dim, block_size) for direct matmul compatibility
- **V stored normally**: (block_size, head_dim)

## Expected Output

```
=== Compiling and Registering Kernels ===
Compiling kernel: .../aic_qk_matmul.cpp (func_id=0)
Compiling kernel: .../aiv_softmax_prepare.cpp (func_id=1)
Compiling kernel: .../aic_pv_matmul.cpp (func_id=2)
Compiling kernel: .../aiv_online_update.cpp (func_id=3)
...
=== build_paged_attention_graph (16x16 framework version) ===
batch=1, num_heads=16, kv_head_num=1, head_dim=16
block_size=16, block_num=1
...
Created 4 tasks
...
=== Comparing Results ===
Comparing out: shape=(256,), dtype=float32
  out: PASS (256/256 elements matched)

============================================================
TEST PASSED
============================================================
```

## Reference

This implementation corresponds to the PyPTO paged_attention.py:
- /data/w00949583/pypto/examples/ir_parser/paged_attention.py

Both implementations use the same Online Softmax algorithm with identical kernel structure.

## See Also

- [Test Framework Documentation](../scripts/README.md)
- [Paged Attention (original)](../paged_attention/) - PTO Tile API only version
