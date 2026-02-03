# Paged Attention Example - Simulation Platform (a2a3sim)

This example demonstrates Paged Attention implementation with AIC/AIV kernel splitting using the thread-based simulation platform.

## Overview

Paged Attention is an efficient attention mechanism that processes KV cache in fixed-size blocks, enabling memory-efficient inference for long sequences. This implementation uses the Online Softmax algorithm for numerically stable incremental computation.

### Algorithm

For each query token, the attention is computed incrementally across KV cache blocks:

```
For each block j:
    sij = Qi @ Kj.T                    # QK MatMul
    pij = softmax(sij * scale)         # Softmax
    oi_new = pij @ Vj                  # PV MatMul
    oi = online_update(oi, oi_new)     # Accumulate with Online Softmax
```

### Kernel Design (AIC/AIV Split)

| Kernel | Core Type | Operation |
|--------|-----------|-----------|
| `aic_qk_matmul` | AIC (Cube) | Q @ K^T matrix multiplication |
| `aiv_softmax_prepare` | AIV (Vector) | scale, rowmax, exp, rowsum |
| `aic_pv_matmul` | AIC (Cube) | P @ V matrix multiplication |
| `aiv_online_update` | AIV (Vector) | Online Softmax accumulation + normalize |

### Task Graph Structure

For each batch, the task dependency pattern is:

```
Block 0: QK -> SF -> PV --+
Block 1: QK -> SF -> PV --+--> UP[0] -> UP[1] -> UP[2] -> UP[3]
Block 2: QK -> SF -> PV --+
Block 3: QK -> SF -> PV --+
```

- **QK/SF/PV chains**: Run in parallel across blocks
- **UP (Online Update)**: Serialized within batch due to accumulator dependency

## Dependencies

- Python 3
- NumPy
- gcc/g++ compiler

No Ascend SDK or CANN toolkit required.

## Quick Start

```bash
# From repository root
python examples/scripts/run_example.py \
  -k examples/paged_attention/kernels \
  -g examples/paged_attention/golden.py \
  -p a2a3sim

# With verbose output
python examples/scripts/run_example.py \
  -k examples/paged_attention/kernels \
  -g examples/paged_attention/golden.py \
  -p a2a3sim \
  -v
```

## Directory Structure

```
paged_attention/
├── README.md                    # This file
├── golden.py                    # Input generation and expected output
└── kernels/
    ├── kernel_config.py         # Kernel configuration
    ├── aic/                      # AIC kernel implementations (Cube unit)
    │   ├── aic_qk_matmul.cpp     # Q @ K^T matrix multiplication
    │   └── aic_pv_matmul.cpp     # P @ V matrix multiplication
    ├── aiv/                      # AIV kernel implementations (Vector unit)
    │   ├── aiv_softmax_prepare.cpp  # Softmax preparation
    │   └── aiv_online_update.cpp    # Online Softmax update + normalize
    └── orchestration/
        └── paged_attention_orch.cpp # Task graph building function
```

## Files

### `golden.py`

Defines input tensors and expected output computation:

```python
__outputs__ = ["out"]
TENSOR_ORDER = ["query", "key_cache", "value_cache", "block_table", 
                "context_lens", "out", "config"]

PARAMS_LIST = [
    {
        "batch": 2,
        "num_heads": 16,
        "kv_head_num": 1,
        "head_dim": 128,
        "block_size": 64,
        "block_num": 4,
        "context_len": 256,
    },
]

def generate_inputs(params: dict) -> dict:
    # Returns all tensors including config

def compute_golden(tensors: dict, params: dict) -> None:
    # Computes expected output using Online Softmax algorithm
```

### `kernels/kernel_config.py`

Defines kernel sources and orchestration function:

```python
KERNELS = [
    {"func_id": 0, "source": ".../aic_qk_matmul.cpp",       "core_type": "aic"},
    {"func_id": 1, "source": ".../aiv_softmax_prepare.cpp", "core_type": "aiv"},
    {"func_id": 2, "source": ".../aic_pv_matmul.cpp",       "core_type": "aic"},
    {"func_id": 3, "source": ".../aiv_online_update.cpp",   "core_type": "aiv"},
]

ORCHESTRATION = {
    "source": ".../paged_attention_orch.cpp",
    "function_name": "build_paged_attention_graph"
}
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `batch` | Batch size | 2 |
| `num_heads` | Number of query heads | 16 |
| `kv_head_num` | Number of KV heads (for GQA) | 1 |
| `head_dim` | Head dimension | 128 |
| `block_size` | KV cache block size | 64 |
| `block_num` | Number of blocks per batch | 4 |
| `context_len` | Context length | 256 |

## Expected Output

```
=== Building Runtime: host_build_graph (platform: a2a3sim) ===
...
=== Compiling and Registering Kernels ===
Compiling kernel: .../aic_qk_matmul.cpp (func_id=0)
Compiling kernel: .../aiv_softmax_prepare.cpp (func_id=1)
Compiling kernel: .../aic_pv_matmul.cpp (func_id=2)
Compiling kernel: .../aiv_online_update.cpp (func_id=3)
...
=== build_paged_attention_graph (multi-head per task) ===
batch=2, num_heads=16, kv_head_num=1, head_dim=128
block_size=64, block_num=4
...
Created 32 tasks
...
=== Comparing Results ===
Comparing out: shape=(4096,), dtype=float32
  out: PASS (4096/4096 elements matched)

============================================================
TEST PASSED
============================================================
```

## Kernels

### AIC Kernels (Cube Unit)

- **aic_qk_matmul.cpp** - Computes Q @ K^T for all heads
- **aic_pv_matmul.cpp** - Computes P @ V for all heads

### AIV Kernels (Vector Unit)

- **aiv_softmax_prepare.cpp** - Computes scale, rowmax, exp, rowsum for softmax
- **aiv_online_update.cpp** - Online Softmax accumulation with fused final normalization

## GQA Support

This implementation supports Grouped Query Attention (GQA) where multiple query heads share a single KV head:

```
heads_per_kv = num_heads / kv_head_num
```

For example, with `num_heads=16` and `kv_head_num=1`, all 16 query heads share the same KV cache.

## See Also

- [Test Framework Documentation](../scripts/README.md)
- [Simple Example](../host_build_graph_sim_example/) - Basic task graph example
- [Main Project README](../../README.md)
