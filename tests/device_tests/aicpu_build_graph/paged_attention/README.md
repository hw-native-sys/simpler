# Paged Attention (Device Test - aicpu_build_graph)

This test demonstrates Paged Attention using the **aicpu_build_graph** runtime, where the AICPU device builds the task graph at runtime via a dlopen'd orchestration plugin while scheduler threads execute tasks concurrently.

The kernel implementations are identical to the `host_build_graph` version. Only the orchestration and runtime configuration differ.

## Overview

Paged Attention is an efficient attention mechanism that processes KV cache in fixed-size blocks, enabling memory-efficient inference for long sequences. This implementation uses:

- **CCE-style codegen** for AIC kernels (Cube unit matmul)
- **PTO Tile API** for AIV kernels (Vector unit operations)
- **Online Softmax** algorithm for numerically stable incremental computation

### Runtime Architecture

In `aicpu_build_graph` mode:
- **1 AICPU builder thread** runs the orchestration plugin (builds the task graph)
- **3 AICPU scheduler threads** execute tasks concurrently with graph construction
- The orchestration plugin is compiled as a `.so`, embedded in Runtime, and dlopen'd on AICPU
- The framework pre-allocates device memory for I/O tensors and populates `orch_args[]`

### Supported Platforms

| Platform | Description |
|----------|-------------|
| a2a3 | Ascend hardware (requires device ID) |

> This test uses bfloat16 data types and production-scale shapes that are not supported by the a2a3sim simulator. It only runs on real hardware.

### Algorithm

For each query token, the attention is computed incrementally across KV cache blocks:

```
For each block j:
    sij = Qi @ Kj^T                    # QK MatMul (AIC)
    mij, lij, pij = softmax_prepare(sij)  # Softmax (AIV)
    oi_new = pij @ Vj                  # PV MatMul (AIC)
    oi = online_update(oi, oi_new, mij, lij)  # Accumulate (AIV)
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
# Run on hardware (specify device ID)
python examples/scripts/run_example.py \
  -k tests/device_tests/aicpu_build_graph/paged_attention/kernels \
  -g tests/device_tests/aicpu_build_graph/paged_attention/golden.py \
  -p a2a3 -d 0

# Run multi-block test case
PA_CASE=Case2 python examples/scripts/run_example.py \
  -k tests/device_tests/aicpu_build_graph/paged_attention/kernels \
  -g tests/device_tests/aicpu_build_graph/paged_attention/golden.py \
  -p a2a3 -d 0
```

## Directory Structure

```
paged_attention/
├── README.md                    # This file
├── golden.py                    # Input generation and expected output
└── kernels/
    ├── kernel_config.py         # Kernel registration config (aicpu_build_graph)
    ├── aic/                      # AIC kernels (CCE codegen style)
    │   ├── aic_qk_matmul.cpp     # Q @ K^T matmul
    │   └── aic_pv_matmul.cpp     # P @ V matmul
    ├── aiv/                      # AIV kernels (PTO Tile API)
    │   ├── aiv_softmax_prepare.cpp  # Softmax preparation
    │   └── aiv_online_update.cpp    # Online Softmax update + normalize
    └── orchestration/
        └── paged_attention_orch.cpp # AICPU task graph builder (dlopen'd plugin)
```

## Test Cases

| Case | batch | num_heads | kv_head_num | head_dim | block_size | context_len | Description |
|------|-------|-----------|-------------|----------|------------|-------------|-------------|
| Case1 | 256 | 16 | 1 | 128 | 128 | 8192 | Default |
| Case2 | 64 | 64 | 1 | 128 | 64 | 8192 | Multi-block |

All test cases use **bfloat16** Q/K/V inputs with GQA (kv_head_num=1).

## Key Differences from host_build_graph Version

| Aspect | host_build_graph | aicpu_build_graph |
|--------|------------------|-------------------|
| Graph building | Host CPU | AICPU device (dlopen'd plugin) |
| I/O memory | Orchestration allocates + copies | Framework pre-manages |
| Task API | `runtime->add_task()` | `api.add_task()` |
| Dependency API | `runtime->add_successor()` | `api.add_successor_conditional()` |
| Task visibility | Implicit | Explicit `api.publish_task()` |
| Thread model | 3 scheduler threads | 1 builder + 3 scheduler threads |

## See Also

- [host_build_graph version](../../host_build_graph/paged_attention/README.md)
- [Test Framework Documentation](../../../../examples/scripts/README.md)
