---
name: kernel-development
description: On-demand guide for writing AICore/AICPU kernels, orchestration code, and golden tests for the PTO Runtime three-program model. Covers AIC (CUBE/matrix), AIV (vector), and orchestration patterns.
---

# Kernel Development Skill

## When to Use

Load this skill when:
- Writing a new AICore kernel (AIC or AIV)
- Creating orchestration code that builds a task graph
- Implementing `golden.py` test harness for a kernel
- Configuring `kernel_config.py` for a new example

## Terminology

- **AIC** = AICore-CUBE — matrix operations (matmul, convolution)
- **AIV** = AICore-VECTOR — element-wise operations (add, mul, activation)
- **AICPU** — control processor that schedules tasks to AICore

## Kernel Source Pattern

AIV kernel example (`kernels/aiv/kernel_add.cpp`):
```cpp
#include "kernel_operator.h"

extern "C" __global__ __aicore__
void kernel_add(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t totalLength) {
    // Use GlobalTensor, LocalTensor, Tile operations
    // Synchronize with pipe_barrier(PIPE_ALL);
}
```

## Orchestration Pattern

Orchestration builds the task dependency graph (`kernels/orchestration/build_graph.cpp`):
```cpp
void build_graph(TaskGraph& graph, TensorMap& tensors) {
    // Create tasks referencing func_id values from kernel_config.py
    // Set dependencies between tasks
    // Map tensors to task arguments
}
```

## Golden Test Pattern

```python
def generate_inputs(params):
    """Return dict of torch tensors (inputs + zero-initialized outputs)."""
    M, N = params.get("M", 128), params.get("N", 128)
    return {
        "a": torch.randn(M, N, dtype=torch.float16),
        "b": torch.randn(M, N, dtype=torch.float16),
        "out_c": torch.zeros(M, N, dtype=torch.float16),
    }

def compute_golden(tensors, params):
    """Compute expected outputs in-place."""
    tensors["out_c"].copy_(tensors["a"] + tensors["b"])
```

## kernel_config.py Template

```python
ORCHESTRATION = {
    "source": "orchestration/build_graph.cpp",
    "function_name": "build_graph",
}

KERNELS = [
    {"func_id": 0, "source": "aiv/kernel_add.cpp", "core_type": "aiv"},
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 3,
    "block_dim": 3,
}
```

## Validation

```bash
python examples/scripts/run_example.py \
    -k examples/<runtime>/<name>/kernels \
    -g examples/<runtime>/<name>/golden.py \
    -p a2a3sim
```
