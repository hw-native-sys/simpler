# PTO Runtime Example - Simulation Platform (a2a3sim)

This example demonstrates how to build and execute task dependency graphs using the thread-based simulation platform, without requiring Ascend hardware.

## Overview

The example implements the formula `(a + b + 1)(a + b + 2)` using a task dependency graph:

- Task 0: `c = a + b`
- Task 1: `d = c + 1`
- Task 2: `e = c + 2`
- Task 3: `f = d * e`

With input values `a=2.0` and `b=3.0`, the expected result is `f = (2+3+1)*(2+3+2) = 42.0`.

## Key Differences from Hardware Example

| Aspect | Hardware (a2a3) | Simulation (a2a3sim) |
|--------|-----------------|----------------------|
| Platform | `-p a2a3` | `-p a2a3sim` |
| Requirements | CANN toolkit, Ascend device | gcc/g++ only |
| Kernel compilation | ccec (Bisheng) compiler | g++ compiler |
| Execution | AICPU/AICore on device | Host threads |
| Kernel format | PTO ISA | Plain C++ loops |

## Dependencies

- Python 3
- NumPy
- gcc/g++ compiler

No Ascend SDK or CANN toolkit required.

## Quick Start

Run the example using the test framework:

```bash
# From repository root
python examples/scripts/run_example.py \
  -k examples/host_build_graph_sim_example/kernels \
  -g examples/host_build_graph_sim_example/golden.py \
  -p a2a3sim

# With verbose output
python examples/scripts/run_example.py \
  -k examples/host_build_graph_sim_example/kernels \
  -g examples/host_build_graph_sim_example/golden.py \
  -p a2a3sim \
  -v
```

## Directory Structure

```
host_build_graph_sim_example/
├── README.md                    # This file
├── golden.py                    # Input generation and expected output
└── kernels/
    ├── kernel_config.py         # Kernel configuration
    ├── aiv/                      # AIV kernel implementations (plain C++)
    │   ├── kernel_add.cpp        # Element-wise tensor addition
    │   ├── kernel_add_scalar.cpp # Add scalar to tensor elements
    │   └── kernel_mul.cpp        # Element-wise tensor multiplication
    └── orchestration/
        └── example_orch.cpp      # Task graph building function
```

## Files

### `golden.py`

Defines input tensors and expected output computation:

```python
__outputs__ = ["f"]           # Output tensor names
TENSOR_ORDER = ["a", "b", "f"]  # Order passed to orchestration function

def generate_inputs(params: dict) -> dict:
    # Returns: {"a": ..., "b": ..., "f": ...}

def compute_golden(tensors: dict, params: dict) -> None:
    # Computes expected output in-place
```

### `kernels/kernel_config.py`

Defines kernel sources and orchestration function:

```python
KERNELS = [
    {"func_id": 0, "core_type": "aiv", "source": ".../kernel_add.cpp"},
    {"func_id": 1, "core_type": "aiv", "source": ".../kernel_add_scalar.cpp"},
    {"func_id": 2, "core_type": "aiv", "source": ".../kernel_mul.cpp"},
]

ORCHESTRATION = {
    "source": ".../example_orch.cpp",
    "function_name": "BuildExampleGraph"
}
```

## Expected Output

```
=== Building Runtime: host_build_graph (platform: a2a3sim) ===
...
=== Compiling and Registering Kernels ===
Compiling kernel: kernels/aiv/kernel_add.cpp (func_id=0)
...
=== Generating Input Tensors ===
Inputs: ['a', 'b']
Outputs: ['f']
...
=== Launching Runtime ===
...
=== Comparing Results ===
Comparing f: shape=(16384,), dtype=float32
  First 10 actual:   [42. 42. 42. 42. 42. 42. 42. 42. 42. 42.]
  First 10 expected: [42. 42. 42. 42. 42. 42. 42. 42. 42. 42.]
  f: PASS (16384/16384 elements matched)

============================================================
TEST PASSED
============================================================
```

## Simulation Architecture

The simulation platform emulates the AICPU/AICore execution model:

- **Kernel loading**: Kernel `.text` sections are mmap'd into executable memory
- **Thread execution**: Host threads emulate AICPU scheduling and AICore computation
- **Memory**: All allocations use host memory (malloc/free)
- **Same API**: Uses identical C API as the real a2a3 platform

## Kernels

Simulation kernels are plain C++ implementations in `kernels/aiv/`:

- **kernel_add.cpp** - Element-wise tensor addition (loop-based)
- **kernel_add_scalar.cpp** - Add scalar to each tensor element (loop-based)
- **kernel_mul.cpp** - Element-wise tensor multiplication (loop-based)

These are compiled with g++ instead of the PTO compiler.

## Troubleshooting

### "binary_data cannot be empty" Error

- Verify correct `-p a2a3sim` parameter is used
- Check if kernel source files exist
- Use `-v` to view detailed compilation logs

### Compilation Errors

- Ensure gcc/g++ is installed and available in PATH
- Check kernel source syntax for C++ errors

## See Also

- [Test Framework Documentation](../scripts/README.md)
- [Hardware Example](../host_build_graph_example/) - Run on real Ascend devices
- [Main Project README](../../README.md)
