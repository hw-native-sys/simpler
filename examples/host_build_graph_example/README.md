# PTO Runtime Example - Hardware Platform (a2a3)

This example demonstrates how to build and execute task dependency graphs on Ascend devices using the PTO Runtime test framework.

## Overview

The example implements the formula `(a + b + 1)(a + b + 2)` using a task dependency graph:

- Task 0: `c = a + b`
- Task 1: `d = c + 1`
- Task 2: `e = c + 2`
- Task 3: `f = d * e`

With input values `a=2.0` and `b=3.0`, the expected result is `f = (2+3+1)*(2+3+2) = 42.0`.

## Dependencies

- Python 3
- NumPy
- CANN Runtime (Ascend) with ASCEND_HOME_PATH set
- gcc/g++ compiler
- PTO-ISA headers (PTO_ISA_ROOT environment variable)

## Quick Start

Run the example using the test framework:

```bash
# From repository root
python examples/scripts/run_example.py \
  -k examples/host_build_graph_example/kernels \
  -g examples/host_build_graph_example/golden.py \
  -p a2a3

# With specific device ID
python examples/scripts/run_example.py \
  -k examples/host_build_graph_example/kernels \
  -g examples/host_build_graph_example/golden.py \
  -p a2a3 \
  -d 9

# With verbose output
python examples/scripts/run_example.py \
  -k examples/host_build_graph_example/kernels \
  -g examples/host_build_graph_example/golden.py \
  -p a2a3 \
  -v
```

## Directory Structure

```
host_build_graph_example/
├── README.md                    # This file
├── golden.py                    # Input generation and expected output
└── kernels/
    ├── kernel_config.py         # Kernel configuration
    ├── aiv/                      # AIV kernel implementations
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
=== Building Runtime: host_build_graph (platform: a2a3) ===
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

## Environment Setup

```bash
# Required environment variables
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PTO_ISA_ROOT=/path/to/pto-isa

# Optional
export PTO_DEVICE_ID=0
```

## Kernels

The example uses runtime kernel compilation. Kernels are compiled using the Bisheng CCE compiler (`ccec`) from the CANN toolkit:

- **kernel_add.cpp** - Element-wise tensor addition using PTO ISA
- **kernel_add_scalar.cpp** - Add a scalar value to each tensor element
- **kernel_mul.cpp** - Element-wise tensor multiplication

## Troubleshooting

### Kernel Compilation Failed

Ensure PTO_ISA_ROOT is set:
```bash
export PTO_ISA_ROOT=/path/to/pto-isa
```

### Device Initialization Failed

- Verify CANN runtime is installed and ASCEND_HOME_PATH is set
- Check that the specified device ID is valid (0-15)
- Ensure you have permission to access the device

### "binary_data cannot be empty" Error

- Verify correct `-p a2a3` parameter is used
- Check if kernel source files exist
- Use `-v` to view detailed compilation logs

## See Also

- [Test Framework Documentation](../scripts/README.md)
- [Simulation Example](../host_build_graph_sim_example/) - Run without Ascend hardware
- [Main Project README](../../README.md)
