# PTO Runtime Python Example - Basic

This example demonstrates how to build and execute task dependency graphs on Ascend devices using the Python bindings for PTO Runtime.

## Overview

The example implements the formula `(a + b + 1)(a + b + 2)` using a task dependency graph with runtime kernel compilation:

- Task 0: `c = a + b`
- Task 1: `d = c + 1`
- Task 2: `e = c + 2`
- Task 3: `f = d * e`

With input values `a=2.0` and `b=3.0`, the expected result is `f = (2+3+1)*(2+3+2) = 42.0`.

## Building

From the runtime directory:

```bash
mkdir -p build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON
make -j
```

## Dependencies

- Python 3
- NumPy
- CANN Runtime (Ascend)
- PTO Runtime Python bindings (built automatically)

## Running the Example

### Set Environment Variables

From the build directory:

```bash
# Set PTO_ISA_ROOT for runtime kernel compilation
export PTO_ISA_ROOT=$(pwd)/_deps/pto-isa-src
```

### Run the Example

```bash
cd ../examples/basic
python3 main.py <device_id>
```

For example, to run on device 9:

```bash
python3 main.py 9
```

## Expected Output

```
=== Runtime Builder Example (Python) ===

=== Compiling Kernels at Runtime ===
All kernels compiled and loaded successfully

=== Allocating Device Memory ===
Allocated 6 tensors (128x128 each, 65536 bytes per tensor)
Initialized input tensors: a=2.0, b=3.0 (all elements)
Expected result: f = (2+3+1)*(2+3+2) = 6*7 = 42.0

=== Creating Task Runtime for Formula ===
Formula: (a + b + 1)(a + b + 2)
Tasks:
  task0: c = a + b
  task1: d = c + 1
  task2: e = c + 2
  task3: f = d * e

Created runtime with 4 tasks
...

=== Executing Runtime ===

=== Validating Results ===
First 10 elements of result:
  f[0] = 42.0
  f[1] = 42.0
  ...

✓ SUCCESS: All 16384 elements are correct (42.0)
Formula verified: (a + b + 1)(a + b + 2) = (2+3+1)*(2+3+2) = 42

=== Success ===
```

## How It Works

1. **Initialize Device**: The example initializes the DeviceRunner with the specified device ID
2. **Compile Kernels**: Three kernels are compiled at runtime:
   - `kernel_add.cpp`: Element-wise addition
   - `kernel_add_scalar.cpp`: Add scalar to each element
   - `kernel_mul.cpp`: Element-wise multiplication
3. **Allocate Memory**: Device memory is allocated for 6 tensors (a, b, c, d, e, f)
4. **Copy Input**: Input data is copied from host to device
5. **Build Runtime**: A task dependency runtime is constructed with proper dependencies
6. **Execute Runtime** (Python): Python calls `runner.run(runtime)` to execute the runtime on device
7. **Validate**: Results are copied back and verified
8. **Cleanup**: All resources are freed

### Execution Flow

The example demonstrates a clean separation of concerns:

**C++ (InitRuntime)**:
- Compiles and loads kernels
- Allocates device memory for tensors
- Initializes input data
- Builds the task dependency runtime

**Python**:
- Orchestrates the overall flow
- Calls `runner.run(runtime)` to execute the runtime on device

**C++ (FinalizeRuntime)**:
- Copies results from device
- Validates computation correctness
- Frees device memory
- Deletes the runtime

## Kernels

The example uses runtime kernel compilation. Kernel source files are in the `kernels/` directory:

- `kernel_add.cpp` - Element-wise tensor addition
- `kernel_add_scalar.cpp` - Add a scalar value to each tensor element
- `kernel_mul.cpp` - Element-wise tensor multiplication

These kernels are compiled at runtime using the Bisheng compiler from the CANN toolkit.

## API Reference

See the main [runtime README](../../README.md) for detailed documentation on the PTO Runtime API.

## Troubleshooting

### Import Error: Cannot import pto_runtime

Make sure PYTHONPATH is set correctly:
```bash
export PYTHONPATH=/path/to/runtime/build/python:$PYTHONPATH
```

### Kernel Compilation Failed

Ensure PTO_ISA_ROOT is set:
```bash
export PTO_ISA_ROOT=/path/to/runtime/build/_deps/pto-isa-src
```

Or set it to your custom PTO-ISA installation path.

### Device Initialization Failed

- Verify CANN runtime is installed and ASCEND_HOME_PATH is set
- Check that the specified device ID is valid (0-15)
- Ensure you have permission to access the device
