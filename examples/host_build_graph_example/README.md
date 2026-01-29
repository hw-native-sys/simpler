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

The example uses Python-driven compilation via RuntimeBuilder:

```python
from runtime_builder import RuntimeBuilder

builder = RuntimeBuilder(platform="a2a3")
host_binary, aicpu_binary, aicore_binary = builder.build("host_build_graph")
```

No separate CMake build step is required - RuntimeBuilder handles compilation automatically.

## Dependencies

- Python 3
- NumPy
- CANN Runtime (Ascend) with ASCEND_HOME_PATH set
- gcc/g++ compiler

## Running the Example

### Set Environment Variables

From the build directory:

```bash
# Set PTO_ISA_ROOT for runtime kernel compilation
export PTO_ISA_ROOT=$(pwd)/_deps/pto-isa-src
```

### Run the Example

```bash
cd examples/host_build_graph_example
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

1. **Build Runtime**: RuntimeBuilder compiles host, AICPU, and AICore binaries
2. **Load Runtime Library**: `bind_host_binary()` loads the host .so via ctypes
3. **Set Device**: Initialize the target device
4. **Compile Orchestration**: Compile the orchestration function that builds the task graph
5. **Compile & Register Kernels**: Compile AIV kernels using PTOCompiler and register them
6. **Initialize Runtime**: Call `runtime.initialize()` with orchestration and input tensors
7. **Execute Runtime**: `launch_runtime()` executes the task graph on device
8. **Finalize**: Copy results back and verify correctness

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

### Import Error: Cannot import bindings

Make sure you are running from the correct directory and the python/ directory is in your path:
```bash
cd examples/host_build_graph_example
python3 main.py
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

## See Also

For a simulation-based version that runs without Ascend hardware, see [host_build_graph_sim_example](../host_build_graph_sim_example/).
