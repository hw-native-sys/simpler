# PTO Runtime Python Example - Simulation (a2a3sim)

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
| Platform | `RuntimeBuilder(platform="a2a3")` | `RuntimeBuilder(platform="a2a3sim")` |
| Requirements | CANN toolkit, Ascend device | gcc/g++ only |
| Kernel compilation | ccec (Bisheng) compiler | g++ compiler |
| Execution | AICPU/AICore on device | Host threads |
| Kernel format | PTO ISA | Plain C++ loops |

## Dependencies

- Python 3
- NumPy
- gcc/g++ compiler

No Ascend SDK or CANN toolkit required.

## Running the Example

From the repository root:

```bash
cd examples/host_build_graph_sim_example
python3 main.py
```

Optional device ID (simulation only, default 0):
```bash
python3 main.py -d 0
```

## Expected Output

```
=== Building Simulation Runtime ===
Available runtimes: ['host_build_graph']

=== Loading Runtime Library ===
Loaded runtime (xxx bytes)

=== Setting Device 0 ===

=== Compiling Orchestration Function ===
Compiled orchestration: xxx bytes

=== Compiling and Registering Simulation Kernels ===
Compiling kernels/aiv/kernel_add.cpp...
Compiling kernels/aiv/kernel_add_scalar.cpp...
Compiling kernels/aiv/kernel_mul.cpp...
All kernels compiled and registered successfully

=== Preparing Input Tensors ===
Created tensors: 16384 elements each
  host_a: all 2.0
  host_b: all 3.0
  host_f: zeros (output)
Expected result: f = (a + b + 1) * (a + b + 2) = (2+3+1)*(2+3+2) = 42.0

=== Creating and Initializing Runtime ===

=== Executing Runtime (Simulation) ===

=== Finalizing and Copying Results ===

=== Validating Results ===
First 10 elements of result (host_f):
  f[0] = 42.0
  f[1] = 42.0
  ...

SUCCESS: All 16384 elements are correct (42.0)
```

## How It Works

1. **Build Runtime**: `RuntimeBuilder(platform="a2a3sim")` compiles host, AICPU, and AICore libraries using g++
2. **Load Runtime Library**: `bind_host_binary()` loads the host .so via ctypes
3. **Set Device**: Records device ID (no actual device initialization in simulation)
4. **Compile Orchestration**: Compile the orchestration function using g++
5. **Compile & Register Kernels**: Compile simulation kernels (plain C++) and register their .text sections
6. **Initialize Runtime**: Call `runtime.initialize()` with orchestration and input tensors
7. **Execute Runtime**: `launch_runtime()` executes using host threads instead of device cores
8. **Finalize**: Results are already in host memory (no copy needed)

### Simulation Architecture

The simulation platform emulates the AICPU/AICore execution model:

- **Kernel loading**: Kernel `.text` sections are mmap'd into executable memory
- **Thread execution**: Host threads emulate AICPU scheduling and AICore computation
- **Memory**: All allocations use host memory (malloc/free)
- **Same API**: Uses identical C API as the real a2a3 platform

## Kernels

Simulation kernels are plain C++ implementations in `kernels/aiv/`:

- `kernel_add.cpp` - Element-wise tensor addition (loop-based)
- `kernel_add_scalar.cpp` - Add scalar to each tensor element (loop-based)
- `kernel_mul.cpp` - Element-wise tensor multiplication (loop-based)

These are compiled with g++ instead of the PTO compiler.

## API Reference

See the main [runtime README](../../README.md) for detailed documentation on the PTO Runtime API.

## See Also

For the hardware version that runs on real Ascend devices, see [host_build_graph_example](../host_build_graph_example/).
