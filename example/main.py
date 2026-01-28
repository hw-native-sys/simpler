#!/usr/bin/env python3
"""
Main Example - PTO Runtime with Dynamic Orchestration

This program demonstrates how to use the runtime with dynamic orchestration
functions that can be compiled and loaded at runtime.

Flow:
1. Python: Load runtime, compile orchestration, register kernels
2. Python: Prepare input tensors (numpy arrays)
3. C++ InitRuntime(): Calls orchestration to allocate device memory, copy data, build graph
4. Python launch_runtime(): Executes the runtime on device
5. C++ FinalizeRuntime(): Copies results back to host, frees device memory

Example usage:
   python main.py [device_id]
"""

import sys
import os
from pathlib import Path
import numpy as np
from ctypes import cdll, c_void_p, c_uint64, c_int, POINTER, CFUNCTYPE

# Add parent directory to path so we can import runtime_bindings
example_root = Path(__file__).parent
runtime_root = Path(__file__).parent.parent
runtime_dir = runtime_root / "python"
sys.path.insert(0, str(runtime_dir))

try:
    from runtime_bindings import load_runtime, register_kernel, set_device, launch_runtime, OrchestrationFunc
    from runtime_builder import RuntimeBuilder
    from pto_compiler import PTOCompiler
    from elf_parser import extract_text_section
except ImportError as e:
    print(f"Error: Cannot import runtime_bindings module: {e}")
    print("Make sure you are running this from the correct directory")
    sys.exit(1)


def main():
    # Check and build runtime if necessary
    builder = RuntimeBuilder()
    print(f"Available runtimes: {builder.list_runtimes()}")
    try:
        host_binary, aicpu_binary, aicore_binary = builder.build("host_build_graph")
    except Exception as e:
        print(f"Error: Failed to build runtime libraries: {e}")
        return -1

    # Parse device ID from command line
    device_id = 9
    if len(sys.argv) > 1:
        try:
            device_id = int(sys.argv[1])
            if device_id < 0 or device_id > 15:
                print(f"Error: deviceId ({device_id}) out of range [0, 15]")
                return -1
        except ValueError:
            print(f"Error: invalid deviceId argument: {sys.argv[1]}")
            return -1

    # Load runtime library and get Runtime class
    print("\n=== Loading Runtime Library ===")
    Runtime = load_runtime(host_binary)
    print(f"Loaded runtime ({len(host_binary)} bytes)")

    # Compile orchestration shared library
    print("\n=== Compiling Orchestration Function ===")
    pto_compiler = PTOCompiler()

    orch_so_path = pto_compiler.compile_orchestration(
        str(example_root / "kernels" / "orchestration" / "example_orch.cpp"),
        extra_include_dirs=[
            str(runtime_root / "src" / "runtime" / "host_build_graph" / "runtime"),  # for runtime.h
            str(runtime_root / "src" / "platform" / "a2a3" / "host"),                 # for devicerunner.h
        ]
    )
    print(f"Compiled orchestration: {orch_so_path}")

    # Load orchestration function from shared library
    orch_lib = cdll.LoadLibrary(orch_so_path)
    orch_lib.BuildExampleGraph.argtypes = [c_void_p, POINTER(c_uint64), c_int]
    orch_lib.BuildExampleGraph.restype = c_int

    def build_example_graph(runtime, args, arg_count):
        """Wrapper to call C++ orchestration function."""
        return orch_lib.BuildExampleGraph(runtime, args, arg_count)

    print("Loaded orchestration function: BuildExampleGraph")

    # Compile and register kernels (Python-side compilation)
    print("\n=== Compiling and Registering Kernels ===")

    pto_isa_root = "/data/wcwxy/workspace/pypto/pto-isa"

    # Compile kernel_add (func_id=0)
    print("Compiling kernel_add.cpp...")
    kernel_add_o = pto_compiler.compile_kernel(
        str(example_root / "kernels" / "aiv" / "kernel_add.cpp"),
        core_type=1,  # AIV
        pto_isa_root=pto_isa_root
    )
    kernel_add_bin = extract_text_section(kernel_add_o)
    register_kernel(0, kernel_add_bin)

    # Compile kernel_add_scalar (func_id=1)
    print("Compiling kernel_add_scalar.cpp...")
    kernel_add_scalar_o = pto_compiler.compile_kernel(
        str(example_root / "kernels" / "aiv" / "kernel_add_scalar.cpp"),
        core_type=1,  # AIV
        pto_isa_root=pto_isa_root
    )
    kernel_add_scalar_bin = extract_text_section(kernel_add_scalar_o)
    register_kernel(1, kernel_add_scalar_bin)

    # Compile kernel_mul (func_id=2)
    print("Compiling kernel_mul.cpp...")
    kernel_mul_o = pto_compiler.compile_kernel(
        str(example_root / "kernels" / "aiv" / "kernel_mul.cpp"),
        core_type=1,  # AIV
        pto_isa_root=pto_isa_root
    )
    kernel_mul_bin = extract_text_section(kernel_mul_o)
    register_kernel(2, kernel_mul_bin)

    print("All kernels compiled and registered successfully")

    # Set device before creating runtime (enables memory allocation)
    print(f"\n=== Setting Device {device_id} ===")
    set_device(device_id)

    # Prepare input tensors
    print("\n=== Preparing Input Tensors ===")
    ROWS = 128
    COLS = 128
    SIZE = ROWS * COLS  # 16384 elements

    # Create numpy arrays for inputs
    host_a = np.full(SIZE, 2.0, dtype=np.float32)
    host_b = np.full(SIZE, 3.0, dtype=np.float32)
    host_f = np.zeros(SIZE, dtype=np.float32)  # Output tensor

    print(f"Created tensors: {SIZE} elements each")
    print(f"  host_a: all 2.0")
    print(f"  host_b: all 3.0")
    print(f"  host_f: zeros (output)")
    print(f"Expected result: f = (a + b + 1) * (a + b + 2) = (2+3+1)*(2+3+2) = 42.0")

    # Build func_args: [host_a_ptr, host_b_ptr, host_f_ptr, size_a, size_b, size_f, SIZE]
    func_args = [
        host_a.ctypes.data,   # host_a pointer
        host_b.ctypes.data,   # host_b pointer
        host_f.ctypes.data,   # host_f pointer (output)
        host_a.nbytes,        # size_a in bytes
        host_b.nbytes,        # size_b in bytes
        host_f.nbytes,        # size_f in bytes
        SIZE,                 # number of elements
    ]

    # Create and initialize runtime
    print("\n=== Creating and Initializing Runtime ===")
    runtime = Runtime()
    runtime.initialize(build_example_graph, func_args)

    # Execute runtime on device
    print("\n=== Executing Runtime on Device ===")
    launch_runtime(runtime,
                 aicpu_thread_num=3,
                 block_dim=3,
                 device_id=device_id,
                 aicpu_binary=aicpu_binary,
                 aicore_binary=aicore_binary)

    # Finalize and copy results back to host
    print("\n=== Finalizing and Copying Results ===")
    runtime.finalize()

    # Validate results
    print("\n=== Validating Results ===")
    print(f"First 10 elements of result (host_f):")
    for i in range(10):
        print(f"  f[{i}] = {host_f[i]}")

    # Check if all elements are correct
    expected = 42.0
    all_correct = np.allclose(host_f, expected, rtol=1e-5)
    error_count = np.sum(~np.isclose(host_f, expected, rtol=1e-5))

    if all_correct:
        print(f"\nSUCCESS: All {SIZE} elements are correct (42.0)")
    else:
        print(f"\nFAILED: {error_count} elements are incorrect")

    return 0 if all_correct else -1


if __name__ == '__main__':
    sys.exit(main())
