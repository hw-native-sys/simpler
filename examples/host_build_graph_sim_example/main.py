#!/usr/bin/env python3
"""
A2A3Sim Python Example - Thread-based Simulation via Python

This demonstrates calling the a2a3sim simulation platform from Python,
following the same flow as host_build_graph_example/main.py:

1. Build simulation runtime as .so via RuntimeBuilder
2. Compile kernels to .o with g++, extract .text, register as binary
3. Compile orchestration to .so, initialize runtime, execute

The simulation uses threads on host to emulate device execution.
Kernel .text sections are loaded into mmap'd executable memory.

Example usage:
    python main.py
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Add parent directory to path so we can import bindings
example_root = Path(__file__).parent
runtime_root = Path(__file__).parent.parent.parent
runtime_dir = runtime_root / "python"
sys.path.insert(0, str(runtime_dir))
sys.path.insert(0, str(example_root))

try:
    from runtime_builder import RuntimeBuilder
    from bindings import bind_host_binary, register_kernel, set_device, launch_runtime
    from elf_parser import extract_text_section
    from kernels.kernel_config import KERNELS, ORCHESTRATION
except ImportError as e:
    print(f"Error: Cannot import module: {e}")
    print("Make sure you are running this from the correct directory")
    sys.exit(1)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="A2A3Sim Python Example")
    parser.add_argument("-d", "--device", type=int, default=0,
                        help="Device ID (simulation, default: 0)")
    args = parser.parse_args()

    device_id = args.device

    # Build simulation runtime
    print("\n=== Building Simulation Runtime ===")
    builder = RuntimeBuilder(platform="a2a3sim")
    pto_compiler = builder.get_pto_compiler()
    print(f"Available runtimes: {builder.list_runtimes()}")
    try:
        host_binary, aicpu_binary, aicore_binary = builder.build("host_build_graph")
    except Exception as e:
        print(f"Error: Failed to build runtime libraries: {e}")
        return -1

    # Load runtime library and get Runtime class
    print("\n=== Loading Runtime Library ===")
    Runtime = bind_host_binary(host_binary)
    print(f"Loaded runtime ({len(host_binary)} bytes)")

    # Set device (no-op in simulation, records device ID)
    print(f"\n=== Setting Device {device_id} ===")
    set_device(device_id)

    # Compile orchestration shared library
    print("\n=== Compiling Orchestration Function ===")

    orch_so_binary = pto_compiler.compile_orchestration(
        ORCHESTRATION["source"],
        extra_include_dirs=[
            str(runtime_root / "src" / "runtime" / "host_build_graph" / "runtime"),  # for runtime.h
        ] + pto_compiler.get_platform_include_dirs()  # for device_runner.h
    )
    print(f"Compiled orchestration: {len(orch_so_binary)} bytes")

    # Compile and register simulation kernels
    print("\n=== Compiling and Registering Simulation Kernels ===")

    # Define pto_isa_root for a2a3 platform (not needed for a2a3sim, but kept for compatibility)
    pto_isa_root = "/data/wcwxy/workspace/pypto/pto-isa"

    for kernel in KERNELS:
        print(f"Compiling {kernel['source']}...")
        # compile_incore handles platform dispatch internally
        kernel_o = pto_compiler.compile_incore(
            kernel["source"],
            core_type=kernel.get("core_type", "aiv"),
            pto_isa_root=pto_isa_root
        )

        # Extract .text section (raw machine code)
        kernel_bin = extract_text_section(kernel_o)
        # Register kernel binary (will be mmap'd as executable memory)
        register_kernel(kernel["func_id"], kernel_bin)

    print("All kernels compiled and registered successfully")

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
    runtime.initialize(orch_so_binary, ORCHESTRATION["function_name"], func_args)

    # Execute runtime (simulation: uses threads)
    print("\n=== Executing Runtime (Simulation) ===")
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
