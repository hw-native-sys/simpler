#!/usr/bin/env python3
"""
Main Example - PTO Runtime with C++ Runtime Builder

This program demonstrates how to use the refactored runtime builder where
the runtime initialization logic is in C++ (runtimemaker.cpp) and Python
orchestrates the runtime execution.

Flow:
1. Python: Load runtime, register kernels
2. C++ InitRuntime(): Allocates tensors, builds task structure, initializes data
3. Python launch_runtime(): Initializes device and executes the runtime
4. C++ FinalizeRuntime(): Validates results, frees tensors, calls destructor

Example usage:
   python main.py [device_id]
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add parent directory to path so we can import runtime_bindings
example_root = Path(__file__).parent
runtime_root = Path(__file__).parent.parent
runtime_dir = runtime_root / "python"
sys.path.insert(0, str(runtime_dir))

try:
    from runtime_bindings import load_runtime, register_kernel, set_device, launch_runtime
    from binary_compiler import BinaryCompiler
    from pto_compiler import PTOCompiler
    from elf_parser import extract_text_section
except ImportError:
    print("Error: Cannot import runtime_bindings module")
    print("Make sure you are running this from the correct directory")
    sys.exit(1)


def check_and_build_runtime():
    """
    Check if runtime libraries exist and build if necessary using BinaryCompiler.

    Returns:
        True if build successful or libraries exist, False otherwise
    """
    print("Building runtime using BinaryCompiler...")

    compiler = BinaryCompiler()

    # Compile AICore kernel
    print("\n[1/3] Compiling AICore kernel...")
    try:
        aicore_include_dirs = [
            str(runtime_root / "src" / "runtime" / "runtime"),
        ]
        aicore_source_dirs = [
            str(runtime_root / "src" / "runtime" / "aicore"),
            str(runtime_root / "src" / "runtime" / "runtime"),
        ]
        aicore_binary = compiler.compile("aicore", aicore_include_dirs, aicore_source_dirs)
    except Exception as e:
        print(f"✗ AICore compilation failed: {e}")
        return None

    # Compile AICPU kernel
    print("\n[2/3] Compiling AICPU kernel...")
    try:
        aicpu_include_dirs = [
            str(runtime_root / "src" / "runtime" / "runtime"),
        ]
        aicpu_source_dirs = [
            str(runtime_root / "src" / "runtime" / "aicpu"),
            str(runtime_root / "src" / "runtime" / "runtime"),
        ]
        aicpu_binary = compiler.compile("aicpu", aicpu_include_dirs, aicpu_source_dirs)
    except Exception as e:
        print(f"✗ AICPU compilation failed: {e}")
        return None

    # Compile Host runtime
    print("\n[3/3] Compiling Host runtime...")
    try:
        host_include_dirs = [
            str(runtime_root / "src" / "runtime" / "runtime"),
        ]
        host_source_dirs = [
            str(runtime_root / "src" / "runtime" / "host"),
            str(runtime_root / "src" / "runtime" / "runtime"),
        ]
        host_binary = compiler.compile("host", host_include_dirs, host_source_dirs)
    except Exception as e:
        print(f"✗ Host runtime compilation failed: {e}")
        return None

    print("\nBuild complete!")

    return (host_binary, aicpu_binary, aicore_binary)



def main():
    # Check and build runtime if necessary
    compile_results = check_and_build_runtime()
    if not compile_results:
        print("Error: Failed to build runtime libraries")
        return -1
    host_binary, aicpu_binary, aicore_binary = compile_results

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

    # Compile and register kernels (Python-side compilation)
    print("\n=== Compiling and Registering Kernels ===")
    pto_compiler = PTOCompiler()

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

    # Create and initialize runtime
    # C++ handles: allocate tensors, build tasks, initialize data
    print("\n=== Creating and Initializing Runtime ===")
    runtime = Runtime()
    runtime.initialize()

    # Execute runtime on device
    # Device init happens inside launch_runtime if not already done
    print("\n=== Executing Runtime on Device ===")
    launch_runtime(runtime,
                 aicpu_thread_num=1,
                 block_dim=1,
                 device_id=device_id,
                 aicpu_binary=aicpu_binary,
                 aicore_binary=aicore_binary)

    # Validate results and cleanup
    # C++ handles: copy results from device, validate, free tensors, call destructor
    print("\n=== Validating Results and Cleaning Up ===")
    runtime.finalize()

    return 0

if __name__ == '__main__':
    sys.exit(main())
