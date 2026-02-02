#!/usr/bin/env python3
"""
Paged Attention AIC/AIV Split Simulation

This example demonstrates paged attention with manual AIC/AIV subgraph splitting:
- AIC (AICore): QK MatMul, PV MatMul
- AIV (AIVector): Softmax Prepare, Online Update, Normalize

Uses a2a3sim platform for thread-based simulation.
"""

import sys
import struct
import argparse
from pathlib import Path
import numpy as np

# Add parent directory to path
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
    from golden import paged_attention_golden, generate_test_data
except ImportError as e:
    print(f"Error: Cannot import module: {e}")
    print("Make sure you are running this from the correct directory")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Paged Attention AIC/AIV Split Simulation")
    parser.add_argument("-d", "--device", type=int, default=0, help="Device ID")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of query heads")
    parser.add_argument("--kv_head_num", type=int, default=1, help="Number of KV heads")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--block_size", type=int, default=64, help="Block size")
    parser.add_argument("--block_num", type=int, default=4, help="Number of blocks per batch")
    parser.add_argument("--context_len", type=int, default=256, help="Context length")
    args = parser.parse_args()

    device_id = args.device

    # Generate test data
    print("\n=== Generating Test Data ===")
    data = generate_test_data(
        batch=args.batch,
        num_heads=args.num_heads,
        kv_head_num=args.kv_head_num,
        head_dim=args.head_dim,
        block_size=args.block_size,
        block_num=args.block_num,
        context_len=args.context_len,
    )
    
    print(f"batch={data['batch']}, num_heads={data['num_heads']}, head_dim={data['head_dim']}")
    print(f"kv_head_num={data['kv_head_num']}, block_size={data['block_size']}, block_num={data['block_num']}")

    # Compute golden result
    print("\n=== Computing Golden Result ===")
    golden_out = paged_attention_golden(
        data["query"],
        data["key_cache"],
        data["value_cache"],
        data["block_table"],
        data["context_lens"],
        data["scale_value"],
    )
    print(f"Golden output shape: {golden_out.shape}")
    print(f"Golden output range: [{golden_out.min():.4f}, {golden_out.max():.4f}]")

    # Build simulation runtime
    print("\n=== Building Simulation Runtime ===")
    builder = RuntimeBuilder(platform="a2a3sim")
    pto_compiler = builder.get_pto_compiler()
    
    try:
        host_binary, aicpu_binary, aicore_binary = builder.build("host_build_graph")
    except Exception as e:
        print(f"Error: Failed to build runtime libraries: {e}")
        return -1

    # Load runtime library
    print("\n=== Loading Runtime Library ===")
    Runtime = bind_host_binary(host_binary)
    set_device(device_id)

    # Compile orchestration
    print("\n=== Compiling Orchestration Function ===")
    orch_so_binary = pto_compiler.compile_orchestration(
        ORCHESTRATION["source"],
        extra_include_dirs=[
            str(runtime_root / "src" / "runtime" / "host_build_graph" / "runtime"),
        ] + pto_compiler.get_platform_include_dirs()
    )
    print(f"Compiled orchestration: {len(orch_so_binary)} bytes")

    # Compile and register kernels
    print("\n=== Compiling and Registering Kernels ===")
    pto_isa_root = "/data/wcwxy/workspace/pypto/pto-isa"

    for kernel in KERNELS:
        print(f"Compiling {Path(kernel['source']).name}...")
        kernel_o = pto_compiler.compile_incore(
            kernel["source"],
            core_type=kernel.get("core_type", "aiv"),
            pto_isa_root=pto_isa_root
        )
        kernel_bin = extract_text_section(kernel_o)
        register_kernel(kernel["func_id"], kernel_bin)

    print("All kernels compiled and registered")

    # Prepare tensors
    print("\n=== Preparing Tensors ===")
    host_query = data["query"].astype(np.float32).flatten()
    host_key_cache = data["key_cache"].astype(np.float32).flatten()
    host_value_cache = data["value_cache"].astype(np.float32).flatten()
    host_block_table = data["block_table"].astype(np.int32).flatten()
    host_context_lens = data["context_lens"].astype(np.int32)
    host_out = np.zeros(data["batch"] * data["num_heads"] * data["head_dim"], dtype=np.float32)

    # Convert scale_value to uint64 bits
    scale_bytes = struct.pack('f', data["scale_value"])
    scale_bits = struct.unpack('I', scale_bytes)[0]

    # Build func_args
    func_args = [
        host_query.ctypes.data,
        host_key_cache.ctypes.data,
        host_value_cache.ctypes.data,
        host_block_table.ctypes.data,
        host_context_lens.ctypes.data,
        host_out.ctypes.data,
        host_query.nbytes,
        host_key_cache.nbytes,
        host_value_cache.nbytes,
        host_block_table.nbytes,
        host_context_lens.nbytes,
        host_out.nbytes,
        data["batch"],
        data["num_heads"],
        data["kv_head_num"],
        data["head_dim"],
        data["block_size"],
        data["block_num"],
        scale_bits,
    ]

    # Create and initialize runtime
    print("\n=== Creating and Initializing Runtime ===")
    runtime = Runtime()
    runtime.initialize(orch_so_binary, ORCHESTRATION["function_name"], func_args)

    # Execute runtime
    print("\n=== Executing Runtime (Simulation) ===")
    launch_runtime(runtime,
                   aicpu_thread_num=3,
                   block_dim=3,
                   device_id=device_id,
                   aicpu_binary=aicpu_binary,
                   aicore_binary=aicore_binary)

    # Finalize
    print("\n=== Finalizing ===")
    runtime.finalize()

    # Validate results
    print("\n=== Validating Results ===")
    sim_out = host_out.reshape(data["batch"], data["num_heads"], data["head_dim"])
    
    print(f"Simulation output shape: {sim_out.shape}")
    print(f"Simulation output range: [{sim_out.min():.4f}, {sim_out.max():.4f}]")

    # Compare with golden
    max_error = np.abs(sim_out - golden_out).max()
    mean_error = np.abs(sim_out - golden_out).mean()
    
    print(f"\nMax absolute error: {max_error:.6f}")
    print(f"Mean absolute error: {mean_error:.6f}")

    # Check tolerance
    rtol = 1e-3
    atol = 1e-3
    all_close = np.allclose(sim_out, golden_out, rtol=rtol, atol=atol)
    error_count = np.sum(~np.isclose(sim_out, golden_out, rtol=rtol, atol=atol))
    total_elements = sim_out.size

    if all_close:
        print(f"\nSUCCESS: All {total_elements} elements match within tolerance")
    else:
        error_ratio = error_count / total_elements * 100
        print(f"\nWARNING: {error_count}/{total_elements} ({error_ratio:.2f}%) elements exceed tolerance")
        if error_ratio < 1.0:
            print("Error ratio < 1%, considered acceptable for bfloat16 precision")

    # Print sample values
    print("\nSample comparison (first 5 elements of first head):")
    for i in range(min(5, data["head_dim"])):
        print(f"  [{i}] golden={golden_out[0,0,i]:.6f}, sim={sim_out[0,0,i]:.6f}, "
              f"diff={abs(golden_out[0,0,i]-sim_out[0,0,i]):.6f}")

    return 0 if all_close else 1


if __name__ == '__main__':
    sys.exit(main())
