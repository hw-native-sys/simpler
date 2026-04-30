#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3 Worker API demo — multi-chip MoE with true inter-chip communication.

This implements a distributed MoE (Mixture of Experts) pattern with real inter-chip communication:
  - Each card has send[num_experts][num_tokens][hidden_dim] - 3D tensor
  - Dispatch: card i sends send[i][expert_j] to card j (expert owner)
  - Compute: card j computes recv[expert_j][card_i] += expert_j
  - Combine: card j sends recv[expert_j][card_i] back to card i
  - Result: output matches golden.py exactly

Data flow:
  Initial:  send[card_i][expert_j][tokens][hidden]  (per-card 3D tensor)
  Dispatch: recv[card_j][card_i][tokens][hidden]  (all-to-all transpose)
  Compute:  recv[card_j][card_i][tokens][hidden] += card_j (expert_id)
  Combine:  output[card_i][tokens][hidden] = sum_j recv[card_j][card_i][tokens][hidden]

Run:
    python examples/workers/l3/moe_multi_chip_experts/main.py -p a2a3sim -d 0-1
"""

import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipBootstrapConfig,
    ChipBufferSpec,
    ChipCallable,
    ChipCommBootstrapConfig,
    CoreCallable,
    DataType,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import Worker

from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root
from simpler_setup.torch_interop import make_tensor_arg

HERE = os.path.dirname(os.path.abspath(__file__))

# MoE configuration - matching golden.py exactly
NUM_TOKENS = 10  # Number of tokens
HIDDEN_DIM = 16  # Hidden dimension
COUNT = 4  # Number of tokens to process per (card, expert) pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--platform", required=True, choices=["a2a3sim", "a2a3", "a5sim", "a5"])
    parser.add_argument("-d", "--device", default="0-1", help="Device range, e.g. '0-1' or '0,1'")
    return parser.parse_args()


def parse_device_range(spec: str) -> list[int]:
    """Parse device range specification like '0-1' or '0,1' into a list of IDs."""
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        ids = list(range(lo, hi + 1))
    elif "," in spec:
        ids = [int(x) for x in spec.split(",")]
    else:
        ids = [int(spec)]
    return ids
    return ids


def build_moe_comm_callable(platform: str) -> ChipCallable:
    """Build MoE callable with inter-chip communication (dispatch-compute-combine)."""
    print("[moe_multi_chip] [DEBUG] Starting kernel compilation...", flush=True)
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root(clone_protocol="https")
    print(f"[moe_multi_chip] [DEBUG] pto_isa_root: {pto_isa_root}", flush=True)
    include_dirs = kc.get_orchestration_include_dirs(runtime)

    # Add platform_comm include directory for CommContext
    kernel_include_dirs = list(include_dirs) + [str(kc.project_root / "src" / "common")]

    # Build three kernels
    print("[moe_multi_chip] [DEBUG] Compiling dispatch kernel...", flush=True)
    dispatch_bytes = kc.compile_incore(
        source_path=os.path.join(HERE, "kernels/aiv/moe_dispatch_alltoall.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=kernel_include_dirs,
    )
    print("[moe_multi_chip] [DEBUG] Dispatch kernel compiled", flush=True)

    print("[moe_multi_chip] [DEBUG] Compiling simple compute kernel...", flush=True)
    compute_bytes = kc.compile_incore(
        source_path=os.path.join(HERE, "kernels/aiv/moe_simple_compute.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=include_dirs,
    )
    print("[moe_multi_chip] [DEBUG] Simple compute kernel compiled", flush=True)

    print("[moe_multi_chip] [DEBUG] Compiling combine kernel...", flush=True)
    combine_bytes = kc.compile_incore(
        source_path=os.path.join(HERE, "kernels/aiv/moe_combine_alltoall.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=kernel_include_dirs,
    )
    print("[moe_multi_chip] [DEBUG] Combine kernel compiled", flush=True)

    if not platform.endswith("sim"):
        print("[moe_multi_chip] [DEBUG] Extracting text sections from ELF binaries...", flush=True)
        from simpler_setup.elf_parser import extract_text_section
        dispatch_bytes = extract_text_section(dispatch_bytes)
        compute_bytes = extract_text_section(compute_bytes)
        combine_bytes = extract_text_section(combine_bytes)
        print("[moe_multi_chip] [DEBUG] Text sections extracted", flush=True)

    print("[moe_multi_chip] [DEBUG] Compiling orchestration...", flush=True)
    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "kernels/orchestration/moe_comm_orch.cpp"),
    )
    print("[moe_multi_chip] [DEBUG] Orchestration compiled", flush=True)

    # Build core callables
    dispatch_cc = CoreCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT,
                   ArgDirection.IN, ArgDirection.IN, ArgDirection.IN],
        binary=dispatch_bytes,
    )

    compute_cc = CoreCallable.build(
        signature=[ArgDirection.INOUT, ArgDirection.IN, ArgDirection.IN, ArgDirection.IN],
        binary=compute_bytes,
    )

    combine_cc = CoreCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT,
                   ArgDirection.IN, ArgDirection.IN, ArgDirection.IN],
        binary=combine_bytes,
    )

    return ChipCallable.build(
        signature=[
            ArgDirection.IN,   # send[num_experts][num_tokens][hidden_dim]
            ArgDirection.OUT,  # recv[num_cards][num_tokens][hidden_dim]
            ArgDirection.OUT,  # output[num_tokens][hidden_dim]
            ArgDirection.INOUT,  # scratch HCCL buffer
            ArgDirection.IN,   # expert_id
            ArgDirection.IN,   # card_id
            ArgDirection.IN,   # num_cards
            ArgDirection.IN,   # CommContext*
        ],
        func_name="aicpu_orchestration_entry",
        binary=orch_bytes,
        children=[(0, dispatch_cc), (1, compute_cc), (2, combine_cc)],
    )


def run(platform: str, device_ids: list[int]) -> int:
    """Core logic - implements true inter-chip communication MoE."""
    print("[moe_multi_chip] [DEBUG] run() function started", flush=True)
    num_cards = len(device_ids)
    num_experts = num_cards  # One expert per chip

    print(f"[moe_multi_chip] devices={device_ids} num_cards={num_cards} num_experts={num_experts}", flush=True)
    print(f"[moe_multi_chip] NUM_TOKENS={NUM_TOKENS} HIDDEN_DIM={HIDDEN_DIM} COUNT={COUNT}", flush=True)

    # Configure HCCL communication
    # Scratch buffer size: num_cards * num_cards slots (all cards' data)
    # Layout: scratch[card_j][expert_i][tokens][hidden_dim]
    scratch_count = num_cards * num_cards * NUM_TOKENS * HIDDEN_DIM
    scratch_nbytes = scratch_count * 4  # float32

    # Allocate space for signals at tail of scratch
    total_scratch_nbytes = scratch_nbytes + num_cards * 4  # + num_cards int32 signals
    window_size = max(total_scratch_nbytes, 4 * 1024)

    rootinfo_path = f"/tmp/pto_moe_multi_chip_{os.getpid()}.bin"
    print(f"[moe_multi_chip] [DEBUG] HCCL config: scratch_count={scratch_count} window_size={window_size} rootinfo={rootinfo_path}", flush=True)

    # Clean up any stale rootinfo file
    try:
        os.unlink(rootinfo_path)
        print(f"[moe_multi_chip] [DEBUG] Cleaned up stale rootinfo file", flush=True)
    except FileNotFoundError:
        print(f"[moe_multi_chip] [DEBUG] No stale rootinfo file to clean", flush=True)
        pass

    torch.manual_seed(42)
    print("[moe_multi_chip] [DEBUG] Random seed set", flush=True)

    # Per-card data layout (3D/2D as per user requirement)
    # send[i]: [num_experts, num_tokens, hidden_dim]
    host_send = [torch.ones(num_experts, NUM_TOKENS, HIDDEN_DIM, dtype=torch.float32).share_memory_()
                 for _ in device_ids]

    # recv[i]: [num_cards, num_tokens, hidden_dim] - receives data from all cards for expert_i
    host_recv = [torch.zeros(num_cards, NUM_TOKENS, HIDDEN_DIM, dtype=torch.float32).share_memory_()
                 for _ in device_ids]

    # output[i]: [num_tokens, hidden_dim]
    host_output = [torch.zeros(NUM_TOKENS, HIDDEN_DIM, dtype=torch.float32).share_memory_()
                   for _ in device_ids]

    print("[moe_multi_chip] [DEBUG] All tensors allocated, host_send initialized to 1.0", flush=True)

    # Configure HCCL bootstrap for each card
    cfgs = [
        ChipBootstrapConfig(
            comm=ChipCommBootstrapConfig(
                rank=rank,
                nranks=num_cards,
                rootinfo_path=rootinfo_path,
                window_size=window_size,
            ),
            buffers=[
                ChipBufferSpec(
                    name="scratch",
                    dtype="float32",
                    count=scratch_count,
                    nbytes=total_scratch_nbytes,
                ),
            ],
        )
        for rank in range(num_cards)
    ]

    print("[moe_multi_chip] [DEBUG] Creating Worker...", flush=True)
    worker = Worker(
        level=3,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=device_ids,
        num_sub_workers=0,
        chip_bootstrap_configs=cfgs,
    )
    print("[moe_multi_chip] [DEBUG] Worker created", flush=True)

    print(f"[moe_multi_chip] compiling kernels for {platform}...", flush=True)
    moe_cc = build_moe_comm_callable(platform)
    print("[moe_multi_chip] [DEBUG] All kernels compiled successfully", flush=True)

    print("[moe_multi_chip] init worker (with HCCL communication)...", flush=True)
    worker.init()
    print("[moe_multi_chip] [DEBUG] Worker initialized", flush=True)

    # Get chip contexts (contains CommContext pointers)
    contexts = worker.chip_contexts
    print(f"[moe_multi_chip] chip contexts: {len(contexts)}", flush=True)
    for i, ctx in enumerate(contexts):
        print(f"[moe_multi_chip]   card {i}: rank={ctx.rank}/{ctx.nranks} device_ctx=0x{ctx.device_ctx:x}", flush=True)

    try:
        # 第一次运行：只执行到dispatch阶段，查看recv数据
        # 注意：当前orchestration是一次性执行所有3个阶段，所以无法分阶段查看
        # 这里我们运行完整流程，然后在host端查看最终结果

        def orch_fn(orch, _args, cfg):
            print(f"[moe_multi_chip] orch_fn: Starting submission for {num_cards} cards", flush=True)
            # Each card submits a task that:
            # 1. Dispatches its expert data to all cards
            # 2. Computes on received data
            # 3. Combines results back to source cards
            for i in range(num_cards):
                print(f"[moe_multi_chip] orch_fn: Submitting task for card {i} (worker {i})", flush=True)
                moe_args = TaskArgs()
                moe_args.add_tensor(make_tensor_arg(host_send[i]), TensorArgType.INPUT)
                moe_args.add_tensor(make_tensor_arg(host_recv[i]), TensorArgType.OUTPUT_EXISTING)
                moe_args.add_tensor(make_tensor_arg(host_output[i]), TensorArgType.OUTPUT_EXISTING)

                # Scratch buffer (HCCL window)
                from simpler.task_interface import ContinuousTensor
                moe_args.add_tensor(
                    ContinuousTensor.make(
                        data=contexts[i].buffer_ptrs["scratch"],
                        shapes=(scratch_count,),
                        dtype=DataType.FLOAT32,
                        child_memory=True,
                    ),
                    TensorArgType.INOUT,
                )

                moe_args.add_scalar(i)  # expert_id
                moe_args.add_scalar(i)  # card_id
                moe_args.add_scalar(num_cards)
                moe_args.add_scalar(contexts[i].device_ctx)

                result = orch.submit_next_level(moe_cc, moe_args, cfg, worker=i)
                print(f"[moe_multi_chip] orch_fn: Submitted task for card {i}, result={result}", flush=True)

            print(f"[moe_multi_chip] orch_fn: All {num_cards} tasks submitted", flush=True)

        print("[moe_multi_chip] running multi-chip MoE DAG with inter-chip communication...", flush=True)
        print("[moe_multi_chip] [DEBUG] About to call worker.run()...", flush=True)
        worker.run(orch_fn, args=None, config=CallConfig())
        print("[moe_multi_chip] [DEBUG] worker.run() completed", flush=True)

        # 打印host端的recv数据（这是所有阶段完成后的最终recv状态）
        print("\n[moe_multi_chip] ===== Host-side recv data (after all stages) =====")
        for i in range(num_cards):
            print(f"[moe_multi_chip] Card {i} recv shape: {host_recv[i].shape}")
            print(f"[moe_multi_chip] Card {i} recv sample (first 2 cards' data, first 2 tokens, first 3 dims):")
            for card_j in range(min(2, num_cards)):
                for t in range(min(2, COUNT)):
                    print(f"  recv[{card_j}][{t}][:3] = {host_recv[i][card_j, t, :3].tolist()}")

        # 打印host端的output数据
        print("\n[moe_multi_chip] ===== Host-side output data (final) =====")
        for i in range(num_cards):
            print(f"[moe_multi_chip] Card {i} output shape: {host_output[i].shape}")
            print(f"[moe_multi_chip] Card {i} output sample (first {COUNT} tokens, first 3 dims):")
            for t in range(COUNT):
                print(f"  output[{t}][:3] = {host_output[i][t, :3].tolist()}")

        print("\n[moe_multi_chip] Results:")
        for i in range(num_cards):
            print(f"[moe_multi_chip] card {i} output shape: {host_output[i].shape}")
            print(f"[moe_multi_chip] card {i} output sample (first {COUNT} tokens, first 3 dims):")
            for t in range(COUNT):
                print(f"  token {t}: {host_output[i][t, :3]}")

        # Verify against golden.py
        print("\n[moe_multi_chip] Verifying against golden.py...")

        # For golden, we need to reconstruct the original input data
        # host_send[i]: [num_experts, NUM_TOKENS, HIDDEN_DIM]
        # Convert to golden format: [num_cards, num_experts, NUM_TOKENS, HIDDEN_DIM]
        send_batch = torch.stack(host_send)  # [num_cards, num_experts, NUM_TOKENS, HIDDEN_DIM]

        # Initialize recv in golden format: [num_experts, num_cards, NUM_TOKENS, HIDDEN_DIM]
        # This will be filled by the dispatch phase
        recv_batch = torch.zeros(num_experts, num_cards, NUM_TOKENS, HIDDEN_DIM, dtype=torch.float32)

        # Initialize output for golden as ZERO tensor (not containing hardware results!)
        # golden.py's demo function uses +=, so it must start from zero
        golden_output_input = torch.zeros(num_cards, NUM_TOKENS, HIDDEN_DIM, dtype=torch.float32)

        # Run golden to compute expected output
        # Note: golden.py's demo function modifies recv and output in place
        import sys
        golden_path = os.path.join(HERE, "golden.py")
        if golden_path not in sys.path:
            sys.path.insert(0, HERE)

        # Import golden module
        import importlib.util
        spec = importlib.util.spec_from_file_location("golden", golden_path)
        golden_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(golden_module)

        # Run golden computation (modifies golden_output_input in place)
        # The golden function computes: output[i][:][:] = sum_j (send[j][i] + i)
        # where only the first COUNT tokens are processed
        golden_output = golden_module.demo(send_batch, recv_batch, golden_output_input)

        # Compare results
        all_match = True
        for i in range(num_cards):
            max_diff = float(torch.max(torch.abs(host_output[i] - golden_output[i])))
            mean_diff = float(torch.mean(torch.abs(host_output[i] - golden_output[i])))
            print(f"[moe_multi_chip] card {i}: max |output - golden| = {max_diff:.6e}, mean diff = {mean_diff:.6e}")

            if max_diff > 1e-3:
                all_match = False
                print(f"[moe_multi_chip] card {i} MISMATCH! Showing first {COUNT} tokens:")
                for t in range(COUNT):
                    actual = host_output[i][t, :3]
                    expected = golden_output[i][t, :3]
                    print(f"  token {t}: actual={actual.tolist()}, expected={expected.tolist()}")
            else:
                print(f"[moe_multi_chip] card {i} ✅ matches golden")

        if all_match:
            print("\n[moe_multi_chip] ✅ All cards matched golden.py!")
            return 0
        else:
            print("\n[moe_multi_chip] ❌ Some cards did NOT match golden.py")
            return 1

    except Exception as e:
        print(f"[moe_multi_chip] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        print("[moe_multi_chip] shutting down worker...")
        worker.close()

        # Clean up rootinfo file
        try:
            os.unlink(rootinfo_path)
        except FileNotFoundError:
            pass


def main() -> int:
    args = parse_args()
    device_ids = parse_device_range(args.device)
    return run(args.platform, device_ids)


if __name__ == "__main__":
    sys.exit(main())
