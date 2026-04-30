#!/usr/bin/env python3
# Test dispatch kernel in isolation

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

# MoE configuration
NUM_TOKENS = 10
HIDDEN_DIM = 16
COUNT = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Test dispatch kernel in isolation")
    parser.add_argument("-p", "--platform", required=True, choices=["a2a3sim", "a2a3", "a5sim", "a5"])
    parser.add_argument("-d", "--device", default="0-1", help="Device range")
    return parser.parse_args()


def parse_device_range(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        return list(range(lo, hi + 1))
    elif "," in spec:
        return [int(x) for x in spec.split(",")]
    else:
        return [int(spec)]


def build_dispatch_only_callable(platform: str) -> ChipCallable:
    """Build callable with ONLY dispatch kernel."""
    print("[Dispatch-Only] Compiling dispatch kernel...", flush=True)
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root(clone_protocol="https")
    include_dirs = kc.get_orchestration_include_dirs(runtime)
    kernel_include_dirs = list(include_dirs) + [str(kc.project_root / "src" / "common")]

    # Compile dispatch kernel
    dispatch_bytes = kc.compile_incore(
        source_path=os.path.join(HERE, "kernels/aiv/moe_dispatch_alltoall.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=kernel_include_dirs,
    )
    print("[Dispatch-Only] Dispatch kernel compiled", flush=True)

    if not platform.endswith("sim"):
        from simpler_setup.elf_parser import extract_text_section
        dispatch_bytes = extract_text_section(dispatch_bytes)
        print("[Dispatch-Only] Text sections extracted", flush=True)

    # Compile orchestration
    print("[Dispatch-Only] Compiling orchestration...", flush=True)
    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "kernels/orchestration/moe_dispatch_only_orch.cpp"),
    )
    print("[Dispatch-Only] Orchestration compiled", flush=True)

    # Build core callable
    dispatch_cc = CoreCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT,
                   ArgDirection.IN, ArgDirection.IN, ArgDirection.IN],
        binary=dispatch_bytes,
    )

    return ChipCallable.build(
        signature=[
            ArgDirection.IN,   # send
            ArgDirection.OUT,  # recv
            ArgDirection.OUT,  # output (unused but needed for signature)
            ArgDirection.INOUT,  # scratch
            ArgDirection.IN,   # expert_id
            ArgDirection.IN,   # card_id
            ArgDirection.IN,   # num_cards
            ArgDirection.IN,   # CommContext*
        ],
        func_name="aicpu_orchestration_entry",
        binary=orch_bytes,
        children=[(0, dispatch_cc)],  # Only dispatch child
    )


def run(platform: str, device_ids: list[int]) -> int:
    print(f"[Dispatch-Only] Testing dispatch on devices {device_ids}", flush=True)
    num_cards = len(device_ids)
    num_experts = num_cards

    # Configure HCCL
    scratch_count = num_cards * num_cards * NUM_TOKENS * HIDDEN_DIM
    scratch_nbytes = scratch_count * 4
    total_scratch_nbytes = scratch_nbytes + num_cards * 4
    window_size = max(total_scratch_nbytes, 4 * 1024)

    rootinfo_path = f"/tmp/pto_dispatch_only_{os.getpid()}.bin"
    try:
        os.unlink(rootinfo_path)
    except FileNotFoundError:
        pass

    torch.manual_seed(42)

    # Allocate tensors with unique values to trace data flow
    # Value encoding: (card_id * 1000000) + (expert_id * 10000) + (token * 100) + dim
    host_send = []
    for i, device_id in enumerate(device_ids):
        send = torch.zeros(num_experts, NUM_TOKENS, HIDDEN_DIM, dtype=torch.float32).share_memory_()
        for expert_j in range(num_experts):
            for t in range(NUM_TOKENS):
                for d in range(HIDDEN_DIM):
                    # Unique value: card_i -> expert_j -> token_t -> dim_d
                    value = float(i * 1000000 + expert_j * 10000 + t * 100 + d)
                    send[expert_j, t, d] = value
        host_send.append(send)

    host_recv = [torch.zeros(num_cards, NUM_TOKENS, HIDDEN_DIM, dtype=torch.float32).share_memory_()
                 for _ in device_ids]
    host_output = [torch.zeros(NUM_TOKENS, HIDDEN_DIM, dtype=torch.float32).share_memory_()
                   for _ in device_ids]

    print(f"[Dispatch-Only] Allocated tensors with unique values", flush=True)
    print(f"[Dispatch-Only] Value encoding: (card_id * 1000000) + (expert_id * 10000) + (token * 100) + dim", flush=True)
    print(f"[Dispatch-Only] Sample: host_send[0][0][0][0] = {host_send[0][0, 0, 0].item()} (card 0, expert 0, token 0, dim 0)", flush=True)

    # Print input values BEFORE running kernel
    print("\n" + "="*80)
    print("[Dispatch-Only] INPUT SEND VALUES (before kernel):")
    print("="*80)
    for i in range(num_cards):
        print(f"\n[Dispatch-Only] Card {i} send values:")
        print(f"  Shape: {host_send[i].shape}")
        for expert_j in range(num_experts):
            print(f"    Expert {expert_j}:")
            for t in range(min(2, COUNT)):
                vals = host_send[i][expert_j, t, :3].tolist()
                print(f"      Token {t}: {vals}")

    # Configure HCCL bootstrap
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

    # Create worker
    worker = Worker(
        level=3,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=device_ids,
        num_sub_workers=0,
        chip_bootstrap_configs=cfgs,
    )

    print(f"[Dispatch-Only] Compiling kernels for {platform}...", flush=True)
    dispatch_cc = build_dispatch_only_callable(platform)
    print("[Dispatch-Only] All kernels compiled successfully", flush=True)

    print("[Dispatch-Only] Initializing worker...", flush=True)
    worker.init()
    contexts = worker.chip_contexts
    print(f"[Dispatch-Only] Worker initialized with {len(contexts)} contexts", flush=True)

    try:
        def orch_fn(orch, _args, cfg):
            print(f"[Dispatch-Only] Submitting tasks for {num_cards} cards", flush=True)
            for i in range(num_cards):
                dispatch_args = TaskArgs()
                dispatch_args.add_tensor(make_tensor_arg(host_send[i]), TensorArgType.INPUT)
                dispatch_args.add_tensor(make_tensor_arg(host_recv[i]), TensorArgType.OUTPUT_EXISTING)
                dispatch_args.add_tensor(make_tensor_arg(host_output[i]), TensorArgType.OUTPUT_EXISTING)

                from simpler.task_interface import ContinuousTensor
                dispatch_args.add_tensor(
                    ContinuousTensor.make(
                        data=contexts[i].buffer_ptrs["scratch"],
                        shapes=(scratch_count,),
                        dtype=DataType.FLOAT32,
                        child_memory=True,
                    ),
                    TensorArgType.INOUT,
                )

                dispatch_args.add_scalar(i)  # expert_id
                dispatch_args.add_scalar(i)  # card_id
                dispatch_args.add_scalar(num_cards)
                dispatch_args.add_scalar(contexts[i].device_ctx)

                result = orch.submit_next_level(dispatch_cc, dispatch_args, cfg, worker=i)
                print(f"[Dispatch-Only] Submitted task for card {i}", flush=True)

        print("[Dispatch-Only] Running dispatch-only test...", flush=True)
        worker.run(orch_fn, args=None, config=CallConfig())
        print("[Dispatch-Only] Test completed", flush=True)

        # Compute golden recv using dispatch logic
        def compute_golden_recv(num_cards, host_send):
            """
            Compute golden recv using dispatch logic:
            For card i (processing expert i):
              recv[i][j][:COUNT][:] = card j's send[expert_i][:COUNT][:]
            NOTE: Dispatch only processes first COUNT tokens, not all NUM_TOKENS!
            """
            golden_recvs = []
            for cardi in range(num_cards):
                recv = torch.zeros(num_cards, NUM_TOKENS, HIDDEN_DIM, dtype=torch.float32)
                for cardj in range(num_cards):
                    # Card i receives from card j: card j's send[expert_i]
                    # expert_i = cardi (because card i processes expert i)
                    # Only copy first COUNT tokens!
                    recv[cardj, :COUNT, :] = host_send[cardj][cardi, :COUNT, :]
                golden_recvs.append(recv)
            return golden_recvs

        golden_recvs = compute_golden_recv(num_cards, host_send)

        # Verify correctness
        print("\n" + "="*80)
        print("[Dispatch-Only] VERIFICATION:")
        print("="*80)
        print("[Dispatch-Only] Comparing actual recv vs golden recv...")
        print(f"[Dispatch-Only] Recv shape: {host_recv[0].shape} (num_cards={num_cards}, NUM_TOKENS={NUM_TOKENS}, HIDDEN_DIM={HIDDEN_DIM})")

        all_match = True
        for i in range(num_cards):
            max_diff = float(torch.max(torch.abs(host_recv[i] - golden_recvs[i])))
            mean_diff = float(torch.mean(torch.abs(host_recv[i] - golden_recvs[i])))
            print(f"[Dispatch-Only] Card {i}: max |recv - golden| = {max_diff:.6e}, mean diff = {mean_diff:.6e}")

            if max_diff > 1e-3:
                all_match = False
                print(f"[Dispatch-Only] Card {i} MISMATCH! Full recv data:")
                for card_j in range(num_cards):
                    for t in range(NUM_TOKENS):
                        print(f"  recv[{card_j}][{t}][:3] = {host_recv[i][card_j, t, :3].tolist()}")
                        print(f"  golden[{card_j}][{t}][:3] = {golden_recvs[i][card_j, t, :3].tolist()}")
            else:
                print(f"[Dispatch-Only] Card {i} ✅ matches golden")

        if all_match:
            print("\n[Dispatch-Only] ✅ All cards matched golden!")
            return 0
        else:
            print("\n[Dispatch-Only] ❌ Some cards did NOT match golden!")
            return 1

    except Exception as e:
        print(f"[Dispatch-Only] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        print("[Dispatch-Only] Shutting down worker...")
        worker.close()
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
