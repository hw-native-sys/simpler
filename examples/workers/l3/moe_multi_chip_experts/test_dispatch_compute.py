#!/usr/bin/env python3
# Test dispatch + compute kernels together

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
    parser = argparse.ArgumentParser(description="Test dispatch + compute kernels")
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


def build_dispatch_compute_callable(platform: str) -> ChipCallable:
    """Build callable with dispatch + compute kernels."""
    print("[Dispatch+Compute] Compiling kernels...", flush=True)
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
    print("[Dispatch+Compute] Dispatch kernel compiled", flush=True)

    # Compile simple compute kernel
    compute_bytes = kc.compile_incore(
        source_path=os.path.join(HERE, "kernels/aiv/moe_simple_compute.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=include_dirs,
    )
    print("[Dispatch+Compute] Compute kernel compiled", flush=True)

    if not platform.endswith("sim"):
        from simpler_setup.elf_parser import extract_text_section
        dispatch_bytes = extract_text_section(dispatch_bytes)
        compute_bytes = extract_text_section(compute_bytes)
        print("[Dispatch+Compute] Text sections extracted", flush=True)

    # Compile orchestration
    print("[Dispatch+Compute] Compiling orchestration...", flush=True)
    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "kernels/orchestration/moe_dispatch_compute_orch.cpp"),
    )
    print("[Dispatch+Compute] Orchestration compiled", flush=True)

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

    return ChipCallable.build(
        signature=[
            ArgDirection.IN,   # send
            ArgDirection.OUT,  # recv
            ArgDirection.OUT,  # output (unused)
            ArgDirection.INOUT,  # scratch
            ArgDirection.IN,   # expert_id
            ArgDirection.IN,   # card_id
            ArgDirection.IN,   # num_cards
            ArgDirection.IN,   # CommContext*
        ],
        func_name="aicpu_orchestration_entry",
        binary=orch_bytes,
        children=[(0, dispatch_cc), (1, compute_cc)],  # Dispatch + Compute
    )


def run(platform: str, device_ids: list[int]) -> int:
    print(f"[Dispatch+Compute] Testing on devices {device_ids}", flush=True)
    num_cards = len(device_ids)
    num_experts = num_cards

    # Configure HCCL
    scratch_count = num_cards * num_cards * NUM_TOKENS * HIDDEN_DIM
    scratch_nbytes = scratch_count * 4
    total_scratch_nbytes = scratch_nbytes + num_cards * 4
    window_size = max(total_scratch_nbytes, 4 * 1024)

    rootinfo_path = f"/tmp/pto_dispatch_compute_{os.getpid()}.bin"
    try:
        os.unlink(rootinfo_path)
    except FileNotFoundError:
        pass

    torch.manual_seed(42)

    # Allocate tensors
    host_send = [torch.ones(num_experts, NUM_TOKENS, HIDDEN_DIM, dtype=torch.float32).share_memory_()
                 for _ in device_ids]
    host_recv = [torch.zeros(num_cards, NUM_TOKENS, HIDDEN_DIM, dtype=torch.float32).share_memory_()
                 for _ in device_ids]
    host_output = [torch.zeros(NUM_TOKENS, HIDDEN_DIM, dtype=torch.float32).share_memory_()
                   for _ in device_ids]

    print(f"[Dispatch+Compute] Allocated tensors: send=1.0, recv=0.0", flush=True)

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

    print(f"[Dispatch+Compute] Compiling kernels for {platform}...", flush=True)
    dispatch_compute_cc = build_dispatch_compute_callable(platform)
    print("[Dispatch+Compute] All kernels compiled successfully", flush=True)

    print("[Dispatch+Compute] Initializing worker...", flush=True)
    worker.init()
    contexts = worker.chip_contexts
    print(f"[Dispatch+Compute] Worker initialized with {len(contexts)} contexts", flush=True)

    try:
        def orch_fn(orch, _args, cfg):
            print(f"[Dispatch+Compute] Submitting tasks for {num_cards} cards", flush=True)
            for i in range(num_cards):
                args = TaskArgs()
                args.add_tensor(make_tensor_arg(host_send[i]), TensorArgType.INPUT)
                args.add_tensor(make_tensor_arg(host_recv[i]), TensorArgType.OUTPUT_EXISTING)
                args.add_tensor(make_tensor_arg(host_output[i]), TensorArgType.OUTPUT_EXISTING)

                from simpler.task_interface import ContinuousTensor
                args.add_tensor(
                    ContinuousTensor.make(
                        data=contexts[i].buffer_ptrs["scratch"],
                        shapes=(scratch_count,),
                        dtype=DataType.FLOAT32,
                        child_memory=True,
                    ),
                    TensorArgType.INOUT,
                )

                args.add_scalar(i)  # expert_id
                args.add_scalar(i)  # card_id
                args.add_scalar(num_cards)
                args.add_scalar(contexts[i].device_ctx)

                result = orch.submit_next_level(dispatch_compute_cc, args, cfg, worker=i)
                print(f"[Dispatch+Compute] Submitted task for card {i}", flush=True)

        print("[Dispatch+Compute] Running dispatch+compute test...", flush=True)
        worker.run(orch_fn, args=None, config=CallConfig())
        print("[Dispatch+Compute] Test completed", flush=True)

        # Print results
        print("\n" + "="*80)
        print("[Dispatch+Compute] RESULTS:")
        print("="*80)

        for i in range(num_cards):
            print(f"\n[Dispatch+Compute] Card {i} recv data (after dispatch+compute):")
            print(f"  Shape: {host_recv[i].shape}")
            print(f"  Expected: recv[i][:4][:] should be 2.0 (1.0 from dispatch + 1.0 from compute)")
            print(f"  Sample data (first 2 cards' data, first {COUNT} tokens, first 3 dims):")

            for card_j in range(num_cards):
                print(f"    recv[{card_j}][:3][:3] = [", end="")
                for t in range(min(3, COUNT)):
                    vals = host_recv[i][card_j, t, :3].tolist()
                    print(f"[{vals[0]:.1f},{vals[1]:.1f},{vals[2]:.1f}]", end="")
                    if t < min(3, COUNT) - 1:
                        print(", ", end="")
                print("]")

        # Verify correctness
        print("\n" + "="*80)
        print("[Dispatch+Compute] VERIFICATION:")
        print("="*80)

        all_correct = True
        for i in range(num_cards):
            for card_j in range(num_cards):
                for t in range(COUNT):
                    for d in range(HIDDEN_DIM):
                        expected = 2.0  # 1.0 (dispatch) + 1.0 (compute)
                        actual = host_recv[i][card_j, t, d].item()
                        if abs(actual - expected) > 1e-5:
                            print(f"[Dispatch+Compute] ERROR: Card {i} recv[{card_j}][{t}][{d}] = {actual}, expected {expected}")
                            all_correct = False

        if all_correct:
            print("[Dispatch+Compute] ✅ All values correct! Dispatch+Compute works perfectly.")
            return 0
        else:
            print("[Dispatch+Compute] ❌ Some values incorrect!")
            return 1

    except Exception as e:
        print(f"[Dispatch+Compute] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        print("[Dispatch+Compute] Shutting down worker...")
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
