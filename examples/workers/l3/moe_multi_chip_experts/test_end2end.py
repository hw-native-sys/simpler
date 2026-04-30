#!/usr/bin/env python3
# Test complete MoE pipeline: Dispatch + Compute + Combine

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
    parser = argparse.ArgumentParser(description="Test complete MoE pipeline (Dispatch + Compute + Combine)")
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


def build_end2end_callable(platform: str) -> ChipCallable:
    """Build callable with dispatch + compute + combine kernels."""
    print("[End2End] Compiling kernels...", flush=True)
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
    print("[End2End] Dispatch kernel compiled", flush=True)

    # Compile compute kernel
    compute_bytes = kc.compile_incore(
        source_path=os.path.join(HERE, "kernels/aiv/moe_simple_compute.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=include_dirs,
    )
    print("[End2End] Compute kernel compiled", flush=True)

    # Compile combine kernel
    combine_bytes = kc.compile_incore(
        source_path=os.path.join(HERE, "kernels/aiv/moe_combine_alltoall2.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=kernel_include_dirs,
    )
    print("[End2End] Combine kernel compiled", flush=True)

    if not platform.endswith("sim"):
        from simpler_setup.elf_parser import extract_text_section
        dispatch_bytes = extract_text_section(dispatch_bytes)
        compute_bytes = extract_text_section(compute_bytes)
        combine_bytes = extract_text_section(combine_bytes)
        print("[End2End] Text sections extracted", flush=True)

    # Compile orchestration
    print("[End2End] Compiling orchestration...", flush=True)
    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "kernels/orchestration/moe_end2end_orch.cpp"),
    )
    print("[End2End] Orchestration compiled", flush=True)

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
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT, ArgDirection.OUT,
                   ArgDirection.IN, ArgDirection.IN, ArgDirection.IN],
        binary=combine_bytes,
    )

    return ChipCallable.build(
        signature=[
            ArgDirection.IN,   # send
            ArgDirection.OUT,  # recv
            ArgDirection.OUT,  # output
            ArgDirection.INOUT,  # scratch
            ArgDirection.INOUT,  # scratch_test
            ArgDirection.OUT,  # scratch_print
            ArgDirection.IN,   # expert_id
            ArgDirection.IN,   # card_id
            ArgDirection.IN,   # num_cards
            ArgDirection.IN,   # CommContext*
        ],
        func_name="aicpu_orchestration_entry",
        binary=orch_bytes,
        children=[(0, dispatch_cc), (1, compute_cc), (2, combine_cc)],  # All three phases
    )


def compute_golden_end2end(num_cards: int, host_send: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Compute golden output for end-to-end pipeline:
    1. Dispatch: send[card_j][expert_i][:COUNT][:] -> recv[card_i][card_j][:COUNT][:]
    2. Compute: recv[card_i][card_j][:COUNT][:] += 1.0
    3. Combine: recv[expert_j][card_i][:COUNT][:] -> output[card_i][expert_j][:COUNT][:]

    Send initialization: unique values using (card * 1000000 + expert * 10000 + token * 100 + dim)
    """
    golden_outputs = []
    for cardi in range(num_cards):
        output = torch.zeros(num_cards, COUNT, HIDDEN_DIM, dtype=torch.float32)
        for expertj in range(num_cards):
            for t in range(COUNT):
                for d in range(HIDDEN_DIM):
                    # After dispatch: recv[cardi][expertj][:][:] = send[expertj][cardi][:][:]
                    # Value from cardi's send[expertj][cardi][t][d]
                    send_value = host_send[cardi][expertj, t, d].item()
                    # After compute: recv += 1.0
                    recv_value = send_value + 1.0
                    # After combine: output[cardi][expertj][t][d] = recv[expertj][cardi][t][d]
                    output[expertj, t, d] = recv_value
        golden_outputs.append(output)

    return golden_outputs


def run(platform: str, device_ids: list[int]) -> int:
    print(f"[End2End] Testing complete MoE pipeline on devices {device_ids}", flush=True)
    num_cards = len(device_ids)
    num_experts = num_cards

    # Configure HCCL
    scratch_count = num_cards * num_cards * NUM_TOKENS * HIDDEN_DIM
    scratch_nbytes = scratch_count * 4
    total_scratch_nbytes = scratch_nbytes + num_cards * 4
    window_size = max(total_scratch_nbytes, 4 * 1024)

    print(f"\n[End2End] Test Configuration:")
    print(f"  Platform: {platform}")
    print(f"  Number of cards: {num_cards}")
    print(f"  Device IDs: {device_ids}")
    print(f"  NUM_TOKENS: {NUM_TOKENS}")
    print(f"  HIDDEN_DIM: {HIDDEN_DIM}")
    print(f"  COUNT (tokens processed): {COUNT}")

    rootinfo_path = f"/tmp/pto_end2end_{os.getpid()}.bin"
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
    host_output = [torch.zeros(num_cards, COUNT, HIDDEN_DIM, dtype=torch.float32).share_memory_()
                   for _ in device_ids]

    # Allocate scratch_print tensor (debug output)
    host_scratch_print = [torch.zeros(scratch_count, dtype=torch.float32).share_memory_()
                          for _ in device_ids]

    print(f"\n[End2End] Allocated tensors:")
    print(f"  send=unique_values, recv=0.0, output=0.0")
    print(f"  Value encoding: (card_id * 1000000) + (expert_id * 10000) + (token * 100) + dim", flush=True)

    # Compute golden output
    print("\n[End2End] Computing golden output...")
    golden_outputs = compute_golden_end2end(num_cards, host_send)
    print("[End2End] Golden output computed", flush=True)

    # Configure HCCL bootstrap with two independent scratch buffers
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
                ChipBufferSpec(
                    name="scratch_test",
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

    print(f"\n[End2End] Compiling kernels for {platform}...", flush=True)
    end2end_cc = build_end2end_callable(platform)
    print("[End2End] All kernels compiled successfully", flush=True)

    print("[End2End] Initializing worker...", flush=True)
    worker.init()
    contexts = worker.chip_contexts
    print(f"[End2End] Worker initialized with {len(contexts)} contexts", flush=True)

    try:
        def orch_fn(orch, _args, cfg):
            print(f"[End2End] Submitting tasks for {num_cards} cards", flush=True)
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
                args.add_tensor(
                    ContinuousTensor.make(
                        data=contexts[i].buffer_ptrs["scratch_test"],
                        shapes=(scratch_count,),
                        dtype=DataType.FLOAT32,
                        child_memory=True,
                    ),
                    TensorArgType.INOUT,
                )
                args.add_tensor(make_tensor_arg(host_scratch_print[i]), TensorArgType.OUTPUT_EXISTING)

                args.add_scalar(i)  # expert_id
                args.add_scalar(i)  # card_id
                args.add_scalar(num_cards)
                args.add_scalar(contexts[i].device_ctx)

                result = orch.submit_next_level(end2end_cc, args, cfg, worker=i)
                print(f"[End2End] Submitted task for card {i}", flush=True)

        print("\n[End2End] Running end-to-end test...", flush=True)

        worker.run(orch_fn, args=None, config=CallConfig())
        print("\n[End2End] End-to-end pipeline completed!", flush=True)

        # Print results
        print("\n" + "="*80)
        print("[End2End] OUTPUT DATA:")
        print("="*80)

        for i in range(num_cards):
            print(f"\n[End2End] Card {i} output data:")
            print(f"  Expected: Each value = send_value + 1.0")
            print(f"  Sample data (first 2 experts, first {COUNT} tokens, first 3 dims):")

            for expert_j in range(min(2, num_cards)):
                print(f"    Expert {expert_j}:")
                for t in range(min(COUNT, 2)):
                    vals = host_output[i][expert_j, t, :3].tolist()
                    golden_vals = golden_outputs[i][expert_j, t, :3].tolist()
                    print(f"      Token {t}: Output={vals}, Golden={golden_vals}")

        # Verify correctness
        print("\n" + "="*80)
        print("[End2End] VERIFICATION:")
        print("="*80)

        all_correct = True
        error_count = 0
        total_checked = 0

        for i in range(num_cards):
            print(f"\n[End2End] Card {i}:")
            card_errors = 0

            for expert_j in range(num_cards):
                for t in range(COUNT):
                    for d in range(HIDDEN_DIM):
                        expected = golden_outputs[i][expert_j, t, d].item()
                        actual = host_output[i][expert_j, t, d].item()
                        total_checked += 1

                        if abs(actual - expected) > 1e-3:
                            card_errors += 1
                            error_count += 1
                            all_correct = False

            if card_errors == 0:
                print(f"  ✓ All {num_cards * COUNT * HIDDEN_DIM} values correct")
            else:
                print(f"  ✗ {card_errors} / {num_cards * COUNT * HIDDEN_DIM} values incorrect")

        print(f"\n  Total: {total_checked - error_count}/{total_checked} correct")

        # Final verdict
        print("\n" + "="*80)
        print("[End2End] FINAL VERDICT:")
        print("="*80)

        if all_correct:
            print("\n[End2End] ✅ All values correct! End-to-end pipeline works perfectly.")
            return 0
        else:
            print("\n[End2End] ❌ Some values incorrect!")
            return 1

    except Exception as e:
        print(f"[End2End] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        print("[End2End] Shutting down worker...")
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
