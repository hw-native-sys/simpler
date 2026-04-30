#!/usr/bin/env python3
# Test combine kernel in isolation with unique integer values per token

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
    parser = argparse.ArgumentParser(description="Test combine kernel in isolation")
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


def build_combine_only_callable(platform: str) -> ChipCallable:
    """Build callable with ONLY combine kernel."""
    print("[Combine-Only] Compiling combine kernel...", flush=True)
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root(clone_protocol="https")
    include_dirs = kc.get_orchestration_include_dirs(runtime)
    kernel_include_dirs = list(include_dirs) + [str(kc.project_root / "src" / "common")]

    # Compile combine kernel
    combine_bytes = kc.compile_incore(
        source_path=os.path.join(HERE, "kernels/aiv/moe_combine_alltoall2.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=kernel_include_dirs,
    )
    print("[Combine-Only] Combine kernel compiled", flush=True)

    if not platform.endswith("sim"):
        from simpler_setup.elf_parser import extract_text_section
        combine_bytes = extract_text_section(combine_bytes)
        print("[Combine-Only] Text sections extracted", flush=True)

    # Compile orchestration
    print("[Combine-Only] Compiling orchestration...", flush=True)
    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "kernels/orchestration/moe_combine_only_orch.cpp"),
    )
    print("[Combine-Only] Orchestration compiled", flush=True)

    # Build core callable
    combine_cc = CoreCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT, ArgDirection.OUT,
                   ArgDirection.IN, ArgDirection.IN, ArgDirection.IN],
        binary=combine_bytes,
    )

    return ChipCallable.build(
        signature=[
            ArgDirection.IN,   # recv
            ArgDirection.OUT,  # output
            ArgDirection.INOUT,  # scratch
            ArgDirection.OUT,  # scratch_print
            ArgDirection.IN,   # card_id
            ArgDirection.IN,   # num_cards
            ArgDirection.IN,   # CommContext*
        ],
        func_name="aicpu_orchestration_entry",
        binary=orch_bytes,
        children=[(0, combine_cc)],  # Only combine child
    )


def compute_golden_output(num_cards: int, host_recv: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Compute golden output using direct store logic:
        output[cardi][expertj][:count][:] = recv[expertj, cardi, :count, :]

    For combine-only test:
    - Each card_j's recv[j] has shape [num_cards, NUM_TOKENS, HIDDEN_DIM]
    - recv[j][i][t][d] = expert_j's processed data for card_i
    - Card i's output[expert_j][:][:] stores expert_j's data for card_i
    """
    golden_outputs = []
    for cardi in range(num_cards):
        output = torch.zeros(num_cards, COUNT, HIDDEN_DIM, dtype=torch.float32)
        for expertj in range(num_cards):
            # recv[expertj][cardi][:][:] = expert_j's processed data for card_i
            # Store to output[expertj][:][:]
            output[expertj, :, :] = host_recv[expertj][cardi, :COUNT, :]
        golden_outputs.append(output)

    return golden_outputs


def initialize_recv_with_unique_integers(num_cards: int, device_id: int) -> torch.Tensor:
    """
    Initialize recv tensor with unique integers for each token.

    Direct store logic (no accumulation):
    - recv[expert_i][card_j][t][d] = expert_i processed data for card_j
    - output[card_j][expert_i][t][d] = recv[expert_i][card_j][t][d] (direct copy)

    Each position gets a unique value to trace data flow:
    value = (expert * 10000) + (card_j * 100) + (t * 10) + d

    This way we can identify which expert's data ended up where.
    """
    recv = torch.zeros(num_cards, NUM_TOKENS, HIDDEN_DIM, dtype=torch.float32).share_memory_()

    for expert_i in range(num_cards):
        for t in range(NUM_TOKENS):
            for d in range(HIDDEN_DIM):
                value = float(expert_i * 10000 + device_id * 100 + t * 10 + d)
                recv[expert_i, t, d] = value

    return recv


def run(platform: str, device_ids: list[int]) -> int:
    print(f"[Combine-Only] Testing combine on devices {device_ids}", flush=True)
    num_cards = len(device_ids)

    print(f"\n[Combine-Only] Test Configuration:")
    print(f"  Platform: {platform}")
    print(f"  Number of cards: {num_cards}")
    print(f"  Device IDs: {device_ids}")
    print(f"  NUM_TOKENS: {NUM_TOKENS}")
    print(f"  HIDDEN_DIM: {HIDDEN_DIM}")
    print(f"  COUNT (tokens processed): {COUNT}")
    print(f"  Total values per card: {num_cards * COUNT * HIDDEN_DIM}")
    print(f"  Total values to verify: {num_cards * num_cards * COUNT * HIDDEN_DIM}")

    # Configure HCCL
    scratch_count = num_cards * num_cards * NUM_TOKENS * HIDDEN_DIM
    scratch_nbytes = scratch_count * 4
    total_scratch_nbytes = scratch_nbytes + num_cards * 4
    window_size = max(total_scratch_nbytes, 4 * 1024)

    print(f"\n[Combine-Only] Memory Configuration:")
    print(f"  Scratch buffer size: {scratch_count} elements = {scratch_nbytes / 1024:.2f} KB")
    print(f"  Total with signals: {total_scratch_nbytes / 1024:.2f} KB")
    print(f"  HCCL window size: {window_size / 1024:.2f} KB")

    rootinfo_path = f"/tmp/pto_combine_only_{os.getpid()}.bin"
    try:
        os.unlink(rootinfo_path)
    except FileNotFoundError:
        pass

    torch.manual_seed(42)

    # Allocate tensors with unique integer values for each token
    host_recv = []
    for i in device_ids:
        recv = initialize_recv_with_unique_integers(num_cards, i)
        host_recv.append(recv)

    host_output = [torch.zeros(num_cards, COUNT, HIDDEN_DIM, dtype=torch.float32).share_memory_()
                   for _ in device_ids]

    # Allocate scratch_print tensors (debug output)
    host_scratch_print = [torch.zeros(scratch_count, dtype=torch.float32).share_memory_()
                          for _ in device_ids]

    # Compute golden output BEFORE running the kernel
    print("\n[Combine-Only] Computing golden output using golden.py logic...")
    golden_outputs = compute_golden_output(num_cards, host_recv)
    print("[Combine-Only] Golden output computed", flush=True)

    print(f"\n[Combine-Only] Allocated tensors: recv=unique_integers, output=0.0", flush=True)

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

    print(f"\n[Combine-Only] Compiling kernels for {platform}...", flush=True)
    combine_cc = build_combine_only_callable(platform)
    print("[Combine-Only] All kernels compiled successfully", flush=True)

    print("[Combine-Only] Initializing worker...", flush=True)
    worker.init()
    contexts = worker.chip_contexts
    print(f"[Combine-Only] Worker initialized with {len(contexts)} contexts", flush=True)

    try:
        def orch_fn(orch, _args, cfg):
            print(f"[Combine-Only] Submitting tasks for {num_cards} cards", flush=True)
            for i in range(num_cards):
                combine_args = TaskArgs()
                combine_args.add_tensor(make_tensor_arg(host_recv[i]), TensorArgType.INPUT)
                combine_args.add_tensor(make_tensor_arg(host_output[i]), TensorArgType.OUTPUT_EXISTING)

                from simpler.task_interface import ContinuousTensor
                combine_args.add_tensor(
                    ContinuousTensor.make(
                        data=contexts[i].buffer_ptrs["scratch"],
                        shapes=(scratch_count,),
                        dtype=DataType.FLOAT32,
                        child_memory=True,
                    ),
                    TensorArgType.INOUT,
                )
                combine_args.add_tensor(make_tensor_arg(host_scratch_print[i]), TensorArgType.OUTPUT_EXISTING)

                combine_args.add_scalar(i)  # card_id
                combine_args.add_scalar(num_cards)
                combine_args.add_scalar(contexts[i].device_ctx)

                result = orch.submit_next_level(combine_cc, combine_args, cfg, worker=i)
                print(f"[Combine-Only] Submitted task for card {i}", flush=True)

        print("[Combine-Only] Running combine-only test...", flush=True)

        # Print what each card will do
        print("\n[Combine-Only] Task breakdown:")
        for i in range(num_cards):
            print(f"  Card {i}: Will combine results from all experts for card {i}")
            print(f"    Input: recv[{i}][expert][{COUNT} tokens][{HIDDEN_DIM} dims]")
            print(f"    Output: output[num_experts={num_cards}][{COUNT} tokens][{HIDDEN_DIM} dims]")

        # Print output initial values BEFORE running kernel
        print("\n" + "="*80)
        print("[Combine-Only] OUTPUT INITIAL VALUES (before kernel):")
        print("="*80)
        for i in range(num_cards):
            print(f"\n[Combine-Only] Card {i} output initial values:")
            print(f"  Shape: {host_output[i].shape}")
            for expert_i in range(num_cards):
                print(f"    Expert {expert_i}:")
                for t in range(COUNT):
                    vals = host_output[i][expert_i, t, :].tolist()
                    print(f"      Token {t}: {vals}")

        worker.run(orch_fn, args=None, config=CallConfig())
        print("\n[Combine-Only] Test completed successfully!", flush=True)

        # Print scratch_print buffer contents for debugging
        print("\n" + "="*80)
        print("[Combine-Only] SCRATCH_PRINT BUFFER CONTENTS (Phase 1 stage-in mirror):")
        print("="*80)

        for i in range(num_cards):
            print(f"\n[Combine-Only] Card {i} scratch_print buffer (device {device_ids[i]}):")
            print(f"  Layout: scratch_print[expert_i][card_j][token][dim]")
            print(f"  Size: [{num_cards}][{num_cards}][{NUM_TOKENS}][{HIDDEN_DIM}]")

            for expert_i in range(num_cards):
                print(f"\n  Expert {expert_i}:")
                for card_j in range(num_cards):
                    print(f"    For card {card_j}:")
                    for t in range(COUNT):
                        offset = expert_i * num_cards * NUM_TOKENS * HIDDEN_DIM + card_j * NUM_TOKENS * HIDDEN_DIM + t * HIDDEN_DIM
                        vals = host_scratch_print[i][offset:offset+HIDDEN_DIM].tolist()
                        print(f"      Token {t}: {vals}")

        # Print results
        print("\n" + "="*80)
        print("[Combine-Only] INPUT RECV DATA:")
        print("="*80)

        for i in range(num_cards):
            print(f"\n[Combine-Only] Card {i} recv data (device {device_ids[i]}):")
            print(f"  Shape: {host_recv[i].shape}")
            for expert_i in range(num_cards):
                print(f"\n  Expert {expert_i}:")
                for t in range(NUM_TOKENS):
                    vals = host_recv[i][expert_i, t, :].tolist()
                    print(f"    Token {t}: {vals}")

        print("\n" + "="*80)
        print("[Combine-Only] OUTPUT DATA (after combine):")
        print("="*80)

        for i in range(num_cards):
            print(f"\n[Combine-Only] Card {i} output data:")
            print(f"  Shape: {host_output[i].shape}")
            for expert_i in range(num_cards):
                print(f"\n  Expert {expert_i}:")
                for t in range(COUNT):
                    vals = host_output[i][expert_i, t, :].tolist()
                    golden_vals = golden_outputs[i][expert_i, t, :].tolist()
                    print(f"\n    Token {t}:")
                    print(f"      Output:  {vals}")
                    print(f"      Golden:  {golden_vals}")
                    match = all(abs(v - g) < 1e-3 for v, g in zip(vals, golden_vals))
                    print(f"      Match: {'✓' if match else '✗'}")

        # Verify correctness by comparing with pre-computed golden output
        print("\n" + "="*80)
        print("[Combine-Only] VERIFICATION SUMMARY:")
        print("="*80)

        all_correct = True
        error_count = 0
        total_checked = 0

        for i in range(num_cards):
            print(f"\n[Combine-Only] Card {i}:")
            card_errors = 0

            for expert_i in range(num_cards):
                for t in range(COUNT):
                    for d in range(HIDDEN_DIM):
                        expected = golden_outputs[i][expert_i, t, d].item()
                        actual = host_output[i][expert_i, t, d].item()
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

        if all_correct:
            print("\n[Combine-Only] ✅ All values correct! Combine kernel works perfectly.")
            return 0
        else:
            print("\n[Combine-Only] ❌ Some values incorrect!")
            return 1

    except Exception as e:
        print(f"[Combine-Only] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        print("[Combine-Only] Shutting down worker...")
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
