#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end distributed allreduce with selectable algorithm variant.

Four algorithm modes are available via --mode:

  onephase   (default) Mesh direct: read full vector from all peers, accumulate.
             Best for small P (2-4), simplest implementation.

  twophase   Mesh RS+AG: reduce-scatter then allgather with mesh barriers.
             Best for medium P, bandwidth-efficient (2*(P-1) chunks vs P-1 vectors).

  ring       Ring RS+AG: chunked reduce-scatter + allgather with neighbor barriers.
             Best for large P (8+), bandwidth-optimal (2*(P-1)/P * N).

  bidirectional_ring  Bidirectional interleaved ring RS+AG implementing the IBing
             algorithm (Zong et al., ACM TACO 2025).  Communication steps halved
             from 2(P-1) to P-1 by sending/receiving in both HCCS directions
             simultaneously.  Best for small-N (α-dominated) workloads.

Each rank owns a private input/output tensor; cross-rank communication happens
strictly inside the kernel, via a communication window scratch slot.

Run examples:
    python main.py -p a2a3sim -d 0-1 --mode onephase
    python main.py -p a2a3sim -d 0-3 --mode twophase
    python main.py -p a2a3sim -d 0-3 --mode ring
    python main.py -p a2a3sim -d 0-3 --mode bidirectional_ring

"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402
from simpler.task_interface import (  # noqa: E402
    ArgDirection,
    CallConfig,
    ChipCallable,
    CommBufferSpec,
    CoreCallable,
    DataType,
    TaskArgs,
    Tensor,
    TensorArgType,
)
from simpler.worker import Worker  # noqa: E402

from simpler_setup.elf_parser import extract_text_section  # noqa: E402
from simpler_setup.kernel_compiler import KernelCompiler  # noqa: E402
from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: E402
from simpler_setup.torch_interop import make_tensor_arg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

ALLREDUCE_COUNT = 256
DTYPE_NBYTES = 4  # float32
K_MAX_SUPPORTED_RANKS = 16

# Kernel and orchestration file mappings per mode.
KERNEL_MAP = {
    "onephase": "kernels/aiv/allreduce_onephase_kernel.cpp",
    "twophase": "kernels/aiv/allreduce_twophase_kernel.cpp",
    "ring": "kernels/aiv/allreduce_ring_kernel.cpp",
    "bidirectional_ring": "kernels/aiv/allreduce_bidirectional_ring_kernel.cpp",
}
ORCH_MAP = {
    "onephase": (
        "kernels/orchestration/allreduce_onephase_orch.cpp",
        "allreduce_orchestration",
        "allreduce_orchestration_config",
    ),
    "twophase": (
        "kernels/orchestration/allreduce_twophase_orch.cpp",
        "allreduce_twophase_orchestration",
        "allreduce_twophase_orchestration_config",
    ),
    "ring": (
        "kernels/orchestration/allreduce_ring_orch.cpp",
        "allreduce_ring_orchestration",
        "allreduce_ring_orchestration_config",
    ),
    "bidirectional_ring": (
        "kernels/orchestration/allreduce_bidirectional_ring_orch.cpp",
        "allreduce_bidirectional_ring_orchestration",
        "allreduce_bidirectional_ring_orchestration_config",
    ),
}


def compute_scratch_params(mode: str, nranks: int) -> tuple[int, int, int]:
    """Compute scratch buffer parameters for the given mode and rank count.

    Returns:
        (float_elems, scratch_nbytes, window_size)
    """
    if mode == "onephase":
        # One-phase: ALLREDUCE_COUNT floats + signal tail (16 int32 slots).
        float_elems = ALLREDUCE_COUNT
        signal_tail_nbytes = K_MAX_SUPPORTED_RANKS * DTYPE_NBYTES
        scratch_nbytes = float_elems * DTYPE_NBYTES + signal_tail_nbytes
    elif mode == "twophase":
        # Two-phase: nranks * chunk_elems floats + 2 signal rows (RS + AG barriers).
        chunk_elems = ALLREDUCE_COUNT // nranks
        float_elems = nranks * chunk_elems
        signal_tail_nbytes = 2 * K_MAX_SUPPORTED_RANKS * DTYPE_NBYTES
        scratch_nbytes = float_elems * DTYPE_NBYTES + signal_tail_nbytes
    elif mode == "ring":
        # Ring: (nranks+1) * chunk_elems floats (P chunks + 1 exchange) + 2*(P-1) signal rows.
        chunk_elems = ALLREDUCE_COUNT // nranks
        float_elems = (nranks + 1) * chunk_elems
        signal_tail_nbytes = 2 * (nranks - 1) * K_MAX_SUPPORTED_RANKS * DTYPE_NBYTES
        scratch_nbytes = float_elems * DTYPE_NBYTES + signal_tail_nbytes
    elif mode == "bidirectional_ring":
        # Two-ring design: ALLREDUCE_COUNT floats + (2*(P-1)+1) signal rows.
        # The +1 accounts for the final sync barrier between AG and stage-out.
        subchunk_elems = ALLREDUCE_COUNT // (2 * nranks)
        float_elems = 2 * nranks * subchunk_elems  # = ALLREDUCE_COUNT
        signal_tail_nbytes = (2 * (nranks - 1) + 1) * K_MAX_SUPPORTED_RANKS * DTYPE_NBYTES
        scratch_nbytes = float_elems * DTYPE_NBYTES + signal_tail_nbytes
    else:
        raise ValueError(f"Unsupported allreduce mode: {mode!r}. Expected one of {tuple(KERNEL_MAP.keys())}.")

    window_size = max(scratch_nbytes, 4 * 1024)
    return float_elems, scratch_nbytes, window_size


def parse_device_range(spec: str) -> list[int]:
    """Parse a device range string like ``0-1`` or a single device id."""
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        ids = list(range(lo, hi + 1))
    else:
        ids = [int(spec)]
    if not (2 <= len(ids) <= K_MAX_SUPPORTED_RANKS):
        raise ValueError(f"allreduce needs between 2 and {K_MAX_SUPPORTED_RANKS} devices, got {len(ids)} ({ids})")
    return ids


def build_chip_callable(platform: str, mode: str, pto_isa_commit: str | None) -> ChipCallable:
    """Compile the selected allreduce kernel + orchestration shim."""
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root(commit=pto_isa_commit, clone_protocol="https")
    include_dirs = kc.get_orchestration_include_dirs(runtime)

    # The kernel resolves CommContext from "platform_comm/comm_context.h",
    # which lives under src/common/. Add that directory on top of the runtime
    # include set so the kernel compile can see it.
    kernel_include_dirs = list(include_dirs) + [str(kc.project_root / "src" / "common")]
    kernel_bytes = kc.compile_incore(
        source_path=os.path.join(HERE, KERNEL_MAP[mode]),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=kernel_include_dirs,
    )
    if not platform.endswith("sim"):
        kernel_bytes = extract_text_section(kernel_bytes)

    orch_path, func_name, config_name = ORCH_MAP[mode]
    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, orch_path),
    )
    core_callable = CoreCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT],
        arg_index=[0, 1, 2],
        binary=kernel_bytes,
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT],
        func_name=func_name,
        config_name=config_name,
        binary=orch_bytes,
        children=[(0, core_callable)],
    )


def expected_output(nranks: int) -> list[float]:
    """output[i] = sum_r (i + r*100) = nranks*i + 100 * nranks*(nranks-1)/2."""
    return [float(nranks * i + 100 * nranks * (nranks - 1) // 2) for i in range(ALLREDUCE_COUNT)]


def run(
    device_ids: list[int],
    platform: str = "a2a3",
    mode: str = "onephase",
    pto_isa_commit: str | None = None,
) -> int:
    """Core logic — callable from both CLI and pytest."""
    nranks = len(device_ids)
    if mode not in KERNEL_MAP:
        raise ValueError(f"Unsupported allreduce mode: {mode!r}. Expected one of {tuple(KERNEL_MAP.keys())}.")
    if not (2 <= nranks <= K_MAX_SUPPORTED_RANKS):
        raise ValueError(f"allreduce needs between 2 and {K_MAX_SUPPORTED_RANKS} devices, got {nranks} ({device_ids})")

    # Ring requires ALLREDUCE_COUNT divisible by nranks; bidirectional_ring
    # (two-ring design) requires ALLREDUCE_COUNT divisible by 2*nranks.
    if mode in ("twophase", "ring") and ALLREDUCE_COUNT % nranks != 0:
        raise ValueError(f"ALLREDUCE_COUNT={ALLREDUCE_COUNT} must be divisible by nranks={nranks} for {mode} mode")
    if mode == "bidirectional_ring" and ALLREDUCE_COUNT % (2 * nranks) != 0:
        raise ValueError(
            f"ALLREDUCE_COUNT={ALLREDUCE_COUNT} must be divisible by 2*nranks={2 * nranks} for {mode} mode"
        )

    float_elems, scratch_nbytes, window_size = compute_scratch_params(mode, nranks)

    print(f"[allreduce] mode={mode} platform={platform} devices={device_ids} nranks={nranks}")

    host_inputs = [
        torch.tensor([i + rank * 100 for i in range(ALLREDUCE_COUNT)], dtype=torch.float32).share_memory_()
        for rank in range(nranks)
    ]
    host_outputs = [torch.zeros(ALLREDUCE_COUNT, dtype=torch.float32).share_memory_() for _ in range(nranks)]

    print(f"[allreduce] compiling {mode} kernel...")
    chip_callable = build_chip_callable(platform, mode, pto_isa_commit)

    worker = Worker(
        level=3,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=device_ids,
        num_sub_workers=0,
    )
    chip_handle = worker.register(chip_callable)

    try:
        print("[allreduce] init worker (forks chip children; base comm is lazy)...")
        worker.init()

        def orch_fn(orch, _args, cfg):
            with orch.allocate_domain(
                name="default",
                workers=list(range(nranks)),
                window_size=window_size,
                buffers=[CommBufferSpec(name="scratch", dtype="float32", count=float_elems, nbytes=scratch_nbytes)],
            ) as handle:
                for i in range(nranks):
                    domain = handle[i]
                    print(
                        f"[allreduce] chip {i}: rank={domain.domain_rank}/{domain.domain_size} "
                        f"window=[0x{domain.local_window_base:x} +{domain.actual_window_size}B] "
                        f"scratch=0x{domain.buffer_ptrs['scratch']:x}"
                    )
                    chip_args = TaskArgs()
                    chip_args.add_tensor(make_tensor_arg(host_inputs[i]), TensorArgType.INPUT)
                    chip_args.add_tensor(make_tensor_arg(host_outputs[i]), TensorArgType.OUTPUT_EXISTING)
                    chip_args.add_tensor(
                        Tensor.make(
                            data=domain.buffer_ptrs["scratch"],
                            shapes=(float_elems,),
                            dtype=DataType.FLOAT32,
                            child_memory=True,
                        ),
                        TensorArgType.INOUT,
                    )
                    chip_args.add_scalar(domain.domain_size)
                    chip_args.add_scalar(domain.device_ctx)
                    orch.submit_next_level(chip_handle, chip_args, cfg, worker=i)

        print(f"[allreduce] running {nranks}-chip allreduce DAG...")
        worker.run(orch_fn, args=None, config=CallConfig())

        expected = torch.tensor(expected_output(nranks), dtype=torch.float32)
        ok = True
        for i in range(nranks):
            max_diff = float(torch.max(torch.abs(host_outputs[i] - expected)))
            print(f"[allreduce] chip {i}: max |out - expected| = {max_diff:.3e}")
            if max_diff > 1e-3:
                ok = False
                for j in range(min(4, ALLREDUCE_COUNT)):
                    print(f"  output[{j}]={float(host_outputs[i][j])!r} expected={float(expected[j])!r}")

        if not ok:
            print(f"[allreduce] {mode} golden check FAILED")
            return 1
        print(f"[allreduce] {mode} all ranks matched golden ✅")
        return 0
    finally:
        worker.close()


def main() -> int:
    """CLI entry point for distributed allreduce with selectable ``--mode``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-p", "--platform", default="a2a3", help="Platform backend, e.g. a2a3 or a2a3sim.")
    parser.add_argument(
        "-d", "--device", default="0-1", help="Device range, e.g. '0-1' or '0-3'. 2 to 16 chips required."
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["onephase", "twophase", "ring", "bidirectional_ring"],
        default="onephase",
        help=(
            "Allreduce algorithm variant: onephase (mesh direct), twophase (mesh RS+AG), "
            "ring (ring RS+AG), bidirectional_ring (interleaved bidirectional ring RS+AG)."
        ),
    )
    parser.add_argument("--pto-isa-commit", default=None, help="Optional PTO ISA commit/tag to fetch before compiling.")
    cli = parser.parse_args()

    return run(parse_device_range(cli.device), platform=cli.platform, mode=cli.mode, pto_isa_commit=cli.pto_isa_commit)


if __name__ == "__main__":
    sys.exit(main())
