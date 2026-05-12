#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end distributed allreduce — symmetric 4-phase pattern.

Each rank owns a private input/output tensor; cross-rank communication happens
strictly inside the kernel, via a communication window scratch slot:

  Phase 1 stage-in      input → my scratch slot (HCCL window)
  Phase 2 device barrier signal matrix cross-rank sync via TNOTIFY/TWAIT
  Phase 3 compute       TLOAD every peer's scratch slot, TADD into acc
  Phase 4 stage-out     TSTORE acc → output

input / output are plain per-rank ``torch.share_memory_()`` tensors — the
parent writes inputs before ``init()`` and reads outputs after ``run()``, and
the framework's TaskArgs path handles H2D / D2H automatically (same as
``multi_chip_dispatch``).  Only ``scratch`` uses the chip bootstrap path
because window buffers can only exist after ``comm_alloc_windows`` has
run — that's what ``chip_bootstrap_configs`` is for.

Run:
    python examples/workers/l3/allreduce_distributed/main.py -p a2a3sim -d 0-1
"""

from __future__ import annotations

import argparse
import os
import sys

# Workaround for the duplicate-libomp abort when homebrew numpy and pip torch
# coexist in one macOS process. Harmless on Linux. Must be set before
# ``import torch``. See docs/troubleshooting/macos-libomp-collision.md.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402
from simpler.task_interface import (  # noqa: E402
    ArgDirection,
    CallConfig,
    ChipBootstrapConfig,
    ChipBufferSpec,
    ChipCallable,
    ChipCommBootstrapConfig,
    ChipContext,
    ContinuousTensor,
    CoreCallable,
    DataType,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import Worker  # noqa: E402

from simpler_setup.elf_parser import extract_text_section  # noqa: E402
from simpler_setup.kernel_compiler import KernelCompiler  # noqa: E402
from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: E402
from simpler_setup.torch_interop import make_tensor_arg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Must match ALLREDUCE_COUNT in kernels/aiv/allreduce_kernel.cpp.
ALLREDUCE_COUNT = 256
DTYPE_NBYTES = 4  # float32
BUFFER_NBYTES = ALLREDUCE_COUNT * DTYPE_NBYTES
# Signal tail: one int32 slot per rank, bounded by kMaxSupportedRanks in
# the kernel.  Signal area sits at the tail of the scratch buffer.
SIGNAL_TAIL_NBYTES = 16 * 4
SCRATCH_NBYTES = BUFFER_NBYTES + SIGNAL_TAIL_NBYTES


def parse_device_range(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        ids = list(range(lo, hi + 1))
    else:
        ids = [int(spec)]
    if len(ids) != 2:
        raise ValueError(f"allreduce_distributed needs exactly 2 devices, got {ids}")
    return ids


def build_chip_callable(platform: str, pto_isa_commit: str | None) -> ChipCallable:
    """Compile the AIV allreduce kernel + its C++ orchestration shim.

    The orchestration forwards three Tensor args (input / output / scratch)
    plus two scalars (nranks, CommContext pointer); the kernel reads
    ``Tensor->buffer.addr + start_offset`` to reach the device pointer.
    """
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root(commit=pto_isa_commit, clone_protocol="https")
    include_dirs = kc.get_orchestration_include_dirs(runtime)

    # The kernel resolves CommContext from "platform_comm/comm_context.h",
    # which lives under src/common/. Add that directory on top of the runtime
    # include set so the kernel compile can see it.
    kernel_include_dirs = list(include_dirs) + [str(kc.project_root / "src" / "common")]
    kernel_bytes = kc.compile_incore(
        source_path=os.path.join(HERE, "kernels/aiv/allreduce_kernel.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=kernel_include_dirs,
    )
    if not platform.endswith("sim"):
        kernel_bytes = extract_text_section(kernel_bytes)

    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "kernels/orchestration/allreduce_orch.cpp"),
    )
    core_callable = CoreCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT],
        binary=kernel_bytes,
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT],
        func_name="allreduce_orchestration",
        config_name="allreduce_orchestration_config",
        binary=orch_bytes,
        children=[(0, core_callable)],
    )


def expected_output(nranks: int) -> list[float]:
    """output[i] = sum_r (i + r*100) = nranks*i + 100 * nranks*(nranks-1)/2."""
    return [float(nranks * i + 100 * nranks * (nranks - 1) // 2) for i in range(ALLREDUCE_COUNT)]


def run(
    device_ids: list[int],
    platform: str = "a2a3",
    pto_isa_commit: str | None = None,
    build: bool = False,
) -> int:
    """Core logic — callable from both CLI and pytest."""
    nranks = len(device_ids)
    # Backends may round up; only needs to hold SCRATCH_NBYTES.  A 4 KB floor
    # keeps us clear of minimum-window-size quirks.
    window_size = max(SCRATCH_NBYTES, 4 * 1024)

    rootinfo_path = f"/tmp/pto_allreduce_distributed_rootinfo_{os.getpid()}.bin"
    try:
        os.unlink(rootinfo_path)
    except FileNotFoundError:
        pass

    print(f"[allreduce] platform={platform} devices={device_ids} nranks={nranks}")

    # --- Per-rank host tensors (input/output) via torch.share_memory_().
    # share_memory_() moves the storage into an mmap region that forked
    # children see at the same virtual address, so ``chip_args.add_tensor``
    # with TensorArgType.INPUT / OUTPUT_EXISTING can hand the kernel a host
    # pointer and the framework handles H2D/D2H transparently — no manual
    # SharedMemory / HostBufferStaging plumbing.
    host_inputs = [
        torch.tensor([i + rank * 100 for i in range(ALLREDUCE_COUNT)], dtype=torch.float32).share_memory_()
        for rank in range(nranks)
    ]
    host_outputs = [torch.zeros(ALLREDUCE_COUNT, dtype=torch.float32).share_memory_() for _ in range(nranks)]

    # --- Scratch bootstrap: one window buffer per chip.  The window exists
    # only after comm_alloc_windows has run, so allocating scratch
    # during bootstrap is the natural lifecycle — no TaskArgs equivalent
    # covers it.  No host staging is needed for scratch.
    cfgs = [
        ChipBootstrapConfig(
            comm=ChipCommBootstrapConfig(
                rank=rank,
                nranks=nranks,
                rootinfo_path=rootinfo_path,
                window_size=window_size,
            ),
            buffers=[
                ChipBufferSpec(
                    name="scratch",
                    dtype="float32",
                    count=ALLREDUCE_COUNT,
                    nbytes=SCRATCH_NBYTES,
                ),
            ],
        )
        for rank in range(nranks)
    ]

    print("[allreduce] compiling kernels...")
    chip_callable = build_chip_callable(platform, pto_isa_commit)

    worker = Worker(
        level=3,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=device_ids,
        num_sub_workers=0,
        chip_bootstrap_configs=cfgs,
        build=build,
    )
    chip_cid = worker.register(chip_callable)

    try:
        print("[allreduce] init worker (forks chip children + bootstraps comm backend)...")
        worker.init()

        contexts: list[ChipContext] = worker.chip_contexts
        assert len(contexts) == nranks
        for i, ctx in enumerate(contexts):
            print(
                f"[allreduce] chip {i}: device={ctx.device_id} rank={ctx.rank}/{ctx.nranks} "
                f"window=[0x{ctx.local_window_base:x} +{ctx.actual_window_size}B] "
                f"scratch=0x{ctx.buffer_ptrs['scratch']:x}"
            )

        def orch_fn(orch, _args, cfg):
            for i, ctx in enumerate(contexts):
                chip_args = TaskArgs()
                chip_args.add_tensor(make_tensor_arg(host_inputs[i]), TensorArgType.INPUT)
                chip_args.add_tensor(make_tensor_arg(host_outputs[i]), TensorArgType.OUTPUT_EXISTING)
                # Scratch is a device pointer into the HCCL window — not a
                # host tensor — so wrap it manually with child_memory=True
                # to skip the runtime's H2D path.
                chip_args.add_tensor(
                    ContinuousTensor.make(
                        data=ctx.buffer_ptrs["scratch"],
                        shapes=(ALLREDUCE_COUNT,),
                        dtype=DataType.FLOAT32,
                        child_memory=True,
                    ),
                    TensorArgType.INOUT,
                )
                chip_args.add_scalar(ctx.nranks)
                chip_args.add_scalar(ctx.device_ctx)
                orch.submit_next_level(chip_cid, chip_args, cfg, worker=i)

        print("[allreduce] running 2-chip allreduce DAG...")
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
            print("[allreduce] golden check FAILED")
            return 1
        print("[allreduce] all ranks matched golden ✅")
        return 0
    finally:
        worker.close()
        try:
            os.unlink(rootinfo_path)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--platform", default="a2a3", help="Platform backend, e.g. a2a3 or a2a3sim.")
    parser.add_argument("-d", "--device", default="0-1", help="Device range, e.g. '0-1'. Two chips required.")
    parser.add_argument(
        "--build", action="store_true", help="Rebuild runtime from source instead of using cached libs."
    )
    parser.add_argument("--pto-isa-commit", default=None, help="Optional PTO ISA commit/tag to fetch before compiling.")
    cli = parser.parse_args()
    return run(
        parse_device_range(cli.device), platform=cli.platform, pto_isa_commit=cli.pto_isa_commit, build=cli.build
    )


if __name__ == "__main__":
    sys.exit(main())
