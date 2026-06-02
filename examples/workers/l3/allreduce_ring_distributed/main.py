#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end distributed ring allreduce — chunked reduce-scatter + allgather.

Each rank owns private input/output tensors.  Cross-rank traffic follows a
logical ring (send to ``(rank+1) % P``, receive from ``(rank-1+P) % P``):

  Stage-in        partition input into P chunk slots in the HCCL window
  Reduce-scatter  (P-1) rounds — each rank ends with the fully reduced chunk
                  it owns (same golden as mesh ``allreduce_distributed``)
  Allgather       (P-1) rounds — propagate reduced chunks around the ring
  Stage-out       concatenate chunks → output

Compared to mesh ``allreduce_distributed/`` (O(P) full-vector remote reads
per rank), ring moves one chunk per round — bandwidth scales as
``2(P-1)/P × M`` for large ``M``.  P=4 is the minimum interesting width for
the schedule; P=2 is supported for regression.

Run:
    python examples/workers/l3/allreduce_ring_distributed/main.py -p a2a3sim -d 0-3

Compare mesh baseline on the same host:
    python examples/workers/l3/allreduce_distributed/main.py -p a2a3sim -d 0-3

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

# Must match ALLREDUCE_COUNT in kernels/aiv/allreduce_ring_kernel.cpp.
ALLREDUCE_COUNT = 256
K_MAX_SUPPORTED_RANKS = 16
CHUNK_MAX = ALLREDUCE_COUNT // 2  # largest chunk (P=2)
SCRATCH_FLOAT_ELEMS = (K_MAX_SUPPORTED_RANKS + 1) * CHUNK_MAX
SIGNAL_SLOTS = 2 * (K_MAX_SUPPORTED_RANKS - 1) * K_MAX_SUPPORTED_RANKS
SCRATCH_NBYTES = SCRATCH_FLOAT_ELEMS * 4 + SIGNAL_SLOTS * 4


def parse_device_range(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        ids = list(range(lo, hi + 1))
    else:
        ids = [int(spec)]
    if not (2 <= len(ids) <= K_MAX_SUPPORTED_RANKS):
        raise ValueError(
            f"allreduce_ring_distributed needs between 2 and {K_MAX_SUPPORTED_RANKS} devices, "
            f"got {len(ids)} ({ids})"
        )
    if ALLREDUCE_COUNT % len(ids) != 0:
        raise ValueError(
            f"ALLREDUCE_COUNT={ALLREDUCE_COUNT} must be divisible by nranks={len(ids)} for even chunking"
        )
    return ids


def build_chip_callable(platform: str, pto_isa_commit: str | None) -> ChipCallable:
    """Compile the ring allreduce AIV kernel + orchestration shim."""
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root(commit=pto_isa_commit, clone_protocol="https")
    include_dirs = kc.get_orchestration_include_dirs(runtime)
    kernel_include_dirs = list(include_dirs) + [str(kc.project_root / "src" / "common")]
    kernel_bytes = kc.compile_incore(
        source_path=os.path.join(HERE, "kernels/aiv/allreduce_ring_kernel.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=kernel_include_dirs,
    )
    if not platform.endswith("sim"):
        kernel_bytes = extract_text_section(kernel_bytes)

    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "kernels/orchestration/allreduce_ring_orch.cpp"),
    )
    core_callable = CoreCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT],
        binary=kernel_bytes,
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT],
        func_name="allreduce_ring_orchestration",
        config_name="allreduce_ring_orchestration_config",
        binary=orch_bytes,
        children=[(0, core_callable)],
    )


def expected_output(nranks: int) -> list[float]:
    """Same golden as mesh allreduce: output[i] = sum_r (i + r*100)."""
    return [float(nranks * i + 100 * nranks * (nranks - 1) // 2) for i in range(ALLREDUCE_COUNT)]


def run(
    device_ids: list[int],
    platform: str = "a2a3",
    pto_isa_commit: str | None = None,
) -> int:
    """Core logic — callable from both CLI and pytest."""
    nranks = len(device_ids)
    window_size = max(SCRATCH_NBYTES, 4 * 1024)

    print(f"[ring-allreduce] platform={platform} devices={device_ids} nranks={nranks}")

    host_inputs = [
        torch.tensor([i + rank * 100 for i in range(ALLREDUCE_COUNT)], dtype=torch.float32).share_memory_()
        for rank in range(nranks)
    ]
    host_outputs = [torch.zeros(ALLREDUCE_COUNT, dtype=torch.float32).share_memory_() for _ in range(nranks)]

    print("[ring-allreduce] compiling kernels...")
    chip_callable = build_chip_callable(platform, pto_isa_commit)

    worker = Worker(
        level=3,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=device_ids,
        num_sub_workers=0,
    )
    chip_cid = worker.register(chip_callable)

    try:
        print("[ring-allreduce] init worker (forks chip children; base comm is lazy)...")
        worker.init()

        def orch_fn(orch, _args, cfg):
            with orch.allocate_domain(
                name="default",
                workers=list(range(nranks)),
                window_size=window_size,
                buffers=[
                    CommBufferSpec(
                        name="scratch",
                        dtype="float32",
                        count=SCRATCH_FLOAT_ELEMS,
                        nbytes=SCRATCH_NBYTES,
                    )
                ],
            ) as handle:
                for i in range(nranks):
                    domain = handle[i]
                    print(
                        f"[ring-allreduce] chip {i}: rank={domain.domain_rank}/{domain.domain_size} "
                        f"window=[0x{domain.local_window_base:x} +{domain.actual_window_size}B] "
                        f"scratch=0x{domain.buffer_ptrs['scratch']:x}"
                    )
                    chip_args = TaskArgs()
                    chip_args.add_tensor(make_tensor_arg(host_inputs[i]), TensorArgType.INPUT)
                    chip_args.add_tensor(make_tensor_arg(host_outputs[i]), TensorArgType.OUTPUT_EXISTING)
                    chip_args.add_tensor(
                        ContinuousTensor.make(
                            data=domain.buffer_ptrs["scratch"],
                            shapes=(SCRATCH_FLOAT_ELEMS,),
                            dtype=DataType.FLOAT32,
                            child_memory=True,
                        ),
                        TensorArgType.INOUT,
                    )
                    chip_args.add_scalar(domain.domain_size)
                    chip_args.add_scalar(domain.device_ctx)
                    orch.submit_next_level(chip_cid, chip_args, cfg, worker=i)

        print(f"[ring-allreduce] running {nranks}-chip ring allreduce DAG...")
        worker.run(orch_fn, args=None, config=CallConfig())

        expected = torch.tensor(expected_output(nranks), dtype=torch.float32)
        ok = True
        for i in range(nranks):
            max_diff = float(torch.max(torch.abs(host_outputs[i] - expected)))
            print(f"[ring-allreduce] chip {i}: max |out - expected| = {max_diff:.3e}")
            if max_diff > 1e-3:
                ok = False
                for j in range(min(4, ALLREDUCE_COUNT)):
                    print(f"  output[{j}]={float(host_outputs[i][j])!r} expected={float(expected[j])!r}")

        if not ok:
            print("[ring-allreduce] golden check FAILED")
            return 1
        print("[ring-allreduce] all ranks matched golden ✅")
        return 0
    finally:
        worker.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-p", "--platform", default="a2a3", help="Platform backend, e.g. a2a3 or a2a3sim.")
    parser.add_argument(
        "-d",
        "--device",
        default="0-3",
        help="Device range, e.g. '0-3' (recommended) or '0-1'. 2 to 16 chips; COUNT must divide nranks.",
    )
    parser.add_argument("--pto-isa-commit", default=None, help="Optional PTO ISA commit/tag to fetch before compiling.")
    cli = parser.parse_args()

    return run(parse_device_range(cli.device), platform=cli.platform, pto_isa_commit=cli.pto_isa_commit)


if __name__ == "__main__":
    sys.exit(main())
