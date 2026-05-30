#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end distributed reduce-scatter — symmetric 4-phase pattern.

Each rank owns a private input of nranks*COUNT_PER_RANK floats (N equal-sized
chunks).  After reduce-scatter, rank r holds the element-wise sum of chunk r
from every rank:

  Phase 1 stage-in      all N input chunks → scratch slots in HCCL window
  Phase 2 device barrier signal matrix cross-rank sync via TNOTIFY/TWAIT
  Phase 3 reduce        acc = my scratch[my_rank*C]; acc += peer's scratch[my_rank*C] for each peer
  Phase 4 stage-out     TSTORE acc → output

Run:
    python examples/workers/l3/reduce_scatter_distributed/main.py -p a2a3sim -d 0-1

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

# Must match COUNT_PER_RANK in kernels/aiv/reduce_scatter_kernel.cpp.
COUNT_PER_RANK = 64
DTYPE_NBYTES = 4  # float32
# Signal tail: one int32 slot per rank, bounded by kMaxSupportedRanks.
SIGNAL_TAIL_NBYTES = 16 * 4  # 64 B
# NOTE: the full scratch size depends on nranks (staging area = nranks * COUNT_PER_RANK floats).
# It is computed inside run() once nranks is known.


def parse_device_range(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        ids = list(range(lo, hi + 1))
    else:
        ids = [int(spec)]
    if not (2 <= len(ids) <= 16):
        raise ValueError(f"reduce_scatter_distributed needs between 2 and 16 devices, got {len(ids)} ({ids})")
    return ids


def build_chip_callable(platform: str, pto_isa_commit: str | None) -> ChipCallable:
    """Compile the AIV reduce-scatter kernel + its C++ orchestration shim."""
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root(commit=pto_isa_commit, clone_protocol="https")
    include_dirs = kc.get_orchestration_include_dirs(runtime)

    # src/common  — for platform_comm/comm_context.h
    kernel_include_dirs = list(include_dirs) + [
        str(kc.project_root / "src" / "common"),
    ]
    kernel_bytes = kc.compile_incore(
        source_path=os.path.join(HERE, "kernels/aiv/reduce_scatter_kernel.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=kernel_include_dirs,
    )
    if not platform.endswith("sim"):
        kernel_bytes = extract_text_section(kernel_bytes)

    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "kernels/orchestration/reduce_scatter_orch.cpp"),
    )
    core_callable = CoreCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT],
        binary=kernel_bytes,
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT],
        func_name="reduce_scatter_orchestration",
        config_name="reduce_scatter_orchestration_config",
        binary=orch_bytes,
        children=[(0, core_callable)],
    )


def expected_output(nranks: int, dest: int) -> list[float]:
    """output[j] = sum_r input_r[dest*C + j] = nranks*(dest*C+j) + 100*nranks*(nranks-1)/2."""
    return [
        float(nranks * (dest * COUNT_PER_RANK + j) + 100 * nranks * (nranks - 1) // 2) for j in range(COUNT_PER_RANK)
    ]


def run(
    device_ids: list[int],
    platform: str = "a2a3",
    pto_isa_commit: str | None = None,
) -> int:
    """Core logic — callable from both CLI and pytest."""
    nranks = len(device_ids)
    # Scratch = nranks * COUNT_PER_RANK floats (staging) + signal tail.
    scratch_nbytes = nranks * COUNT_PER_RANK * DTYPE_NBYTES + SIGNAL_TAIL_NBYTES
    window_size = max(scratch_nbytes, 4 * 1024)

    print(f"[reduce_scatter] platform={platform} devices={device_ids} nranks={nranks}")

    host_inputs = [
        torch.tensor([i + rank * 100 for i in range(nranks * COUNT_PER_RANK)], dtype=torch.float32).share_memory_()
        for rank in range(nranks)
    ]
    host_outputs = [torch.zeros(COUNT_PER_RANK, dtype=torch.float32).share_memory_() for _ in range(nranks)]

    print("[reduce_scatter] compiling kernels...")
    chip_callable = build_chip_callable(platform, pto_isa_commit)

    worker = Worker(
        level=3,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=device_ids,
        num_sub_workers=0,
    )
    chip_handle = worker.register(chip_callable)

    try:
        print("[reduce_scatter] init worker (forks chip children; base comm is lazy)...")
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
                        count=nranks * COUNT_PER_RANK,
                        nbytes=scratch_nbytes,
                    )
                ],
            ) as handle:
                for i in range(nranks):
                    domain = handle[i]
                    print(
                        f"[reduce_scatter] chip {i}: rank={domain.domain_rank}/{domain.domain_size} "
                        f"window=[0x{domain.local_window_base:x} +{domain.actual_window_size}B] "
                        f"scratch=0x{domain.buffer_ptrs['scratch']:x}"
                    )
                    chip_args = TaskArgs()
                    chip_args.add_tensor(make_tensor_arg(host_inputs[i]), TensorArgType.INPUT)
                    chip_args.add_tensor(make_tensor_arg(host_outputs[i]), TensorArgType.OUTPUT_EXISTING)
                    chip_args.add_tensor(
                        ContinuousTensor.make(
                            data=domain.buffer_ptrs["scratch"],
                            shapes=(nranks * COUNT_PER_RANK,),
                            dtype=DataType.FLOAT32,
                            child_memory=True,
                        ),
                        TensorArgType.INOUT,
                    )
                    chip_args.add_scalar(domain.domain_size)
                    chip_args.add_scalar(domain.device_ctx)
                    orch.submit_next_level(chip_handle, chip_args, cfg, worker=i)

        print(f"[reduce_scatter] running {nranks}-chip reduce-scatter DAG...")
        worker.run(orch_fn, args=None, config=CallConfig())

        ok = True
        for i in range(nranks):
            expected = torch.tensor(expected_output(nranks, i), dtype=torch.float32)
            max_diff = float(torch.max(torch.abs(host_outputs[i] - expected)))
            print(f"[reduce_scatter] chip {i}: max |out - expected| = {max_diff:.3e}")
            if max_diff > 1e-3:
                ok = False
                for j in range(min(4, COUNT_PER_RANK)):
                    print(f"  output[{j}]={float(host_outputs[i][j])!r} expected={float(expected[j])!r}")

        if not ok:
            print("[reduce_scatter] golden check FAILED")
            return 1
        print("[reduce_scatter] all ranks matched golden ✅")
        return 0
    finally:
        worker.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--platform", default="a2a3", help="Platform backend, e.g. a2a3 or a2a3sim.")
    parser.add_argument(
        "-d", "--device", default="0-1", help="Device range, e.g. '0-1' or '0-3'. 2 to 16 chips required."
    )
    parser.add_argument("--pto-isa-commit", default=None, help="Optional PTO ISA commit/tag to fetch before compiling.")
    cli = parser.parse_args()

    return run(parse_device_range(cli.device), platform=cli.platform, pto_isa_commit=cli.pto_isa_commit)


if __name__ == "__main__":
    sys.exit(main())
