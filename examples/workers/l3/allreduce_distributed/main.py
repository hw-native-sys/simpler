#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end distributed allreduce over the Worker(chip_bootstrap_configs=...) path.

The kernel (ported verbatim from #307) reads every rank's contribution out of
the HCCL window via CommRemotePtr and sums them into each rank's own window
slot.  The distributed bring-up stack this exercises, bottom up:

  - HCCL backend                        comm_init / comm_alloc_windows
  - ChipWorker.comm_* wrappers          host-side bootstrap of the communicator
  - ChipBootstrapChannel                chip child publishes SUCCESS to the parent
  - mailbox atomics                     parent/child sync without torn reads
  - error propagation                   bootstrap failures raise from Worker.init()
  - ChipWorker.bootstrap_context        one-shot per-chip bring-up
  - Worker(chip_bootstrap_configs=...)  Worker-level orchestration

These are the components that compose the bring-up — not framework hierarchy
levels (see docs/hierarchical_level_runtime.md for the L0–L6 topology).

Hardware only.  The sim backend's CommRemotePtr uses a different addressing
scheme; sim support is out of scope for this demo.

Run:
    python examples/workers/l3/allreduce_distributed/main.py -d 0-1
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from multiprocessing.shared_memory import SharedMemory

from simpler.task_interface import (
    ArgDirection,
    ChipBootstrapConfig,
    ChipBufferSpec,
    ChipCallable,
    ChipCallConfig,
    ChipCommBootstrapConfig,
    ChipContext,
    ContinuousTensor,
    CoreCallable,
    DataType,
    HostBufferStaging,
    TaskArgs,
)
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

HERE = os.path.dirname(os.path.abspath(__file__))

# Must match ALLREDUCE_COUNT in kernels/aiv/allreduce_kernel.cpp.
ALLREDUCE_COUNT = 256
DTYPE_NBYTES = 4  # float32
NOTIFY_COUNT = 1
NOTIFY_NBYTES = 4  # int32


def parse_device_range(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        ids = list(range(lo, hi + 1))
    else:
        ids = [int(spec)]
    if len(ids) != 2:
        raise ValueError(f"allreduce_distributed needs exactly 2 devices, got {ids}")
    return ids


def build_chip_callable(platform: str) -> ChipCallable:
    """Compile the AIV allreduce kernel + its C++ orchestration shim.

    The kernel consumes 4 tensors (input, recv_window, output, notify_counter)
    and one scalar (device_ctx).
    """
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root(clone_protocol="https")
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
    # Hardware path: strip the ELF down to the .text section the loader wants.
    kernel_bytes = extract_text_section(kernel_bytes)

    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "kernels/orchestration/allreduce_orch.cpp"),
    )
    core_callable = CoreCallable.build(signature=[], binary=kernel_bytes)
    return ChipCallable.build(
        signature=[
            ArgDirection.IN,
            ArgDirection.INOUT,
            ArgDirection.OUT,
            ArgDirection.INOUT,
        ],
        func_name="allreduce_orchestration",
        binary=orch_bytes,
        children=[(0, core_callable)],
    )


def make_rank_input(rank: int) -> list[float]:
    """Rank r contributes input[i] = i + r*100; matches PR #307's golden."""
    return [float(i + rank * 100) for i in range(ALLREDUCE_COUNT)]


def expected_output(nranks: int) -> list[float]:
    """output[i] = sum_r (i + r*100) = nranks*i + 100 * nranks*(nranks-1)/2."""
    return [float(nranks * i + 100 * nranks * (nranks - 1) // 2) for i in range(ALLREDUCE_COUNT)]


def pack_f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def pack_i32(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}i", *values)


def run(device_ids: list[int]) -> int:
    """Core logic — callable from both CLI and pytest."""
    nranks = len(device_ids)
    buffer_nbytes = ALLREDUCE_COUNT * DTYPE_NBYTES
    recv_window_nbytes = nranks * buffer_nbytes
    notify_nbytes = NOTIFY_COUNT * NOTIFY_NBYTES
    window_size = 4 * 1024 * 1024  # HCCL may round up; actual size surfaces via ChipContext.

    rootinfo_path = f"/tmp/pto_allreduce_distributed_rootinfo_{os.getpid()}.bin"
    try:
        os.unlink(rootinfo_path)
    except FileNotFoundError:
        pass

    print(f"[allreduce] devices={device_ids} nranks={nranks}")

    # Per-rank input SharedMemory — parent writes the bytes, child reads via
    # HostBufferStaging during bootstrap_context().  Parent unlinks right
    # after worker.init() returns (child has already finished copy_to at that
    # point).
    input_shms: list[SharedMemory] = []
    recv_window_shms: list[SharedMemory] = []
    notify_counter_shms: list[SharedMemory] = []
    output_shms: list[SharedMemory] = []
    for rank in range(nranks):
        shm = SharedMemory(create=True, size=buffer_nbytes)
        assert shm.buf is not None
        shm.buf[:buffer_nbytes] = pack_f32(make_rank_input(rank))
        input_shms.append(shm)

        recv_shm = SharedMemory(create=True, size=recv_window_nbytes)
        assert recv_shm.buf is not None
        recv_shm.buf[:recv_window_nbytes] = pack_f32([0.0] * (nranks * ALLREDUCE_COUNT))
        recv_window_shms.append(recv_shm)

        notify_shm = SharedMemory(create=True, size=notify_nbytes)
        assert notify_shm.buf is not None
        notify_shm.buf[:notify_nbytes] = pack_i32([0] * NOTIFY_COUNT)
        notify_counter_shms.append(notify_shm)

        out_shm = SharedMemory(create=True, size=buffer_nbytes)
        output_shms.append(out_shm)

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
                    name="input",
                    dtype="float32",
                    count=ALLREDUCE_COUNT,
                    placement="window",
                    nbytes=buffer_nbytes,
                    load_from_host=True,
                ),
                ChipBufferSpec(
                    name="recv_window",
                    dtype="float32",
                    count=nranks * ALLREDUCE_COUNT,
                    placement="window",
                    nbytes=recv_window_nbytes,
                    load_from_host=True,
                ),
                ChipBufferSpec(
                    name="output",
                    dtype="float32",
                    count=ALLREDUCE_COUNT,
                    placement="window",
                    nbytes=buffer_nbytes,
                    store_to_host=True,
                ),
                ChipBufferSpec(
                    name="notify_counter",
                    dtype="int32",
                    count=NOTIFY_COUNT,
                    placement="window",
                    nbytes=notify_nbytes,
                    load_from_host=True,
                ),
            ],
            host_inputs=[
                HostBufferStaging(name="input", shm_name=input_shms[rank].name, size=buffer_nbytes),
                HostBufferStaging(name="recv_window", shm_name=recv_window_shms[rank].name, size=recv_window_nbytes),
                HostBufferStaging(name="notify_counter", shm_name=notify_counter_shms[rank].name, size=notify_nbytes),
            ],
            host_outputs=[HostBufferStaging(name="output", shm_name=output_shms[rank].name, size=buffer_nbytes)],
        )
        for rank in range(nranks)
    ]

    print("[allreduce] compiling kernels...")
    chip_callable = build_chip_callable("a2a3")

    worker = Worker(
        level=3,
        platform="a2a3",
        runtime="tensormap_and_ringbuffer",
        device_ids=device_ids,
        num_sub_workers=0,
        chip_bootstrap_configs=cfgs,
    )

    try:
        print("[allreduce] init worker (forks chip children + bootstraps HCCL)...")
        worker.init()

        # Child has copied staged buffers by now. Drop our
        # copies so the shm segments don't outlive the run.
        for shm in input_shms:
            shm.close()
            shm.unlink()
        input_shms.clear()
        for shm in recv_window_shms:
            shm.close()
            shm.unlink()
        recv_window_shms.clear()
        for shm in notify_counter_shms:
            shm.close()
            shm.unlink()
        notify_counter_shms.clear()

        contexts: list[ChipContext] = worker.chip_contexts
        assert len(contexts) == nranks
        for i, ctx in enumerate(contexts):
            print(
                f"[allreduce] chip {i}: device={ctx.device_id} rank={ctx.rank}/{ctx.nranks} "
                f"window=[0x{ctx.local_window_base:x} +{ctx.actual_window_size}B] "
                f"buffers={ {k: hex(v) for k, v in ctx.buffer_ptrs.items()} }"
            )

        def orch_fn(orch, _args, cfg):
            for i, ctx in enumerate(contexts):
                chip_args = TaskArgs()
                chip_args.add_tensor(
                    ContinuousTensor.make(
                        ctx.buffer_ptrs["input"], (1, ALLREDUCE_COUNT), DataType.FLOAT32, child_memory=True
                    )
                )
                chip_args.add_tensor(
                    ContinuousTensor.make(
                        ctx.buffer_ptrs["recv_window"],
                        (nranks, ALLREDUCE_COUNT),
                        DataType.FLOAT32,
                        child_memory=True,
                    )
                )
                chip_args.add_tensor(
                    ContinuousTensor.make(
                        ctx.buffer_ptrs["output"], (1, ALLREDUCE_COUNT), DataType.FLOAT32, child_memory=True
                    )
                )
                chip_args.add_tensor(
                    ContinuousTensor.make(
                        ctx.buffer_ptrs["notify_counter"], (NOTIFY_COUNT,), DataType.INT32, child_memory=True
                    )
                )
                chip_args.add_scalar(ctx.device_ctx)
                orch.submit_next_level(chip_callable, chip_args, cfg, worker=i)

        print("[allreduce] running 2-chip allreduce DAG...")
        worker.run(orch_fn, args=None, config=ChipCallConfig())

        # Child has flushed store_to_host buffers to SharedMemory by now.
        expected = expected_output(nranks)
        ok = True
        for i in range(nranks):
            out_shm = output_shms[i]
            assert out_shm.buf is not None
            got = list(struct.unpack(f"<{ALLREDUCE_COUNT}f", bytes(out_shm.buf[:buffer_nbytes])))

            max_diff = max(abs(a - b) for a, b in zip(got, expected))
            print(f"[allreduce] chip {i}: max |out - expected| = {max_diff:.3e}")
            sample_count = min(8, ALLREDUCE_COUNT)
            print(f"[allreduce] chip {i}: out[:{sample_count}] = {got[:sample_count]}")
            print(f"[allreduce] chip {i}: expected[:{sample_count}] = {expected[:sample_count]}")
            if max_diff > 1e-3:
                ok = False
                for j in range(min(4, ALLREDUCE_COUNT)):
                    print(f"  output[{j}]={got[j]!r} expected={expected[j]!r}")

        if not ok:
            print("[allreduce] golden check FAILED")
            return 1
        print("[allreduce] all ranks matched golden ✅")
        return 0
    finally:
        worker.close()
        for shm in input_shms:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
        for shm in recv_window_shms:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
        for shm in notify_counter_shms:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
        for shm in output_shms:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
        try:
            os.unlink(rootinfo_path)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--device", default="0-1", help="Device range, e.g. '0-1'. Two chips required.")
    cli = parser.parse_args()
    return run(parse_device_range(cli.device))


if __name__ == "__main__":
    sys.exit(main())
