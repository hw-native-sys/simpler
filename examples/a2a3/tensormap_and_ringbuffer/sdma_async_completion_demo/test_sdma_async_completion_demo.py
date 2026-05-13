#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SDMA TGET_ASYNC deferred completion smoke test for onboard a2a3.

Each rank stages its input inside the HCCL window.  The deferred producer
TGET_ASYNCs the peer rank's input into local ``out`` and registers the PTO
AsyncEvent through ``defer_pto_async_event``.  The consumer depends on the
producer output and writes ``result = out + 1``.  Correct ``out`` and
``result`` therefore validate both the SDMA completion polling and the
deferred-release dependency path.
"""

from __future__ import annotations

import argparse
import ctypes
import os
from multiprocessing.shared_memory import SharedMemory

import torch
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipBootstrapConfig,
    ChipBufferSpec,
    ChipCallable,
    ChipContext,
    CommDomain,
    CommDomainPlan,
    ContinuousTensor,
    CoreCallable,
    DataType,
    HostBufferStaging,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root
from simpler_setup.torch_interop import make_tensor_arg

HERE = os.path.dirname(os.path.abspath(__file__))
N = 128 * 128
DTYPE_NBYTES = 4
COMM_MAX_RANK_NUM = 64


class _CommContextStruct(ctypes.Structure):
    _fields_ = [
        ("workSpace", ctypes.c_uint64),
        ("workSpaceSize", ctypes.c_uint64),
        ("rankId", ctypes.c_uint32),
        ("rankNum", ctypes.c_uint32),
        ("winSize", ctypes.c_uint64),
        ("windowsIn", ctypes.c_uint64 * COMM_MAX_RANK_NUM),
        ("windowsOut", ctypes.c_uint64 * COMM_MAX_RANK_NUM),
    ]


def parse_device_range(spec: str) -> list[int]:
    if "," in spec:
        return [int(x) for x in spec.split(",") if x]
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        return list(range(lo, hi + 1))
    return [int(spec)]


def build_chip_callable(platform: str, pto_isa_commit: str | None, clone_protocol: str) -> ChipCallable:
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root(commit=pto_isa_commit, clone_protocol=clone_protocol)
    include_dirs = kc.get_orchestration_include_dirs(runtime)
    extra_includes = list(include_dirs) + [str(kc.project_root / "src" / "common")]

    children = []
    for func_id, rel in [
        (0, "kernels/aiv/kernel_sdma_tget_async.cpp"),
        (1, "kernels/aiv/kernel_consumer.cpp"),
    ]:
        kernel = kc.compile_incore(
            source_path=os.path.join(HERE, rel),
            core_type="aiv",
            pto_isa_root=pto_isa_root,
            extra_include_dirs=extra_includes,
        )
        if not platform.endswith("sim"):
            kernel = extract_text_section(kernel)
        children.append(
            (
                func_id,
                CoreCallable.build(
                    signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.OUT, ArgDirection.IN],
                    binary=kernel,
                ),
            )
        )

    orch = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "kernels/orchestration/sdma_async_completion_orch.cpp"),
        extra_include_dirs=[str(kc.project_root / "src" / "common")],
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.OUT, ArgDirection.IN],
        func_name="sdma_async_completion_orchestration",
        binary=orch,
        children=children,
    )


def read_comm_context(worker: Worker, ctx: ChipContext, domain_name: str) -> _CommContextStruct:
    domain = ctx.domains[domain_name]
    host_ctx = _CommContextStruct()
    worker.copy_from(
        ctypes.addressof(host_ctx),
        domain.device_ctx,
        ctypes.sizeof(host_ctx),
        worker_id=ctx.worker_index,
    )
    return host_ctx


def require_sdma_workspace(worker: Worker, contexts: list[ChipContext], domain_name: str) -> None:
    missing = []
    for ctx in contexts:
        host_ctx = read_comm_context(worker, ctx, domain_name)
        if host_ctx.workSpace == 0 or host_ctx.workSpaceSize == 0:
            missing.append(ctx.worker_index)
    if missing:
        raise RuntimeError(
            "sdma_async_completion_demo requires an a2a3 host runtime built "
            "with SIMPLER_ENABLE_PTO_SDMA_WORKSPACE=1; missing SDMA workspace "
            f"on worker indices {missing}. Rebuild with --build after setting "
            "SIMPLER_ENABLE_PTO_SDMA_WORKSPACE=1 and PTO_ISA_ROOT."
        )


def run(
    platform: str = "a2a3",
    device_ids: list[int] | None = None,
    pto_isa_commit: str | None = None,
    build: bool = False,
) -> int:
    if device_ids is None:
        device_ids = [0, 1]
    nranks = len(device_ids)
    if nranks != 2:
        raise ValueError(f"sdma_async_completion_demo needs exactly 2 devices, got {device_ids}")
    if platform.endswith("sim"):
        raise ValueError("sdma_async_completion_demo requires onboard a2a3 hardware")

    input_nbytes = N * DTYPE_NBYTES
    window_size = max(input_nbytes, 4 * 1024)

    inputs = [
        torch.tensor([float(rank * 1000 + (i % 251)) / 10.0 for i in range(N)], dtype=torch.float32)
        for rank in range(nranks)
    ]
    out = [torch.zeros(N, dtype=torch.float32).share_memory_() for _ in range(nranks)]
    result = [torch.zeros(N, dtype=torch.float32).share_memory_() for _ in range(nranks)]

    input_shms = [SharedMemory(create=True, size=input_nbytes) for _ in range(nranks)]
    for rank, shm in enumerate(input_shms):
        buf = shm.buf
        if buf is None:
            raise RuntimeError("SharedMemory buffer is unavailable")
        buf[:input_nbytes] = inputs[rank].numpy().tobytes()

    comm_plan = CommDomainPlan(
        domains=[
            CommDomain(
                name="default",
                worker_indices=list(range(nranks)),
                window_size=window_size,
                buffers=[
                    ChipBufferSpec(
                        name="input_window",
                        dtype="float32",
                        count=N,
                        nbytes=input_nbytes,
                        load_from_host=True,
                    ),
                ],
            )
        ]
    )
    cfgs = [
        ChipBootstrapConfig(
            comm=comm_plan.bootstrap_for_worker(rank),
            host_inputs=[
                HostBufferStaging(
                    domain_name="default",
                    name="input_window",
                    shm_name=input_shms[rank].name,
                    size=input_nbytes,
                )
            ],
        )
        for rank in range(nranks)
    ]

    chip_callable = build_chip_callable(platform, pto_isa_commit, "https")
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
        worker.init()
        contexts: list[ChipContext] = worker.chip_contexts
        require_sdma_workspace(worker, contexts, "default")

        def orch_fn(orch, _args, cfg):
            for rank, ctx in enumerate(contexts):
                domain = ctx.domains["default"]
                args = TaskArgs()
                args.add_tensor(
                    ContinuousTensor.make(
                        data=domain.buffer_ptrs["input_window"],
                        shapes=(N,),
                        dtype=DataType.FLOAT32,
                        child_memory=True,
                    ),
                    TensorArgType.INPUT,
                )
                args.add_tensor(make_tensor_arg(out[rank]), TensorArgType.OUTPUT_EXISTING)
                args.add_tensor(make_tensor_arg(result[rank]), TensorArgType.OUTPUT_EXISTING)
                args.add_scalar(domain.device_ctx)
                orch.submit_next_level(chip_cid, args, cfg, worker=rank)

        worker.run(orch_fn, args=None, config=CallConfig())

        ok = True
        for rank in range(nranks):
            peer = 1 - rank
            expected_out = inputs[peer]
            expected_result = expected_out + 1.0
            max_out = float(torch.max(torch.abs(out[rank] - expected_out)))
            max_result = float(torch.max(torch.abs(result[rank] - expected_result)))
            print(f"[sdma_async_completion_demo] rank {rank}: max_out={max_out:.3e} max_result={max_result:.3e}")
            ok = ok and max_out <= 1e-3 and max_result <= 1e-3
        return 0 if ok else 1
    finally:
        worker.close()
        for shm in input_shms:
            shm.close()
            shm.unlink()


def test_sdma_async_completion_demo() -> None:
    assert run("a2a3", [0, 1]) == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3")
    parser.add_argument("-d", "--device", default="0-1")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--pto-isa-commit", default=None)
    args = parser.parse_args()
    return run(args.platform, parse_device_range(args.device), args.pto_isa_commit, build=args.build)


if __name__ == "__main__":
    raise SystemExit(main())
