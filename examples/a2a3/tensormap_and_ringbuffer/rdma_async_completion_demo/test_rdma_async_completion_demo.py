#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Direct SDK: repeated IN_DIE PUT/GET with one retained two-device domain."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from functools import partial

import pytest
from simpler.task_interface import ArgDirection, CallConfig, ChipCallable, CommBufferSpec, CoreCallable, TaskArgs
from simpler.worker import Worker
from simpler.workspace_config import RdmaConfig

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

RDMA_WORKSPACE_REQUIRED = "requires SIMPLER_ENABLE_PTO_RDMA_WORKSPACE=ON"

pytestmark = [
    pytest.mark.coverage_runtime("tensormap_and_ringbuffer"),
    pytest.mark.coverage_skip(reason=RDMA_WORKSPACE_REQUIRED, platforms=["a2a3"]),
]

WINDOW_BYTES = 1024


def _build_rdma_callable(operation: str) -> ChipCallable:
    """Compile the packaged templates using the ordinary SDK; parameters remain dynamic."""
    compiler = KernelCompiler("a2a3")
    runtime = "tensormap_and_ringbuffer"
    source = compiler.project_root / "src/a2a3/runtime/tensormap_and_ringbuffer/builtins/rdma"
    signature = [ArgDirection.IN] * 6
    binary = compiler.compile_incore(
        str(source / f"rdma_{operation}.cpp"),
        "aiv",
        ensure_pto_isa_root(),
        extra_include_dirs=compiler.get_orchestration_include_dirs(runtime),
    )
    core = CoreCallable.build(signature=signature, binary=extract_text_section(binary))
    orchestration = compiler.compile_orchestration(runtime, str(source / "rdma_orchestration.cpp"))
    return ChipCallable.build(
        signature=signature,
        binary=orchestration,
        func_name="aicpu_orchestration_entry",
        config_name="aicpu_orchestration_config",
        children=[(0, core)],
    )


@dataclass
class Transfer:
    callable_handle: object
    operation: str
    iteration: int
    nbytes: int
    qp_index: int
    local_offset: int
    remote_offset: int
    roce_ips: tuple[str, ...] = ()
    identity: object = None
    contexts: list[int] = field(default_factory=list)
    received: list[bytes] = field(default_factory=list)

    def pattern(self, rank: int) -> bytes:
        return bytes((rank * 71 + self.iteration * 19 + i) % 256 for i in range(self.nbytes))


def rdma_host_orchestration(orch, _args, config, *, request: Transfer) -> None:
    """Reuse one domain, stage both sources, then submit one scalar-only task per rank."""
    domain = orch.get_or_allocate_domain(
        name="rdma-repeated",
        members=[0, 1],
        window_size=WINDOW_BYTES,
        buffers=[CommBufferSpec(name="window", dtype="uint8", count=WINDOW_BYTES, nbytes=WINDOW_BYTES)],
        retain_after_run=True,
        workspace_engines={"rdma": RdmaConfig(qp_num=2, roce_ips=request.roce_ips)},
    )
    request.identity = domain.ref
    request.contexts = [domain.device_context(rank, workspace_engines=("rdma",)) for rank in range(2)]
    source_offset = request.local_offset if request.operation == "put" else request.remote_offset
    # Complete Host staging on both devices before either device may read its peer.
    for rank in range(2):
        orch.copy_to_domain(domain, rank, bytes(WINDOW_BYTES))
        orch.copy_to_domain(domain, rank, request.pattern(rank), offset=source_offset)
    for rank in range(2):
        args = TaskArgs()
        for scalar in (
            request.contexts[rank],
            1 - rank,
            request.local_offset,
            request.remote_offset,
            request.nbytes,
            request.qp_index,
        ):
            args.add_scalar(scalar)
        orch.submit_next_level(request.callable_handle, args, config, worker=rank)


def _readback(orch, _args, _config, *, request: Transfer) -> None:
    domain = orch.get_domain("rdma-repeated")
    destination = request.remote_offset if request.operation == "put" else request.local_offset
    request.received = [orch.copy_from_domain(domain, rank, request.nbytes, offset=destination) for rank in range(2)]


def run(device_ids: list[int], *, build: bool = False, roce_ips: tuple[str, ...] = ()) -> None:
    if len(device_ids) != 2 or len(set(device_ids)) != 2:
        raise ValueError("two distinct A3 device IDs are required")
    worker = Worker(
        level=3,
        platform="a2a3",
        runtime="tensormap_and_ringbuffer",
        device_ids=device_ids,
        num_sub_workers=0,
        build=build,
    )
    handles = {op: worker.register(_build_rdma_callable(op)) for op in ("put", "get")}
    first_identity = None
    first_contexts = None
    try:
        worker.init()
        for iteration, operation in enumerate(("put", "get", "put", "get")):
            request = Transfer(
                handles[operation],
                operation,
                iteration,
                64 + iteration * 64,
                iteration % 2,
                64 + iteration * 64,
                512 + iteration * 64,
                roce_ips,
            )
            # run() joins FIN and the RDMA completion, not just kernel submission.
            worker.run(partial(rdma_host_orchestration, request=request), config=CallConfig())
            if first_identity is None:
                first_identity, first_contexts = request.identity, request.contexts
            assert request.identity == first_identity and request.contexts == first_contexts
            # A second callback reads only after the previous run's distributed fence.
            worker.run(partial(_readback, request=request), config=CallConfig())
            for rank, received in enumerate(request.received):
                assert received == request.pattern(1 - rank), (operation, iteration, rank)
    finally:
        worker.close()


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.device_count(2)
def test_rdma_async_completion_demo(st_platform, st_device_ids) -> None:
    if os.environ.get("SIMPLER_ENABLE_PTO_RDMA_WORKSPACE", "").strip().upper() not in {"1", "ON", "TRUE", "YES", "Y"}:
        pytest.skip(RDMA_WORKSPACE_REQUIRED)
    if st_platform != "a2a3":
        pytest.skip("IN_DIE requires A3 hardware")
    run([int(device) for device in st_device_ids[:2]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs=2, type=int, default=[0, 1])
    parser.add_argument("--roce-ips", nargs=2, default=())
    parser.add_argument("--build", action="store_true")
    options = parser.parse_args()
    run(options.devices, build=options.build, roce_ips=tuple(options.roce_ips))
