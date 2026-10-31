# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Two-chip explicit Host views: MTE, PTO SDMA completion, and retained domains."""

import struct
from pathlib import Path

import pytest
from simpler.buffer import AccessMode
from simpler.task_interface import (
    ArgDirection as D,
)
from simpler.task_interface import (
    ChipCallable,
    CommBufferSpec,
    CoreCallable,
    DataType,
    HostWindowSpec,
    TaskArgs,
    TensorArgType,
    TensorTransfer,
)
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root
from tests.st.worker.comm_domain.host_domain_kernels import HOST_RUNTIMES, build_chip_callable, require_a3_host_fabric

N = 128 * 128
SIZE = N * 4
RUNTIME = "tensormap_and_ringbuffer"
ROOT = Path(__file__).resolve().parents[4]


def build_sdma(platform):
    compiler = KernelCompiler(platform=platform)
    include = compiler.get_orchestration_include_dirs(RUNTIME)
    sources = [
        Path(__file__).parent / "kernels/host_sdma.cpp",
        ROOT / "examples/a2a3/tensormap_and_ringbuffer/sdma_async_completion_demo/kernels/aiv/kernel_consumer.cpp",
    ]
    children = []
    for index, source in enumerate(sources):
        binary = compiler.compile_incore(
            source_path=str(source), core_type="aiv", pto_isa_root=ensure_pto_isa_root(), extra_include_dirs=include
        )
        children.append((index, CoreCallable.build(signature=[D.IN, D.OUT], binary=extract_text_section(binary))))
    orchestration = compiler.compile_orchestration(
        runtime_name=RUNTIME, source_path=str(Path(__file__).parent / "kernels/host_sdma_orch.cpp")
    )
    return ChipCallable.build(
        signature=[D.IN, D.OUT, D.OUT], func_name="host_sdma", binary=orchestration, children=children
    )


@pytest.mark.platforms(["a2a3sim", "a2a3"])
@pytest.mark.device_count(2)
@pytest.mark.parametrize(
    ("runtime", "engine"),
    [
        pytest.param(RUNTIME, "mte", marks=pytest.mark.runtime(RUNTIME)),
        pytest.param("host_build_graph", "mte", marks=pytest.mark.runtime("host_build_graph")),
        pytest.param(RUNTIME, "sdma", marks=[pytest.mark.runtime(RUNTIME), pytest.mark.sdma]),
    ],
)
@pytest.mark.parametrize("host_output", [False, True])
def test_host_domain_two_chip_transfer(st_platform, st_device_ids, engine, host_output, runtime):
    if engine == "sdma" and st_platform.endswith("sim"):
        pytest.skip("SDMA completion uses onboard device queues")
    worker = Worker(
        level=3,
        platform=st_platform,
        runtime=runtime,
        device_ids=list(map(int, st_device_ids[:2])),
        num_sub_workers=0,
        enable_sdma=engine == "sdma",
    )
    chip = worker.register(build_sdma(st_platform) if engine == "sdma" else build_chip_callable(st_platform, runtime))
    request = dict(
        name="host",
        members=[0, 1],
        window_size=SIZE * 2,
        buffers=[CommBufferSpec("input", "float32", N, SIZE), CommBufferSpec("output", "float32", N, SIZE)],
        host_window=HostWindowSpec(SIZE * 3),
        retain_after_run=True,
    )
    domains = []
    try:
        worker.init()
        require_a3_host_fabric(st_platform)
        results = [worker.create_buffer(SIZE) for _ in range(2)]
        for iteration in range(2):
            value = 3.0 + iteration
            data = struct.pack(f"<{N}f", *([value] * N))

            def transfer(orch, _args, cfg):
                domain = orch.get_or_allocate_domain(**request)
                if domains:
                    assert domain is domains[0]
                else:
                    domains.append(domain)
                orch.copy_to_host_domain(domain, data)
                for rank in range(2):
                    orch.copy_to_domain(domain, rank, data, buffer="input")
                views = [
                    domain.map_view(
                        domain.host().tensor(
                            (N,), DataType.FLOAT32, byte_offset=(rank + 1) * SIZE if host_output else 0
                        ),
                        domain_rank=rank,
                        access=AccessMode.WRITE if host_output else AccessMode.READ,
                    )
                    for rank in range(2)
                ]
                for rank, mapped in enumerate(views):
                    ctx = domain.member(rank)
                    source = ctx.buffers["input"].tensor((N,), DataType.FLOAT32) if host_output else mapped
                    destination = mapped if host_output else ctx.buffers["output"].tensor((N,), DataType.FLOAT32)
                    args = TaskArgs()
                    args.add_tensor(source, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
                    if engine == "mte":
                        args.add_tensor(source, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
                    args.add_tensor(
                        destination,
                        TensorArgType.INPUT if host_output and engine == "mte" else TensorArgType.OUTPUT_EXISTING,
                        transfer=TensorTransfer.NONE,
                    )
                    if engine == "sdma":
                        args.add_tensor(results[rank].tensor((N,), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
                    orch.submit_next_level(chip, args, cfg, worker=rank)

            worker.run(transfer)
            expected = value * 2 if engine == "mte" else value

            def read(orch, _args, _cfg):
                for rank in range(2):
                    if host_output:
                        raw = orch.copy_from_host_domain(domains[0], SIZE, offset=(rank + 1) * SIZE)
                    else:
                        raw = orch.copy_from_domain(domains[0], rank, SIZE, buffer="output")
                    assert struct.unpack(f"<{N}f", raw) == (expected,) * N

            worker.run(read)
            if engine == "sdma":
                for result in results:
                    assert struct.unpack_from(f"<{N}f", result.shm.buf) == (expected + 1,) * N
        domains[0].release()
        assert domains[0].freed
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim", "a2a3"])
@pytest.mark.parametrize("runtime", HOST_RUNTIMES)
@pytest.mark.device_count(2)
def test_host_domain_release_after_accepted_tasks(st_platform, st_device_ids, runtime):
    worker = Worker(
        level=3, platform=st_platform, runtime=runtime, device_ids=list(map(int, st_device_ids[:2])), num_sub_workers=0
    )
    chip = worker.register(build_chip_callable(st_platform, runtime))
    saved = []
    try:
        worker.init()
        require_a3_host_fabric(st_platform)
        outputs = [worker.create_buffer(SIZE) for _ in range(2)]

        def submit(orch, _args, cfg):
            domain = orch.allocate_domain(
                name="closing",
                members=[0, 1],
                window_size=SIZE,
                buffers=[CommBufferSpec("unused", "float32", N, SIZE)],
                host_window=HostWindowSpec(SIZE),
            )
            orch.copy_to_host_domain(domain, struct.pack(f"<{N}f", *([7.0] * N)))
            views = [
                domain.map_view(domain.host().tensor((N,), DataType.FLOAT32), domain_rank=rank, access=AccessMode.READ)
                for rank in range(2)
            ]
            for rank, view in enumerate(views):
                args = TaskArgs()
                args.add_tensor(view, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
                args.add_tensor(view, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
                args.add_tensor(outputs[rank].tensor((N,), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
                orch.submit_next_level(chip, args, cfg, worker=rank)
            domain.release()
            assert domain.released and not domain.freed
            saved.append(domain)

        worker.run(submit)
        assert saved[0].freed
        for output in outputs:
            assert struct.unpack_from(f"<{N}f", output.shm.buf) == (14.0,) * N
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim", "a2a3"])
@pytest.mark.parametrize("runtime", HOST_RUNTIMES)
@pytest.mark.device_count(2)
def test_device_selected_bf16_rows(st_platform, st_device_ids, runtime):
    callable_obj = build_chip_callable(
        st_platform, runtime, kernel_source=Path(__file__).parent / "kernels/host_sparse_read.cpp"
    )
    worker = Worker(
        level=3, platform=st_platform, runtime=runtime, device_ids=list(map(int, st_device_ids[:2])), num_sub_workers=0
    )
    chip = worker.register(callable_obj)
    table_size = 128 * 256 * 2
    output_size = 6 * 256 * 2
    rows = [[0, 127, 7, 7, 128, 0xFFFFFFFF], [127, 0, 0xFFFFFFFF, 128, 9, 9]]
    saved = []
    try:
        worker.init()
        require_a3_host_fabric(st_platform)
        ids = [worker.create_buffer(6 * 4) for _ in range(2)]
        results = [worker.create_buffer(output_size) for _ in range(2)]
        for rank in range(2):
            struct.pack_into("<6I", ids[rank].shm.buf, 0, *rows[rank])
        for iteration in range(2):
            table = [0x3F00 + (r + c + 17 * iteration) % 128 for r in range(128) for c in range(256)]

            def fetch(orch, _args, cfg):
                domain = orch.get_or_allocate_domain(
                    name="table",
                    members=[0, 1],
                    window_size=64,
                    buffers=[CommBufferSpec("unused", "uint8", 64, 64)],
                    host_window=HostWindowSpec(table_size),
                    retain_after_run=True,
                )
                if saved:
                    assert domain is saved[0]
                else:
                    saved.append(domain)
                # CPU initialization updates the Host backing, not a device replica.
                orch.copy_to_host_domain(domain, struct.pack(f"<{len(table)}H", *table))
                views = [
                    domain.map_view(
                        domain.host().tensor((128, 256), DataType.BFLOAT16), domain_rank=rank, access=AccessMode.READ
                    )
                    for rank in range(2)
                ]
                for rank, view in enumerate(views):
                    args = TaskArgs()
                    args.add_tensor(view, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
                    args.add_tensor(ids[rank].tensor((6,), DataType.UINT32), TensorArgType.INPUT)
                    args.add_tensor(results[rank].tensor((6, 256), DataType.BFLOAT16), TensorArgType.OUTPUT_EXISTING)
                    orch.submit_next_level(chip, args, cfg, worker=rank)

            worker.run(fetch)
            for rank in range(2):
                expected = [table[r * 256 + c] if r < 128 else 0 for r in rows[rank] for c in range(256)]
                assert list(struct.unpack_from("<1536H", results[rank].shm.buf)) == expected
        saved[0].release()
    finally:
        worker.close()
