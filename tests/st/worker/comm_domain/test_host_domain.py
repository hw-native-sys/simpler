# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Worker-owned Host mappings for L2/L3, MTE and PTO SDMA."""

import ctypes
import struct
from pathlib import Path

import pytest
from simpler.buffer import AccessMode
from simpler.task_interface import (
    ArgDirection,
    ChipCallable,
    CommBufferSpec,
    CoreCallable,
    DataType,
    TaskArgs,
    TensorArgType,
    TensorTransfer,
)
from simpler.task_interface import (
    ArgDirection as D,
)
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root
from simpler_setup.scene_test import compile_chip_callable_spec

HOST_RUNTIMES = (
    pytest.param("tensormap_and_ringbuffer", marks=pytest.mark.runtime("tensormap_and_ringbuffer")),
    pytest.param("host_build_graph", marks=pytest.mark.runtime("host_build_graph")),
)


def create_host_buffer(worker, nbytes):
    return worker.create_buffer(nbytes)


def build_chip_callable(platform, runtime="tensormap_and_ringbuffer", *, kernel_source=None):
    root = Path(__file__).resolve().parents[4]
    kernels = root / "examples/workers/l3/multi_chip_dispatch/kernels"
    if runtime == "host_build_graph":
        orchestration_source = (
            root / "examples/workers/l4/compute_then_tload_mixed_l3/kernels/orchestration/local_add_orch.cpp"
        )
        func_name = "local_add_orchestration"
    else:
        orchestration_source = kernels / "orchestration/vector_add_orch.cpp"
        func_name = "vector_add_orchestration"
    compiler = KernelCompiler(platform=platform)
    binary = compiler.compile_incore(
        source_path=str(kernel_source if kernel_source is not None else kernels / "aiv/vector_add_kernel.cpp"),
        core_type="aiv",
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=compiler.get_orchestration_include_dirs(runtime),
    )
    if not platform.endswith("sim"):
        binary = extract_text_section(binary)
    signature = [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT]
    orchestration = compiler.compile_orchestration(runtime_name=runtime, source_path=str(orchestration_source))
    return ChipCallable.build(
        signature=signature,
        func_name=func_name,
        binary=orchestration,
        children=[(0, CoreCallable.build(signature=signature, binary=binary))],
    )


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


@pytest.mark.runtime(RUNTIME)
@pytest.mark.device_count(2)
@pytest.mark.parametrize(
    ("engine", "host_output"),
    [
        pytest.param("mte", True, id="host-write-mte", marks=pytest.mark.platforms(["a2a3sim", "a2a3"])),
        pytest.param("sdma", False, id="host-read-sdma", marks=[pytest.mark.platforms(["a2a3"]), pytest.mark.sdma]),
        pytest.param("sdma", True, id="host-write-sdma", marks=[pytest.mark.platforms(["a2a3"]), pytest.mark.sdma]),
    ],
)
def test_host_domain_two_chip_transfer(st_platform, st_device_ids, engine, host_output):
    runtime = RUNTIME
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
        buffers=[
            CommBufferSpec("input", "float32", N, SIZE),
            CommBufferSpec("output", "float32", N, SIZE),
        ],
        retain_after_run=True,
    )
    domains = []
    try:
        worker.init()
        results = [worker.create_buffer(SIZE) for _ in range(2)]
        host = create_host_buffer(worker, SIZE * 3)
        mapped_views = []
        for iteration in range(2):
            value = 3.0 + iteration
            data = struct.pack(f"<{N}f", *([value] * N))

            def transfer(orch, _args, cfg):
                domain = orch.get_or_allocate_domain(**request)
                if domains:
                    assert domain is domains[0]
                else:
                    domains.append(domain)
                host.buffer[:SIZE] = data
                for rank in range(2):
                    orch.copy_to_domain(domain, rank, data, buffer="input")
                views = [
                    worker.map_view(
                        host.tensor((N,), DataType.FLOAT32, byte_offset=(rank + 1) * SIZE if host_output else 0),
                        target=rank,
                        access=AccessMode.WRITE if host_output else AccessMode.READ,
                    )
                    for rank in range(2)
                ]
                mapped_views.extend(views)
                device_buffer = "input" if host_output else "output"
                device_views = [
                    worker.map_view(
                        domain.member(rank).buffers[device_buffer].tensor((N,), DataType.FLOAT32),
                        target=rank,
                        access=AccessMode.READ if host_output else AccessMode.WRITE,
                    )
                    for rank in range(2)
                ]
                for rank, mapped in enumerate(views):
                    source = device_views[rank] if host_output else mapped
                    destination = mapped if host_output else device_views[rank]
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
                        raw = host.buffer[(rank + 1) * SIZE : (rank + 2) * SIZE]
                    else:
                        raw = orch.copy_from_domain(domains[0], rank, SIZE, buffer="output")
                    assert struct.unpack(f"<{N}f", raw) == (expected,) * N

            worker.run(read)
            if engine == "sdma":
                for result in results:
                    assert struct.unpack_from(f"<{N}f", result.buffer) == (expected + 1,) * N
        domains[0].release()
        assert domains[0].freed
        assert not host.closed
        for view in mapped_views:
            worker.unmap_view(view)
        worker.release_buffer(host)
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim", "a2a3"])
@pytest.mark.parametrize(
    "runtime",
    [
        pytest.param(RUNTIME, marks=[pytest.mark.runtime(RUNTIME), pytest.mark.manual]),
        pytest.param("host_build_graph", marks=pytest.mark.runtime("host_build_graph")),
    ],
)
@pytest.mark.device_count(2)
def test_host_domain_release_after_accepted_tasks(st_platform, st_device_ids, runtime):
    worker = Worker(
        level=3, platform=st_platform, runtime=runtime, device_ids=list(map(int, st_device_ids[:2])), num_sub_workers=0
    )
    chip = worker.register(build_chip_callable(st_platform, runtime))
    saved = []
    try:
        worker.init()
        outputs = [worker.create_buffer(SIZE) for _ in range(2)]
        host = create_host_buffer(worker, SIZE)
        host.buffer[:] = struct.pack(f"<{N}f", *([7.0] * N))
        mapped_views = []

        def submit(orch, _args, cfg):
            domain = orch.allocate_domain(
                name="closing",
                members=[0, 1],
                window_size=SIZE,
                buffers=[
                    CommBufferSpec("unused", "float32", N, SIZE),
                ],
            )
            views = [
                worker.map_view(host.tensor((N,), DataType.FLOAT32), target=rank, access=AccessMode.READ)
                for rank in range(2)
            ]
            mapped_views.extend(views)
            for rank, view in enumerate(views):
                args = TaskArgs()
                args.add_tensor(view, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
                args.add_tensor(view, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
                args.add_tensor(outputs[rank].tensor((N,), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
                orch.submit_next_level(chip, args, cfg, worker=rank)
            # Submitted uses, rather than domain lifetime, fence mapping cleanup.
            with pytest.raises(RuntimeError):
                worker.unmap_view(views[0])
            domain.release()
            assert domain.released and not domain.freed
            saved.append(domain)

        worker.run(submit)
        assert saved[0].freed
        assert not host.closed
        for view in mapped_views:
            worker.unmap_view(view)
        worker.release_buffer(host)
        for output in outputs:
            assert struct.unpack_from(f"<{N}f", output.buffer) == (14.0,) * N
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim", "a2a3"])
@pytest.mark.manual
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
        ids = [worker.create_buffer(6 * 4) for _ in range(2)]
        results = [worker.create_buffer(output_size) for _ in range(2)]
        host = create_host_buffer(worker, table_size)
        mapped_views = []
        for rank in range(2):
            struct.pack_into("<6I", ids[rank].buffer, 0, *rows[rank])
        for iteration in range(2):
            table = [0x3F00 + (r + c + 17 * iteration) % 128 for r in range(128) for c in range(256)]

            def fetch(orch, _args, cfg):
                domain = orch.get_or_allocate_domain(
                    name="table",
                    members=[0, 1],
                    window_size=64,
                    buffers=[
                        CommBufferSpec("unused", "uint8", 64, 64),
                    ],
                    retain_after_run=True,
                )
                if saved:
                    assert domain is saved[0]
                else:
                    saved.append(domain)
                # CPU initialization updates the Host backing, not a device replica.
                host.buffer[:] = struct.pack(f"<{len(table)}H", *table)
                views = [
                    worker.map_view(
                        host.tensor((128, 256), DataType.BFLOAT16),
                        target=rank,
                        access=AccessMode.READ,
                    )
                    for rank in range(2)
                ]
                mapped_views.extend(views)
                for rank, view in enumerate(views):
                    args = TaskArgs()
                    args.add_tensor(view, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
                    args.add_tensor(ids[rank].tensor((6,), DataType.UINT32), TensorArgType.INPUT)
                    args.add_tensor(results[rank].tensor((6, 256), DataType.BFLOAT16), TensorArgType.OUTPUT_EXISTING)
                    orch.submit_next_level(chip, args, cfg, worker=rank)

            worker.run(fetch)
            for rank in range(2):
                expected = [table[r * 256 + c] if r < 128 else 0 for r in rows[rank] for c in range(256)]
                assert list(struct.unpack_from("<1536H", results[rank].buffer)) == expected
        saved[0].release()
        for view in mapped_views:
            worker.unmap_view(view)
        worker.release_buffer(host)
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim", "a2a3"])
@pytest.mark.runtime(RUNTIME)
@pytest.mark.device_count(1)
def test_direct_host_two_live_buffers(st_platform, st_device_ids):
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    chip = worker.register(build_chip_callable(st_platform, RUNTIME))
    try:
        worker.init()
        sources = [create_host_buffer(worker, SIZE) for _ in range(2)]
        result = worker.create_buffer(SIZE)
        for iteration, source in enumerate(sources):
            payload = struct.pack(f"<{N}f", *([float(iteration + 2)] * N))
            ctypes.memmove(source.base, payload, SIZE)
            native = source.tensor((N,), DataType.FLOAT32)
            view = worker.map_view(native)
            assert native.mapping_ref == 0 and view.mapping_ref != 0
            assert native.buffer.identity == view.buffer.identity
            args = TaskArgs()
            args.add_tensor(native, TensorArgType.INPUT, transfer=TensorTransfer.H2D)
            args.add_tensor(view, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
            args.add_tensor(result.tensor((N,), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
            worker.run(chip, args)
            assert struct.unpack_from(f"<{N}f", result.buffer) == (float(2 * (iteration + 2)),) * N
            worker.unmap_view(view)
            with pytest.raises(ValueError, match="live explicit mapping"):
                worker.run(chip, args)
            remapped = worker.map_view(native, access=AccessMode.READ)
            assert remapped.mapping_ref != view.mapping_ref
            retry = TaskArgs()
            retry.add_tensor(native, TensorArgType.INPUT, transfer=TensorTransfer.H2D)
            retry.add_tensor(remapped, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
            retry.add_tensor(result.tensor((N,), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
            worker.run(chip, retry)
            assert struct.unpack_from(f"<{N}f", result.buffer) == (float(2 * (iteration + 2)),) * N
            worker.unmap_view(remapped)
            worker.release_buffer(source)
            with pytest.raises(ValueError):
                worker.map_view(native)
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim", "a2a3"])
@pytest.mark.runtime("host_build_graph")
@pytest.mark.device_count(1)
def test_direct_host_native_control_and_mapped_data(st_platform, st_device_ids):
    runtime = "host_build_graph"
    kernels = ROOT / "examples/workers/l2/framework_eager/kernels"
    chip = compile_chip_callable_spec(
        {
            "orchestration": {
                "source": str(kernels / "value_arguments_orchestration.cpp"),
                "function_name": "framework_host_value_arguments",
                "signature": [ArgDirection.IN, ArgDirection.OUT, ArgDirection.IN] + [ArgDirection.SCALAR] * 4,
            },
            "incores": [
                {
                    "func_id": 0,
                    "source": str(kernels / "aiv/value_arguments.cpp"),
                    "core_type": "aiv",
                    "signature": [ArgDirection.IN, ArgDirection.OUT, ArgDirection.SCALAR, ArgDirection.SCALAR],
                }
            ],
        },
        st_platform,
        runtime,
        f"host-domain-native-control-{st_platform}",
    )
    worker = Worker(level=2, platform=st_platform, runtime=runtime, device_id=int(st_device_ids[0]))
    handle = worker.register(chip)
    count, offset = 64 * 64, 64
    try:
        worker.init()
        source = create_host_buffer(worker, offset + count * 4)
        native = source.tensor((1,), DataType.INT32)
        mapped = worker.map_view(source.tensor((64, 64), DataType.FLOAT32, byte_offset=offset), access=AccessMode.READ)
        assert native.buffer.identity == mapped.buffer.identity and native.mapping_ref == 0 and mapped.mapping_ref
        result = worker.create_buffer(count * 4)
        for epoch, control in enumerate((3, -7)):
            values = [float(i % 17 + epoch + 1) for i in range(count)]
            ctypes.memmove(source.base, struct.pack("<i", control), 4)
            ctypes.memmove(source.base + offset, struct.pack(f"<{count}f", *values), count * 4)
            args = TaskArgs()
            args.add_tensor(mapped, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
            args.add_tensor(result.tensor((64, 64), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
            args.add_tensor(native, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
            for value in (ctypes.c_float(2), ctypes.c_int8(0), ctypes.c_uint64(0), ctypes.c_double(1)):
                args.add_scalar(value)
            worker.run(handle, args)
            assert struct.unpack_from(f"<{count}f", result.buffer) == tuple(2 * value + control for value in values)
        worker.unmap_view(mapped)
        worker.release_buffer(source)
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim", "a2a3"])
@pytest.mark.manual
@pytest.mark.parametrize(
    ("runtime", "access"),
    [
        pytest.param(
            "host_build_graph",
            AccessMode.WRITE,
            id="write-host_build_graph",
            marks=pytest.mark.runtime("host_build_graph"),
        ),
        pytest.param(RUNTIME, AccessMode.READWRITE, id=f"readwrite-{RUNTIME}", marks=pytest.mark.runtime(RUNTIME)),
    ],
)
@pytest.mark.device_count(1)
def test_direct_host_mapped_writeback(st_platform, st_device_ids, runtime, access):
    readwrite = access == AccessMode.READWRITE
    shape = (64, 64) if readwrite else (N,)
    count = 64 * 64 if readwrite else N
    if readwrite:
        kernels = ROOT / "examples/workers/l2/framework_eager/kernels"
        signature = [ArgDirection.IN, ArgDirection.INOUT]
        chip = compile_chip_callable_spec(
            {
                "orchestration": {
                    "source": str(kernels / "alias_inout_orchestration.cpp"),
                    "function_name": "framework_alias_inout",
                    "signature": signature,
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": str(kernels / "aiv/alias_inout.cpp"),
                        "core_type": "aiv",
                        "signature": signature,
                    }
                ],
            },
            st_platform,
            runtime,
            f"host-domain-writeback-{st_platform}-{runtime}",
        )
    else:
        chip = build_chip_callable(st_platform, runtime)
    worker = Worker(level=2, platform=st_platform, runtime=runtime, device_id=int(st_device_ids[0]))
    handle = worker.register(chip)
    offset, extent = 64, count * 4
    try:
        worker.init()
        source = worker.create_buffer(extent)
        target = create_host_buffer(worker, extent + 128)
        ctypes.memset(target.base, 0xA5, extent + 128)
        expected = [1.0] * count
        ctypes.memmove(target.base + offset, struct.pack(f"<{count}f", *expected), extent)
        mapped = worker.map_view(target.tensor(shape, DataType.FLOAT32, byte_offset=offset), access=access)
        for epoch in range(2):
            values = [float(i % 17 + epoch + 2) for i in range(count)]
            struct.pack_into(f"<{count}f", source.buffer, 0, *values)
            args = TaskArgs()
            args.add_tensor(source.tensor(shape, DataType.FLOAT32), TensorArgType.INPUT)
            if not readwrite:
                args.add_tensor(source.tensor(shape, DataType.FLOAT32), TensorArgType.INPUT)
            args.add_tensor(
                mapped,
                TensorArgType.INOUT if readwrite else TensorArgType.OUTPUT_EXISTING,
                transfer=TensorTransfer.NONE,
            )
            worker.run(handle, args)
            expected = [a + b for a, b in zip(expected, values)] if readwrite else [2 * value for value in values]
            raw = ctypes.string_at(target.base, extent + 128)
            assert struct.unpack_from(f"<{count}f", raw, offset) == tuple(expected)
            assert raw[:offset] == raw[-64:] == b"\xa5" * 64
        worker.unmap_view(mapped)
        worker.release_buffer(target)
    finally:
        worker.close()
