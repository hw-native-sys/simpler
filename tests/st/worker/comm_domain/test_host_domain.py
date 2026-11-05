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
    CoreCallable,
    DataType,
    TaskArgs,
    TensorArgType,
    TensorTransfer,
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
