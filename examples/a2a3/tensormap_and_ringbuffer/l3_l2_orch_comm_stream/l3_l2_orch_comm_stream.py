#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Closed-loop L3-L2 in-flight orchestration communication stream example."""

from __future__ import annotations

import ctypes
import os
import struct

from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, ChipCallable, CoreCallable, DataType, TaskArgs, scalar_to_uint64
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

_RUNTIME = "tensormap_and_ringbuffer"
_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH_SRC = os.path.join(_HERE, "kernels", "orchestration", "l3_l2_orch_comm_orch.cpp")
_AIV_SRC = os.path.join(_HERE, "kernels", "aiv", "kernel_l3_l2_transform.cpp")
_HEADER = struct.Struct("<QII")
_HEADER_BYTES = 64
_NUMEL = 128 * 128
_NBYTES = _NUMEL * 4
_INPUT_OFFSET = _HEADER_BYTES
_OUTPUT_OFFSET = _INPUT_OFFSET + _NBYTES
_PAYLOAD_BYTES = _OUTPUT_OFFSET + _NBYTES
_ROUNDS = 3
_SCALAR = ctypes.c_float(7.0)


def _build_chip_callable(platform: str) -> ChipCallable:
    kc = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    inc_dirs = kc.get_orchestration_include_dirs(_RUNTIME)
    extra_common = [str(kc.project_root / "src" / "common")]

    aiv = kc.compile_incore(
        _AIV_SRC,
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=inc_dirs,
    )
    if not platform.endswith("sim"):
        aiv = extract_text_section(aiv)

    orch = kc.compile_orchestration(
        runtime_name=_RUNTIME,
        source_path=_ORCH_SRC,
        extra_include_dirs=extra_common,
    )
    return ChipCallable.build(
        signature=[],
        func_name="l3_l2_orch_comm_orchestration",
        binary=orch,
        children=[(0, CoreCallable.build(signature=[D.IN, D.OUT], binary=aiv))],
    )


def _float_view(tensor):
    return (ctypes.c_float * _NUMEL).from_address(int(tensor.data))


def _byte_view(tensor):
    return (ctypes.c_uint8 * int(tensor.nbytes())).from_address(int(tensor.data))


def _write_header(header_tensor, seq: int, opcode: int) -> None:
    buf = _byte_view(header_tensor)
    for i in range(_HEADER_BYTES):
        buf[i] = 0
    _HEADER.pack_into(buf, 0, seq, opcode, 0)


def _fill_input(input_tensor, round_idx: int) -> list[float]:
    values = _float_view(input_tensor)
    expected = []
    for i in range(_NUMEL):
        value = float(round_idx * 1000 + (i % 251)) / 16.0
        values[i] = value
        expected.append(value + float(_SCALAR.value))
    return expected


def _assert_output_matches(output_tensor, expected: list[float]) -> None:
    values = _float_view(output_tensor)
    for i, want in enumerate(expected):
        got = float(values[i])
        assert abs(got - want) <= 1e-5, f"output[{i}] expected {want}, got {got}"


def run_closed_loop_stream(platform: str, device_id: int) -> None:
    chip_callable = _build_chip_callable(platform)
    worker = Worker(
        level=3,
        device_ids=[int(device_id)],
        num_sub_workers=0,
        platform=platform,
        runtime=_RUNTIME,
    )
    handle = worker.register(chip_callable)
    worker.init()
    try:
        config = CallConfig()
        config.block_dim = 1
        config.aicpu_thread_num = 2

        def orch(orch_handle, _args, cfg):
            region = orch_handle.create_l3_l2_region(worker_id=0, payload_bytes=_PAYLOAD_BYTES)
            header = orch_handle.alloc([_HEADER_BYTES], DataType.UINT8)
            host_input = orch_handle.alloc([_NUMEL], DataType.FLOAT32)
            host_output = orch_handle.alloc([_NUMEL], DataType.FLOAT32)

            task_args = TaskArgs()
            for scalar in region.descriptor_scalars():
                task_args.add_scalar(int(scalar))
            task_args.add_scalar(_INPUT_OFFSET)
            task_args.add_scalar(_OUTPUT_OFFSET)
            task_args.add_scalar(_NUMEL)
            task_args.add_scalar(DataType.FLOAT32.value)
            task_args.add_scalar(_NBYTES)
            task_args.add_scalar(scalar_to_uint64(_SCALAR))
            orch_handle.submit_next_level(handle, task_args, cfg, worker=0)

            for seq in range(1, _ROUNDS + 1):
                expected = _fill_input(host_input, seq)
                region.payload_write(_INPUT_OFFSET, host_input, nbytes=_NBYTES)
                _write_header(header, seq, 1)
                region.payload_write(0, header, nbytes=_HEADER.size)
                region.notify(seq)
                region.wait(seq, timeout=5.0)
                region.payload_read(_OUTPUT_OFFSET, host_output, nbytes=_NBYTES)
                _assert_output_matches(host_output, expected)

            stop_seq = _ROUNDS + 1
            _write_header(header, stop_seq, 2)
            region.payload_write(0, header, nbytes=_HEADER.size)
            region.notify(stop_seq)

        worker.run(orch, args=None, config=config)
    finally:
        worker.close()
