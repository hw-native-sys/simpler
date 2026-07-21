#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3-L2 message queue example with L2 input-window processing.

This file is both a runnable example and a pytest scene-test entry.
"""

from __future__ import annotations

import os
import struct

import pytest
from simpler.l3_l2_message_queue import L3L2QueueOpcode
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, ChipCallable, CoreCallable, TaskArgs
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

_RUNTIME = "tensormap_and_ringbuffer"
_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH_SRC = os.path.join(_HERE, "kernels", "orchestration", "l3_l2_message_queue_orch.cpp")
_AIV_SRC = os.path.join(_HERE, "kernels", "aiv", "kernel_queue_transform.cpp")
_TIMEOUT_S = 5.0
_QUEUE_DEPTH = 8
_INPUT_ARENA_BYTES = 512 * 1024
_OUTPUT_ARENA_BYTES = 512 * 1024
_INPUT_HEADER = "<QQ"
_OUTPUT_HEADER = "<QQQ"
_INPUT_HEADER_BYTES = 64
_OUTPUT_HEADER_BYTES = 64
# The queue seq is transport ordering only. This example uses payload headers
# for application-level request correlation and output kind metadata.
_TILE_ROWS = 128
_TILE_COLS = 128
_TILE_ELEMS = _TILE_ROWS * _TILE_COLS


def _build_core_callable(signature: list[D], binary: bytes) -> CoreCallable:
    try:
        return CoreCallable.build(signature=signature, binary=binary)
    except ValueError as exc:
        if "arg_index" not in str(exc):
            raise
        return CoreCallable.build(
            signature=signature,
            arg_index=list(range(len(signature))),
            binary=binary,
        )


def _build_chip_callable(platform: str) -> ChipCallable:
    kc = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    inc_dirs = kc.get_orchestration_include_dirs(_RUNTIME)

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
        extra_include_dirs=[str(kc.project_root / "src" / "common")],
    )
    return ChipCallable.build(
        signature=[],
        func_name="l3_l2_message_queue_orchestration",
        binary=orch,
        children=[(0, _build_core_callable([D.IN, D.IN, D.OUT], aiv))],
    )


def _pack_input(request_id: int, mode: int, values: list[float]) -> bytes:
    header = struct.pack(_INPUT_HEADER, request_id, mode)
    return header + bytes(_INPUT_HEADER_BYTES - len(header)) + struct.pack(f"<{len(values)}f", *values)


def _pack_output(request_id: int, kind: int, aux: int, values: list[float]) -> bytes:
    header = struct.pack(_OUTPUT_HEADER, request_id, kind, aux)
    return header + bytes(_OUTPUT_HEADER_BYTES - len(header)) + struct.pack(f"<{len(values)}f", *values)


def _tile(base: float) -> list[float]:
    return [base + float(i % _TILE_COLS) for i in range(_TILE_ELEMS)]


def _add_scalar(values: list[float], scalar: float) -> list[float]:
    return [value + scalar for value in values]


def _add_tiles(left: list[float], right: list[float]) -> list[float]:
    return [x + y for x, y in zip(left, right)]


def _input_payloads() -> list[bytes]:
    inputs = _input_tiles()
    return [
        _pack_input(101, 1, inputs[0]),
        _pack_input(102, 2, inputs[1]),
        _pack_input(103, 3, inputs[2]),
        _pack_input(104, 3, inputs[3]),
    ]


def _expected_outputs() -> list[bytes]:
    inputs = _input_tiles()
    return [
        _pack_output(102, 20, 0, _add_scalar(inputs[1], 20.0)),
        _pack_output(101, 10, 0, _add_scalar(inputs[0], 10.0)),
        _pack_output(101, 11, 0, _add_scalar(inputs[0], 11.0)),
        _pack_output(103, 30, 104, _add_tiles(inputs[2], inputs[3])),
    ]


def _read_expected_outputs(queue, expected_outputs: list[bytes]) -> None:
    for expected in expected_outputs:
        message = queue.output.peek(timeout=_TIMEOUT_S)
        assert message.opcode == L3L2QueueOpcode.DATA
        assert message.payload_nbytes == len(expected)
        assert _read_message_payload(queue, message) == expected
        queue.output.release(message)


def _input_tiles() -> list[list[float]]:
    return [_tile(1.0), _tile(100.0), _tile(1000.0), _tile(2000.0)]


def _read_message_payload(queue, message) -> bytes:
    if message.payload_nbytes == 0:
        queue.output.read_into(message, None)
        return b""
    output = bytearray(message.payload_nbytes)
    queue.output.read_into(message, output)
    return bytes(output)


def run_l3_l2_message_queue_example(platform: str, device_id: int) -> None:
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
            queue = orch_handle.create_l3_l2_queue(
                worker_id=0,
                depth=_QUEUE_DEPTH,
                input_arena_bytes=_INPUT_ARENA_BYTES,
                output_arena_bytes=_OUTPUT_ARENA_BYTES,
            )

            task_args = TaskArgs()
            for scalar in queue.l2_task_arg_scalars():
                task_args.add_scalar(int(scalar))
            orch_handle.submit_next_level(handle, task_args, cfg, worker=0)

            payloads = _input_payloads()
            expected_outputs = _expected_outputs()

            for payload in payloads[:2]:
                queue.input.enqueue(payload, nbytes=len(payload), timeout=_TIMEOUT_S)
            _read_expected_outputs(queue, expected_outputs[:3])

            for payload in payloads[2:]:
                queue.input.enqueue(payload, nbytes=len(payload), timeout=_TIMEOUT_S)
            queue.request_stop(timeout=_TIMEOUT_S)
            _read_expected_outputs(queue, expected_outputs[3:])

            queue.free()

        worker.run(orch, args=None, config=config)
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim", "a2a3", "a5sim", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime("tensormap_and_ringbuffer")
def test_l3_l2_message_queue(st_platform, st_device_ids):
    run_l3_l2_message_queue_example(st_platform, int(st_device_ids[0]))
