#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3-L2 message queue example with ordinary host-buffer staging."""

from __future__ import annotations

import ctypes
import os
import struct

from simpler.l3_l2_message_queue import L3L2QueueOpcode
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, ChipCallable, CoreCallable, TaskArgs, scalar_to_uint64
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

_RUNTIME = "tensormap_and_ringbuffer"
_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH_SRC = os.path.join(_HERE, "kernels", "orchestration", "l3_l2_message_queue_orch.cpp")
_AIV_SRC = os.path.join(_HERE, "kernels", "aiv", "kernel_queue_transform.cpp")
_TIMEOUT_S = 5.0
_QUEUE_DEPTH = 4
_INPUT_ARENA_BYTES = 4096
_OUTPUT_ARENA_BYTES = 4096
_SCALAR = ctypes.c_float(7.0)


def _build_core_callable(signature: list[D], binary: bytes) -> CoreCallable:
    try:
        return CoreCallable.build(signature=signature, binary=binary)
    except ValueError as exc:
        if "arg_index" not in str(exc):
            raise
        return CoreCallable.build(signature=signature, arg_index=list(range(len(signature))), binary=binary)


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
        func_name="l3_l2_message_queue_orchestration",
        binary=orch,
        children=[(0, _build_core_callable([D.IN, D.OUT], aiv))],
    )


def _pack_floats(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _unpack_floats(data: bytes | bytearray) -> tuple[float, ...]:
    if not data:
        return ()
    return struct.unpack(f"<{len(data) // 4}f", bytes(data))


def _input_payloads() -> list[bytes]:
    return [
        _pack_floats([float(i) for i in range(16)]),
        _pack_floats([float(100 + i) / 8.0 for i in range(32)]),
        _pack_floats([float(1000 + i) / 16.0 for i in range(48)]),
        b"",
    ]


def _expected_payload(payload: bytes) -> bytes:
    if not payload:
        return b""
    values = [value + float(_SCALAR.value) for value in _unpack_floats(payload)]
    return _pack_floats(values)


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
            task_args.add_scalar(scalar_to_uint64(_SCALAR))
            orch_handle.submit_next_level(handle, task_args, cfg, worker=0)

            payloads = _input_payloads()
            for payload in payloads:
                if payload:
                    queue.input.enqueue(payload, nbytes=len(payload), timeout=_TIMEOUT_S)
                else:
                    queue.input.enqueue(None, nbytes=0, timeout=_TIMEOUT_S)
            queue.request_stop(timeout=_TIMEOUT_S)

            for seq, payload in enumerate(payloads, start=1):
                message = queue.output.peek(timeout=_TIMEOUT_S)
                assert message.seq == seq
                assert message.opcode == L3L2QueueOpcode.DATA
                assert message.payload_nbytes == len(payload)
                if payload:
                    output = bytearray(message.payload_nbytes)
                    queue.output.read_into(message, output)
                    assert bytes(output) == _expected_payload(payload)
                else:
                    queue.output.read_into(message, None)
                queue.output.release(message)

            queue.free()

        worker.run(orch, args=None, config=config)
    finally:
        worker.close()
