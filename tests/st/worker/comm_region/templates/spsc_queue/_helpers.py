# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Private L3 SPSC queue scene-test fixtures."""

from __future__ import annotations

import importlib.util
import os
import struct
import sys
from typing import Any

from simpler import comm_endpoints as ce
from simpler.comm_region_template import SpscQueue, SpscQueueOpcode, _SpscQueueEndpointBinding
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, ChipCallable, CoreCallable, TaskArgs
from simpler.worker import Worker, attach_exception_note

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

_RUNTIME = "tensormap_and_ringbuffer"
_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH_SRC = os.path.join(_HERE, "kernels", "orchestration", "spsc_queue_orch.cpp")
_AIV_SRC = os.path.join(_HERE, "kernels", "aiv", "kernel_queue_transform.cpp")
_TIMEOUT_S = 5.0
_QUEUE_DEPTH = 4
_INPUT_ARENA_BYTES = 256 * 1024
_OUTPUT_ARENA_BYTES = 512 * 1024
_CMD_ECHO = 1
_CMD_ERROR = 2
_CMD_COMPUTE = 3
_HEADER = struct.Struct("<Q")
_TILE_ROWS = 128
_TILE_COLS = 128
_TILE_ELEMS = _TILE_ROWS * _TILE_COLS
_COMPUTE_SCALAR = 1.0
_WRAP_NBYTES = 200 * 1024
_POST_STOP_MARKER = struct.pack("<Q", 0x11)
_SMALL_ECHO = struct.pack("<Q", _CMD_ECHO)
_SENTINEL_HEAD = 0xD100000000000001
_SENTINEL_TAIL = 0xD100000000000002

SUCCESS_PLATFORMS = ["a2a3sim", "a2a3", "a5sim", "a5"]
FAULT_PLATFORMS = ["a2a3sim", "a5sim"]


def build_chip_callable(platform: str) -> ChipCallable:
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
    try:
        child = CoreCallable.build(signature=[D.IN, D.OUT], binary=aiv)
    except ValueError as exc:
        if "arg_index" not in str(exc):
            raise
        child = CoreCallable.build(signature=[D.IN, D.OUT], arg_index=[0, 1], binary=aiv)
    return ChipCallable.build(
        signature=[],
        func_name="spsc_queue_orchestration",
        binary=orch,
        children=[(0, child)],
    )


def stream_config() -> CallConfig:
    config = CallConfig()
    config.aicpu_thread_num = 2
    return config


def close_owned_workers(primary: BaseException | None, *workers: Any) -> None:
    first_cleanup: BaseException | None = None
    for worker in workers:
        if worker is None:
            continue
        try:
            worker.close()
        except BaseException as cleanup:
            if primary is not None:
                try:
                    attach_exception_note(primary, f"{type(cleanup).__name__}: {cleanup}")
                except BaseException:
                    pass
            elif first_cleanup is None:
                first_cleanup = cleanup
            else:
                try:
                    attach_exception_note(first_cleanup, f"{type(cleanup).__name__}: {cleanup}")
                except BaseException:
                    pass
    if primary is None and first_cleanup is not None:
        raise first_cleanup


def make_l3_worker(platform: str, device_id: int) -> tuple[Worker, Any]:
    worker = Worker(
        level=3,
        device_ids=[int(device_id)],
        num_sub_workers=0,
        platform=platform,
        runtime=_RUNTIME,
    )
    try:
        handle = worker.register(build_chip_callable(platform))
        worker.init()
        return worker, handle
    except BaseException as primary:
        close_owned_workers(primary, worker)
        raise


def region_lifecycle():
    name = "comm_region_recursive_single_owner_helpers"
    module = sys.modules.get(name)
    if module is not None:
        return module
    path = os.path.abspath(os.path.join(_HERE, "..", "..", "recursive_single_owner", "_helpers.py"))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("region lifecycle helpers are missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def create_queue(orch_handle, *, host_path: str = "L3", peer_path: str = "L3/L2[0]") -> SpscQueue:
    host = ce.at(host_path, ce.HOST_CPU)
    device = ce.at(peer_path, ce.DEVICE_AICPU)
    queue = orch_handle.create_spsc_queue(
        initiator=host,
        peer=device,
        depth=_QUEUE_DEPTH,
        input_arena_bytes=_INPUT_ARENA_BYTES,
        output_arena_bytes=_OUTPUT_ARENA_BYTES,
    )
    assert isinstance(queue, SpscQueue)
    assert queue.layout.depth == _QUEUE_DEPTH
    assert queue.layout.input_arena_bytes == _INPUT_ARENA_BYTES
    assert queue.layout.output_arena_bytes == _OUTPUT_ARENA_BYTES
    assert queue.layout.input_arena_bytes != queue.layout.output_arena_bytes
    scalars = queue.peer_task_arg_scalars()
    assert len(scalars) == 10
    binding = _SpscQueueEndpointBinding.from_scalars(scalars)
    assert list(binding.to_scalars()) == scalars
    assert binding.transaction_id != 0
    return queue


def queue_task_args(queue: SpscQueue, *, sentinels: bool = False) -> TaskArgs:
    task_args = TaskArgs()
    if sentinels:
        task_args.add_scalar(_SENTINEL_HEAD)
    for scalar in queue.peer_task_arg_scalars():
        task_args.add_scalar(int(scalar))
    if sentinels:
        task_args.add_scalar(_SENTINEL_TAIL)
    return task_args


def forwarded_task_args(args) -> TaskArgs:
    count = int(args.scalar_count())
    start = 0
    end = count
    if count >= 2 and int(args.scalar(0)) == _SENTINEL_HEAD and int(args.scalar(count - 1)) == _SENTINEL_TAIL:
        start = 1
        end = count - 1
    task_args = TaskArgs()
    for index in range(start, end):
        task_args.add_scalar(int(args.scalar(index)))
    return task_args


def submit_queue(orch_handle, chip_handle, queue: SpscQueue, cfg, *, worker_id: int = 0) -> None:
    orch_handle.submit_next_level(chip_handle, queue_task_args(queue), cfg, worker=int(worker_id))


def sentinel_task_args(middle: int) -> TaskArgs:
    task_args = TaskArgs()
    task_args.add_scalar(_SENTINEL_HEAD)
    task_args.add_scalar(int(middle))
    task_args.add_scalar(_SENTINEL_TAIL)
    return task_args


def assert_driver_sentinels(args, middle: int) -> None:
    assert int(args.scalar_count()) == 3
    assert int(args.scalar(0)) == _SENTINEL_HEAD
    assert int(args.scalar(1)) == int(middle)
    assert int(args.scalar(2)) == _SENTINEL_TAIL


def echo_once(queue: SpscQueue, *, free: bool) -> None:
    enqueue_bytes(queue, echo_payload(b"once"))
    opcode, payload, _offset = drain_message(queue)
    assert opcode is SpscQueueOpcode.DATA
    assert payload == echo_payload(b"once")
    queue.request_stop(timeout=_TIMEOUT_S)
    opcode, payload, _offset = drain_message(queue)
    assert opcode is SpscQueueOpcode.DATA
    assert payload == _POST_STOP_MARKER
    if not free:
        return
    queue.free()
    try:
        queue.input.try_enqueue(_SMALL_ECHO, len(_SMALL_ECHO))
    except RuntimeError as exc:
        if "released" not in str(exc):
            raise
    else:
        raise AssertionError("enqueue after free succeeded")


def enqueue_bytes(queue: SpscQueue, payload: bytes) -> None:
    queue.input.enqueue(payload, nbytes=len(payload), timeout=_TIMEOUT_S)


def drain_message(queue: SpscQueue) -> tuple[SpscQueueOpcode, bytes, int]:
    message = queue.output.peek(timeout=_TIMEOUT_S)
    payload = _read_payload(queue, message)
    offset = int(message.payload_offset)
    opcode = message.opcode
    queue.output.release(message)
    return opcode, payload, offset


def _read_payload(queue: SpscQueue, message) -> bytes:
    if message.payload_nbytes == 0:
        queue.output.read_into(message, None)
        return b""
    buf = bytearray(message.payload_nbytes)
    queue.output.read_into(message, buf)
    return bytes(buf)


def echo_payload(body: bytes) -> bytes:
    return _HEADER.pack(_CMD_ECHO) + body


def error_payload(body: bytes) -> bytes:
    return _HEADER.pack(_CMD_ERROR) + body


def compute_payload(base: float) -> bytes:
    tile = struct.pack(f"<{_TILE_ELEMS}f", *[(base + float(i % _TILE_COLS)) for i in range(_TILE_ELEMS)])
    return _HEADER.pack(_CMD_COMPUTE) + tile


def expected_compute_payload(base: float) -> bytes:
    tile = struct.pack(
        f"<{_TILE_ELEMS}f", *[(base + float(i % _TILE_COLS) + _COMPUTE_SCALAR) for i in range(_TILE_ELEMS)]
    )
    return _HEADER.pack(_CMD_COMPUTE) + tile


def wrap_payload(seed: int) -> bytes:
    body = bytes((seed + i) & 0xFF for i in range(_WRAP_NBYTES - _HEADER.size))
    return echo_payload(body)
