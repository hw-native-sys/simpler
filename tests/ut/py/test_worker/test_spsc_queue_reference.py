# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Example-private SPSC reference tests against a public Region fake."""

from __future__ import annotations

import copy
import ctypes
import pickle
import struct
import sys
from pathlib import Path
from typing import Any

import pytest
from simpler.comm_region import NotifyOp, RegionInstanceState, SignalTestResult, WaitCmp

_EXAMPLE_DIR = Path(__file__).resolve().parents[4] / "examples" / "workers" / "l3" / "worker_chip_message_queue"
sys.path.insert(0, str(_EXAMPLE_DIR))

from spsc_queue import (  # noqa: E402
    DESCRIPTOR_BYTES,
    SpscQueue,
    SpscQueueLayout,
    SpscQueueMessage,
    SpscQueueOpcode,
    _SpscQueueState,
    create_spsc_queue,
    decode_spsc_queue_descriptor,
    encode_spsc_queue_descriptor,
)

_LAYOUT_GOLDEN = (
    (1, 64, 64, 32, 64, 128, 192),
    (4, 128, 192, 128, 256, 384, 576),
    (8, 192, 64, 256, 512, 704, 768),
    (2, 64, 128, 64, 128, 192, 320),
)
_DESCRIPTOR_GOLDEN_FIELDS = (7, 3, 128, 16)
_DESCRIPTOR_GOLDEN_BYTES = bytes(
    [
        0x07,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x03,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x80,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x10,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
    ]
)
_REGION_SCALARS = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]
_DENIED_REGION_ATTRS = (
    "_local_view",
    "_allocation_identity",
    "_consumer",
    "_provider",
    "_ensure_usable",
    "_counter_notify",
    "_counter_test",
    "_counter_wait",
    "_worker",
)


def _require_signed_int32(value: object) -> int:
    if type(value) is not int or value < -2147483648 or value > 2147483647:
        raise OverflowError("signed int32 counter operand out of range")
    return value


def _compare_counter(observed: int, operand: int, cmp: WaitCmp) -> bool:
    if cmp is WaitCmp.EQ:
        return observed == operand
    if cmp is WaitCmp.NE:
        return observed != operand
    if cmp is WaitCmp.GT:
        return observed > operand
    if cmp is WaitCmp.GE:
        return observed >= operand
    if cmp is WaitCmp.LT:
        return observed < operand
    if cmp is WaitCmp.LE:
        return observed <= operand
    return False


class _MemoryCounter:
    def __init__(self, region: _PublicRegion, offset: int) -> None:
        self._region = region
        self._offset = int(offset)

    def test(self, cmp_value: int, cmp: WaitCmp) -> SignalTestResult:
        operand = _require_signed_int32(cmp_value)
        observed = ctypes.c_int32(int(self._region.counters.get(self._offset, 0)) & 0xFFFFFFFF).value
        return SignalTestResult(matched=_compare_counter(observed, operand, WaitCmp(cmp)), observed=observed)

    def wait(self, cmp_value: int, cmp: WaitCmp, timeout: float) -> int:
        operand = _require_signed_int32(cmp_value)
        if timeout is None or float(timeout) <= 0:
            raise ValueError("region counter wait requires a positive timeout")
        if self._region.on_wait is not None:
            self._region.on_wait(self._offset)
        result = self.test(operand, cmp)
        if result.matched:
            return int(result.observed)
        raise TimeoutError(f"queue counter wait timed out; observed={result.observed}")

    def notify(self, value: int, op: NotifyOp = NotifyOp.Set) -> None:
        operand = _require_signed_int32(value)
        if self._region.fail_notify is not None:
            raise self._region.fail_notify
        if op is NotifyOp.Add:
            current = ctypes.c_int32(int(self._region.counters.get(self._offset, 0)) & 0xFFFFFFFF).value
            self._region.counters[self._offset] = ctypes.c_int32((current + operand) & 0xFFFFFFFF).value
        else:
            self._region.counters[self._offset] = operand


class _PublicRegion:
    """Public Region stand-in. Private attribute access fails the reference boundary."""

    def __init__(self, payload_bytes: int, counter_bytes: int, scalars: list[int] | None = None) -> None:
        self.payload = bytearray(int(payload_bytes))
        self.counters: dict[int, int] = {}
        self.scalars = list(_REGION_SCALARS if scalars is None else scalars)
        self.on_wait = None
        self.fail_notify: BaseException | None = None
        self.fail_payload: BaseException | None = None
        self.descriptor_calls = 0
        self.freed = False
        self._state = RegionInstanceState.LIVE

    def __getattribute__(self, name: str) -> Any:
        if name in _DENIED_REGION_ATTRS or (name.startswith("_") and not name.startswith("__")):
            raise AssertionError(f"reference SPSC touched private region attribute {name}")
        return object.__getattribute__(self, name)

    @property
    def state(self) -> RegionInstanceState:
        return object.__getattribute__(self, "_state")

    def payload_write(self, offset: int, host_buffer: Any, nbytes: int | None = None) -> None:
        if self.fail_payload is not None:
            raise self.fail_payload
        view = memoryview(host_buffer)
        size = int(view.nbytes if nbytes is None else nbytes)
        self.payload[int(offset) : int(offset) + size] = bytes(view[:size])

    def payload_read(self, offset: int, host_buffer: Any, nbytes: int | None = None) -> None:
        if self.fail_payload is not None:
            raise self.fail_payload
        view = memoryview(host_buffer)
        size = int(view.nbytes if nbytes is None else nbytes)
        view[:size] = bytes(self.payload[int(offset) : int(offset) + size])

    def counter(self, offset: int) -> _MemoryCounter:
        return _MemoryCounter(self, offset)

    def descriptor_scalars(self) -> list[int]:
        self.descriptor_calls += 1
        return list(self.scalars)

    def free(self) -> None:
        self.freed = True


def _layout(depth: int, input_arena_bytes: int, output_arena_bytes: int) -> SpscQueueLayout:
    return SpscQueueLayout.create(
        depth=depth, input_arena_bytes=input_arena_bytes, output_arena_bytes=output_arena_bytes
    )


def _queue(depth: int = 4, input_arena_bytes: int = 128, output_arena_bytes: int = 192, **region_kwargs: Any):
    layout = _layout(depth, input_arena_bytes, output_arena_bytes)
    region = _PublicRegion(layout.payload_bytes, layout.counter_bytes)
    for name, value in region_kwargs.items():
        setattr(region, name, value)
    return region, SpscQueue(region, layout)


def _plant_output(region: _PublicRegion, queue: SpscQueue, seq: int, opcode: SpscQueueOpcode, payload: bytes) -> int:
    layout = queue.layout
    if payload:
        offset = layout.output_arena_offset
        region.payload[offset : offset + len(payload)] = payload
    else:
        offset = 0
    slot_index = (int(seq) - 1) & (layout.depth - 1)
    slot = layout.output_desc_offset + slot_index * DESCRIPTOR_BYTES
    region.payload[slot : slot + DESCRIPTOR_BYTES] = encode_spsc_queue_descriptor(seq, opcode, offset, len(payload))
    region.counters[layout.output_desc_tail_offset] = int(seq)
    return offset


@pytest.mark.parametrize(
    ("depth", "input_arena_bytes", "output_arena_bytes", "output_desc", "input_arena", "output_arena", "payload"),
    _LAYOUT_GOLDEN,
)
def test_layout_golden_vectors(
    depth, input_arena_bytes, output_arena_bytes, output_desc, input_arena, output_arena, payload
):
    layout = _layout(depth, input_arena_bytes, output_arena_bytes)
    assert layout.input_desc_offset == 0
    assert layout.output_desc_offset == output_desc
    assert layout.input_arena_offset == input_arena
    assert layout.output_arena_offset == output_arena
    assert layout.payload_bytes == payload
    assert layout.counter_bytes == 384
    assert layout.input_desc_tail_offset == 0
    assert layout.input_desc_head_offset == 64
    assert layout.output_desc_tail_offset == 128
    assert layout.output_desc_head_offset == 192
    assert layout.initiator_abort_offset == 256
    assert layout.peer_abort_offset == 320


def test_layout_max_depth_and_invalid_values():
    depth = 1 << 30
    layout = _layout(depth, 64, 64)
    assert layout.output_desc_offset == depth * 32
    assert layout.payload_bytes == depth * 64 + 128
    for args in ((3, 64, 64), (0, 64, 64), (2, 65, 64), (2, 64, 63), (True, 64, 64)):
        with pytest.raises((TypeError, ValueError)):
            _layout(*args)
    with pytest.raises(ValueError, match="overflowed uint64"):
        _layout(2, (1 << 64) - 64, 64)


def test_opcode_and_descriptor_golden_vector():
    assert [int(item) for item in SpscQueueOpcode] == [0, 1, 2, 3]
    encoded = encode_spsc_queue_descriptor(
        *_DESCRIPTOR_GOLDEN_FIELDS[:1], SpscQueueOpcode.ERROR, *_DESCRIPTOR_GOLDEN_FIELDS[2:]
    )
    assert encoded == _DESCRIPTOR_GOLDEN_BYTES
    assert decode_spsc_queue_descriptor(encoded) == _DESCRIPTOR_GOLDEN_FIELDS
    assert struct.calcsize("<QQQQ") == DESCRIPTOR_BYTES
    with pytest.raises(ValueError):
        decode_spsc_queue_descriptor(_DESCRIPTOR_GOLDEN_BYTES[:-1])


def test_peer_task_args_forward_region_descriptor_then_config():
    region, queue = _queue(depth=4, input_arena_bytes=128, output_arena_bytes=192)
    assert region.descriptor_calls == 0
    scalars = queue.peer_task_arg_scalars()
    assert scalars[:8] == _REGION_SCALARS
    assert scalars[8:] == [4, 128, 192]
    assert len(scalars) == 11
    assert region.descriptor_calls == 1
    scalars[0] = 0
    assert region.descriptor_scalars() == _REGION_SCALARS


def test_create_helper_uses_only_public_create_region():
    seen: dict[str, Any] = {}

    class _Orch:
        def create_region(self, *, members, topology, payload_bytes, counter_bytes):
            seen["members"] = members
            seen["provider"] = topology.provider
            seen["payload_bytes"] = payload_bytes
            seen["counter_bytes"] = counter_bytes
            return _PublicRegion(payload_bytes, counter_bytes)

        def __getattr__(self, name: str):
            raise AssertionError(f"reference helper touched orchestrator.{name}")

    initiator = object()
    peer = object()
    queue = create_spsc_queue(
        _Orch(), initiator=initiator, peer=peer, depth=4, input_arena_bytes=128, output_arena_bytes=192
    )
    assert seen["members"] == (initiator, peer)
    assert seen["provider"] is peer
    assert seen["payload_bytes"] == queue.layout.payload_bytes
    assert seen["counter_bytes"] == queue.layout.counter_bytes
    assert queue.peer_task_arg_scalars()[8:] == [4, 128, 192]


def test_queue_refuses_copy_and_logical_free_keeps_region_live():
    region, queue = _queue()
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(queue)
    with pytest.raises(TypeError, match="cannot be copied"):
        pickle.dumps(queue)
    queue.free()
    queue.free()
    assert queue._state is _SpscQueueState.RELEASED
    assert region.state is RegionInstanceState.LIVE
    assert region.freed is False
    with pytest.raises(RuntimeError, match="released"):
        queue.input.try_enqueue(b"abc", 3)


def test_try_enqueue_oversize_does_not_touch_shared_state():
    region, queue = _queue(input_arena_bytes=64, output_arena_bytes=64)
    assert queue.input.try_enqueue(b"x" * 128, 128) is False
    assert region.counters == {}
    assert bytes(region.payload) == bytes(len(region.payload))
    with pytest.raises(ValueError, match="exceeds input arena"):
        queue.input.enqueue(b"x" * 128, 128, timeout=0.1)


def test_duplex_wrap_stop_and_error_use_public_region_ops():
    region, queue = _queue(depth=4, input_arena_bytes=64, output_arena_bytes=64)
    assert queue.input.try_enqueue(None, 0) is True
    assert queue.input.try_enqueue(b"x" * 48, 48) is True
    region.counters[queue.layout.input_desc_head_offset] = 2
    assert queue.input.try_enqueue(b"y" * 32, 32) is True
    slot = queue.layout.input_desc_offset + 2 * DESCRIPTOR_BYTES
    seq, opcode, offset, nbytes = decode_spsc_queue_descriptor(bytes(region.payload[slot : slot + DESCRIPTOR_BYTES]))
    assert (seq, opcode, nbytes, offset) == (3, int(SpscQueueOpcode.DATA), 32, queue.layout.input_arena_offset)
    assert queue.try_request_stop() is True
    assert queue.input.try_enqueue(b"no", 2) is False

    out_offset = _plant_output(region, queue, 1, SpscQueueOpcode.ERROR, b"err-data")
    handle = queue.output.try_peek()
    assert handle is not None
    assert handle.opcode is SpscQueueOpcode.ERROR
    dest = bytearray(8)
    queue.output.read_into(handle, dest)
    assert bytes(dest) == b"err-data"
    queue.output.release(handle)
    assert out_offset == queue.layout.output_arena_offset


def test_output_ownership_timeout_and_abort_notify_note():
    region, queue = _queue()
    with pytest.raises(TimeoutError, match="timed out"):
        queue.output.peek(0.01)
    assert queue._state is _SpscQueueState.LIVE

    def set_abort(_offset: int) -> None:
        region.counters[queue.layout.peer_abort_offset] = 1

    region.on_wait = set_abort
    with pytest.raises(RuntimeError, match="remote abort"):
        queue.output.peek(0.01)
    assert queue._state is _SpscQueueState.POISONED_REMOTE

    region, queue = _queue()
    _plant_output(region, queue, 1, SpscQueueOpcode.DATA, b"abc")
    handle = queue.output.try_peek()
    forged = SpscQueueMessage(handle.seq, handle.opcode, handle.payload_offset, handle.payload_nbytes)
    with pytest.raises(RuntimeError, match="not active"):
        queue.output.release(forged)
    assert queue._state is _SpscQueueState.POISONED_LOCAL
    assert region.counters[queue.layout.initiator_abort_offset] == 1

    region, queue = _queue()
    region.fail_payload = RuntimeError("payload write failed")
    region.fail_notify = RuntimeError("abort notify failed")
    with pytest.raises(RuntimeError, match="payload write failed") as excinfo:
        queue.input.try_enqueue(b"hi", 2)
    assert queue._state is _SpscQueueState.POISONED_LOCAL
    assert any("abort notify failed" in note for note in getattr(excinfo.value, "__notes__", []))


def test_zero_byte_enqueue_and_exact_depth_publish_order():
    _region, queue = _queue()
    with pytest.raises(ValueError, match="buffer == None"):
        queue.input.try_enqueue(b"", 0)
    with pytest.raises(ValueError, match="STOP must be zero-byte"):
        queue.input._try_enqueue(None, 4, SpscQueueOpcode.STOP)

    region, queue = _queue(depth=1, input_arena_bytes=64, output_arena_bytes=64)
    writes: list[int] = []
    original = region.payload_write

    def record_write(offset, host_buffer, nbytes=None):
        writes.append(int(offset))
        return original(offset, host_buffer, nbytes)

    region.payload_write = record_write
    assert queue.input.try_enqueue(b"abcd", 4) is True
    assert writes[0] == queue.layout.input_arena_offset
    assert writes[1] == queue.layout.input_desc_offset + 8
    assert writes[2] == queue.layout.input_desc_offset
    assert queue.input.try_enqueue(b"efgh", 4) is False

    def advance_head(_offset: int) -> None:
        region.counters[queue.layout.input_desc_head_offset] = 1

    region.on_wait = advance_head
    queue.input.enqueue(b"ijkl", 4, timeout=1.0)
    assert queue._input_tail == 2


def test_closed_region_expires_without_touching_counters():
    region, queue = _queue()
    region._state = RegionInstanceState.CLOSED
    with pytest.raises(RuntimeError, match="expired"):
        queue.input.try_enqueue(b"ab", 2)
    assert queue._state is _SpscQueueState.EXPIRED
    assert region.counters == {}


def test_output_early_wrap_poisons_once():
    region, queue = _queue(depth=4, input_arena_bytes=64, output_arena_bytes=64)
    out = queue.layout.output_arena_offset
    _plant_output(region, queue, 1, SpscQueueOpcode.DATA, b"abcdefgh")
    handle = queue.output.try_peek()
    queue.output.release(handle)
    region.payload[out : out + 8] = b"xxxxxxxx"
    slot = queue.layout.output_desc_offset + DESCRIPTOR_BYTES
    region.payload[slot : slot + DESCRIPTOR_BYTES] = encode_spsc_queue_descriptor(2, SpscQueueOpcode.DATA, out, 8)
    region.counters[queue.layout.output_desc_tail_offset] = 2
    with pytest.raises(RuntimeError, match="payload replay offset mismatch"):
        queue.output.try_peek()
    assert queue._output_head == 1
    assert region.counters[queue.layout.initiator_abort_offset] == 1
    with pytest.raises(RuntimeError, match="payload replay offset mismatch"):
        queue.output.try_peek()
    assert region.counters[queue.layout.initiator_abort_offset] == 1


def test_free_rejects_acquired_message_without_region_free():
    region, queue = _queue(depth=4, input_arena_bytes=64, output_arena_bytes=64)
    _plant_output(region, queue, 1, SpscQueueOpcode.DATA, b"abcd")
    handle = queue.output.try_peek()
    queue.free()
    with pytest.raises(RuntimeError, match="released"):
        queue.output.read_into(handle, bytearray(4))
    with pytest.raises(RuntimeError, match="released"):
        queue.output.release(handle)
    assert region.state is RegionInstanceState.LIVE
    assert region.freed is False
