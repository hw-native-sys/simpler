# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Example-private duplex SPSC queue on the public Region API."""

from __future__ import annotations

import ctypes
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable, NoReturn

import simpler.comm_endpoints as ce
from simpler.comm_region import NotifyOp, RegionInstanceState, WaitCmp

_UINT64_MAX = (1 << 64) - 1
DESCRIPTOR_BYTES = 32
_ARENA_ALIGNMENT = 64
_DESCRIPTOR_RING_ALIGNMENT = 8
_COUNTER_BYTES = 384
_MAX_DEPTH = 1 << 30
_INPUT_DESC_TAIL_OFFSET = 0
_INPUT_DESC_HEAD_OFFSET = 64
_OUTPUT_DESC_TAIL_OFFSET = 128
_OUTPUT_DESC_HEAD_OFFSET = 192
_INITIATOR_ABORT_OFFSET = 256
_PEER_ABORT_OFFSET = 320

_DESCRIPTOR_STRUCT = struct.Struct("<QQQQ")
_DESCRIPTOR_FIELDS_STRUCT = struct.Struct("<QQQ")


class SpscQueueOpcode(IntEnum):
    INVALID = 0
    DATA = 1
    STOP = 2
    ERROR = 3


class _SpscQueueState(Enum):
    LIVE = "live"
    RELEASED = "released"
    POISONED_LOCAL = "poisoned_local"
    POISONED_REMOTE = "poisoned_remote"
    EXPIRED = "expired"


def _counter_low32(value: int) -> int:
    return ctypes.c_int32(int(value) & 0xFFFFFFFF).value


def _payload_expected_offset(cursor: int, nbytes: int, arena_offset: int, arena_bytes: int) -> int:
    arena_pos = int(cursor) % int(arena_bytes)
    if arena_pos + int(nbytes) > int(arena_bytes):
        return int(arena_offset)
    return int(arena_offset) + arena_pos


def _require_exact_u64(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    if value < 0 or value > _UINT64_MAX:
        raise ValueError(f"{name} overflowed uint64")
    return value


def _checked_add_u64(lhs: int, rhs: int) -> int:
    result = lhs + rhs
    if lhs < 0 or rhs < 0 or result > _UINT64_MAX:
        raise ValueError("SPSC queue layout calculation overflowed uint64")
    return result


def _checked_mul_u64(lhs: int, rhs: int) -> int:
    if lhs < 0 or rhs < 0:
        raise ValueError("SPSC queue layout calculation overflowed uint64")
    result = lhs * rhs
    if result > _UINT64_MAX:
        raise ValueError("SPSC queue layout calculation overflowed uint64")
    return result


def _align_up_u64(value: int, align: int) -> int:
    if align <= 0:
        raise ValueError("SPSC queue layout calculation overflowed uint64")
    remainder = value % align
    bump = 0 if remainder == 0 else align - remainder
    return _checked_add_u64(value, bump)


def encode_spsc_queue_descriptor(
    seq: object,
    opcode: object,
    payload_offset: object,
    payload_nbytes: object,
) -> bytes:
    opcode_value = int(opcode) if isinstance(opcode, SpscQueueOpcode) else opcode
    return _DESCRIPTOR_STRUCT.pack(
        _require_exact_u64("seq", seq),
        _require_exact_u64("opcode", opcode_value),
        _require_exact_u64("payload_offset", payload_offset),
        _require_exact_u64("payload_nbytes", payload_nbytes),
    )


def decode_spsc_queue_descriptor(data: bytes) -> tuple[int, int, int, int]:
    raw = bytes(data)
    if len(raw) != DESCRIPTOR_BYTES:
        raise ValueError("descriptor requires exactly 32 bytes")
    seq, opcode, payload_offset, payload_nbytes = _DESCRIPTOR_STRUCT.unpack(raw)
    return int(seq), int(opcode), int(payload_offset), int(payload_nbytes)


def _reject_copy(obj: object) -> NoReturn:
    raise TypeError(f"{type(obj).__name__} cannot be copied")


def _attach_exception_note(error: BaseException, note: str) -> None:
    adder = getattr(error, "add_note", None)
    if callable(adder):
        adder(note)
        return
    notes = getattr(error, "__notes__", None)
    if isinstance(notes, list):
        notes.append(note)
        return
    object.__setattr__(error, "__notes__", [note])


def _admit_bytes_like(obj: object, nbytes: int, *, writable: bool) -> object:
    try:
        view = memoryview(obj)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("SPSC queue requires a contiguous host buffer") from exc
    if not view.c_contiguous:
        raise ValueError("SPSC queue ordinary host buffer must be C-contiguous")
    if writable and view.readonly:
        raise ValueError("SPSC queue output target must be writable")
    if int(view.nbytes) < int(nbytes):
        raise ValueError(f"SPSC queue nbytes={int(nbytes)} exceeds ordinary host buffer size {int(view.nbytes)}")
    return obj


def _admit_spsc_payload(obj: object, nbytes: int, *, writable: bool, allow_none: bool) -> object:
    if int(nbytes) == 0 and allow_none:
        if obj is not None:
            raise ValueError("SPSC queue zero-byte payload requires buffer == None")
        return None
    if obj is None:
        raise ValueError("SPSC queue nonzero payload requires a host buffer")
    try:
        return _admit_bytes_like(obj, nbytes, writable=writable)
    except ValueError:
        if hasattr(obj, "nbytes") and hasattr(obj, "base"):
            available = int(obj.nbytes)
            if available < int(nbytes):
                raise ValueError(f"SPSC queue nbytes={nbytes} exceeds registered buffer size {available}") from None
            return obj
        raise


def _admit_spsc_payload_kind(obj: object, *, writable: bool) -> None:
    if obj is None:
        return
    _admit_spsc_payload(obj, 0, writable=writable, allow_none=False)


@dataclass(frozen=True)
class SpscQueueLayout:
    depth: int
    input_arena_bytes: int
    output_arena_bytes: int
    input_desc_offset: int
    output_desc_offset: int
    input_arena_offset: int
    output_arena_offset: int
    payload_bytes: int
    input_desc_tail_offset: int
    input_desc_head_offset: int
    output_desc_tail_offset: int
    output_desc_head_offset: int
    initiator_abort_offset: int
    peer_abort_offset: int
    counter_bytes: int

    @classmethod
    def create(cls, *, depth: object, input_arena_bytes: object, output_arena_bytes: object) -> SpscQueueLayout:
        depth_value = _require_exact_u64("depth", depth)
        input_bytes = _require_exact_u64("input_arena_bytes", input_arena_bytes)
        output_bytes = _require_exact_u64("output_arena_bytes", output_arena_bytes)
        if depth_value < 1 or depth_value & (depth_value - 1) != 0 or depth_value > _MAX_DEPTH:
            raise ValueError("depth must be a power of two and <= 2^30")
        if input_bytes < 1 or input_bytes % _ARENA_ALIGNMENT != 0:
            raise ValueError("input_arena_bytes must be a positive 64-byte multiple")
        if output_bytes < 1 or output_bytes % _ARENA_ALIGNMENT != 0:
            raise ValueError("output_arena_bytes must be a positive 64-byte multiple")

        desc_ring_bytes = _checked_mul_u64(depth_value, DESCRIPTOR_BYTES)
        input_desc_offset = 0
        output_desc_offset = _checked_add_u64(input_desc_offset, desc_ring_bytes)
        desc_end = _checked_add_u64(output_desc_offset, desc_ring_bytes)
        input_arena_offset = _align_up_u64(desc_end, _ARENA_ALIGNMENT)
        input_arena_end = _checked_add_u64(input_arena_offset, input_bytes)
        output_arena_offset = _align_up_u64(input_arena_end, _ARENA_ALIGNMENT)
        payload_bytes = _checked_add_u64(output_arena_offset, output_bytes)
        if (
            output_desc_offset % _DESCRIPTOR_RING_ALIGNMENT != 0
            or input_arena_offset % _ARENA_ALIGNMENT != 0
            or output_arena_offset % _ARENA_ALIGNMENT != 0
        ):
            raise ValueError("SPSC queue layout alignment is invalid")
        return cls(
            depth=depth_value,
            input_arena_bytes=input_bytes,
            output_arena_bytes=output_bytes,
            input_desc_offset=input_desc_offset,
            output_desc_offset=output_desc_offset,
            input_arena_offset=input_arena_offset,
            output_arena_offset=output_arena_offset,
            payload_bytes=payload_bytes,
            input_desc_tail_offset=_INPUT_DESC_TAIL_OFFSET,
            input_desc_head_offset=_INPUT_DESC_HEAD_OFFSET,
            output_desc_tail_offset=_OUTPUT_DESC_TAIL_OFFSET,
            output_desc_head_offset=_OUTPUT_DESC_HEAD_OFFSET,
            initiator_abort_offset=_INITIATOR_ABORT_OFFSET,
            peer_abort_offset=_PEER_ABORT_OFFSET,
            counter_bytes=_COUNTER_BYTES,
        )


@dataclass(frozen=True)
class SpscQueueMessage:
    seq: int
    opcode: SpscQueueOpcode
    payload_offset: int
    payload_nbytes: int
    _owner_token: object | None = field(default=None, repr=False, compare=False)


class SpscQueue:
    def __init__(self, region: object, layout: SpscQueueLayout) -> None:
        if not isinstance(layout, SpscQueueLayout):
            raise TypeError("SPSC queue requires an SpscQueueLayout")
        self._region = region
        self._layout = layout
        self._state = _SpscQueueState.LIVE
        self._first_error: BaseException | None = None
        self._owner_token = object()
        self._input_head = 0
        self._input_tail = 0
        self._input_payload_head = 0
        self._input_payload_tail = 0
        self._input_stop_published = False
        self._output_head = 0
        self._output_tail = 0
        self._output_payload_head = 0
        self._output_active: SpscQueueMessage | None = None
        self._desc_fields = bytearray(24)
        self._desc_seq = bytearray(8)
        self._desc_read = bytearray(DESCRIPTOR_BYTES)
        self.input = _SpscQueueInitiatorInput(self)
        self.output = _SpscQueueInitiatorOutput(self)

    @property
    def layout(self) -> SpscQueueLayout:
        return self._layout

    def peer_task_arg_scalars(self) -> list[int]:
        self._ensure_usable()
        return [
            *self._region.descriptor_scalars(),
            self._layout.depth,
            self._layout.input_arena_bytes,
            self._layout.output_arena_bytes,
        ]

    def try_request_stop(self) -> bool:
        return self.input._try_enqueue(None, 0, SpscQueueOpcode.STOP)

    def request_stop(self, timeout: float) -> None:
        self.input._enqueue(None, 0, SpscQueueOpcode.STOP, timeout)

    def free(self) -> None:
        if self._state is _SpscQueueState.RELEASED:
            return
        self._state = _SpscQueueState.RELEASED

    def _ensure_usable(self) -> None:
        if self._state is _SpscQueueState.RELEASED:
            raise RuntimeError("SPSC queue has been released")
        if self._state is _SpscQueueState.POISONED_REMOTE:
            raise RuntimeError("SPSC queue is remote-aborted")
        if self._state is _SpscQueueState.POISONED_LOCAL:
            if self._first_error is not None:
                raise self._first_error
            raise RuntimeError("SPSC queue is poisoned")
        if self._state is _SpscQueueState.EXPIRED:
            raise RuntimeError("SPSC queue expired after the region instance left LIVE")
        try:
            live = self._region.state is RegionInstanceState.LIVE
        except BaseException:
            live = False
        if not live:
            self._state = _SpscQueueState.EXPIRED
            raise RuntimeError("SPSC queue expired after the region instance left LIVE")

    def _poison_local(self, error: BaseException) -> None:
        if self._state is not _SpscQueueState.LIVE:
            return
        self._first_error = error
        self._state = _SpscQueueState.POISONED_LOCAL
        try:
            self._region.counter(self._layout.initiator_abort_offset).notify(1, NotifyOp.Set)
        except BaseException as abort_error:
            _attach_exception_note(error, f"initiator abort notify failed: {abort_error}")

    def _run_primitive(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._ensure_usable()
        try:
            return fn(*args, **kwargs)
        except (TimeoutError, ValueError, TypeError):
            raise
        except BaseException as exc:
            self._poison_local(exc)
            raise

    def _signal_test(self, offset: int, cmp_value: int, cmp: WaitCmp) -> Any:
        return self._run_primitive(lambda: self._region.counter(offset).test(_counter_low32(cmp_value), cmp))

    def _signal_notify(self, offset: int, value: int) -> None:
        self._run_primitive(lambda: self._region.counter(offset).notify(_counter_low32(value), NotifyOp.Set))

    def _sample_peer_abort(self) -> None:
        result = self._signal_test(self._layout.peer_abort_offset, 1, WaitCmp.GE)
        if result.matched:
            self._state = _SpscQueueState.POISONED_REMOTE
            raise RuntimeError("SPSC queue remote abort observed")

    def _refresh_counter(self, offset: int, local_value: int) -> int:
        result = self._signal_test(offset, local_value, WaitCmp.NE)
        if not result.matched:
            return local_value
        observed = int(result.observed) & 0xFFFF_FFFF
        local_low = local_value & 0xFFFF_FFFF
        delta = ctypes.c_int32((observed - local_low) & 0xFFFF_FFFF).value
        if delta < 0 or delta > int(self._layout.depth):
            error = RuntimeError("SPSC queue counter reconstruction failed")
            self._poison_local(error)
            raise error
        return local_value + delta

    def _wait_remaining_seconds(self, deadline_ns: int) -> float:
        remaining_ns = int(deadline_ns) - time.monotonic_ns()
        if remaining_ns <= 0:
            return 0.0
        return remaining_ns / 1_000_000_000

    def _wait_progress(self, offset: int, local_value: int, deadline_ns: int) -> None:
        self._sample_peer_abort()
        remaining = self._wait_remaining_seconds(deadline_ns)
        if remaining <= 0:
            self._sample_peer_abort()
            raise TimeoutError("SPSC queue operation timed out")
        try:
            self._run_primitive(
                lambda: self._region.counter(offset).wait(_counter_low32(local_value), WaitCmp.NE, remaining)
            )
        except TimeoutError:
            self._sample_peer_abort()
            raise TimeoutError("SPSC queue operation timed out") from None
        self._sample_peer_abort()

    def _blocking_deadline(self, timeout: float) -> int:
        if timeout is None or float(timeout) <= 0:
            raise ValueError("SPSC queue blocking operations require a positive timeout")
        return time.monotonic_ns() + int(float(timeout) * 1_000_000_000)

    def _write_descriptor(
        self, offset: int, seq: int, opcode: SpscQueueOpcode, payload_offset: int, nbytes: int
    ) -> None:
        self._desc_fields[:] = _DESCRIPTOR_FIELDS_STRUCT.pack(int(opcode), int(payload_offset), int(nbytes))
        self._desc_seq[:] = struct.pack("<Q", int(seq))
        self._run_primitive(self._region.payload_write, offset + 8, self._desc_fields, 24)
        self._run_primitive(self._region.payload_write, offset, self._desc_seq, 8)

    def _read_descriptor(self, offset: int) -> SpscQueueMessage:
        self._run_primitive(self._region.payload_read, offset, self._desc_read, DESCRIPTOR_BYTES)
        seq, opcode_value, payload_offset, payload_nbytes = decode_spsc_queue_descriptor(bytes(self._desc_read))
        try:
            opcode = SpscQueueOpcode(opcode_value)
        except ValueError:
            error = RuntimeError("SPSC queue observed invalid descriptor opcode")
            self._poison_local(error)
            raise error from None
        return SpscQueueMessage(
            seq=int(seq),
            opcode=opcode,
            payload_offset=int(payload_offset),
            payload_nbytes=int(payload_nbytes),
        )

    def _advance_payload_head(
        self,
        cursor: int,
        payload_offset: int,
        payload_nbytes: int,
        arena_offset: int,
        arena_bytes: int,
    ) -> int:
        if payload_nbytes == 0:
            return cursor
        expected_offset = _payload_expected_offset(cursor, payload_nbytes, arena_offset, arena_bytes)
        if expected_offset != payload_offset:
            error = RuntimeError("SPSC queue payload replay offset mismatch")
            self._poison_local(error)
            raise error
        arena_pos = cursor % arena_bytes
        if arena_pos + payload_nbytes > arena_bytes:
            cursor += arena_bytes - arena_pos
        return cursor + payload_nbytes

    def _replay_released_input_descriptors(self, old_head: int, new_head: int) -> None:
        cursor = old_head
        while cursor < new_head:
            slot_index = cursor & (self._layout.depth - 1)
            slot_offset = self._layout.input_desc_offset + slot_index * DESCRIPTOR_BYTES
            message = self._read_descriptor(slot_offset)
            if message.seq != cursor + 1:
                error = RuntimeError("SPSC queue input release replay seq mismatch")
                self._poison_local(error)
                raise error
            self._input_payload_head = self._advance_payload_head(
                self._input_payload_head,
                message.payload_offset,
                message.payload_nbytes,
                self._layout.input_arena_offset,
                self._layout.input_arena_bytes,
            )
            cursor += 1

    def _reserve_input_payload(self, nbytes: int, next_payload_tail: int) -> tuple[int, int] | None:
        arena_bytes = self._layout.input_arena_bytes
        expected_offset = _payload_expected_offset(
            next_payload_tail, nbytes, self._layout.input_arena_offset, arena_bytes
        )
        arena_pos = next_payload_tail % arena_bytes
        if arena_pos + nbytes > arena_bytes:
            next_payload_tail += arena_bytes - arena_pos
        if next_payload_tail + nbytes - self._input_payload_head > arena_bytes:
            return None
        return expected_offset, next_payload_tail

    def __copy__(self) -> SpscQueue:
        _reject_copy(self)

    def __deepcopy__(self, memo: dict[int, object]) -> SpscQueue:
        _reject_copy(self)

    def __reduce__(self) -> tuple[object, ...]:
        _reject_copy(self)

    def __getstate__(self) -> object:
        _reject_copy(self)


class _SpscQueueInitiatorInput:
    def __init__(self, queue: SpscQueue) -> None:
        self._queue = queue

    def try_enqueue(self, buffer_or_none: object, nbytes: int) -> bool:
        queue = self._queue
        queue._ensure_usable()
        admitted = _admit_spsc_payload(buffer_or_none, int(nbytes), writable=False, allow_none=True)
        return self._try_enqueue(admitted, nbytes, SpscQueueOpcode.DATA)

    def enqueue(self, buffer_or_none: object, nbytes: int, timeout: float) -> None:
        queue = self._queue
        queue._ensure_usable()
        admitted = _admit_spsc_payload(buffer_or_none, int(nbytes), writable=False, allow_none=True)
        self._enqueue(admitted, nbytes, SpscQueueOpcode.DATA, timeout)

    def _enqueue(self, buffer_or_none: object, nbytes: int, opcode: SpscQueueOpcode, timeout: float) -> None:
        nbytes = int(nbytes)
        if nbytes > int(self._queue._layout.input_arena_bytes):
            raise ValueError("SPSC queue payload exceeds input arena capacity")
        deadline_ns = self._queue._blocking_deadline(timeout)
        while True:
            if self._try_enqueue(buffer_or_none, nbytes, opcode):
                return
            if self._queue._input_stop_published:
                raise RuntimeError("SPSC queue input is stopped")
            self._queue._wait_progress(self._queue._layout.input_desc_head_offset, self._queue._input_head, deadline_ns)

    def _try_enqueue(self, buffer_or_none: object, nbytes: int, opcode: SpscQueueOpcode) -> bool:
        queue = self._queue
        nbytes = int(nbytes)
        if nbytes < 0:
            raise ValueError("SPSC queue nbytes must be non-negative")
        try:
            opcode = SpscQueueOpcode(opcode)
        except ValueError as exc:
            raise ValueError("SPSC queue input opcode must be DATA or STOP") from exc
        if opcode not in (SpscQueueOpcode.DATA, SpscQueueOpcode.STOP):
            raise ValueError("SPSC queue input opcode must be DATA or STOP")
        if opcode is SpscQueueOpcode.STOP and nbytes != 0:
            raise ValueError("SPSC queue STOP must be zero-byte")
        if nbytes == 0:
            if buffer_or_none is not None:
                raise ValueError("SPSC queue zero-byte enqueue requires buffer_or_none == None")
        elif buffer_or_none is None:
            raise ValueError("SPSC queue nonzero enqueue requires a host buffer")
        queue._ensure_usable()
        if nbytes > int(queue._layout.input_arena_bytes):
            return False
        if queue._input_stop_published:
            return False
        queue._sample_peer_abort()
        if nbytes != 0:
            self._require_enqueue_source(buffer_or_none, nbytes)
        old_head = queue._input_head
        queue._input_head = queue._refresh_counter(queue._layout.input_desc_head_offset, queue._input_head)
        if queue._input_head != old_head:
            queue._replay_released_input_descriptors(old_head, queue._input_head)
        if queue._input_tail - queue._input_head >= queue._layout.depth:
            return False
        payload_offset = 0
        next_payload_tail = queue._input_payload_tail
        if nbytes != 0:
            reserved = queue._reserve_input_payload(nbytes, next_payload_tail)
            if reserved is None:
                return False
            payload_offset, next_payload_tail = reserved
            queue._run_primitive(queue._region.payload_write, payload_offset, buffer_or_none, nbytes)
        seq = queue._input_tail + 1
        slot_index = queue._input_tail & (queue._layout.depth - 1)
        slot_offset = queue._layout.input_desc_offset + slot_index * DESCRIPTOR_BYTES
        queue._write_descriptor(slot_offset, seq, opcode, payload_offset, nbytes)
        queue._input_tail += 1
        queue._input_payload_tail = next_payload_tail + nbytes
        queue._signal_notify(queue._layout.input_desc_tail_offset, queue._input_tail)
        if opcode is SpscQueueOpcode.STOP:
            queue._input_stop_published = True
        return True

    def _require_enqueue_source(self, buffer_or_none: Any, nbytes: int) -> None:
        try:
            view = memoryview(buffer_or_none)
        except TypeError as exc:
            if hasattr(buffer_or_none, "nbytes") and hasattr(buffer_or_none, "base"):
                available = int(buffer_or_none.nbytes)
                if available < int(nbytes):
                    raise ValueError(f"SPSC queue nbytes={nbytes} exceeds registered buffer size {available}") from None
                return
            raise ValueError("SPSC queue requires a contiguous host buffer") from exc
        if not view.c_contiguous:
            raise ValueError("SPSC queue ordinary host buffer must be C-contiguous")
        if int(view.nbytes) < int(nbytes):
            raise ValueError(f"SPSC queue nbytes={nbytes} exceeds ordinary host buffer size {int(view.nbytes)}")


class _SpscQueueInitiatorOutput:
    def __init__(self, queue: SpscQueue) -> None:
        self._queue = queue

    def try_peek(self) -> SpscQueueMessage | None:
        queue = self._queue
        queue._ensure_usable()
        queue._sample_peer_abort()
        if queue._output_active is not None:
            return queue._output_active
        queue._output_tail = queue._refresh_counter(queue._layout.output_desc_tail_offset, queue._output_tail)
        if queue._output_tail == queue._output_head:
            return None
        slot_index = queue._output_head & (queue._layout.depth - 1)
        slot_offset = queue._layout.output_desc_offset + slot_index * DESCRIPTOR_BYTES
        message = queue._read_descriptor(slot_offset)
        if message.seq != queue._output_head + 1:
            error = RuntimeError("SPSC queue output descriptor seq mismatch")
            queue._poison_local(error)
            raise error
        if message.opcode not in (SpscQueueOpcode.DATA, SpscQueueOpcode.ERROR):
            error = RuntimeError("SPSC queue output descriptor must be DATA or ERROR")
            queue._poison_local(error)
            raise error
        if message.payload_nbytes == 0:
            if message.payload_offset != 0:
                error = RuntimeError("SPSC queue zero-byte output descriptor has nonzero offset")
                queue._poison_local(error)
                raise error
        else:
            begin = queue._layout.output_arena_offset
            end = begin + queue._layout.output_arena_bytes
            if message.payload_offset < begin or message.payload_offset + message.payload_nbytes > end:
                error = RuntimeError("SPSC queue output payload outside output arena")
                queue._poison_local(error)
                raise error
            queue._advance_payload_head(
                queue._output_payload_head,
                message.payload_offset,
                message.payload_nbytes,
                queue._layout.output_arena_offset,
                queue._layout.output_arena_bytes,
            )
        owned = SpscQueueMessage(
            seq=message.seq,
            opcode=message.opcode,
            payload_offset=message.payload_offset,
            payload_nbytes=message.payload_nbytes,
            _owner_token=queue._owner_token,
        )
        queue._output_active = owned
        return owned

    def peek(self, timeout: float) -> SpscQueueMessage:
        queue = self._queue
        deadline_ns = queue._blocking_deadline(timeout)
        while True:
            message = self.try_peek()
            if message is not None:
                return message
            queue._wait_progress(queue._layout.output_desc_tail_offset, queue._output_tail, deadline_ns)

    def read_into(self, handle: SpscQueueMessage, buffer: object) -> None:
        queue = self._queue
        queue._ensure_usable()
        self._require_active_handle(handle, ownership_violation=True)
        if int(handle.payload_nbytes) == 0:
            _admit_spsc_payload_kind(buffer, writable=True)
            return
        admitted = _admit_spsc_payload(buffer, int(handle.payload_nbytes), writable=True, allow_none=False)
        queue._run_primitive(queue._region.payload_read, handle.payload_offset, admitted, handle.payload_nbytes)

    def release(self, handle: SpscQueueMessage) -> None:
        queue = self._queue
        queue._ensure_usable()
        self._require_active_handle(handle, ownership_violation=True)
        queue._output_payload_head = queue._advance_payload_head(
            queue._output_payload_head,
            handle.payload_offset,
            handle.payload_nbytes,
            queue._layout.output_arena_offset,
            queue._layout.output_arena_bytes,
        )
        queue._output_head += 1
        queue._output_active = None
        queue._signal_notify(queue._layout.output_desc_head_offset, queue._output_head)

    def _require_active_handle(self, handle: SpscQueueMessage, *, ownership_violation: bool) -> None:
        queue = self._queue
        active = queue._output_active
        if (
            active is not None
            and active is handle
            and handle._owner_token is queue._owner_token
            and handle.seq == active.seq
            and handle.opcode == active.opcode
            and handle.payload_offset == active.payload_offset
            and handle.payload_nbytes == active.payload_nbytes
        ):
            return
        if ownership_violation:
            error = RuntimeError("SPSC queue output handle is not active")
            queue._poison_local(error)
            raise error
        raise RuntimeError("SPSC queue output handle is not active")


def create_spsc_queue(
    orch: object,
    *,
    initiator: ce.EndpointSelector,
    peer: ce.EndpointSelector,
    depth: int,
    input_arena_bytes: int,
    output_arena_bytes: int,
) -> SpscQueue:
    layout = SpscQueueLayout.create(
        depth=depth,
        input_arena_bytes=input_arena_bytes,
        output_arena_bytes=output_arena_bytes,
    )
    region = orch.create_region(
        members=(initiator, peer),
        topology=ce.SingleOwner(provider=peer),
        payload_bytes=layout.payload_bytes,
        counter_bytes=layout.counter_bytes,
    )
    return SpscQueue(region, layout)
