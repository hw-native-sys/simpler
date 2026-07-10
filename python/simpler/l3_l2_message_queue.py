# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3-side L3-L2 SPSC message queue wrapper."""

from __future__ import annotations

import ctypes
import struct
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from .l3_l2_orch_comm import (
    L3L2OrchRegion,
    NotifyOp,
    WaitCmp,
)
from .task_interface import DataType, Tensor

L3L2_QUEUE_MAGIC = 0x4C335132
L3L2_QUEUE_ABI_MAJOR = 1
L3L2_QUEUE_ABI_MINOR = 1
L3L2_QUEUE_DESC_SLOT_BYTES = 32
L3L2_QUEUE_PAYLOAD_ARENA_ALIGNMENT = 64
L3L2_QUEUE_COUNTER_STRIDE = 64
L3L2_QUEUE_INPUT_DESC_TAIL_OFFSET = 0
L3L2_QUEUE_INPUT_DESC_HEAD_OFFSET = 64
L3L2_QUEUE_OUTPUT_DESC_TAIL_OFFSET = 128
L3L2_QUEUE_OUTPUT_DESC_HEAD_OFFSET = 192
L3L2_QUEUE_L3_ABORT_FLAG_OFFSET = 256
L3L2_QUEUE_L2_ABORT_FLAG_OFFSET = 320
L3L2_QUEUE_COUNTER_BYTES = 384
L3L2_QUEUE_MAX_DEPTH = 1 << 30
_UINT64_MAX = (1 << 64) - 1

_DESC = struct.Struct("<4Q")
_POLL_INTERVAL_S = 0.00005


@dataclass(frozen=True)
class _HostByteSpan:
    nbytes: int
    ptr: int | None
    view: memoryview | None


class L3L2QueueOpcode(IntEnum):
    INVALID = 0
    DATA = 1
    STOP = 2
    ERROR = 3


class _QueueState(IntEnum):
    LIVE = 0
    RELEASED = 1
    POISONED_LOCAL = 2
    POISONED_REMOTE = 3
    EXPIRED = 4


@dataclass(frozen=True)
class L3L2QueueLayout:
    depth: int
    input_desc_offset: int
    output_desc_offset: int
    input_arena_offset: int
    output_arena_offset: int
    input_arena_bytes: int
    output_arena_bytes: int
    payload_bytes: int
    input_desc_tail_offset: int
    input_desc_head_offset: int
    output_desc_tail_offset: int
    output_desc_head_offset: int
    l3_abort_flag_offset: int
    l2_abort_flag_offset: int
    counter_bytes: int


@dataclass(frozen=True)
class L3L2QueueMessage:
    seq: int
    opcode: L3L2QueueOpcode
    payload_offset: int
    payload_nbytes: int


def l3_l2_queue_magic_version() -> int:
    return (L3L2_QUEUE_MAGIC << 32) | (L3L2_QUEUE_ABI_MAJOR << 16) | L3L2_QUEUE_ABI_MINOR


def _align_up(value: int, align: int) -> int:
    if value < 0 or value > _UINT64_MAX:
        raise ValueError("L3-L2 queue layout calculation overflowed uint64")
    remainder = value % align
    bump = 0 if remainder == 0 else align - remainder
    result = value + bump
    if result > _UINT64_MAX:
        raise ValueError("L3-L2 queue layout calculation overflowed uint64")
    return result


def _checked_add_u64(lhs: int, rhs: int) -> int:
    result = lhs + rhs
    if lhs < 0 or rhs < 0 or result > _UINT64_MAX:
        raise ValueError("L3-L2 queue layout calculation overflowed uint64")
    return result


def _tensor_like_nbytes(buffer: Any) -> int | None:
    nbytes_attr: Any = getattr(buffer, "nbytes", None)
    if nbytes_attr is not None:
        nbytes_value: Any = nbytes_attr() if callable(nbytes_attr) else nbytes_attr
        return int(nbytes_value)
    numel: Any = getattr(buffer, "numel", None)
    element_size: Any = getattr(buffer, "element_size", None)
    if callable(numel) and callable(element_size):
        numel_value: Any = numel()
        element_size_value: Any = element_size()
        return int(numel_value) * int(element_size_value)
    return None


def _host_byte_span(buffer: Any, nbytes: int, *, writable: bool) -> _HostByteSpan:
    nbytes = int(nbytes)
    try:
        view = memoryview(buffer)
    except TypeError:
        view = None

    if view is not None:
        if not view.c_contiguous:
            raise ValueError("L3-L2 queue ordinary host buffer must be C-contiguous")
        if int(view.nbytes) < nbytes:
            raise ValueError(f"L3-L2 queue nbytes={nbytes} exceeds ordinary host buffer size {int(view.nbytes)}")
        try:
            byte_view = view if view.itemsize == 1 and view.format in {"B", "b", "c"} else view.cast("B")
        except (TypeError, ValueError) as exc:
            raise ValueError("L3-L2 queue ordinary host buffer must be viewable as bytes") from exc
        if writable and byte_view.readonly:
            raise ValueError("L3-L2 queue output target must be a writable ordinary host buffer")
        ptr = None
        if not byte_view.readonly:
            ptr = ctypes.addressof(ctypes.c_char.from_buffer(byte_view))
        return _HostByteSpan(nbytes=nbytes, ptr=ptr, view=byte_view)

    data_ptr: Any = getattr(buffer, "data_ptr", None)
    if callable(data_ptr):
        is_contiguous = getattr(buffer, "is_contiguous", None)
        if callable(is_contiguous) and not bool(is_contiguous()):
            raise ValueError("L3-L2 queue ordinary host tensor-like buffer must be contiguous")
        device = getattr(buffer, "device", None)
        device_type = getattr(device, "type", device)
        if device_type is not None and str(device_type) != "cpu":
            raise ValueError("L3-L2 queue ordinary host tensor-like buffer must be on CPU")
        available = _tensor_like_nbytes(buffer)
        if available is None:
            raise ValueError("L3-L2 queue ordinary host tensor-like buffer must expose nbytes")
        if available < nbytes:
            raise ValueError(f"L3-L2 queue nbytes={nbytes} exceeds ordinary host buffer size {available}")
        ptr_value: Any = data_ptr()
        ptr = int(ptr_value)
        if ptr <= 0 and nbytes > 0:
            raise ValueError("L3-L2 queue ordinary host tensor-like buffer must have a nonzero data_ptr")
        return _HostByteSpan(nbytes=nbytes, ptr=ptr, view=None)

    access = "writable" if writable else "readable"
    raise ValueError(f"L3-L2 queue requires a registered Tensor or {access} contiguous ordinary host buffer")


def make_l3_l2_queue_layout(depth: int, input_arena_bytes: int, output_arena_bytes: int) -> L3L2QueueLayout:
    depth = int(depth)
    input_arena_bytes = int(input_arena_bytes)
    output_arena_bytes = int(output_arena_bytes)
    if depth <= 0 or depth & (depth - 1) != 0 or depth > L3L2_QUEUE_MAX_DEPTH:
        raise ValueError("L3-L2 queue depth must be a power of two and <= 2^30")
    if input_arena_bytes <= 0 or input_arena_bytes % L3L2_QUEUE_PAYLOAD_ARENA_ALIGNMENT != 0:
        raise ValueError("L3-L2 queue input_arena_bytes must be a positive 64-byte multiple")
    if output_arena_bytes <= 0 or output_arena_bytes % L3L2_QUEUE_PAYLOAD_ARENA_ALIGNMENT != 0:
        raise ValueError("L3-L2 queue output_arena_bytes must be a positive 64-byte multiple")

    desc_ring_bytes = depth * L3L2_QUEUE_DESC_SLOT_BYTES
    if desc_ring_bytes > _UINT64_MAX:
        raise ValueError("L3-L2 queue layout calculation overflowed uint64")
    input_desc_offset = 0
    output_desc_offset = _checked_add_u64(input_desc_offset, desc_ring_bytes)
    desc_end = _checked_add_u64(output_desc_offset, desc_ring_bytes)
    input_arena_offset = _align_up(desc_end, L3L2_QUEUE_PAYLOAD_ARENA_ALIGNMENT)
    input_arena_end = _checked_add_u64(input_arena_offset, input_arena_bytes)
    output_arena_offset = _align_up(input_arena_end, L3L2_QUEUE_PAYLOAD_ARENA_ALIGNMENT)
    payload_bytes = _checked_add_u64(output_arena_offset, output_arena_bytes)
    return L3L2QueueLayout(
        depth=depth,
        input_desc_offset=input_desc_offset,
        output_desc_offset=output_desc_offset,
        input_arena_offset=input_arena_offset,
        output_arena_offset=output_arena_offset,
        input_arena_bytes=input_arena_bytes,
        output_arena_bytes=output_arena_bytes,
        payload_bytes=payload_bytes,
        input_desc_tail_offset=L3L2_QUEUE_INPUT_DESC_TAIL_OFFSET,
        input_desc_head_offset=L3L2_QUEUE_INPUT_DESC_HEAD_OFFSET,
        output_desc_tail_offset=L3L2_QUEUE_OUTPUT_DESC_TAIL_OFFSET,
        output_desc_head_offset=L3L2_QUEUE_OUTPUT_DESC_HEAD_OFFSET,
        l3_abort_flag_offset=L3L2_QUEUE_L3_ABORT_FLAG_OFFSET,
        l2_abort_flag_offset=L3L2_QUEUE_L2_ABORT_FLAG_OFFSET,
        counter_bytes=L3L2_QUEUE_COUNTER_BYTES,
    )


def create_l3_l2_queue(
    orch: Any,
    *,
    worker_id: int,
    depth: int,
    input_arena_bytes: int,
    output_arena_bytes: int,
) -> L3L2Queue:
    layout = make_l3_l2_queue_layout(depth, input_arena_bytes, output_arena_bytes)
    region = orch.create_l3_l2_region(
        worker_id=int(worker_id),
        payload_bytes=layout.payload_bytes,
        counter_bytes=layout.counter_bytes,
    )
    try:
        desc_fields = orch.alloc([24], DataType.UINT8)
        desc_seq = orch.alloc([8], DataType.UINT8)
        desc_read = orch.alloc([L3L2_QUEUE_DESC_SLOT_BYTES], DataType.UINT8)
        for offset in (
            layout.input_desc_tail_offset,
            layout.input_desc_head_offset,
            layout.output_desc_tail_offset,
            layout.output_desc_head_offset,
            layout.l3_abort_flag_offset,
            layout.l2_abort_flag_offset,
        ):
            region.counter(offset).notify(0, NotifyOp.Set)
    except Exception:
        try:
            region.free()
        except Exception:
            pass
        raise
    return L3L2Queue(orch, region, layout, desc_fields, desc_seq, desc_read)


class L3L2Queue:
    def __init__(
        self,
        orch: Any,
        region: L3L2OrchRegion,
        layout: L3L2QueueLayout,
        desc_fields: Tensor,
        desc_seq: Tensor,
        desc_read: Tensor,
    ) -> None:
        self._orch = orch
        self._region = region
        self._layout = layout
        self._desc_fields = desc_fields
        self._desc_seq = desc_seq
        self._desc_read = desc_read
        self._state = _QueueState.LIVE
        self._input_head = 0
        self._input_tail = 0
        self._output_head = 0
        self._output_tail = 0
        self._input_payload_tail = 0
        self._input_payload_head = 0
        self._output_payload_head = 0
        self._output_active: L3L2QueueMessage | None = None
        self._staging_tensor: Tensor | None = None
        self._staging_nbytes = 0
        self._stop_published = False
        self.input = _L3InputQueue(self)
        self.output = _L3OutputQueue(self)

    @property
    def region(self) -> L3L2OrchRegion:
        return self._region

    @property
    def layout(self) -> L3L2QueueLayout:
        return self._layout

    @property
    def magic_version(self) -> int:
        return l3_l2_queue_magic_version()

    def l2_task_arg_scalars(self) -> list[int]:
        self._ensure_live()
        return [
            *self._region.descriptor_scalars(),
            self.magic_version,
            self._layout.depth,
            self._layout.input_arena_bytes,
            self._layout.output_arena_bytes,
            self._layout.payload_bytes,
            self._layout.counter_bytes,
        ]

    def try_request_stop(self) -> bool:
        return self.input._try_enqueue(None, 0, L3L2QueueOpcode.STOP)

    def request_stop(self, timeout: float) -> None:
        self.input._enqueue(None, 0, L3L2QueueOpcode.STOP, timeout)

    def free(self) -> None:
        if self._state == _QueueState.RELEASED:
            return
        self._state = _QueueState.RELEASED
        self._region.free()

    def _ensure_live(self) -> None:
        if self._state == _QueueState.RELEASED:
            raise RuntimeError("L3-L2 queue has been released")
        if self._state == _QueueState.POISONED_REMOTE:
            raise RuntimeError("L3-L2 queue is remote-aborted")
        if self._state == _QueueState.POISONED_LOCAL:
            raise RuntimeError("L3-L2 queue is poisoned")
        if self._state == _QueueState.EXPIRED:
            raise RuntimeError("L3-L2 queue expired after orchestration run")
        if getattr(self._region, "_expired", False):
            self._state = _QueueState.EXPIRED
            raise RuntimeError("L3-L2 queue expired after orchestration run")
        self._region._ensure_live()

    def _validate_registered_buffer(self, buffer: Any, nbytes: int) -> Tensor:
        if not isinstance(buffer, Tensor):
            raise ValueError("L3-L2 queue requires a registered Tensor returned by orch.alloc(...)")
        self._region._owner._validate_l3_l2_orch_comm_host_buffer(buffer)
        if int(nbytes) > int(buffer.nbytes()):
            raise ValueError(f"L3-L2 queue nbytes={nbytes} exceeds registered Tensor size {int(buffer.nbytes())}")
        return buffer

    def _registered_buffer_or_none(self, buffer: Any, nbytes: int) -> Tensor | None:
        if not isinstance(buffer, Tensor):
            return None
        return self._validate_registered_buffer(buffer, nbytes)

    def _ensure_staging_capacity(self, nbytes: int) -> Tensor:
        nbytes = int(nbytes)
        if self._staging_tensor is None or self._staging_nbytes < nbytes:
            tensor: Tensor = self._orch.alloc([nbytes], DataType.UINT8)
            self._staging_tensor = tensor
            self._staging_nbytes = int(tensor.nbytes())
            return tensor
        return self._staging_tensor

    def _copy_host_span_to_tensor(self, span: _HostByteSpan, tensor: Tensor) -> None:
        if span.nbytes == 0:
            return
        if span.ptr is not None:
            ctypes.memmove(int(tensor.data), span.ptr, span.nbytes)
            return
        assert span.view is not None
        ctypes.memmove(int(tensor.data), span.view[: span.nbytes].tobytes(), span.nbytes)

    def _copy_tensor_to_host_span(self, tensor: Tensor, span: _HostByteSpan) -> None:
        if span.nbytes == 0:
            return
        if span.ptr is not None:
            ctypes.memmove(span.ptr, int(tensor.data), span.nbytes)
            return
        assert span.view is not None
        span.view[: span.nbytes] = ctypes.string_at(int(tensor.data), span.nbytes)

    def _refresh_counter(self, offset: int, local_value: int, depth: int) -> int:
        result = self._signal_test(offset, local_value & 0xFFFF_FFFF, WaitCmp.NE)
        if not result.matched:
            return local_value
        observed = int(result.observed) & 0xFFFF_FFFF
        local_low = local_value & 0xFFFF_FFFF
        delta = ctypes.c_int32((observed - local_low) & 0xFFFF_FFFF).value
        if delta < 0 or delta > depth:
            self._poison_local()
            raise RuntimeError("L3-L2 queue counter reconstruction failed")
        return local_value + delta

    def _sample_peer_abort_after_timeout(self) -> None:
        result = self._signal_test(self._layout.l2_abort_flag_offset, 1, WaitCmp.GE)
        if result.matched:
            self._state = _QueueState.POISONED_REMOTE
            raise RuntimeError("L3-L2 queue remote abort observed")
        raise TimeoutError("L3-L2 queue operation timed out")

    def _poison_local(self) -> None:
        if self._state != _QueueState.LIVE:
            return
        self._state = _QueueState.POISONED_LOCAL
        try:
            l3_host_mapping = getattr(self._region, "_l3_host_mapping", None)
            if l3_host_mapping is not None:
                self._region._direct_counter_notify(self._layout.l3_abort_flag_offset, 1, NotifyOp.Set)
            else:
                self._region.counter(self._layout.l3_abort_flag_offset).notify(1, NotifyOp.Set)
        except Exception:
            pass

    def _run_primitive(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception:
            self._poison_local()
            raise

    def _signal_test(self, offset: int, cmp_value: int, cmp: WaitCmp) -> Any:
        return self._run_primitive(lambda: self._region.counter(offset).test(cmp_value, cmp))

    def _signal_notify(self, offset: int, value: int) -> None:
        self._run_primitive(lambda: self._region.counter(offset).notify(value, NotifyOp.Set))

    def _write_descriptor(
        self, offset: int, seq: int, opcode: L3L2QueueOpcode, payload_offset: int, nbytes: int
    ) -> None:
        fields_buf = (ctypes.c_uint8 * 24).from_address(int(self._desc_fields.data))
        fields_buf[:] = _DESC.pack(0, int(opcode), int(payload_offset), int(nbytes))[8:]
        seq_buf = (ctypes.c_uint8 * 8).from_address(int(self._desc_seq.data))
        seq_buf[:] = struct.pack("<Q", int(seq))
        self._run_primitive(self._region.payload_write, offset + 8, self._desc_fields, nbytes=24)
        self._run_primitive(self._region.payload_write, offset, self._desc_seq, nbytes=8)

    def _read_descriptor(self, offset: int) -> L3L2QueueMessage:
        self._run_primitive(self._region.payload_read, offset, self._desc_read, nbytes=L3L2_QUEUE_DESC_SLOT_BYTES)
        raw = ctypes.string_at(int(self._desc_read.data), L3L2_QUEUE_DESC_SLOT_BYTES)
        seq, opcode_value, payload_offset, payload_nbytes = _DESC.unpack(raw)
        try:
            opcode = L3L2QueueOpcode(opcode_value)
        except ValueError:
            self._poison_local()
            raise RuntimeError("L3-L2 queue observed invalid descriptor opcode") from None
        return L3L2QueueMessage(
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
        expected_offset = arena_offset + (cursor % arena_bytes)
        if expected_offset != payload_offset:
            if payload_offset != arena_offset:
                self._poison_local()
                raise RuntimeError("L3-L2 queue payload replay offset mismatch")
            cursor += arena_bytes - (cursor % arena_bytes)
        return cursor + payload_nbytes

    def _replay_released_input_descriptors(self, old_head: int, new_head: int) -> None:
        cursor = old_head
        while cursor < new_head:
            slot_index = cursor & (self._layout.depth - 1)
            slot_offset = self._layout.input_desc_offset + slot_index * L3L2_QUEUE_DESC_SLOT_BYTES
            message = self._read_descriptor(slot_offset)
            if message.seq != cursor + 1:
                self._poison_local()
                raise RuntimeError("L3-L2 queue input release replay seq mismatch")
            self._input_payload_head = self._advance_payload_head(
                self._input_payload_head,
                message.payload_offset,
                message.payload_nbytes,
                self._layout.input_arena_offset,
                self._layout.input_arena_bytes,
            )
            cursor += 1


class _L3InputQueue:
    def __init__(self, queue: L3L2Queue) -> None:
        self._queue = queue

    def enqueue(self, buffer_or_none: Any, nbytes: int, timeout: float) -> None:
        self._enqueue(buffer_or_none, nbytes, L3L2QueueOpcode.DATA, timeout)

    def try_enqueue(self, buffer_or_none: Any, nbytes: int) -> bool:
        return self._try_enqueue(buffer_or_none, nbytes, L3L2QueueOpcode.DATA)

    def _enqueue(self, buffer_or_none: Any, nbytes: int, opcode: L3L2QueueOpcode, timeout: float) -> None:
        if timeout is None or float(timeout) <= 0:
            raise ValueError("L3-L2 queue blocking operations require a positive timeout")
        deadline = time.monotonic() + float(timeout)
        while True:
            if self._try_enqueue(buffer_or_none, nbytes, opcode):
                return
            if self._queue._stop_published:
                raise RuntimeError("L3-L2 queue input is stopped")
            if time.monotonic() >= deadline:
                self._queue._sample_peer_abort_after_timeout()
            time.sleep(_POLL_INTERVAL_S)

    def _try_enqueue(self, buffer_or_none: Any, nbytes: int, opcode: L3L2QueueOpcode) -> bool:
        queue = self._queue
        nbytes = int(nbytes)
        if nbytes < 0:
            raise ValueError("L3-L2 queue nbytes must be non-negative")
        payload_tensor, staged_span = self._resolve_payload_source(buffer_or_none, nbytes)

        queue._ensure_live()
        if queue._stop_published:
            return False
        if opcode == L3L2QueueOpcode.STOP and nbytes != 0:
            raise ValueError("L3-L2 queue STOP must be zero-byte")
        if nbytes > queue._layout.input_arena_bytes:
            return False
        old_head = queue._input_head
        queue._input_head = queue._refresh_counter(
            queue._layout.input_desc_head_offset, queue._input_head, queue._layout.depth
        )
        if queue._input_head != old_head:
            queue._replay_released_input_descriptors(old_head, queue._input_head)
        if queue._input_tail - queue._input_head >= queue._layout.depth:
            return False

        payload_offset = 0
        next_payload_tail = queue._input_payload_tail
        if nbytes != 0:
            payload_offset, next_payload_tail = self._reserve_payload_range(nbytes, next_payload_tail)
            if payload_offset < 0:
                return False
            if staged_span is not None:
                payload_tensor = self._materialize_staged_payload(buffer_or_none, staged_span, nbytes)
            queue._run_primitive(queue._region.payload_write, payload_offset, payload_tensor, nbytes=nbytes)
            queue._input_payload_tail = next_payload_tail + nbytes

        seq = queue._input_tail + 1
        slot_index = queue._input_tail & (queue._layout.depth - 1)
        slot_offset = queue._layout.input_desc_offset + slot_index * L3L2_QUEUE_DESC_SLOT_BYTES
        queue._write_descriptor(slot_offset, seq, opcode, payload_offset, nbytes)
        queue._input_tail += 1
        queue._signal_notify(queue._layout.input_desc_tail_offset, queue._input_tail)
        if opcode == L3L2QueueOpcode.STOP:
            queue._stop_published = True
        return True

    def _resolve_payload_source(self, buffer_or_none: Any, nbytes: int) -> tuple[Any | None, _HostByteSpan | None]:
        if nbytes == 0:
            if buffer_or_none is not None:
                raise ValueError("L3-L2 queue zero-byte enqueue requires buffer_or_none == None")
            return None, None
        payload_tensor = self._queue._registered_buffer_or_none(buffer_or_none, nbytes)
        if payload_tensor is not None:
            return payload_tensor, None
        return None, _host_byte_span(buffer_or_none, nbytes, writable=False)

    def _reserve_payload_range(self, nbytes: int, next_payload_tail: int) -> tuple[int, int]:
        queue = self._queue
        arena_pos = next_payload_tail % queue._layout.input_arena_bytes
        if arena_pos + nbytes > queue._layout.input_arena_bytes:
            next_payload_tail += queue._layout.input_arena_bytes - arena_pos
            arena_pos = 0
        if next_payload_tail + nbytes - queue._input_payload_head > queue._layout.input_arena_bytes:
            return -1, next_payload_tail
        return queue._layout.input_arena_offset + arena_pos, next_payload_tail

    def _materialize_staged_payload(self, buffer_or_none: Any, staged_span: _HostByteSpan, nbytes: int) -> Any:
        queue = self._queue
        if getattr(queue._region, "_l3_host_mapping", None) is not None:
            return buffer_or_none
        payload_tensor = queue._ensure_staging_capacity(nbytes)
        queue._copy_host_span_to_tensor(staged_span, payload_tensor)
        return payload_tensor


class _L3OutputQueue:
    def __init__(self, queue: L3L2Queue) -> None:
        self._queue = queue

    def try_peek(self) -> L3L2QueueMessage | None:
        queue = self._queue
        queue._ensure_live()
        if queue._output_active is not None:
            return queue._output_active
        queue._output_tail = queue._refresh_counter(
            queue._layout.output_desc_tail_offset, queue._output_tail, queue._layout.depth
        )
        if queue._output_tail == queue._output_head:
            return None
        slot_index = queue._output_head & (queue._layout.depth - 1)
        slot_offset = queue._layout.output_desc_offset + slot_index * L3L2_QUEUE_DESC_SLOT_BYTES
        message = queue._read_descriptor(slot_offset)
        if message.seq != queue._output_head + 1:
            queue._poison_local()
            raise RuntimeError("L3-L2 queue output descriptor seq mismatch")
        if message.opcode == L3L2QueueOpcode.STOP:
            queue._poison_local()
            raise RuntimeError("L3-L2 queue output descriptor cannot be STOP")
        if message.payload_nbytes == 0:
            if message.payload_offset != 0:
                queue._poison_local()
                raise RuntimeError("L3-L2 queue zero-byte output descriptor has nonzero offset")
        else:
            begin = queue._layout.output_arena_offset
            end = begin + queue._layout.output_arena_bytes
            if message.payload_offset < begin or message.payload_offset + message.payload_nbytes > end:
                queue._poison_local()
                raise RuntimeError("L3-L2 queue output payload outside output arena")
            queue._advance_payload_head(
                queue._output_payload_head,
                message.payload_offset,
                message.payload_nbytes,
                queue._layout.output_arena_offset,
                queue._layout.output_arena_bytes,
            )
        queue._output_active = message
        return message

    def peek(self, timeout: float) -> L3L2QueueMessage:
        if timeout is None or float(timeout) <= 0:
            raise ValueError("L3-L2 queue blocking operations require a positive timeout")
        deadline = time.monotonic() + float(timeout)
        while True:
            message = self.try_peek()
            if message is not None:
                return message
            if time.monotonic() >= deadline:
                self._queue._sample_peer_abort_after_timeout()
            time.sleep(_POLL_INTERVAL_S)

    def read_into(self, handle: L3L2QueueMessage, buffer: Any) -> None:
        queue = self._queue
        queue._ensure_live()
        if queue._output_active != handle:
            raise RuntimeError("L3-L2 queue output handle is not active")
        if handle.payload_nbytes == 0:
            if buffer is not None:
                raise ValueError("L3-L2 queue zero-byte output read requires buffer == None")
            return
        target = queue._registered_buffer_or_none(buffer, handle.payload_nbytes)
        target_span = None
        if target is None:
            target_span = _host_byte_span(buffer, handle.payload_nbytes, writable=True)
            if getattr(queue._region, "_l3_host_mapping", None) is None:
                target = queue._ensure_staging_capacity(handle.payload_nbytes)
            else:
                target = buffer
        queue._run_primitive(queue._region.payload_read, handle.payload_offset, target, nbytes=handle.payload_nbytes)
        if target_span is not None and getattr(queue._region, "_l3_host_mapping", None) is None:
            queue._copy_tensor_to_host_span(target, target_span)

    def release(self, handle: L3L2QueueMessage) -> None:
        queue = self._queue
        queue._ensure_live()
        if queue._output_active != handle:
            queue._poison_local()
            raise RuntimeError("L3-L2 queue output handle is not active")
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

    def dequeue_into(self, buffer: Any, timeout: float) -> L3L2QueueMessage:
        handle = self.peek(timeout)
        self.read_into(handle, buffer)
        self.release(handle)
        return handle

    def try_dequeue_into(self, buffer: Any) -> L3L2QueueMessage | None:
        handle = self.try_peek()
        if handle is None:
            return None
        self.read_into(handle, buffer)
        self.release(handle)
        return handle
