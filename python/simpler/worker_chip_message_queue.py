# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3-side L3-L2 SPSC message queue compatibility facade."""

from __future__ import annotations

import operator
from typing import Any

from .buffer import AccessMode, Buffer
from .comm_endpoints import DEVICE_AICPU, HOST_CPU, SingleOwner, _format_worker_path, at
from .comm_provider import RegionPartKind
from .comm_region_template import (
    _SPSC_QUEUE_COUNTER_BYTES,
    _SPSC_QUEUE_DESCRIPTOR_BYTES,
    _SPSC_QUEUE_INITIATOR_ABORT_OFFSET,
    _SPSC_QUEUE_PEER_ABORT_OFFSET,
    SpscQueueOpcode,
    _BoundSpscQueue,
    _RegionTemplateCoordinator,
    _RegionTemplatePlacementRequest,
    _SpscQueueConfig,
    _SpscQueueLayout,
    _SpscQueueMessage,
    _SpscQueueSlot,
    _SpscQueueTemplate,
    _TemplateSlotBindingRequest,
)
from .worker_chip_orch_comm import WorkerChipOrchRegion, worker_chip_orch_region_desc_from_local_views

WORKER_CHIP_QUEUE_DESC_SLOT_BYTES = _SPSC_QUEUE_DESCRIPTOR_BYTES
WORKER_CHIP_QUEUE_COUNTER_BYTES = _SPSC_QUEUE_COUNTER_BYTES
WORKER_CHIP_QUEUE_WORKER_ABORT_FLAG_OFFSET = _SPSC_QUEUE_INITIATOR_ABORT_OFFSET
WORKER_CHIP_QUEUE_CHIP_ABORT_FLAG_OFFSET = _SPSC_QUEUE_PEER_ABORT_OFFSET

WorkerChipQueueOpcode = SpscQueueOpcode
WorkerChipQueueMessage = _SpscQueueMessage
WorkerChipQueueLayout = _SpscQueueLayout


def _require_worker_chip_id(value: object) -> int:
    if type(value) is bool:
        raise TypeError("worker_id must not be bool")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError("worker_id must support operator.index()") from exc


def _admit_bytes_like(obj: object, nbytes: int, *, writable: bool) -> object:
    try:
        view = memoryview(obj)
    except TypeError as exc:
        raise ValueError("worker-chip queue requires a registered HOST Buffer or contiguous host buffer") from exc
    if not view.c_contiguous:
        raise ValueError("worker-chip queue ordinary host buffer must be C-contiguous")
    if writable and view.readonly:
        raise ValueError("worker-chip queue output target must be writable")
    if int(view.nbytes) < int(nbytes):
        raise ValueError(
            f"worker-chip queue nbytes={int(nbytes)} exceeds ordinary host buffer size {int(view.nbytes)}"
        )
    return obj


def _admit_registered_buffer(worker: Any, obj: Buffer, nbytes: int, *, writable: bool) -> Buffer:
    if obj.closed:
        raise ValueError("worker-chip queue buffer is closed")
    worker._validate_worker_chip_orch_comm_host_buffer(obj)
    needed = AccessMode.WRITE if writable else AccessMode.READ
    if obj.access not in (needed, AccessMode.READWRITE):
        raise ValueError(
            f"worker-chip queue buffer grants {obj.access.name} but this direction needs {needed.name}"
        )
    if int(obj.nbytes) < int(nbytes):
        raise ValueError(f"worker-chip queue nbytes={int(nbytes)} exceeds registered buffer size {int(obj.nbytes)}")
    return obj


def _admit_worker_chip_payload(
    worker: Any, obj: object, nbytes: int, *, writable: bool, allow_none: bool
) -> object:
    if int(nbytes) == 0:
        if allow_none:
            if obj is not None:
                raise ValueError("worker-chip queue zero-byte payload requires buffer == None")
            return None
    if obj is None:
        raise ValueError("worker-chip queue nonzero payload requires a host buffer")
    if isinstance(obj, Buffer):
        return _admit_registered_buffer(worker, obj, nbytes, writable=writable)
    return _admit_bytes_like(obj, nbytes, writable=writable)


def _admit_worker_chip_payload_kind(worker: Any, obj: object, *, writable: bool) -> None:
    if obj is None:
        return
    _admit_worker_chip_payload(worker, obj, 0, writable=writable, allow_none=False)


def _require_live_bound_queue(lane: Any) -> None:
    lane._queue._ensure_usable()


def make_worker_chip_queue_layout(depth: int, input_arena_bytes: int, output_arena_bytes: int) -> _SpscQueueLayout:
    return _SpscQueueLayout.create(
        _SpscQueueConfig(
            depth=depth,
            input_arena_bytes=input_arena_bytes,
            output_arena_bytes=output_arena_bytes,
        )
    )


def create_worker_chip_queue(
    orch: Any,
    *,
    worker_id: object,
    depth: object,
    input_arena_bytes: object,
    output_arena_bytes: object,
) -> WorkerChipQueue:
    worker = orch._worker
    worker_id = _require_worker_chip_id(worker_id)
    worker._validate_worker_chip_id(worker_id)
    root_path = _format_worker_path(int(worker.level))
    provider_path = _format_worker_path(2, parent_path=root_path, index=worker_id)
    host = at(root_path, HOST_CPU)
    peer = at(provider_path, DEVICE_AICPU)
    placement = _RegionTemplatePlacementRequest(
        members=(host, peer),
        topology=SingleOwner(provider=peer),
        slot_bindings=(
            _TemplateSlotBindingRequest(slot=_SpscQueueSlot.INITIATOR, endpoint=host),
            _TemplateSlotBindingRequest(slot=_SpscQueueSlot.PEER, endpoint=peer),
        ),
    )
    config = _SpscQueueConfig(
        depth=depth,
        input_arena_bytes=input_arena_bytes,
        output_arena_bytes=output_arena_bytes,
    )
    return _RegionTemplateCoordinator(worker).create(
        template=_SpscQueueTemplate(),
        config=config,
        placement=placement,
        result_projector=lambda bound: _project_worker_chip_queue(worker, bound),
    )


def _project_worker_chip_queue(worker: Any, bound: object) -> WorkerChipQueue:
    if not isinstance(bound, _BoundSpscQueue):
        raise TypeError("worker-chip queue projector requires a bound SPSC queue")
    instance = bound._instance
    payload_view = instance.local_view(RegionPartKind.PAYLOAD)
    counter_view = instance.local_view(RegionPartKind.COUNTER)
    if payload_view is None or counter_view is None:
        raise RuntimeError("worker-chip queue projector requires PAYLOAD and COUNTER local views")
    desc = worker_chip_orch_region_desc_from_local_views(instance.provider_resource_id, payload_view, counter_view)
    region = WorkerChipOrchRegion(worker, instance, desc)
    return WorkerChipQueue(worker, bound, region)


class _WorkerChipQueueInput:
    def __init__(self, worker: Any, bound: _BoundSpscQueue) -> None:
        self._worker = worker
        self._lane = bound.input

    def try_enqueue(self, buffer_or_none: object, nbytes: int) -> bool:
        _require_live_bound_queue(self._lane)
        admitted = _admit_worker_chip_payload(
            self._worker, buffer_or_none, nbytes, writable=False, allow_none=True
        )
        return self._lane.try_enqueue(admitted, nbytes)

    def enqueue(self, buffer_or_none: object, nbytes: int, timeout: float) -> None:
        _require_live_bound_queue(self._lane)
        admitted = _admit_worker_chip_payload(
            self._worker, buffer_or_none, nbytes, writable=False, allow_none=True
        )
        self._lane.enqueue(admitted, nbytes, timeout)


class _WorkerChipQueueOutput:
    def __init__(self, worker: Any, bound: _BoundSpscQueue) -> None:
        self._worker = worker
        self._lane = bound.output

    def try_peek(self) -> _SpscQueueMessage | None:
        return self._lane.try_peek()

    def peek(self, timeout: float) -> _SpscQueueMessage:
        return self._lane.peek(timeout)

    def read_into(self, handle: _SpscQueueMessage, buffer: object) -> None:
        _require_live_bound_queue(self._lane)
        self._lane._require_active_handle(handle, ownership_violation=True)
        admitted = _admit_worker_chip_payload(
            self._worker, buffer, handle.payload_nbytes, writable=True, allow_none=True
        )
        self._lane.read_into(handle, admitted)

    def release(self, handle: _SpscQueueMessage) -> None:
        self._lane.release(handle)

    def dequeue_into(self, buffer: object, timeout: float) -> _SpscQueueMessage:
        _require_live_bound_queue(self._lane)
        _admit_worker_chip_payload_kind(self._worker, buffer, writable=True)
        handle = self._lane.peek(timeout)
        admitted = _admit_worker_chip_payload(
            self._worker, buffer, handle.payload_nbytes, writable=True, allow_none=True
        )
        self._lane.read_into(handle, admitted)
        self._lane.release(handle)
        return handle

    def try_dequeue_into(self, buffer: object) -> _SpscQueueMessage | None:
        _require_live_bound_queue(self._lane)
        _admit_worker_chip_payload_kind(self._worker, buffer, writable=True)
        handle = self._lane.try_peek()
        if handle is None:
            return None
        admitted = _admit_worker_chip_payload(
            self._worker, buffer, handle.payload_nbytes, writable=True, allow_none=True
        )
        self._lane.read_into(handle, admitted)
        self._lane.release(handle)
        return handle


class WorkerChipQueue:
    def __init__(self, worker: Any, bound: _BoundSpscQueue, region: WorkerChipOrchRegion) -> None:
        self._bound = bound
        self._region = region
        self.input = _WorkerChipQueueInput(worker, bound)
        self.output = _WorkerChipQueueOutput(worker, bound)

    @property
    def region(self) -> WorkerChipOrchRegion:
        return self._region

    @property
    def layout(self) -> _SpscQueueLayout:
        return self._bound.layout

    @property
    def magic_version(self) -> int:
        return int(self._bound.endpoint_binding.magic_version)

    def chip_task_arg_scalars(self) -> list[int]:
        self._bound._ensure_usable()
        return list(self._bound.endpoint_binding.to_scalars())

    def try_request_stop(self) -> bool:
        return self._bound.try_request_stop()

    def request_stop(self, timeout: float) -> None:
        self._bound.request_stop(timeout)

    def free(self) -> None:
        self._bound.free()
        self._region.free()
