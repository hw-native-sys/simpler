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

from typing import Any

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


def make_worker_chip_queue_layout(depth: int, input_arena_bytes: int, output_arena_bytes: int) -> _SpscQueueLayout:
    return _SpscQueueLayout.create(
        _SpscQueueConfig(
            depth=int(depth),
            input_arena_bytes=int(input_arena_bytes),
            output_arena_bytes=int(output_arena_bytes),
        )
    )


def create_worker_chip_queue(
    orch: Any,
    *,
    worker_id: int,
    depth: int,
    input_arena_bytes: int,
    output_arena_bytes: int,
) -> WorkerChipQueue:
    worker = orch._worker
    worker._validate_worker_chip_id(int(worker_id))
    root_path = _format_worker_path(int(worker.level))
    provider_path = _format_worker_path(2, parent_path=root_path, index=int(worker_id))
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
        depth=int(depth),
        input_arena_bytes=int(input_arena_bytes),
        output_arena_bytes=int(output_arena_bytes),
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
    return WorkerChipQueue(bound, region)


class WorkerChipQueue:
    def __init__(self, bound: _BoundSpscQueue, region: WorkerChipOrchRegion) -> None:
        self._bound = bound
        self._region = region
        self.input = bound.input
        self.output = bound.output

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
