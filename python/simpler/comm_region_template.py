# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Internal Region Template types and the duplex SPSC queue ABI."""

from __future__ import annotations

import ctypes
import struct
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, NoReturn, Protocol, TypeVar

from .comm_endpoints import (
    BackendPlan,
    BackendResolver,
    EndpointRecord,
    EndpointSelector,
    RegionLayoutSpec,
    SingleOwner,
    UnsupportedRegionPlan,
)
from .comm_provider import RegionPartKind, validate_independent_local_views
from .comm_region import (
    MaterializationContext,
    MaterializationRefusal,
    NotifyOp,
    RefusalReason,
    RegionInstanceState,
    WaitCmp,
    materialize_region_instance,
)

_UINT64_MAX = (1 << 64) - 1
_SESSION_INSTANCE_ID_BYTES = 8

_SPSC_QUEUE_MAGIC = 0x53505351
_SPSC_QUEUE_ABI_MAJOR = 1
_SPSC_QUEUE_ABI_MINOR = 0
_SPSC_QUEUE_MAGIC_VERSION = (_SPSC_QUEUE_MAGIC << 32) | (_SPSC_QUEUE_ABI_MAJOR << 16) | _SPSC_QUEUE_ABI_MINOR
_SPSC_QUEUE_ENDPOINT_BINDING_SCALAR_COUNT = 10
_SPSC_QUEUE_DESCRIPTOR_BYTES = 32
_SPSC_QUEUE_ARENA_ALIGNMENT = 64
_SPSC_QUEUE_DESCRIPTOR_RING_ALIGNMENT = 8
_SPSC_QUEUE_COUNTER_STRIDE = 64
_SPSC_QUEUE_COUNTER_BYTES = 384
_SPSC_QUEUE_MAX_DEPTH = 1 << 30
_SPSC_QUEUE_INPUT_DESC_TAIL_OFFSET = 0
_SPSC_QUEUE_INPUT_DESC_HEAD_OFFSET = 64
_SPSC_QUEUE_OUTPUT_DESC_TAIL_OFFSET = 128
_SPSC_QUEUE_OUTPUT_DESC_HEAD_OFFSET = 192
_SPSC_QUEUE_INITIATOR_ABORT_OFFSET = 256
_SPSC_QUEUE_PEER_ABORT_OFFSET = 320

_DESCRIPTOR_STRUCT = struct.Struct("<QQQQ")
_SESSION_STRUCT = struct.Struct("<Q")

_TProjected = TypeVar("_TProjected")


class SpscQueueOpcode(IntEnum):
    INVALID = 0
    DATA = 1
    STOP = 2
    ERROR = 3


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


def session_instance_id_to_bits(session_instance_id: bytes) -> int:
    if not isinstance(session_instance_id, (bytes, bytearray, memoryview)):
        raise TypeError("session_instance_id must be 8 opaque bytes")
    raw = bytes(session_instance_id)
    if len(raw) != _SESSION_INSTANCE_ID_BYTES:
        raise ValueError("session_instance_id must be 8 opaque bytes")
    return int(_SESSION_STRUCT.unpack(raw)[0])


def session_instance_id_from_bits(bits: object) -> bytes:
    return _SESSION_STRUCT.pack(_require_exact_u64("session_instance_id_bits", bits))


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
    if len(raw) != _SPSC_QUEUE_DESCRIPTOR_BYTES:
        raise ValueError("descriptor requires exactly 32 bytes")
    seq, opcode, payload_offset, payload_nbytes = _DESCRIPTOR_STRUCT.unpack(raw)
    return int(seq), int(opcode), int(payload_offset), int(payload_nbytes)


@dataclass(frozen=True)
class _SpscQueueConfig:
    depth: int
    input_arena_bytes: int
    output_arena_bytes: int


@dataclass(frozen=True)
class _SpscQueueLayout:
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
    def create(cls, config: _SpscQueueConfig) -> _SpscQueueLayout:
        depth = _require_exact_u64("depth", config.depth)
        input_arena_bytes = _require_exact_u64("input_arena_bytes", config.input_arena_bytes)
        output_arena_bytes = _require_exact_u64("output_arena_bytes", config.output_arena_bytes)
        if depth < 1 or depth & (depth - 1) != 0 or depth > _SPSC_QUEUE_MAX_DEPTH:
            raise ValueError("depth must be a power of two and <= 2^30")
        if input_arena_bytes < 1 or input_arena_bytes % _SPSC_QUEUE_ARENA_ALIGNMENT != 0:
            raise ValueError("input_arena_bytes must be a positive 64-byte multiple")
        if output_arena_bytes < 1 or output_arena_bytes % _SPSC_QUEUE_ARENA_ALIGNMENT != 0:
            raise ValueError("output_arena_bytes must be a positive 64-byte multiple")

        desc_ring_bytes = _checked_mul_u64(depth, _SPSC_QUEUE_DESCRIPTOR_BYTES)
        input_desc_offset = 0
        output_desc_offset = _checked_add_u64(input_desc_offset, desc_ring_bytes)
        desc_end = _checked_add_u64(output_desc_offset, desc_ring_bytes)
        input_arena_offset = _align_up_u64(desc_end, _SPSC_QUEUE_ARENA_ALIGNMENT)
        input_arena_end = _checked_add_u64(input_arena_offset, input_arena_bytes)
        output_arena_offset = _align_up_u64(input_arena_end, _SPSC_QUEUE_ARENA_ALIGNMENT)
        payload_bytes = _checked_add_u64(output_arena_offset, output_arena_bytes)
        if (
            output_desc_offset % _SPSC_QUEUE_DESCRIPTOR_RING_ALIGNMENT != 0
            or input_arena_offset % _SPSC_QUEUE_ARENA_ALIGNMENT != 0
            or output_arena_offset % _SPSC_QUEUE_ARENA_ALIGNMENT != 0
        ):
            raise ValueError("SPSC queue layout alignment is invalid")
        return cls(
            depth=depth,
            input_arena_bytes=input_arena_bytes,
            output_arena_bytes=output_arena_bytes,
            input_desc_offset=input_desc_offset,
            output_desc_offset=output_desc_offset,
            input_arena_offset=input_arena_offset,
            output_arena_offset=output_arena_offset,
            payload_bytes=payload_bytes,
            input_desc_tail_offset=_SPSC_QUEUE_INPUT_DESC_TAIL_OFFSET,
            input_desc_head_offset=_SPSC_QUEUE_INPUT_DESC_HEAD_OFFSET,
            output_desc_tail_offset=_SPSC_QUEUE_OUTPUT_DESC_TAIL_OFFSET,
            output_desc_head_offset=_SPSC_QUEUE_OUTPUT_DESC_HEAD_OFFSET,
            initiator_abort_offset=_SPSC_QUEUE_INITIATOR_ABORT_OFFSET,
            peer_abort_offset=_SPSC_QUEUE_PEER_ABORT_OFFSET,
            counter_bytes=_SPSC_QUEUE_COUNTER_BYTES,
        )


@dataclass(frozen=True)
class SpscQueueEndpointBinding:
    magic_version: int
    session_instance_id_bits: int
    transaction_id: int
    payload_base: int
    payload_bytes: int
    counter_base: int
    counter_bytes: int
    depth: int
    input_arena_bytes: int
    output_arena_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "magic_version", _require_exact_u64("magic_version", self.magic_version))
        object.__setattr__(
            self,
            "session_instance_id_bits",
            _require_exact_u64("session_instance_id_bits", self.session_instance_id_bits),
        )
        object.__setattr__(self, "transaction_id", _require_exact_u64("transaction_id", self.transaction_id))
        object.__setattr__(self, "payload_base", _require_exact_u64("payload_base", self.payload_base))
        object.__setattr__(self, "payload_bytes", _require_exact_u64("payload_bytes", self.payload_bytes))
        object.__setattr__(self, "counter_base", _require_exact_u64("counter_base", self.counter_base))
        object.__setattr__(self, "counter_bytes", _require_exact_u64("counter_bytes", self.counter_bytes))
        object.__setattr__(self, "depth", _require_exact_u64("depth", self.depth))
        object.__setattr__(self, "input_arena_bytes", _require_exact_u64("input_arena_bytes", self.input_arena_bytes))
        object.__setattr__(
            self, "output_arena_bytes", _require_exact_u64("output_arena_bytes", self.output_arena_bytes)
        )
        if self.magic_version != _SPSC_QUEUE_MAGIC_VERSION:
            raise ValueError("binding magic_version is not SPSQ ABI 1.0")

    def to_scalars(self) -> tuple[int, ...]:
        return (
            self.magic_version,
            self.session_instance_id_bits,
            self.transaction_id,
            self.payload_base,
            self.payload_bytes,
            self.counter_base,
            self.counter_bytes,
            self.depth,
            self.input_arena_bytes,
            self.output_arena_bytes,
        )

    @classmethod
    def from_scalars(cls, scalars: Sequence[object]) -> SpscQueueEndpointBinding:
        try:
            count = len(scalars)
        except TypeError as exc:
            raise TypeError("binding scalars must be a sequence of 10 uint64 values") from exc
        if count != _SPSC_QUEUE_ENDPOINT_BINDING_SCALAR_COUNT:
            raise ValueError("binding requires exactly 10 uint64 scalars")
        values = tuple(_require_exact_u64(f"binding[{index}]", scalars[index]) for index in range(count))
        if values[0] != _SPSC_QUEUE_MAGIC_VERSION:
            raise ValueError("binding magic_version is not SPSQ ABI 1.0")
        return cls(
            magic_version=values[0],
            session_instance_id_bits=values[1],
            transaction_id=values[2],
            payload_base=values[3],
            payload_bytes=values[4],
            counter_base=values[5],
            counter_bytes=values[6],
            depth=values[7],
            input_arena_bytes=values[8],
            output_arena_bytes=values[9],
        )


_DESCRIPTOR_FIELDS_STRUCT = struct.Struct("<QQQ")


class _SpscQueueSlot(Enum):
    INITIATOR = "initiator"
    PEER = "peer"


class _SpscQueuePlanState(Enum):
    AVAILABLE = "available"
    CONSUMED = "consumed"


class _SpscQueueState(Enum):
    LIVE = "live"
    RELEASED = "released"
    POISONED_LOCAL = "poisoned_local"
    POISONED_REMOTE = "poisoned_remote"
    EXPIRED = "expired"


@dataclass(frozen=True)
class _TemplateSlotBindingRequest:
    slot: Enum
    endpoint: EndpointSelector


@dataclass(frozen=True)
class _RegionTemplatePlacementRequest:
    members: tuple[EndpointSelector, ...]
    topology: SingleOwner
    slot_bindings: tuple[_TemplateSlotBindingRequest, ...]


@dataclass(frozen=True)
class _ResolvedTemplateSlot:
    slot: Enum
    endpoint: EndpointRecord


@dataclass(frozen=True)
class _ResolvedTemplateSlots:
    bindings: tuple[_ResolvedTemplateSlot, ...]

    def endpoint(self, slot: Enum) -> EndpointRecord:
        matches = [binding.endpoint for binding in self.bindings if binding.slot == slot]
        if len(matches) != 1:
            raise ValueError(f"template slot {slot!r} is not uniquely bound")
        return matches[0]


class _RegionTemplate(Protocol):
    def plan(self, config: object) -> _RegionTemplatePlan: ...


class _RegionTemplatePlan(Protocol):
    @property
    def region_layout(self) -> RegionLayoutSpec: ...

    def _bind(self, instance: object, slots: _ResolvedTemplateSlots) -> object: ...


def _reject_copy(obj: object) -> NoReturn:
    raise TypeError(f"{type(obj).__name__} cannot be copied")


def _resolve_template_slots(
    registry: object,
    resolved_members: Sequence[EndpointRecord],
    slot_bindings: Sequence[_TemplateSlotBindingRequest],
    required_slots: Sequence[Enum],
) -> _ResolvedTemplateSlots:
    required = tuple(required_slots)
    member_by_identity = {record.identity: record for record in resolved_members}
    seen: dict[Enum, EndpointRecord] = {}
    resolved: list[_ResolvedTemplateSlot] = []
    for binding in slot_bindings:
        if not isinstance(binding, _TemplateSlotBindingRequest):
            raise TypeError("slot_bindings must contain _TemplateSlotBindingRequest values")
        slot = binding.slot
        if slot not in required:
            raise ValueError(f"unknown template slot: {slot!r}")
        if slot in seen:
            raise ValueError(f"duplicate template slot: {slot!r}")
        records = registry.resolve_members((binding.endpoint,))
        if len(records) != 1:
            raise ValueError(f"template slot {slot!r} must resolve to exactly one endpoint")
        record = records[0]
        member = member_by_identity.get(record.identity)
        if member is None:
            raise ValueError(f"template slot {slot!r} endpoint is not a region member")
        seen[slot] = member
        resolved.append(_ResolvedTemplateSlot(slot=slot, endpoint=member))
    for slot in required:
        if slot not in seen:
            raise ValueError(f"missing template slot: {slot!r}")
    return _ResolvedTemplateSlots(bindings=tuple(resolved))


def _require_distinct_initiator_peer(slots: _ResolvedTemplateSlots) -> tuple[EndpointRecord, EndpointRecord]:
    initiator = slots.endpoint(_SpscQueueSlot.INITIATOR)
    peer = slots.endpoint(_SpscQueueSlot.PEER)
    if initiator.identity == peer.identity:
        raise ValueError("INITIATOR and PEER must bind different endpoint identities")
    return initiator, peer


class _SpscQueueTemplate:
    required_slots = (_SpscQueueSlot.INITIATOR, _SpscQueueSlot.PEER)

    def plan(self, config: object) -> _SpscQueuePlan:
        if not isinstance(config, _SpscQueueConfig):
            raise TypeError("SPSC queue template.plan requires _SpscQueueConfig")
        layout = _SpscQueueLayout.create(config)
        return _SpscQueuePlan(
            config=config,
            layout=layout,
            region_layout=RegionLayoutSpec(payload_bytes=layout.payload_bytes, counter_bytes=layout.counter_bytes),
        )


class _SpscQueuePlan:
    def __init__(self, config: _SpscQueueConfig, layout: _SpscQueueLayout, region_layout: RegionLayoutSpec) -> None:
        self._config = config
        self._layout = layout
        self._region_layout = region_layout
        self._state = _SpscQueuePlanState.AVAILABLE
        self._bind_lock = threading.Lock()

    @property
    def region_layout(self) -> RegionLayoutSpec:
        return self._region_layout

    def _consume_for_bind(self) -> None:
        with self._bind_lock:
            if self._state is not _SpscQueuePlanState.AVAILABLE:
                raise RuntimeError("queue plan bind token is already consumed")
            self._state = _SpscQueuePlanState.CONSUMED

    def _bind(self, instance: object, slots: _ResolvedTemplateSlots) -> _BoundSpscQueue:
        self._consume_for_bind()
        if instance.state is not RegionInstanceState.LIVE:
            raise ValueError("queue bind requires a LIVE region instance")
        layout = instance.layout
        if int(layout.payload_bytes) != int(self._layout.payload_bytes):
            raise ValueError("region payload_bytes do not match the queue plan")
        if int(layout.counter_bytes) != int(self._layout.counter_bytes):
            raise ValueError("region counter_bytes do not match the queue plan")
        _require_distinct_initiator_peer(slots)
        payload_view = instance.local_view(RegionPartKind.PAYLOAD)
        counter_view = instance.local_view(RegionPartKind.COUNTER)
        if payload_view is None or counter_view is None:
            raise ValueError("queue bind requires PAYLOAD and COUNTER local views")
        validate_independent_local_views(payload_view, counter_view)
        if int(payload_view.logical_bytes) != int(self._layout.payload_bytes):
            raise ValueError("PAYLOAD local view does not match the queue plan")
        if int(counter_view.logical_bytes) != int(self._layout.counter_bytes):
            raise ValueError("COUNTER local view does not match the queue plan")
        session, transaction_id = instance._allocation_identity
        binding = SpscQueueEndpointBinding(
            magic_version=_SPSC_QUEUE_MAGIC_VERSION,
            session_instance_id_bits=session_instance_id_to_bits(session),
            transaction_id=int(transaction_id),
            payload_base=int(payload_view.local_base),
            payload_bytes=int(payload_view.logical_bytes),
            counter_base=int(counter_view.local_base),
            counter_bytes=int(counter_view.logical_bytes),
            depth=int(self._layout.depth),
            input_arena_bytes=int(self._layout.input_arena_bytes),
            output_arena_bytes=int(self._layout.output_arena_bytes),
        )
        return _BoundSpscQueue(instance, self._layout, binding)

    def __copy__(self) -> _SpscQueuePlan:
        _reject_copy(self)

    def __deepcopy__(self, memo: dict[int, object]) -> _SpscQueuePlan:
        _reject_copy(self)

    def __reduce__(self) -> tuple[object, ...]:
        _reject_copy(self)

    def __getstate__(self) -> object:
        _reject_copy(self)


class _RegionTemplateCoordinator:
    def __init__(self, worker: object) -> None:
        self._worker = worker

    def create(
        self,
        *,
        template: _RegionTemplate,
        config: object,
        placement: _RegionTemplatePlacementRequest,
        result_projector: Callable[[object], _TProjected],
    ) -> _TProjected:
        if not isinstance(placement, _RegionTemplatePlacementRequest):
            raise TypeError("region template create requires _RegionTemplatePlacementRequest")
        if not callable(result_projector):
            raise TypeError("region template create requires a result projector")
        worker = self._worker
        with worker._operation_lease("region_template.create"), worker._control_admission("region_template.create"):
            return self._create_in_transaction(template, config, placement, result_projector)

    def _create_in_transaction(
        self,
        template: _RegionTemplate,
        config: object,
        placement: _RegionTemplatePlacementRequest,
        result_projector: Callable[[object], _TProjected],
    ) -> _TProjected:
        worker = self._worker
        instance = None
        published = False
        try:
            plan = template.plan(config)
            registry = worker._get_endpoint_registry()
            resolved_region = registry.resolve_region_spec(placement.members, placement.topology)
            required = tuple(getattr(template, "required_slots", ()))
            if not required:
                required = tuple(binding.slot for binding in placement.slot_bindings)
            slots = _resolve_template_slots(registry, resolved_region.members, placement.slot_bindings, required)
            _require_distinct_initiator_peer(slots)
            backend_plan = BackendResolver(registry, worker._get_region_access_service()).plan(
                resolved_region, plan.region_layout
            )
            if isinstance(backend_plan, UnsupportedRegionPlan):
                raise MaterializationRefusal(RefusalReason.UNSUPPORTED_PLAN, backend_plan.message)
            if not isinstance(backend_plan, BackendPlan):
                raise MaterializationRefusal(
                    RefusalReason.UNSUPPORTED_PLAN,
                    "materialized region requires a BackendPlan",
                )
            instance = materialize_region_instance(
                MaterializationContext(
                    worker=worker,
                    registry=registry,
                    plan=backend_plan,
                    layout=plan.region_layout,
                )
            )
            self._prove_initiator_peer_access(instance, slots)
            bound = plan._bind(instance, slots)
            projected = result_projector(bound)
            published = True
            return projected
        except BaseException:
            if instance is not None and not published:
                self._rollback_unpublished(instance)
            raise

    def _prove_initiator_peer_access(self, instance: object, slots: _ResolvedTemplateSlots) -> None:
        initiator, peer = _require_distinct_initiator_peer(slots)
        if instance.consumer.identity != initiator.identity:
            raise ValueError("INITIATOR slot does not match the materialized consumer access endpoint")
        if instance.provider.identity != peer.identity:
            raise ValueError("PEER slot does not match the materialized provider-local view endpoint")

    def _rollback_unpublished(self, instance: object) -> None:
        try:
            live = instance.state is RegionInstanceState.LIVE
        except BaseException:
            live = False
        if not live:
            return
        try:
            self._worker._region_instance_registry.close(instance)
        except BaseException as cleanup_error:
            self._worker._record_unreclaimable(
                "region template create: materialized instance could not be reclaimed; no further work is admitted",
                cleanup_error,
            )


@dataclass(frozen=True)
class _SpscQueueMessage:
    seq: int
    opcode: SpscQueueOpcode
    payload_offset: int
    payload_nbytes: int
    _owner_token: object | None = field(default=None, repr=False, compare=False)


class _BoundSpscQueue:
    def __init__(self, instance: object, layout: _SpscQueueLayout, binding: SpscQueueEndpointBinding) -> None:
        self._instance = instance
        self._layout = layout
        self._endpoint_binding = binding
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
        self._output_active: _SpscQueueMessage | None = None
        self._desc_fields = bytearray(24)
        self._desc_seq = bytearray(8)
        self._desc_read = bytearray(_SPSC_QUEUE_DESCRIPTOR_BYTES)
        self.input = _SpscQueueInitiatorInput(self)
        self.output = _SpscQueueInitiatorOutput(self)

    @property
    def layout(self) -> _SpscQueueLayout:
        return self._layout

    @property
    def endpoint_binding(self) -> SpscQueueEndpointBinding:
        return self._endpoint_binding

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
            live = self._instance.state is RegionInstanceState.LIVE
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
            self._instance.counter(self._layout.initiator_abort_offset).notify(1, NotifyOp.Set)
        except BaseException as abort_error:
            adder = getattr(error, "add_note", None)
            if callable(adder):
                adder(f"initiator abort notify failed: {abort_error}")

    def _run_primitive(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._ensure_usable()
        try:
            return fn(*args, **kwargs)
        except TimeoutError:
            raise
        except BaseException as exc:
            self._poison_local(exc)
            raise

    def _signal_test(self, offset: int, cmp_value: int, cmp: WaitCmp) -> Any:
        return self._run_primitive(lambda: self._instance.counter(offset).test(int(cmp_value), cmp))

    def _signal_notify(self, offset: int, value: int) -> None:
        self._run_primitive(lambda: self._instance.counter(offset).notify(int(value), NotifyOp.Set))

    def _sample_peer_abort(self) -> None:
        result = self._signal_test(self._layout.peer_abort_offset, 1, WaitCmp.GE)
        if result.matched:
            self._state = _SpscQueueState.POISONED_REMOTE
            raise RuntimeError("SPSC queue remote abort observed")

    def _refresh_counter(self, offset: int, local_value: int) -> int:
        result = self._signal_test(offset, local_value & 0xFFFF_FFFF, WaitCmp.NE)
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
                lambda: self._instance.counter(offset).wait(local_value & 0xFFFF_FFFF, WaitCmp.NE, remaining)
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
        self._run_primitive(self._instance.payload_write, offset + 8, self._desc_fields, 24)
        self._run_primitive(self._instance.payload_write, offset, self._desc_seq, 8)

    def _read_descriptor(self, offset: int) -> _SpscQueueMessage:
        self._run_primitive(self._instance.payload_read, offset, self._desc_read, _SPSC_QUEUE_DESCRIPTOR_BYTES)
        seq, opcode_value, payload_offset, payload_nbytes = decode_spsc_queue_descriptor(bytes(self._desc_read))
        try:
            opcode = SpscQueueOpcode(opcode_value)
        except ValueError:
            error = RuntimeError("SPSC queue observed invalid descriptor opcode")
            self._poison_local(error)
            raise error from None
        return _SpscQueueMessage(
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
                error = RuntimeError("SPSC queue payload replay offset mismatch")
                self._poison_local(error)
                raise error
            cursor += arena_bytes - (cursor % arena_bytes)
        return cursor + payload_nbytes

    def _replay_released_input_descriptors(self, old_head: int, new_head: int) -> None:
        cursor = old_head
        while cursor < new_head:
            slot_index = cursor & (self._layout.depth - 1)
            slot_offset = self._layout.input_desc_offset + slot_index * _SPSC_QUEUE_DESCRIPTOR_BYTES
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
        arena_pos = next_payload_tail % self._layout.input_arena_bytes
        if arena_pos + nbytes > self._layout.input_arena_bytes:
            next_payload_tail += self._layout.input_arena_bytes - arena_pos
            arena_pos = 0
        if next_payload_tail + nbytes - self._input_payload_head > self._layout.input_arena_bytes:
            return None
        return self._layout.input_arena_offset + arena_pos, next_payload_tail

    def __copy__(self) -> _BoundSpscQueue:
        _reject_copy(self)

    def __deepcopy__(self, memo: dict[int, object]) -> _BoundSpscQueue:
        _reject_copy(self)

    def __reduce__(self) -> tuple[object, ...]:
        _reject_copy(self)

    def __getstate__(self) -> object:
        _reject_copy(self)


class _SpscQueueInitiatorInput:
    def __init__(self, queue: _BoundSpscQueue) -> None:
        self._queue = queue

    def try_enqueue(self, buffer_or_none: object, nbytes: int) -> bool:
        return self._try_enqueue(buffer_or_none, nbytes, SpscQueueOpcode.DATA)

    def enqueue(self, buffer_or_none: object, nbytes: int, timeout: float) -> None:
        self._enqueue(buffer_or_none, nbytes, SpscQueueOpcode.DATA, timeout)

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
            queue._run_primitive(queue._instance.payload_write, payload_offset, buffer_or_none, nbytes)
        seq = queue._input_tail + 1
        slot_index = queue._input_tail & (queue._layout.depth - 1)
        slot_offset = queue._layout.input_desc_offset + slot_index * _SPSC_QUEUE_DESCRIPTOR_BYTES
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
            raise ValueError("SPSC queue requires a registered Buffer or contiguous host buffer") from exc
        if not view.c_contiguous:
            raise ValueError("SPSC queue ordinary host buffer must be C-contiguous")
        if int(view.nbytes) < int(nbytes):
            raise ValueError(f"SPSC queue nbytes={nbytes} exceeds ordinary host buffer size {int(view.nbytes)}")


class _SpscQueueInitiatorOutput:
    def __init__(self, queue: _BoundSpscQueue) -> None:
        self._queue = queue

    def try_peek(self) -> _SpscQueueMessage | None:
        queue = self._queue
        queue._ensure_usable()
        queue._sample_peer_abort()
        if queue._output_active is not None:
            return queue._output_active
        queue._output_tail = queue._refresh_counter(queue._layout.output_desc_tail_offset, queue._output_tail)
        if queue._output_tail == queue._output_head:
            return None
        slot_index = queue._output_head & (queue._layout.depth - 1)
        slot_offset = queue._layout.output_desc_offset + slot_index * _SPSC_QUEUE_DESCRIPTOR_BYTES
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
        owned = _SpscQueueMessage(
            seq=message.seq,
            opcode=message.opcode,
            payload_offset=message.payload_offset,
            payload_nbytes=message.payload_nbytes,
            _owner_token=queue._owner_token,
        )
        queue._output_active = owned
        return owned

    def peek(self, timeout: float) -> _SpscQueueMessage:
        queue = self._queue
        deadline_ns = queue._blocking_deadline(timeout)
        while True:
            message = self.try_peek()
            if message is not None:
                return message
            queue._wait_progress(queue._layout.output_desc_tail_offset, queue._output_tail, deadline_ns)

    def read_into(self, handle: _SpscQueueMessage, buffer: object) -> None:
        queue = self._queue
        queue._ensure_usable()
        self._require_active_handle(handle, ownership_violation=True)
        if handle.payload_nbytes == 0:
            if buffer is not None:
                raise ValueError("SPSC queue zero-byte output read requires buffer == None")
            return
        if buffer is None:
            raise ValueError("SPSC queue nonzero output read requires a writable host buffer")
        self._require_read_destination(buffer, handle.payload_nbytes)
        queue._run_primitive(queue._instance.payload_read, handle.payload_offset, buffer, handle.payload_nbytes)

    def release(self, handle: _SpscQueueMessage) -> None:
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

    def dequeue_into(self, buffer: object, timeout: float) -> _SpscQueueMessage:
        handle = self.peek(timeout)
        self.read_into(handle, buffer)
        self.release(handle)
        return handle

    def try_dequeue_into(self, buffer: object) -> _SpscQueueMessage | None:
        handle = self.try_peek()
        if handle is None:
            return None
        self.read_into(handle, buffer)
        self.release(handle)
        return handle

    def _require_active_handle(self, handle: _SpscQueueMessage, *, ownership_violation: bool) -> None:
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

    def _require_read_destination(self, buffer: Any, nbytes: int) -> None:
        try:
            view = memoryview(buffer)
        except TypeError as exc:
            if hasattr(buffer, "nbytes") and hasattr(buffer, "base"):
                available = int(buffer.nbytes)
                if available < int(nbytes):
                    raise ValueError(f"SPSC queue nbytes={nbytes} exceeds registered buffer size {available}") from None
                return
            raise ValueError("SPSC queue requires a registered Buffer or writable contiguous host buffer") from exc
        if not view.c_contiguous:
            raise ValueError("SPSC queue ordinary host buffer must be C-contiguous")
        if view.readonly:
            raise ValueError("SPSC queue output target must be a writable ordinary host buffer")
        if int(view.nbytes) < int(nbytes):
            raise ValueError(f"SPSC queue nbytes={nbytes} exceeds ordinary host buffer size {int(view.nbytes)}")
