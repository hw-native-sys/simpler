# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415

from __future__ import annotations

import copy
import inspect
import pickle
import struct
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pytest
from simpler.comm_endpoints import (
    DEVICE_AICPU,
    HOST_CPU,
    EndpointIdentity,
    EndpointRecord,
    EndpointRegistry,
    RegionLayoutSpec,
    SingleOwner,
    at,
)
from simpler.comm_provider import RegionPartKind, RegionPartLocalView
from simpler.comm_region import (
    MaterializationError,
    NotifyOp,
    RegionInstanceState,
    SignalTestResult,
    WaitCmp,
)
from simpler.comm_region_template import (
    _SPSC_QUEUE_ABI_MAJOR,
    _SPSC_QUEUE_ABI_MINOR,
    _SPSC_QUEUE_ARENA_ALIGNMENT,
    _SPSC_QUEUE_COUNTER_BYTES,
    _SPSC_QUEUE_COUNTER_STRIDE,
    _SPSC_QUEUE_DESCRIPTOR_BYTES,
    _SPSC_QUEUE_ENDPOINT_BINDING_SCALAR_COUNT,
    _SPSC_QUEUE_INITIATOR_ABORT_OFFSET,
    _SPSC_QUEUE_INPUT_DESC_HEAD_OFFSET,
    _SPSC_QUEUE_INPUT_DESC_TAIL_OFFSET,
    _SPSC_QUEUE_MAGIC,
    _SPSC_QUEUE_MAGIC_VERSION,
    _SPSC_QUEUE_MAX_DEPTH,
    _SPSC_QUEUE_OUTPUT_DESC_HEAD_OFFSET,
    _SPSC_QUEUE_OUTPUT_DESC_TAIL_OFFSET,
    _SPSC_QUEUE_PEER_ABORT_OFFSET,
    SpscQueueEndpointBinding,
    SpscQueueOpcode,
    _align_up_u64,
    _BoundSpscQueue,
    _checked_add_u64,
    _checked_mul_u64,
    _RegionTemplateCoordinator,
    _RegionTemplatePlacementRequest,
    _require_distinct_initiator_peer,
    _resolve_template_slots,
    _SpscQueueConfig,
    _SpscQueueLayout,
    _SpscQueueMessage,
    _SpscQueuePlan,
    _SpscQueuePlanState,
    _SpscQueueSlot,
    _SpscQueueState,
    _SpscQueueTemplate,
    _TemplateSlotBindingRequest,
    decode_spsc_queue_descriptor,
    encode_spsc_queue_descriptor,
    session_instance_id_from_bits,
    session_instance_id_to_bits,
)
from simpler.worker import Worker, _Lifecycle

# Golden vectors shared with tests/ut/cpp/common/test_region_template.cpp.
_LAYOUT_GOLDEN = (
    (1, 64, 64, 32, 64, 128, 192),
    (4, 128, 192, 128, 256, 384, 576),
    (8, 192, 64, 256, 512, 704, 768),
    (2, 64, 128, 64, 128, 192, 320),
)
_MAX_DEPTH_LAYOUT = (
    _SPSC_QUEUE_MAX_DEPTH,
    64,
    64,
    _SPSC_QUEUE_MAX_DEPTH * 32,
    _SPSC_QUEUE_MAX_DEPTH * 64,
    _SPSC_QUEUE_MAX_DEPTH * 64 + 64,
    _SPSC_QUEUE_MAX_DEPTH * 64 + 128,
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
_SESSION_GOLDEN_BYTES = bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08])
_SESSION_GOLDEN_BITS = 0x0807060504030201
_BINDING_GOLDEN: tuple[int, ...] = (
    0x5350535100010000,
    _SESSION_GOLDEN_BITS,
    42,
    0x1000,
    192,
    0x2000,
    384,
    1,
    64,
    64,
)


def _layout(depth, input_arena_bytes, output_arena_bytes):
    return _SpscQueueLayout.create(
        _SpscQueueConfig(depth=depth, input_arena_bytes=input_arena_bytes, output_arena_bytes=output_arena_bytes)
    )


def test_packed_magic_version_is_spsq_abi_1_0():
    assert _SPSC_QUEUE_MAGIC == 0x53505351
    assert _SPSC_QUEUE_ABI_MAJOR == 1
    assert _SPSC_QUEUE_ABI_MINOR == 0
    assert _SPSC_QUEUE_MAGIC_VERSION == 0x5350535100010000
    packed = (_SPSC_QUEUE_MAGIC << 32) | (_SPSC_QUEUE_ABI_MAJOR << 16) | _SPSC_QUEUE_ABI_MINOR
    assert _SPSC_QUEUE_MAGIC_VERSION == packed
    assert _SPSC_QUEUE_ENDPOINT_BINDING_SCALAR_COUNT == 10
    assert _SPSC_QUEUE_DESCRIPTOR_BYTES == 32
    assert _SPSC_QUEUE_ARENA_ALIGNMENT == 64
    assert _SPSC_QUEUE_COUNTER_STRIDE == 64
    assert _SPSC_QUEUE_COUNTER_BYTES == 384
    assert _SPSC_QUEUE_MAX_DEPTH == 1 << 30


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
    assert layout.input_arena_bytes == input_arena_bytes
    assert layout.output_arena_bytes == output_arena_bytes
    assert layout.input_arena_offset % 64 == 0
    assert layout.output_arena_offset % 64 == 0
    assert layout.input_desc_tail_offset == _SPSC_QUEUE_INPUT_DESC_TAIL_OFFSET
    assert layout.input_desc_head_offset == _SPSC_QUEUE_INPUT_DESC_HEAD_OFFSET
    assert layout.output_desc_tail_offset == _SPSC_QUEUE_OUTPUT_DESC_TAIL_OFFSET
    assert layout.output_desc_head_offset == _SPSC_QUEUE_OUTPUT_DESC_HEAD_OFFSET
    assert layout.initiator_abort_offset == _SPSC_QUEUE_INITIATOR_ABORT_OFFSET
    assert layout.peer_abort_offset == _SPSC_QUEUE_PEER_ABORT_OFFSET
    assert layout.counter_bytes == _SPSC_QUEUE_COUNTER_BYTES


def test_layout_max_depth_golden_vector():
    depth, input_arena_bytes, output_arena_bytes, output_desc, input_arena, output_arena, payload = _MAX_DEPTH_LAYOUT
    layout = _layout(depth, input_arena_bytes, output_arena_bytes)
    assert layout.depth == depth
    assert layout.output_desc_offset == output_desc
    assert layout.input_arena_offset == input_arena
    assert layout.output_arena_offset == output_arena
    assert layout.payload_bytes == payload
    assert layout.input_arena_offset % 64 == 0
    assert layout.output_arena_offset % 64 == 0


def test_layout_is_deterministic_and_has_no_runtime_side_effect():
    config = _SpscQueueConfig(depth=4, input_arena_bytes=128, output_arena_bytes=192)
    first = _SpscQueueLayout.create(config)
    second = _SpscQueueLayout.create(config)
    assert first == second
    assert copy.copy(first) == first


@pytest.mark.parametrize(
    ("depth", "input_arena_bytes", "output_arena_bytes"),
    [
        (3, 64, 64),
        (0, 64, 64),
        ((1 << 30) + 1, 64, 64),
        (1 << 31, 64, 64),
        (2, 0, 64),
        (2, 65, 64),
        (2, 64, 0),
        (2, 64, 63),
        (True, 64, 64),
        (2, True, 64),
        (2, 64, True),
        (-1, 64, 64),
        (2, -64, 64),
        (1.0, 64, 64),
    ],
)
def test_layout_rejects_invalid_depth_and_arena_values(depth, input_arena_bytes, output_arena_bytes):
    with pytest.raises((TypeError, ValueError)):
        _layout(depth, input_arena_bytes, output_arena_bytes)


def test_layout_add_overflow_input_arena():
    with pytest.raises(ValueError, match="overflowed uint64"):
        _layout(2, (1 << 64) - 64, 64)


def test_layout_add_overflow_output_arena():
    with pytest.raises(ValueError, match="overflowed uint64"):
        _layout(1, 64, (1 << 64) - 64)


def test_checked_mul_add_align_overflow_boundaries():
    with pytest.raises(ValueError, match="overflowed uint64"):
        _checked_mul_u64(1 << 63, 2)
    with pytest.raises(ValueError, match="overflowed uint64"):
        _checked_add_u64((1 << 64) - 1, 1)
    with pytest.raises(ValueError, match="overflowed uint64"):
        _align_up_u64((1 << 64) - 32, 64)
    assert _checked_mul_u64(_SPSC_QUEUE_MAX_DEPTH, 32) == _SPSC_QUEUE_MAX_DEPTH * 32
    assert _align_up_u64(64, 64) == 64
    assert _align_up_u64(65, 64) == 128


def test_opcode_values_are_fixed():
    assert int(SpscQueueOpcode.INVALID) == 0
    assert int(SpscQueueOpcode.DATA) == 1
    assert int(SpscQueueOpcode.STOP) == 2
    assert int(SpscQueueOpcode.ERROR) == 3


def test_descriptor_bytes_golden_vector():
    encoded = encode_spsc_queue_descriptor(
        seq=_DESCRIPTOR_GOLDEN_FIELDS[0],
        opcode=SpscQueueOpcode.ERROR,
        payload_offset=_DESCRIPTOR_GOLDEN_FIELDS[2],
        payload_nbytes=_DESCRIPTOR_GOLDEN_FIELDS[3],
    )
    assert encoded == _DESCRIPTOR_GOLDEN_BYTES
    assert decode_spsc_queue_descriptor(encoded) == _DESCRIPTOR_GOLDEN_FIELDS
    assert struct.calcsize("<QQQQ") == 32


def test_descriptor_rejects_non_exact_size_and_scalars():
    with pytest.raises(ValueError):
        decode_spsc_queue_descriptor(_DESCRIPTOR_GOLDEN_BYTES[:-1])
    with pytest.raises(TypeError):
        encode_spsc_queue_descriptor(True, 1, 0, 0)
    with pytest.raises(ValueError):
        encode_spsc_queue_descriptor(-1, 1, 0, 0)


def test_session_identity_little_endian_round_trip():
    assert session_instance_id_to_bits(_SESSION_GOLDEN_BYTES) == _SESSION_GOLDEN_BITS
    assert session_instance_id_from_bits(_SESSION_GOLDEN_BITS) == _SESSION_GOLDEN_BYTES
    assert session_instance_id_from_bits(session_instance_id_to_bits(_SESSION_GOLDEN_BYTES)) == _SESSION_GOLDEN_BYTES


def test_binding_exact_10_scalar_golden_vector():
    binding = SpscQueueEndpointBinding(*_BINDING_GOLDEN)
    scalars = binding.to_scalars()
    assert len(scalars) == 10
    assert scalars == _BINDING_GOLDEN
    assert SpscQueueEndpointBinding.from_scalars(list(scalars)) == binding
    assert SpscQueueEndpointBinding.from_scalars(scalars).to_scalars() == scalars


@pytest.mark.parametrize(
    "scalars",
    [
        _BINDING_GOLDEN[:-1],
        _BINDING_GOLDEN + (0,),
        (),
    ],
)
def test_binding_rejects_non_exact_count(scalars):
    with pytest.raises(ValueError, match="exactly 10"):
        SpscQueueEndpointBinding.from_scalars(scalars)


def test_binding_rejects_bool_range_and_version_mismatch():
    with pytest.raises(TypeError):
        SpscQueueEndpointBinding.from_scalars((True,) + _BINDING_GOLDEN[1:])
    negative = list(_BINDING_GOLDEN)
    negative[2] = -1
    with pytest.raises(ValueError):
        SpscQueueEndpointBinding.from_scalars(negative)
    over = list(_BINDING_GOLDEN)
    over[3] = 1 << 64
    with pytest.raises(ValueError):
        SpscQueueEndpointBinding.from_scalars(over)
    wrong_magic = list(_BINDING_GOLDEN)
    wrong_magic[0] = 0x4C33513200010001
    with pytest.raises(ValueError, match="SPSQ ABI 1.0"):
        SpscQueueEndpointBinding.from_scalars(wrong_magic)
    wrong_major = list(_BINDING_GOLDEN)
    wrong_major[0] = (_SPSC_QUEUE_MAGIC << 32) | (2 << 16) | 0
    with pytest.raises(ValueError, match="SPSQ ABI 1.0"):
        SpscQueueEndpointBinding.from_scalars(wrong_major)
    wrong_minor = list(_BINDING_GOLDEN)
    wrong_minor[0] = (_SPSC_QUEUE_MAGIC << 32) | (1 << 16) | 1
    with pytest.raises(ValueError, match="SPSQ ABI 1.0"):
        SpscQueueEndpointBinding.from_scalars(wrong_minor)


class _UnknownTemplateSlot(Enum):
    EXTRA = "extra"


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
    def __init__(self, region: _MemoryRegion, offset: int) -> None:
        self._region = region
        self._offset = int(offset)

    def test(self, cmp_value: int, cmp: WaitCmp) -> SignalTestResult:
        observed = int(self._region.counters.get(self._offset, 0))
        return SignalTestResult(matched=_compare_counter(observed, int(cmp_value), WaitCmp(cmp)), observed=observed)

    def wait(self, cmp_value: int, cmp: WaitCmp, timeout: float) -> int:
        if timeout is None or float(timeout) <= 0:
            raise ValueError("region counter wait requires a positive timeout")
        if self._region.on_wait is not None:
            self._region.on_wait(self._offset)
        result = self.test(cmp_value, cmp)
        if result.matched:
            return int(result.observed)
        raise TimeoutError(f"queue counter wait timed out; observed={result.observed}")

    def notify(self, value: int, op: NotifyOp = NotifyOp.Set) -> None:
        if self._region.fail_notify is not None:
            raise self._region.fail_notify
        if op is NotifyOp.Add:
            self._region.counters[self._offset] = int(self._region.counters.get(self._offset, 0)) + int(value)
        else:
            self._region.counters[self._offset] = int(value) & 0xFFFF_FFFF


@dataclass
class _MemoryRegion:
    layout: RegionLayoutSpec
    consumer: EndpointRecord
    provider: EndpointRecord
    session: bytes
    transaction_id: int
    payload_base: int = 0x1000
    counter_base: int = 0x2000
    payload: bytearray = None  # type: ignore[assignment]
    counters: dict[int, int] = None  # type: ignore[assignment]
    on_wait: object = None
    fail_notify: BaseException | None = None
    fail_payload: BaseException | None = None
    _state: RegionInstanceState = RegionInstanceState.LIVE

    def __post_init__(self) -> None:
        if self.payload is None:
            self.payload = bytearray(int(self.layout.payload_bytes))
        if self.counters is None:
            self.counters = {}

    @property
    def state(self) -> RegionInstanceState:
        return self._state

    @property
    def _allocation_identity(self) -> tuple[bytes, int]:
        if len(bytes(self.session)) != 8 or type(self.transaction_id) is not int or self.transaction_id < 1:
            raise MaterializationError("region allocation identity is incomplete")
        return bytes(self.session), int(self.transaction_id)

    def local_view(self, part: RegionPartKind) -> RegionPartLocalView | None:
        if self._state is not RegionInstanceState.LIVE:
            return None
        if part is RegionPartKind.PAYLOAD:
            return RegionPartLocalView(RegionPartKind.PAYLOAD, self.payload_base, int(self.layout.payload_bytes))
        if part is RegionPartKind.COUNTER:
            return RegionPartLocalView(RegionPartKind.COUNTER, self.counter_base, int(self.layout.counter_bytes))
        raise ValueError("region part must be PAYLOAD or COUNTER")

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


def _endpoint_bundle():
    session = bytes(range(8))
    host = EndpointRecord(EndpointIdentity(session, 1, 0), "L3", HOST_CPU, 0)
    peer = EndpointRecord(EndpointIdentity(session, 1, 1), "L3/L2[1]", DEVICE_AICPU, 0)
    extra = EndpointRecord(EndpointIdentity(session, 1, 2), "L3/L2[0]", DEVICE_AICPU, 0)
    registry = EndpointRegistry(
        root_level=3, session_instance_id=session, registry_epoch=1, records=(host, peer, extra)
    )
    return session, host, peer, extra, registry


def _bind_memory_queue(depth=4, input_arena_bytes=128, output_arena_bytes=192, **region_kwargs):
    session, host, peer, _extra, registry = _endpoint_bundle()
    config = _SpscQueueConfig(depth, input_arena_bytes, output_arena_bytes)
    plan = _SpscQueueTemplate().plan(config)
    members = registry.resolve_members((at(host.path, HOST_CPU), at(peer.path, DEVICE_AICPU)))
    slots = _resolve_template_slots(
        registry, members, _queue_placement(host, peer).slot_bindings, _SpscQueueTemplate.required_slots
    )
    region = _MemoryRegion(
        layout=plan.region_layout,
        consumer=host,
        provider=peer,
        session=session,
        transaction_id=42,
        **region_kwargs,
    )
    return plan, region, slots, plan._bind(region, slots)


def _queue_placement(host: EndpointRecord, peer: EndpointRecord) -> _RegionTemplatePlacementRequest:
    return _RegionTemplatePlacementRequest(
        members=(at(host.path, host.deployment), at(peer.path, peer.deployment)),
        topology=SingleOwner(provider=at(peer.path, peer.deployment)),
        slot_bindings=(
            _TemplateSlotBindingRequest(_SpscQueueSlot.INITIATOR, at(host.path, host.deployment)),
            _TemplateSlotBindingRequest(_SpscQueueSlot.PEER, at(peer.path, peer.deployment)),
        ),
    )


def test_plan_is_deterministic_and_has_no_runtime_side_effect():
    config = _SpscQueueConfig(4, 128, 192)
    first = _SpscQueueTemplate().plan(config)
    second = _SpscQueueTemplate().plan(config)
    assert first.region_layout == second.region_layout
    assert first._layout == second._layout
    assert first._state is _SpscQueuePlanState.AVAILABLE
    assert second._state is _SpscQueuePlanState.AVAILABLE
    assert first.region_layout == RegionLayoutSpec(payload_bytes=576, counter_bytes=_SPSC_QUEUE_COUNTER_BYTES)


def test_plan_refuses_copy_deepcopy_and_pickle():
    plan = _SpscQueueTemplate().plan(_SpscQueueConfig(4, 128, 128))
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(plan)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.deepcopy(plan)
    with pytest.raises(TypeError, match="cannot be copied"):
        pickle.dumps(plan)


def test_slot_resolve_rejects_missing_duplicate_unknown_and_non_member():
    _session, host, peer, extra, registry = _endpoint_bundle()
    members = registry.resolve_members((at("L3", HOST_CPU), at("L3/L2[1]", DEVICE_AICPU)))
    required = _SpscQueueTemplate.required_slots
    with pytest.raises(ValueError, match="missing"):
        _resolve_template_slots(
            registry,
            members,
            (_TemplateSlotBindingRequest(_SpscQueueSlot.INITIATOR, at("L3", HOST_CPU)),),
            required,
        )
    with pytest.raises(ValueError, match="duplicate"):
        _resolve_template_slots(
            registry,
            members,
            (
                _TemplateSlotBindingRequest(_SpscQueueSlot.INITIATOR, at("L3", HOST_CPU)),
                _TemplateSlotBindingRequest(_SpscQueueSlot.INITIATOR, at("L3/L2[1]", DEVICE_AICPU)),
            ),
            required,
        )
    with pytest.raises(ValueError, match="unknown"):
        _resolve_template_slots(
            registry,
            members,
            (
                _TemplateSlotBindingRequest(_SpscQueueSlot.INITIATOR, at("L3", HOST_CPU)),
                _TemplateSlotBindingRequest(_SpscQueueSlot.PEER, at("L3/L2[1]", DEVICE_AICPU)),
                _TemplateSlotBindingRequest(_UnknownTemplateSlot.EXTRA, at("L3/L2[0]", DEVICE_AICPU)),
            ),
            required,
        )
    with pytest.raises(ValueError, match="not a region member"):
        _resolve_template_slots(
            registry,
            members,
            (
                _TemplateSlotBindingRequest(_SpscQueueSlot.INITIATOR, at("L3", HOST_CPU)),
                _TemplateSlotBindingRequest(_SpscQueueSlot.PEER, at("L3/L2[0]", DEVICE_AICPU)),
            ),
            required,
        )
    with pytest.raises(ValueError, match="different endpoint"):
        same = _resolve_template_slots(
            registry,
            registry.resolve_members((at("L3", HOST_CPU),)),
            (
                _TemplateSlotBindingRequest(_SpscQueueSlot.INITIATOR, at("L3", HOST_CPU)),
                _TemplateSlotBindingRequest(_SpscQueueSlot.PEER, at("L3", HOST_CPU)),
            ),
            required,
        )
        _require_distinct_initiator_peer(same)


def test_concurrent_first_bind_consumes_once_and_second_bind_does_not_touch_instance():
    session, host, peer, _extra, registry = _endpoint_bundle()
    plan = _SpscQueueTemplate().plan(_SpscQueueConfig(4, 128, 128))
    members = registry.resolve_members((at("L3", HOST_CPU), at("L3/L2[1]", DEVICE_AICPU)))
    slots = _resolve_template_slots(
        registry, members, _queue_placement(host, peer).slot_bindings, _SpscQueueTemplate.required_slots
    )
    region = _MemoryRegion(plan.region_layout, host, peer, session, 7)
    results: list[object] = []
    errors: list[BaseException] = []

    def attempt() -> None:
        try:
            results.append(plan._bind(region, slots))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 1
    assert isinstance(results[0], _BoundSpscQueue)
    assert plan._state is _SpscQueuePlanState.CONSUMED
    assert all("already consumed" in str(error) for error in errors)

    class _Boom:
        def __getattribute__(self, name: str):
            raise AssertionError(f"second bind touched {name}")

    with pytest.raises(RuntimeError, match="already consumed"):
        plan._bind(_Boom(), slots)


def test_bind_rejects_layout_mismatch_and_incomplete_identity():
    session, host, peer, _extra, registry = _endpoint_bundle()
    plan = _SpscQueueTemplate().plan(_SpscQueueConfig(4, 128, 128))
    members = registry.resolve_members((at("L3", HOST_CPU), at("L3/L2[1]", DEVICE_AICPU)))
    slots = _resolve_template_slots(
        registry, members, _queue_placement(host, peer).slot_bindings, _SpscQueueTemplate.required_slots
    )
    region = _MemoryRegion(RegionLayoutSpec(64, 384), host, peer, session, 7)
    with pytest.raises(ValueError, match="payload_bytes"):
        plan._bind(region, slots)
    assert plan._state is _SpscQueuePlanState.CONSUMED

    plan = _SpscQueueTemplate().plan(_SpscQueueConfig(4, 128, 128))
    region = _MemoryRegion(plan.region_layout, host, peer, session, 0)
    with pytest.raises(MaterializationError, match="allocation identity"):
        plan._bind(region, slots)


def test_bound_queue_refuses_copy_and_does_not_own_physical_cleanup():
    _plan, region, _slots, queue = _bind_memory_queue()
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(queue)
    with pytest.raises(TypeError, match="cannot be copied"):
        pickle.dumps(queue)
    queue.free()
    assert queue._state is _SpscQueueState.RELEASED
    assert region.state is RegionInstanceState.LIVE
    with pytest.raises(RuntimeError, match="released"):
        queue.input.try_enqueue(b"abc", 3)


def test_try_enqueue_oversize_and_blocking_oversize_do_not_touch_shared_state():
    _plan, region, _slots, queue = _bind_memory_queue(input_arena_bytes=64, output_arena_bytes=64)
    assert queue.input.try_enqueue(b"x" * 128, 128) is False
    assert region.counters == {}
    assert bytes(region.payload) == bytes(len(region.payload))
    with pytest.raises(ValueError, match="exceeds input arena"):
        queue.input.enqueue(None, 128, timeout=0.1)
    assert region.counters == {}


def test_duplex_data_zero_byte_wrap_replay_stop_and_error():
    _plan, region, _slots, queue = _bind_memory_queue(depth=4, input_arena_bytes=64, output_arena_bytes=64)
    assert queue.input.try_enqueue(None, 0) is True
    assert queue.input.try_enqueue(b"x" * 48, 48) is True
    region.counters[queue._layout.input_desc_head_offset] = 2
    wrap_ok = queue.input.try_enqueue(b"y" * 32, 32)
    assert wrap_ok is True
    slot = queue._layout.input_desc_offset + 2 * _SPSC_QUEUE_DESCRIPTOR_BYTES
    seq, opcode, offset, nbytes = decode_spsc_queue_descriptor(bytes(region.payload[slot : slot + 32]))
    assert seq == 3
    assert opcode == int(SpscQueueOpcode.DATA)
    assert nbytes == 32
    assert offset == queue._layout.input_arena_offset

    assert queue.try_request_stop() is True
    assert queue.input.try_enqueue(b"no", 2) is False

    out_payload = b"err-data"
    out_offset = queue._layout.output_arena_offset
    region.payload[out_offset : out_offset + 8] = out_payload
    desc = encode_spsc_queue_descriptor(1, SpscQueueOpcode.ERROR, out_offset, 8)
    region.payload[queue._layout.output_desc_offset : queue._layout.output_desc_offset + 32] = desc
    region.counters[queue._layout.output_desc_tail_offset] = 1
    handle = queue.output.try_peek()
    assert handle is not None
    assert handle.opcode is SpscQueueOpcode.ERROR
    dest = bytearray(8)
    queue.output.read_into(handle, dest)
    assert bytes(dest) == out_payload
    queue.output.release(handle)

    data2 = b"more"
    desc2 = encode_spsc_queue_descriptor(2, SpscQueueOpcode.DATA, out_offset, 4)
    region.payload[out_offset : out_offset + 4] = data2
    region.payload[queue._layout.output_desc_offset + 32 : queue._layout.output_desc_offset + 64] = desc2
    region.counters[queue._layout.output_desc_tail_offset] = 2
    handle = queue.output.try_peek()
    assert handle is not None
    assert handle.opcode is SpscQueueOpcode.DATA
    dest = bytearray(4)
    queue.output.read_into(handle, dest)
    queue.output.release(handle)
    assert bytes(dest) == data2


def test_output_handle_ownership_and_wait_timeout_abort_failure():
    _plan, region, _slots, queue = _bind_memory_queue()
    with pytest.raises(TimeoutError, match="timed out"):
        queue.output.peek(0.01)
    assert queue._state is _SpscQueueState.LIVE

    def set_abort(_offset: int) -> None:
        region.counters[queue._layout.peer_abort_offset] = 1

    region.on_wait = set_abort
    with pytest.raises(RuntimeError, match="remote abort"):
        queue.output.peek(0.01)
    assert queue._state is _SpscQueueState.POISONED_REMOTE

    _plan, region, _slots, queue = _bind_memory_queue()
    out_offset = queue._layout.output_arena_offset
    region.payload[out_offset : out_offset + 3] = b"abc"
    region.payload[queue._layout.output_desc_offset : queue._layout.output_desc_offset + 32] = (
        encode_spsc_queue_descriptor(1, SpscQueueOpcode.DATA, out_offset, 3)
    )
    region.counters[queue._layout.output_desc_tail_offset] = 1
    handle = queue.output.try_peek()
    assert handle is not None
    forged = _SpscQueueMessage(handle.seq, handle.opcode, handle.payload_offset, handle.payload_nbytes)
    with pytest.raises(RuntimeError, match="not active"):
        queue.output.release(forged)
    assert queue._state is _SpscQueueState.POISONED_LOCAL
    assert queue._layout.initiator_abort_offset in region.counters
    assert region.counters[queue._layout.initiator_abort_offset] == 1

    _plan, region, _slots, queue = _bind_memory_queue()
    region.fail_payload = RuntimeError("payload write failed")
    region.fail_notify = RuntimeError("abort notify failed")
    with pytest.raises(RuntimeError, match="payload write failed") as excinfo:
        queue.input.try_enqueue(b"hi", 2)
    assert queue._state is _SpscQueueState.POISONED_LOCAL
    assert queue._first_error is excinfo.value
    notes = getattr(excinfo.value, "__notes__", [])
    assert any("abort notify failed" in note for note in notes)


def test_zero_byte_enqueue_rejects_non_none_and_stop_is_zero_byte():
    _plan, _region, _slots, queue = _bind_memory_queue()
    with pytest.raises(ValueError, match="buffer_or_none == None"):
        queue.input.try_enqueue(b"", 0)
    with pytest.raises(ValueError, match="STOP must be zero-byte"):
        queue.input._try_enqueue(None, 4, SpscQueueOpcode.STOP)


def test_exact_depth_capacity_blocking_wake_dequeue_and_expiration():
    _plan, region, _slots, queue = _bind_memory_queue(depth=1, input_arena_bytes=64, output_arena_bytes=64)
    writes: list[tuple[int, int]] = []
    original_write = region.payload_write

    def record_write(offset, host_buffer, nbytes=None):
        writes.append((int(offset), int(nbytes if nbytes is not None else len(host_buffer))))
        return original_write(offset, host_buffer, nbytes)

    region.payload_write = record_write  # type: ignore[method-assign]
    assert queue.input.try_enqueue(b"abcd", 4) is True
    assert writes[0][0] == queue._layout.input_arena_offset
    assert writes[1][0] == 8
    assert writes[2][0] == 0
    assert queue.input.try_enqueue(b"efgh", 4) is False

    def advance_head(_offset: int) -> None:
        region.counters[queue._layout.input_desc_head_offset] = 1

    region.on_wait = advance_head
    queue.input.enqueue(b"ijkl", 4, timeout=1.0)
    assert queue._input_tail == 2

    out_offset = queue._layout.output_arena_offset
    region.payload[out_offset : out_offset + 4] = b"wxyz"
    region.payload[queue._layout.output_desc_offset : queue._layout.output_desc_offset + 32] = (
        encode_spsc_queue_descriptor(1, SpscQueueOpcode.DATA, out_offset, 4)
    )
    region.counters[queue._layout.output_desc_tail_offset] = 1
    dest = bytearray(4)
    message = queue.output.try_dequeue_into(dest)
    assert message is not None
    assert bytes(dest) == b"wxyz"
    with pytest.raises(RuntimeError, match="not active"):
        queue.output.release(message)

    _plan, region, _slots, queue = _bind_memory_queue()
    region._state = RegionInstanceState.CLOSED
    with pytest.raises(RuntimeError, match="expired"):
        queue.input.try_enqueue(b"ab", 2)
    assert queue._state is _SpscQueueState.EXPIRED
    assert region.counters == {}


def test_output_stop_poisons_and_handle_invalidates_after_release():
    _plan, region, _slots, queue = _bind_memory_queue()
    region.payload[queue._layout.output_desc_offset : queue._layout.output_desc_offset + 32] = (
        encode_spsc_queue_descriptor(1, SpscQueueOpcode.STOP, 0, 0)
    )
    region.counters[queue._layout.output_desc_tail_offset] = 1
    with pytest.raises(RuntimeError, match="DATA or ERROR"):
        queue.output.try_peek()
    assert queue._state is _SpscQueueState.POISONED_LOCAL


class _FakeLease:
    def __init__(self, calls: list[tuple], name: str, handle: int, *, fail_close: bool = False) -> None:
        self._calls = calls
        self._name = name
        self.handle = handle
        self.closed = False
        self._fail_close = fail_close

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._calls.append(("mapping_close", self._name))
        if self._fail_close:
            raise RuntimeError("mapping close failed")


class _FakeNativeWorker:
    def __init__(self, calls: list[tuple], *, fail_release: bool = False) -> None:
        self._calls = calls
        self._fail_release = fail_release
        self._last_resource_id = 42

    def control_payload(self, _worker_type, worker_id, sub_cmd, payload, _timeout):
        from simpler.comm_provider import (
            PosixShmImport,
            ProviderReleaseResult,
            ProviderReleaseStatus,
            RegionAllocationResult,
            RegionExportDescriptor,
            RegionPartExportDescriptor,
            RegionPartKind,
            RegionPartLocalView,
        )
        from simpler.comm_provider_control import (
            DelegatedAllocateReply,
            DelegatedAllocateReplyTag,
            DelegatedRegionOperation,
            DelegatedReleaseReply,
            DelegatedReleaseReplyTag,
            encode_reply,
            parse_request,
            publish_reply,
        )
        from simpler.worker import _CTRL_DELEGATED_REGION

        assert int(sub_cmd) == _CTRL_DELEGATED_REGION
        staged = bytearray(payload)
        envelope = parse_request(staged)
        if envelope.operation is DelegatedRegionOperation.DELEGATED_ALLOCATE:
            request = envelope.decode_terminal()
            spec = request.spec
            self._calls.append(("allocate", int(spec.payload.logical_bytes), int(spec.counter.logical_bytes)))
            result = RegionAllocationResult(
                provider_resource_id=42,
                export_descriptor=RegionExportDescriptor(
                    payload=RegionPartExportDescriptor(
                        spec.payload.planned_backing_kind,
                        int(spec.payload.logical_bytes),
                        int(spec.payload.logical_bytes),
                        PosixShmImport("/pto_payload_42"),
                    ),
                    counter=RegionPartExportDescriptor(
                        spec.counter.planned_backing_kind,
                        int(spec.counter.logical_bytes),
                        int(spec.counter.logical_bytes),
                        PosixShmImport("/pto_counter_42"),
                    ),
                ),
            )
            payload_view = RegionPartLocalView(RegionPartKind.PAYLOAD, 0x1000, int(spec.payload.logical_bytes))
            counter_view = RegionPartLocalView(RegionPartKind.COUNTER, 0x2000, int(spec.counter.logical_bytes))
            committed = encode_reply(
                DelegatedAllocateReply(
                    tag=DelegatedAllocateReplyTag.ALLOCATED,
                    session_instance_id=envelope.session_instance_id,
                    transaction_id=envelope.transaction_id,
                    result=result,
                    payload_view=payload_view,
                    counter_view=counter_view,
                )
            )
            publish_reply(memoryview(staged), committed)
            return bytes(staged)
        self._calls.append(("release", int(worker_id), int(self._last_resource_id)))
        request = envelope.decode_terminal()
        committed = encode_reply(
            DelegatedReleaseReply(
                tag=DelegatedReleaseReplyTag.RELEASED,
                session_instance_id=envelope.session_instance_id,
                transaction_id=request.transaction_id,
                result=ProviderReleaseResult(
                    provider_resource_id=int(self._last_resource_id),
                    status=ProviderReleaseStatus.RELEASED,
                ),
            )
        )
        publish_reply(memoryview(staged), committed)
        if self._fail_release:
            raise RuntimeError("release failed")
        return bytes(staged)


@pytest.fixture
def region_worker(monkeypatch):
    def build(*, fail_mapping_close: bool = False, fail_release: bool = False, device_ids=(8, 9)):
        worker = Worker(level=3, device_ids=list(device_ids))
        worker._lifecycle = _Lifecycle.READY
        worker._worker = object()
        worker._next_level_worker_ids = list(range(len(device_ids)))
        worker._config = {**worker._config, "platform": "a2a3sim", "device_ids": list(device_ids)}
        calls: list[tuple] = []
        leases: list[_FakeLease] = []
        worker._worker = _FakeNativeWorker(calls, fail_release=fail_release)
        monkeypatch.setattr(worker, "_consume_worker_host_mapped_cleanup_error", lambda _api: None)

        def fake_import(_worker_id, _resource_id, export):
            name = "payload" if not leases else "counter"
            lease = _FakeLease(calls, name, handle=100 + len(leases), fail_close=fail_mapping_close)
            calls.append(("import", name, int(export.logical_bytes)))
            leases.append(lease)
            return lease

        monkeypatch.setattr(worker, "_import_region_part_lease", fake_import)
        return worker, calls, leases

    return build


def test_coordinator_create_projects_bound_queue_without_escaping_instance(region_worker):
    worker, calls, _leases = region_worker()
    captured: dict[str, object] = {}

    def projector(bound):
        captured["bound"] = bound
        captured["instance"] = bound._instance
        return bound

    host_sel = at("L3", HOST_CPU)
    peer_sel = at("L3/L2[1]", DEVICE_AICPU)
    placement = _RegionTemplatePlacementRequest(
        members=(host_sel, peer_sel),
        topology=SingleOwner(provider=peer_sel),
        slot_bindings=(
            _TemplateSlotBindingRequest(_SpscQueueSlot.INITIATOR, host_sel),
            _TemplateSlotBindingRequest(_SpscQueueSlot.PEER, peer_sel),
        ),
    )
    result = _RegionTemplateCoordinator(worker).create(
        template=_SpscQueueTemplate(),
        config=_SpscQueueConfig(4, 128, 192),
        placement=placement,
        result_projector=projector,
    )
    assert result is captured["bound"]
    assert isinstance(result, _BoundSpscQueue)
    assert result is not captured["instance"]
    assert result.endpoint_binding.to_scalars()[0] == _SPSC_QUEUE_MAGIC_VERSION
    assert len(result.endpoint_binding.to_scalars()) == 10
    assert result._desc_fields == bytearray(24)
    assert result._desc_seq == bytearray(8)
    assert result._desc_read == bytearray(32)
    assert ("allocate", 576, 384) in calls
    assert not any(item[0] == "counter_notify" for item in calls)


def test_coordinator_bind_and_projector_failure_close_instance(region_worker):
    worker, calls, _leases = region_worker()
    host_sel = at("L3", HOST_CPU)
    peer_sel = at("L3/L2[1]", DEVICE_AICPU)
    placement = _RegionTemplatePlacementRequest(
        members=(host_sel, peer_sel),
        topology=SingleOwner(provider=peer_sel),
        slot_bindings=(
            _TemplateSlotBindingRequest(_SpscQueueSlot.INITIATOR, host_sel),
            _TemplateSlotBindingRequest(_SpscQueueSlot.PEER, peer_sel),
        ),
    )

    def fail_bind(self, instance, slots):
        self._consume_for_bind()
        raise RuntimeError("injected bind failure")

    original = _SpscQueuePlan._bind
    _SpscQueuePlan._bind = fail_bind
    try:
        with pytest.raises(RuntimeError, match="injected bind failure"):
            _RegionTemplateCoordinator(worker).create(
                template=_SpscQueueTemplate(),
                config=_SpscQueueConfig(4, 64, 64),
                placement=placement,
                result_projector=lambda bound: bound,
            )
    finally:
        _SpscQueuePlan._bind = original
    assert any(item[0] == "release" for item in calls)
    assert worker._region_instance_registry._instances == {}


def test_coordinator_cleanup_failure_records_unreclaimable_and_keeps_first_error(region_worker):
    worker, calls, _leases = region_worker(fail_mapping_close=True)
    host_sel = at("L3", HOST_CPU)
    peer_sel = at("L3/L2[1]", DEVICE_AICPU)
    placement = _RegionTemplatePlacementRequest(
        members=(host_sel, peer_sel),
        topology=SingleOwner(provider=peer_sel),
        slot_bindings=(
            _TemplateSlotBindingRequest(_SpscQueueSlot.INITIATOR, host_sel),
            _TemplateSlotBindingRequest(_SpscQueueSlot.PEER, peer_sel),
        ),
    )
    with pytest.raises(RuntimeError, match="projector failed") as excinfo:
        _RegionTemplateCoordinator(worker).create(
            template=_SpscQueueTemplate(),
            config=_SpscQueueConfig(4, 64, 64),
            placement=placement,
            result_projector=lambda _bound: (_ for _ in ()).throw(RuntimeError("projector failed")),
        )
    assert str(excinfo.value) == "projector failed"
    assert worker._ordered_cleanup_error is not None


def test_production_facade_uses_coordinator():
    from simpler import worker_chip_message_queue
    from simpler.worker_chip_message_queue import create_worker_chip_queue

    source = inspect.getsource(create_worker_chip_queue)
    assert "_RegionTemplateCoordinator" in source
    assert "create_worker_chip_region" not in source
    module_source = inspect.getsource(worker_chip_message_queue)
    assert "time.sleep" not in module_source
    assert "orch.alloc" not in module_source
