# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Unit tests for the private region materializer."""

import dataclasses
import struct
import threading
from types import SimpleNamespace
from typing import Any, Optional, Union

import pytest
from simpler import comm_endpoints as ce
from simpler.comm_provider import (
    ProviderReleaseResult,
    ProviderReleaseStatus,
    RegionAllocationError,
    RegionControlErrorKind,
    RegionOperationKind,
    RegionPartKind,
)
from simpler.comm_region import (
    CounterPart,
    DelegatedSingleOwnerRegionShape,
    HostVmmCopyAccess,
    MaterializationContext,
    MaterializationError,
    MaterializationRefusal,
    NotifyOp,
    PayloadPart,
    RefusalReason,
    RegionCounter,
    RegionInstance,
    RegionInstanceRegistry,
    RegionInstanceState,
    RegionPartSpan,
    SignalTestResult,
    WaitCmp,
    materialize_region_instance,
    project_region_allocation_spec,
    validate_single_owner_region_shape,
)
from simpler.orchestrator import _callback_frame_for, _callback_run
from simpler.worker import Worker, _Lifecycle, _RunResources


def _ready(worker: Worker) -> Worker:
    worker._lifecycle = _Lifecycle.READY
    return worker


def _l3(device_ids=(0,)) -> Worker:
    worker = _ready(Worker(level=3, device_ids=list(device_ids)))
    worker._worker = object()
    worker._next_level_worker_ids = list(range(len(device_ids)))
    return worker


def _l4_with_local_l3(device_ids=(0,)) -> Worker:
    child = Worker(level=3, device_ids=list(device_ids), num_sub_workers=0)
    worker = Worker(level=4, num_sub_workers=0)
    worker.add_worker(child)
    return _ready(worker)


def _context(worker: Worker, members, topology, layout=None) -> MaterializationContext:
    layout = layout or ce.RegionLayoutSpec(payload_bytes=64, counter_bytes=128)
    registry = worker._get_endpoint_registry()
    resolved = registry.resolve_region_spec(members, topology)
    plan = ce.BackendResolver(registry, worker._get_region_access_service()).plan(resolved, layout)
    return MaterializationContext(
        worker=worker,
        registry=registry,
        plan=plan,
        layout=layout,
    )


def _accepted_context(worker: Optional[Worker] = None) -> MaterializationContext:
    worker = worker or _l3(device_ids=[8, 9])
    return _context(
        worker,
        [ce.at("L3", ce.HOST_CPU), ce.at("L3/L2[1]", ce.DEVICE_AICPU)],
        ce.SingleOwner(provider=ce.at("L3/L2[1]", ce.DEVICE_AICPU)),
    )


def _assert_refusal(ctx: MaterializationContext, reason: RefusalReason) -> None:
    with pytest.raises(MaterializationRefusal) as excinfo:
        validate_single_owner_region_shape(ctx)
    assert excinfo.value.reason is reason


def _attachments(part: ce.RegionPartPlan) -> dict[ce.EndpointIdentity, ce.MemberAttachmentPlan]:
    return {attachment.member: attachment for attachment in part.attachments}


def _materialize_default_region(worker: Worker):
    return worker._materialize_region_instance(
        [ce.at("L3", ce.HOST_CPU), ce.at("L3/L2[1]", ce.DEVICE_AICPU)],
        ce.SingleOwner(provider=ce.at("L3/L2[1]", ce.DEVICE_AICPU)),
        ce.RegionLayoutSpec(payload_bytes=64, counter_bytes=128),
    )


def _tracked(worker: Worker) -> tuple[RegionInstance, ...]:
    return tuple(worker._region_instance_registry._instances.values())


def _manual_two_member_context(
    worker: Worker,
    provider: ce.EndpointRecord,
    consumer: ce.EndpointRecord,
    layout: Optional[ce.RegionLayoutSpec] = None,
) -> MaterializationContext:
    layout = layout or ce.RegionLayoutSpec(payload_bytes=64, counter_bytes=128)
    registry = ce.EndpointRegistry(
        root_level=3,
        session_instance_id=worker._owner_instance_id,
        registry_epoch=worker._endpoint_registry_epoch,
        records=(consumer, provider),
    )
    provider_attachment = ce.MemberAttachmentPlan(
        member=provider.identity,
        role=ce.AttachmentRole.PROVIDER,
        adapter_kind=None,
        adapter_profile=None,
    )
    consumer_attachment = ce.MemberAttachmentPlan(
        member=consumer.identity,
        role=ce.AttachmentRole.CONSUMER,
        adapter_kind=ce.AdapterKind.OWNER_DELEGATED_COPY,
        adapter_profile=ce.AdapterProfile.HOST_VMM_COPY,
    )
    part_attachments = (provider_attachment, consumer_attachment)
    plan = ce.BackendPlan(
        ordered_members=(consumer.identity, provider.identity),
        payload=ce.RegionPartPlan(
            part=ce.RegionPartKind.PAYLOAD,
            backend_kind=ce.BackendKind.VMM_WINDOW,
            attachments=part_attachments,
        ),
        counter=ce.RegionPartPlan(
            part=ce.RegionPartKind.COUNTER,
            backend_kind=ce.BackendKind.VMM_WINDOW,
            attachments=part_attachments,
        ),
        topology_plan=ce.SingleOwnerPlan(provider_endpoint=provider.identity),
    )
    return MaterializationContext(worker=worker, registry=registry, plan=plan, layout=layout)


def test_payload_part_validates_before_issue_and_uses_byte_span():
    access = _FakeAccess()
    part = PayloadPart(RegionPartSpan(offset=16, nbytes=8), access)
    payload = bytearray(b"abcdefgh")

    part.write(4, payload, 4)

    assert len(access.write_calls) == 1
    span, offset, _host_ptr, nbytes = access.write_calls[0]
    assert span == RegionPartSpan(offset=16, nbytes=8)
    assert (offset, nbytes) == (4, 4)

    with pytest.raises(ValueError, match="payload range"):
        part.write(6, payload, 4)
    assert len(access.write_calls) == 1


def test_host_vmm_copy_access_passes_absolute_mapping_offsets(monkeypatch):
    calls: list[tuple[str, int, int, int, int]] = []

    monkeypatch.setattr(
        "simpler.comm_region._host_vmm_copy_to",
        lambda handle, offset, host_ptr, nbytes: calls.append(("write", handle, offset, host_ptr, nbytes)),
    )
    monkeypatch.setattr(
        "simpler.comm_region._host_vmm_copy_from",
        lambda handle, offset, host_ptr, nbytes: calls.append(("read", handle, offset, host_ptr, nbytes)),
    )

    access = HostVmmCopyAccess(handle=7)
    span = RegionPartSpan(offset=64, nbytes=16)
    access.write_bytes(span, 4, 1234, 8)
    access.read_bytes(span, 8, 5678, 4)

    assert calls == [("write", 7, 68, 1234, 8), ("read", 7, 72, 5678, 4)]


def test_counter_part_owns_signal_semantics(monkeypatch):
    access = _FakeAccess()
    part = CounterPart(RegionPartSpan(offset=64, nbytes=16), access, handle=11)
    calls: list[tuple[Any, ...]] = []

    monkeypatch.setattr(
        "simpler.comm_region._region_counter_notify",
        lambda handle, offset, value, op: calls.append(("notify", handle, offset, value, op)),
    )
    monkeypatch.setattr(
        "simpler.comm_region._region_counter_test",
        lambda handle, offset, value, cmp: calls.append(("test", handle, offset, value, cmp)) or (False, 3),
    )
    monkeypatch.setattr(
        "simpler.comm_region._region_counter_wait",
        lambda handle, offset, value, cmp, timeout_ns: calls.append(("wait", handle, offset, value, cmp, timeout_ns))
        or (0, 0, 9, True, ""),
    )

    counter = part.counter(4)
    assert isinstance(counter, RegionCounter)
    counter.notify(5, NotifyOp.Set)
    result = counter.test(7, WaitCmp.GE)
    observed = counter.wait(9, WaitCmp.EQ, 0.25)

    assert result == SignalTestResult(matched=False, observed=3)
    assert observed == 9
    assert calls == [
        ("notify", 11, 68, 5, int(NotifyOp.Set)),
        ("test", 11, 68, 7, int(WaitCmp.GE)),
        ("wait", 11, 68, 9, int(WaitCmp.EQ), 250_000_000),
    ]

    with pytest.raises(ValueError, match="counter offset"):
        part.counter(2)
    with pytest.raises(ValueError, match="positive timeout"):
        counter.wait(1, WaitCmp.EQ, 0)


class _FakeAccess:
    def __init__(self) -> None:
        self.write_calls: list[tuple[RegionPartSpan, int, int, int]] = []
        self.read_calls: list[tuple[RegionPartSpan, int, int, int]] = []
        self.fail_write = False

    def write_bytes(self, span: RegionPartSpan, offset: int, host_ptr: int, nbytes: int) -> None:
        self.write_calls.append((span, int(offset), int(host_ptr), int(nbytes)))
        if self.fail_write:
            raise RuntimeError("issued write failed")

    def read_bytes(self, span: RegionPartSpan, offset: int, host_ptr: int, nbytes: int) -> None:
        self.read_calls.append((span, int(offset), int(host_ptr), int(nbytes)))


def test_shape_validation_accepts_l3_host_to_local_l2_aicpu_copy_plan():
    ctx = _accepted_context()
    shape = validate_single_owner_region_shape(ctx)

    assert shape.consumer.path == "L3"
    assert shape.consumer.deployment is ce.HOST_CPU
    assert shape.provider.path == "L3/L2[1]"
    assert shape.provider.deployment is ce.DEVICE_AICPU
    assert shape.worker_id == 1


def test_shape_validation_accepts_l4_host_to_descendant_aicpu_plan():
    worker = _l4_with_local_l3(device_ids=[4])
    ctx = _context(
        worker,
        [ce.at("L4", ce.HOST_CPU), ce.at("L4/L3[0]/L2[0]", ce.DEVICE_AICPU)],
        ce.SingleOwner(provider=ce.at("L4/L3[0]/L2[0]", ce.DEVICE_AICPU)),
    )
    shape = validate_single_owner_region_shape(ctx)
    assert isinstance(shape, DelegatedSingleOwnerRegionShape)
    assert shape.consumer.path == "L4"
    assert shape.provider.path == "L4/L3[0]/L2[0]"
    assert shape.first_hop_child_id == 0
    assert shape.provider_device_id == 4


def test_shape_validation_rejects_aicore_provider_until_it_has_a_materializer():
    worker = _l3(device_ids=[0])
    ctx = _context(
        worker,
        [ce.at("L3", ce.HOST_CPU), ce.at("L3/L2[0]", ce.DEVICE_AICORE)],
        ce.SingleOwner(provider=ce.at("L3/L2[0]", ce.DEVICE_AICORE)),
    )

    _assert_refusal(ctx, RefusalReason.UNSUPPORTED_PROVIDER_DEPLOYMENT)


def test_shape_validation_rejects_non_l2_child_provider_path():
    worker = _l3(device_ids=[0])
    session_id = worker._owner_instance_id
    registry_epoch = worker._endpoint_registry_epoch
    consumer = ce.EndpointRecord(
        identity=ce.EndpointIdentity(session_id, registry_epoch, 0),
        path="L3",
        deployment=ce.HOST_CPU,
        node_scope_id=0,
    )
    provider = ce.EndpointRecord(
        identity=ce.EndpointIdentity(session_id, registry_epoch, 1),
        path="L3/L1[0]",
        deployment=ce.DEVICE_AICPU,
        node_scope_id=0,
    )

    _assert_refusal(_manual_two_member_context(worker, provider, consumer), RefusalReason.NEEDS_DELEGATION)


def test_shape_validation_translates_worker_chip_id_value_error():
    worker = _l3(device_ids=[0])
    session_id = worker._owner_instance_id
    registry_epoch = worker._endpoint_registry_epoch
    consumer = ce.EndpointRecord(
        identity=ce.EndpointIdentity(session_id, registry_epoch, 0),
        path="L3",
        deployment=ce.HOST_CPU,
        node_scope_id=0,
    )
    provider = ce.EndpointRecord(
        identity=ce.EndpointIdentity(session_id, registry_epoch, 1),
        path="L3/L2[1]",
        deployment=ce.DEVICE_AICPU,
        node_scope_id=0,
    )

    _assert_refusal(_manual_two_member_context(worker, provider, consumer), RefusalReason.UNSUPPORTED_MEMBER_SHAPE)


def test_shape_validation_rejects_unsupported_plan_and_non_vmm_backing():
    unsupported = _context(
        _l3(device_ids=[0]),
        [ce.at("L3", ce.HOST_CPU), ce.at("L3/L2[0]", ce.DEVICE_AICPU)],
        ce.SingleOwner(provider=ce.at("L3", ce.HOST_CPU)),
    )
    _assert_refusal(unsupported, RefusalReason.UNSUPPORTED_PLAN)

    ctx = _accepted_context()
    posix_payload = dataclasses.replace(ctx.plan.payload, backend_kind=ce.BackendKind.POSIX_SHM)
    _assert_refusal(
        dataclasses.replace(ctx, plan=dataclasses.replace(ctx.plan, payload=posix_payload)),
        RefusalReason.UNSUPPORTED_BACKEND_KIND,
    )


def test_shape_validation_rejects_direct_map_and_extra_consumers():
    ctx = _accepted_context()
    host = validate_single_owner_region_shape(ctx).consumer.identity
    payload_by_member = _attachments(ctx.plan.payload)
    payload_by_member[host] = dataclasses.replace(
        payload_by_member[host],
        adapter_kind=ce.AdapterKind.DIRECT_MAP,
        adapter_profile=ce.AdapterProfile.HOST_SVM_MAP,
    )
    direct_map_payload = dataclasses.replace(ctx.plan.payload, attachments=tuple(payload_by_member.values()))
    _assert_refusal(
        dataclasses.replace(ctx, plan=dataclasses.replace(ctx.plan, payload=direct_map_payload)),
        RefusalReason.UNSUPPORTED_ATTACHMENT,
    )

    worker = _l3(device_ids=[0])
    decisions: dict[
        tuple[Any, ce.AdapterKind, ce.AdapterProfile],
        Union[ce.RegionAccessDecision, bool],
    ] = {
        (
            ce.BackendKind.VMM_WINDOW,
            ce.AdapterKind.OWNER_DELEGATED_COPY,
            ce.AdapterProfile.HOST_VMM_COPY,
        ): True,
        (
            ce.BackendKind.VMM_WINDOW,
            ce.AdapterKind.DEVICE_PEER,
            ce.AdapterProfile.DEVICE_VMM_PEER_IMPORT,
        ): True,
    }
    worker._region_access_service = ce.StaticRegionAccessService(decisions)
    extra = _context(
        worker,
        [ce.at("L3", ce.HOST_CPU), ce.at("L3/L2[0]", ce.DEVICE_AICPU), ce.at("L3/L2[0]", ce.DEVICE_AICORE)],
        ce.SingleOwner(provider=ce.at("L3/L2[0]", ce.DEVICE_AICPU)),
    )
    _assert_refusal(extra, RefusalReason.UNSUPPORTED_MEMBER_SHAPE)


def test_shape_validation_rejects_remote_endpoint_member():
    worker = _l3(device_ids=[0])
    session_id = worker._owner_instance_id
    registry_epoch = worker._endpoint_registry_epoch
    consumer = ce.EndpointRecord(
        identity=ce.EndpointIdentity(session_id, registry_epoch, 0),
        path="L3",
        deployment=ce.HOST_CPU,
        node_scope_id=0,
    )
    remote_provider = ce.EndpointRecord(
        identity=ce.EndpointIdentity(session_id, registry_epoch, 1),
        path="L3/L2[0]",
        deployment=ce.DEVICE_AICPU,
        node_scope_id=1,
    )

    _assert_refusal(
        _manual_two_member_context(worker, remote_provider, consumer),
        RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
    )


def test_shape_validation_rejects_device_peer_consumer_attachment():
    ctx = _accepted_context()
    host = validate_single_owner_region_shape(ctx).consumer.identity
    payload_by_member = _attachments(ctx.plan.payload)
    payload_by_member[host] = dataclasses.replace(
        payload_by_member[host],
        adapter_kind=ce.AdapterKind.DEVICE_PEER,
        adapter_profile=ce.AdapterProfile.DEVICE_VMM_PEER_IMPORT,
    )
    device_peer_payload = dataclasses.replace(ctx.plan.payload, attachments=tuple(payload_by_member.values()))

    _assert_refusal(
        dataclasses.replace(ctx, plan=dataclasses.replace(ctx.plan, payload=device_peer_payload)),
        RefusalReason.UNSUPPORTED_ATTACHMENT,
    )


def test_shape_validation_rejects_duplicate_attachment_members():
    ctx = _accepted_context()
    duplicate_payload = dataclasses.replace(
        ctx.plan.payload,
        attachments=ctx.plan.payload.attachments + (ctx.plan.payload.attachments[-1],),
    )

    _assert_refusal(
        dataclasses.replace(ctx, plan=dataclasses.replace(ctx.plan, payload=duplicate_payload)),
        RefusalReason.UNSUPPORTED_ATTACHMENT,
    )


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
    def __init__(
        self,
        calls: list[tuple],
        *,
        fail_release: bool = False,
        allocate_error: Optional[BaseException] = None,
        mutate_success=None,
    ) -> None:
        self._calls = calls
        self._fail_release = fail_release
        self._allocate_error = allocate_error
        self._mutate_success = mutate_success
        self._last_resource_id = 42

    def control_payload(self, _worker_type, worker_id, sub_cmd, payload, _timeout):
        from simpler.comm_provider import ProviderReleaseResult, ProviderReleaseStatus, RegionAllocationError
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
            if self._allocate_error is not None:
                if isinstance(self._allocate_error, RegionAllocationError):
                    committed = encode_reply(
                        DelegatedAllocateReply(
                            tag=DelegatedAllocateReplyTag.ERROR,
                            session_instance_id=envelope.session_instance_id,
                            transaction_id=envelope.transaction_id,
                            error=self._allocate_error,
                        )
                    )
                    publish_reply(memoryview(staged), committed)
                    return bytes(staged)
                raise self._allocate_error
            result, payload_view, counter_view = _committed_success(spec)
            self._last_resource_id = int(result.provider_resource_id)
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
            if self._mutate_success is not None:
                self._mutate_success(staged)
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


def _committed_success(spec, *, resource_id: int = 42, shm_names=None, payload_base: int = 0, counter_base: int = 64):
    from simpler.comm_provider import (
        PosixShmImport,
        RegionAllocationResult,
        RegionExportDescriptor,
        RegionPartExportDescriptor,
        RegionPartKind,
        RegionPartLocalView,
    )

    names = shm_names or (f"/pto_payload_{resource_id}", f"/pto_counter_{resource_id}")
    result = RegionAllocationResult(
        provider_resource_id=int(resource_id),
        export_descriptor=RegionExportDescriptor(
            payload=RegionPartExportDescriptor(
                spec.payload.planned_backing_kind,
                int(spec.payload.logical_bytes),
                int(spec.payload.logical_bytes),
                PosixShmImport(names[0]),
            ),
            counter=RegionPartExportDescriptor(
                spec.counter.planned_backing_kind,
                int(spec.counter.logical_bytes),
                int(spec.counter.logical_bytes),
                PosixShmImport(names[1]),
            ),
        ),
    )
    payload_view = RegionPartLocalView(RegionPartKind.PAYLOAD, int(payload_base), int(spec.payload.logical_bytes))
    counter_view = RegionPartLocalView(RegionPartKind.COUNTER, int(counter_base), int(spec.counter.logical_bytes))
    return result, payload_view, counter_view


@pytest.fixture
def region_worker(monkeypatch):
    def build(
        *,
        fail_mapping_close: bool = False,
        fail_release: bool = False,
        fail_first_import: bool = False,
        fail_second_import: bool = False,
        allocate_error: Optional[BaseException] = None,
        mutate_success=None,
        device_ids=(8, 9),
    ):
        worker = _l3(device_ids=device_ids)
        worker._config = {**worker._config, "platform": "a2a3sim", "device_ids": list(device_ids)}
        calls: list[tuple] = []
        leases: list[_FakeLease] = []
        worker._worker = _FakeNativeWorker(
            calls,
            fail_release=fail_release,
            allocate_error=allocate_error,
            mutate_success=mutate_success,
        )
        monkeypatch.setattr(worker, "_consume_worker_host_mapped_cleanup_error", lambda _api: None)

        def fake_import(_worker_id, _resource_id, export):
            name = "payload" if not leases else "counter"
            lease = _FakeLease(calls, name, handle=100 + len(leases), fail_close=fail_mapping_close)
            calls.append(("import", name, int(export.logical_bytes)))
            if fail_first_import and name == "payload":
                raise RuntimeError("first import failed")
            if fail_second_import and name == "counter":
                raise RuntimeError("second import failed")
            leases.append(lease)
            return lease

        monkeypatch.setattr(worker, "_import_region_part_lease", fake_import)
        return worker, calls, leases

    return build


def test_worker_materializes_region_instance_and_closes_single_region(region_worker):
    worker, calls, leases = region_worker()
    instance = _materialize_default_region(worker)

    assert instance.state is RegionInstanceState.LIVE
    assert instance.worker_id == 1
    assert instance._payload_mapping is leases[0]
    assert instance._counter_mapping is leases[1]
    assert instance._payload_mapping is not None
    assert instance._counter_mapping is not None
    assert instance._payload_part is not None
    assert instance._counter_part is not None
    assert instance._payload_mapping is not instance._counter_mapping
    assert instance._payload_mapping.handle != instance._counter_mapping.handle
    assert instance._payload_part.span == RegionPartSpan(offset=0, nbytes=64)
    assert instance._counter_part.span == RegionPartSpan(offset=0, nbytes=128)

    with worker._control_reservation("test_region_instance"):
        instance.close()
        instance.close()
    assert instance.state is RegionInstanceState.CLOSED
    assert _tracked(worker) == ()
    assert calls == [
        ("allocate", 64, 128),
        ("import", "payload", 64),
        ("import", "counter", 128),
        ("mapping_close", "payload"),
        ("mapping_close", "counter"),
        ("release", 1, 42),
    ]
    assert all(lease.closed for lease in leases)


def test_region_instance_constructs_separate_w4_part_slots_and_routes_access(region_worker, monkeypatch):
    worker, _calls, _leases = region_worker()
    instance = _materialize_default_region(worker)
    routed: list[tuple[Any, ...]] = []

    def payload_write(self, offset, host_buffer, nbytes=None):
        routed.append(("payload_write", self.span, offset, host_buffer, nbytes))

    def payload_read(self, offset, host_buffer, nbytes=None):
        routed.append(("payload_read", self.span, offset, host_buffer, nbytes))

    def counter(self, offset):
        routed.append(("counter", self.span, offset))
        return "w4-counter"

    monkeypatch.setattr(PayloadPart, "write", payload_write)
    monkeypatch.setattr(PayloadPart, "read", payload_read)
    monkeypatch.setattr(CounterPart, "counter", counter)

    assert instance._payload_part is not instance._counter_part
    with worker._control_reservation("test_region_instance"):
        instance.payload_write(4, "src", nbytes=8)
        instance.payload_read(12, "dst")
        assert instance.counter(16) == "w4-counter"

    assert routed == [
        ("payload_write", RegionPartSpan(offset=0, nbytes=64), 4, "src", 8),
        ("payload_read", RegionPartSpan(offset=0, nbytes=64), 12, "dst", None),
        ("counter", RegionPartSpan(offset=0, nbytes=128), 16),
    ]


def test_region_instance_rejects_access_construction_from_non_host_vmm_copy_attachment(region_worker):
    worker, calls, _leases = region_worker()
    ctx = _accepted_context(worker)
    consumer = validate_single_owner_region_shape(ctx).consumer.identity
    payload_by_member = _attachments(ctx.plan.payload)
    payload_by_member[consumer] = dataclasses.replace(
        payload_by_member[consumer],
        adapter_kind=ce.AdapterKind.DIRECT_MAP,
        adapter_profile=ce.AdapterProfile.HOST_SHM_MAP,
    )
    bad_payload = dataclasses.replace(ctx.plan.payload, attachments=tuple(payload_by_member.values()))
    bad_ctx = dataclasses.replace(ctx, plan=dataclasses.replace(ctx.plan, payload=bad_payload))

    with pytest.raises(MaterializationRefusal) as excinfo:
        from simpler.comm_region import materialize_region_instance

        materialize_region_instance(bad_ctx)

    assert excinfo.value.reason is RefusalReason.UNSUPPORTED_ATTACHMENT
    assert calls == []
    assert worker._region_instance_registry._instances == {}


def test_live_region_instance_close_reuses_one_shot_cleanup(region_worker):
    worker, calls, _leases = region_worker()
    instance = _materialize_default_region(worker)

    with worker._control_reservation("test_region_instance"):
        instance.close()
        instance.close()

    assert instance.state is RegionInstanceState.CLOSED
    assert _tracked(worker) == ()
    assert calls == [
        ("allocate", 64, 128),
        ("import", "payload", 64),
        ("import", "counter", 128),
        ("mapping_close", "payload"),
        ("mapping_close", "counter"),
        ("release", 1, 42),
    ]
    with worker._control_reservation("test_region_instance"):
        instance.close()
    assert instance.state is RegionInstanceState.CLOSED
    assert _tracked(worker) == ()
    with pytest.raises(MaterializationError, match="not live"):
        instance.payload_write(0, "src")


def test_region_instance_close_failure_marks_failed_and_poisons_worker(region_worker):
    worker, calls, _leases = region_worker(fail_mapping_close=True)
    instance = _materialize_default_region(worker)

    with worker._control_reservation("test_region_instance"):
        with pytest.raises(RuntimeError, match="mapping close failed"):
            instance.close()

    assert instance.state is RegionInstanceState.CLOSE_FAILED
    assert _tracked(worker) == ()
    with pytest.raises(RuntimeError, match="no further work is admitted"):
        worker._require_no_ordered_cleanup_failure("test")
    with pytest.raises(RuntimeError, match="not live"):
        instance.payload_write(0, "src")
    with pytest.raises(RuntimeError, match="mapping close failed"):
        instance.close()
    assert calls == [
        ("allocate", 64, 128),
        ("import", "payload", 64),
        ("import", "counter", 128),
        ("mapping_close", "payload"),
        ("mapping_close", "counter"),
        ("release", 1, 42),
    ]


def test_region_instance_close_failure_replays_cached_error(region_worker):
    worker, calls, _leases = region_worker(fail_mapping_close=True)
    instance = _materialize_default_region(worker)

    with worker._control_reservation("test_region_instance"):
        with pytest.raises(RuntimeError, match="mapping close failed") as first_excinfo:
            instance.close()
        with pytest.raises(RuntimeError, match="mapping close failed") as second_excinfo:
            instance.close()

    assert instance.state is RegionInstanceState.CLOSE_FAILED
    assert _tracked(worker) == ()
    assert second_excinfo.value is first_excinfo.value
    assert calls == [
        ("allocate", 64, 128),
        ("import", "payload", 64),
        ("import", "counter", 128),
        ("mapping_close", "payload"),
        ("mapping_close", "counter"),
        ("release", 1, 42),
    ]


def test_callback_region_close_before_submit_retired_from_run_cleanup(region_worker):
    worker, calls, _leases = region_worker()
    resources = _RunResources()
    worker._building_run_resources = resources
    try:
        with _callback_run(17, worker):
            instance = _materialize_default_region(worker)
            instance.close()
    finally:
        worker._building_run_resources = None

    assert instance.state is RegionInstanceState.CLOSED
    assert _tracked(worker) == ()
    worker._region_instance_registry.cleanup_run(resources)
    assert _tracked(worker) == ()
    assert calls == [
        ("allocate", 64, 128),
        ("import", "payload", 64),
        ("import", "counter", 128),
        ("mapping_close", "payload"),
        ("mapping_close", "counter"),
        ("release", 1, 42),
    ]


def test_callback_region_run_cleanup_then_later_close_is_idempotent(region_worker):
    worker, calls, _leases = region_worker()
    resources = _RunResources()
    worker._building_run_resources = resources
    try:
        with _callback_run(18, worker):
            instance = _materialize_default_region(worker)
            frame = _callback_frame_for(worker)
            assert frame is not None
            frame.has_submitted_task = True
            with pytest.raises(RuntimeError, match="cannot follow a task submission"):
                instance.close()
    finally:
        worker._building_run_resources = None

    worker._region_instance_registry.cleanup_run(resources)

    assert instance.state is RegionInstanceState.CLOSED
    assert _tracked(worker) == ()
    instance.close()
    assert instance.state is RegionInstanceState.CLOSED
    assert _tracked(worker) == ()
    assert calls == [
        ("allocate", 64, 128),
        ("import", "payload", 64),
        ("import", "counter", 128),
        ("mapping_close", "payload"),
        ("mapping_close", "counter"),
        ("release", 1, 42),
    ]


def test_callback_region_data_plane_after_submit_uses_mapped_view(region_worker, monkeypatch):
    worker, _calls, _leases = region_worker()
    routed: list[str] = []

    def payload_write(self, offset, host_buffer, nbytes=None):
        routed.append("payload_write")

    def payload_read(self, offset, host_buffer, nbytes=None):
        routed.append("payload_read")

    def counter(self, offset):
        routed.append("counter")
        return "counter"

    monkeypatch.setattr(PayloadPart, "write", payload_write)
    monkeypatch.setattr(PayloadPart, "read", payload_read)
    monkeypatch.setattr(CounterPart, "counter", counter)

    resources = _RunResources()
    worker._building_run_resources = resources
    try:
        with _callback_run(19, worker):
            instance = _materialize_default_region(worker)
            frame = _callback_frame_for(worker)
            assert frame is not None
            frame.has_submitted_task = True
            instance.payload_write(0, "src")
            instance.payload_read(0, "dst")
            assert instance.counter(0) == "counter"
            with pytest.raises(RuntimeError, match="cannot follow a task submission"):
                instance.close()
    finally:
        worker._building_run_resources = None

    assert routed == ["payload_write", "payload_read", "counter"]


def test_materialize_tracks_before_allocate_and_create_failure_is_not_live(region_worker):
    worker, calls, _leases = region_worker(allocate_error=RuntimeError("create failed"))

    with pytest.raises(RuntimeError, match="create failed"):
        worker._materialize_region_instance(
            [ce.at("L3", ce.HOST_CPU), ce.at("L3/L2[1]", ce.DEVICE_AICPU)],
            ce.SingleOwner(provider=ce.at("L3/L2[1]", ce.DEVICE_AICPU)),
            ce.RegionLayoutSpec(payload_bytes=64, counter_bytes=128),
        )

    assert calls == [("allocate", 64, 128)]
    assert _tracked(worker) == ()
    assert worker._delegated_session_fatal is not None


def test_materialize_tracks_before_allocate(region_worker, monkeypatch):
    worker, calls, _leases = region_worker()
    seen: list[int] = []
    native = worker._worker
    original = native.control_payload

    def observe(*args, **kwargs):
        seen.append(len(worker._region_instance_registry._instances))
        return original(*args, **kwargs)

    native.control_payload = observe
    instance = _materialize_default_region(worker)
    assert seen == [1]
    assert instance.state is RegionInstanceState.LIVE
    assert calls[:1] == [("allocate", 64, 128)]


def test_first_import_failure_releases_once_and_poisons(region_worker):
    worker, calls, leases = region_worker(fail_first_import=True)
    with pytest.raises(RuntimeError, match="first import failed"):
        _materialize_default_region(worker)

    assert _tracked(worker) == ()
    worker._require_no_ordered_cleanup_failure("test")
    assert calls == [
        ("allocate", 64, 128),
        ("import", "payload", 64),
        ("release", 1, 42),
    ]
    assert leases == []


def test_second_import_failure_closes_first_lease_and_poisons(region_worker):
    worker, calls, leases = region_worker(fail_second_import=True)
    with pytest.raises(RuntimeError, match="second import failed"):
        _materialize_default_region(worker)

    assert _tracked(worker) == ()
    worker._require_no_ordered_cleanup_failure("test")
    assert calls == [
        ("allocate", 64, 128),
        ("import", "payload", 64),
        ("import", "counter", 128),
        ("mapping_close", "payload"),
        ("release", 1, 42),
    ]
    assert leases[0].closed is True
    assert len(leases) == 1


def _overlap_allocated_counter_view(staged: bytearray) -> None:
    from simpler.comm_provider import RegionPartKind
    from simpler.comm_provider_control import ALLOCATE_COUNTER_VIEW_OFFSET

    struct.pack_into("<IIQQ", staged, ALLOCATE_COUNTER_VIEW_OFFSET, int(RegionPartKind.COUNTER), 0, 0, 128)


def test_invalid_success_reply_releases_once_and_never_imports(region_worker):
    worker, calls, leases = region_worker(mutate_success=_overlap_allocated_counter_view)
    with pytest.raises((RuntimeError, ValueError), match="must not overlap"):
        _materialize_default_region(worker)

    assert _tracked(worker) == ()
    assert worker._delegated_session_fatal is not None
    assert calls == [("allocate", 64, 128)]
    assert leases == []


def test_invalid_success_compensation_debt_releases_once_and_survives_until_sweep(region_worker):
    worker, calls, leases = region_worker(fail_release=True, mutate_success=_overlap_allocated_counter_view)
    with pytest.raises((RuntimeError, ValueError), match="must not overlap"):
        _materialize_default_region(worker)

    assert _tracked(worker) == ()
    assert worker._delegated_session_fatal is not None
    assert calls == [("allocate", 64, 128)]
    assert leases == []


def test_successful_close_then_materialize_and_close_again(region_worker):
    worker, calls, _leases = region_worker()
    first = _materialize_default_region(worker)
    with worker._control_reservation("test_region_instance"):
        first.close()
    assert first.state is RegionInstanceState.CLOSED
    assert _tracked(worker) == ()

    second = _materialize_default_region(worker)
    assert second.state is RegionInstanceState.LIVE
    assert second is not first
    with worker._control_reservation("test_region_instance"):
        second.close()
    assert second.state is RegionInstanceState.CLOSED
    assert _tracked(worker) == ()
    assert calls.count(("allocate", 64, 128)) == 2
    assert sum(1 for item in calls if item[0] == "import") == 4
    assert sum(1 for item in calls if item[0] == "mapping_close") == 4
    assert [item for item in calls if item[0] == "release"] == [
        ("release", 1, 42),
        ("release", 1, 42),
    ]


def test_committed_local_base_zero_is_not_treated_as_absent(region_worker):
    worker, _calls, _leases = region_worker()
    instance = _materialize_default_region(worker)
    assert instance._payload_part is not None
    assert instance._counter_part is not None
    assert instance._payload_part.span.offset == 0
    assert instance._counter_part.span.offset == 0
    assert instance.state is RegionInstanceState.LIVE


def test_allocation_error_without_debt_retires_closed_without_release(region_worker):
    from simpler.comm_provider import RegionAllocationError, RegionControlErrorKind, RegionOperationKind, RegionPartKind

    error = RegionAllocationError(
        provisional_resource_id=9,
        control_kind=RegionControlErrorKind.BACKEND_FAILURE,
        failed_part=RegionPartKind.PAYLOAD,
        failed_operation=RegionOperationKind.MATERIALIZE,
        cleanup_debt_remaining=False,
        message="create failed cleanly",
    )
    worker, calls, _leases = region_worker(allocate_error=error)
    with pytest.raises(RegionAllocationError):
        _materialize_default_region(worker)

    assert _tracked(worker) == ()
    worker._require_no_ordered_cleanup_failure("test")
    assert worker._delegated_session_fatal is None
    assert calls == [("allocate", 64, 128)]


def test_allocation_error_with_debt_is_close_failed_without_release_client(region_worker):
    from simpler.comm_provider import RegionAllocationError, RegionControlErrorKind, RegionOperationKind, RegionPartKind

    error = RegionAllocationError(
        provisional_resource_id=9,
        control_kind=RegionControlErrorKind.BACKEND_FAILURE,
        failed_part=RegionPartKind.PAYLOAD,
        failed_operation=RegionOperationKind.MATERIALIZE,
        cleanup_debt_remaining=True,
        message="create left debt",
    )
    worker, calls, _leases = region_worker(allocate_error=error)
    with pytest.raises(RegionAllocationError) as excinfo:
        _materialize_default_region(worker)
    assert excinfo.value.control_kind is RegionControlErrorKind.BACKEND_FAILURE
    assert excinfo.value.cleanup_debt_remaining is True

    tracked = _tracked(worker)
    assert len(tracked) == 1
    assert tracked[0]._state is RegionInstanceState.CLOSE_FAILED
    assert tracked[0]._delegated_release_edge is False
    assert worker._delegated_session_fatal is not None
    assert calls == [("allocate", 64, 128)]


def test_registry_run_membership_cleanup_closes_only_that_run(region_worker):
    worker, calls, _leases = region_worker()
    first = _RunResources()
    second = _RunResources()
    worker._building_run_resources = first
    try:
        first_instance = _materialize_default_region(worker)
    finally:
        worker._building_run_resources = None
    worker._building_run_resources = second
    try:
        second_instance = _materialize_default_region(worker)
    finally:
        worker._building_run_resources = None

    assert {first_instance, second_instance} == set(_tracked(worker))
    worker._region_instance_registry.cleanup_run(first)
    assert first_instance.state is RegionInstanceState.CLOSED
    assert second_instance.state is RegionInstanceState.LIVE
    assert _tracked(worker) == (second_instance,)
    assert [item for item in calls if item[0] == "release"] == [("release", 1, 42)]

    worker._region_instance_registry.cleanup_run(second)
    assert second_instance.state is RegionInstanceState.CLOSED
    assert _tracked(worker) == ()
    assert [item for item in calls if item[0] == "release"] == [("release", 1, 42), ("release", 1, 42)]


def test_registry_run_cleanup_closes_live_instances(region_worker):
    worker, calls, _leases = region_worker()
    resources = _RunResources()
    worker._building_run_resources = resources
    try:
        instance = _materialize_default_region(worker)
    finally:
        worker._building_run_resources = None

    assert instance.state is RegionInstanceState.LIVE
    assert _tracked(worker) == (instance,)
    worker._region_instance_registry.cleanup_run(resources)
    assert instance.state is RegionInstanceState.CLOSED
    assert _tracked(worker) == ()
    assert calls == [
        ("allocate", 64, 128),
        ("import", "payload", 64),
        ("import", "counter", 128),
        ("mapping_close", "payload"),
        ("mapping_close", "counter"),
        ("release", 1, 42),
    ]


def test_allocation_error_with_debt_survives_run_cleanup_until_sweep_without_release(region_worker):
    from simpler.comm_provider import RegionAllocationError, RegionControlErrorKind, RegionOperationKind, RegionPartKind

    error = RegionAllocationError(
        provisional_resource_id=9,
        control_kind=RegionControlErrorKind.BACKEND_FAILURE,
        failed_part=RegionPartKind.PAYLOAD,
        failed_operation=RegionOperationKind.MATERIALIZE,
        cleanup_debt_remaining=True,
        message="create left debt",
    )
    worker, calls, _leases = region_worker(allocate_error=error)
    resources = _RunResources()
    worker._building_run_resources = resources
    try:
        with pytest.raises(RegionAllocationError) as excinfo:
            _materialize_default_region(worker)
        assert excinfo.value.control_kind is RegionControlErrorKind.BACKEND_FAILURE
        assert excinfo.value.cleanup_debt_remaining is True
    finally:
        worker._building_run_resources = None

    tracked = _tracked(worker)
    assert len(tracked) == 1
    instance = tracked[0]
    assert instance._state is RegionInstanceState.CLOSE_FAILED
    assert instance._delegated_release_edge is False
    worker._region_instance_registry.cleanup_run(resources)
    assert _tracked(worker) == (instance,)
    assert calls == [("allocate", 64, 128)]
    worker._region_instance_registry.sweep()
    assert _tracked(worker) == ()
    assert calls == [("allocate", 64, 128)]


def test_provider_release_failure_keeps_diagnostic_until_sweep(region_worker):
    worker, calls, _leases = region_worker(fail_release=True)
    instance = _materialize_default_region(worker)
    with worker._control_reservation("test_region_instance"):
        with pytest.raises(RuntimeError, match="release failed"):
            instance.close()

    assert instance.state is RegionInstanceState.CLOSE_FAILED
    assert _tracked(worker) == (instance,)
    worker._region_instance_registry.cleanup_run(None)
    assert _tracked(worker) == (instance,)
    release_calls = [item for item in calls if item[0] == "release"]
    assert release_calls == [("release", 1, 42)]
    worker._region_instance_registry.sweep()
    assert _tracked(worker) == ()
    assert [item for item in calls if item[0] == "release"] == release_calls


def test_unpublished_close_failed_survives_cleanup_run_until_sweep(region_worker):
    worker, calls, _leases = region_worker(fail_second_import=True, fail_release=True)
    resources = _RunResources()
    worker._building_run_resources = resources
    try:
        with pytest.raises(RuntimeError, match="second import failed"):
            _materialize_default_region(worker)
    finally:
        worker._building_run_resources = None

    tracked = _tracked(worker)
    assert len(tracked) == 1
    instance = tracked[0]
    assert instance._state is RegionInstanceState.CLOSE_FAILED
    worker._region_instance_registry.cleanup_run(resources)
    assert _tracked(worker) == (instance,)
    assert calls.count(("release", 1, 42)) == 1
    worker._region_instance_registry.sweep()
    assert _tracked(worker) == ()
    assert calls.count(("release", 1, 42)) == 1


def test_data_plane_does_not_consult_registry(region_worker, monkeypatch):
    worker, _calls, _leases = region_worker()
    instance = _materialize_default_region(worker)
    registry = worker._region_instance_registry

    def boom(*_args, **_kwargs):
        raise AssertionError("data-plane must not consult the region instance registry")

    for name in (
        "track",
        "close",
        "cleanup_run",
        "sweep",
        "record_data_plane_failure",
        "_iter_run",
        "_settle",
        "_retire",
    ):
        monkeypatch.setattr(registry, name, boom)
    monkeypatch.setattr(PayloadPart, "write", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(PayloadPart, "read", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(CounterPart, "counter", lambda *_args, **_kwargs: "counter")

    with worker._control_reservation("test_region_instance"):
        instance.payload_write(0, "src")
        instance.payload_read(0, "dst")
        assert instance.counter(0) == "counter"

    assert not hasattr(RegionInstanceRegistry, "get")
    assert not hasattr(RegionInstanceRegistry, "find")
    assert not hasattr(RegionInstanceRegistry, "__getitem__")


def test_live_region_instance_access_requires_control_context(region_worker):
    worker, calls, _leases = region_worker()
    instance = _materialize_default_region(worker)

    with pytest.raises(RuntimeError, match="active orchestration/control context"):
        instance.payload_write(0, "src")
    with pytest.raises(RuntimeError, match="active orchestration/control context"):
        instance.counter(0)
    with pytest.raises(RuntimeError, match="active orchestration/control context"):
        instance.close()

    with worker._control_reservation("test_region_instance"):
        instance.close()

    assert calls == [
        ("allocate", 64, 128),
        ("import", "payload", 64),
        ("import", "counter", 128),
        ("mapping_close", "payload"),
        ("mapping_close", "counter"),
        ("release", 1, 42),
    ]


def test_local_view_returns_completed_materialization_only(region_worker):
    worker, _calls, _leases = region_worker()
    instance = _materialize_default_region(worker)
    assert instance.provider_resource_id == 42
    payload = instance.local_view(RegionPartKind.PAYLOAD)
    counter = instance.local_view(RegionPartKind.COUNTER)
    assert payload is instance._payload_local_view
    assert counter is instance._counter_local_view
    assert payload is not None and counter is not None
    instance._state = RegionInstanceState.CLOSED
    assert instance.local_view(RegionPartKind.PAYLOAD) is None
    assert instance.local_view(RegionPartKind.COUNTER) is None


def test_live_resource_inventory_reports_registry_instance_count(region_worker):
    worker, _calls, _leases = region_worker()
    assert "region instance" not in worker._describe_live_resources()
    _materialize_default_region(worker)
    assert "1 region instance(s)" in worker._describe_live_resources()


def test_record_data_plane_failure_matches_single_region(region_worker):
    worker, calls, _leases = region_worker()
    resources = _RunResources()
    worker._building_run_resources = resources
    try:
        instance = _materialize_default_region(worker)
    finally:
        worker._building_run_resources = None
    error = RuntimeError("issued local operation failed")
    worker._region_instance_registry.record_data_plane_failure(resources, instance.provider_resource_id, error)
    assert instance.data_plane_error is error
    assert instance._close_attempted is False
    assert instance.state is RegionInstanceState.LIVE
    assert [item for item in calls if item[0] == "release"] == []


def test_record_data_plane_failure_matches_only_target_among_multiple(region_worker):
    worker, calls, _leases = region_worker()
    resources = _RunResources()
    worker._building_run_resources = resources
    try:
        first = _materialize_default_region(worker)
        second = _materialize_default_region(worker)
    finally:
        worker._building_run_resources = None
    second._provider_resource_id = 43
    error = RuntimeError("issued local operation failed")
    worker._region_instance_registry.record_data_plane_failure(resources, first.provider_resource_id, error)
    assert first.data_plane_error is error
    assert second.data_plane_error is None
    assert first._close_attempted is False
    assert second._close_attempted is False
    assert [item for item in calls if item[0] == "release"] == []


def test_record_data_plane_failure_unknown_resource_is_routing_error(region_worker):
    worker, _calls, _leases = region_worker()
    resources = _RunResources()
    worker._building_run_resources = resources
    try:
        instance = _materialize_default_region(worker)
    finally:
        worker._building_run_resources = None
    with pytest.raises(MaterializationError, match="no region instance for resource 99"):
        worker._region_instance_registry.record_data_plane_failure(resources, 99, RuntimeError("unused"))
    assert instance.data_plane_error is None
    assert instance._close_attempted is False


def test_record_data_plane_failure_duplicate_match_is_invariant_error(region_worker):
    worker, _calls, _leases = region_worker()
    resources = _RunResources()
    worker._building_run_resources = resources
    try:
        first = _materialize_default_region(worker)
        second = _materialize_default_region(worker)
    finally:
        worker._building_run_resources = None
    second._provider_resource_id = first.provider_resource_id
    with pytest.raises(MaterializationError, match="duplicate region instances for resource 42"):
        worker._region_instance_registry.record_data_plane_failure(
            resources, first.provider_resource_id, RuntimeError("unused")
        )
    assert first.data_plane_error is None
    assert second.data_plane_error is None


def _endpoint_error(resource_id: int) -> RuntimeError:
    return RuntimeError(
        f"L3-L2 endpoint error op=payload_write kind=5 region={int(resource_id)} msg=issued local operation failed"
    )


def test_endpoint_unknown_resource_poisons_worker(region_worker):
    worker, calls, _leases = region_worker()
    resources = _RunResources()
    worker._building_run_resources = resources
    try:
        instance = _materialize_default_region(worker)
    finally:
        worker._building_run_resources = None
    assert worker._poison_worker_chip_region_from_endpoint_error(_endpoint_error(99), resources) is True
    assert instance.data_plane_error is None
    assert instance._close_attempted is False
    assert [item for item in calls if item[0] == "release"] == []
    with pytest.raises(RuntimeError, match="no further work is admitted"):
        worker._require_no_ordered_cleanup_failure("test")
    with pytest.raises(RuntimeError, match="no further work is admitted"):
        _materialize_default_region(worker)


def test_endpoint_duplicate_match_poisons_worker(region_worker):
    worker, calls, _leases = region_worker()
    resources = _RunResources()
    worker._building_run_resources = resources
    try:
        first = _materialize_default_region(worker)
        second = _materialize_default_region(worker)
    finally:
        worker._building_run_resources = None
    second._provider_resource_id = first.provider_resource_id
    assert worker._poison_worker_chip_region_from_endpoint_error(_endpoint_error(first.provider_resource_id), resources)
    assert first.data_plane_error is None
    assert second.data_plane_error is None
    assert [item for item in calls if item[0] == "release"] == []
    with pytest.raises(RuntimeError, match="no further work is admitted"):
        worker._require_no_ordered_cleanup_failure("test")


def _queue_endpoint_error(session: bytes, transaction_id: int) -> RuntimeError:
    bits = int.from_bytes(bytes(session), "little")
    return RuntimeError(
        "SPSC queue endpoint error op=init kind=6 "
        f"session=0x{bits:016x} transaction={int(transaction_id)} msg=issued local operation failed"
    )


def test_allocation_identity_returns_bound_session_and_transaction(region_worker):
    worker, _calls, _leases = region_worker()
    resources = _RunResources()
    worker._building_run_resources = resources
    try:
        instance = _materialize_default_region(worker)
    finally:
        worker._building_run_resources = None
    session, transaction_id = instance._allocation_identity
    assert isinstance(session, bytes)
    assert len(session) == 8
    assert type(transaction_id) is int
    assert transaction_id >= 1
    instance._delegated_session_instance_id = None
    with pytest.raises(MaterializationError, match="incomplete"):
        _ = instance._allocation_identity


def test_record_data_plane_failure_by_allocation_identity_matches_single_region(region_worker):
    worker, calls, _leases = region_worker()
    resources = _RunResources()
    worker._building_run_resources = resources
    try:
        instance = _materialize_default_region(worker)
    finally:
        worker._building_run_resources = None
    session, transaction_id = instance._allocation_identity
    error = RuntimeError("issued local operation failed")
    worker._region_instance_registry.record_data_plane_failure_by_allocation_identity(
        resources, session, transaction_id, error
    )
    assert instance.data_plane_error is error
    assert instance._close_attempted is False
    assert instance.state is RegionInstanceState.LIVE
    assert [item for item in calls if item[0] == "release"] == []
    later = RuntimeError("second failure")
    worker._region_instance_registry.record_data_plane_failure_by_allocation_identity(
        resources, session, transaction_id, later
    )
    assert instance.data_plane_error is error


def test_record_data_plane_failure_by_allocation_identity_unknown_is_routing_error(region_worker):
    worker, _calls, _leases = region_worker()
    resources = _RunResources()
    other = _RunResources()
    worker._building_run_resources = resources
    try:
        instance = _materialize_default_region(worker)
    finally:
        worker._building_run_resources = None
    session, transaction_id = instance._allocation_identity
    with pytest.raises(MaterializationError, match="no region instance for allocation identity"):
        worker._region_instance_registry.record_data_plane_failure_by_allocation_identity(
            other, session, transaction_id, RuntimeError("unused")
        )
    with pytest.raises(MaterializationError, match="no region instance for allocation identity"):
        worker._region_instance_registry.record_data_plane_failure_by_allocation_identity(
            resources, b"\x00" * 8, 99, RuntimeError("unused")
        )
    assert instance.data_plane_error is None
    assert instance._close_attempted is False


def test_record_data_plane_failure_by_allocation_identity_duplicate_match_is_invariant_error(region_worker):
    worker, _calls, _leases = region_worker()
    resources = _RunResources()
    worker._building_run_resources = resources
    try:
        first = _materialize_default_region(worker)
        second = _materialize_default_region(worker)
    finally:
        worker._building_run_resources = None
    session, transaction_id = first._allocation_identity
    second._delegated_session_instance_id = first._delegated_session_instance_id
    second._delegated_transaction_id = first._delegated_transaction_id
    with pytest.raises(MaterializationError, match="duplicate region instances for allocation identity"):
        worker._region_instance_registry.record_data_plane_failure_by_allocation_identity(
            resources, session, transaction_id, RuntimeError("unused")
        )
    assert first.data_plane_error is None
    assert second.data_plane_error is None


def test_queue_endpoint_error_correlates_unique_allocation_identity(region_worker):
    worker, calls, _leases = region_worker()
    resources = _RunResources()
    worker._building_run_resources = resources
    try:
        instance = _materialize_default_region(worker)
    finally:
        worker._building_run_resources = None
    session, transaction_id = instance._allocation_identity
    error = _queue_endpoint_error(session, transaction_id)
    assert worker._poison_spsc_queue_from_endpoint_error(error, resources) is True
    assert instance.data_plane_error is error
    assert instance._close_attempted is False
    assert [item for item in calls if item[0] == "release"] == []
    with pytest.raises(RuntimeError, match="no further work is admitted"):
        worker._require_no_ordered_cleanup_failure("test")


def test_queue_endpoint_error_unknown_identity_fail_closed(region_worker):
    worker, _calls, _leases = region_worker()
    resources = _RunResources()
    worker._building_run_resources = resources
    try:
        instance = _materialize_default_region(worker)
    finally:
        worker._building_run_resources = None
    error = _queue_endpoint_error(b"\xff" * 8, 1)
    assert worker._poison_spsc_queue_from_endpoint_error(error, resources) is True
    assert instance.data_plane_error is None
    with pytest.raises(RuntimeError, match="no further work is admitted"):
        worker._require_no_ordered_cleanup_failure("test")


def test_queue_endpoint_error_malformed_does_not_modify_instance_or_fall_back(region_worker):
    worker, _calls, _leases = region_worker()
    resources = _RunResources()
    worker._building_run_resources = resources
    try:
        instance = _materialize_default_region(worker)
    finally:
        worker._building_run_resources = None
    error = RuntimeError(
        "SPSC queue endpoint error op=init kind=2 msg=invalid queue binding "
        f"L3-L2 endpoint error op=payload_write kind=5 region={int(instance.provider_resource_id)} "
        "msg=issued local operation failed"
    )
    assert worker._poison_spsc_queue_from_endpoint_error(error, resources) is True
    assert instance.data_plane_error is None
    with pytest.raises(RuntimeError, match="no further work is admitted"):
        worker._require_no_ordered_cleanup_failure("test")


def test_queue_endpoint_error_zero_transaction_does_not_modify_instance(region_worker):
    worker, _calls, _leases = region_worker()
    resources = _RunResources()
    worker._building_run_resources = resources
    try:
        instance = _materialize_default_region(worker)
    finally:
        worker._building_run_resources = None
    session, _transaction_id = instance._allocation_identity
    error = _queue_endpoint_error(session, 0)
    assert worker._poison_spsc_queue_from_endpoint_error(error, resources) is True
    assert instance.data_plane_error is None
    with pytest.raises(RuntimeError, match="no further work is admitted"):
        worker._require_no_ordered_cleanup_failure("test")


def test_legacy_region_parser_still_handles_independent_region_marker(region_worker):
    worker, _calls, _leases = region_worker()
    resources = _RunResources()
    worker._building_run_resources = resources
    try:
        instance = _materialize_default_region(worker)
    finally:
        worker._building_run_resources = None
    error = _endpoint_error(instance.provider_resource_id)
    assert worker._poison_spsc_queue_from_endpoint_error(error, resources) is False
    assert worker._poison_worker_chip_region_from_endpoint_error(error, resources) is True
    assert instance.data_plane_error is error


def test_data_plane_poison_still_completes_registry_cleanup_and_release(region_worker):
    worker, calls, _leases = region_worker()
    resources = _RunResources()
    worker._building_run_resources = resources
    try:
        instance = _materialize_default_region(worker)
    finally:
        worker._building_run_resources = None
    error = _endpoint_error(instance.provider_resource_id)
    assert worker._poison_worker_chip_region_from_endpoint_error(error, resources) is True
    assert instance.data_plane_error is error
    assert instance._close_attempted is False
    assert instance.state is RegionInstanceState.LIVE
    worker._region_instance_registry.cleanup_run(resources)
    assert instance.state is RegionInstanceState.CLOSED
    assert calls.count(("release", 1, 42)) == 1
    with pytest.raises(RuntimeError, match="no further work is admitted"):
        worker._require_no_ordered_cleanup_failure("test")
    with pytest.raises(RuntimeError, match="no further work is admitted"):
        _materialize_default_region(worker)


def test_shape_validation_rejects_foreign_registry_context():
    owner = _accepted_context(_l3(device_ids=[0, 1]))
    foreign_worker = _l3(device_ids=[0, 1])

    with pytest.raises(MaterializationRefusal) as excinfo:
        validate_single_owner_region_shape(dataclasses.replace(owner, worker=foreign_worker))

    assert excinfo.value.reason is RefusalReason.REGISTRY_MISMATCH


def test_shape_validation_does_not_translate_worker_readiness_errors():
    ctx = _accepted_context()
    ctx.worker._worker = None

    shape = validate_single_owner_region_shape(ctx)
    assert shape.provider.path == "L3/L2[1]"
    assert shape.first_hop_child_id == 1


def test_shape_validation_rejects_stale_registry_epoch():
    ctx = _accepted_context(_l3(device_ids=[0, 1]))
    ctx.worker._invalidate_endpoint_registry()

    with pytest.raises(MaterializationRefusal) as excinfo:
        validate_single_owner_region_shape(ctx)

    assert excinfo.value.reason is RefusalReason.REGISTRY_MISMATCH


def test_projection_copies_admitted_part_backing_and_layout_bytes():
    ctx = _accepted_context()
    shape = validate_single_owner_region_shape(ctx)
    spec = project_region_allocation_spec(ctx.plan, ctx.layout)

    assert shape.worker_id == 1
    assert spec.payload.planned_backing_kind is ctx.plan.payload.backend_kind
    assert spec.counter.planned_backing_kind is ctx.plan.counter.backend_kind
    assert spec.payload.logical_bytes == ctx.layout.payload_bytes
    assert spec.counter.logical_bytes == ctx.layout.counter_bytes
    assert spec.payload.planned_backing_kind is ce.BackendKind.VMM_WINDOW
    assert spec.counter.planned_backing_kind is ce.BackendKind.VMM_WINDOW
    assert not hasattr(spec, "provider")
    assert not hasattr(spec.payload, "device_id")
    assert not hasattr(spec.payload, "worker_id")


def test_projection_copies_plan_backing_without_selecting_capability():
    ctx = _accepted_context()
    posix_plan = dataclasses.replace(
        ctx.plan,
        payload=dataclasses.replace(ctx.plan.payload, backend_kind=ce.BackendKind.POSIX_SHM),
        counter=dataclasses.replace(ctx.plan.counter, backend_kind=ce.BackendKind.POSIX_SHM),
    )

    spec = project_region_allocation_spec(posix_plan, ctx.layout)

    assert spec.payload.planned_backing_kind is ce.BackendKind.POSIX_SHM
    assert spec.counter.planned_backing_kind is ce.BackendKind.POSIX_SHM


def test_projection_refuses_zero_layout_before_store_invocation():
    ctx = _accepted_context()
    zero_layout = ce.RegionLayoutSpec(payload_bytes=0, counter_bytes=128)

    with pytest.raises(ValueError, match="logical_bytes must be positive"):
        project_region_allocation_spec(ctx.plan, zero_layout)


def test_projection_requires_a_real_backend_plan():
    ctx = _accepted_context()
    unsupported = ce.UnsupportedRegionPlan(
        reason=ce.BackendUnsupportedReason.NO_DEFAULT_PROVIDER,
        message="no default SingleOwner provider is available",
    )

    with pytest.raises(TypeError, match="requires a BackendPlan"):
        project_region_allocation_spec(unsupported, ctx.layout)
    with pytest.raises(TypeError, match="requires a RegionLayoutSpec"):
        project_region_allocation_spec(ctx.plan, object())


def test_compatibility_create_projects_admitted_w2_plan_not_byte_counts(monkeypatch):
    worker = _l3(device_ids=[8, 9])
    spec = worker._project_admitted_worker_chip_region_spec(1, 64, 128)

    assert spec.payload.planned_backing_kind is ce.BackendKind.VMM_WINDOW
    assert spec.counter.planned_backing_kind is ce.BackendKind.VMM_WINDOW
    assert spec.payload.logical_bytes == 64
    assert spec.counter.logical_bytes == 128


def test_compatibility_create_does_not_allocate_when_admission_refuses(monkeypatch):
    worker = _l3(device_ids=[8, 9])
    calls: list[str] = []
    monkeypatch.setattr(worker, "_consume_worker_host_mapped_cleanup_error", lambda _api: None)

    def refuse(_ctx):
        calls.append("admit")
        raise MaterializationRefusal(RefusalReason.UNSUPPORTED_BACKEND_KIND, "refused")

    def payload(*_args, **_kwargs):
        calls.append("allocate")
        raise AssertionError("unadmitted plans must not reach delegated dispatch")

    monkeypatch.setattr("simpler.comm_region.validate_single_owner_region_shape", refuse)
    worker._worker = type("Native", (), {"control_payload": staticmethod(payload)})()

    with pytest.raises(MaterializationRefusal) as excinfo:
        worker._create_worker_chip_region(1, 64, 128)

    assert excinfo.value.reason is RefusalReason.UNSUPPORTED_BACKEND_KIND
    assert calls == ["admit"]


def test_compatibility_create_uses_projection_result_not_a_fabricated_spec(monkeypatch):
    worker = _l3(device_ids=[8, 9])
    worker._config = {**worker._config, "platform": "a2a3sim", "device_ids": [8, 9]}
    monkeypatch.setattr(worker, "_consume_worker_host_mapped_cleanup_error", lambda _api: None)
    monkeypatch.setattr(worker, "_import_region_part_lease", lambda *_args, **_kwargs: _FakeLease([], "payload", 1))
    captured: dict[str, Any] = {}

    def payload(_worker_type, worker_id, sub_cmd, staged, _timeout):
        from simpler.comm_provider_control import (
            DelegatedAllocateReply,
            DelegatedAllocateReplyTag,
            DelegatedRegionOperation,
            encode_reply,
            parse_request,
            publish_reply,
        )
        from simpler.worker import _CTRL_DELEGATED_REGION

        assert int(sub_cmd) == _CTRL_DELEGATED_REGION
        buf = bytearray(staged)
        envelope = parse_request(buf)
        request = envelope.decode_terminal()
        captured["worker_id"] = int(worker_id)
        captured["request"] = request
        captured["plan"] = worker._get_endpoint_registry()
        if envelope.operation is DelegatedRegionOperation.DELEGATED_ALLOCATE:
            result, payload_view, counter_view = _committed_success(request.spec, resource_id=1)
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
            publish_reply(memoryview(buf), committed)
        return bytes(buf)

    worker._worker = type("Native", (), {"control_payload": staticmethod(payload)})()
    region = worker._create_worker_chip_region(1, 64, 128)
    request = captured["request"]
    assert int(captured["worker_id"]) == 1
    assert request.payload_logical_bytes == 64
    assert request.counter_logical_bytes == 128
    assert request.payload_backend_kind is ce.BackendKind.VMM_WINDOW
    assert request.counter_backend_kind is ce.BackendKind.VMM_WINDOW
    registry = worker._get_endpoint_registry()
    plan = region._instance.plan
    provider = registry.record_for(plan.topology_plan.provider_endpoint)
    consumer = next(registry.record_for(member) for member in plan.ordered_members if member != provider.identity)
    assert provider.path == "L3/L2[1]"
    assert provider.deployment is ce.DEVICE_AICPU
    assert consumer.path == "L3"
    assert consumer.deployment is ce.HOST_CPU


class _StoreControlMailbox:
    def __init__(
        self,
        store,
        *,
        fail_before_allocate: Optional[BaseException] = None,
        fail_after_allocate: Optional[BaseException] = None,
        reply_mode: str = "handler",
    ) -> None:
        from simpler.comm_provider_control import ProviderTransactionTable

        self.store = store
        self.table = ProviderTransactionTable()
        self.allocate_calls = 0
        self.release_calls = 0
        self.released_ids: list[int] = []
        self._fail_before_allocate = fail_before_allocate
        self._fail_after_allocate = fail_after_allocate
        self._reply_mode = reply_mode

    def control_payload(self, _worker_type, worker_id, sub_cmd, payload, _timeout):
        from simpler.comm_provider_control import (
            DelegatedRegionOperation,
            handle_terminal_delegated_region,
            parse_reply,
            parse_request,
        )
        from simpler.worker import _CTRL_DELEGATED_REGION

        del worker_id
        assert int(sub_cmd) == _CTRL_DELEGATED_REGION
        staged = bytearray(payload)
        envelope = parse_request(staged)
        if envelope.operation is DelegatedRegionOperation.DELEGATED_ALLOCATE:
            self.allocate_calls += 1
            if self._fail_before_allocate is not None:
                raise self._fail_before_allocate
            if self._reply_mode == "empty":
                return bytes(len(staged))
            if self._reply_mode == "malformed":
                staged[:] = b"\x00" * len(staged)
                staged[12:16] = (1).to_bytes(4, "little")
                return bytes(staged)
            handle_terminal_delegated_region(memoryview(staged), self.table, self.store)
            if self._fail_after_allocate is not None:
                raise self._fail_after_allocate
            return bytes(staged)
        self.release_calls += 1
        handle_terminal_delegated_region(memoryview(staged), self.table, self.store)
        outcome = parse_reply(staged).decode_outcome()
        resource_id = getattr(getattr(outcome, "result", None), "provider_resource_id", 0)
        self.released_ids.append(int(resource_id or 0))
        return bytes(staged)


def _live_control_worker(mailbox, monkeypatch, device_ids=(8, 9)):
    worker = _l3(device_ids=device_ids)
    worker._config = {**worker._config, "platform": "a2a3sim", "device_ids": list(device_ids)}
    worker._worker = mailbox
    monkeypatch.setattr(worker, "_consume_worker_host_mapped_cleanup_error", lambda _api: None)
    return worker


def _assert_poisoned(worker, *, cause: Optional[BaseException] = None) -> RuntimeError:
    with pytest.raises(RuntimeError, match="no further work is admitted") as poison_info:
        worker._require_no_delegated_session_fatal("test")
    if cause is not None:
        chain: list[BaseException] = []
        current: Optional[BaseException] = poison_info.value
        while current is not None and current not in chain:
            chain.append(current)
            current = current.__cause__
        assert any(item is cause or item == cause for item in chain)
    return poison_info.value


def test_handler_commit_then_mailbox_error_releases_once_and_poisons(monkeypatch):
    from simpler.comm_provider import ProviderRegionStore

    from tests.ut.py.test_worker.test_comm_provider import FakeShellFactory, _sim_context

    factory = FakeShellFactory()
    store = ProviderRegionStore(_sim_context(), _shell_factory=factory)
    mailbox = _StoreControlMailbox(store, fail_after_allocate=RuntimeError("mailbox down after commit"))
    worker = _live_control_worker(mailbox, monkeypatch)
    with pytest.raises(RuntimeError, match="mailbox down after commit") as exc_info:
        _materialize_default_region(worker)
    assert _tracked(worker) == ()
    assert mailbox.release_calls == 0
    assert factory.payloads[0].release_count == 0
    assert factory.counters[0].release_count == 0
    _assert_poisoned(worker, cause=exc_info.value)
    with pytest.raises(RuntimeError, match="no further work is admitted"):
        _materialize_default_region(worker)


def test_dispatch_empty_reply_does_not_release_and_poisons(monkeypatch):
    from simpler.comm_provider import ProviderRegionStore, RegionControlError

    from tests.ut.py.test_worker.test_comm_provider import FakeShellFactory, _sim_context

    store = ProviderRegionStore(_sim_context(), _shell_factory=FakeShellFactory())
    mailbox = _StoreControlMailbox(store, reply_mode="empty")
    worker = _live_control_worker(mailbox, monkeypatch)
    with pytest.raises(RegionControlError):
        _materialize_default_region(worker)
    assert _tracked(worker) == ()
    assert mailbox.release_calls == 0
    _assert_poisoned(worker)
    with pytest.raises(RuntimeError, match="no further work is admitted"):
        _materialize_default_region(worker)


def test_dispatch_malformed_reply_does_not_release_and_poisons(monkeypatch):
    from simpler.comm_provider import ProviderRegionStore, RegionControlError

    from tests.ut.py.test_worker.test_comm_provider import FakeShellFactory, _sim_context

    store = ProviderRegionStore(_sim_context(), _shell_factory=FakeShellFactory())
    mailbox = _StoreControlMailbox(store, reply_mode="malformed")
    worker = _live_control_worker(mailbox, monkeypatch)
    with pytest.raises(RegionControlError):
        _materialize_default_region(worker)
    assert _tracked(worker) == ()
    assert mailbox.release_calls == 0
    _assert_poisoned(worker)
    with pytest.raises(RuntimeError, match="no further work is admitted"):
        _materialize_default_region(worker)


def test_request_encode_failure_does_not_dispatch_or_poison(monkeypatch):
    from simpler.comm_provider import ProviderRegionStore

    from tests.ut.py.test_worker.test_comm_provider import FakeShellFactory, _sim_context

    store = ProviderRegionStore(_sim_context(), _shell_factory=FakeShellFactory())
    mailbox = _StoreControlMailbox(store)
    worker = _live_control_worker(mailbox, monkeypatch)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("encode failed")

    monkeypatch.setattr("simpler.comm_region.encode_request", _boom)
    with pytest.raises(RuntimeError, match="encode failed"):
        _materialize_default_region(worker)
    assert mailbox.allocate_calls == 0
    assert mailbox.release_calls == 0
    assert _tracked(worker) == ()
    worker._require_no_ordered_cleanup_failure("test")


def test_mailbox_error_before_handler_poisons_and_keeps_transport_type(monkeypatch):
    from simpler.comm_provider import ProviderRegionStore

    from tests.ut.py.test_worker.test_comm_provider import FakeShellFactory, _sim_context

    store = ProviderRegionStore(_sim_context(), _shell_factory=FakeShellFactory())
    mailbox = _StoreControlMailbox(store, fail_before_allocate=RuntimeError("mailbox down before handler"))
    worker = _live_control_worker(mailbox, monkeypatch)
    with pytest.raises(RuntimeError, match="mailbox down before handler") as exc_info:
        _materialize_default_region(worker)
    assert mailbox.allocate_calls == 1
    assert mailbox.release_calls == 0
    assert _tracked(worker) == ()
    _assert_poisoned(worker, cause=exc_info.value)
    with pytest.raises(RuntimeError, match="no further work is admitted") as admission_info:
        _materialize_default_region(worker)
    assert admission_info.value.__cause__ is worker._delegated_session_fatal


def test_store_lifecycle_allocate_is_terminal_ambiguity(monkeypatch):
    from simpler.comm_provider import ProviderRegionStore, RegionControlError, RegionControlErrorKind

    from tests.ut.py.test_worker.test_comm_provider import FakeShellFactory, _sim_context

    factory = FakeShellFactory()
    store = ProviderRegionStore(_sim_context(), _shell_factory=factory)
    store.sweep()
    mailbox = _StoreControlMailbox(store)
    worker = _live_control_worker(mailbox, monkeypatch)
    with pytest.raises(RegionControlError) as exc_info:
        _materialize_default_region(worker)
    assert exc_info.value.kind is RegionControlErrorKind.STORE_LIFECYCLE
    assert mailbox.release_calls == 0
    assert factory.world.calls == []
    assert _tracked(worker) == ()
    _assert_poisoned(worker)
    with pytest.raises(RuntimeError, match="no further work is admitted"):
        _materialize_default_region(worker)


_DELEGATED_SESSION_A = b"\x10\x11\x12\x13\x14\x15\x16\x17"
_DELEGATED_SESSION_B = b"\x20\x21\x22\x23\x24\x25\x26\x27"


def _endpoint_session(session: bytes = _DELEGATED_SESSION_A, epoch: int = 3) -> SimpleNamespace:
    return SimpleNamespace(session_instance_id=session, registry_epoch=epoch)


class _RecordingLease:
    def __init__(self, name: str, calls: list[str], fail: Optional[BaseException] = None) -> None:
        self.name = name
        self.calls = calls
        self.fail = fail
        self.handle = 1

    def close(self) -> None:
        self.calls.append(self.name)
        if self.fail is not None:
            raise self.fail


def test_delegated_allocate_dispatch_is_serialized_and_burns_ids():
    registry = RegionInstanceRegistry()
    endpoints = _endpoint_session()
    holding = threading.Event()
    second_entered = threading.Event()
    order: list[tuple[str, int]] = []

    def first() -> None:
        with registry._delegated_allocate_dispatch(registry=endpoints, expected_registry_epoch=3) as ident:
            order.append(("first", ident[1]))
            holding.set()
            assert not second_entered.wait(timeout=0.2)
        order.append(("first_done", ident[1]))

    def second() -> None:
        holding.wait(timeout=2)
        with registry._delegated_allocate_dispatch(registry=endpoints, expected_registry_epoch=3) as ident:
            second_entered.set()
            order.append(("second", ident[1]))

    t1 = threading.Thread(target=first)
    t2 = threading.Thread(target=second)
    t1.start()
    t2.start()
    t1.join(timeout=2)
    t2.join(timeout=2)
    assert order == [("first", 1), ("first_done", 1), ("second", 2)]
    assert registry._next_delegated_transaction_id == 3
    assert registry._delegated_session_instance_id == _DELEGATED_SESSION_A


def test_delegated_allocate_dispatch_burns_id_when_encode_or_dispatch_fails():
    registry = RegionInstanceRegistry()
    endpoints = _endpoint_session()
    with pytest.raises(RuntimeError, match="encode failed"):
        with registry._delegated_allocate_dispatch(registry=endpoints, expected_registry_epoch=3) as ident:
            assert ident == (_DELEGATED_SESSION_A, 1)
            raise RuntimeError("encode failed")
    with registry._delegated_allocate_dispatch(registry=endpoints, expected_registry_epoch=3) as ident:
        assert ident == (_DELEGATED_SESSION_A, 2)


def test_delegated_allocate_dispatch_rejects_session_and_epoch_mismatch_before_side_effect():
    registry = RegionInstanceRegistry()
    with registry._delegated_allocate_dispatch(registry=_endpoint_session(), expected_registry_epoch=3) as ident:
        assert ident[1] == 1
    with pytest.raises(MaterializationRefusal) as session_exc:
        with registry._delegated_allocate_dispatch(
            registry=_endpoint_session(_DELEGATED_SESSION_B), expected_registry_epoch=3
        ):
            raise AssertionError("mismatch must fail before yield")
    assert session_exc.value.reason is RefusalReason.REGISTRY_MISMATCH
    with pytest.raises(MaterializationRefusal) as epoch_exc:
        with registry._delegated_allocate_dispatch(registry=_endpoint_session(epoch=4), expected_registry_epoch=3):
            raise AssertionError("epoch mismatch must fail before yield")
    assert epoch_exc.value.reason is RefusalReason.REGISTRY_MISMATCH
    with registry._delegated_allocate_dispatch(registry=_endpoint_session(), expected_registry_epoch=3) as ident:
        assert ident[1] == 2


def test_delegated_allocate_dispatch_closes_admission_at_uint64_limit():
    registry = RegionInstanceRegistry()
    registry._next_delegated_transaction_id = (1 << 64) - 1
    with registry._delegated_allocate_dispatch(registry=_endpoint_session(), expected_registry_epoch=3) as ident:
        assert ident[1] == (1 << 64) - 1
    with pytest.raises(MaterializationError, match="exhausted"):
        with registry._delegated_allocate_dispatch(registry=_endpoint_session(), expected_registry_epoch=3):
            raise AssertionError("must not wrap")
    registry._next_delegated_transaction_id = 0
    registry._delegated_admission_closed = False
    with pytest.raises(MaterializationError, match="0 is illegal"):
        with registry._delegated_allocate_dispatch(registry=_endpoint_session(), expected_registry_epoch=3):
            raise AssertionError("must not recycle 0")


def test_clean_retirement_keeps_allocator_and_has_no_lookup_api():
    ctx = _accepted_context()
    shape = validate_single_owner_region_shape(ctx)
    instance = RegionInstance.planned(ctx, shape)
    registry = ctx.worker._region_instance_registry
    with registry._delegated_allocate_dispatch(registry=_endpoint_session(), expected_registry_epoch=3) as ident:
        assert ident[1] == 1
    registry.track(instance, None)
    registry._retire(id(instance))
    assert id(instance) not in registry._instances
    with registry._delegated_allocate_dispatch(registry=_endpoint_session(), expected_registry_epoch=3) as ident:
        assert ident[1] == 2
    assert not hasattr(RegionInstanceRegistry, "get")
    assert not hasattr(RegionInstanceRegistry, "find")
    assert not hasattr(RegionInstanceRegistry, "__getitem__")


def test_delegated_identity_requires_track_and_skips_uncertainty_release():
    ctx = _accepted_context()
    shape = validate_single_owner_region_shape(ctx)
    instance = RegionInstance.planned(ctx, shape)
    releases: list[tuple[bytes, int, bytes]] = []

    def _dispatch(*, session_instance_id, transaction_id, provider_path):
        releases.append((session_instance_id, transaction_id, provider_path))
        return ProviderReleaseResult(provider_resource_id=11, status=ProviderReleaseStatus.RELEASED)

    ctx.worker._dispatch_delegated_release = _dispatch
    with pytest.raises(MaterializationError, match="tracked"):
        instance._bind_delegated_identity(_DELEGATED_SESSION_A, 1, b"L3/L2[1]")
    ctx.worker._region_instance_registry.track(instance, None)
    instance._bind_delegated_identity(_DELEGATED_SESSION_A, 1, b"L3/L2[1]")
    instance._state = RegionInstanceState.LIVE
    mapping_calls: list[str] = []
    instance._payload_mapping = _RecordingLease("payload", mapping_calls)
    instance._counter_mapping = _RecordingLease("counter", mapping_calls)
    instance._close_owned(poison_on_error=False)
    assert releases == []
    assert mapping_calls == ["payload", "counter"]
    assert instance._delegated_release_edge is False
    assert instance._provider_resource_id == 0


def test_committed_allocated_installs_release_edge_and_close_sends_once():
    ctx = _accepted_context()
    shape = validate_single_owner_region_shape(ctx)
    instance = RegionInstance.planned(ctx, shape)
    ctx.worker._region_instance_registry.track(instance, None)
    instance._bind_delegated_identity(_DELEGATED_SESSION_A, 4, b"L3/L2[1]")
    instance._commit_delegated_allocation(11)
    assert instance.provider_resource_id == 11
    assert instance._delegated_allocation_committed is True
    assert instance._delegated_release_edge is True
    releases: list[dict[str, object]] = []

    def _dispatch(**kwargs):
        releases.append(kwargs)
        return ProviderReleaseResult(provider_resource_id=11, status=ProviderReleaseStatus.RELEASED)

    ctx.worker._dispatch_delegated_release = _dispatch
    mapping_calls: list[str] = []
    payload_error = RuntimeError("payload mapping close failed")
    instance._state = RegionInstanceState.LIVE
    instance._payload_mapping = _RecordingLease("payload", mapping_calls, fail=payload_error)
    instance._counter_mapping = _RecordingLease("counter", mapping_calls)
    with pytest.raises(RuntimeError, match="payload mapping close failed"):
        instance._close_owned(poison_on_error=False)
    assert mapping_calls == ["payload", "counter"]
    assert releases == [
        {
            "session_instance_id": _DELEGATED_SESSION_A,
            "transaction_id": 4,
            "provider_path": b"L3/L2[1]",
        }
    ]
    with pytest.raises(RuntimeError, match="payload mapping close failed"):
        instance._close_owned(poison_on_error=False)
    assert len(releases) == 1


def test_delegated_shape_accepts_l3_and_l4():
    l3_ctx = _accepted_context()
    l3_shape = validate_single_owner_region_shape(l3_ctx)
    assert isinstance(l3_shape, DelegatedSingleOwnerRegionShape)
    assert l3_shape.consumer.path == "L3"
    assert l3_shape.provider.path == "L3/L2[1]"
    assert l3_shape.initiator_path == b"L3"
    assert l3_shape.provider_path == b"L3/L2[1]"
    assert l3_shape.first_hop_child_id == 1
    assert l3_shape.provider_device_id == 9

    l4_worker = _l4_with_local_l3(device_ids=[4])
    l4_ctx = _context(
        l4_worker,
        [ce.at("L4", ce.HOST_CPU), ce.at("L4/L3[0]/L2[0]", ce.DEVICE_AICPU)],
        ce.SingleOwner(provider=ce.at("L4/L3[0]/L2[0]", ce.DEVICE_AICPU)),
    )
    l4_shape = validate_single_owner_region_shape(l4_ctx)
    assert l4_shape.consumer.path == "L4"
    assert l4_shape.provider.path == "L4/L3[0]/L2[0]"
    assert l4_shape.first_hop_child_id == 0
    assert l4_shape.provider_device_id == 4


def test_delegated_shape_refuses_aicore_without_dispatch():
    worker = _l3(device_ids=[0])
    ctx = _context(
        worker,
        [ce.at("L3", ce.HOST_CPU), ce.at("L3/L2[0]", ce.DEVICE_AICORE)],
        ce.SingleOwner(provider=ce.at("L3/L2[0]", ce.DEVICE_AICORE)),
    )
    worker._dispatch_delegated_allocate = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no wire"))
    with pytest.raises(MaterializationRefusal) as excinfo:
        validate_single_owner_region_shape(ctx)
    assert excinfo.value.reason is RefusalReason.UNSUPPORTED_PROVIDER_DEPLOYMENT
    with pytest.raises(MaterializationRefusal) as core:
        materialize_region_instance(ctx)
    assert core.value.reason is RefusalReason.UNSUPPORTED_PROVIDER_DEPLOYMENT


def _posix_allocated_reply(session: bytes, transaction_id: int, *, payload_bytes: int, counter_bytes: int):
    from simpler.comm_provider import (
        PosixShmImport,
        RegionAllocationResult,
        RegionExportDescriptor,
        RegionPartExportDescriptor,
        RegionPartKind,
        RegionPartLocalView,
    )
    from simpler.comm_provider_control import DelegatedAllocateReply, DelegatedAllocateReplyTag, encode_reply

    result = RegionAllocationResult(
        provider_resource_id=11,
        export_descriptor=RegionExportDescriptor(
            payload=RegionPartExportDescriptor(
                planned_backing_kind=ce.BackendKind.VMM_WINDOW,
                logical_bytes=payload_bytes,
                mapping_bytes=payload_bytes,
                import_capability=PosixShmImport(shm_name="/pto_payload_a"),
            ),
            counter=RegionPartExportDescriptor(
                planned_backing_kind=ce.BackendKind.VMM_WINDOW,
                logical_bytes=counter_bytes,
                mapping_bytes=counter_bytes,
                import_capability=PosixShmImport(shm_name="/pto_counter_a"),
            ),
        ),
    )
    payload = RegionPartLocalView(part=RegionPartKind.PAYLOAD, local_base=0x1000, logical_bytes=payload_bytes)
    counter = RegionPartLocalView(part=RegionPartKind.COUNTER, local_base=0x2000, logical_bytes=counter_bytes)
    return encode_reply(
        DelegatedAllocateReply(
            tag=DelegatedAllocateReplyTag.ALLOCATED,
            session_instance_id=session,
            transaction_id=transaction_id,
            result=result,
            payload_view=payload,
            counter_view=counter,
        )
    )


class _TestDelegatedImportFailure(BaseException):
    pass


def _posix_inconsistent_allocated_reply(session: bytes, transaction_id: int, *, payload_bytes: int, counter_bytes: int):
    from simpler.comm_provider import (
        PosixShmImport,
        RegionAllocationResult,
        RegionExportDescriptor,
        RegionPartExportDescriptor,
        RegionPartKind,
        RegionPartLocalView,
    )
    from simpler.comm_provider_control import DelegatedAllocateReply, DelegatedAllocateReplyTag, encode_reply

    result = RegionAllocationResult(
        provider_resource_id=11,
        export_descriptor=RegionExportDescriptor(
            payload=RegionPartExportDescriptor(
                planned_backing_kind=ce.BackendKind.VMM_WINDOW,
                logical_bytes=payload_bytes,
                mapping_bytes=payload_bytes,
                import_capability=PosixShmImport(shm_name="/pto_same_token"),
            ),
            counter=RegionPartExportDescriptor(
                planned_backing_kind=ce.BackendKind.VMM_WINDOW,
                logical_bytes=counter_bytes,
                mapping_bytes=counter_bytes,
                import_capability=PosixShmImport(shm_name="/pto_same_token"),
            ),
        ),
    )
    payload = RegionPartLocalView(part=RegionPartKind.PAYLOAD, local_base=0x1000, logical_bytes=payload_bytes)
    counter = RegionPartLocalView(part=RegionPartKind.COUNTER, local_base=0x2000, logical_bytes=counter_bytes)
    return encode_reply(
        DelegatedAllocateReply(
            tag=DelegatedAllocateReplyTag.ALLOCATED,
            session_instance_id=session,
            transaction_id=transaction_id,
            result=result,
            payload_view=payload,
            counter_view=counter,
        )
    )


def _backend_failure_reply(session: bytes, transaction_id: int):
    from simpler.comm_provider_control import DelegatedAllocateReply, DelegatedAllocateReplyTag, encode_reply

    return encode_reply(
        DelegatedAllocateReply(
            tag=DelegatedAllocateReplyTag.ERROR,
            session_instance_id=session,
            transaction_id=transaction_id,
            error=RegionAllocationError(
                provisional_resource_id=7,
                control_kind=RegionControlErrorKind.BACKEND_FAILURE,
                failed_part=RegionPartKind.COUNTER,
                failed_operation=RegionOperationKind.ZERO_BYTES,
                cleanup_debt_remaining=False,
            ),
        )
    )


def _install_delegated_dispatch(
    worker: Worker,
    dispatches: list[bytes],
    *,
    close_calls: Optional[list[str]] = None,
    import_calls: Optional[list[str]] = None,
    fail_on_import: Optional[int] = None,
    import_error: Optional[BaseException] = None,
    reply_factory=None,
) -> None:
    from simpler.comm_provider import PosixShmImport
    from simpler.comm_provider_control import parse_request

    close_calls = close_calls if close_calls is not None else []
    import_calls = import_calls if import_calls is not None else []

    def _dispatch(staged):
        view = staged if isinstance(staged, memoryview) else memoryview(staged)
        envelope = parse_request(view)
        request = envelope.decode_terminal()
        dispatches.append(view.tobytes())
        factory = reply_factory if reply_factory is not None else _posix_allocated_reply
        return factory(
            request.session_instance_id,
            request.transaction_id,
            payload_bytes=int(request.payload_logical_bytes),
            counter_bytes=int(request.counter_logical_bytes),
        )

    def _import(*_args, **_kwargs):
        name = "payload" if len(import_calls) == 0 else "counter"
        import_calls.append(name)
        if fail_on_import is not None and len(import_calls) == fail_on_import:
            assert import_error is not None
            raise import_error
        return _RecordingLease(name, close_calls)

    worker._dispatch_delegated_allocate = _dispatch
    worker._import_region_part_lease = _import
    worker._provider_import_capability_type = lambda: PosixShmImport


def test_l3_and_l4_plans_use_the_same_delegated_materializer_core():
    l3_dispatches: list[bytes] = []
    l3_ctx = _accepted_context()
    _install_delegated_dispatch(l3_ctx.worker, l3_dispatches)
    l3_instance = materialize_region_instance(l3_ctx)
    assert l3_instance.state is RegionInstanceState.LIVE
    assert l3_instance.provider_resource_id == 11
    assert l3_instance._delegated_release_edge is True
    assert len(l3_dispatches) == 1

    l4_dispatches: list[bytes] = []
    l4_worker = _l4_with_local_l3(device_ids=[4])
    l4_ctx = _context(
        l4_worker,
        [ce.at("L4", ce.HOST_CPU), ce.at("L4/L3[0]/L2[0]", ce.DEVICE_AICPU)],
        ce.SingleOwner(provider=ce.at("L4/L3[0]/L2[0]", ce.DEVICE_AICPU)),
    )
    _install_delegated_dispatch(l4_worker, l4_dispatches)
    l4_instance = materialize_region_instance(l4_ctx)
    assert l4_instance.state is RegionInstanceState.LIVE
    assert l4_instance._delegated_provider_path == b"L4/L3[0]/L2[0]"
    assert len(l4_dispatches) == 1
    from simpler.comm_provider_control import parse_request

    l3_req = parse_request(l3_dispatches[0]).decode_terminal()
    l4_req = parse_request(l4_dispatches[0]).decode_terminal()
    assert type(l3_req) is type(l4_req)
    assert l3_req.provider_path == b"L3/L2[1]"
    assert l4_req.provider_path == b"L4/L3[0]/L2[0]"
    assert len(l3_dispatches[0]) >= 256
    assert len(l4_dispatches[0]) >= 256


@pytest.mark.parametrize(
    ("fail_on_import", "error_factory", "expected_imports", "expected_closes"),
    [
        (1, lambda: RuntimeError("payload import failed"), ["payload"], []),
        (2, lambda: RuntimeError("counter import failed"), ["payload", "counter"], ["payload"]),
        (1, lambda: _TestDelegatedImportFailure("payload import failed"), ["payload"], []),
        (2, lambda: _TestDelegatedImportFailure("counter import failed"), ["payload", "counter"], ["payload"]),
    ],
)
def test_delegated_import_failure_sends_one_release(fail_on_import, error_factory, expected_imports, expected_closes):
    ctx = _accepted_context()
    captured: dict[str, RegionInstance] = {}
    original_track = ctx.worker._region_instance_registry.track

    def _track(instance, resources):
        captured["instance"] = instance
        return original_track(instance, resources)

    ctx.worker._region_instance_registry.track = _track  # type: ignore[method-assign]
    dispatches: list[bytes] = []
    releases: list[dict[str, object]] = []
    close_calls: list[str] = []
    import_calls: list[str] = []
    error = error_factory()
    _install_delegated_dispatch(
        ctx.worker,
        dispatches,
        close_calls=close_calls,
        import_calls=import_calls,
        fail_on_import=fail_on_import,
        import_error=error,
    )

    def _release(**kwargs):
        releases.append(kwargs)
        return ProviderReleaseResult(provider_resource_id=11, status=ProviderReleaseStatus.RELEASED)

    ctx.worker._dispatch_delegated_release = _release
    with pytest.raises(type(error)) as excinfo:
        materialize_region_instance(ctx)
    assert excinfo.value is error
    assert import_calls == expected_imports
    assert close_calls == expected_closes
    assert len(dispatches) == 1
    assert len(dispatches[0]) >= 256
    assert len(releases) == 1
    assert releases[0] == {
        "session_instance_id": ctx.registry.session_instance_id,
        "transaction_id": 1,
        "provider_path": b"L3/L2[1]",
    }
    instance = captured["instance"]
    assert instance._delegated_release_edge is False
    assert id(instance) not in ctx.worker._region_instance_registry._instances
    assert ctx.worker._region_instance_registry._next_delegated_transaction_id == 2


def test_delegated_malformed_allocated_reply_does_not_import_or_release():
    ctx = _accepted_context()
    captured: dict[str, RegionInstance] = {}
    original_track = ctx.worker._region_instance_registry.track

    def _track(instance, resources):
        captured["instance"] = instance
        return original_track(instance, resources)

    ctx.worker._region_instance_registry.track = _track  # type: ignore[method-assign]
    dispatches: list[bytes] = []
    releases: list[dict[str, object]] = []
    close_calls: list[str] = []
    import_calls: list[str] = []
    _install_delegated_dispatch(
        ctx.worker,
        dispatches,
        close_calls=close_calls,
        import_calls=import_calls,
        reply_factory=_posix_inconsistent_allocated_reply,
    )
    ctx.worker._dispatch_delegated_release = lambda **kwargs: releases.append(kwargs) or ProviderReleaseResult(
        provider_resource_id=11, status=ProviderReleaseStatus.RELEASED
    )
    with pytest.raises(RuntimeError, match="POSIX shm tokens must be distinct") as excinfo:
        materialize_region_instance(ctx)
    instance = captured["instance"]
    fatal = ctx.worker._delegated_session_fatal
    assert fatal is not None
    assert excinfo.value is fatal
    assert instance._delegated_allocation_committed is False
    assert instance._delegated_release_edge is False
    assert instance._provider_resource_id == 0
    assert import_calls == []
    assert close_calls == []
    assert releases == []
    assert ctx.worker._region_instance_registry._delegated_admission_closed is True
    assert id(instance) not in ctx.worker._region_instance_registry._instances


def test_delegated_clean_backend_failure_preserves_typed_fields():
    ctx = _accepted_context()
    captured: dict[str, RegionInstance] = {}
    original_track = ctx.worker._region_instance_registry.track

    def _track(instance, resources):
        captured["instance"] = instance
        return original_track(instance, resources)

    ctx.worker._region_instance_registry.track = _track  # type: ignore[method-assign]
    dispatches: list[bytes] = []
    releases: list[dict[str, object]] = []
    close_calls: list[str] = []
    import_calls: list[str] = []

    def _error_reply(session, transaction_id, **_kwargs):
        return _backend_failure_reply(session, transaction_id)

    _install_delegated_dispatch(
        ctx.worker,
        dispatches,
        close_calls=close_calls,
        import_calls=import_calls,
        reply_factory=_error_reply,
    )
    ctx.worker._dispatch_delegated_release = lambda **kwargs: releases.append(kwargs) or ProviderReleaseResult(
        provider_resource_id=7, status=ProviderReleaseStatus.RELEASED
    )
    with pytest.raises(RegionAllocationError) as excinfo:
        materialize_region_instance(ctx)
    error = excinfo.value
    assert error.control_kind is RegionControlErrorKind.BACKEND_FAILURE
    assert error.failed_part is RegionPartKind.COUNTER
    assert error.failed_operation is RegionOperationKind.ZERO_BYTES
    assert error.provisional_resource_id == 7
    assert error.cleanup_debt_remaining is False
    assert ctx.worker._delegated_session_fatal is None
    assert ctx.worker._region_instance_registry._delegated_admission_closed is False
    assert import_calls == []
    assert close_calls == []
    assert releases == []
    instance = captured["instance"]
    assert instance._delegated_release_edge is False
    assert ctx.worker._region_instance_registry._next_delegated_transaction_id == 2
    success_dispatches: list[bytes] = []
    _install_delegated_dispatch(ctx.worker, success_dispatches)
    live = materialize_region_instance(ctx)
    assert live._delegated_transaction_id == 2
    assert live.state is RegionInstanceState.LIVE
