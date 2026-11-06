# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


import os
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from simpler import worker as worker_module
from simpler.buffer import (
    AccessMode,
    AddressSpace,
    BackendKind,
    ImportContext,
    ImportRegistry,
    create_host_shared_buffer,
    mint_owner_instance_id,
    select_adapter,
)
from simpler.comm_endpoints import (
    DEVICE_AICORE,
    DEVICE_AICPU,
    HOST_CPU,
    AdapterKind,
    AdapterProfile,
    BufferAccessQuery,
    DefaultRegionAccessService,
    _AdapterCandidate,
)
from simpler.task_interface import (
    ArgDirection,
    DataType,
    TaskArgs,
    Tensor,
    TensorArgType,
    TensorTransfer,
    _materialize_chip_task_args,
)
from simpler.worker import Worker


@pytest.fixture
def host_worker():
    worker = Worker(
        level=3, platform="a2a3sim", runtime="tensormap_and_ringbuffer", device_ids=[0, 1], num_sub_workers=0
    )
    worker.init()
    try:
        yield worker
    finally:
        worker.close()


@pytest.fixture
def direct_worker(monkeypatch):
    worker = Worker(level=2, platform="a2a3sim", runtime="tensormap_and_ringbuffer", device_id=0)
    chip = MagicMock()
    chip._execution_environment = (os.getpid(), 1, 1, 0, False)
    worker._lifecycle = worker_module._Lifecycle.READY
    worker._host_runtime_path = None
    worker._chip_worker = chip
    monkeypatch.setattr(worker, "_device_control_admission", lambda _api: nullcontext())
    return worker


@pytest.fixture
def direct_host(direct_worker):
    buffer = direct_worker.create_buffer(128)
    native = buffer.tensor((16,), DataType.FLOAT32)
    try:
        yield direct_worker, buffer, native
    finally:
        direct_worker._source_context_failed = False
        direct_worker._chip_run_touched_identities.clear()
        resource = direct_worker._host_buffers.get(buffer.identity)
        if resource is not None:
            resource.closed = buffer.closed = False
            for reference in tuple(resource._mapping_targets):
                direct_worker.unmap_view(native._with_mapping_ref(reference))
        direct_worker.release_buffer(buffer)


def test_host_maps_local_endpoints_without_domain(host_worker):
    buffers = [host_worker.create_buffer(128) for _ in range(2)]
    for buffer in buffers:
        native = buffer.tensor((16,), DataType.FLOAT32)
        views = [host_worker.map_view(native, target=worker, access=AccessMode.READ) for worker in range(2)]
        for worker, view in enumerate(views):
            args = TaskArgs()
            args.add_tensor(view, transfer=TensorTransfer.NONE)
            with host_worker._child_prov_lock:
                host_worker._child_prov_check_dispatch_locked(args, worker, api="test")
                with pytest.raises(ValueError, match="target worker"):
                    host_worker._child_prov_check_dispatch_locked(args, 1 - worker, api="test")
            host_worker.unmap_view(view)
            with pytest.raises(ValueError, match="live explicit mapping"):
                host_worker._host_buffers[buffer.identity].validate(view, worker)
        host_worker.release_buffer(buffer)
    assert not host_worker._live_domains


def test_create_buffer_uses_existing_allocator(direct_worker, monkeypatch):
    direct_worker.level = 4
    direct_worker._config["platform"] = "a2a3"
    allocate = Mock(return_value=object())
    monkeypatch.setattr(direct_worker, "_create_buffer_locked", allocate)
    assert direct_worker.create_buffer(67) is allocate.return_value
    allocate.assert_called_once_with(67)


@pytest.mark.parametrize("target", [None, True, -1, 2, "0"])
def test_host_map_rejects_invalid_local_endpoint(host_worker, target):
    buffer = host_worker.create_buffer(128)
    try:
        with pytest.raises(ValueError, match="target"):
            host_worker.map_view(buffer.tensor((16,), DataType.FLOAT32), target=target)
    finally:
        host_worker.release_buffer(buffer)


def test_map_view_issues_only_requested_record(direct_host):
    worker, buffer, native = direct_host
    mapped = worker.map_view(native, access=AccessMode.READ)
    registry = worker._chip_import_registry
    assert registry.resolve_tensor(mapped).base > 0
    args = TaskArgs()
    target = SimpleNamespace(sig_count=1, sig=lambda _index: ArgDirection.OUT)
    args.add_tensor(
        native._with_mapping_ref(mapped.mapping_ref + 1), TensorArgType.OUTPUT_EXISTING, transfer=TensorTransfer.NONE
    )
    with pytest.raises(ValueError, match="no granted attachment"):
        _materialize_chip_task_args(args, registry, target)
    args.clear()
    args.add_tensor(mapped, transfer=TensorTransfer.NONE)
    with pytest.raises(ValueError, match="mapping grant"):
        _materialize_chip_task_args(args, registry, target)


def test_unmap_preserves_native_and_remap_never_revives_old_reference(direct_host):
    worker, buffer, native = direct_host
    first = worker.map_view(native, access=AccessMode.READ)
    worker.unmap_view(first)
    assert not buffer.closed
    buffer.buffer[:4] = b"live"
    imported = worker._chip_import_registry.resolve_tensor(native)
    assert bytes(imported.shm.buf[:4]) == b"live"
    with pytest.raises(ValueError, match="no granted attachment"):
        worker._chip_import_registry.resolve_tensor(first)
    second = worker.map_view(native, access=AccessMode.READ)
    assert second.mapping_ref != first.mapping_ref
    with pytest.raises(ValueError, match="no granted attachment"):
        worker._chip_import_registry.resolve_tensor(first)
    worker.unmap_view(first)
    assert worker._chip_import_registry.resolve_tensor(second).base > 0


def test_unmap_shared_import_only_detaches_last_record(direct_host):
    worker, buffer, native = direct_host
    read = worker.map_view(native, access=AccessMode.READ)
    write = worker.map_view(native, access=AccessMode.WRITE)
    shared = worker._chip_import_registry.resolve_tensor(write).host_backing
    assert worker._chip_import_registry.resolve_tensor(read).host_backing is shared
    worker.unmap_view(read)
    assert worker._chip_import_registry.resolve_tensor(write).base > 0
    worker.unmap_view(write)


def test_unmap_failure_revokes_and_retains_cleanup_until_retry(direct_host, monkeypatch):
    worker, buffer, native = direct_host
    mapped = worker.map_view(native, access=AccessMode.READ)
    imported = worker._chip_import_registry.resolve_tensor(mapped).host_backing
    original = imported.release
    attempts = []

    def release():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("injected detach failure")
        original()

    monkeypatch.setattr(imported, "release", release)
    with pytest.raises(RuntimeError, match="injected detach"):
        worker.unmap_view(mapped)
    with pytest.raises(ValueError, match="not live"):
        worker._chip_import_registry.resolve_tensor(mapped)
    with pytest.raises(RuntimeError, match="cleanup is unconfirmed"):
        worker.map_view(native)
    worker.unmap_view(mapped)
    remapped = worker.map_view(native)
    assert remapped.mapping_ref != mapped.mapping_ref


@pytest.mark.parametrize("unmap", [False, True])
def test_host_view_lost_reply_keeps_revoked_route_and_blocks_reuse(direct_host, monkeypatch, unmap):
    worker, buffer, native = direct_host
    first = worker.map_view(native)
    resource = worker._host_buffers[buffer.identity]
    original = resource._view_control

    def lost_reply(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("lost reply")

    monkeypatch.setattr(resource, "_view_control", lost_reply)
    with pytest.raises(RuntimeError, match="lost reply"):
        worker.unmap_view(first) if unmap else worker.map_view(native)
    # Only the originally returned Tensor is needed for failed unmap;
    # a failed map is recoverable by retrying map_view, without guessing r.
    if unmap:
        with pytest.raises(RuntimeError, match="cleanup is unconfirmed"):
            worker.map_view(native)
    monkeypatch.setattr(resource, "_view_control", original)
    if unmap:
        worker.unmap_view(first)
    fresh = worker.map_view(native)
    assert fresh.mapping_ref > first.mapping_ref


def test_unmap_refuses_inflight_before_revocation(direct_host):
    worker, buffer, native = direct_host
    mapped = worker.map_view(buffer.tensor((16,), DataType.FLOAT32))
    worker._chip_run_touched_identities[1] = {buffer.identity}
    with pytest.raises(RuntimeError, match="in-flight"):
        worker.unmap_view(mapped)
    assert worker._chip_import_registry.resolve_tensor(mapped).base > 0
    worker._chip_run_touched_identities.clear()
    worker.unmap_view(mapped)


def test_mapping_reference_exhaustion_does_not_reuse_retired_ids(direct_host):
    worker, buffer, native = direct_host
    worker.map_view(native)
    worker._host_buffers[buffer.identity]._next_reference = 254
    for reference in range(254, 256):
        mapped = worker.map_view(native)
        assert mapped.mapping_ref == reference
        worker.unmap_view(mapped)
    with pytest.raises(RuntimeError, match="references exhausted"):
        worker.map_view(native)


@pytest.mark.parametrize("owner_closed", [False, True])
def test_mapped_record_rejects_closed_owner_or_unconfirmed_completion(direct_host, owner_closed):
    worker, buffer, native = direct_host
    mapped = worker.map_view(buffer.tensor((16,), DataType.FLOAT32))
    if owner_closed:
        worker._host_buffers[buffer.identity].closed = True
    else:
        worker._source_context_failed = True
    with pytest.raises(ValueError, match="not live"):
        worker._chip_import_registry.resolve_tensor(mapped)


_REFS = {AccessMode.READWRITE: 1, AccessMode.READ: 2, AccessMode.WRITE: 3}


@pytest.fixture
def mapping():
    buffer = create_host_shared_buffer(64, mint_owner_instance_id(), 1)
    environment = (os.getpid(), 1, 1, 0, False)
    registry = ImportRegistry(
        ImportContext(
            DEVICE_AICPU,
            platform="a2a3",
            runtime="tensormap_and_ringbuffer",
            execution_environment=environment,
            current_environment=lambda: environment,
        )
    )
    for access in (AccessMode.READWRITE, AccessMode.READ, AccessMode.WRITE):
        registry.register_attachment(
            buffer.to_descriptor(),
            8192,
            mapping_ref=_REFS[access],
            access=access,
            profile=AdapterProfile.HOST_REGISTER_MAP,
            live=lambda: True,
        )
    try:
        yield buffer, registry
    finally:
        registry.close()
        buffer.close()


@pytest.mark.parametrize(
    "access,tag,direction,allowed",
    [
        (AccessMode.READ, TensorArgType.INPUT, ArgDirection.IN, True),
        (AccessMode.READ, TensorArgType.INPUT, ArgDirection.OUT, False),
        (AccessMode.READ, TensorArgType.INOUT, ArgDirection.INOUT, False),
        (AccessMode.WRITE, TensorArgType.INPUT, ArgDirection.OUT, True),
        (AccessMode.WRITE, TensorArgType.INPUT, ArgDirection.IN, False),
        (AccessMode.READWRITE, TensorArgType.INOUT, ArgDirection.INOUT, True),
    ],
)
def test_selected_mapping_grant(mapping, access, tag, direction, allowed):
    buffer, registry = mapping
    native = buffer.tensor((16,), DataType.FLOAT32)
    view = native._with_mapping_ref(_REFS[access])
    args = TaskArgs()
    args.add_tensor(view, tag, transfer=TensorTransfer.NONE)
    target = SimpleNamespace(sig_count=1, sig=lambda index: direction)
    if allowed:
        assert _materialize_chip_task_args(args, registry, target).tensor(0).data == 8192
    else:
        with pytest.raises(ValueError, match="mapping grant"):
            _materialize_chip_task_args(args, registry, target)
    assert native.mapping_ref == 0 and view.buffer == native.buffer


def test_mapped_tensor_cannot_replace_its_mapping_reference(mapping):
    buffer, _ = mapping
    read = buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(_REFS[AccessMode.READ])
    assert read._with_mapping_ref(read.mapping_ref) == read
    with pytest.raises(ValueError):
        read._with_mapping_ref(_REFS[AccessMode.WRITE])


def test_mapping_rejects_grant_and_descriptor_changes(mapping):
    buffer, registry = mapping
    with pytest.raises(ValueError, match="conflicts"):
        registry.register_attachment(
            buffer.to_descriptor(),
            8192,
            mapping_ref=_REFS[AccessMode.READ],
            access=AccessMode.READWRITE,
            profile=AdapterProfile.HOST_REGISTER_MAP,
            live=lambda: True,
        )
    forged = replace(buffer, nbytes=128).tensor((16,), DataType.FLOAT32)._with_mapping_ref(_REFS[AccessMode.READ])
    with pytest.raises(ValueError, match="descriptor differs"):
        registry.resolve_tensor(forged)
    active = buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(_REFS[AccessMode.READ])
    assert registry.resolve_tensor(active).base == 8192
    registry.unregister(buffer.identity)
    with pytest.raises(ValueError, match="no granted attachment"):
        registry.resolve_tensor(active)


def test_read_mapping_preflight_precedes_other_argument_materialization(mapping, monkeypatch):
    buffer, registry = mapping
    args = TaskArgs()
    args.add_tensor(buffer.tensor((16,), DataType.FLOAT32), transfer=TensorTransfer.H2D)
    view = buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(_REFS[AccessMode.READ])
    args.add_tensor(view, TensorArgType.OUTPUT, transfer=TensorTransfer.NONE)
    monkeypatch.setattr(registry, "materialize", lambda _: pytest.fail("must reject before ordinary import"))
    target = SimpleNamespace(sig_count=2, sig=lambda _index: ArgDirection.OUT)
    with pytest.raises(ValueError, match="mapping grant"):
        _materialize_chip_task_args(args, registry, target)


@pytest.mark.parametrize(
    "target", [None, SimpleNamespace(sig_count=0), SimpleNamespace(sig_count=2, sig=lambda _: None)]
)
def test_mapping_requires_callable_direction_before_materialization(mapping, monkeypatch, target):
    buffer, registry = mapping
    args = TaskArgs()
    args.add_tensor(buffer.tensor((16,), DataType.FLOAT32), transfer=TensorTransfer.H2D)
    view = buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(_REFS[AccessMode.READ])
    args.add_tensor(view, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
    monkeypatch.setattr(registry, "materialize", lambda _: pytest.fail("must reject before ordinary import"))
    with pytest.raises(ValueError, match="declared callable direction"):
        _materialize_chip_task_args(args, registry, target)


def test_attachment_cannot_widen_source_grant(mapping):
    buffer, registry = mapping
    buffer.access = AccessMode.READ
    registry = ImportRegistry(registry._context)
    with pytest.raises(ValueError, match="exceeds source grant"):
        registry.register_attachment(
            buffer.to_descriptor(),
            8192,
            mapping_ref=5,
            access=AccessMode.READWRITE,
            profile=AdapterProfile.HOST_REGISTER_MAP,
            live=lambda: True,
        )


def test_close_attachment_revokes_only_its_record_and_cannot_be_registered_again(mapping):
    buffer, registry = mapping
    native = buffer.tensor((16,), DataType.FLOAT32)
    native_import = registry.materialize(buffer.to_descriptor())
    read = native._with_mapping_ref(_REFS[AccessMode.READ])
    write = native._with_mapping_ref(_REFS[AccessMode.WRITE])
    registry.close_attachment(buffer.identity, read.mapping_ref)
    registry.close_attachment(buffer.identity, read.mapping_ref)
    with pytest.raises(ValueError, match="not live"):
        registry.resolve_tensor(read)
    assert registry.resolve_tensor(write).base == 8192
    assert registry.has_active_attachments(buffer.identity)
    with pytest.raises(ValueError, match="conflicts"):
        registry.register_attachment(
            buffer.to_descriptor(),
            8192,
            mapping_ref=read.mapping_ref,
            access=AccessMode.READ,
            profile=AdapterProfile.HOST_REGISTER_MAP,
            live=lambda: True,
        )
    registry.close_attachment(buffer.identity, write.mapping_ref)
    registry.close_attachment(buffer.identity, _REFS[AccessMode.READWRITE])
    assert not registry.has_active_attachments(buffer.identity)
    assert registry.resolve_tensor(native) is native_import
    registry.unregister(buffer.identity)
    assert not registry.has_active_attachments(buffer.identity)
    with pytest.raises(ValueError, match="no granted attachment"):
        registry.resolve_tensor(read)


@pytest.mark.parametrize("index", range(5))
def test_mapping_rejects_changed_target_environment(mapping, index):
    buffer, original = mapping
    environment = original._context.execution_environment
    current = [environment]
    registry = ImportRegistry(replace(original._context, current_environment=lambda: current[0]))
    registry.register_attachment(
        buffer.to_descriptor(),
        8192,
        mapping_ref=1,
        profile=AdapterProfile.HOST_REGISTER_MAP,
        live=lambda: True,
    )
    view = buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(1)
    assert registry.resolve_tensor(view).execution_environment == environment
    changed = list(environment)
    changed[index] = not changed[index] if index == 4 else changed[index] + 1
    current[0] = tuple(changed)
    with pytest.raises(ValueError, match="not live"):
        registry.resolve_tensor(view)
    current[0] = environment
    with pytest.raises(ValueError, match="not live"):
        registry.resolve_tensor(view)


@pytest.mark.parametrize(
    "context_changes",
    [
        {"execution_environment": None},
        {"execution_environment": (1, 2)},
        {"current_environment": None},
        {"current_environment": lambda: None},
    ],
)
def test_mapping_registration_requires_current_target_environment(mapping, context_changes):
    buffer, original = mapping
    registry = ImportRegistry(replace(original._context, **context_changes))
    with pytest.raises(ValueError, match="execution environment"):
        registry.register_attachment(
            buffer.to_descriptor(),
            8192,
            mapping_ref=1,
            profile=AdapterProfile.HOST_REGISTER_MAP,
            live=lambda: True,
        )


@pytest.mark.parametrize(
    "profile,deployment,platform,supported",
    [
        (AdapterProfile.HOST_REGISTER_MAP, DEVICE_AICPU, "a2a3", True),
        (AdapterProfile.HOST_REGISTER_MAP, DEVICE_AICORE, "a2a3", True),
        (AdapterProfile.HOST_SHM_MAP, DEVICE_AICPU, "a2a3sim", True),
        (AdapterProfile.HOST_SHM_MAP, HOST_CPU, "a2a3sim", True),
        (AdapterProfile.HOST_REGISTER_MAP, HOST_CPU, "a2a3", False),
        (AdapterProfile.HOST_REGISTER_MAP, DEVICE_AICPU, "a5", False),
        (AdapterProfile.HOST_REGISTER_MAP, DEVICE_AICPU, "a2a3sim", True),
        (AdapterProfile.HOST_SHM_MAP, DEVICE_AICPU, "a2a3", False),
        (AdapterProfile.REMOTE_COPY, DEVICE_AICPU, "a2a3", False),
        (AdapterProfile.DEVICE_LOCAL, DEVICE_AICPU, "a2a3", False),
    ],
)
def test_confirmed_host_adapter_uses_platform_and_consumer_policy(mapping, profile, deployment, platform, supported):
    buffer, original = mapping
    context = replace(original._context, deployment=deployment, platform=platform)
    kwargs = {"confirmed_profile": profile, "mapped": deployment is not HOST_CPU}
    if supported:
        assert select_adapter(buffer.to_descriptor(), context, **kwargs) == (AdapterKind.DIRECT_MAP, profile)
    else:
        with pytest.raises(ValueError, match="UNSUPPORTED_ENDPOINT_RELATION"):
            select_adapter(buffer.to_descriptor(), context, **kwargs)


@pytest.mark.parametrize(
    "query_changes",
    [
        {"source_address_space": AddressSpace.DEVICE},
        {"same_node": False},
        {"runtime": "unknown"},
        {"platform": "a5"},
    ],
)
def test_confirmed_profile_cannot_bypass_default_service_by_stripping_diagnostic(query_changes):
    query = BufferAccessQuery(
        BackendKind.POSIX_SHM,
        DEVICE_AICPU,
        same_node=True,
        platform="a2a3",
        runtime="tensormap_and_ringbuffer",
        confirmed_profile=AdapterProfile.HOST_REGISTER_MAP,
        source_address_space=AddressSpace.HOST,
    )
    candidate = _AdapterCandidate(AdapterKind.DIRECT_MAP, AdapterProfile.HOST_REGISTER_MAP)
    decision = DefaultRegionAccessService().evaluate_buffer_access(replace(query, **query_changes), candidate)
    assert not decision.supported


class TestExplicitMappedTensor:
    @pytest.fixture
    def mapping_context(self):
        environment = (os.getpid(), 1, 1, 0, False)
        return ImportContext(
            DEVICE_AICPU,
            platform="a2a3",
            runtime="tensormap_and_ringbuffer",
            execution_environment=environment,
            current_environment=lambda: environment,
        )

    def test_native_and_mapped_arguments_keep_one_backing(self, mapping_context):
        from types import SimpleNamespace  # noqa: PLC0415

        from simpler.task_interface import ArgDirection, TensorArgType, _materialize_chip_task_args  # noqa: PLC0415

        backing = create_host_shared_buffer(128, mint_owner_instance_id(), 1)
        registry = ImportRegistry(mapping_context)
        try:
            native = Tensor(backing, shapes=(8,), dtype=DataType.FLOAT32, byte_offset=16)
            mapped = native._with_mapping_ref(1)
            registry.register_attachment(
                backing.to_descriptor(),
                0x8000,
                mapping_ref=1,
                profile=AdapterProfile.HOST_REGISTER_MAP,
                live=lambda: True,
            )
            args = TaskArgs()
            args.add_tensor(native, TensorArgType.INPUT)
            args.add_tensor(mapped, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
            target = SimpleNamespace(sig_count=2, sig=lambda _index: ArgDirection.IN)
            materialized = _materialize_chip_task_args(args, registry, target)
            assert mapped.buffer == native.buffer
            assert mapped.buffer.address_space == AddressSpace.HOST
            assert mapped.mapping_ref == 1 and native.mapping_ref == 0
            assert materialized.tensor(0).data != materialized.tensor(1).data
            assert materialized.tensor(1).data == 0x8010
            assert materialized.tensor(0).address_space == AddressSpace.HOST
            assert materialized.tensor(1).address_space == AddressSpace.DEVICE
            with pytest.raises(ValueError, match="transfer NONE"):
                TaskArgs().add_tensor(mapped)
        finally:
            registry.close()
            backing.close()


def test_device_free_retires_the_selected_mapping(direct_worker):
    direct_worker._chip_worker.malloc.return_value = 4096
    buffer = direct_worker.malloc(64)
    mapped = direct_worker.map_view(buffer.tensor((16,), DataType.FLOAT32), access=AccessMode.READ)
    registry = direct_worker._ensure_chip_import_registry()
    assert registry.resolve_tensor(mapped).base == 4096
    direct_worker.free(buffer)
    assert buffer.identity not in direct_worker._device_mappings
    with pytest.raises(ValueError, match="no granted attachment"):
        registry.resolve_tensor(mapped)


@pytest.mark.parametrize("failure_marker", ["native", "touched", "abandoned"])
def test_unknown_completion_retains_device_import_before_native_teardown(direct_host, failure_marker):
    worker, _buffer, native = direct_host
    mapped = worker.map_view(native, access=AccessMode.READ)
    registry = worker._chip_import_registry
    assert registry is not None
    imported = registry.resolve_tensor(mapped)
    release = Mock(wraps=imported.host_backing.release)
    imported.host_backing.release = release
    abandoned = MagicMock()
    abandoned._resources.touched_identities = {native.buffer.identity}
    if failure_marker == "native":
        worker._source_context_failed = True
    elif failure_marker == "touched":
        worker._chip_run_touched_identities[1] = {native.buffer.identity}
    else:
        worker._abandoned_run_handles.append(abandoned)
    try:
        worker._close_chip_import_registry()
        assert worker._chip_import_registry is registry
        release.assert_not_called()
        with pytest.raises(RuntimeError, match="unconfirmed|in-flight"):
            worker._release_host_buffer(native.buffer.identity)
        release.assert_not_called()
    finally:
        if abandoned in worker._abandoned_run_handles:
            worker._abandoned_run_handles.remove(abandoned)


def test_chip_import_cleanup_requires_confirmed_native_lane_close():
    registry, chip = MagicMock(), MagicMock()
    chip._close_chip_run_lane.side_effect = RuntimeError("unproven native run")
    with pytest.raises(RuntimeError, match="unproven"):
        worker_module._teardown_chip_process_resources(registry, chip, MagicMock(), MagicMock())
    registry.close.assert_not_called()


def test_worker_close_unmaps_remaining_local_views(host_worker):
    buffer = host_worker.create_buffer(128)
    native = buffer.tensor((16,), DataType.FLOAT32)
    mapped = host_worker.map_view(native, target=1)
    resource = host_worker._host_buffers[buffer.identity]
    resource.validate(mapped, 1)
    host_worker.close()
    assert buffer.closed and not resource._mapping_targets
    assert buffer.identity not in host_worker._host_buffers
