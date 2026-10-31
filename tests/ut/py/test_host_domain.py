# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import ctypes
import os
import struct
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from simpler import host_domain_backend
from simpler import worker as worker_module
from simpler.buffer import AccessMode, AddressSpace, ImportContext, ImportRegistry
from simpler.comm_endpoints import DEVICE_AICPU, AdapterProfile
from simpler.global_comm_domain import GLOBAL_DOMAIN_PROFILE_A3_FABRIC, GLOBAL_DOMAIN_PROFILE_IDS
from simpler.host_domain import HostDomainResource
from simpler.host_domain_backend import A3HostBacking, host_attachment_profile
from simpler.task_interface import (
    ArgDirection,
    CommBufferSpec,
    DataType,
    HostWindowSpec,
    TaskArgs,
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
    chip.comm_init.return_value = 9
    chip._execution_environment = (os.getpid(), 1, 1, 0, False)
    worker._lifecycle = worker_module._Lifecycle.READY

    def allocate_domain(_comm, _allocation, ranks, rank, size, committed):
        assert (ranks, rank, size) == ([0], 0, 64)
        ctypes.c_uint64.from_address(committed).value = 1
        return 1, 2, 64

    chip.comm_alloc_domain_windows.side_effect = allocate_domain
    chip.comm_attach_domain_host.return_value = 8192
    worker._chip_worker = chip
    monkeypatch.setattr(worker, "_device_control_admission", lambda _api: nullcontext())
    monkeypatch.setattr(worker, "_comm_plan_rootinfo_path", lambda: "/test/direct-host")
    return worker


@pytest.fixture
def direct_host(direct_worker):
    buffer = direct_worker.create_buffer(128, host_window=HostWindowSpec(128))
    try:
        yield direct_worker, buffer, buffer.tensor((16,), DataType.FLOAT32)
    finally:
        direct_worker._source_context_failed = False
        direct_worker._chip_run_touched_identities.clear()
        if buffer.identity in direct_worker._direct_host_domains:
            direct_worker.release_buffer(buffer)


def request(**changes):
    result = dict(
        name="host",
        members=[1, 0],
        window_size=4096,
        buffers=[CommBufferSpec("output", "float32", 16, 64)],
        host_window=HostWindowSpec(128),
        retain_after_run=True,
    )
    result.update(changes)
    return result


def allocate(worker, **changes):
    saved = []
    worker.run(lambda orch, _args, _cfg: saved.append(orch.get_or_allocate_domain(**request(**changes))))
    return saved[0]


def test_host_domain_preserves_native_identity_and_target_views(host_worker):
    domain = allocate(host_worker)
    source = domain.host().tensor((16,), DataType.FLOAT32)
    views = [domain.map_view(source, domain_rank=rank) for rank in range(2)]
    assert source.mapping_ref == 0
    assert source.buffer.address_space == AddressSpace.HOST
    assert all(view.buffer == source.buffer for view in views)
    assert [view.mapping_ref for view in views] == [1, 2]
    for rank, view in enumerate(views):
        args = TaskArgs()
        args.add_tensor(view, transfer=TensorTransfer.NONE)
        with host_worker._child_prov_lock:
            host_worker._child_prov_check_dispatch_locked(args, 1 - rank, api="test")
            with pytest.raises(ValueError, match="target worker"):
                host_worker._child_prov_check_dispatch_locked(args, rank, api="test")
    native_args = TaskArgs()
    native_args.add_tensor(source)
    with host_worker._child_prov_lock, pytest.raises(ValueError, match="explicit mapping"):
        host_worker._child_prov_check_dispatch_locked(native_args, 0, api="test")
    domain.release()
    assert domain.freed
    with pytest.raises(RuntimeError, match="released"):
        domain.map_view(source, domain_rank=0)


def test_host_domain_cpu_copy_and_configuration_reuse(host_worker):
    saved = []
    data = struct.pack("<16f", *range(16))

    def fill(orch, _args, _cfg):
        domain = orch.get_or_allocate_domain(**request())
        orch.copy_to_host_domain(domain, data, offset=64)
        assert orch.copy_from_host_domain(domain, len(data), offset=64) == data
        assert orch.get_or_allocate_domain(**request()) is domain
        with pytest.raises(ValueError, match="configuration differs"):
            orch.get_or_allocate_domain(**request(host_window=HostWindowSpec(256)))
        with pytest.raises(ValueError, match="exceeds"):
            orch.copy_to_host_domain(domain, data, offset=68)
        saved.append(domain)

    host_worker.run(fill)
    assert not saved[0].freed
    saved[0].release()
    assert saved[0].freed


@pytest.mark.parametrize("writing", [False, True])
def test_host_copy_uses_existing_control_admission(host_worker, monkeypatch, writing):
    domain = allocate(host_worker)
    resource = domain._host_resource
    copied = MagicMock(wraps=resource.copy)
    monkeypatch.setattr(resource, "copy", copied)

    def reject(api):
        assert api == "Host domain copy"
        raise RuntimeError("control refused")

    with monkeypatch.context() as patch:
        patch.setattr(host_worker._orch, "_control_admission", reject)
        with pytest.raises(RuntimeError, match="control refused"):
            if writing:
                host_worker._orch.copy_to_host_domain(domain, b"modified")
            else:
                host_worker._orch.copy_from_host_domain(domain, 8)
    copied.assert_not_called()
    domain.release()


def test_host_domain_stale_tensor_cannot_rebind_to_same_name(host_worker):
    domain = allocate(host_worker)
    stale = domain.map_view(domain.host().tensor((16,), DataType.FLOAT32), domain_rank=0)
    domain.release()
    replacement = allocate(host_worker)
    assert replacement.host().identity != stale.buffer.identity
    args = TaskArgs()
    args.add_tensor(stale, transfer=TensorTransfer.NONE)
    with host_worker._child_prov_lock, pytest.raises(ValueError, match="not a live"):
        host_worker._child_prov_check_dispatch_locked(args, 1, api="test")
    replacement.release()


def test_host_domain_native_reference_requires_endpoint_grant(host_worker):
    domain = allocate(host_worker)
    tensor = domain.host().tensor((16,), DataType.FLOAT32)
    registry = ImportRegistry(ImportContext(DEVICE_AICPU))
    with pytest.raises(ValueError, match="no granted attachment"):
        registry.resolve_tensor(tensor)
    domain.release()


def test_shared_host_backing_keeps_logical_bounds_and_endpoint_profile(host_worker):
    domain = allocate(host_worker)
    resource = domain._host_resource
    assert domain.host().nbytes == 128
    assert resource.export.mapping_size % os.sysconf("SC_PAGE_SIZE") == 0
    assert resource.export.mapping_size >= domain.host().nbytes
    assert host_attachment_profile(resource.export, "a2a3sim") == AdapterProfile.HOST_SHM_MAP
    with pytest.raises(ValueError, match="does not match"):
        host_attachment_profile(resource.export, "a2a3")
    with pytest.raises(ValueError, match="exceeds"):
        resource.copy(128, b"x", writing=True)
    domain.release()


def test_fabric_backing_preserves_logical_extent_and_authorizes_actual_pids(monkeypatch):
    calls = []

    class Native:
        base = 4096

        def __init__(self, runtime_path):
            assert runtime_path == "/test/host_runtime.so"

        def require_mapping_support(self):
            pass

        def allocate(self, device, numa, size):
            calls.append((device, numa, size))
            return 2097152, bytes(128)

        def authorize(self, pids):
            calls.append(tuple(pids))

        def release(self):
            calls.append("release")

    monkeypatch.setattr(host_domain_backend, "_HostDomainMemory", Native)
    backing = A3HostBacking(host_runtime_path="/test/host_runtime.so")
    base, export = backing.allocate(HostWindowSpec(128, numa_id=1), None, 2, 2)
    assert base == 4096
    assert export.mapping_size == 2097152
    assert export.profile_id == GLOBAL_DOMAIN_PROFILE_IDS[GLOBAL_DOMAIN_PROFILE_A3_FABRIC]
    assert host_attachment_profile(export, "a2a3") == AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT
    with pytest.raises(ValueError, match="does not match"):
        host_attachment_profile(export, "a2a3sim")
    backing.authorize([31, 42])
    backing.release()
    assert calls == [(2, 1, 128), (31, 42), "release"]


@pytest.mark.parametrize("kwargs", [dict(nbytes=0), dict(nbytes=63), dict(nbytes=True), dict(nbytes=64, numa_id=-1)])
def test_host_window_spec_rejects_invalid_allocation(kwargs):
    with pytest.raises((ValueError, TypeError)):
        HostWindowSpec(**kwargs)


def test_owner_cleanup_retries_without_replaying_chip_release(host_worker, monkeypatch):
    domain = allocate(host_worker)
    resource = domain._host_resource
    original = resource.backend.release
    attempts = []

    def fail_once():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("injected Host owner free failure")
        original()

    monkeypatch.setattr(resource.backend, "release", fail_once)
    with pytest.raises(RuntimeError, match="injected Host owner"):
        domain.release()
    assert resource.views_released
    assert host_worker._domain_free_results[domain.allocation_id] is None

    def no_replay(**_kwargs):
        raise AssertionError("confirmed chip collective must not replay")

    monkeypatch.setattr(host_worker, "_dispatch_control_domain", no_replay)
    host_worker.close()
    assert len(attempts) == 2
    assert domain.freed and host_worker._cleanup_journal.empty


def test_failed_creation_without_any_import_releases_owner(host_worker, monkeypatch):
    captured = []
    original = HostDomainResource.allocate

    def capture(resource):
        original(resource)
        captured.append(resource)

    monkeypatch.setattr(HostDomainResource, "allocate", capture)

    def refuse(**_kwargs):
        raise RuntimeError("all children refused before allocation")

    monkeypatch.setattr(host_worker, "_dispatch_control_domain", refuse)
    with pytest.raises(RuntimeError, match="all children refused"):
        allocate(host_worker)
    assert captured[0].views_released
    host_worker.close()
    assert captured[0].closed and host_worker._cleanup_journal.empty


def test_direct_host_buffer_uses_existing_domain_and_rejects_live_release(direct_worker):
    worker = direct_worker
    chip = worker._chip_worker
    buffer = worker._create_direct_host_buffer(128, HostWindowSpec(128))
    resource = worker._direct_host_domains[buffer.identity][0]
    native = buffer.tensor((32,), DataType.FLOAT32)
    mapped = resource.map_view(native, 0)
    assert native.mapping_ref == 0
    assert mapped.buffer.identity == native.buffer.identity
    assert worker._chip_import_registry is not None
    assert worker._chip_import_registry.resolve_tensor(mapped).base == 8192
    args = TaskArgs()
    args.add_tensor(native, transfer=TensorTransfer.H2D)
    args.add_tensor(mapped, transfer=TensorTransfer.NONE)
    worker._child_prov_check_dispatch_locked(args, 0, api="test")
    target = SimpleNamespace(sig_count=2, sig=lambda _index: ArgDirection.IN)
    resolved = _materialize_chip_task_args(args, worker._chip_import_registry, target)
    assert resolved.tensor(0).data == buffer.base
    assert resolved.tensor(0).address_space == AddressSpace.HOST
    assert resolved.tensor(1).data == 8192
    assert resolved.tensor(1).address_space == AddressSpace.DEVICE
    with pytest.raises(ValueError, match="target worker"):
        worker._child_prov_check_dispatch_locked(args, 1, api="test")
    worker._chip_run_touched_identities[1] = {buffer.identity}
    with pytest.raises(RuntimeError, match="in-flight"):
        worker._release_direct_host_resource(buffer.identity)
    assert not resource.closed
    worker._chip_run_touched_identities.clear()
    worker._release_direct_host_resource(buffer.identity)
    chip.comm_release_domain_windows.assert_called_once()
    assert buffer.closed
    assert not worker._direct_host_domains
    assert worker._cleanup_journal.empty
    with pytest.raises(ValueError, match="not a live allocation"):
        worker._child_prov_check_dispatch_locked(args, 0, api="test")
    with pytest.raises(ValueError, match="no granted attachment"):
        _materialize_chip_task_args(args, worker._chip_import_registry, target)


def test_direct_host_buffer_size_mismatch_has_no_side_effects():
    worker = Worker(level=2, platform="a2a3sim", runtime="tensormap_and_ringbuffer", device_id=0)
    with pytest.raises(ValueError, match="requested buffer size"):
        worker._create_direct_host_buffer(128, HostWindowSpec(64))
    assert not worker._direct_host_domains


def test_mapping_racing_release_cannot_retain_dispatch_permission(host_worker, monkeypatch):
    domain = allocate(host_worker)
    source = domain.host().tensor((16,), DataType.FLOAT32)
    entered, resume = threading.Event(), threading.Event()
    original = domain._host_resource.map_view
    failures = []

    def delayed(tensor, worker, **kwargs):
        entered.set()
        assert resume.wait(5)
        return original(tensor, worker, **kwargs)

    def mapping():
        try:
            domain.map_view(source, domain_rank=0)
        except ValueError as exc:
            failures.append(str(exc))

    monkeypatch.setattr(domain._host_resource, "map_view", delayed)
    thread = threading.Thread(target=mapping)
    thread.start()
    try:
        assert entered.wait(5)
        domain.release()
    finally:
        resume.set()
        thread.join(5)
    assert not thread.is_alive()
    assert failures == ["Host mapping requires the live domain backing"]


def test_host_validation_reuses_registered_descriptor(host_worker, monkeypatch):
    domain = allocate(host_worker)
    source = domain.host().tensor((16,), DataType.FLOAT32)
    resource = domain._host_resource

    def no_rebuild(_buffer):
        raise AssertionError("dispatch must reuse the registered descriptor")

    with monkeypatch.context() as patch:
        patch.setattr(type(resource.buffer), "to_descriptor", no_rebuild)
        mapped = resource.map_view(source, resource.workers[0])
        resource.validate(mapped, resource.workers[0])
    resource.buffer.nbytes = 64
    changed = resource.buffer.tensor((16,), DataType.FLOAT32)
    with pytest.raises(ValueError, match="differs from its source registration"):
        resource.validate(changed, resource.workers[0], native=True)
    domain.release()


def test_host_buffer_close_revokes_views_but_retains_owner_until_domain_release(host_worker):
    domain = allocate(host_worker)
    resource = domain._host_resource
    buffer = domain.host()
    native = buffer.tensor((16,), DataType.FLOAT32)
    buffer.close()
    assert buffer.closed and not resource.closed
    assert resource.backend.shared is not None and not resource.backend.shared.closed
    with pytest.raises(RuntimeError, match="not live"):
        resource.request()
    with pytest.raises(ValueError, match="live"):
        resource.map_view(native, 0)
    domain.release()
    assert resource.closed and resource.backend.shared.closed


def test_direct_release_is_serialized_and_removes_cleanup_debt(direct_worker, monkeypatch):
    from simpler.buffer import buffer_reference  # noqa: PLC0415

    worker = direct_worker
    chip = worker._chip_worker
    buffer = worker._create_direct_host_buffer(128, HostWindowSpec(128))
    with pytest.raises(ValueError, match="original owning"):
        worker.release_buffer(buffer_reference(buffer.to_descriptor()))
    assert not buffer.closed
    gate, lock, errors = threading.Barrier(2), threading.Lock(), []

    def synchronized(_worker):
        gate.wait(timeout=5)
        return lock

    monkeypatch.setattr(worker, "_child_prov_worker_lock", synchronized)

    def release():
        try:
            worker._release_direct_host_resource(buffer.identity)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=release) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads)
    assert not errors and worker._cleanup_journal.empty
    chip.comm_release_domain_windows.assert_called_once()


def test_map_view_issues_only_requested_record(direct_host):
    worker, buffer, native = direct_host
    mapped = worker.map_view(native, access=AccessMode.READ)
    registry = worker._chip_import_registry
    assert registry.resolve_tensor(mapped).base == 8192
    with pytest.raises(ValueError, match="no granted attachment"):
        registry.resolve_tensor(native._with_mapping_ref(mapped.mapping_ref + 1))
    args = TaskArgs()
    args.add_tensor(mapped, transfer=TensorTransfer.NONE)
    target = SimpleNamespace(sig_count=1, sig=lambda _index: ArgDirection.OUT)
    with pytest.raises(ValueError, match="mapping grant"):
        _materialize_chip_task_args(args, registry, target)


def test_unmap_preserves_native_and_remap_never_revives_old_reference(direct_host):
    worker, buffer, native = direct_host
    first = worker.map_view(native, access=AccessMode.READ)
    worker.unmap_view(first)
    assert not buffer.closed and worker._chip_import_registry.resolve_tensor(native).base == buffer.base
    with pytest.raises(ValueError, match="not live"):
        worker._chip_import_registry.resolve_tensor(first)
    second = worker.map_view(native, access=AccessMode.READ)
    assert second.mapping_ref != first.mapping_ref
    with pytest.raises(ValueError, match="not live"):
        worker._chip_import_registry.resolve_tensor(first)
    worker.unmap_view(first)
    assert worker._chip_import_registry.resolve_tensor(second).base == 8192
    worker._chip_worker.comm_detach_domain_host.assert_called_once()


def test_unmap_shared_import_only_detaches_last_record(direct_host):
    worker, buffer, native = direct_host
    read = worker.map_view(native, access=AccessMode.READ)
    write = worker.map_view(native, access=AccessMode.WRITE)
    worker.unmap_view(read)
    worker._chip_worker.comm_detach_domain_host.assert_not_called()
    assert worker._chip_import_registry.resolve_tensor(write).base == 8192
    worker.unmap_view(write)
    worker._chip_worker.comm_detach_domain_host.assert_called_once()


def test_unmap_failure_revokes_and_retains_cleanup_until_retry(direct_host):
    worker, buffer, native = direct_host
    mapped = worker.map_view(native, access=AccessMode.READ)
    worker._chip_worker.comm_detach_domain_host.side_effect = [RuntimeError("injected detach failure"), None]
    with pytest.raises(RuntimeError, match="injected detach"):
        worker.unmap_view(mapped)
    with pytest.raises(ValueError, match="not live"):
        worker._chip_import_registry.resolve_tensor(mapped)
    with pytest.raises(RuntimeError, match="cleanup is unconfirmed"):
        worker.map_view(native)
    worker.unmap_view(mapped)
    worker._chip_worker.comm_detach_domain_host.side_effect = None
    remapped = worker.map_view(native)
    assert remapped.mapping_ref != mapped.mapping_ref


@pytest.mark.parametrize("unmap", [False, True])
def test_host_view_lost_reply_keeps_revoked_route_and_blocks_reuse(direct_host, monkeypatch, unmap):
    worker, buffer, native = direct_host
    resource = worker._direct_host_domains[buffer.identity][0]
    original = resource._view_control
    first = worker.map_view(native)

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
    assert worker._chip_import_registry.resolve_tensor(mapped).base == 8192
    worker._chip_run_touched_identities.clear()
    worker.unmap_view(mapped)


def test_mapping_reference_exhaustion_does_not_reuse_retired_ids(direct_host):
    worker, buffer, native = direct_host
    for reference in range(1, 256):
        mapped = worker.map_view(native)
        assert mapped.mapping_ref == reference
        worker.unmap_view(mapped)
    with pytest.raises(RuntimeError, match="references exhausted"):
        worker.map_view(native)


def test_domain_unmap_then_remap_uses_existing_control_path(host_worker):
    domain = allocate(host_worker)
    native = domain.host().tensor((16,), DataType.FLOAT32)
    first = domain.map_view(native, domain_rank=0, access=AccessMode.READ)
    domain.unmap_view(first)
    second = domain.map_view(native, domain_rank=0, access=AccessMode.READ)
    assert second.mapping_ref != first.mapping_ref
    with pytest.raises(ValueError, match="live explicit mapping"):
        domain._host_resource.validate(first, domain.workers[0])
    domain._host_resource.validate(second, domain.workers[0])
    domain.release()


@pytest.mark.parametrize("owner_closed", [False, True])
def test_mapped_record_rejects_closed_owner_or_unconfirmed_completion(direct_host, owner_closed):
    worker, buffer, native = direct_host
    mapped = worker.map_view(buffer.tensor((16,), DataType.FLOAT32))
    if owner_closed:
        worker._direct_host_domains[buffer.identity][0].closed = True
    else:
        worker._source_context_failed = True
    with pytest.raises(ValueError, match="not live"):
        worker._chip_import_registry.resolve_tensor(mapped)


def test_host_allocation_rejects_borrowed_context_before_side_effects(direct_worker):
    worker = direct_worker
    worker._chip_worker._execution_environment = (os.getpid(), 1, 1, 0, True)
    with pytest.raises(ValueError, match="owned execution context"):
        worker._create_direct_host_buffer(128, HostWindowSpec(128))
    worker._chip_worker.comm_init.assert_not_called()
    assert not worker._direct_host_domains
