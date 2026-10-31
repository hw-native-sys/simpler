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
import shutil
import struct
import subprocess
import threading
from contextlib import nullcontext
from dataclasses import replace
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest
from simpler import host_domain_backend
from simpler import worker as worker_module
from simpler.buffer import (
    AccessMode,
    AddressSpace,
    BackendKind,
    Buffer,
    CanonicalIdentity,
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
from simpler.global_comm_domain import GLOBAL_DOMAIN_PROFILE_A3_FABRIC, GLOBAL_DOMAIN_PROFILE_IDS
from simpler.host_domain import HostDomainResource
from simpler.host_domain_backend import A3HostBacking, host_attachment_profile
from simpler.task_interface import (
    ArgDirection,
    CommBufferSpec,
    DataType,
    HostWindowSpec,
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
        if getattr(worker, "_expected_domain_debt", False):
            _close_with_domain_debt(worker)
        else:
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


@pytest.mark.parametrize("startup_abort", [False, True])
def test_unconfirmed_host_owner_does_not_prevent_child_shutdown(monkeypatch, startup_abort):
    worker = Worker(level=3, num_sub_workers=0)
    events = []

    def retained():
        events.append("owner")
        raise RuntimeError("Host owner retained: unconfirmed views")

    worker._cleanup_journal.add("host-domain", "local", retained)
    monkeypatch.setattr(worker, "_broadcast_child_shutdown", lambda _shms: events.append("shutdown"))
    monkeypatch.setattr(worker, "_reclaim_child_groups", lambda _deadline: events.append("reap"))
    monkeypatch.setattr(worker, "_abort_hierarchical", lambda **_kwargs: events.append("abort"))
    with pytest.raises(RuntimeError, match="unconfirmed views"):
        worker._teardown_worker_tree(startup_abort=startup_abort)
    assert events[0] == "owner" and events[-1] == "owner"
    assert ("abort" if startup_abort else "reap") in events[1:-1]
    assert not worker._cleanup_journal.empty


def test_unrelated_pretransport_failure_still_fences_teardown(monkeypatch):
    worker = Worker(level=3, num_sub_workers=0)
    monkeypatch.setattr(worker, "_flush_pending_remote_frees", Mock(side_effect=RuntimeError("remote free owed")))
    shutdown = Mock()
    monkeypatch.setattr(worker, "_broadcast_child_shutdown", shutdown)
    with pytest.raises(RuntimeError, match="remote free owed"):
        worker._teardown_worker_tree(startup_abort=False)
    shutdown.assert_not_called()


def test_direct_host_release_failure_allows_finalize_but_retains_owner():
    worker = Worker(level=2, platform="a2a3sim", device_id=0)
    chip = Mock()
    chip.workspace_report.return_value = ("disabled", {})
    chip.comm_release_domain_windows.side_effect = RuntimeError("release failed")
    worker._chip_worker = chip
    resource = SimpleNamespace(closed=False, buffer=None, views_released=False, close=Mock())
    worker._direct_host_domains["identity"] = (resource, 9, 11, ctypes.c_uint64(1))
    worker._cleanup_journal.add("host-domain", "direct-11", lambda: worker._release_direct_host_resource("identity"))
    with pytest.raises(RuntimeError, match="native teardown did not confirm"):
        worker._teardown_worker_tree(startup_abort=False)
    chip.finalize.assert_called_once()
    chip.comm_release_domain_windows.assert_called_once()
    assert worker._chip_worker is None
    assert not resource.views_released
    resource.close.assert_not_called()
    assert "identity" in worker._direct_host_domains
    with pytest.raises(RuntimeError, match="native teardown did not confirm"):
        worker._teardown_worker_tree(startup_abort=False)
    chip.comm_release_domain_windows.assert_called_once()


def test_child_journal_survivor_is_not_treated_as_view_release(monkeypatch):
    worker = Worker(level=3, num_sub_workers=0)
    owner_release = Mock(side_effect=RuntimeError("Host views unknown"))
    worker._cleanup_journal.add("host-domain", "local", owner_release)

    def reap(_deadline):
        worker._cleanup_journal.add("child", "survivor", Mock(side_effect=RuntimeError("child still alive")))

    monkeypatch.setattr(worker, "_reclaim_child_groups", reap)
    with pytest.raises(RuntimeError, match="child still alive"):
        worker._teardown_worker_tree(startup_abort=False)
    assert not worker._chip_pids
    assert len(worker._cleanup_journal) == 2


def test_a3_registration_uses_runtime_identity_and_checks_authorization(monkeypatch):
    native = MagicMock()
    monkeypatch.setattr(host_domain_backend, "_HostDomainMemory", lambda _path: native)
    chip = MagicMock()
    chip.host_backing_process_identity.return_value = 12345
    owner = host_domain_backend.host_backing_type("a2a3")(host_runtime_path="/test/host_runtime.so")
    native.require_mapping_support.assert_called_once_with()
    owner.authorize([host_domain_backend.host_process_identity("a2a3", chip)])
    native.authorize.assert_called_once_with([12345])


def test_optional_host_identity_does_not_enable_unsupported_backing(monkeypatch):
    def unexpected_fabric_call():
        raise AssertionError("non-Fabric communicator requested a Fabric identity")

    chip = MagicMock()
    chip.host_backing_process_identity.side_effect = unexpected_fabric_call
    assert host_domain_backend.host_process_identity("a2a3sim", chip) == 0
    host_domain_backend.host_backing_type("a2a3sim")().authorize([0, 0])
    assert host_domain_backend.host_process_identity("a5", chip) == 0
    with pytest.raises(ValueError, match="not implemented"):
        host_domain_backend.host_backing_type("a5")


@pytest.mark.parametrize("fail_once", [False, True])
def test_parent_host_bootstrap_reads_pid_replies_before_authorizing_members(monkeypatch, fail_once):
    worker = Worker(level=3, platform="a2a3sim", device_ids=[12, 13], num_sub_workers=0)
    pids, requests, resource = [12345, 23456], [], MagicMock()
    path = "/shared/\u8def\u5f84/rootinfo"

    def control(chip_idx, name):
        nonlocal fail_once
        request = SharedMemory(name=name)
        try:
            buf = request.buf
            assert buf is not None
            rank, count, want_pid, reply_pid = worker_module._COMM_INIT_HEADER.unpack_from(buf)
            assert (rank, count, reply_pid) == (chip_idx, 2, 0)
            offset = worker_module._COMM_INIT_HEADER.size
            assert bytes(buf[offset : offset + len(path.encode()) + 1]) == path.encode() + b"\0"
            requests.append((rank, want_pid))
            if fail_once and want_pid and rank == 1:
                fail_once = False
                raise RuntimeError("PID reply failed")
            worker_module._COMM_INIT_HEADER.pack_into(buf, 0, rank, count, want_pid, pids[rank] if want_pid else 0)
        finally:
            request.close()

    monkeypatch.setattr(worker, "_comm_plan_rootinfo_path", lambda: path)
    worker._worker = MagicMock(control_comm_init=control)
    try:
        worker._ensure_comm_base()
        assert worker._comm_base_ready and not worker._host_identity_ready
        assert worker._host_process_ids == {0: 0, 1: 0}
        if fail_once:
            with pytest.raises(RuntimeError, match="PID reply failed"):
                worker._ensure_comm_base(require_host_identity=True)
            assert not worker._host_identity_ready and worker._host_process_ids == {0: 0, 1: 0}
        worker._ensure_comm_base(require_host_identity=True)
        assert worker._host_identity_ready and worker._host_process_ids == dict(enumerate(pids))
        worker._allocate_host_domain_backing(resource, (1, 0))
        assert resource.mock_calls == [call.allocate(), call.backend.authorize([23456, 12345])]
        before = list(requests)
        worker._ensure_comm_base(require_host_identity=True)
        assert requests == before and sorted(requests[:2]) == [(0, 0), (1, 0)]
    finally:
        worker._worker = None
        worker.close()


def test_host_mapping_runtime_support_is_checked_before_allocation(monkeypatch):
    native = MagicMock()
    native.require_mapping_support.side_effect = RuntimeError("missing Host detach entry")
    monkeypatch.setattr(host_domain_backend, "_HostDomainMemory", lambda _path: native)
    with pytest.raises(RuntimeError, match="missing Host detach"):
        host_domain_backend.A3HostBacking(host_runtime_path="/test/host_runtime.so")
    native.allocate.assert_not_called()


_REFS = {AccessMode.READWRITE: 1, AccessMode.READ: 2, AccessMode.WRITE: 3}


@pytest.fixture
def mapping(request):
    identity = CanonicalIdentity(mint_owner_instance_id(), 1, 1)
    buffer = Buffer(
        identity=identity,
        address_space=AddressSpace.HOST,
        access=AccessMode.READWRITE,
        backend_kind=BackendKind.REMOTE_SIDECAR,
        nbytes=64,
        base=4096,
    )
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
    for access in getattr(request, "param", (AccessMode.READWRITE, AccessMode.READ, AccessMode.WRITE)):
        registry.register_attachment(
            buffer.to_descriptor(),
            8192,
            mapping_ref=_REFS[access],
            access=access,
            profile=AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT,
            live=lambda: True,
        )
    return buffer, registry


@pytest.mark.parametrize(
    "access,tag,direction,allowed",
    [
        (AccessMode.READ, TensorArgType.INPUT, ArgDirection.IN, True),
        (AccessMode.READ, TensorArgType.OUTPUT, ArgDirection.OUT, False),
        (AccessMode.READ, TensorArgType.INOUT, ArgDirection.INOUT, False),
        (AccessMode.WRITE, TensorArgType.OUTPUT, ArgDirection.OUT, True),
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


def test_callable_direction_cannot_be_weakened_by_input_tag(mapping):
    buffer, registry = mapping
    view = buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(_REFS[AccessMode.READ])
    args = TaskArgs()
    args.add_tensor(view, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
    target = SimpleNamespace(sig_count=1, sig=lambda index: ArgDirection.OUT)
    with pytest.raises(ValueError, match="mapping grant"):
        _materialize_chip_task_args(args, registry, target)


@pytest.mark.parametrize(
    "mapping,write_granted",
    [((AccessMode.READ,), False), ((AccessMode.READ, AccessMode.WRITE), True)],
    indirect=["mapping"],
    ids=["read-only", "read-and-write"],
)
def test_native_selector_requires_an_existing_target_grant(mapping, write_granted):
    buffer, registry = mapping
    native = buffer.tensor((16,), DataType.FLOAT32)
    read = native._with_mapping_ref(_REFS[AccessMode.READ])
    write = native._with_mapping_ref(_REFS[AccessMode.WRITE])
    args = TaskArgs()
    args.add_tensor(write, TensorArgType.OUTPUT_EXISTING, transfer=TensorTransfer.NONE)
    target = SimpleNamespace(sig_count=1, sig=lambda _index: ArgDirection.OUT)
    if write_granted:
        assert _materialize_chip_task_args(args, registry, target).tensor(0).data == 8192
    else:
        with pytest.raises(ValueError, match="no granted attachment"):
            _materialize_chip_task_args(args, registry, target)
    assert read.mapping_ref == _REFS[AccessMode.READ] and read.buffer == write.buffer


def test_mapped_tensor_cannot_replace_its_mapping_reference(mapping):
    buffer, _ = mapping
    read = buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(_REFS[AccessMode.READ])
    assert read._with_mapping_ref(read.mapping_ref) == read
    with pytest.raises(ValueError):
        read._with_mapping_ref(_REFS[AccessMode.WRITE])


def test_mapping_rejects_grant_change_and_unregistered_reference(mapping):
    buffer, registry = mapping
    with pytest.raises(ValueError, match="conflicts"):
        registry.register_attachment(
            buffer.to_descriptor(),
            8192,
            mapping_ref=_REFS[AccessMode.READ],
            access=AccessMode.READWRITE,
            profile=AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT,
            live=lambda: True,
        )
    view = buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(4)
    with pytest.raises(ValueError, match="no granted attachment"):
        registry.resolve_tensor(view)
    registry.unregister(buffer.identity)
    with pytest.raises(ValueError, match="no granted attachment"):
        registry.resolve_tensor(buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(_REFS[AccessMode.READ]))


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
            profile=AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT,
            live=lambda: True,
        )


def test_host_profile_does_not_implicitly_materialize_a_sidecar(mapping):
    buffer, _ = mapping
    fresh = ImportRegistry(mapping[1]._context)
    with pytest.raises(ValueError, match="UNSUPPORTED_BACKEND_KIND"):
        fresh.materialize(buffer.to_descriptor())
    view = buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(_REFS[AccessMode.READ])
    with pytest.raises(ValueError, match="no granted attachment"):
        fresh.resolve_tensor(view)
    fresh.register_attachment(
        buffer.to_descriptor(),
        8192,
        mapping_ref=view.mapping_ref,
        access=AccessMode.READ,
        profile=AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT,
        live=lambda: True,
    )
    assert fresh.resolve_tensor(view).base == 8192


def test_close_attachment_revokes_only_its_record_and_cannot_be_registered_again(mapping):
    buffer, registry = mapping
    native = buffer.tensor((16,), DataType.FLOAT32)
    registry.register_attachment(
        buffer.to_descriptor(),
        4096,
        mapping_ref=0,
        profile=AdapterProfile.HOST_VMM_MAP,
        live=lambda: True,
    )
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
            profile=AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT,
            live=lambda: True,
        )
    registry.close_attachment(buffer.identity, write.mapping_ref)
    registry.close_attachment(buffer.identity, _REFS[AccessMode.READWRITE])
    assert not registry.has_active_attachments(buffer.identity)
    assert registry.resolve_tensor(native).base == 4096
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
        profile=AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT,
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
            profile=AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT,
            live=lambda: True,
        )


@pytest.mark.parametrize(
    "profile,deployment,platform,supported",
    [
        (AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT, DEVICE_AICPU, "a2a3", True),
        (AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT, DEVICE_AICORE, "a2a3", True),
        (AdapterProfile.HOST_SHM_MAP, DEVICE_AICPU, "a2a3sim", True),
        (AdapterProfile.HOST_SHM_MAP, HOST_CPU, "a2a3sim", True),
        (AdapterProfile.HOST_VMM_MAP, HOST_CPU, "a2a3", True),
        (AdapterProfile.HOST_VMM_MAP, DEVICE_AICPU, "a2a3", False),
        (AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT, HOST_CPU, "a2a3", False),
        (AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT, DEVICE_AICPU, "a5", False),
        (AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT, DEVICE_AICPU, "a2a3sim", False),
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
        BackendKind.REMOTE_SIDECAR,
        DEVICE_AICPU,
        same_node=True,
        platform="a2a3",
        runtime="tensormap_and_ringbuffer",
        confirmed_profile=AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT,
        source_address_space=AddressSpace.HOST,
    )
    candidate = _AdapterCandidate(AdapterKind.DIRECT_MAP, AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT)
    decision = DefaultRegionAccessService().evaluate_buffer_access(replace(query, **query_changes), candidate)
    assert not decision.supported


def test_host_attachment_cleanup_faults(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[3]
    source = (root / "src/a2a3/platform/onboard/host/comm_hccl.cpp").read_text()

    def section(start, end):
        return source[source.index(start) : source.index(end, source.index(start))]

    fixture = (Path(__file__).parent / "support/host_comm_cleanup.cpp").read_text()
    declarations, cases = fixture.split("// PRODUCTION_BODIES")
    bodies = [
        section("struct DomainAllocation {", "static_assert(sizeof(CommGlobalDomainDescriptor)"),
        section("static aclError release_domain_host(", "static aclError release_base_windows("),
        section(
            "DomainAllocation::~DomainAllocation()",
            "// ============================================================================",
        ),
        section('extern "C" int comm_attach_domain_host(', 'extern "C" int\ncomm_release_domain_windows'),
        section('extern "C" int\ncomm_release_domain_windows', 'extern "C" int comm_global_domain_prepare'),
    ]
    # Only SDK/runtime boundaries are stubbed; the selected production bodies are compiled verbatim.
    unit = tmp_path / "host_comm_cleanup.cpp"
    unit.write_text(
        declarations
        + bodies[0]
        + "\n"
        + cases.split("// TEST_CASES")[0]
        + "\n".join(bodies[1:])
        + cases.split("// TEST_CASES")[1]
    )
    executable = tmp_path / "host_comm_cleanup"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I" + str(root / "src/common"),
            "-I" + str(root / "src/common/platform/include"),
            str(unit),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)


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
                profile=AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT,
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

    def test_mapping_requires_confirmed_live_target(self, mapping_context):
        backing = create_host_shared_buffer(64, mint_owner_instance_id(), 2)
        registry = ImportRegistry(mapping_context)
        live = [True]
        try:
            mapped = backing.tensor((16,), DataType.FLOAT32)._with_mapping_ref(7)
            with pytest.raises(ValueError, match="no granted attachment"):
                registry.resolve_tensor(mapped)
            registry.register_attachment(
                backing.to_descriptor(),
                0x8000,
                mapping_ref=7,
                profile=AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT,
                live=lambda: live[0],
            )
            assert registry.resolve_tensor(mapped).base == 0x8000
            forged = replace(backing, nbytes=128).tensor((16,), DataType.FLOAT32)._with_mapping_ref(7)
            with pytest.raises(ValueError, match="descriptor differs"):
                registry.resolve_tensor(forged)
            live[0] = False
            with pytest.raises(ValueError, match="not live"):
                registry.resolve_tensor(mapped)
            registry.unregister(backing.identity)
            with pytest.raises(ValueError, match="no granted attachment"):
                registry.resolve_tensor(mapped)
        finally:
            registry.close()
            backing.close()


def _close_with_domain_debt(worker):
    worker._expected_domain_debt = True
    with pytest.raises(RuntimeError, match="unconfirmed"):
        worker.close()
    assert not worker._cleanup_journal.empty and worker._worker is None
    assert not worker._chip_pids and not worker._next_level_pids
    assert not worker._chip_shms and not worker._next_level_shms


@pytest.mark.parametrize("completed", [False, True])
def test_host_owner_survives_unconfirmed_domain_release(host_worker, monkeypatch, completed):
    worker, members = host_worker, [1, 0]
    chip_pids = set(worker._chip_pids)
    assert chip_pids
    saved = []
    worker.run(lambda orch, _args, _cfg: saved.append(orch.allocate_domain(**request(members=members))))
    domain = saved[0]
    resource = domain._host_resource
    assert resource is not None
    args = TaskArgs()
    args.add_tensor(
        domain.map_view(domain.host().tensor((16,), DataType.FLOAT32), domain_rank=0), transfer=TensorTransfer.NONE
    )
    original = worker._release_domain_claimed
    failure, attempts = RuntimeError("injected Host domain release failure"), []

    def release(handle):
        attempts.append(handle)
        if completed:
            original(handle)
        raise failure

    try:
        with monkeypatch.context() as patch:
            patch.setattr(worker, "_release_domain_claimed", release)
            for free in (domain.release, lambda: worker._release_domain_now(domain)):
                with pytest.raises(RuntimeError, match="injected Host domain release failure"):
                    free()
            assert domain.released and not domain.freed
            with worker._child_prov_lock, pytest.raises(ValueError, match="not a live allocation"):
                worker._child_prov_check_dispatch_locked(args, members[0], api="test")
            assert worker._domain_free_results[domain.allocation_id] is failure
            assert resource.views_released is completed and not resource.backend.shared.closed
            _close_with_domain_debt(worker)
            assert chip_pids <= worker.teardown_reports().keys()
            assert resource.backend.shared.closed is completed
            assert attempts == [domain]
    finally:
        try:
            if worker._worker is not None:
                _close_with_domain_debt(worker)
            assert chip_pids <= worker.teardown_reports().keys()
        finally:
            # The test's SHM is disposable only after all chip processes have been reaped.
            if chip_pids <= worker.teardown_reports().keys():
                resource.backend.release()
