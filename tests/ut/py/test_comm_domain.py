# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import copy
import struct
import threading
from types import SimpleNamespace
from typing import Any

import pytest
import simpler.worker as worker_mod
from simpler.orchestrator import _callback_run
from simpler.task_interface import CommBufferSpec, CommDomainHandle, DataType, GlobalCommDomainHandle, TaskArgs
from simpler.worker import Worker


@pytest.fixture(params=[3, 4])
def domain_worker(request):
    child = Worker(
        level=3,
        platform="a2a3sim",
        runtime="tensormap_and_ringbuffer",
        device_ids=[0, 1],
        num_sub_workers=0,
        comm_profile="sim",
        global_device_ranks=[10, 20],
    )
    worker = child
    members = [1, 0]
    try:
        if request.param == 4:
            worker = Worker(level=4, num_sub_workers=0)
            node = worker.add_worker(child)
            members = [(node, 1), (node, 0)]
        worker.init()
        yield worker, members
    finally:
        worker.close()


def _request(members, **changes) -> dict[str, Any]:
    config = dict(
        name="tp",
        members=members,
        window_size=4096,
        buffers=[CommBufferSpec("payload", "uint8", 32, 64)],
        retain_after_run=True,
    )
    config.update(changes)
    return config


def _check_domain_dispatch_locked(worker, args, target):
    worker._prepare_buffer_references(args, None, target)
    worker._child_prov_check_dispatch_locked(args, target, api="test")


@pytest.mark.parametrize("sizes", [(4,), (64, 16), (80, 64), (4096, 4096, 16, 16, 16)])
def test_unaligned_buffer_sizes_are_rejected_before_allocation(domain_worker, monkeypatch, sizes):
    worker, members = domain_worker
    specs = [CommBufferSpec(f"buffer_{i}", "uint8", size, size) for i, size in enumerate(sizes)]

    def unexpected_allocation(**_kwargs):
        pytest.fail("invalid buffer layout reached the allocation backend")

    monkeypatch.setattr(worker, "_allocate_domain", unexpected_allocation)
    monkeypatch.setattr(worker, "_allocate_global_domain", unexpected_allocation)

    def build(orch, _args, _cfg):
        for allocate in (orch.allocate_domain, orch.get_or_allocate_domain):
            with pytest.raises(ValueError, match="buffer_.*nbytes.*64"):
                allocate(**_request(members, window_size=16384, buffers=specs))
        assert worker.live_domains == {}
        assert tuple(spec.nbytes for spec in specs) == sizes

    worker.run(build)


def test_unaligned_window_size_is_rejected_before_allocation(domain_worker, monkeypatch):
    worker, members = domain_worker

    def unexpected_allocation(**_kwargs):
        pytest.fail("unaligned rank stride reached the allocation backend")

    monkeypatch.setattr(worker, "_allocate_domain", unexpected_allocation)
    monkeypatch.setattr(worker, "_allocate_global_domain", unexpected_allocation)

    def build(orch, _args, _cfg):
        for allocate in (orch.allocate_domain, orch.get_or_allocate_domain):
            with pytest.raises(ValueError, match="window_size.*64"):
                allocate(**_request(members, window_size=65))
        assert worker.live_domains == {}

    worker.run(build)


def test_caller_padded_buffers_keep_exact_offsets_and_logical_sizes(domain_worker):
    worker, members = domain_worker
    specs = [
        CommBufferSpec("signal", "int32", 1, 64),
        CommBufferSpec("payload", "uint8", 80, 128),
        CommBufferSpec("tail", "int32", 1, 64),
    ]

    def build(orch, _args, _cfg):
        with pytest.raises(ValueError, match="buffers exceed window_size"):
            orch.allocate_domain(**_request(members, window_size=192, buffers=specs))
        assert worker.live_domains == {}
        with orch.allocate_domain(**_request(members, window_size=256, buffers=specs)) as domain:
            for name, offset, nbytes in (("signal", 0, 64), ("payload", 64, 128), ("tail", 192, 64)):
                assert domain.buffer_range(name) == (offset, nbytes)
                if worker.level == 3:
                    for rank in range(len(members)):
                        context = domain.local_context(rank)
                        assert context.buffers[name].base == context.local_window_base + offset
                        assert context.buffers[name].base % 64 == 0
                        assert context.actual_window_size == 256
            orch.copy_to_domain(domain, 0, b"x" * 80, buffer="payload")
            assert orch.copy_from_domain(domain, 0, 80, buffer="payload") == b"x" * 80
            assert domain._config is not None
            assert [(spec.count, spec.nbytes) for spec in domain._config.specs()] == [(1, 64), (80, 128), (1, 64)]

    worker.run(build)


def test_reuse_snapshot_data_and_second_run_release(domain_worker):
    worker, members = domain_worker
    request = _request(members)
    saved = []

    def first(orch, _args, _cfg):
        handle = orch.get_or_allocate_domain(**request)
        assert type(handle) is CommDomainHandle is GlobalCommDomainHandle
        assert orch.get_or_allocate_domain(**request) is handle
        assert orch.get_domain("tp") is handle
        assert orch.get_domain(handle.ref) is handle and not handle.borrowed
        payload = handle.member(0).buffers["payload"]
        assert payload.tensor([8], DataType.UINT8).buffer.identity == payload.identity
        assert handle.buffers[0].dtype == "uint8" and handle.buffers[0].count == 32
        assert handle.requested_window_size == 4096
        args = TaskArgs()
        args.add_tensor(payload.tensor([8], DataType.UINT8))
        target = members[0] if worker.level == 3 else members[0][0]
        with worker._child_prov_lock:
            _check_domain_dispatch_locked(worker, args, target)
            with pytest.raises(ValueError, match="wrong worker|stale"):
                _check_domain_dispatch_locked(worker, args, 999)
        assert handle.member(0).local_worker_id == 1
        assert [handle.member(rank).global_device_rank for rank in range(2)] == [20, 10]
        orch.copy_to_domain(handle, 0, b"retained", buffer="payload", offset=3)
        saved.append((handle, orch))

    worker.run(first)
    handle, orch = saved[0]
    allocation_count = worker._next_alloc_id
    assert not handle.released and not handle.freed
    assert worker.live_domains == {"tp": handle}
    for name in ("tp", "missing"):
        with pytest.raises(RuntimeError, match="callback"):
            orch.get_or_allocate_domain(**dict(request, name=name))
    # Mutating a caller-owned spec cannot rewrite the captured configuration.
    request["buffers"][0].dtype = "int8"

    def second(orch, _args, _cfg):
        with pytest.raises(ValueError, match="buffers"):
            orch.get_or_allocate_domain(**request)
        same = orch.get_or_allocate_domain(**_request(members))
        assert same is handle and worker._next_alloc_id == allocation_count
        assert orch.copy_from_domain(same, 0, 8, buffer="payload", offset=3) == b"retained"
        orch.release_domain(same)
        assert same.released and not same.freed
        with pytest.raises(KeyError):
            orch.get_domain("tp")
        with pytest.raises(RuntimeError, match="released"):
            same.member(0)

    worker.run(second)
    assert handle.freed and worker.live_domains == {}
    handle.release()


def test_configuration_conflicts_have_no_device_effect(domain_worker, monkeypatch):
    worker, members = domain_worker
    request = _request(members)
    mutations = [dict(members=list(reversed(members))), dict(window_size=8192), dict(retain_after_run=False)]
    for field, value in (
        ("name", "other"),
        ("dtype", "int8"),
        ("count", 8),
        ("nbytes", 128),
        ("load_from_host", True),
        ("store_to_host", True),
    ):
        specs = copy.deepcopy(request["buffers"])
        setattr(specs[0], field, value)
        mutations.append(dict(buffers=specs))

    def build(orch, _args, _cfg):
        handle = orch.allocate_domain(**request)
        allocated = worker._next_alloc_id
        with pytest.raises(ValueError, match="already live"):
            orch.allocate_domain(**request)
        for mutation in mutations:
            with pytest.raises(ValueError, match="configuration differs"):
                orch.get_or_allocate_domain(**dict(request, **mutation))
        child = worker if worker.level == 3 else worker._next_level_workers[0]
        with monkeypatch.context() as patch:
            patch.setitem(child._config, "global_device_ranks", [30, 40])
            with pytest.raises(ValueError, match="topology"):
                orch.get_or_allocate_domain(**request)
        assert worker._next_alloc_id == allocated
        assert orch.get_domain("tp") is handle
        handle.release()

    worker.run(build)


def test_release_rebuild_does_not_remove_replacement(domain_worker):
    worker, members = domain_worker
    saved = []

    def build(orch, _args, _cfg):
        old = orch.allocate_domain(**_request(members))
        old.release()
        new = orch.allocate_domain(**_request(members))
        assert old.domain_id != new.domain_id
        assert orch.get_domain("tp") is new
        saved.extend((old, new))

    worker.run(build)
    old, new = saved
    assert old.freed and not new.freed
    assert worker.live_domains == {"tp": new}
    worker.run(lambda orch, _args, _cfg: orch.release_domain(orch.get_domain("tp")))
    assert new.freed


def test_context_exit_and_worker_close(domain_worker):
    worker, members = domain_worker
    saved = []

    def build(orch, _args, _cfg):
        handle = orch.allocate_domain(**_request(members))
        with pytest.raises(LookupError):
            with orch.get_or_allocate_domain(**_request(members)) as same:
                assert same is handle
                raise LookupError("body failure")
        assert handle.released and not handle.freed
        saved.append(handle)
        saved.append(orch.allocate_domain(**_request(members)))

    worker.run(build)
    assert saved[0].freed and not saved[1].freed
    worker.close()
    assert saved[1].released and saved[1].freed


def test_missing_lookup(domain_worker):
    worker, members = domain_worker

    def build(orch, _args, _cfg):
        with pytest.raises(KeyError):
            orch.get_domain("missing")
        orch.allocate_domain(**_request(members)).release()

    worker.run(build)


def test_post_allocation_failure_does_not_publish_an_incomplete_handle(monkeypatch):
    worker = Worker(level=3, platform="a2a3sim", runtime="tensormap_and_ringbuffer", device_ids=[0], num_sub_workers=0)
    try:
        worker.init()

        def fail_metadata(*_args):
            raise RuntimeError("metadata assembly failed")

        def build(orch, _args, _cfg):
            with monkeypatch.context() as patch:
                patch.setattr(worker_mod, "GlobalDomainMember", fail_metadata)
                with pytest.raises(RuntimeError, match="metadata assembly failed"):
                    orch.get_or_allocate_domain(**_request([0]))
            with pytest.raises(KeyError):
                orch.get_domain("tp")

        worker.run(build)
        assert worker.live_domains == {}
        worker.run(lambda orch, _args, _cfg: orch.get_or_allocate_domain(**_request([0])).release())
    finally:
        worker.close()


def test_failed_creation_stays_cleanup_owned_without_becoming_public(monkeypatch):
    worker = Worker(level=3, device_ids=[0], num_sub_workers=0)
    worker._worker = SimpleNamespace()
    worker._lifecycle = worker_mod._Lifecycle.READY
    resources = worker_mod._RunResources()
    worker._building_run_resources = resources
    released = []
    failed = []
    first = True

    def in_memory_shms(_count, action, **kwargs):
        owner = SimpleNamespace(create=lambda size: SimpleNamespace(buf=bytearray(size)))
        result = action(owner)
        if kwargs.get("after_success"):
            kwargs["after_success"]()
        return result

    def dispatch(**kwargs):
        nonlocal first
        if kwargs["op"] == "release":
            released.append(kwargs["allocation_id"])
            return
        request = kwargs["request_shms"][0].buf
        count = worker_mod._DOMAIN_REQ_HEADER.unpack_from(request)[4]
        reply = kwargs["reply_shms"][0].buf
        worker_mod._DOMAIN_REPLY_HEADER.pack_into(reply, 0, 1, 0xC0, 0x1000, count + 1 if first else count, 8192)
        if first:
            first = False
            failed.append(worker._live_domains["tp"])
        else:
            struct.pack_into("<Q", reply, worker_mod._DOMAIN_REPLY_HEADER.size, 0x1000)

    monkeypatch.setattr(worker_mod, "_run_with_owned_shared_memory", in_memory_shms)
    monkeypatch.setattr(worker, "_ensure_comm_base", lambda: None)
    monkeypatch.setattr(worker, "_dispatch_control_domain", dispatch)
    request = dict(
        name="tp", members=[0], window_size=4096, buffers=[CommBufferSpec("x", "uint8", 4, 64)], retain_after_run=True
    )
    try:
        with _callback_run(1, worker):
            with pytest.raises(RuntimeError, match="reply buffer_count"):
                worker._comm_domain(**request, reuse=False)
            old = failed[0]
            assert worker.live_domains == {}
            with pytest.raises(KeyError):
                worker._get_domain("tp")
            assert old.released and not old.freed
            assert old in resources.pending_release_domains
            new = worker._comm_domain(**request, reuse=True)
            assert new.requested_window_size == 4096 and new.mapping_size == 8192
            assert new is not old and new._config is not None
            assert new.member(0).global_device_rank == 0
            assert new.allocation_id != old.allocation_id
            worker._execute_pending_domain_releases(resources)
            assert old.freed and released == [old.allocation_id]
            assert not new.released and not new.freed
            assert worker._get_domain("tp") is new
    finally:
        # Fake native objects contain no device allocations or operating-system mappings.
        worker._worker = None
        worker._building_run_resources = None
        worker._live_domains.clear()
        worker.close()


def test_nonretained_and_failed_retained_runs_are_closed(domain_worker):
    worker, members = domain_worker
    saved = []
    worker.run(
        lambda orch, _args, _cfg: saved.append(orch.allocate_domain(**_request(members, retain_after_run=False)))
    )
    assert saved[0].released and saved[0].freed

    def failing(orch, _args, _cfg):
        saved.append(orch.get_or_allocate_domain(**_request(members)))
        raise LookupError("orchestration failed")

    with pytest.raises(LookupError, match="orchestration failed"):
        worker.run(failing)
    assert saved[1].released and saved[1].freed
    assert worker.live_domains == {}


def test_reuse_after_submission_has_no_device_control(domain_worker):
    from simpler.orchestrator import _admit_task_submission  # noqa: PLC0415

    worker, members = domain_worker

    def build(orch, _args, _cfg):
        domain = orch.allocate_domain(**_request(members))
        allocation_count = worker._next_alloc_id
        _admit_task_submission(worker)
        assert orch.get_or_allocate_domain(**_request(members)) is domain
        assert worker._next_alloc_id == allocation_count
        with pytest.raises(ValueError, match="configuration differs"):
            orch.get_or_allocate_domain(**_request(members, window_size=8192))
        if worker.level == 3:
            with pytest.raises(RuntimeError, match="direct device control"):
                orch.allocate_domain(**_request(members, name="new"))

    worker.run(build)


def test_legacy_entrypoints_share_reuse_and_release(domain_worker):
    worker, members = domain_worker

    def noop(_orch, _args, _cfg):
        pass

    task = worker.register(noop) if worker.level >= 4 else None

    def build(orch, _args, _cfg):
        request = _request(members)
        if worker.level == 3:
            with pytest.raises(RuntimeError, match="L4"):
                orch.allocate_global_domain(**request)
            request["workers"] = request.pop("members")
            domain = orch.allocate_domain(**request)
        else:
            assert task is not None
            orch.submit_next_level(task, TaskArgs(), worker=members[0][0])
            with pytest.raises(TypeError, match="L3"):
                orch.allocate_domain(**dict(request, workers=request["members"], members=None))
            domain = orch.allocate_global_domain(**request)
            orch.copy_to_global_domain(domain, 0, b"legacy", buffer="payload")
            assert orch.copy_from_domain(domain, 0, 6, buffer="payload") == b"legacy"
        assert orch.get_or_allocate_domain(**_request(members)) is domain
        with pytest.raises(ValueError, match="already live"):
            orch.allocate_domain(**_request(members))
        orch.release_domain(domain)

    worker.run(build)


@pytest.mark.parametrize("release_api", ["handle", "orchestrator"])
def test_retained_domain_release_after_run(domain_worker, release_api, monkeypatch):
    worker, members = domain_worker
    saved = []
    worker.run(lambda orch, _args, _cfg: saved.append(orch.allocate_domain(**_request(members))))
    domain = saved[0]
    release_calls = []
    backend_name = "_release_domain_claimed" if worker.level == 3 else "_release_global_domain_claimed"
    backend = getattr(worker, backend_name)

    def release_backend(handle):
        release_calls.append(handle)
        backend(handle)

    monkeypatch.setattr(worker, backend_name, release_backend)
    orch = worker._orch
    assert orch is not None
    release = domain.release if release_api == "handle" else lambda: orch.release_domain(domain)
    release()
    assert domain.released and domain.freed and worker.live_domains == {}
    domain.release()
    orch.release_domain(domain)
    orch.release_global_domain(domain)
    assert release_calls == [domain]


def test_retained_domain_serial_access_and_run_registration(domain_worker):
    worker, members = domain_worker
    saved = []
    worker.run(lambda orch, _args, _cfg: saved.append(orch.allocate_domain(**_request(members))))
    domain = saved[0]
    orch = worker._orch
    assert orch is not None
    assert orch.get_domain("tp") is domain
    assert domain.member(0).local_worker_id == 1
    assert domain.buffer_range("payload") == (0, 64)
    if worker.level == 3:
        assert domain.local_context(0) is domain[1]
    orch.copy_to_domain(domain, 0, b"serial", buffer="payload")
    assert orch.copy_from_domain(domain, 0, 6, buffer="payload") == b"serial"
    if worker.level >= 4:
        orch.copy_to_global_domain(domain, 0, b"legacy", buffer="payload")
        assert orch.copy_from_global_domain(domain, 0, 6, buffer="payload") == b"legacy"
    assert not domain.released and not domain.freed

    def lookup(orch, _args, _cfg):
        current = orch.get_domain("tp")
        assert current is domain
        assert current.buffer_range("payload") == (0, 64)
        resources = worker._building_run_resources
        assert resources is not None and resources.requires_ordered_cleanup
        table = resources.live_domains if worker.level == 3 else resources.live_global_domains
        assert table.get("tp") is current

    worker.run(lookup)
    with domain as current:
        assert current is domain
    assert domain.freed


def test_accepted_callback_can_release_while_close_drains(domain_worker):
    worker, members = domain_worker
    saved = []
    request = _request(members, buffers=[CommBufferSpec(name, "uint8", 4, 64) for name in ("payload", "tail")])

    def allocate(orch, _args, _cfg):
        domain = orch.allocate_domain(**request)
        orch.copy_to_domain(domain, 0, b"abcdefgh", offset=60)
        saved.append(domain)

    worker.run(allocate)
    entered, resume = threading.Event(), threading.Event()
    errors = []

    def build(orch, _args, _cfg):
        entered.set()
        assert resume.wait(10)
        domain = orch.get_or_allocate_domain(**request)
        assert orch.get_domain("tp") is domain
        assert domain.member(0).local_worker_id == 1
        with pytest.raises(RuntimeError, match="READY"):
            with worker._operation_lease("fresh operation"):
                pass
        orch.copy_to_domain(domain, 0, b"copy", offset=62)
        assert orch.copy_from_domain(domain, 0, 8, offset=60) == b"abcopygh"
        domain.release()
        assert domain.released and not domain.freed

    def run():
        try:
            worker.run(build)
        except BaseException as exc:
            errors.append(exc)

    def resume_after_close_claim():
        try:
            with worker._hierarchical_start_cv:
                assert worker._hierarchical_start_cv.wait_for(
                    lambda: worker._lifecycle is worker_mod._Lifecycle.CLOSED, timeout=10
                )
            with pytest.raises(RuntimeError, match="READY"):
                worker.create_buffer(1)
            assert worker._orch is not None
            with _callback_run(0, object()), pytest.raises(RuntimeError, match="callback"):
                worker._orch.get_or_allocate_domain(**request)
        except BaseException as exc:
            errors.append(exc)
        finally:
            resume.set()

    runner = threading.Thread(target=run)
    observer = threading.Thread(target=resume_after_close_claim)
    runner.start()
    assert entered.wait(10)
    observer.start()
    try:
        # Native teardown belongs to the init-owner thread; close waits for the accepted callback.
        worker.close()
    finally:
        resume.set()
        runner.join(10)
        observer.join(10)
    assert not runner.is_alive() and not observer.is_alive()
    assert not errors
    assert saved[0].freed


def test_domain_reference_preserves_views_and_revokes_before_failed_backend_release(monkeypatch):
    from dataclasses import replace  # noqa: PLC0415
    from multiprocessing.shared_memory import SharedMemory  # noqa: PLC0415

    from simpler.buffer import ImportContext, ImportRegistry, buffer_reference, wrap_vmm_window  # noqa: PLC0415
    from simpler.comm_endpoints import DEVICE_AICPU  # noqa: PLC0415
    from simpler.global_comm_domain import (  # noqa: PLC0415
        GlobalDomainBuffer,
        GlobalDomainCommand,
        GlobalDomainMember,
        GlobalDomainPhase,
        decode_domain_command,
        encode_domain_command,
    )
    from simpler.task_interface import _materialize_chip_task_args  # noqa: PLC0415

    command = GlobalDomainCommand(
        phase=GlobalDomainPhase.PREPARE_EXPORT,
        domain_id=7,
        generation=3,
        name="tp",
        profile="sim",
        window_size=4096,
        members=(GlobalDomainMember(2, 0, 10, 0),),
        buffers=(GlobalDomainBuffer("prefix", 16, "uint8", 16), GlobalDomainBuffer("payload", 32, "float32", 8)),
        owner_instance_id=b"creator1",
    )
    received = decode_domain_command(encode_domain_command(command))
    assert received == command
    worker = Worker(level=3, device_ids=(0,), num_sub_workers=0)
    granted = wrap_vmm_window(4112, 32, worker._owner_instance_id, 41)
    with worker._child_prov_lock:
        worker._record_device_alloc(granted, domain_allocation_id=7)
    logical = buffer_reference(granted)
    assert logical.base == 0 and logical.body == b"" and logical.owner_worker_path_id == 0
    first = logical.tensor([2], DataType.FLOAT32, byte_offset=4)
    second = logical.tensor([2], DataType.FLOAT32, byte_offset=12)
    assert first.buffer.identity == second.buffer.identity == granted.identity
    entry = worker_mod._L2GlobalDomain(
        domain_id=7,
        generation=3,
        domain_rank=0,
        rank_count=1,
        descriptor=worker_mod.GlobalDomainDescriptor(worker_mod.GLOBAL_DOMAIN_VERSION, 1, 0, 1, 4096, b"test"),
        local_window_base=4096,
        mapping_size=4096,
        requested_window_size=4096,
        device_ctx=1,
    )
    store = worker_mod._L2GlobalDomainStore({7: entry})
    registry = ImportRegistry(
        ImportContext(deployment=DEVICE_AICPU, device_owner_instance_id=worker._owner_instance_id)
    )

    def materialize():
        backing = worker._reexport(first.buffer)
        args = TaskArgs()
        for tensor in (first, second):
            args.add_tensor(backing.tensor(tensor.shapes, tensor.dtype, byte_offset=tensor.byte_offset))
        with worker._child_prov_lock:
            worker._child_prov_check_dispatch_locked(args, 0, api="test")
            with pytest.raises(ValueError, match="wrong worker|stale"):
                worker._child_prov_check_dispatch_locked(args, 1, api="test")
        return _materialize_chip_task_args(args, registry)

    native = materialize()
    assert native.tensor(0).data == 4116 and native.tensor(1).data == 4124
    assert registry.require(granted.identity).base == granted.base
    changed = replace(logical)
    changed.nbytes += 1
    with pytest.raises(ValueError, match="registered backing"):
        worker._reexport(changed.to_descriptor())
    with pytest.raises(ValueError, match="creator identity"):
        encode_domain_command(replace(command, owner_instance_id=b""))

    class Chip:
        fail = True

        def comm_global_domain_release(self, _domain_id):
            if self.fail:
                raise RuntimeError("release failed")

    def release_payload(_buf):
        size = worker_mod.LOCAL_RELEASE_REQUEST.size
        shm = SharedMemory(create=True, size=size)
        shm.unlink()
        payload = shm.buf
        assert payload is not None
        worker_mod.LOCAL_RELEASE_REQUEST.pack_into(
            payload, 0, worker_mod.LOCAL_DOMAIN_MAGIC, worker_mod.GLOBAL_DOMAIN_VERSION, 7, 3
        )
        return shm, payload, size

    monkeypatch.setattr(worker_mod, "_open_global_domain_payload", release_payload)
    chip = Chip()
    worker._drop_domain_allocs(7)
    with pytest.raises(RuntimeError, match="release failed"):
        worker_mod._handle_ctrl_global_domain_release(chip, memoryview(bytearray()), store)
    assert store.domains[7] is entry
    with pytest.raises(ValueError, match="stale"):
        materialize()
    chip.fail = False
    worker_mod._handle_ctrl_global_domain_release(chip, memoryview(bytearray()), store)
    assert store.domains == {}
    with pytest.raises(ValueError, match="stale"):
        materialize()
    registry.close()
    worker.close()


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("host_nbytes", [16, 32])
def test_domain_and_host_descriptors_cannot_alias_identity(reverse, separate, host_nbytes):
    from simpler.buffer import ImportContext, ImportRegistry, wrap_fork_inherited, wrap_vmm_window  # noqa: PLC0415
    from simpler.comm_endpoints import DEVICE_AICPU  # noqa: PLC0415
    from simpler.task_interface import _materialize_chip_task_args  # noqa: PLC0415

    domain = wrap_vmm_window(4096, 32, b"creator1", 1)
    host = wrap_fork_inherited(8192, host_nbytes, b"creator1", 1)
    ordered = (host, domain) if reverse else (domain, host)
    registry = ImportRegistry(ImportContext(deployment=DEVICE_AICPU, device_owner_instance_id=b"creator1"))

    def materialize(buffers):
        args = TaskArgs()
        for buffer in buffers:
            args.add_tensor(buffer.tensor([8], DataType.UINT8))
        return _materialize_chip_task_args(args, registry)

    try:
        if separate:
            first = materialize(ordered[:1])
            assert first.tensor(0).data == (8192 if reverse else 4096)
        with pytest.raises(ValueError, match="different descriptor"):
            materialize(ordered[1:] if separate else ordered)
        assert materialize(ordered[:1]).tensor(0).data == (8192 if reverse else 4096)
    finally:
        registry.close()


@pytest.fixture
def logical_buffer_worker():
    from simpler.buffer import buffer_reference, wrap_vmm_window  # noqa: PLC0415

    worker = Worker(level=4, num_sub_workers=0)
    actual = wrap_vmm_window(4096, 64, b"node-own", 7, "L3")
    reference = buffer_reference(actual, owner_worker_id=3, remote_ref=(41, 2))
    with worker._child_prov_lock:
        worker._record_device_alloc(reference, domain_allocation_id=7)
    try:
        yield worker, actual, reference
    finally:
        worker.close()


def test_buffer_reference_projects_storage_without_changing_canonical_identity(logical_buffer_worker):
    from dataclasses import replace  # noqa: PLC0415

    from simpler.buffer import BackendKind, buffer_reference  # noqa: PLC0415

    _worker, actual, reference = logical_buffer_worker
    assert reference.identity == actual.identity
    assert reference.backend_kind == BackendKind.REMOTE_SIDECAR
    assert reference.nbytes == actual.nbytes and reference.access == actual.access
    assert reference.address_space == actual.address_space
    assert reference.base == 0 and reference.body == b"" and reference.owner_worker_path_id == 0
    assert reference.to_descriptor() == buffer_reference(actual.to_descriptor()).to_descriptor()
    assert replace(reference)._remote_ref == (41, 2)


@pytest.mark.parametrize("corruption", ["identity", "generation", "extent", "access", "path", "target", "stale"])
def test_buffer_reference_submission_revalidates_private_registration(logical_buffer_worker, corruption):
    from dataclasses import replace  # noqa: PLC0415

    from simpler.buffer import AccessMode, CanonicalIdentity  # noqa: PLC0415
    from simpler.task_interface import _empty_remote_sidecar_for  # noqa: PLC0415

    worker, _actual, reference = logical_buffer_worker
    changes = {
        "identity": {"identity": CanonicalIdentity(b"forged!!", 7, 1)},
        "generation": {"identity": CanonicalIdentity(b"node-own", 7, 2)},
        "extent": {"nbytes": 65},
        "access": {"access": AccessMode.READ},
        "path": {"owner_worker_path_id": 1},
    }
    args = TaskArgs()
    args.add_tensor(replace(reference, **changes.get(corruption, {})).tensor([4], DataType.UINT8, byte_offset=4))
    if corruption == "stale":
        worker._drop_domain_allocs(7)
    with worker._child_prov_lock, pytest.raises(ValueError, match="stale|registration"):
        worker._prepare_buffer_references(args, _empty_remote_sidecar_for(args), 4 if corruption == "target" else 3)


@pytest.mark.parametrize("remote", [False, True])
def test_buffer_reference_cannot_cross_dispatcher_kind(logical_buffer_worker, remote):
    from dataclasses import replace  # noqa: PLC0415

    from simpler.task_interface import _empty_remote_sidecar_for  # noqa: PLC0415

    worker, _actual, reference = logical_buffer_worker
    if remote:
        reference = replace(reference, _remote_ref=None)
        with worker._child_prov_lock:
            worker._record_device_alloc(reference, domain_allocation_id=7)
    args = TaskArgs()
    args.add_tensor(reference.tensor([4], DataType.UINT8))
    with worker._child_prov_lock, pytest.raises(ValueError, match="dispatcher"):
        worker._prepare_buffer_references(args, _empty_remote_sidecar_for(args) if remote else None, 3)


def test_buffer_reference_does_not_widen_registered_access(logical_buffer_worker):
    from dataclasses import replace  # noqa: PLC0415

    from simpler.buffer import AccessMode  # noqa: PLC0415
    from simpler.task_interface import TensorArgType, _empty_remote_sidecar_for  # noqa: PLC0415

    worker, _actual, reference = logical_buffer_worker
    reference = replace(reference, access=AccessMode.READ)
    with worker._child_prov_lock:
        worker._record_device_alloc(reference, domain_allocation_id=7)
    for tag in (TensorArgType.OUTPUT_EXISTING, TensorArgType.INOUT):
        args = TaskArgs()
        with pytest.raises(ValueError, match="access"):
            args.add_tensor(reference.tensor([4], DataType.UINT8), tag)
    args = TaskArgs()
    args.add_tensor(reference.tensor([4], DataType.UINT8), TensorArgType.INPUT)
    with worker._child_prov_lock:
        assert worker._prepare_buffer_references(args, _empty_remote_sidecar_for(args), 3).tensors[0].present


def test_automatic_sidecar_mixes_with_remote_refs_without_mutating_cached_args(logical_buffer_worker):
    from simpler.task_interface import (  # noqa: PLC0415
        RemoteAddressSpace,
        RemoteBufferHandle,
        RemoteTensorRef,
        TensorArgType,
        _remote_sidecar_for,
        _validate_remote_sidecar_access,
    )

    worker, _actual, reference = logical_buffer_worker
    ordinary = RemoteBufferHandle._from_remote_allocation(
        worker_id=3,
        buffer_id=90,
        generation=5,
        address_space=RemoteAddressSpace.REMOTE_DEVICE,
        nbytes=64,
        remote_addr=8192,
        rkey_or_token=0,
    )
    args = TaskArgs()
    args.add_tensor(reference.tensor([4], DataType.UINT8, byte_offset=4), TensorArgType.INOUT)
    args.add_tensor(RemoteTensorRef(ordinary, offset=8, shape=(4,), dtype=DataType.UINT8))
    args.add_scalar(19)
    sidecar = _remote_sidecar_for(args)
    assert sidecar is not None and sidecar.tensors[0] is None
    original_tensor = args.tensor(0)
    _validate_remote_sidecar_access(args, sidecar)
    with worker._child_prov_lock:
        prepared = worker._prepare_buffer_references(args, sidecar, 3)
        assert worker._prepare_buffer_references(args, sidecar, 3) == prepared
    assert _remote_sidecar_for(args) == sidecar and sidecar.tensors[0] is None
    assert args.tensor(0).buffer == original_tensor.buffer and args.tensor(0).byte_offset == 4
    assert args.tag(0) == TensorArgType.INOUT and args.scalar(0) == 19
    generated = prepared.tensors[0]
    assert generated is not None and generated.handle is None
    assert (generated.desc.buffer_id, generated.desc.generation, generated.desc.offset) == (41, 2, 4)
    assert prepared.tensors[1] == sidecar.tensors[1] and prepared.tensors[1].handle is ordinary
    reference._remote_ref = (999, 999)
    with worker._child_prov_lock:
        assert worker._prepare_buffer_references(args, sidecar, 3) == prepared


def test_release_and_rebuild_revoke_old_tensor_and_reference(domain_worker):
    worker, members = domain_worker
    saved = []

    def build(orch, _args, _cfg):
        old = orch.allocate_domain(**_request(members))
        ref = old.ref
        payload = old.member(0).buffers["payload"]
        args = TaskArgs()
        args.add_tensor(payload.tensor([8], DataType.UINT8))
        old.release()
        target = members[0] if worker.level == 3 else members[0][0]
        with worker._child_prov_lock, pytest.raises(ValueError, match="stale"):
            _check_domain_dispatch_locked(worker, args, target)
        new = orch.allocate_domain(**_request(members))
        assert new.ref != ref and new.member(0).buffers["payload"].identity != payload.identity
        with pytest.raises(KeyError):
            orch.get_domain(ref)
        with worker._child_prov_lock, pytest.raises(ValueError, match="stale"):
            _check_domain_dispatch_locked(worker, args, target)
        saved.extend((old, new))

    worker.run(build)
    assert saved[0].freed and not saved[1].freed

    def finish(orch, _args, _cfg):
        current = orch.get_domain("tp")
        args = TaskArgs()
        args.add_tensor(current.member(0).buffers["payload"].tensor([8], DataType.UINT8))
        with worker._child_prov_lock:
            target = members[0] if worker.level == 3 else members[0][0]
            _check_domain_dispatch_locked(worker, args, target)
        current.release()

    worker.run(finish)
    assert saved[1].freed


def test_borrowed_local_context_does_not_alias_another_nodes_chip_id():
    from simpler.global_comm_domain import GlobalDomainMember  # noqa: PLC0415
    from simpler.task_interface import ChipDomainContext  # noqa: PLC0415

    local = ChipDomainContext("tp", 0, 2, 1, 4096, 4096, {})
    borrowed = CommDomainHandle(
        name="tp",
        members=(GlobalDomainMember(0, 0, 10, 0), GlobalDomainMember(1, 0, 20, 1)),
        contexts={0: local},
        borrowed=True,
        _release_fn=None,
    )
    assert borrowed.local_context(0) is local
    with pytest.raises(KeyError, match="not mapped"):
        borrowed.local_context(1)


def test_interrupted_release_before_admission_preserves_live_handle():
    import contextlib  # noqa: PLC0415

    calls = []

    def reject(_handle):
        raise KeyboardInterrupt("release not admitted")

    domain = CommDomainHandle(name="not-admitted", _release_fn=calls.append, _release_scope=reject)
    with pytest.raises(KeyboardInterrupt, match="not admitted"):
        domain.release()
    assert not domain.released and not domain.freed

    domain._release_scope = contextlib.nullcontext
    domain.release()
    domain.release()
    assert domain.released and calls == [domain]


@pytest.mark.parametrize("fail", [False, True])
def test_cached_tensor_use_enrolls_without_lookup(domain_worker, fail):
    worker, members = domain_worker
    saved = []
    worker.run(lambda orch, _args, _cfg: saved.append(orch.allocate_domain(**_request(members))))
    domain = saved[0]
    args = TaskArgs()
    args.add_tensor(domain.member(0).buffers["payload"].tensor([8], DataType.UINT8))

    def consume(_orch, _args, _cfg):
        worker._record_touched_identities(args)
        if fail:
            raise LookupError("consumer failed")

    if fail:
        with pytest.raises(LookupError, match="consumer failed"):
            worker.submit(consume)
        assert domain.released and domain.freed
    else:
        run = worker.submit(consume)
        table = run._resources.live_domains if worker.level == 3 else run._resources.live_global_domains
        assert table.get("tp") is domain and run._resources.requires_ordered_cleanup
        domain.release()
        assert domain.released and not domain.freed
        run.wait()
        assert domain.freed


def test_foreign_orchestrator_cannot_release_domain(domain_worker):
    from simpler.orchestrator import Orchestrator  # noqa: PLC0415

    worker, members = domain_worker
    saved = []
    worker.run(lambda orch, _args, _cfg: saved.append(orch.allocate_domain(**_request(members))))
    other = Worker(level=4, num_sub_workers=0)
    with pytest.raises(ValueError, match="creating Worker"):
        Orchestrator(worker._orch._o, other).release_domain(saved[0])
    assert not saved[0].released


def test_cached_tensor_release_inside_callback_orders_cleanup(domain_worker):
    worker, members = domain_worker
    saved = []
    worker.run(lambda orch, _args, _cfg: saved.append(orch.allocate_domain(**_request(members))))
    domain = saved[0]
    args = TaskArgs()
    args.add_tensor(domain.member(0).buffers["payload"].tensor([8], DataType.UINT8))

    def consume(_orch, _args, _cfg):
        worker._record_touched_identities(args)
        domain.release()
        assert worker._building_run_resources.requires_ordered_cleanup
        assert domain.released and not domain.freed

    worker.run(consume)
    assert domain.freed


@pytest.mark.parametrize("retained", [False, True])
def test_interrupted_pending_publication_keeps_cleanup_owner(domain_worker, monkeypatch, retained):
    worker, members = domain_worker
    saved, freed = [], []
    backend = "_release_domain_claimed" if worker.level == 3 else "_release_global_domain_claimed"
    original = getattr(worker, backend)

    def free(handle):
        original(handle)
        freed.append(handle)

    monkeypatch.setattr(worker, backend, free)

    class InterruptedQueue(list):
        def append(self, _handle):
            raise KeyboardInterrupt("pending publication")

    if retained:
        worker.run(lambda orch, _args, _cfg: saved.append(orch.allocate_domain(**_request(members))))

    def build(orch, _args, _cfg):
        if retained:
            domain = saved[0]
            args = TaskArgs()
            args.add_tensor(domain.member(0).buffers["payload"].tensor([8], DataType.UINT8))
            worker._record_touched_identities(args)
        else:
            domain = orch.allocate_domain(**_request(members))
            saved.append(domain)
        resources = worker._building_run_resources
        queue = "pending_release_domains" if worker.level == 3 else "pending_release_global_domains"
        setattr(resources, queue, InterruptedQueue())
        with pytest.raises(KeyboardInterrupt, match="pending publication"):
            domain.release()
        table = resources.live_domains if worker.level == 3 else resources.live_global_domains
        assert table.get("tp") is domain and resources.requires_ordered_cleanup
        owners = worker._live_domains if worker.level == 3 else worker._live_global_domains
        assert owners.get("tp") is domain
        assert domain.released and not domain.freed

    worker.run(build)
    assert saved[0].freed and freed == saved


@pytest.mark.parametrize("domain_worker", [4], indirect=True)
@pytest.mark.parametrize("before_entry", [False, True])
def test_interrupted_global_free_remains_close_owned(domain_worker, monkeypatch, before_entry):
    worker, members = domain_worker
    saved = []
    worker.run(lambda orch, _args, _cfg: saved.append(orch.allocate_domain(**_request(members))))
    domain = saved[0]

    def interrupt(_domain):
        raise KeyboardInterrupt("free interrupted")

    with monkeypatch.context() as patch:
        boundary = "_free_global_domain_after_fence" if before_entry else "_release_global_domain_now"
        patch.setattr(worker, boundary, interrupt)
        with pytest.raises(KeyboardInterrupt, match="free interrupted"):
            domain.release()
    assert domain.released and not domain.freed
    if before_entry:
        assert worker._live_global_domains.get(domain.name) is domain
    else:
        assert worker._failed_global_domain_releases.get(domain.domain_id) is domain
    worker.close()
    assert domain.freed and worker._failed_global_domain_releases == {}


@pytest.mark.parametrize("domain_worker", [3, 4], indirect=True)
def test_failed_pending_release_remains_worker_owned(domain_worker, monkeypatch):
    worker, members = domain_worker
    saved = []

    def build(orch, _args, _cfg):
        domain = orch.allocate_domain(**_request(members))
        saved.append(domain)
        domain.release()

    def interrupt(_handle):
        raise RuntimeError("pending free failed")

    with monkeypatch.context() as patch:
        boundary = "_free_domain_after_fence" if worker.level == 3 else "_free_global_domain_after_fence"
        patch.setattr(worker, boundary, interrupt)
        with pytest.raises(RuntimeError, match="pending free failed"):
            worker.run(build)
    assert saved[0].released and not saved[0].freed
    worker.close()
    assert saved[0].freed


@pytest.mark.parametrize("domain_worker", [3], indirect=True)
@pytest.mark.parametrize("release_path", ["explicit", "automatic", "external", "failed_consumer", "close"])
@pytest.mark.parametrize(
    "teardown_failure", [None, "_reclaim_child_groups", "_close_worker_chip_orch_comm", "journal", "proof"]
)
def test_cached_release_failure_waits_for_tree_teardown(domain_worker, monkeypatch, teardown_failure, release_path):
    worker, members = domain_worker
    saved, calls = [], []

    def build(orch, _args, _cfg):
        domain = orch.allocate_domain(**_request(members, retain_after_run=release_path != "automatic"))
        saved.append(domain)
        if release_path == "explicit":
            domain.release()

    def fail_consumer(orch, _args, _cfg):
        orch.get_domain("tp")
        raise RuntimeError("consumer failed")

    def fail_release(domain):
        calls.append(domain)
        raise RuntimeError("backend release failed")

    teardown = getattr(worker, teardown_failure) if teardown_failure not in (None, "journal", "proof") else lambda: None
    interrupted = []
    withheld_proofs = set()
    record_completion = worker_mod._record_chip_teardown_completion

    def withhold_completion(shm, pid, _finalized):
        record_completion(shm, pid, withheld_proofs)

    def fail_teardown(*args):
        assert not domain.freed
        if not interrupted:
            interrupted.append(True)
            raise RuntimeError("tree teardown failed")
        return teardown(*args)

    try:
        with monkeypatch.context() as patch:
            if release_path in ("external", "failed_consumer", "close"):
                worker.run(build)
            patch.setattr(worker, "_release_domain_claimed", fail_release)
            with pytest.raises(RuntimeError, match="consumer failed|backend release failed"):
                if release_path == "external":
                    saved[0].release()
                elif release_path == "failed_consumer":
                    worker.run(fail_consumer)
                elif release_path == "close":
                    worker.close()
                else:
                    worker.run(build)
            domain = saved[0]
            assert domain.released and not domain.freed
            assert isinstance(worker._domain_free_results[domain.allocation_id], RuntimeError)
            if teardown_failure == "proof":
                with monkeypatch.context() as teardown_patch:
                    teardown_patch.setattr(worker_mod, "_record_chip_teardown_completion", withhold_completion)
                    for _ in range(2):
                        with pytest.raises(RuntimeError, match="reclamation is unconfirmed"):
                            worker.close()
                        assert not domain.freed and not worker._cleanup_journal.empty
                assert not worker._chip_pids and not worker._chip_shms
                worker._finalized_chip_pids.update(withheld_proofs)
            elif teardown_failure is not None:
                with monkeypatch.context() as teardown_patch:
                    if teardown_failure == "journal":
                        worker._cleanup_journal.add_once("shm", "interrupted mailbox", fail_teardown)
                    else:
                        teardown_patch.setattr(worker, teardown_failure, fail_teardown)
                    with pytest.raises(RuntimeError, match="tree teardown failed"):
                        worker.close()
                assert not domain.freed and not worker._cleanup_journal.empty
                if teardown_failure == "journal":
                    assert not worker._chip_pids and not worker._chip_shms
            worker.close()
            assert domain.freed and worker._cleanup_journal.empty
            assert worker._worker is None and not worker._chip_pids and not worker._chip_shms
            domain.release()
            assert calls == [domain]
    finally:
        # The injected release never reached a device. Allow test cleanup even
        # when an assertion fails against an implementation without fallback.
        if saved and not saved[0].freed:
            worker._finalized_chip_pids.update(withheld_proofs)
            worker._domain_free_results.pop(saved[0].allocation_id, None)
            worker.close()


@pytest.mark.parametrize("domain_worker", [4], indirect=True)
def test_freed_global_domain_retires_duplicate_cleanup_debt(domain_worker, monkeypatch):
    worker, members = domain_worker
    saved = []
    worker.run(lambda orch, _args, _cfg: saved.append(orch.allocate_domain(**_request(members))))
    domain = saved[0]
    domain.release()
    assert domain.freed
    worker._failed_global_domain_releases[domain.domain_id] = domain

    def double_free(_domain):
        pytest.fail("a freed domain must not replay the backend release")

    monkeypatch.setattr(worker, "_release_global_domain_now", double_free)
    worker._free_global_domain_after_fence(domain)
    assert worker._failed_global_domain_releases == {}


@pytest.mark.parametrize("domain_worker", [4], indirect=True)
def test_successful_global_free_interrupted_before_freed_publication(domain_worker, monkeypatch):
    worker, members = domain_worker
    saved, calls = [], []
    worker.run(lambda orch, _args, _cfg: saved.append(orch.allocate_domain(**_request(members))))
    domain = saved[0]
    release_now, release_claimed = worker._release_global_domain_now, worker._release_global_domain_claimed

    def counted_release(handle):
        calls.append(handle)
        release_claimed(handle)

    def interrupt_publication(handle):
        release_now(handle)
        raise KeyboardInterrupt("freed publication interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(worker, "_release_global_domain_claimed", counted_release)
        with monkeypatch.context() as interrupt_patch:
            interrupt_patch.setattr(worker, "_release_global_domain_now", interrupt_publication)
            with pytest.raises(KeyboardInterrupt, match="freed publication interrupted"):
                domain.release()
        assert not domain.freed and worker._global_domain_free_results[domain.domain_id] is None
        worker.close()
        assert domain.freed and not worker._failed_global_domain_releases
        assert calls == [domain]


@pytest.mark.parametrize("domain_worker", [4], indirect=True)
@pytest.mark.parametrize("release_path", ["explicit", "automatic", "external", "failed_consumer", "close"])
@pytest.mark.parametrize("backend_completed", [False, True])
def test_cached_global_release_failure_allows_tree_teardown(
    domain_worker, monkeypatch, release_path, backend_completed
):
    worker, members = domain_worker
    saved, calls = [], []
    release_claimed = worker._release_global_domain_claimed
    mailboxes = tuple(shm.name for shm in worker._next_level_shms)

    def build(orch, _args, _cfg):
        domain = orch.allocate_domain(**_request(members, retain_after_run=release_path != "automatic"))
        saved.append(domain)
        if release_path == "explicit":
            domain.release()

    def fail_consumer(orch, _args, _cfg):
        orch.get_domain("tp")
        raise RuntimeError("consumer failed")

    def fail_release(domain):
        calls.append(domain)
        if backend_completed:
            release_claimed(domain)
        raise RuntimeError("global backend release failed")

    try:
        with monkeypatch.context() as patch:
            if release_path in ("external", "failed_consumer", "close"):
                worker.run(build)
            patch.setattr(worker, "_release_global_domain_claimed", fail_release)
            with pytest.raises(RuntimeError, match="consumer failed|global backend release failed"):
                if release_path == "external":
                    saved[0].release()
                elif release_path == "failed_consumer":
                    worker.run(fail_consumer)
                elif release_path == "close":
                    worker.close()
                else:
                    worker.run(build)
            domain = saved[0]
            for _ in range(2):
                with pytest.raises(RuntimeError, match="reclamation is unconfirmed"):
                    worker.close()
                assert worker._worker is None
                assert not worker._next_level_pids and not worker._next_level_shms
                for name in mailboxes:
                    with pytest.raises(FileNotFoundError):
                        worker_mod.SharedMemory(name=name)
                assert domain.released and not domain.freed
                assert worker._failed_global_domain_releases[domain.domain_id] is domain
                assert calls == [domain]
    finally:
        # Restore only this test's deliberately injected unknown record. On the
        # unfixed code, replay only when no backend operation actually ran.
        if worker._worker is not None and saved:
            if backend_completed:
                worker._global_domain_free_results[saved[0].domain_id] = None
            else:
                worker._global_domain_free_results.pop(saved[0].domain_id, None)
            worker._free_global_domain_after_fence(saved[0])
        worker._failed_global_domain_releases.clear()


@pytest.mark.parametrize("domain_worker", [3, 4], indirect=True)
@pytest.mark.parametrize("action", ["release", "lookup"])
@pytest.mark.parametrize("waiter_thread", [False, True])
@pytest.mark.timeout(10)
def test_predecessor_destructor_can_access_domain(domain_worker, action, waiter_thread, monkeypatch):
    worker, members = domain_worker
    saved, results = [], []

    class Graph:
        def __call__(self, orch, _args, _cfg):
            saved.append(orch.allocate_domain(**_request(members)))

        def __del__(self):
            try:
                if action == "release":
                    saved[0].release()
                else:
                    assert worker.live_domains["tp"] is saved[0]
                results.append("done")
            except BaseException as exc:
                results.append(exc)

    first = worker.submit(Graph())
    elected, resume = threading.Event(), threading.Event()
    errors = []

    def pause_waiter(phase):
        if phase == "after_election":
            elected.set()
            assert resume.wait(5)

    def wait_first():
        try:
            first.wait()
        except BaseException as exc:
            errors.append(exc)

    handoff = first._wait_for_handoff

    def resume_waiter():
        resume.set()
        handoff()

    thread = threading.Thread(target=wait_first)
    if waiter_thread:
        first._wait_boundary_hook = pause_waiter
        monkeypatch.setattr(first, "_wait_for_handoff", resume_waiter)
        thread.start()
    try:
        if waiter_thread:
            assert elected.wait(5)
        worker.submit(lambda _orch, _args, _cfg: None).wait()
    finally:
        resume.set()
        if waiter_thread:
            thread.join(5)
            assert not thread.is_alive()
    first.wait()
    assert not errors
    assert results == ["done"]
    assert saved[0].freed == (action == "release")


def test_detach_retains_same_owner_and_run_fence(domain_worker):
    worker, members = domain_worker
    saved = []

    def build(orch, _args, _cfg):
        domain = orch.allocate_domain(**_request(members, retain_after_run=False))
        worker.detach_persistent_domain(domain)
        resources = worker._building_run_resources
        table = resources.live_domains if worker.level == 3 else resources.live_global_domains
        assert table.get("tp") is domain
        assert domain.retain_after_run
        assert orch.get_or_allocate_domain(**_request(members)) is domain
        with pytest.raises(ValueError, match="already retained"):
            worker.detach_persistent_domain(domain)
        saved.append(domain)

    worker.run(build)
    assert worker.live_domains == {"tp": saved[0]} and not saved[0].freed
    with pytest.raises(RuntimeError, match="callback"):
        worker.detach_persistent_domain(saved[0])
    worker.close()
    assert saved[0].released and saved[0].freed


@pytest.mark.parametrize("conflict", [False, True])
def test_concurrent_creation_uses_existing_callback_serializer(domain_worker, conflict):
    worker, members = domain_worker
    entered, resume, second_entered = threading.Event(), threading.Event(), threading.Event()
    saved, errors = [], []

    def first(orch, _args, _cfg):
        entered.set()
        assert resume.wait(10)
        saved.append(orch.get_or_allocate_domain(**_request(members)))

    def second(orch, _args, _cfg):
        second_entered.set()
        saved.append(orch.get_or_allocate_domain(**_request(members, window_size=8192 if conflict else 4096)))

    def run(callback):
        try:
            worker.run(callback)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(callback,)) for callback in (first, second)]
    threads[0].start()
    try:
        assert entered.wait(10)
        threads[1].start()
        assert not second_entered.wait(0.05)
    finally:
        resume.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(10)
                assert not thread.is_alive()
    assert worker._next_alloc_id == 1
    if conflict:
        assert len(errors) == 1 and isinstance(errors[0], ValueError)
        assert "window_size" in str(errors[0])
    else:
        assert errors == [] and len(saved) == 2 and saved[0] is saved[1]


def test_external_release_waits_for_callback_use_publication(domain_worker):
    worker, members = domain_worker
    saved = []
    worker.run(lambda orch, _args, _cfg: saved.append(orch.allocate_domain(**_request(members))))
    domain = saved[0]
    args = TaskArgs()
    args.add_tensor(domain.member(0).buffers["payload"].tensor([8], DataType.UINT8))
    entered, resume, release_done = threading.Event(), threading.Event(), threading.Event()
    errors = []

    def consume(_orch, _args, _cfg):
        worker._record_touched_identities(args)
        entered.set()
        assert resume.wait(10)
        assert not domain.released and not domain.freed

    def invoke(action):
        try:
            action()
        except BaseException as exc:
            errors.append(exc)

    def release():
        domain.release()
        release_done.set()

    runner = threading.Thread(target=invoke, args=(lambda: worker.run(consume),))
    releaser = threading.Thread(target=invoke, args=(release,))
    runner.start()
    try:
        assert entered.wait(10)
        releaser.start()
        assert not release_done.wait(0.05)
    finally:
        resume.set()
        for thread in (runner, releaser):
            if thread.ident is not None:
                thread.join(10)
                assert not thread.is_alive()
    assert errors == [] and release_done.is_set() and domain.freed


def test_staging_copy_excludes_release_until_completion(domain_worker, monkeypatch):
    worker, members = domain_worker
    saved = []
    worker.run(lambda orch, _args, _cfg: saved.append(orch.allocate_domain(**_request(members))))
    domain = saved[0]
    orch = worker._orch
    original = orch._copy_domain_locked
    entered, resume, release_done = threading.Event(), threading.Event(), threading.Event()
    errors = []

    def copy(*args, **kwargs):
        entered.set()
        assert resume.wait(10)
        assert not domain.released and not domain.freed
        return original(*args, **kwargs)

    monkeypatch.setattr(orch, "_copy_domain_locked", copy)

    def invoke(action):
        try:
            action()
        except BaseException as exc:
            errors.append(exc)

    def release():
        domain.release()
        release_done.set()

    copier = threading.Thread(target=invoke, args=(lambda: orch.copy_to_domain(domain, 0, b"safe", buffer="payload"),))
    releaser = threading.Thread(target=invoke, args=(release,))
    copier.start()
    try:
        assert entered.wait(10)
        releaser.start()
        assert not release_done.wait(0.05)
    finally:
        resume.set()
        for thread in (copier, releaser):
            if thread.ident is not None:
                thread.join(10)
                assert not thread.is_alive()
    assert errors == [] and release_done.is_set() and domain.freed


@pytest.mark.parametrize("domain_worker", [3], indirect=True)
@pytest.mark.parametrize("writing", [False, True])
def test_ordinary_copy_registers_cached_domain_before_failure(domain_worker, writing):
    worker, members = domain_worker
    saved = []
    worker.run(lambda orch, _args, _cfg: saved.append(orch.allocate_domain(**_request(members))))
    domain = saved[0]
    device = domain.member(0).buffers["payload"]
    host = worker.create_buffer(4)
    host.shm.buf[:4] = b"copy"

    def consume(orch, _args, _cfg):
        if writing:
            orch.copy_to(device, host, nbytes=4)
        else:
            orch.copy_from(host, device, nbytes=4)
        raise LookupError("copy consumer failed")

    with pytest.raises(LookupError, match="consumer failed"):
        worker.run(consume)
    assert domain.released and domain.freed and worker.live_domains == {}
