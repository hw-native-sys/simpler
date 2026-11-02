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
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import simpler.worker as worker_mod
from simpler.buffer import AccessMode, buffer_reference, wrap_vmm_window
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
        if getattr(worker, "_expected_domain_debt", False):
            _close_with_domain_debt(worker)
        else:
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


def _close_with_domain_debt(worker):
    worker._expected_domain_debt = True
    with pytest.raises(RuntimeError, match="unconfirmed"):
        worker.close()
    assert not worker._cleanup_journal.empty and worker._worker is None
    assert not worker._chip_pids and not worker._next_level_pids
    assert not worker._chip_shms and not worker._next_level_shms


def _retained(worker, members):
    saved = []
    worker.run(lambda orch, _args, _cfg: saved.append(orch.allocate_domain(**_request(members))))
    return saved[0]


def test_explicit_empty_workspace_selection_round_trips_to_chip(domain_worker):
    worker, members = domain_worker
    saved = []
    request = _request(members, workspace_engines=[])
    worker.run(lambda orch, _args, _cfg: saved.append(orch.allocate_domain(**request)))
    assert saved[0]._config.workspace_mask() == 0
    worker.run(lambda orch, _args, _cfg: saved.append(orch.get_or_allocate_domain(**request)))
    assert saved[0] is saved[1]
    worker.run(lambda orch, _args, _cfg: saved.append(orch.get_or_allocate_domain(**_request(members))))
    assert saved[0] is saved[2]


def _tensor_args(domain):
    args = TaskArgs()
    args.add_tensor(domain.member(0).buffers["payload"].tensor([8], DataType.UINT8))
    return args


def _authorize(worker, args, target):
    worker._prepare_buffer_references(args, None, target)
    worker._child_prov_check_dispatch_locked(args, target, api="test")


@contextmanager
def _race(first, second, entered, resume):
    errors = []

    def invoke(action):
        try:
            action()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=invoke, args=(action,)) for action in (first, second)]
    threads[0].start()
    try:
        assert entered.wait(5)
        threads[1].start()
        yield
    finally:
        resume.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(5)
                assert not thread.is_alive()
    assert errors == []


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
        args, ref, identity = _tensor_args(old), old.ref, old.member(0).buffers["payload"].identity
        old.release()
        target = members[0] if worker.level == 3 else members[0][0]
        with worker._child_prov_lock, pytest.raises(ValueError, match="stale"):
            _authorize(worker, args, target)
        new = orch.allocate_domain(**_request(members))
        assert old.domain_id != new.domain_id
        assert new.ref != ref and new.member(0).buffers["payload"].identity != identity
        with pytest.raises(KeyError):
            orch.get_domain(ref)
        with worker._child_prov_lock, pytest.raises(ValueError, match="stale"):
            _authorize(worker, args, target)
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
        worker_mod._DOMAIN_REPLY_HEADER.pack_into(reply, 0, 1, 0xC0, 0x1000, 8192, count + 1 if first else count)
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


def test_reuse_after_submission_has_no_device_control(domain_worker):
    from simpler.orchestrator import _admit_task_submission  # noqa: PLC0415

    worker, members = domain_worker

    def build(orch, _args, _cfg):
        domain = orch.allocate_domain(**_request(members))
        count = worker._next_alloc_id
        _admit_task_submission(worker)
        assert orch.get_or_allocate_domain(**_request(members)) is domain
        assert worker._next_alloc_id == count
        with pytest.raises(ValueError, match="configuration differs"):
            orch.get_or_allocate_domain(**_request(members, window_size=8192))
        if worker.level == 3:
            with pytest.raises(RuntimeError, match="direct device control"):
                orch.allocate_domain(**_request(members, name="new"))

    worker.run(build)


@pytest.mark.parametrize("use", ["cached", "release_inside", "failed", "copy_to", "copy_from"])
def test_cached_domain_use_enrolls_current_run(domain_worker, use):
    worker, members = domain_worker
    if use.startswith("copy") and worker.level != 3:
        pytest.skip("ordinary copy is an L3 API")
    domain = _retained(worker, members)
    args = _tensor_args(domain)
    host = worker.create_buffer(8) if use.startswith("copy") else None

    def consume(orch, _args, _cfg):
        if host is None:
            worker._record_touched_identities(args)
        elif use == "copy_to":
            orch.copy_to(domain.member(0).buffers["payload"], host, nbytes=8)
        else:
            orch.copy_from(host, domain.member(0).buffers["payload"], nbytes=8)
        if use in ("failed", "copy_to", "copy_from"):
            raise LookupError("consumer failed")
        if use == "release_inside":
            domain.release()
            assert domain.released and not domain.freed
            assert worker._building_run_resources.requires_ordered_cleanup

    if use in ("failed", "copy_to", "copy_from"):
        with pytest.raises(LookupError, match="consumer failed"):
            worker.submit(consume)
        assert domain.released and domain.freed and worker.live_domains == {}
    else:
        run = worker.submit(consume)
        table = run._resources.live_domains if worker.level == 3 else run._resources.live_global_domains
        assert run._resources.requires_ordered_cleanup
        if use == "release_inside":
            pending = (
                run._resources.pending_release_domains
                if worker.level == 3
                else run._resources.pending_release_global_domains
            )
            assert domain in pending and "tp" not in table
        else:
            assert table.get("tp") is domain
        domain.release()
        assert domain.released and not domain.freed
        run.wait()
        assert domain.freed


def test_release_admission_and_worker_ownership(domain_worker):
    from simpler.orchestrator import Orchestrator  # noqa: PLC0415

    worker, members = domain_worker
    domain = _retained(worker, members)
    other = Worker(level=4, num_sub_workers=0)
    with pytest.raises(ValueError, match="creating Worker"):
        Orchestrator(worker._orch._o, other).release_domain(domain)
    scope = domain._release_scope

    def interrupt(_domain):
        raise KeyboardInterrupt("not admitted")

    domain._release_scope = interrupt
    with pytest.raises(KeyboardInterrupt, match="not admitted"):
        domain.release()
    assert not domain.released and not domain.freed
    domain._release_scope = scope
    domain.release()
    domain.release()
    assert domain.freed


@pytest.mark.parametrize("retained", [False, True])
def test_interrupted_pending_publication_preserves_owner(domain_worker, retained):
    worker, members = domain_worker
    saved = [_retained(worker, members)] if retained else []

    class InterruptedQueue(list):
        def append(self, _handle):
            raise KeyboardInterrupt("pending publication")

    def build(orch, _args, _cfg):
        domain = saved[0] if retained else orch.allocate_domain(**_request(members))
        if not retained:
            saved.append(domain)
        resources = worker._building_run_resources
        queue = "pending_release_domains" if worker.level == 3 else "pending_release_global_domains"
        setattr(resources, queue, InterruptedQueue())
        with pytest.raises(KeyboardInterrupt, match="pending publication"):
            domain.release()
        table = resources.live_domains if worker.level == 3 else resources.live_global_domains
        assert table.get("tp") is domain and resources.requires_ordered_cleanup
        assert domain.released and not domain.freed

    worker.run(build)
    assert saved[0].freed


@pytest.mark.parametrize("path", ["explicit", "automatic", "external", "failed_consumer", "close"])
@pytest.mark.parametrize("outcome", ["unknown", "completed"])
def test_failed_release_keeps_debt_without_blocking_tree_teardown(domain_worker, monkeypatch, path, outcome):
    worker, members = domain_worker
    saved, calls, cached = [], [], []
    backend = "_release_domain_claimed" if worker.level == 3 else "_release_global_domain_claimed"
    original = getattr(worker, backend)
    mailboxes = tuple(shm.name for shm in (*worker._chip_shms, *worker._next_level_shms))

    def build(orch, _args, _cfg):
        saved.append(orch.allocate_domain(**_request(members, retain_after_run=path != "automatic")))
        cached.append(_tensor_args(saved[0]))
        if path == "explicit":
            saved[0].release()

    def fail_consumer(orch, _args, _cfg):
        orch.get_domain("tp")
        raise RuntimeError("consumer failed")

    def release(domain):
        calls.append(domain)
        if outcome == "completed":
            original(domain)
        raise RuntimeError("backend release failed")

    try:
        if path in ("external", "failed_consumer", "close"):
            worker.run(build)
        with monkeypatch.context() as patch:
            patch.setattr(worker, backend, release)
            with pytest.raises(
                RuntimeError, match="unconfirmed" if path == "close" else "consumer failed|release failed"
            ):
                if path == "external":
                    saved[0].release()
                elif path == "failed_consumer":
                    worker.run(fail_consumer)
                elif path == "close":
                    worker.close()
                else:
                    worker.run(build)
            if path == "close":
                assert worker._worker is None and not worker._chip_pids and not worker._next_level_pids
        domain = saved[0]
        assert domain.released and not domain.freed
        target = members[0] if worker.level == 3 else members[0][0]
        with worker._child_prov_lock, pytest.raises(ValueError, match="stale"):
            _authorize(worker, cached[0], target)
        for _ in range(2):
            with pytest.raises(RuntimeError, match="unconfirmed"):
                worker.close()
            assert not domain.freed and not worker._cleanup_journal.empty
            assert worker._worker is None and not worker._chip_pids and not worker._next_level_pids
            for name in mailboxes:
                with pytest.raises(FileNotFoundError):
                    worker_mod.SharedMemory(name=name)
        assert calls == [domain]
    finally:
        if saved and not saved[0].freed and not worker._cleanup_journal.empty:
            _close_with_domain_debt(worker)


@pytest.mark.parametrize("pending", [False, True])
def test_interrupted_free_entry_remains_close_owned(domain_worker, monkeypatch, pending):
    worker, members = domain_worker
    saved = []

    def build(orch, _args, _cfg):
        saved.append(orch.allocate_domain(**_request(members)))
        if pending:
            saved[0].release()

    def interrupt(_handle):
        raise RuntimeError("before free entry")

    boundary = "_free_domain_after_fence" if worker.level == 3 else "_free_global_domain_after_fence"
    try:
        if not pending:
            worker.run(build)
            boundary = "_release_domain_now" if worker.level == 3 else "_release_global_domain_now"
        with monkeypatch.context() as patch:
            patch.setattr(worker, boundary, interrupt)
            with pytest.raises(RuntimeError, match="before free entry"):
                worker.run(build) if pending else saved[0].release()
        assert saved[0].released and not saved[0].freed
        assert not worker._cleanup_journal.empty
        with pytest.raises(RuntimeError, match="unconfirmed"):
            worker.close()
        assert not saved[0].freed and worker._worker is None
    finally:
        if saved and not saved[0].freed and not worker._cleanup_journal.empty:
            _close_with_domain_debt(worker)


@pytest.mark.parametrize("domain_worker", [4], indirect=True)
def test_confirmed_global_free_retires_interrupted_publication(domain_worker, monkeypatch):
    worker, members = domain_worker
    domain = _retained(worker, members)
    original, calls = worker._release_global_domain_now, []

    def interrupt(handle):
        original(handle)
        calls.append(handle)
        raise KeyboardInterrupt("freed publication interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(worker, "_release_global_domain_now", interrupt)
        with pytest.raises(KeyboardInterrupt, match="publication"):
            domain.release()
    assert not domain.freed and worker._global_domain_free_results[domain.domain_id] is None
    worker.close()
    assert domain.freed and calls == [domain] and worker._cleanup_journal.empty
    worker._defer_domain_retirement(domain)
    assert worker._cleanup_journal.drive_kinds({"domain-retirement"}) is None
    assert worker._cleanup_journal.empty


@pytest.mark.parametrize("conflict", [False, True])
def test_concurrent_creation_is_serialized(domain_worker, conflict):
    worker, members = domain_worker
    entered, resume, second_entered = threading.Event(), threading.Event(), threading.Event()
    saved = []

    def first(orch, _args, _cfg):
        entered.set()
        assert resume.wait(5)
        saved.append(orch.get_or_allocate_domain(**_request(members)))

    def second(orch, _args, _cfg):
        second_entered.set()
        with pytest.raises(ValueError, match="window_size") if conflict else nullcontext():
            saved.append(orch.get_or_allocate_domain(**_request(members, window_size=8192 if conflict else 4096)))

    with _race(lambda: worker.run(first), lambda: worker.run(second), entered, resume):
        assert not second_entered.wait(0.05)
    assert worker._next_alloc_id == 1
    assert len(saved) == (1 if conflict else 2)
    assert all(domain is saved[0] for domain in saved)


@pytest.mark.parametrize("operation", ["cached", "staging"])
def test_release_waits_for_use_publication_and_staging(domain_worker, monkeypatch, operation):
    worker, members = domain_worker
    domain = _retained(worker, members)
    args = _tensor_args(domain)
    entered, resume, released = threading.Event(), threading.Event(), threading.Event()

    def pause():
        entered.set()
        assert resume.wait(5)
        assert not domain.released and not domain.freed

    def consume(_orch, _args, _cfg):
        worker._record_touched_identities(args)
        pause()

    def release():
        domain.release()
        released.set()

    original = worker._orch._copy_domain_locked

    def copy(*args, **kwargs):
        pause()
        return original(*args, **kwargs)

    if operation == "staging":
        monkeypatch.setattr(worker._orch, "_copy_domain_locked", copy)

    def action():
        if operation == "cached":
            worker.run(consume)
        else:
            worker._orch.copy_to_domain(domain, 0, b"safe")

    with _race(action, release, entered, resume):
        assert not released.wait(0.05)
    assert released.is_set() and domain.freed


def test_domain_lookup_lease_drains_before_close(domain_worker, monkeypatch):
    worker, members = domain_worker
    domain = _retained(worker, members)
    entered, resume = threading.Event(), threading.Event()
    original, found = worker._domain_serialization, []

    @contextmanager
    def pause():
        entered.set()
        assert resume.wait(5)
        with original():
            yield

    def observe_close():
        with worker._hierarchical_start_cv:
            assert worker._hierarchical_start_cv.wait_for(
                lambda: worker._lifecycle is worker_mod._Lifecycle.CLOSED, timeout=5
            )
        assert found == []
        with pytest.raises(RuntimeError, match="READY"):
            worker._get_domain(domain.ref)
        resume.set()

    monkeypatch.setattr(worker, "_domain_serialization", pause)
    with _race(lambda: found.append(worker._get_domain(domain.ref)), observe_close, entered, resume):
        worker.close()
    assert found == [domain] and domain.freed


@pytest.mark.parametrize("action", ["release", "lookup"])
@pytest.mark.parametrize("waiter", [False, True])
@pytest.mark.timeout(10)
def test_predecessor_destructor_can_access_domain(domain_worker, monkeypatch, action, waiter):
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
    entered, resume = threading.Event(), threading.Event()

    def pause(phase):
        if phase == "after_election":
            entered.set()
            assert resume.wait(5)

    handoff = first._wait_for_handoff

    def resume_waiter():
        resume.set()
        handoff()

    if waiter:
        first._wait_boundary_hook = pause
        monkeypatch.setattr(first, "_wait_for_handoff", resume_waiter)
        with _race(first.wait, lambda: worker.submit(lambda *_: None).wait(), entered, resume):
            pass
    else:
        worker.submit(lambda *_: None).wait()
    first.wait()
    assert results == ["done"] and saved[0].freed == (action == "release")


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "identity",
        "generation",
        "extent",
        "access",
        "path",
        "backend",
        "target",
        "stale",
        "dispatcher",
        "local_dispatcher",
    ],
)
def test_logical_buffer_registration_and_automatic_sidecar(mutation):
    from simpler.buffer import BackendKind, CanonicalIdentity  # noqa: PLC0415
    from simpler.task_interface import (  # noqa: PLC0415
        RemoteAddressSpace,
        RemoteBufferHandle,
        RemoteTensorRef,
        TensorArgType,
        _remote_sidecar_for,
    )

    worker = Worker(level=4, num_sub_workers=0)
    actual = wrap_vmm_window(4096, 64, b"node-own", 7, "L3")
    reference = buffer_reference(actual, owner_worker_id=3, remote_ref=(41, 2))
    assert reference.identity == actual.identity and reference.nbytes == actual.nbytes
    assert reference.backend_kind == BackendKind.REMOTE_SIDECAR and reference.access == actual.access
    assert reference.base == 0 and reference.body == b"" and reference.owner_worker_path_id == 0
    assert reference.address_space == actual.address_space and replace(reference)._remote_ref == (41, 2)
    changes = {
        "identity": {"identity": CanonicalIdentity(b"forged!!", 7, 1)},
        "generation": {"identity": CanonicalIdentity(b"node-own", 7, 2)},
        "extent": {"nbytes": 65},
        "access": {"access": AccessMode.READ},
        "path": {"owner_worker_path_id": 1},
        "backend": {"backend_kind": BackendKind.VMM_WINDOW, "body": (4096).to_bytes(8, "little")},
    }
    if mutation == "local_dispatcher":
        reference = replace(reference, _remote_ref=None)
    with worker._child_prov_lock:
        worker._record_device_alloc(reference, domain_allocation_id=7)
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
    args.add_tensor(
        replace(reference, **changes.get(mutation, {})).tensor([4], DataType.UINT8, byte_offset=4),
        TensorArgType.INPUT if mutation == "access" else TensorArgType.INOUT,
    )
    args.add_tensor(RemoteTensorRef(ordinary, offset=8, shape=(4,), dtype=DataType.UINT8))
    args.add_scalar(19)
    sidecar = _remote_sidecar_for(args)
    assert sidecar is not None
    if mutation == "stale":
        worker._drop_domain_allocs(7)
    try:
        with worker._child_prov_lock:
            with (
                pytest.raises(ValueError, match="stale|registration|registered|dispatcher")
                if mutation
                else nullcontext()
            ):
                prepared = worker._prepare_buffer_references(
                    args, None if mutation == "dispatcher" else sidecar, 4 if mutation == "target" else 3
                )
                worker._child_prov_check_dispatch_locked(args, 3, api="test")
            if mutation is None:
                assert prepared is not None
                assert worker._prepare_buffer_references(args, sidecar, 3) == prepared
                assert _remote_sidecar_for(args) == sidecar and sidecar.tensors[0] is None
                assert args.tensor(0).buffer == reference.to_descriptor() and args.tensor(0).byte_offset == 4
                assert args.scalar(0) == 19 and args.tag(0) == TensorArgType.INOUT
                ordinary_sidecar = prepared.tensors[1]
                assert ordinary_sidecar is not None and ordinary_sidecar.handle is ordinary
                generated = prepared.tensors[0]
                assert generated is not None and generated.handle is None
                assert (generated.desc.buffer_id, generated.desc.generation, generated.desc.offset) == (41, 2, 4)
                reference._remote_ref = (999, 999)
                assert worker._prepare_buffer_references(args, sidecar, 3) == prepared
                read = replace(reference, access=AccessMode.READ)
                for tag in (TensorArgType.OUTPUT_EXISTING, TensorArgType.INOUT):
                    with pytest.raises(ValueError, match="access"):
                        TaskArgs().add_tensor(read.tensor([4], DataType.UINT8), tag)
                worker._record_device_alloc(read, domain_allocation_id=7)
                read_args = TaskArgs()
                read_args.add_tensor(read.tensor([4], DataType.UINT8), TensorArgType.INPUT)
                read_prepared = worker._prepare_buffer_references(read_args, replace(sidecar, tensors=(None,)), 3)
                assert read_prepared is not None
                read_sidecar = read_prepared.tensors[0]
                assert read_sidecar is not None and read_sidecar.present
    finally:
        worker.close()


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("host_nbytes", [16, 32])
def test_domain_and_host_descriptors_cannot_alias_identity(reverse, separate, host_nbytes):
    from simpler.buffer import ImportContext, ImportRegistry, wrap_fork_inherited  # noqa: PLC0415
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
            assert materialize(ordered[:1]).tensor(0).data == (8192 if reverse else 4096)
        with pytest.raises(ValueError, match="different descriptor"):
            materialize(ordered[1:] if separate else ordered)
        assert materialize(ordered[:1]).tensor(0).data == (8192 if reverse else 4096)
    finally:
        registry.close()


@pytest.mark.parametrize("domain_worker", [3], indirect=True)
def test_registered_views_resolve_and_revalidate_after_revocation(domain_worker):
    from simpler.buffer import ImportContext, ImportRegistry  # noqa: PLC0415
    from simpler.comm_endpoints import DEVICE_AICPU  # noqa: PLC0415
    from simpler.task_interface import _materialize_chip_task_args  # noqa: PLC0415

    worker, members = domain_worker
    domain = _retained(worker, members)
    actual = domain.local_context(0).buffers["payload"]
    logical = buffer_reference(actual)
    registry = ImportRegistry(
        ImportContext(deployment=DEVICE_AICPU, device_owner_instance_id=worker._owner_instance_id)
    )
    try:
        backing = worker._reexport(logical.to_descriptor())
        args = TaskArgs()
        for offset in (4, 12):
            args.add_tensor(backing.tensor([2], DataType.FLOAT32, byte_offset=offset))
        native = _materialize_chip_task_args(args, registry)
        assert (native.tensor(0).data, native.tensor(1).data) == (actual.base + 4, actual.base + 12)
        assert args.tensor(0).buffer.identity == args.tensor(1).buffer.identity == actual.identity
        with worker._child_prov_lock, pytest.raises(ValueError, match="wrong worker|stale"):
            worker._child_prov_check_dispatch_locked(args, 0, api="test")
        with pytest.raises(ValueError, match="registered backing"):
            worker._reexport(replace(logical, nbytes=logical.nbytes + 1).to_descriptor())
        domain.release()
        with pytest.raises(ValueError, match="stale"):
            worker._reexport(logical.to_descriptor())
    finally:
        registry.close()
