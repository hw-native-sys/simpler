# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import threading
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace
from typing import cast

import pytest
import simpler.worker as worker_mod
from simpler.task_interface import CommDomainHandle
from simpler.worker import Worker


@pytest.mark.parametrize("runtime", ["host_build_graph", "tensormap_and_ringbuffer"])
@pytest.mark.parametrize("release_path", ["failed_predecessor", "queued_predecessor_release"])
def test_retained_domain_release_preserves_later_registered_use(monkeypatch, runtime, release_path):
    """An admitted overlapping use retains memory through its last user.

    Register two uses directly to exercise the cleanup boundary independently of
    concurrent admission; both runs may register while the predecessor is active.
    No Worker is initialized and no native device operation is performed.
    """
    worker = Worker(level=3, platform="a2a3", runtime=runtime, device_ids=[0, 1], num_sub_workers=0)
    domain = CommDomainHandle(
        name="shared",
        workers=(0, 1),
        allocation_id=7,
        retain_after_run=True,
        _release_fn=lambda _handle: None,
    )
    first = worker_mod._RunResources()
    successor = worker_mod._RunResources()
    releases = []
    monkeypatch.setattr(worker_mod, "_callback_frame_for", lambda _worker: object())
    monkeypatch.setattr(worker, "_release_domain_now", lambda handle: releases.append(handle))
    worker._worker = object()
    worker._live_domains[domain.name] = domain
    try:
        for resources in (first, successor):
            worker._building_run_resources = resources
            worker._use_domain(domain)
        worker._building_run_resources = None
        assert first.live_domains[domain.name] is successor.live_domains[domain.name] is domain
        first.domains_fenced = True
        if release_path == "failed_predecessor":
            worker._release_all_live_domains(first, include_retained=True)
        else:
            domain._released = True
            worker._release_domain_handle(domain, first)
            worker._execute_pending_domain_releases(first)
        assert not domain.freed, "predecessor cleanup reclaimed a later registered user's domain"
        assert releases == [], "backend reclamation ran before the successor's retirement"
        worker._retire_run_domains(first)
        assert not domain.freed
        worker._retire_run_domains(successor)
        assert domain.freed and releases == [domain]
        worker._retire_run_domains(successor)
        assert releases == [domain]
    finally:
        worker._worker = None
        worker._building_run_resources = None
        worker._live_domains.clear()


@pytest.fixture(params=["host_build_graph", "tensormap_and_ringbuffer"])
def retained_state(monkeypatch, request):
    worker = Worker(level=3, platform="a2a3", runtime=request.param, device_ids=[0, 1], num_sub_workers=0)
    domain = CommDomainHandle(
        name="shared", workers=(0, 1), allocation_id=7, retain_after_run=True, _release_fn=lambda _handle: None
    )
    releases = []
    worker._worker = object()
    worker._live_domains[domain.name] = domain
    monkeypatch.setattr(worker_mod, "_callback_frame_for", lambda _worker: object())
    monkeypatch.setattr(worker, "_release_domain_now", lambda handle: releases.append(handle))
    try:
        yield worker, domain, releases
    finally:
        worker._worker = None
        worker._building_run_resources = None
        worker._live_domains.clear()


@pytest.mark.parametrize("publication", ["handle_uses", "run_uses", "run_live"])
@pytest.mark.parametrize("committed", [False, True])
def test_interrupted_use_publication_leaves_no_retired_owner(retained_state, publication, committed):
    worker, domain, releases = retained_state
    resources = worker_mod._RunResources()

    class InterruptedDict(dict):
        armed = True

        def __setitem__(self, key, value):
            if self.armed:
                self.armed = False
                if committed:
                    super().__setitem__(key, value)
                raise KeyboardInterrupt("use publication")
            super().__setitem__(key, value)

    if publication == "handle_uses":
        domain._run_uses = InterruptedDict()
    elif publication == "run_uses":
        resources.retained_domain_uses = InterruptedDict()
    else:
        resources.live_domains = InterruptedDict()
    worker._building_run_resources = resources
    with pytest.raises(KeyboardInterrupt, match="use publication"):
        worker._use_domain(domain)
    worker._building_run_resources = None
    assert not resources.domains_fenced and releases == []
    worker._retire_run_domains(resources)
    assert resources.retained_domain_uses == {}
    assert all(use is not resources for use in domain._run_uses.values()), "retired run leaked in the domain use ledger"
    assert releases == [] and not domain.released


@pytest.mark.parametrize("wait_result", ["timeout", "context_refused"])
def test_unknown_native_fence_retains_domain_until_confirmed_retry(retained_state, monkeypatch, wait_result):
    worker, domain, releases = retained_state
    resources = worker_mod._RunResources()
    worker._building_run_resources = resources
    worker._use_domain(domain)
    worker._building_run_resources = None
    domain._released = True
    worker._release_domain_handle(domain, resources)
    native_run_releases = []
    worker._orch = SimpleNamespace(
        _release_run=lambda run_id: native_run_releases.append(run_id), _run_done=lambda _run_id: False
    )
    handle = worker_mod.RunHandle(worker, 9, (domain,), resources)
    worker._accepted_run_handles.add(handle)

    def unknown_fence(_run_id, _timeout):
        if wait_result == "context_refused":
            raise worker_mod.ExecutionContextRefused("no native completion verdict")
        return False

    monkeypatch.setattr(worker, "_wait_run_handle", unknown_fence)
    error_type = worker_mod.ExecutionContextRefused if wait_result == "context_refused" else TimeoutError
    with pytest.raises(error_type):
        handle.wait(0.01)
    assert not handle.done and not resources.domains_fenced and not resources.retired
    assert domain._run_uses[id(resources)] is resources
    assert domain in resources.pending_release_domains
    assert not domain.freed and releases == [] and native_run_releases == []
    monkeypatch.setattr(worker, "_wait_run_handle", lambda _run_id, _timeout: True)
    handle.wait(1)
    handle.wait(1)
    assert domain.freed and releases == [domain]
    assert domain._run_uses == {} and resources.retained_domain_uses == {}
    assert native_run_releases == [9]


@pytest.mark.parametrize("runtime", ["host_build_graph", "tensormap_and_ringbuffer"])
def test_close_waits_for_accepted_callback_before_domain_reclamation(monkeypatch, runtime):
    worker = Worker(level=3, num_sub_workers=1, runtime=runtime)
    domain = CommDomainHandle(
        name="shared",
        workers=(),
        allocation_id=7,
        retain_after_run=True,
        _release_fn=lambda _handle: None,
        _release_scope=worker._domain_release_scope,
    )
    entered = threading.Event()
    resume = threading.Event()
    releases = []
    errors = []
    saved = []
    monkeypatch.setattr(worker, "_release_domain_now", lambda handle: releases.append(handle))

    def build(_orch, _args, _cfg):
        worker._use_domain(domain)
        saved.append(worker._building_run_resources)
        domain.release()
        entered.set()
        assert resume.wait(10)

    def submit():
        try:
            worker.submit(build).wait(10)
        except BaseException as exc:
            errors.append(exc)

    def release_after_close_claim():
        try:
            with worker._hierarchical_start_cv:
                assert worker._hierarchical_start_cv.wait_for(
                    lambda: worker._lifecycle is worker_mod._Lifecycle.CLOSED, timeout=10
                )
            assert not saved[0].domains_fenced and releases == [] and not domain.freed
        except BaseException as exc:
            errors.append(exc)
        finally:
            resume.set()

    submitter = threading.Thread(target=submit, daemon=True)
    observer = threading.Thread(target=release_after_close_claim, daemon=True)
    try:
        worker.init()
        worker._live_domains[domain.name] = domain
        submitter.start()
        assert entered.wait(10)
        observer.start()
        worker.close()
    finally:
        resume.set()
        if submitter.ident is not None:
            submitter.join(10)
        if observer.ident is not None:
            observer.join(10)
        worker.close()
    assert not submitter.is_alive() and not observer.is_alive() and errors == []
    assert domain.freed and releases == [domain]
    assert domain._run_uses == {}


def test_pure_retained_reuse_allows_preparation_but_release_keeps_cleanup_gate(retained_state):
    worker, domain, _releases = retained_state
    resources = worker_mod._RunResources()
    worker._building_run_resources = resources
    worker._use_domain(domain)
    worker._building_run_resources = None
    handle = worker_mod.RunHandle(worker, 21, (domain,), resources)
    worker._accepted_run_handles.add(handle)
    assert worker._cleanup_bearing_predecessor() is None, "pure retained reuse unnecessarily serializes preparation"
    domain._released = True
    worker._release_domain_handle(domain, resources)
    assert worker._cleanup_bearing_predecessor() is handle


@pytest.mark.parametrize("reason", ["existing_cleanup", "nonretained_domain"])
def test_retained_reuse_does_not_erase_real_cleanup_debt(retained_state, reason):
    worker, domain, _releases = retained_state
    resources = worker_mod._RunResources()
    if reason == "existing_cleanup":
        resources.requires_ordered_cleanup = True
    else:
        domain.retain_after_run = False
    worker._building_run_resources = resources
    worker._use_domain(domain)
    worker._building_run_resources = None
    handle = worker_mod.RunHandle(worker, 22, (domain,), resources)
    worker._accepted_run_handles.add(handle)
    assert worker._cleanup_bearing_predecessor() is handle


@pytest.mark.parametrize("runtime", ["host_build_graph", "tensormap_and_ringbuffer"])
def test_domain_release_control_round_precedes_successor_allocation(monkeypatch, runtime):
    """Different collective controls must occupy every chip in the same order."""

    worker = Worker(level=3, platform="a2a3", runtime=runtime, device_ids=[0, 1], num_sub_workers=0)
    entered = [threading.Event(), threading.Event()]
    resume = threading.Event()
    allocated = threading.Event()
    attempted = threading.Event()
    errors = []

    class Controls:
        def control_release_domain(self, chip, _request):
            entered[chip].set()
            assert resume.wait(10)

        def control_alloc_domain(self, _chip, _request, _reply):
            allocated.set()

    monkeypatch.setattr(worker, "_worker", Controls())
    requests = {i: cast(SharedMemory, SimpleNamespace(name=f"request-{i}")) for i in range(2)}
    replies = {i: cast(SharedMemory, SimpleNamespace(name=f"reply-{i}")) for i in range(2)}

    def dispatch(op):
        try:
            if op == "alloc":
                attempted.set()
            worker._dispatch_control_domain(
                workers=(0, 1),
                request_shms=requests,
                reply_shms=replies if op == "alloc" else None,
                op=op,
                allocation_id=7 if op == "release" else 8,
            )
        except BaseException as exc:
            errors.append(exc)

    release = threading.Thread(target=dispatch, args=("release",))
    allocate = threading.Thread(target=dispatch, args=("alloc",))
    release.start()
    try:
        assert all(event.wait(10) for event in entered)
        allocate.start()
        assert attempted.wait(10)
        assert not allocated.wait(0.2), "successor allocation overlapped a release collective"
    finally:
        resume.set()
        release.join(10)
        if allocate.ident is not None:
            allocate.join(10)
        worker._worker = None
    assert not release.is_alive() and not allocate.is_alive()
    assert errors == []
    assert allocated.is_set()
