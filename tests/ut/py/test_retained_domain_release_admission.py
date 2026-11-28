# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Lifetime admission after a retained domain is revoked but not reclaimed."""

import threading

import pytest
import simpler.worker as worker_mod
from simpler.task_interface import CommDomainHandle
from simpler.worker import Worker


@pytest.mark.parametrize("runtime", ["host_build_graph", "tensormap_and_ringbuffer"])
def test_released_domain_retains_a_later_native_context_user(monkeypatch, runtime):
    """A's release outlives A; C can prepare while B still holds the backing."""
    worker = Worker(level=3, platform="a2a3", runtime=runtime, device_ids=[0, 1], num_sub_workers=0)
    first = worker_mod._RunResources()
    second = worker_mod._RunResources()
    third = worker_mod._RunResources()
    domain = CommDomainHandle(
        name="retiring",
        workers=(0, 1),
        allocation_id=7,
        retain_after_run=True,
        _release_fn=lambda handle: worker._release_domain_handle(handle, first),
    )
    reclaimed = []
    monkeypatch.setattr(worker_mod, "_callback_frame_for", lambda _worker: object())
    monkeypatch.setattr(worker, "_release_domain_now", lambda handle: reclaimed.append(handle))
    worker._worker = object()
    worker._live_domains[domain.name] = domain
    try:
        worker._building_run_resources = first
        worker._use_domain(domain)
        worker._building_run_resources = second
        worker._retain_native_domains(second)
        assert second.retained_domain_uses[domain.allocation_id] is domain
        assert not second.live_domains

        domain.release()
        worker._retire_run_domains(first)
        assert domain.released and not domain.freed
        assert not reclaimed

        # A has retired, so its ordered-cleanup admission barrier no longer
        # protects D. The native joined-domain table still contains D until
        # physical reclamation and a later C will conservatively borrow it.
        worker._building_run_resources = third
        worker._retain_native_domains(third)
        assert not third.live_domains
        worker._retire_run_domains(second)
        assert not domain.freed, "B reclaimed D without retaining the later native context user C"
        assert not reclaimed
        worker._retire_run_domains(third)
        assert domain.freed
        assert reclaimed == [domain]
    finally:
        worker._worker = None
        worker._building_run_resources = None
        worker._live_domains.clear()


@pytest.mark.parametrize("runtime", ["host_build_graph", "tensormap_and_ringbuffer"])
def test_admission_after_inflight_free_does_not_retain_freed_backing(monkeypatch, runtime):
    worker = Worker(level=3, platform="a2a3", runtime=runtime, device_ids=[0, 1], num_sub_workers=0)
    domain = CommDomainHandle(
        name="retiring",
        workers=(0, 1),
        allocation_id=7,
        retain_after_run=True,
        _release_fn=lambda _handle: None,
    )
    successor = worker_mod._RunResources()
    release_entered = threading.Event()
    finish_release = threading.Event()
    admission_attempted = threading.Event()
    errors = []
    underlying = worker._domain_lifetime_mu

    class ObservedLock:
        def __enter__(self):
            if threading.current_thread() is admission:
                admission_attempted.set()
            return underlying.__enter__()

        def __exit__(self, *args):
            return underlying.__exit__(*args)

    def release_backend(_domain):
        release_entered.set()
        assert finish_release.wait(5), "test did not release the backend gate"

    def release():
        try:
            assert worker._free_domain_after_fence(domain)
        except BaseException as error:
            errors.append(error)

    def admit():
        try:
            worker._retain_native_domains(successor)
        except BaseException as error:
            errors.append(error)

    admission = threading.Thread(target=admit)
    retirement = threading.Thread(target=release)
    monkeypatch.setattr(worker, "_domain_lifetime_mu", ObservedLock())
    monkeypatch.setattr(worker, "_release_domain_now", release_backend)
    worker._worker = object()
    domain._released = True
    worker._retained_domain_backings[domain.allocation_id] = domain
    try:
        retirement.start()
        assert release_entered.wait(5), "release did not reach its backend"
        admission.start()
        assert admission_attempted.wait(5), "admission did not reach its lifetime snapshot"
    finally:
        finish_release.set()
        retirement.join(5)
        if admission.ident is not None:
            admission.join(5)
        worker._worker = None
        worker._live_domains.clear()
    assert not retirement.is_alive() and not admission.is_alive()
    assert not errors
    assert domain.freed
    assert not successor.retained_domain_uses
    assert id(successor) not in domain._run_uses
    assert domain.allocation_id not in worker._retained_domain_backings
