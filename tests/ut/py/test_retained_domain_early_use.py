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
from unittest.mock import MagicMock

import pytest
import simpler.orchestrator as orch_mod
import simpler.worker as worker_mod
from simpler.buffer import wrap_vmm_window
from simpler.orchestrator import Orchestrator
from simpler.task_interface import CommDomainHandle, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker


@pytest.fixture(params=["host_build_graph", "tensormap_and_ringbuffer"])
def cached_domain(monkeypatch, request):
    worker = Worker(level=3, platform="a2a3", runtime=request.param, device_ids=[0, 1], num_sub_workers=0)
    worker._chip_shms = [None, None]
    domain = CommDomainHandle(
        name="shared", workers=(0, 1), allocation_id=7, retain_after_run=True, _release_fn=lambda _handle: None
    )
    buffer = wrap_vmm_window(0x1000, 64, worker._owner_instance_id, buffer_id=7, owner_worker_id=0)
    with worker._child_prov_lock:
        worker._record_device_alloc(buffer, domain_allocation_id=7)
    worker._live_domains[domain.name] = domain
    resources = worker_mod._RunResources()
    worker._building_run_resources = resources
    monkeypatch.setattr(worker_mod, "_callback_frame_for", lambda _worker: object())
    monkeypatch.setattr(
        orch_mod, "_require_handle", lambda *_args, **_kwargs: (b"d" * 32, "NEXT_LEVEL", "LOCAL_CHIP", [])
    )
    monkeypatch.setattr(orch_mod, "_admit_task_submission", lambda _worker: None)
    args = TaskArgs()
    args.add_tensor(buffer.tensor([16], DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
    try:
        yield worker, domain, resources, args
    finally:
        worker._worker = None
        worker._building_run_resources = None
        worker._live_domains.clear()


@pytest.mark.parametrize("group", [False, True])
def test_cached_domain_use_is_registered_before_native_submission(cached_domain, group):
    worker, domain, resources, args = cached_domain
    native = MagicMock()

    def observe_native_submit(*_args):
        assert resources.live_domains.get(domain.name) is domain, "cached use was not registered before native submit"
        assert domain._run_uses.get(id(resources)) is resources
        acquired = worker._child_prov_lock.acquire(blocking=False)
        if acquired:
            worker._child_prov_lock.release()
        assert not acquired, "provenance must remain guarded through native submit"

    native.submit_next_level.side_effect = observe_native_submit
    native.submit_next_level_group.side_effect = observe_native_submit
    orch = Orchestrator(native, worker)
    if group:
        orch.submit_next_level_group(object(), [args], None, workers=[0])
    else:
        orch.submit_next_level(object(), args, None, worker=0)


def test_rejected_group_does_not_register_a_partial_domain_use(cached_domain):
    worker, domain, resources, args = cached_domain
    native = MagicMock()
    with pytest.raises(ValueError, match="not a live allocation"):
        Orchestrator(native, worker).submit_next_level_group(object(), [args, args], None, workers=[0, 1])
    native.submit_next_level_group.assert_not_called()
    assert resources.retained_domain_uses == {}
    assert domain._run_uses == {}


@pytest.mark.parametrize("release_path", ["direct", "retire_last_use"])
def test_backend_release_does_not_hold_domain_lock_while_waiting_for_provenance(
    cached_domain, monkeypatch, release_path
):
    worker, domain, resources, _args = cached_domain
    worker._worker = MagicMock()
    if release_path == "retire_last_use":
        worker._use_domain(domain)
    domain._released = True
    reached_provenance = threading.Event()
    errors = []
    original_drop = worker._drop_domain_allocs

    def drop(allocation_id):
        reached_provenance.set()
        original_drop(allocation_id)

    def release():
        try:
            if release_path == "retire_last_use":
                worker._retire_run_domains(resources)
            else:
                worker._free_domain_after_fence(domain)
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(worker, "_drop_domain_allocs", drop)
    monkeypatch.setattr(worker, "_release_domain_now", lambda _handle: None)
    thread = threading.Thread(target=release, daemon=True)
    try:
        with worker._child_prov_lock:
            thread.start()
            assert reached_provenance.wait(5), "release did not reach the provenance boundary"
            acquired = domain._run_use_lock.acquire(blocking=False)
            if acquired:
                domain._run_use_lock.release()
            assert acquired, "release holds domain lock while waiting for provenance: reverse lock order"
    finally:
        thread.join(5)
        assert not thread.is_alive()
    assert errors == []
    assert domain.freed


def test_failed_predecessor_revokes_new_submissions_while_successor_keeps_backing(cached_domain, monkeypatch):
    worker, domain, successor, args = cached_domain
    first = worker_mod._RunResources()
    for resources in (first, successor):
        worker._building_run_resources = resources
        worker._use_domain(domain)
    worker._building_run_resources = None
    worker._worker = MagicMock()
    freed = []
    monkeypatch.setattr(worker, "_release_domain_now", lambda handle: freed.append(handle))
    first.domains_fenced = True
    worker._release_all_live_domains(first, include_retained=True)
    assert domain.released and not domain.freed
    assert freed == []
    with worker._child_prov_lock, pytest.raises(ValueError, match="not a live allocation"):
        worker._child_prov_check_dispatch_locked(args, 0, api="submit_next_level")
    worker._retire_run_domains(first)
    worker._retire_run_domains(successor)
    assert domain.freed and freed == [domain]
