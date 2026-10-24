# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Group epochs, registry invalidation and unchanged ordinary admission."""

from __future__ import annotations

import dataclasses
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import simpler.mpi_group_mailbox as mailbox_mod
import simpler.worker as worker_mod
from simpler import comm_endpoints as ce
from simpler.comm_region import (
    MaterializationContext,
    MaterializationRefusal,
    RefusalReason,
    _validate_registry_matches_worker,
)
from simpler.orchestrator import _admit_task_submission, _callback_run, direct_control
from test_mpi_l3_group import _FakeProcess, _InertThread, _mpi_worker, _ReadyMailbox


@pytest.fixture
def session():
    worker = _mpi_worker()
    group = worker._mpi_l3_groups[0]
    process = _FakeProcess()
    mailbox = _ReadyMailbox(world_size=len(group.ranks))
    group.process = cast(Any, process)
    group.mailbox = mailbox
    group.session_active = True
    worker._lifecycle = worker_mod._Lifecycle.READY
    try:
        yield worker, group, process, mailbox
    finally:
        group.process = None
        group.monitor_thread = None
        worker._worker = None
        worker.close()


def test_failed_registration_preserves_published_groups(monkeypatch):
    worker = _mpi_worker()
    group = worker._mpi_l3_groups[0]
    ids = tuple(worker._mpi_worker_ids)
    ranks = dict(worker._mpi_rank_by_worker_id)
    registry = worker._get_endpoint_registry()
    monkeypatch.setattr(worker, "_new_remote_session_id", lambda: group.ranks[0].session_id)
    try:
        with pytest.raises(ValueError, match="already registered"):
            worker.add_mpirun_worker_group(group.spec)
        assert worker._mpi_l3_groups == [group]
        assert tuple(worker._mpi_worker_ids) == ids
        assert worker._mpi_rank_by_worker_id == ranks
        assert worker._get_endpoint_registry() is registry
    finally:
        worker.close()


def test_registry_view_rejects_old_session_with_current_directory_version(session):
    worker, group, _process, _mailbox = session
    registry = worker._get_endpoint_registry()
    views = [registry.mpi_session_for_endpoint(record) for record in registry.records]
    assert {view.session_epoch for view in views if view is not None} == {1}
    worker._validate_endpoint_registry(registry, registry.records)
    worker._invalidate_endpoint_registry()
    current = worker._get_endpoint_registry()
    assert group.session_epoch == 1
    with pytest.raises(ce.EndpointResolveError) as old_directory:
        worker._validate_endpoint_registry(registry, registry.records)
    assert old_directory.value.reason is ce.EndpointResolveReason.REGISTRY_MISMATCH

    layout = ce.RegionLayoutSpec(payload_bytes=64, counter_bytes=0)
    provider = ce.at("L4/L3[0]/L2[0]", ce.DEVICE_AICPU)
    resolved = current.resolve_region_spec([ce.at("L4", ce.HOST_CPU), provider], ce.SingleOwner(provider=provider))
    plan = ce.BackendResolver(current, worker._get_region_access_service()).plan(resolved, layout)
    assert isinstance(plan, ce.BackendPlan)

    # A fault-injected generation change isolates epoch checking from directory invalidation.
    group.session_epoch += 1
    with pytest.raises(ce.EndpointResolveError) as old_session:
        worker._validate_endpoint_registry(current, current.records)
    assert old_session.value.reason is ce.EndpointResolveReason.MPI_SESSION_UNAVAILABLE
    ctx = MaterializationContext(
        worker=worker,
        registry=current,
        plan=plan,
        layout=layout,
    )
    with pytest.raises(MaterializationRefusal) as refusal:
        _validate_registry_matches_worker(ctx)
    assert refusal.value.reason is RefusalReason.MPI_SESSION_UNAVAILABLE

    snapshot = worker._endpoint_topology_snapshot()
    entries = []
    for entry in snapshot.entries:
        view = entry.mpi_session
        if view is not None and view.rank == group.ranks[-1].rank:
            view = dataclasses.replace(view, session_epoch=1)
            entry = dataclasses.replace(entry, mpi_session=view)
        entries.append(entry)
    with pytest.raises(ValueError, match="inconsistent epoch"):
        ce.EndpointRegistry.from_snapshot(dataclasses.replace(snapshot, entries=tuple(entries)), registry_epoch=0)
    view = next(view for view in views if view is not None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cast(Any, view).session_epoch = 99
    with pytest.raises(ValueError, match="session_epoch"):
        dataclasses.replace(view, session_epoch=0)


def test_failure_revokes_once_and_refuses_failed_group(session):
    worker, group, process, _mailbox = session
    registry = worker._get_endpoint_registry()
    cause = RuntimeError("observed MPI exit")
    assert worker._expire_mpi_session(group, process, 1, cause)
    assert not worker._expire_mpi_session(group, process, 1, RuntimeError("duplicate"))
    assert group.session_epoch == 2
    assert group.session_failure is cause
    assert not group.session_active
    with worker._operation_lease("test", requires_execution_context=False):
        pass
    with pytest.raises(ce.EndpointResolveError):
        worker._validate_endpoint_registry(registry, registry.records)
    current = worker._get_endpoint_registry()
    selected = current.resolve_region_spec([ce.at("L4/L3[0]", ce.HOST_CPU)], ce.SingleOwner())
    with pytest.raises(ce.EndpointResolveError) as unavailable:
        ce.BackendResolver(current, ce.StaticRegionAccessService({})).plan(
            selected, ce.RegionLayoutSpec(payload_bytes=64, counter_bytes=0)
        )
    assert unavailable.value.reason is ce.EndpointResolveReason.MPI_SESSION_UNAVAILABLE


@pytest.mark.parametrize("identity_field", ["session_instance_id", "registry_epoch"])
def test_registry_rejects_mpi_record_from_other_registry(session, identity_field):
    worker, _group, _process, _mailbox = session
    registry = worker._get_endpoint_registry()
    with pytest.raises(ValueError, match="registry identity"):
        ce.EndpointRegistry(
            root_level=registry.root_level,
            session_instance_id=b"x" * 16 if identity_field == "session_instance_id" else registry.session_instance_id,
            registry_epoch=registry.registry_epoch + (identity_field == "registry_epoch"),
            records=registry.records,
            mpi_sessions={
                record.endpoint_id: view
                for record in registry.records
                if (view := registry.mpi_session_for_endpoint(record)) is not None
            },
        )


def test_cleanup_mailbox_lock_timeout_retains_resources_for_retry(session, tmp_path):
    worker, group, _process, mailbox = session
    group.process = None
    ready_dir = tmp_path / "mpi-ready"
    ready_dir.mkdir()
    group.ready_dir = str(ready_dir)
    group.manifest_path = str(ready_dir / "group.json")
    group.mailbox_lock.acquire()
    try:
        failures = worker._close_mpirun_group(group, timeout_s=0.0)
        assert any("mailbox mapping is still in use" in failure for failure in failures)
        assert group.mailbox is mailbox
        assert not mailbox.closed
        assert group.ready_dir == str(ready_dir) and ready_dir.exists()
        assert group.manifest_path is not None
    finally:
        group.mailbox_lock.release()
    assert not worker._close_mpirun_group(group, timeout_s=0.0)
    assert mailbox.closed
    assert not ready_dir.exists()


def test_observation_distinguishes_busy_mailbox_from_failed_channel(session):
    worker, group, _process, mailbox = session
    registry = worker._get_endpoint_registry()
    group.mailbox_lock = MagicMock(wraps=group.mailbox_lock)
    group.mailbox_lock.acquire()
    group.mailbox_lock.reset_mock()
    try:
        with pytest.raises(ce.EndpointResolveError, match="busy"):
            worker._observe_mpi_sessions(registry, registry.records)
    finally:
        group.mailbox_lock.release()
    assert group.mailbox_lock.acquire.call_args.kwargs.get("timeout", 0) > 0
    assert group.session_epoch == 1 and group.session_failure is None
    worker._observe_mpi_sessions(registry, registry.records)
    assert group.session_epoch == 1 and group.session_failure is None
    mailbox.mark_terminal("channel failed")
    worker._observe_mpi_sessions(registry, registry.records)
    assert group.session_epoch == 2 and group.session_failure is not None
    with pytest.raises(ce.EndpointResolveError):
        worker._validate_endpoint_registry(registry, registry.records)


@pytest.mark.parametrize("stale", ["process", "epoch"])
def test_delayed_event_cannot_revoke_current_session(session, stale):
    worker, group, process, mailbox = session
    epoch = worker._endpoint_registry_epoch
    assert not worker._expire_mpi_session(
        group,
        process if stale == "epoch" else _FakeProcess(),
        0 if stale == "epoch" else 1,
        RuntimeError("late event"),
    )
    assert group.session_epoch == 1 and group.session_active
    assert group.session_failure is None
    assert worker._endpoint_registry_epoch == epoch
    assert mailbox.group_state is mailbox_mod.MailboxGroupState.READY


def test_monitor_uses_captured_generation_after_wait(session):
    worker, group, process, _mailbox = session

    def exit_after_generation_change():
        group.session_epoch += 1
        return 17

    process.wait = exit_after_generation_change
    worker._monitor_mpirun_group(group, process, 1)
    assert group.session_failure is None
    assert group.session_epoch == 2
    assert group.session_active


def test_group_failure_preserves_main_admission_and_healthy_group_views():
    worker = _mpi_worker()
    group = worker._mpi_l3_groups[0]
    worker.add_mpirun_worker_group(group.spec)
    healthy = worker._mpi_l3_groups[1]
    for current_group in worker._mpi_l3_groups:
        current_group.process = cast(Any, _FakeProcess())
        current_group.mailbox = _ReadyMailbox(world_size=len(current_group.ranks))
        current_group.session_active = True
    worker._lifecycle = worker_mod._Lifecycle.READY
    registry = worker._get_endpoint_registry()
    native = MagicMock()
    native.await_run_admission.side_effect = lambda _run_id: worker._expire_mpi_session(
        group, group.process, 1, RuntimeError("exit while waiting")
    )
    try:
        with _callback_run(7, worker):
            with direct_control(worker, native, "test"):
                pass
            _admit_task_submission(worker)
        native.await_run_admission.assert_called_once_with(7)
        with worker._operation_lease("test", requires_execution_context=False):
            pass
        with worker._control_reservation("test"):
            pass
        assert group.session_epoch == 2 and healthy.session_epoch == 1
        with pytest.raises(ce.EndpointResolveError) as stale:
            worker._validate_endpoint_registry(registry, ())
        assert stale.value.reason is ce.EndpointResolveReason.REGISTRY_MISMATCH

        current = worker._get_endpoint_registry()
        selected = current.resolve_region_spec(
            [ce.at(f"L4/L3[{healthy.ranks[0].worker_id}]", ce.HOST_CPU)], ce.SingleOwner()
        )
        layout = ce.RegionLayoutSpec(payload_bytes=64, counter_bytes=0)
        plan = worker._plan_current_region(current, selected, layout)
        assert isinstance(plan, ce.BackendPlan)
        _validate_registry_matches_worker(MaterializationContext(worker, current, plan, layout))
        failed = current.resolve_region_spec([ce.at("L4/L3[0]", ce.HOST_CPU)], ce.SingleOwner())
        with pytest.raises(ce.EndpointResolveError) as unavailable:
            worker._plan_current_region(current, failed, layout)
        assert unavailable.value.reason is ce.EndpointResolveReason.MPI_SESSION_UNAVAILABLE
        with pytest.raises(ce.EndpointResolveError):
            worker._validate_endpoint_registry(current, (*selected.members, *failed.members))
        with worker._hierarchical_start_cv:
            with pytest.raises(RuntimeError, match="MPI group"):
                worker._require_mpi_startup_locked()
    finally:
        for current_group in worker._mpi_l3_groups:
            current_group.process = None
            current_group.monitor_thread = None
        worker._worker = None
        worker.close()


def test_retirement_is_idempotent_and_allows_cleanup(session):
    worker, group, _process, mailbox = session
    worker._mark_mpirun_groups_closing([group])
    worker._mark_mpirun_groups_closing([group])
    assert group.session_epoch == 2
    assert not group.session_active
    assert group.session_failure is None
    group.process = None
    assert not worker._close_mpirun_group(group, timeout_s=0.0)
    assert mailbox.closed
    assert group.session_epoch == 2


@pytest.mark.parametrize("exit_point", ["before_attach", "during_attach"])
def test_activation_cannot_publish_an_expired_session(monkeypatch, exit_point):
    worker = _mpi_worker()
    group = worker._mpi_l3_groups[0]
    native = MagicMock()
    worker._worker = native
    process = _FakeProcess()
    mailbox = _ReadyMailbox(world_size=len(group.ranks))
    monkeypatch.setattr(mailbox_mod.MpiGroupMailbox, "create", lambda **_kwargs: mailbox)
    monkeypatch.setattr(worker_mod.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(worker_mod.threading, "Thread", _InertThread)
    if exit_point == "before_attach":
        process.returncode = 17
    else:
        native.add_mpi_group_mailbox.side_effect = lambda *_args: worker._expire_mpi_session(
            group, group.process, group.session_epoch, RuntimeError("exit during attach")
        )
    try:
        with pytest.raises(RuntimeError, match="MPI"):
            worker._activate_mpirun_worker_groups(worker_mod.time.monotonic() + 5.0)
        assert group.session_epoch == 2
        assert not group.session_active
        assert group.session_failure is not None
        if exit_point == "before_attach":
            native.add_mpi_group_mailbox.assert_not_called()
    finally:
        group.process = None
        group.monitor_thread = None
        worker._worker = None
        worker.close()
