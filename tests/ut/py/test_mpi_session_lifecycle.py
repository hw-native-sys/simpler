# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MPI invalidation, admission, and mailbox lifetime boundaries."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
import simpler.mpi_group_mailbox as mailbox_mod
import simpler.worker as worker_mod
from simpler.comm_endpoints import MpiSessionState
from simpler.mpi_group_mailbox import MailboxGroupState
from simpler.orchestrator import _admit_task_submission, _callback_run, direct_control
from test_mpi_l3_group import _FakeProcess, _InertThread, _mpi_worker, _ReadyMailbox


@pytest.fixture
def active_group():
    worker = _mpi_worker()
    group = worker._mpi_l3_groups[0]
    mailbox = _ReadyMailbox(world_size=2)
    process = _FakeProcess()
    group.state = MpiSessionState.ACTIVE
    group.mailbox = mailbox
    group.process = process
    try:
        yield worker, group, mailbox, process
    finally:
        group.process = None
        group.monitor_thread = None
        group.native_attached = False
        worker.close()


@pytest.fixture
def activation_group(monkeypatch):
    worker = _mpi_worker()
    group = worker._mpi_l3_groups[0]
    mailbox = _ReadyMailbox(world_size=2)
    process = _FakeProcess()
    native = MagicMock()
    worker._worker = native
    monkeypatch.setattr(mailbox_mod.MpiGroupMailbox, "create", lambda **_kwargs: mailbox)
    monkeypatch.setattr(worker_mod.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(worker_mod.threading, "Thread", _InertThread)
    try:
        yield worker, group, mailbox, process, native
    finally:
        group.process = None
        group.monitor_thread = None
        native.close()
        group.native_attached = False
        worker._worker = None
        worker.close()


def _expire(worker, group, process, cause, **changes):
    values = dict(
        expected_process=process,
        expected_session_ids=tuple(rank.session_id for rank in group.ranks),
        kind=worker_mod._MpiSessionFailureKind.PROCESS_EXIT,
    )
    return worker._expire_mpi_group_session(group, cause=cause, **(values | changes))


@contextmanager
def _mapping_reader(group):
    held = threading.Event()
    release = threading.Event()
    errors = []

    def read():
        try:
            with group.mailbox_lock:
                held.set()
                if not release.wait(timeout=5.0):
                    raise AssertionError("mapping reader was not released within the test budget")
        except BaseException as error:
            errors.append(error)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        assert held.wait(timeout=2.0)
        yield
    finally:
        release.set()
        reader.join(timeout=5.0)
        assert not reader.is_alive()
        assert not errors, errors


def test_expiry_publishes_terminal_first_cause_and_registry_invalidation(active_group, monkeypatch):
    worker, group, mailbox, process = active_group
    registry = worker._get_endpoint_registry()
    worker._region_access_service = object()
    epoch = worker._endpoint_registry_epoch
    cause = RuntimeError("exit 17")
    notification = RuntimeError("mapping notification failed")
    notify = MagicMock(side_effect=notification)
    monkeypatch.setattr(mailbox, "mark_terminal", notify)
    assert _expire(worker, group, process, cause)
    assert group.state is MpiSessionState.TERMINAL
    assert group.failure_cause is cause
    assert group.failure_kind is worker_mod._MpiSessionFailureKind.PROCESS_EXIT
    assert worker._mpi_session_failure.__cause__ is cause
    assert worker._endpoint_registry_epoch == epoch + 1
    assert worker._endpoint_registry is None
    assert worker._region_access_service is None
    assert registry.registry_epoch == epoch
    assert group.notification_error is notification
    assert not _expire(worker, group, process, RuntimeError("late notification"))
    assert worker._endpoint_registry_epoch == epoch + 1
    assert group.failure_cause is cause
    notify.assert_called_once_with(str(cause))
    with pytest.raises(RuntimeError, match="MPI") as failure:
        worker._require_no_mpi_session_failure("test")
    assert failure.value.__cause__ is worker._mpi_session_failure


def test_stale_failure_event_cannot_expire_current_group(active_group):
    worker, group, mailbox, process = active_group
    epoch = worker._endpoint_registry_epoch
    assert not _expire(worker, group, process, RuntimeError("stale event"), expected_process=_FakeProcess())
    assert group.state is MpiSessionState.ACTIVE
    assert worker._mpi_session_failure is None
    assert worker._endpoint_registry_epoch == epoch
    assert mailbox.group_state is MailboxGroupState.READY


def test_ready_mailbox_cannot_hide_a_process_exit_before_attach(activation_group):
    worker, group, _mailbox, process, native = activation_group
    process.returncode = 17
    with pytest.raises(RuntimeError, match="exited before READY"):
        worker._activate_mpirun_worker_groups(worker_mod.time.monotonic() + 5.0)
    assert group.state is MpiSessionState.TERMINAL
    assert group.failure_kind is worker_mod._MpiSessionFailureKind.STARTUP_FAILURE
    assert not group.native_attached
    native.add_mpi_group_mailbox.assert_not_called()


def test_failure_during_attach_cannot_be_overwritten_by_active(activation_group):
    worker, group, _mailbox, process, native = activation_group
    cause = RuntimeError("exit during attach")

    def exit_during_attach(*_args):
        assert group.state is MpiSessionState.STARTING
        assert group.native_attached
        return _expire(worker, group, process, cause)

    native.add_mpi_group_mailbox.side_effect = exit_during_attach
    with pytest.raises(RuntimeError, match="MPI"):
        worker._activate_mpirun_worker_groups(worker_mod.time.monotonic() + 5.0)
    assert group.state is MpiSessionState.TERMINAL
    assert group.failure_cause is cause
    assert group.failure_kind is worker_mod._MpiSessionFailureKind.PROCESS_EXIT
    assert group.native_attached


@pytest.mark.parametrize(
    ("budget", "expected_waits", "expected_join"),
    [
        (100.0, [5.0, 2.0, 2.0], 1.0),
        (4.0, [0.0, 1.0, 2.0], 1.0),
        (1.0, [0.0, 0.0, 1.0], 0.0),
        (0.0, [0.0, 0.0, 0.0], 0.0),
    ],
    ids=["capped", "short", "reap-only", "exhausted"],
)
def test_mpi_close_reserves_time_for_signal_escalation_and_monitor(
    active_group, monkeypatch, budget, expected_waits, expected_join
):
    worker, group, mailbox, _process = active_group
    now = [100.0]
    wait_budgets = []
    process = MagicMock(pid=4242, returncode=None)
    process.poll.side_effect = lambda: process.returncode
    monitor = MagicMock()
    monitor.is_alive.return_value = True
    send_signal = MagicMock()

    def wait(*, timeout):
        assert timeout >= 0
        assert not mailbox.closed
        wait_budgets.append(timeout)
        now[0] += timeout
        if len(wait_budgets) < 3:
            raise worker_mod.subprocess.TimeoutExpired("mpirun", timeout)
        process.returncode = -worker_mod.signal.SIGKILL
        return process.returncode

    def join(*, timeout):
        assert timeout >= 0
        assert not mailbox.closed
        now[0] += timeout
        monitor.is_alive.return_value = False

    process.wait.side_effect = wait
    monitor.join.side_effect = join
    group.process = process
    group.monitor_thread = monitor
    monkeypatch.setattr(worker_mod, "_monotonic", lambda: now[0])
    monkeypatch.setattr(worker_mod.os, "killpg", send_signal)

    assert not worker._close_mpirun_group(group, deadline=now[0] + budget)
    assert wait_budgets == pytest.approx(expected_waits)
    assert send_signal.call_args_list == [
        ((process.pid, worker_mod.signal.SIGTERM),),
        ((process.pid, worker_mod.signal.SIGKILL),),
    ]
    monitor.join.assert_called_once_with(timeout=pytest.approx(expected_join))
    assert now[0] - 100.0 <= budget
    assert group.process is None and group.monitor_thread is None
    assert group.mailbox is None and mailbox.closed


def test_mpi_groups_keep_one_deadline_despite_dispatch_delay(active_group, monkeypatch):
    worker, group, _mailbox, _process = active_group
    groups = [group, object()]
    now = [100.0]
    observations = []

    def close_group(current, *, deadline):
        now[0] += 7.0
        observations.append((current, deadline, max(0.0, deadline - now[0])))
        return []

    monkeypatch.setattr(worker_mod, "_monotonic", lambda: now[0])
    monkeypatch.setattr(worker, "_mpi_l3_groups", groups)
    monkeypatch.setattr(worker, "_close_mpirun_group", close_group)
    worker._close_mpirun_groups(timeout_s=10.0)
    assert observations == [(groups[1], 110.0, 3.0), (groups[0], 110.0, 0.0)]


def test_busy_mailbox_is_retained_until_mapping_guard_is_released(active_group):
    worker, group, mailbox, _process = active_group
    group.process = None
    with _mapping_reader(group):
        assert _expire(worker, group, None, RuntimeError("exit while mapping is busy"))
        assert group.state is MpiSessionState.TERMINAL
        assert worker._mpi_session_failure is not None
        assert mailbox.group_state is MailboxGroupState.READY
        failures = worker._close_mpirun_group(group, deadline=worker_mod._monotonic())
        assert failures
        assert group.mailbox is mailbox
        assert not mailbox.closed
    assert not worker._close_mpirun_group(group, deadline=worker_mod._monotonic() + 0.1)
    assert mailbox.closed


def test_close_joins_monitor_without_holding_mapping_guard(active_group, monkeypatch):
    worker, group, mailbox, _process = active_group
    group.process = None
    attempted = threading.Event()
    joining = threading.Event()
    errors = []
    results = []

    def monitor_body():
        try:
            attempted.set()
            with group.mailbox_lock:
                assert not mailbox.closed
        except BaseException as error:
            errors.append(error)

    monitor = threading.Thread(target=monitor_body, daemon=True)
    original_join = monitor.join

    def observed_join(timeout=None):
        joining.set()
        original_join(timeout=timeout)

    monkeypatch.setattr(monitor, "join", observed_join)
    group.monitor_thread = monitor

    def close_group():
        try:
            results.append(worker._close_mpirun_group(group, deadline=worker_mod._monotonic() + 5.0))
        except BaseException as error:
            errors.append(error)

    closer = threading.Thread(target=close_group, daemon=True)
    group.mailbox_lock.acquire()
    try:
        monitor.start()
        assert attempted.wait(timeout=2.0)
        closer.start()
        assert joining.wait(timeout=2.0)
    finally:
        group.mailbox_lock.release()
        original_join(timeout=5.0)
        if closer.ident is not None:
            closer.join(timeout=5.0)
    assert not monitor.is_alive() and not closer.is_alive()
    assert not errors
    assert results == [[]]
    assert mailbox.closed


def test_native_close_failure_keeps_attachment_and_journal_until_retry(active_group):
    worker, group, mailbox, _process = active_group
    group.process = None
    group.native_attached = True
    native = MagicMock()
    native.close.side_effect = [RuntimeError("native close failed"), None, None]
    worker._worker = native
    worker._lifecycle = worker_mod._Lifecycle.READY
    worker._mpi_native_init_incomplete = True
    with pytest.raises(RuntimeError, match="native close failed"):
        worker.close()
    assert group.native_attached
    assert group.mailbox is mailbox
    assert not mailbox.closed
    assert worker._worker is native
    assert not worker._cleanup_journal.empty
    try:
        with pytest.raises(RuntimeError, match="release is unproven"):
            worker.close()
        assert group.native_attached and not mailbox.closed
        assert worker._worker is native and not worker._cleanup_journal.empty
    finally:
        worker._mpi_native_init_incomplete = False
    worker.close()
    assert not group.native_attached
    assert mailbox.closed
    assert worker._worker is None
    assert worker._cleanup_journal.empty


@pytest.mark.parametrize("boundary", ["submit", "control"])
def test_waiting_operation_rechecks_mpi_failure_before_dispatch(active_group, monkeypatch, boundary):
    worker, group, _mailbox, process = active_group
    dispatch = MagicMock()
    monkeypatch.setattr(worker, "_submit_l3_locked", dispatch)

    @contextmanager
    def admission_after_failure():
        assert _expire(worker, group, process, RuntimeError("failure while waiting"))
        yield

    if boundary == "submit":
        monkeypatch.setattr(worker._submit_mu, "exclusive", admission_after_failure)
        with pytest.raises(RuntimeError, match="MPI"):
            worker._submit_locked(lambda *_args: None, None, None)
    else:
        monkeypatch.setattr(worker._submit_mu, "shared", admission_after_failure)
        with pytest.raises(RuntimeError, match="MPI"), worker._control_reservation("test"):
            dispatch()
    dispatch.assert_not_called()


@pytest.mark.parametrize("boundary", ["begin", "callback"])
def test_mpi_failure_cancels_an_accepted_run_without_publishing_success(active_group, boundary):
    worker, group, _mailbox, process = active_group
    worker._lifecycle = worker_mod._Lifecycle.READY
    worker._worker = MagicMock()
    native = MagicMock()
    worker._orch = native

    def begin():
        if boundary == "begin":
            assert _expire(worker, group, process, RuntimeError("exit during begin"))
        return 23

    def cancel(run_id, _error):
        assert run_id == 23
        assert any(handle._run_id == run_id for handle in worker._accepted_run_handles)

    native._begin_run.side_effect = begin
    native._fail_run_submission.side_effect = cancel
    callback = MagicMock(side_effect=lambda *_args: _expire(worker, group, process, RuntimeError("exit in graph")))
    with pytest.raises(RuntimeError, match="MPI"):
        worker.submit(callback)
    assert callback.call_count == (boundary == "callback")
    native._fail_run_submission.assert_called_once()
    native._close_run_submission.assert_not_called()
    native._release_run.assert_called_once_with(23)
    assert not worker._accepted_run_handles
    assert worker._building_run_resources is None
    assert worker._active_ops == 0


def test_control_wait_reentry_and_task_submission_refuse_but_cleanup_remains_available(active_group):
    worker, group, _mailbox, process = active_group
    native = MagicMock()
    native.await_run_admission.side_effect = lambda _run_id: _expire(
        worker, group, process, RuntimeError("exit during FIFO admission")
    )
    with worker._control_reservation("outer"):
        with _callback_run(23, worker):
            with pytest.raises(RuntimeError, match="MPI"), direct_control(worker, native, "test"):
                pytest.fail("control body executed after MPI failure")
            native.await_run_admission.assert_called_once_with(23)
            with pytest.raises(RuntimeError, match="MPI"):
                _admit_task_submission(worker)
        with pytest.raises(RuntimeError, match="MPI"), worker._control_reservation("inner"):
            pytest.fail("reentrant control bypassed MPI failure")
    group.process = None
    assert not worker._close_mpirun_group(group, deadline=worker_mod._monotonic() + 0.1)
