# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MPI L3 group activation and transport-isolation tests."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import simpler.mpi_group_mailbox as mailbox_mod
import simpler.worker as worker_mod
from simpler import mpi_l3_session
from simpler.mpi_group_mailbox import MAILBOX_SIZE, MailboxGroupState
from simpler.remote_l3_protocol import ControlName
from simpler.worker import MpiL3GroupSpec, Worker


class _ReadyMailbox:
    def __init__(self, world_size: int) -> None:
        self.world_size = int(world_size)
        self.group_state = MailboxGroupState.READY
        self.address = 0x12340000
        self.closed = False
        self.terminal_messages: list[str] = []

    def manifest(self):
        return {
            "name": "pytest-mpi-mailbox",
            "mailbox_bytes": MAILBOX_SIZE,
            "world_size": self.world_size,
        }

    def terminal_reason(self):
        return self.terminal_messages[-1] if self.terminal_messages else ""

    def mark_terminal(self, message):
        self.group_state = MailboxGroupState.TERMINAL
        self.terminal_messages.append(str(message))

    def close(self, *, unlink=False):
        assert unlink
        self.closed = True


class _FakeProcess:
    pid = 4242
    returncode = None

    def poll(self):
        return self.returncode


class _InertThread:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def start(self):
        return None

    def join(self, timeout=None):
        return None

    def is_alive(self):
        return False


class _ControlledExitProcess(_FakeProcess):
    def __init__(self) -> None:
        self.wait_entered = threading.Event()
        self.exit_requested = threading.Event()

    def wait(self, timeout=None):
        self.wait_entered.set()
        if not self.exit_requested.wait(timeout=5.0 if timeout is None else timeout):
            raise AssertionError("MPI process exit was not released within the test budget")
        return self.returncode

    def exit(self) -> None:
        self.returncode = 17
        self.exit_requested.set()


@pytest.fixture
def observed_mpi_exit() -> Iterator[tuple[Worker, _ReadyMailbox, _ControlledExitProcess, threading.Thread]]:
    worker = _mpi_worker()
    mailbox = _ReadyMailbox(world_size=2)
    process = _ControlledExitProcess()
    group = worker._mpi_l3_groups[0]
    group.mailbox = mailbox
    group.process = cast(Any, process)
    group.state = worker_mod.MpiSessionState.ACTIVE
    worker._lifecycle = worker_mod._Lifecycle.READY
    monitor_errors: list[BaseException] = []

    def monitor() -> None:
        try:
            worker._monitor_mpirun_group(group, cast(Any, process))
        except BaseException as error:
            monitor_errors.append(error)

    thread = threading.Thread(target=monitor, daemon=True)
    group.monitor_thread = thread
    thread.start()
    try:
        assert process.wait_entered.wait(timeout=5.0)
        yield worker, mailbox, process, thread
    finally:
        process.exit()
        thread.join(timeout=5.0)
        assert not thread.is_alive(), "MPI monitor did not finish within the test budget"
        group.process = None
        group.monitor_thread = None
        group.mailbox = None
        mailbox.close(unlink=True)
        worker.close()
        assert not monitor_errors, monitor_errors


def _mpi_worker() -> Worker:
    worker = Worker(level=4, num_sub_workers=0)
    worker.add_mpirun_worker_group(
        MpiL3GroupSpec(
            hosts=("127.0.0.1", "127.0.0.1"),
            platform="a2a3sim",
            device_ids_by_rank=((0,), (1,)),
            global_device_ranks_by_rank=((0,), (1,)),
        )
    )
    return worker


def test_mpi_activation_attaches_only_mailbox_endpoint(monkeypatch):
    worker = _mpi_worker()
    mailbox = _ReadyMailbox(world_size=2)
    native_worker = MagicMock()
    worker._worker = native_worker
    monkeypatch.setattr(mailbox_mod.MpiGroupMailbox, "create", lambda **_kwargs: mailbox)
    monkeypatch.setattr(worker_mod.subprocess, "Popen", lambda *_args, **_kwargs: _FakeProcess())
    monkeypatch.setattr(worker_mod.threading, "Thread", _InertThread)
    try:
        worker._activate_mpirun_worker_groups(worker_mod.time.monotonic() + 10.0)
        native_worker.add_mpi_group_mailbox.assert_called_once()
        native_worker.add_remote_l3_socket.assert_not_called()
        group = worker._mpi_l3_groups[0]
        assert group.state is worker_mod.MpiSessionState.ACTIVE
        assert group.manifest_path is not None
        with open(group.manifest_path, encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        assert manifest["mailbox"]["name"] == "pytest-mpi-mailbox"
        assert "command_port" not in json.dumps(manifest)
        assert "health_port" not in json.dumps(manifest)
        assert "listen_host" not in json.dumps(manifest)
        assert "connect_host" not in json.dumps(manifest)
    finally:
        group = worker._mpi_l3_groups[0]
        group.process = None
        group.monitor_thread = None
        native_worker.close()
        group.native_attached = False
        worker._worker = None
        worker.close()
        assert mailbox.closed


def test_unexpected_mpirun_exit_marks_group_terminal():
    worker = _mpi_worker()
    mailbox = _ReadyMailbox(world_size=2)
    group = worker._mpi_l3_groups[0]
    group.mailbox = mailbox
    process = _FakeProcess()
    process.returncode = 17
    process.wait = lambda: 17
    group.process = cast(Any, process)
    group.state = worker_mod.MpiSessionState.ACTIVE
    worker._monitor_mpirun_group(group, cast(Any, process))
    try:
        assert mailbox.group_state is MailboxGroupState.TERMINAL
        assert "status 17" in mailbox.terminal_reason()
    finally:
        group.process = None
        group.mailbox = None
        worker.close()


def test_observed_mpirun_exit_rejects_new_operation_lease(observed_mpi_exit):
    worker, mailbox, process, thread = observed_mpi_exit
    registry = worker._get_endpoint_registry()
    epoch = worker._endpoint_registry_epoch
    with worker._operation_lease("mpi test"):
        assert worker._active_ops == 1
    assert worker._active_ops == 0

    process.exit()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert mailbox.group_state is MailboxGroupState.TERMINAL
    assert worker._endpoint_registry_epoch > epoch
    assert worker._endpoint_registry is not registry
    assert registry.registry_epoch == epoch
    with pytest.raises(RuntimeError, match="MPI"):
        with worker._operation_lease("mpi test"):
            pytest.fail("an observed MPI group exit admitted new work")
    assert worker._active_ops == 0


def test_failed_mpi_registration_preserves_existing_group_and_registry(monkeypatch):
    worker = _mpi_worker()
    groups = tuple(worker._mpi_l3_groups)
    worker_ids = tuple(worker._mpi_worker_ids)
    ranks = dict(worker._mpi_rank_by_worker_id)
    registry = worker._get_endpoint_registry()
    epoch = worker._endpoint_registry_epoch
    existing_session_id = groups[0].ranks[0].session_id
    unused_session_id = next(
        value for value in range(1, 4) if value not in {item.session_id for item in ranks.values()}
    )
    monkeypatch.setattr(
        worker, "_new_remote_session_id", MagicMock(side_effect=[unused_session_id, existing_session_id])
    )
    try:
        with pytest.raises(ValueError, match="distinct"):
            worker.add_mpirun_worker_group(groups[0].spec)
        assert tuple(worker._mpi_l3_groups) == groups
        assert tuple(worker._mpi_worker_ids) == worker_ids
        assert worker._mpi_rank_by_worker_id == ranks
        assert worker._endpoint_registry is registry
        assert worker._endpoint_registry_epoch == epoch
    finally:
        worker.close()


def test_successful_mpi_registration_publishes_all_members_and_invalidates_cache():
    worker = Worker(level=4, num_sub_workers=0)
    old_registry = worker._get_endpoint_registry()
    epoch = worker._endpoint_registry_epoch
    try:
        ids = worker.add_mpirun_worker_group(
            MpiL3GroupSpec(
                hosts=("127.0.0.1", "127.0.0.1"),
                platform="a2a3sim",
                device_ids_by_rank=((0,), (1,)),
            )
        )
        assert tuple(worker._mpi_worker_ids) == ids
        assert set(worker._mpi_rank_by_worker_id) == set(ids)
        group = worker._mpi_l3_groups[0]
        assert tuple(item.worker_id for item in group.ranks) == ids
        assert tuple(item.rank for item in group.ranks) == (0, 1)
        assert worker._endpoint_registry is None
        assert worker._endpoint_registry_epoch == epoch + 1
        assert old_registry.registry_epoch == epoch
    finally:
        worker.close()


def test_mpi_close_retains_mailbox_while_monitor_is_alive():
    worker = _mpi_worker()
    group = worker._mpi_l3_groups[0]
    mailbox = _ReadyMailbox(world_size=2)
    monitor = MagicMock(spec=threading.Thread)
    monitor.is_alive.return_value = True
    group.mailbox = mailbox
    group.monitor_thread = monitor
    try:
        failures = worker._close_mpirun_group(group, deadline=worker_mod._monotonic() + 0.01)
        monitor.join.assert_called_once()
        assert 0.0 <= monitor.join.call_args.kwargs["timeout"] <= 0.01
        assert not mailbox.closed
        assert group.mailbox is mailbox
        assert group.monitor_thread is monitor
        assert failures
    finally:
        group.monitor_thread = None
        group.mailbox = None
        mailbox.close(unlink=True)
        worker.close()


def test_mpi_spec_tcp_fields_are_optional_and_control_18_stays_reserved():
    spec = MpiL3GroupSpec(
        hosts=("127.0.0.1", "127.0.0.1"),
        platform="a2a3sim",
        device_ids_by_rank=((0,), (1,)),
    )
    assert spec.command_port_base is None
    assert spec.health_port_base is None
    assert spec.session_listen_hosts == ()
    assert spec.connect_hosts == ()
    assert worker_mod._CTRL_COMMITTED_DEVICE_MEMORY == 18
    assert 18 not in {int(control) for control in ControlName}


def test_mpi_rank_worker_uses_parent_topology_id_before_init(monkeypatch):
    captured = {}

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            self._topology_worker_id = None
            captured["worker"] = self

        def init(self, *args, **kwargs):
            raise RuntimeError("stop after worker construction")

        def close(self):
            pass

    class FailedStartupComm:
        def Get_rank(self):
            return 1

        def allgather(self, value):
            return [value]

        def bcast(self, value, root=0):
            return False, "stop after startup"

    monkeypatch.setattr(mpi_l3_session, "Worker", FakeWorker)
    monkeypatch.setattr(mpi_l3_session, "_install_manifest_dispatcher_registry", lambda manifest: {})
    monkeypatch.setattr(mpi_l3_session, "_install_manifest_inner_registry", lambda manifest, worker: {})

    rc = mpi_l3_session._run_group_session(
        dispatch_comm=FailedStartupComm(),
        domain_comm=None,
        group_manifest={"worker_ids": [3, 7]},
        manifest={
            "worker_id": 7,
            "platform": "a2a3sim",
            "runtime": "tensormap_and_ringbuffer",
            "device_ids": [1],
            "num_sub_workers": 0,
            "heap_ring_size": None,
            "startup_remaining_s": 1.0,
            "session_timeout_s": 1.0,
        },
    )

    assert rc == 1
    assert captured["worker"]._topology_worker_id == 7
