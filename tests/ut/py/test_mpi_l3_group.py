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
from typing import Any, cast
from unittest.mock import MagicMock

import simpler.mpi_group_mailbox as mailbox_mod
import simpler.worker as worker_mod
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
            "protocol_version": 1,
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
    worker._monitor_mpirun_group(group, cast(Any, process))
    try:
        assert mailbox.group_state is MailboxGroupState.TERMINAL
        assert "status 17" in mailbox.terminal_reason()
    finally:
        group.process = None
        group.mailbox = None
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
