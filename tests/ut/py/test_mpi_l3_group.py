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
from simpler import mpi_l3_session
from simpler.mpi_group_mailbox import (
    MAILBOX_SIZE,
    MailboxGroupState,
    MailboxOpcode,
    MailboxRequest,
    MailboxRequestState,
    MailboxTarget,
)
from simpler.remote_l3_protocol import ControlName, FrameHeader, FrameType, encode_frame
from simpler.session_cleanup import (
    GROUP_CLEANUP_RECORD_NAME,
    CleanupOutcome,
    ProcessExit,
    ShutdownCommand,
    decode_runner_record,
    encode_runner_record,
)
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


class _OrderedMailbox:
    """Rank-0 mailbox that records the order of the calls teardown makes on it."""

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.world_size = 1
        self.group_state = MailboxGroupState.READY
        self.request_state = MailboxRequestState.SHUTDOWN_READY
        self.closed = False

    def publish_ready(self):
        self.calls.append("publish_ready")

    def wait_request_state(self, _expected, *, timeout_s):
        return None

    def accept_request(self, *, last_sequence_id):
        self.request_state = MailboxRequestState.TASK_ACCEPTED
        return MailboxRequest(
            sequence_id=int(last_sequence_id) + 1,
            opcode=MailboxOpcode.SHUTDOWN,
            target=MailboxTarget.GROUP,
            target_rank=-1,
            payloads=(mpi_l3_session._shutdown_frame(_RANK0_MANIFEST),),
        )

    def complete_shutdown(self, *, sequence_id):
        self.calls.append("complete_shutdown")

    def publish_closed(self):
        self.calls.append("publish_closed")

    def close(self, *, unlink=False):
        self.calls.append("mailbox_close")
        self.closed = True

    def mark_terminal(self, reason):
        self.calls.append("mark_terminal")
        self.group_state = MailboxGroupState.TERMINAL

    def terminal_reason(self):
        return ""


class _Rank0Comm:
    """Single-rank stand-in for the dispatch and domain communicators."""

    def Get_rank(self):  # noqa: N802 -- mpi4py spelling
        return 0

    def Get_size(self):  # noqa: N802 -- mpi4py spelling
        return 1

    def allgather(self, value):
        return [value]

    def bcast(self, value, root=0):
        return value

    def gather(self, value, root=0):
        return [value]


_RANK0_MANIFEST = {
    "session_id": 11,
    "worker_id": 7,
    "platform": "a2a3sim",
    "runtime": "tensormap_and_ringbuffer",
    "device_ids": [0],
    "num_sub_workers": 0,
    "heap_ring_size": None,
    "startup_remaining_s": 5.0,
    "session_timeout_s": 5.0,
}


def _drive_rank0_group_session(monkeypatch, tmp_path, *, release_error):
    """Run one rank-0 group session through a SHUTDOWN and return what it left behind."""
    calls: list[str] = []
    mailbox = _OrderedMailbox(calls)

    class _RecordingWorker:
        def __init__(self, *_args, **_kwargs):
            self._topology_worker_id = None

        def init(self, *_args, **_kwargs):
            return None

        def close(self):
            calls.append("worker_close")
            if release_error is not None:
                raise release_error

    def _hello_then_exit(connection, manifest, *_rest):
        connection.sendall(
            encode_frame(FrameHeader(FrameType.HELLO, int(manifest["session_id"]), int(manifest["worker_id"]), 0), b"")
        )

    monkeypatch.setattr(mpi_l3_session, "Worker", _RecordingWorker)
    monkeypatch.setattr(mpi_l3_session, "_install_manifest_dispatcher_registry", lambda _manifest: {})
    monkeypatch.setattr(mpi_l3_session, "_install_manifest_inner_registry", lambda _manifest, _worker: {})
    monkeypatch.setattr(mpi_l3_session, "open_rank_mailbox", lambda _manifest, *, rank: mailbox)
    monkeypatch.setattr(mpi_l3_session, "_run_command_loop", _hello_then_exit)

    manifest_path = tmp_path / "group.json"
    manifest_path.write_text("{}", encoding="utf-8")
    comm = _Rank0Comm()
    error: BaseException | None = None
    rc = None
    try:
        rc = mpi_l3_session._run_group_session(
            dispatch_comm=comm,
            domain_comm=comm,
            group_manifest={"worker_ids": [7], "mailbox": {"name": "pytest"}},
            manifest=dict(_RANK0_MANIFEST),
            group_manifest_path=str(manifest_path),
        )
    except BaseException as exc:  # noqa: BLE001 -- the test asserts on what propagated
        error = exc
    record_path = tmp_path / GROUP_CLEANUP_RECORD_NAME
    record = decode_runner_record(record_path.read_bytes()) if record_path.exists() else None
    return calls, record, rc, error, mailbox


def test_rank0_publishes_closed_only_after_its_release(monkeypatch, tmp_path):
    # The two words answer different questions: SHUTDOWN_DONE belongs in the
    # dispatch loop because L4's native close polls it with the GIL held, while
    # CLOSED must stand behind a finished release.
    calls, record, rc, error, mailbox = _drive_rank0_group_session(monkeypatch, tmp_path, release_error=None)
    assert error is None
    assert rc == 0
    assert calls == ["publish_ready", "complete_shutdown", "worker_close", "publish_closed", "mailbox_close"]
    assert mailbox.closed is True
    assert record is not None
    assert record.committed is True
    assert record.rank == 0
    assert record.worker_id == 7
    assert record.command is ShutdownCommand.OBSERVED
    assert record.cleanup is CleanupOutcome.SUCCEEDED


def test_rank0_records_a_failed_release_and_still_finishes_the_mailbox(monkeypatch, tmp_path):
    calls, record, _rc, error, mailbox = _drive_rank0_group_session(
        monkeypatch, tmp_path, release_error=RuntimeError("mailbox unlink refused")
    )
    # A failed release propagates to the caller, and it reaches the record and
    # the mailbox state first: neither is stranded by the failure.
    assert isinstance(error, RuntimeError)
    assert calls == ["publish_ready", "complete_shutdown", "worker_close", "publish_closed", "mailbox_close"]
    assert mailbox.closed is True
    assert record is not None
    assert record.cleanup is CleanupOutcome.FAILED
    assert "mailbox unlink refused" in record.failures[0]


def _exited_process(returncode: int) -> Any:
    process = _FakeProcess()
    process.returncode = returncode
    process.wait = lambda timeout=None: returncode
    return process


def _stage_group_record(worker: Worker, tmp_path, *, record: bytes | None) -> Any:
    group = worker._mpi_l3_groups[0]
    ready_dir = tmp_path / "manifest"
    ready_dir.mkdir()
    if record is not None:
        (ready_dir / GROUP_CLEANUP_RECORD_NAME).write_bytes(record)
    group.ready_dir = str(ready_dir)
    group.manifest_path = str(ready_dir / "group.json")
    group.monitor_thread = None
    group.mailbox = None
    return group


def test_closing_a_group_reads_rank0_record_into_the_ledger(tmp_path):
    worker = _mpi_worker()
    group = worker._mpi_l3_groups[0]
    rank0, rank1 = group.ranks
    _stage_group_record(
        worker,
        tmp_path,
        record=encode_runner_record(
            worker_id=rank0.worker_id,
            command=ShutdownCommand.OBSERVED,
            cleanup=CleanupOutcome.SUCCEEDED,
            rank=0,
        ),
    )
    group.process = _exited_process(0)
    rows: dict[int, Any] = {}
    try:
        assert worker._close_mpirun_group(group, timeout_s=0.1, reports=rows) == []
        # Only rank 0 shares a filesystem with this launcher, so only its row
        # carries a release result.
        assert rows[rank0.worker_id].committed is True
        assert rows[rank0.worker_id].cleanup is CleanupOutcome.SUCCEEDED
        assert rows[rank1.worker_id].committed is False
        assert rows[rank1.worker_id].cleanup is CleanupOutcome.UNKNOWN
        # The command and the process are group facts and reach every row.
        for row in rows.values():
            assert row.command is ShutdownCommand.OBSERVED
            assert row.process_exit is ProcessExit.EXITED
            assert row.exit_status == 0
    finally:
        group.process = None
        worker.close()


def test_a_rank0_release_failure_is_recorded_without_failing_the_close(tmp_path):
    worker = _mpi_worker()
    group = worker._mpi_l3_groups[0]
    rank0 = group.ranks[0]
    _stage_group_record(
        worker,
        tmp_path,
        record=encode_runner_record(
            worker_id=rank0.worker_id,
            command=ShutdownCommand.OBSERVED,
            cleanup=CleanupOutcome.FAILED,
            failures=["unlink refused"],
            rank=0,
        ),
    )
    group.process = _exited_process(1)
    rows: dict[int, Any] = {}
    try:
        # The release happened in another process and no retry here changes it,
        # so it lands in the ledger rather than in this close's failures. What
        # fails a close is teardown debt this Worker owns and can retry.
        assert worker._close_mpirun_group(group, timeout_s=0.1, reports=rows) == []
        row = rows[rank0.worker_id]
        assert row.cleanup is CleanupOutcome.FAILED
        assert row.failures == ("unlink refused",)
        assert "cleanup=failed@rank0" in row.describe()
    finally:
        group.process = None
        worker.close()


def test_a_group_reaped_without_a_record_still_reports_its_exit(tmp_path):
    # The process half is this Worker own observation, so it stands even when
    # nothing was written, which is what separates a silent reap from a group
    # that is still running.
    worker = _mpi_worker()
    group = worker._mpi_l3_groups[0]
    rank0 = group.ranks[0]
    _stage_group_record(worker, tmp_path, record=None)
    group.process = _exited_process(9)
    rows: dict[int, Any] = {}
    try:
        worker._close_mpirun_group(group, timeout_s=0.1, reports=rows)
        row = rows[rank0.worker_id]
        assert row.committed is False
        assert row.command is ShutdownCommand.UNKNOWN
        assert row.cleanup is CleanupOutcome.UNKNOWN
        assert row.process_exit is ProcessExit.EXITED
        assert row.exit_status == 9
    finally:
        group.process = None
        worker.close()


def test_a_retried_close_does_not_weaken_a_row_it_already_wrote(tmp_path):
    # A retry runs with the process handle and the manifest directory already
    # dropped, so it observes strictly less; the verdict must not regress.
    worker = _mpi_worker()
    group = worker._mpi_l3_groups[0]
    rank0 = group.ranks[0]
    _stage_group_record(
        worker,
        tmp_path,
        record=encode_runner_record(
            worker_id=rank0.worker_id,
            command=ShutdownCommand.OBSERVED,
            cleanup=CleanupOutcome.SUCCEEDED,
            rank=0,
        ),
    )
    group.process = _exited_process(0)
    rows: dict[int, Any] = {}
    try:
        worker._close_mpirun_group(group, timeout_s=0.1, reports=rows)
        first = rows[rank0.worker_id]
        assert first.committed is True
        worker._close_mpirun_group(group, timeout_s=0.1, reports=rows)
        assert rows[rank0.worker_id] == first
    finally:
        group.process = None
        worker.close()
