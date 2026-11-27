# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cancellation reaches an executing nested graph while the parent still owns its task."""

import os
import secrets
import shutil
import tempfile
import threading
import time
import weakref
from pathlib import Path
from unittest.mock import Mock

import pytest
from simpler._parent_cancel import SessionCancellation
from simpler.buffer import create_host_shared_buffer, mint_owner_instance_id
from simpler.mpi_group_mailbox import (
    MailboxGroupState,
    MailboxOpcode,
    MailboxTarget,
    MpiGroupError,
    MpiGroupMailbox,
    MpiRankError,
)
from simpler.mpi_l3_session import _group_reply_errors  # noqa: PLC2701
from simpler.remote_l3_protocol import FrameHeader, FrameType, encode_completion, encode_frame
from simpler.task_interface import TaskArgs
from simpler.worker import MpiL3GroupSpec, RemoteCallable, RemoteWorkerSpec, Worker

from tests.ut.py.test_callable_identity import _free_tcp_port, _RemoteL3Daemon  # noqa: PLC2701


def _marker(token, name):
    return Path(tempfile.gettempdir()) / f"simpler-cancel-{token}-{name}"


def _wait_until(predicate):
    deadline = time.monotonic() + 6
    while not predicate():
        if time.monotonic() >= deadline:
            raise RuntimeError("peer progress deadline expired")
        time.sleep(0.01)


def run_peer(orch, args, _config):
    token, failing = args.scalar(0), args.scalar(1)
    if failing == 1:
        _wait_until(_marker(token, "started").exists)
        raise RuntimeError("test origin 106")
    _marker(token, "started").touch()
    _wait_until(orch._o._current_run_failed_for_test)
    _marker(token, "cancelled").touch()
    if failing == 2:
        # Keep this rank in its callback past the configured group timeout.
        time.sleep(20)


def check_peer_exit(worker, nodes, remote, *, group=False, stuck_peer=False):
    token = secrets.randbits(62)
    target = RemoteCallable(__name__ + ":run_peer") if remote else run_peer
    handle = worker.register(target, workers=nodes) if remote else worker.register(target)
    worker.init()

    def parent(orch, _args, config):
        group_args = []
        for index, node in enumerate(nodes):
            args = TaskArgs()
            args.add_scalar(token)
            args.add_scalar(2 if stuck_peer and index == 0 else index)
            if group:
                group_args.append(args)
            else:
                orch.submit_next_level(handle, args, config, worker=node)
        if group:
            orch.submit_next_level_group(handle, group_args, config, workers=nodes)

    try:
        with pytest.raises(RuntimeError, match="test origin 106") as raised:
            worker.run(parent)
        assert _marker(token, "cancelled").exists()
        if stuck_peer:
            assert "timed out" in str(raised.value)
    finally:
        for name in ("started", "cancelled"):
            _marker(token, name).unlink(missing_ok=True)


def test_local_nested_running_graph_receives_parent_failure():
    with Worker(level=4, num_sub_workers=0) as worker:
        nodes = [worker.add_worker(Worker(level=3, num_sub_workers=1)) for _ in range(2)]
        check_peer_exit(worker, nodes, False)


@pytest.mark.parametrize("fail_diagnostic", [False, True], ids=["poll-first", "diagnostic-oom"])
def test_unsettled_parent_cancel_retains_running_work_and_refuses_a_successor(tmp_path, monkeypatch, fail_diagnostic):
    started, release = tmp_path / "started", tmp_path / "release"

    def hold_resource(_args):
        started.touch()
        _wait_until(release.exists)

    with Worker(level=3, num_sub_workers=1) as worker:
        sub = worker.register(hold_resource)
        worker.init()
        assert worker._orch is not None
        real_cancel = worker._orch._request_parent_cancel

        def fail_cleanup(run_id):
            real_cancel(run_id)
            raise MemoryError("injected cancellation cleanup failure")

        request = Mock(side_effect=fail_cleanup)
        monkeypatch.setattr(worker._orch, "_request_parent_cancel", request)
        backing = create_host_shared_buffer(128, mint_owner_instance_id(), 1)

        def graph(orch, _args, _cfg):
            worker._register_worker_chip_orch_comm_host_buffer(backing)
            orch.submit_sub(sub)

        # The graph owns the mapping until the RunHandle releases its keepalive.
        release_backing = weakref.finalize(graph, backing.close)
        cancel = SessionCancellation()
        try:
            with cancel.task(1, worker):
                handle = worker.submit(graph)
                del graph
                _wait_until(started.exists)
                notifier = threading.Thread(target=cancel.request, args=(1,), daemon=True)
                notifier.start()
                notifier.join(1)
                assert not notifier.is_alive()
                assert request.call_count == 2
                if fail_diagnostic:
                    native_wait = worker._orch._wait_run_for

                    def exhaust_diagnostic(run_id, timeout):
                        with pytest.raises(RuntimeError, match="peer error 108.*cleanup failed"):
                            native_wait(run_id, timeout)
                        raise MemoryError("failed to construct the cancellation diagnostic")

                    monkeypatch.setattr(worker._orch, "_wait_run_for", exhaust_diagnostic)
                    with pytest.raises(MemoryError, match="cancellation diagnostic"):
                        handle.result(timeout=2)
                else:
                    # No waiter has observed or cached the failure yet.
                    assert handle.done
                    with pytest.raises(RuntimeError, match="peer error 108.*cleanup failed"):
                        handle.result(timeout=2)
                assert handle._resources.worker_chip_orch_comm_host_buffers[backing.base] == 128
                assert release_backing.alive
                assert handle in worker._abandoned_run_handles and handle._keepalive is not None
                with pytest.raises(RuntimeError, match="no further work is admitted"):
                    worker.submit(lambda *_args: None)
        finally:
            release.touch()
    assert not release_backing.alive
    assert backing.closed


def test_tcp_running_graph_receives_parent_failure(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(Path(__file__).parent), os.getenv("PYTHONPATH", ""))))
    port = _free_tcp_port()
    daemon = _RemoteL3Daemon(port)
    try:
        daemon.await_ready()
        with Worker(level=4, num_sub_workers=0, remote_session_timeout_s=15) as worker:
            nodes = [
                worker.add_remote_worker(
                    RemoteWorkerSpec(
                        endpoint=f"127.0.0.1:{port}",
                        platform="a2a3sim",
                        device_ids=(),
                        num_sub_workers=1,
                        comm_profile="sim",
                    )
                )
                for _ in range(2)
            ]
            check_peer_exit(worker, nodes, True)
    finally:
        daemon.stop()


def mpi_controller(context):
    worker = context.create_worker(num_sub_workers=0)
    check_peer_exit(worker, [s.worker_id for s in context.topology.executors], True)


def test_mpi_group_keeps_origin_when_another_rank_loses_its_transport():
    frame = encode_frame(FrameHeader(FrameType.COMPLETION, 1, 1, 7), encode_completion(7, 106, "original 106"))
    lost = MpiRankError(0, "ConnectionError", "peer lost after cancellation")
    gathered = [(0, True, b"", lost), (1, True, frame, None)]
    errors, terminal = _group_reply_errors(gathered, origin=MpiRankError(1, "RemoteOperationError", "original 106"))
    assert terminal
    assert [error.rank for error in errors] == [1, 0]
    assert errors[0].message == "original 106"
    assert errors[1] == lost
    mailbox = MpiGroupMailbox.create(world_size=2)
    try:
        mailbox.publish_ready()
        mailbox.write_request(
            sequence_id=7,
            opcode=MailboxOpcode.TASK,
            target=MailboxTarget.PER_RANK,
            target_rank=-1,
            payloads=(b"a", b"b"),
        )
        mailbox.accept_request(last_sequence_id=0)
        mailbox.fail_request(sequence_id=7, errors=errors, terminal=terminal)
        with pytest.raises(MpiGroupError, match="original 106.*peer lost after cancellation"):
            mailbox.read_result(sequence_id=7)
        assert mailbox.group_state is MailboxGroupState.TERMINAL
    finally:
        mailbox.close(unlink=True)


@pytest.mark.parametrize("gathered", [False, True])
def test_mpirun_exit_preserves_the_active_or_gathered_diagnostic(gathered):
    mailbox = MpiGroupMailbox.create(world_size=2)
    origin = MpiRankError(1, "RemoteOperationError", "original 106")
    try:
        mailbox.publish_ready()
        mailbox.write_request(
            sequence_id=7, opcode=MailboxOpcode.TASK, target=MailboxTarget.GROUP, target_rank=-1, payloads=(b"task",)
        )
        mailbox.accept_request(last_sequence_id=0)
        mailbox.publish_first_error(7, origin)
        if gathered:
            mailbox.fail_request(
                sequence_id=7, errors=(origin, MpiRankError(0, "ConnectionError", "reply lost")), terminal=True
            )
        Worker._monitor_mpirun_group(Mock(closing=False, mailbox=mailbox), Mock(wait=Mock(return_value=7)))
        with pytest.raises(MpiGroupError, match="original 106") as raised:
            mailbox.read_result(sequence_id=7)
        assert ("reply lost" if gathered else "exited unexpectedly with status 7") in str(raised.value)
        assert mailbox.group_state is MailboxGroupState.TERMINAL
    finally:
        mailbox.close(unlink=True)


@pytest.mark.parametrize("stuck_peer", [False, True])
def test_mpi_group_cancels_running_peer_before_gather(monkeypatch, stuck_peer):
    mpi = pytest.importorskip("mpi4py.MPI")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if not launcher:
        pytest.skip("requires an MPI launcher")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(Path(__file__).parent), os.getenv("PYTHONPATH", ""))))
    launcher_args = ("-host", "localhost") if mpi.get_vendor()[0] == "MPICH" else ()
    with Worker(level=4, num_sub_workers=0, remote_session_timeout_s=8 if stuck_peer else 15) as worker:
        nodes = worker.add_mpirun_worker_group(
            MpiL3GroupSpec(
                hosts=("localhost", "localhost"),
                platform="a2a3sim",
                device_ids_by_rank=((0,), (1,)),
                num_sub_workers_by_rank=(1, 1),
                mpirun_path=launcher,
                mpirun_args=launcher_args,
            )
        )
        check_peer_exit(worker, nodes, True, group=True, stuck_peer=stuck_peer)
