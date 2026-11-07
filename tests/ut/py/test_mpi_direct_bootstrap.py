# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415 -- bootstrap tests import optional native dependencies at point of use
import socket
import struct
import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from simpler import mpi_direct_runtime as runtime
from simpler import mpi_direct_supervisor as supervisor
from simpler.mpi_direct_topology import MpiDirectTopology


def topology():
    return MpiDirectTopology.from_dict(
        {
            "controller_host": "127.0.0.1",
            "executor_ranks": [
                {"rank": 1, "worker_id": 0, "host": "127.0.0.1", "platform": "a2a3sim", "enable_sdma": True}
            ],
        }
    )


def test_bootstrap_stream_transfers_bytes_larger_than_gate():
    from simpler.mpi_direct_bootstrap import encode_startup, receive_startup, send_startup

    record = {"startup_artifacts": [{"payload": bytes(range(256)) * 2048}], "parameters": {"x": 1}}
    payload = encode_startup(record)
    left, right = socket.socketpair()
    with left, right:
        errors = []

        def send():
            try:
                send_startup(left, payload, time.monotonic() + 5)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=send)
        thread.start()
        assert receive_startup(right, time.monotonic() + 5) == record
        thread.join(5)
        assert not thread.is_alive() and not errors


def test_bootstrap_rejects_oversized_frame_before_reading_body():
    from simpler.mpi_direct_bootstrap import MAX_STARTUP_BYTES, receive_startup

    left, right = socket.socketpair()
    with left, right:
        left.sendall(struct.pack("!Q32s", MAX_STARTUP_BYTES + 1, b"0" * 32))
        with pytest.raises(ValueError, match="length"):
            receive_startup(right, time.monotonic() + 1)


def test_bootstrap_rejects_hash_mismatch():
    from simpler.mpi_direct_bootstrap import receive_startup

    left, right = socket.socketpair()
    with left, right:
        left.sendall(struct.pack("!Q32s", 2, b"0" * 32) + b"{}")
        with pytest.raises(ValueError, match="digest"):
            receive_startup(right, time.monotonic() + 1)


def test_executor_bootstrap_stages_before_init(monkeypatch):
    from simpler import remote_l3_session, startup_preparation
    from simpler.remote_l3_protocol import CallableKind

    events = []
    worker = SimpleNamespace(init=lambda **kw: events.append(("init", kw)))
    handle = object()
    monkeypatch.setattr(
        startup_preparation, "_install_node_artifacts", lambda x: events.append("artifacts") or {"abc": "root"}
    )
    monkeypatch.setattr(
        remote_l3_session,
        "_install_manifest_inner_registry",
        lambda m, w: events.append("registry") or {(CallableKind.CHIP_CALLABLE, b"abc"): handle},
    )
    monkeypatch.setattr(remote_l3_session, "_install_manifest_dispatcher_registry", lambda m: {})

    def hook(desc, context):
        events.append("hook")
        assert context.rank == 0
        assert context.chip_handles == {b"abc".hex(): handle}
        return "state"

    monkeypatch.setattr(startup_preparation, "_invoke_node_hook", hook)
    state = runtime._prepare_executor(worker, topology().executors[0], {}, 123)
    assert state[2] == "state"
    assert events == ["artifacts", "registry", "hook", ("init", {"_startup_deadline": 123})]
    assert topology().executors[0].enable_sdma is True


@pytest.mark.parametrize("drained", [False, True])
def test_hook_close_requires_proven_worker_drain(monkeypatch, drained):
    events = []
    worker = SimpleNamespace(close=lambda: events.append("worker"), _has_live_resources=lambda: not drained)
    hook = SimpleNamespace(close=lambda: events.append("hook"))
    if drained:
        runtime._close_executor_resources(worker, hook)
        assert events == ["worker", "hook"]
    else:
        with pytest.raises(RuntimeError, match="unconfirmed"):
            runtime._close_executor_resources(worker, hook)
        assert events == ["worker"]


def test_authenticated_gate_serves_record_then_waits_for_ready():
    from simpler.mpi_direct_bootstrap import encode_startup, receive_startup
    from simpler.mpi_direct_protocol import _gate_recv, _gate_send

    topo = topology()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    port = listener.getsockname()[1]
    errors = []

    def serve():
        try:
            supervisor._startup_gate(
                topo,
                "secret",
                listener,
                Mock(poll=lambda: None),
                {0: encode_startup({"parameters": {"hello": "world"}})},
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=serve)
    thread.start()
    deadline = time.monotonic() + 5
    with socket.create_connection(("127.0.0.1", port)) as peer:
        _gate_send(peer, {"token": "secret", "rank": 0, "state": "bootstrap"})
        assert _gate_recv(peer, deadline)["state"] == "bootstrap"
        assert receive_startup(peer, deadline) == {"parameters": {"hello": "world"}}
    peers = [socket.create_connection(("127.0.0.1", port)) for _ in range(2)]
    try:
        for rank, peer in enumerate(peers):
            _gate_send(peer, {"token": "secret", "rank": rank, "state": "ready"})
        for peer in peers:
            assert _gate_recv(peer, deadline)["state"] == "go_mpi"
    finally:
        for peer in peers:
            peer.close()
    thread.join(5)
    assert not thread.is_alive() and not errors


def test_executor_forwards_bootstrap_registry_and_completion(monkeypatch):
    import sys

    captured = {}
    callback = Mock()
    hook = SimpleNamespace(task_completed=callback, close=Mock())
    worker = SimpleNamespace(close=Mock(), _has_live_resources=lambda: False)

    def loop(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setitem(sys.modules, "simpler.remote_l3_session", SimpleNamespace(_run_command_loop=loop))
    monkeypatch.setattr(runtime, "_ExecutorFrameSocket", lambda *args: SimpleNamespace(close=Mock()))
    inner, dispatch = {"inner": 1}, {"dispatch": 1}
    assert (
        runtime._run_executor(
            None, None, topology(), 1, topology().executors[0], worker, startup_state=(dispatch, inner, hook)
        )
        == 0
    )
    assert captured["args"][3:5] == (inner, dispatch)
    assert captured["kwargs"]["task_completed"] == callback
    hook.close.assert_called_once()


def test_controller_worker_installs_initial_registry(monkeypatch):
    from simpler import remote_l3_session
    from simpler import worker as worker_mod

    context = object.__new__(runtime.MpiDirectControllerContext)
    context._startup_record = {"parameters": {"x": 1}, "inner_l3_worker": []}
    context.startup_registry = {}
    context.attach = Mock()
    worker = object()
    monkeypatch.setattr(worker_mod, "Worker", lambda **kw: worker)
    expected = {"handle": "value"}
    monkeypatch.setattr(remote_l3_session, "_install_manifest_inner_registry", lambda record, target: expected)
    assert context.create_worker() is worker
    assert context.startup_registry == expected


def test_failed_prefork_init_retains_hook_for_cleanup(monkeypatch):
    from simpler import remote_l3_session, startup_preparation

    hook = SimpleNamespace(close=Mock())
    monkeypatch.setattr(startup_preparation, "_install_node_artifacts", lambda _: {})
    monkeypatch.setattr(remote_l3_session, "_install_manifest_inner_registry", lambda *a: {})
    monkeypatch.setattr(remote_l3_session, "_install_manifest_dispatcher_registry", lambda *a: {})
    monkeypatch.setattr(startup_preparation, "_invoke_node_hook", lambda *a: hook)
    state = [None]
    worker = SimpleNamespace(init=Mock(side_effect=TimeoutError("init deadline")))
    with pytest.raises(TimeoutError, match="init deadline"):
        runtime._prepare_executor(worker, topology().executors[0], {}, 123, state)
    assert state[0] is hook
    hook.close.assert_not_called()


def test_rank_failure_notifies_waiting_ready_peer():
    from simpler.mpi_direct_protocol import _gate_recv, _gate_send

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    port = listener.getsockname()[1]
    errors = []

    def serve():
        try:
            supervisor._startup_gate(topology(), "secret", listener, Mock(poll=lambda: None))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=serve)
    thread.start()
    with socket.create_connection(("127.0.0.1", port)) as ready:
        _gate_send(ready, {"token": "secret", "rank": 0, "state": "ready"})
        with socket.create_connection(("127.0.0.1", port)) as failed:
            _gate_send(failed, {"token": "secret", "rank": 1, "state": "failed", "error": "hook failure"})
        response = _gate_recv(ready, time.monotonic() + 5)
        assert response["state"] == "failed"
        assert "rank 1" in response["error"] and "hook failure" in response["error"]
    thread.join(5)
    assert len(errors) == 1


def test_executor_failstops_when_hook_storage_is_still_live(monkeypatch):
    import sys

    worker = SimpleNamespace(close=Mock(), _has_live_resources=lambda: True)
    hook = SimpleNamespace(close=Mock())
    monkeypatch.setitem(
        sys.modules, "simpler.remote_l3_session", SimpleNamespace(_run_command_loop=lambda *a, **k: None)
    )
    monkeypatch.setattr(runtime, "_ExecutorFrameSocket", lambda *args: SimpleNamespace(close=Mock()))

    def exit_rank(code):
        raise SystemExit(code)

    monkeypatch.setattr(runtime.os, "_exit", exit_rank)
    with pytest.raises(SystemExit) as error:
        runtime._run_executor(None, None, topology(), 1, topology().executors[0], worker, startup_state=({}, {}, hook))
    assert error.value.code == 124
    hook.close.assert_not_called()


def test_startup_codec_supports_local_launcher_file():
    from simpler.mpi_direct_bootstrap import decode_startup, encode_startup

    record = {"startup_by_rank": {"0": {"parameters": {"nested": [b"binary", {}, 1, None]}}}}
    assert decode_startup(encode_startup(record)) == record


def test_real_mpi_bootstraps_packaged_controller_and_prefork_hooks(tmp_path):
    import gzip
    import hashlib
    import io
    import json
    import random
    import shutil
    import tarfile
    import textwrap

    mpirun = shutil.which("mpiexec") or shutil.which("mpirun")
    if not mpirun:
        pytest.skip("MPI launcher unavailable")
    assert mpirun is not None
    pytest.importorskip("mpi4py")
    prefix = "direct_bootstrap_artifact"
    source = (
        textwrap.dedent("""
        import pathlib
        from simpler.task_interface import TaskArgs
        from simpler.worker import RemoteCallable
        class State:
            def __init__(self, context):
                self.context = context
                self.completed = 0
            def task_completed(self):
                self.completed += 1
            def close(self):
                assert self.completed == 1
                assert not self.context.inner_worker._has_live_resources()
                pathlib.Path(self.context.parameters['marker']).write_text('closed')
        def hook(context):
            assert not context.inner_worker._initialized
            assert context.rank == context.parameters['node_index']
            return State(context)
        def orch(_orch, args, _config):
            assert args.scalar(0) == 42
        def main(context):
            assert context.startup_parameters == {'answer': 42}
            worker = context.create_worker(num_sub_workers=0)
            handle = worker.register(RemoteCallable('direct_bootstrap_artifact.entry:orch'),
                                     workers=[spec.worker_id for spec in context.topology.executors])
            worker.init()
            def parent(orch, _args, config):
                for spec in context.topology.executors:
                    args = TaskArgs()
                    args.add_scalar(42)
                    orch.submit_next_level(handle, args, config, worker=spec.worker_id)
            worker.run(parent)
    """).encode()
        + b"\n# "
        + random.Random(42).randbytes(128000).hex().encode()
    )
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        info = tarfile.TarInfo("entry.py")
        info.size = len(source)
        archive.addfile(info, io.BytesIO(source))
    payload = gzip.compress(stream.getvalue(), mtime=0)
    artifact = {
        "protocol_version": 1,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "module_prefix": prefix,
        "payload": payload,
    }
    manifest = {
        "controller_host": "127.0.0.1",
        "startup_timeout_s": 30,
        "session_timeout_s": 10,
        "executor_ranks": [
            {"rank": rank, "worker_id": rank - 1, "host": "127.0.0.1", "platform": "a2a3sim"} for rank in (1, 2)
        ],
    }
    topology_path = tmp_path / "topology.json"
    topology_path.write_text(json.dumps(manifest))
    records: dict[int, dict[str, Any]] = {rank: {"startup_artifacts": [artifact]} for rank in (0, 1, 2)}
    records[0]["parameters"] = {"answer": 42}
    for rank in (1, 2):
        records[rank]["startup_hook"] = {
            "import_path": prefix + ".entry:hook",
            "parameters": {"node_index": rank - 1, "marker": str(tmp_path / str(rank))},
        }
    reports = {}
    assert (
        supervisor.run_supervisor(
            str(topology_path),
            prefix + ".entry:main",
            mpirun_path=mpirun,
            startup_by_rank=records,
            cleanup_reports=reports,
        )
        == 0
    )
    assert set(reports) == {0, 1, 2}
    assert all(report.sequence == 2 and report.cleanup == "succeeded" for report in reports.values())
    assert all((tmp_path / str(rank)).read_text() == "closed" for rank in (1, 2))


@pytest.mark.parametrize("stale_context", [{}, {"platform": "wrong", "runtime": "wrong"}])
def test_executor_registry_uses_topology_descriptor_context(monkeypatch, stale_context):
    from simpler import remote_l3_session, startup_preparation

    captured = []
    monkeypatch.setattr(startup_preparation, "_install_node_artifacts", lambda _: {})
    monkeypatch.setattr(startup_preparation, "_invoke_node_hook", lambda *args: None)
    monkeypatch.setattr(
        remote_l3_session, "_install_manifest_inner_registry", lambda record, worker: captured.append(record) or {}
    )
    monkeypatch.setattr(remote_l3_session, "_install_manifest_dispatcher_registry", lambda record: {})
    worker = SimpleNamespace(init=Mock())
    spec = topology().executors[0]
    record = {"inner_l3_worker": [], **stale_context}
    runtime._prepare_executor(worker, spec, record, 123)
    assert captured[0]["platform"] == spec.platform
    assert captured[0]["runtime"] == spec.runtime
    assert record == {"inner_l3_worker": [], **stale_context}


@pytest.mark.parametrize("stale_context", [{}, {"platform": "wrong", "runtime": "wrong"}])
def test_controller_registry_uses_requested_worker_descriptor_context(monkeypatch, stale_context):
    from simpler import remote_l3_session
    from simpler import worker as worker_mod

    context = object.__new__(runtime.MpiDirectControllerContext)
    context._startup_record = {"inner_l3_worker": [], **stale_context}
    context.attach = Mock()
    worker = object()
    monkeypatch.setattr(worker_mod, "Worker", lambda **kwargs: worker)
    captured = []
    monkeypatch.setattr(
        remote_l3_session, "_install_manifest_inner_registry", lambda record, target: captured.append(record) or {}
    )
    assert context.create_worker(platform="a2a3sim", runtime="host_build_graph") is worker
    assert captured[0]["platform"] == "a2a3sim"
    assert captured[0]["runtime"] == "host_build_graph"
    assert context._startup_record == {"inner_l3_worker": [], **stale_context}
