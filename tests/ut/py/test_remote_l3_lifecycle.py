# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import contextlib
import json
import os
import socket
import struct
import threading
import time
from typing import cast

import pytest
from simpler import remote_l3_session, remote_l3_worker


def _manifest(**extra):
    manifest = {
        "session_id": 1,
        "worker_id": 0,
        "parent_worker_level": 4,
        "remote_worker_level": 3,
        "platform": "a2a3sim",
        "transport": "sim",
        "listen_host": "127.0.0.1",
        "connect_host": "127.0.0.1",
        "session_timeout_s": 0.01,
    }
    manifest.update(extra)
    return manifest


def test_read_runner_ready_times_out_without_payload():
    ready_r, ready_w = os.pipe()
    try:
        with pytest.raises(TimeoutError):
            remote_l3_worker._read_runner_ready(ready_r, time.monotonic() + 0.01)
    finally:
        os.close(ready_r)
        os.close(ready_w)


class _Clock:
    """Deterministic monotonic clock advanced by explicit ticks."""

    def __init__(self, t0: float = 1000.0):
        self.t = t0

    def monotonic(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_start_session_hands_runner_absolute_deadline_and_single_ready_wait(monkeypatch):
    # The daemon builds ONE absolute deadline up front; the runner-ready wait
    # derives from it (so the daemon's own Popen/setup time is charged, not a
    # fresh full slice), and the runner is handed that same ABSOLUTE deadline —
    # so the runner's deadline cannot be re-amplified past the daemon's.
    clock = _Clock(1000.0)
    monkeypatch.setattr(remote_l3_worker.time, "monotonic", clock.monotonic)
    captured: dict = {}

    def fake_pipe():
        clock.advance(1.0)  # daemon setup before the runner manifest is written
        return (11, 12)

    class _FakeTmp:
        name = "/tmp/fake-remote-l3.json"

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def write(self, *_a):
            pass

    class _FakePopen:
        pid = 999

        def __init__(self, *_a, **_k):
            clock.advance(2.0)  # Popen + interpreter spawn charged to runner-ready

        def wait(self, timeout=None):
            return 0

    class _SyncThread:
        def __init__(self, *, target, args, daemon):
            self._target, self._args = target, args

        def start(self):
            self._target(*self._args)

    def fake_read_ready(fd, deadline):
        captured["deadline"] = deadline
        return {"ok": True}

    monkeypatch.setattr(remote_l3_worker.os, "pipe", fake_pipe)
    monkeypatch.setattr(remote_l3_worker.os, "close", lambda fd: None)
    monkeypatch.setattr(remote_l3_worker.os, "unlink", lambda p: None)
    monkeypatch.setattr(remote_l3_worker.tempfile, "NamedTemporaryFile", lambda *a, **k: _FakeTmp())
    monkeypatch.setattr(remote_l3_worker.json, "dump", lambda obj, f, **k: captured.__setitem__("manifest", obj))
    monkeypatch.setattr(remote_l3_worker.subprocess, "Popen", lambda *a, **k: _FakePopen())
    monkeypatch.setattr(remote_l3_worker.threading, "Thread", _SyncThread)
    monkeypatch.setattr(remote_l3_worker, "_read_runner_ready", fake_read_ready)

    reply, _proc = remote_l3_worker._start_session(_manifest(startup_remaining_s=50.0, session_timeout_s=30.0))

    # One deadline = 1000 + 50; the runner-ready wait uses it (the 2s Popen is
    # charged), NOT a fresh now()+50 rebuilt after Popen.
    assert captured["deadline"] == 1050.0
    # The runner is handed that same ABSOLUTE monotonic deadline, so its own
    # deadline equals the daemon's (spawn/import time charged) rather than a
    # rebuilt now()+remaining that would run past it.
    assert captured["manifest"]["startup_deadline_monotonic"] == 1050.0
    assert reply["ok"] is True


def test_start_session_kills_runner_on_ready_timeout(monkeypatch):
    class FakePopen:
        pid = 12345

        def __init__(self, *args, **kwargs):
            self.terminated = False
            self.killed = False
            self.wait_calls = 0

        def poll(self):
            return -9 if self.killed else None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            # Model a runner that does not exit on its own or on the cooperative
            # SIGTERM, so cleanup must escalate to the killpg/kill backstop.
            self.wait_calls += 1
            raise remote_l3_worker.subprocess.TimeoutExpired(
                cmd="runner", timeout=timeout if timeout is not None else 0.0
            )

    fake_proc = FakePopen()

    def fake_popen(*args, **kwargs):
        return fake_proc

    def fake_read_ready(fd, deadline):
        raise TimeoutError("ready timeout")

    monkeypatch.setattr(remote_l3_worker.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(remote_l3_worker, "_read_runner_ready", fake_read_ready)

    with pytest.raises(TimeoutError):
        remote_l3_worker._start_session(_manifest(startup_remaining_s=30.0))

    # Cooperative SIGTERM first, then the hard SIGKILL backstop.
    assert fake_proc.terminated
    assert fake_proc.killed
    assert fake_proc.wait_calls >= 1


def test_start_session_returns_live_runner_without_reaping(monkeypatch):
    class FakePopen:
        pid = 12345

        def __init__(self, *args, **kwargs):
            self.wait_calls = 0

        def wait(self, timeout=None):
            self.wait_calls += 1
            return 0

    fake_proc = FakePopen()

    monkeypatch.setattr(remote_l3_worker.subprocess, "Popen", lambda *args, **kwargs: fake_proc)
    monkeypatch.setattr(remote_l3_worker, "_read_runner_ready", lambda fd, timeout_s: {"ok": True})

    reply, proc = remote_l3_worker._start_session(_manifest(startup_remaining_s=30.0))

    # A ready runner is handed back live for the caller to hand off or reclaim;
    # _start_session itself neither reaps nor kills it.
    assert reply["ok"] is True
    assert reply["pid"] == fake_proc.pid
    assert proc is fake_proc
    assert fake_proc.wait_calls == 0


def test_start_session_returns_none_proc_when_runner_reports_not_ok(monkeypatch):
    class FakePopen:
        pid = 777

        def wait(self, timeout=None):
            return 0

    fake_proc = FakePopen()
    reclaimed: list = []

    monkeypatch.setattr(remote_l3_worker.subprocess, "Popen", lambda *args, **kwargs: fake_proc)
    monkeypatch.setattr(remote_l3_worker, "_read_runner_ready", lambda fd, timeout_s: {"ok": False})
    monkeypatch.setattr(remote_l3_worker, "_wait_or_kill_runner", lambda p, **kw: reclaimed.append(p))

    reply, proc = remote_l3_worker._start_session(_manifest(startup_remaining_s=30.0))

    # A failed handshake is killed+reaped exactly once inside _start_session, so
    # the caller gets no runner to reclaim.
    assert reply["ok"] is False
    assert proc is None
    assert reclaimed == [fake_proc]


def test_run_session_bounds_command_accept_by_startup_deadline_not_session_timeout(monkeypatch):
    # Waiting for the parent to attach is still the attach phase: command accept
    # must be bounded by the remaining startup budget, not session_timeout_s.
    clock = _Clock(1000.0)
    monkeypatch.setattr(remote_l3_session.time, "monotonic", clock.monotonic)

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            pass

        def init(self, *args, **kwargs):
            pass

        def close(self):
            pass

    class FakeCommandSock:
        def __init__(self):
            self.timeout = None

        def getsockname(self):
            return ("127.0.0.1", 12345)

        def settimeout(self, timeout):
            self.timeout = timeout

        def accept(self):
            if self.timeout is None:
                raise AssertionError("command accept has no timeout")
            raise socket.timeout("command attach timed out")

        def close(self):
            pass

    class FakeHealthSock:
        def getsockname(self):
            return ("127.0.0.1", 12346)

        def close(self):
            pass

    command_sock = FakeCommandSock()
    sockets = [command_sock, FakeHealthSock()]
    ready_r, ready_w = os.pipe()

    monkeypatch.setattr(remote_l3_session, "Worker", FakeWorker)
    monkeypatch.setattr(remote_l3_session, "_install_manifest_dispatcher_registry", lambda manifest: {})
    monkeypatch.setattr(remote_l3_session, "_install_manifest_inner_registry", lambda manifest, worker: {})
    monkeypatch.setattr(remote_l3_session, "_bind_listener", lambda host: sockets.pop(0))
    monkeypatch.setattr(remote_l3_session, "_health_loop", lambda *args: None)

    try:
        # 5s startup budget (deadline 1005), 30s runtime timeout — deliberately different.
        rc = remote_l3_session.run_session(
            _manifest(startup_deadline_monotonic=1005.0, session_timeout_s=30.0), ready_w
        )
        assert rc == 1
    finally:
        os.close(ready_r)

    # accept timeout = startup_deadline - now() = 5.0, NOT the 30s session timeout.
    assert command_sock.timeout == 5.0


def test_run_session_forces_command_conn_blocking_for_idle(monkeypatch):
    # A finite timeout on the command connection would self-destruct a healthy but
    # idle session (read_frame idle-waiting for the next command would raise). The
    # accept()'d socket inherits socket.getdefaulttimeout() (a user module could
    # have set it), so the runner must explicitly force it blocking.
    clock = _Clock(1000.0)
    monkeypatch.setattr(remote_l3_session.time, "monotonic", clock.monotonic)

    class FakeConn:
        def __init__(self):
            self.timeout = 0.05  # hostile: a non-None default timeout was inherited

        def settimeout(self, t):
            self.timeout = t

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    fake_conn = FakeConn()

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            pass

        def init(self, *args, **kwargs):
            pass

        def close(self):
            pass

    class FakeCommandSock:
        def getsockname(self):
            return ("127.0.0.1", 12345)

        def settimeout(self, timeout):
            pass

        def accept(self):
            return fake_conn, ("127.0.0.1", 1)

        def close(self):
            pass

    class FakeHealthSock:
        def getsockname(self):
            return ("127.0.0.1", 12346)

        def close(self):
            pass

    sockets = [FakeCommandSock(), FakeHealthSock()]
    ready_r, ready_w = os.pipe()

    monkeypatch.setattr(remote_l3_session, "Worker", FakeWorker)
    monkeypatch.setattr(remote_l3_session, "_install_manifest_dispatcher_registry", lambda manifest: {})
    monkeypatch.setattr(remote_l3_session, "_install_manifest_inner_registry", lambda manifest, worker: {})
    monkeypatch.setattr(remote_l3_session, "_bind_listener", lambda host: sockets.pop(0))
    monkeypatch.setattr(remote_l3_session, "_health_loop", lambda *args: None)
    monkeypatch.setattr(remote_l3_session, "_run_command_loop", lambda *args, **kwargs: None)

    try:
        rc = remote_l3_session.run_session(
            _manifest(startup_deadline_monotonic=1005.0, session_timeout_s=0.05), ready_w
        )
        assert rc == 0
    finally:
        os.close(ready_r)

    # Forced blocking before the command loop, regardless of the inherited default.
    assert fake_conn.timeout is None


def test_run_session_bounds_subtree_by_startup_remaining_not_session_timeout(monkeypatch):
    """The inner subtree deadline comes from the parent's startup_remaining_s
    (its slice of the single root startup budget), not the runtime command
    session_timeout_s — the two are deliberately different here."""

    captured = {}

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def init(self, *args, _startup_deadline=None, **kwargs):
            captured["deadline"] = _startup_deadline
            captured["at"] = time.monotonic()

        def close(self):
            self.closed = True

    class FakeCommandSock:
        def getsockname(self):
            return ("127.0.0.1", 12345)

        def settimeout(self, timeout):
            pass

        def accept(self):
            raise socket.timeout("stop after init")

        def close(self):
            pass

    class FakeHealthSock:
        def getsockname(self):
            return ("127.0.0.1", 12346)

        def close(self):
            pass

    sockets = [FakeCommandSock(), FakeHealthSock()]
    ready_r, ready_w = os.pipe()

    monkeypatch.setattr(remote_l3_session, "Worker", FakeWorker)
    monkeypatch.setattr(remote_l3_session, "_install_manifest_dispatcher_registry", lambda manifest: {})
    monkeypatch.setattr(remote_l3_session, "_install_manifest_inner_registry", lambda manifest, worker: {})
    monkeypatch.setattr(remote_l3_session, "_bind_listener", lambda host: sockets.pop(0))
    monkeypatch.setattr(remote_l3_session, "_health_loop", lambda *args: None)

    try:
        remote_l3_session.run_session(_manifest(session_timeout_s=0.01, startup_remaining_s=50.0), ready_w)
    finally:
        os.close(ready_r)

    budget = captured["deadline"] - captured["at"]
    # ~50s startup budget, not the 0.01s runtime command timeout.
    assert 40.0 < budget <= 50.0


def test_run_session_builds_deadline_before_registry_install(monkeypatch):
    # The runner establishes its single deadline before registry install, so that
    # time is charged against the budget rather than restarting a fresh slice at
    # inner init.
    clock = _Clock(1000.0)
    monkeypatch.setattr(remote_l3_session.time, "monotonic", clock.monotonic)
    captured: dict = {}

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            pass

        def init(self, *args, _startup_deadline=None, **kwargs):
            captured["deadline"] = _startup_deadline

        def close(self):
            pass

    def slow_dispatch_registry(manifest):
        clock.advance(5.0)  # registry install consumes the budget
        return {}

    class FakeCommandSock:
        def getsockname(self):
            return ("127.0.0.1", 12345)

        def settimeout(self, timeout):
            pass

        def accept(self):
            raise socket.timeout("stop after init")

        def close(self):
            pass

    class FakeHealthSock:
        def getsockname(self):
            return ("127.0.0.1", 12346)

        def close(self):
            pass

    sockets = [FakeCommandSock(), FakeHealthSock()]
    ready_r, ready_w = os.pipe()

    monkeypatch.setattr(remote_l3_session, "Worker", FakeWorker)
    monkeypatch.setattr(remote_l3_session, "_install_manifest_dispatcher_registry", slow_dispatch_registry)
    monkeypatch.setattr(remote_l3_session, "_install_manifest_inner_registry", lambda manifest, worker: {})
    monkeypatch.setattr(remote_l3_session, "_bind_listener", lambda host: sockets.pop(0))
    monkeypatch.setattr(remote_l3_session, "_health_loop", lambda *args: None)

    try:
        remote_l3_session.run_session(_manifest(session_timeout_s=0.01, startup_remaining_s=50.0), ready_w)
    finally:
        os.close(ready_r)

    # deadline built at t=1000 (before the 5s registry install) => 1050, NOT the
    # 1005 + 50 = 1055 a post-registry rebuild would give.
    assert captured["deadline"] == 1050.0


def test_run_session_uses_daemon_absolute_deadline_verbatim(monkeypatch):
    # Given the daemon's shared-clock absolute deadline, the runner uses it as-is
    # — its deadline equals the daemon's regardless of spawn/import/registry time.
    clock = _Clock(1000.0)
    monkeypatch.setattr(remote_l3_session.time, "monotonic", clock.monotonic)
    captured: dict = {}

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            pass

        def init(self, *args, _startup_deadline=None, **kwargs):
            captured["deadline"] = _startup_deadline

        def close(self):
            pass

    def slow_dispatch_registry(manifest):
        clock.advance(7.0)  # spawn/import/registry cost — must NOT extend the deadline
        return {}

    class FakeCommandSock:
        def getsockname(self):
            return ("127.0.0.1", 12345)

        def settimeout(self, timeout):
            pass

        def accept(self):
            raise socket.timeout("stop after init")

        def close(self):
            pass

    class FakeHealthSock:
        def getsockname(self):
            return ("127.0.0.1", 12346)

        def close(self):
            pass

    sockets = [FakeCommandSock(), FakeHealthSock()]
    ready_r, ready_w = os.pipe()

    monkeypatch.setattr(remote_l3_session, "Worker", FakeWorker)
    monkeypatch.setattr(remote_l3_session, "_install_manifest_dispatcher_registry", slow_dispatch_registry)
    monkeypatch.setattr(remote_l3_session, "_install_manifest_inner_registry", lambda manifest, worker: {})
    monkeypatch.setattr(remote_l3_session, "_bind_listener", lambda host: sockets.pop(0))
    monkeypatch.setattr(remote_l3_session, "_health_loop", lambda *args: None)

    try:
        remote_l3_session.run_session(_manifest(startup_deadline_monotonic=1042.0, session_timeout_s=0.01), ready_w)
    finally:
        os.close(ready_r)

    # Uses the daemon's absolute deadline verbatim (1042), not now()+something.
    assert captured["deadline"] == 1042.0


def test_health_loop_closes_active_connection_on_stop():
    stop = threading.Event()

    class FakeConn:
        def __init__(self):
            self.closed = False

        def settimeout(self, timeout):
            pass

        def sendall(self, data):
            stop.set()

        def close(self):
            self.closed = True

    class FakeSock:
        def __init__(self, conn):
            self.conn = conn
            self.closed = False

        def settimeout(self, timeout):
            pass

        def accept(self):
            return self.conn, ("127.0.0.1", 1)

        def close(self):
            self.closed = True

    conn = FakeConn()
    sock = FakeSock(conn)

    remote_l3_session._health_loop(cast(socket.socket, sock), stop, session_id=1, worker_id=0)

    assert sock.closed
    assert conn.closed


class _FakeProc:
    """A runner that never exits on its own — models a ready, live session."""

    pid = 4242

    def __init__(self):
        self.terminated = False
        self.killed = False

    def wait(self, timeout=None):
        raise remote_l3_worker.subprocess.TimeoutExpired(cmd="runner", timeout=timeout if timeout is not None else 0.0)

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class _SyncThread:
    """Runs the reaper target inline so hand-off is deterministically observable
    (no scheduler race): construction+start executes the target immediately."""

    def __init__(self, *, target, args, daemon):
        self._target = target
        self._args = args

    def start(self):
        self._target(*self._args)


def _start_must_not_run(manifest):
    raise AssertionError("_start_session must not run when the handshake read fails")


def _serve(monkeypatch, *, read=None, start=None, send=None, thread: type = _SyncThread):
    """Install the common _serve_connection collaborators and return the spy
    lists (sent, reaped, reclaimed)."""
    sent: list = []
    reaped: list = []
    reclaimed: list = []
    if read is not None:
        monkeypatch.setattr(remote_l3_worker, "_read_json", read)
    if start is not None:
        monkeypatch.setattr(remote_l3_worker, "_start_session", start)
    if send is None:
        send = lambda conn, payload: sent.append(payload)  # noqa: E731
    monkeypatch.setattr(remote_l3_worker, "_send_json", send)
    monkeypatch.setattr(remote_l3_worker, "_reap_session_runner", lambda p: reaped.append(p))
    monkeypatch.setattr(remote_l3_worker, "_wait_or_kill_runner", lambda p, **kw: reclaimed.append(p))
    monkeypatch.setattr(remote_l3_worker.threading, "Thread", thread)
    return sent, reaped, reclaimed


def test_serve_connection_hands_off_runner_after_send_returns(monkeypatch):
    proc = _FakeProc()
    sent, reaped, reclaimed = _serve(
        monkeypatch,
        read=lambda conn: {"manifest": True},
        start=lambda manifest: ({"ok": True, "pid": proc.pid}, proc),
    )

    remote_l3_worker._serve_connection(cast(socket.socket, object()))

    assert sent == [{"ok": True, "pid": proc.pid}]
    assert reaped == [proc]  # send returned → runner handed to the reaper exactly once
    assert reclaimed == []  # ownership transferred, so the finally does not reclaim


def test_serve_connection_reclaims_runner_when_send_raises(monkeypatch):
    proc = _FakeProc()

    def broken_send(conn, payload):
        raise BrokenPipeError("parent disconnected before reply")

    _sent, reaped, reclaimed = _serve(
        monkeypatch,
        read=lambda conn: {"manifest": True},
        start=lambda manifest: ({"ok": True, "pid": proc.pid}, proc),
        send=broken_send,
    )

    # A dead parent on the reply must not escape this connection.
    remote_l3_worker._serve_connection(cast(socket.socket, object()))

    assert reclaimed == [proc]  # undelivered runner reclaimed exactly once
    assert reaped == []  # never handed off


def test_serve_connection_reclaims_runner_when_reaper_launch_fails(monkeypatch):
    proc = _FakeProc()

    class _FailingThread:
        def __init__(self, *, target, args, daemon):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    sent, reaped, reclaimed = _serve(
        monkeypatch,
        read=lambda conn: {"manifest": True},
        start=lambda manifest: ({"ok": True, "pid": proc.pid}, proc),
        thread=_FailingThread,
    )

    # Thread exhaustion at reaper launch must neither escape the connection nor
    # orphan the runner.
    remote_l3_worker._serve_connection(cast(socket.socket, object()))

    assert sent == [{"ok": True, "pid": proc.pid}]  # reply was delivered
    assert reaped == []
    assert reclaimed == [proc]  # runner reclaimed exactly once


def test_serve_connection_swallows_error_reply_send_failure(monkeypatch):
    def bad_start(manifest):
        raise ValueError("bad manifest")

    def broken_send(conn, payload):
        raise ConnectionResetError("parent disconnected before error reply")

    _sent, _reaped, reclaimed = _serve(
        monkeypatch,
        read=lambda conn: {"manifest": True},
        start=bad_start,
        send=broken_send,
    )

    # No runner was ever created and the error reply cannot land — nothing to
    # reclaim, nothing escapes.
    remote_l3_worker._serve_connection(cast(socket.socket, object()))

    assert reclaimed == []


def test_serve_connection_survives_truncated_frame(monkeypatch):
    def eof(conn):
        raise EOFError("remote daemon socket closed")

    _sent, _reaped, reclaimed = _serve(
        monkeypatch,
        read=eof,
        start=_start_must_not_run,
    )

    # A truncated / closed frame is an ordinary handshake failure: no runner, no
    # escape.
    remote_l3_worker._serve_connection(cast(socket.socket, object()))

    assert reclaimed == []


def test_serve_connection_reports_real_error_to_live_parent(monkeypatch):
    def bad_start(manifest):
        raise ValueError("bad manifest")

    sent, _reaped, _reclaimed = _serve(
        monkeypatch,
        read=lambda conn: {"manifest": True},
        start=bad_start,
    )

    remote_l3_worker._serve_connection(cast(socket.socket, object()))

    assert len(sent) == 1
    assert sent[0]["ok"] is False
    assert "ValueError" in sent[0]["error"]


def test_serve_connection_propagates_control_exception_from_handshake(monkeypatch):
    def interrupt(conn):
        raise KeyboardInterrupt

    sent, _reaped, reclaimed = _serve(
        monkeypatch,
        read=interrupt,
        start=_start_must_not_run,
    )

    # KeyboardInterrupt is BaseException: it propagates for clean shutdown rather
    # than being caught into a spurious error reply.
    with pytest.raises(KeyboardInterrupt):
        remote_l3_worker._serve_connection(cast(socket.socket, object()))

    assert sent == []
    assert reclaimed == []


def test_serve_connection_reclaims_runner_and_propagates_control_exception_on_send(monkeypatch):
    proc = _FakeProc()

    def interrupt_send(conn, payload):
        raise KeyboardInterrupt

    _sent, reaped, reclaimed = _serve(
        monkeypatch,
        read=lambda conn: {"manifest": True},
        start=lambda manifest: ({"ok": True, "pid": proc.pid}, proc),
        send=interrupt_send,
    )

    # The control exception unwinds, but the live runner is reclaimed exactly
    # once before it propagates.
    with pytest.raises(KeyboardInterrupt):
        remote_l3_worker._serve_connection(cast(socket.socket, object()))

    assert reaped == []
    assert reclaimed == [proc]


class _FakeConn:
    def __init__(self):
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False


class _FakeServer:
    def __init__(self, conns):
        self._conns = list(conns)

    def accept(self):
        if self._conns:
            return self._conns.pop(0), ("127.0.0.1", 0)
        raise OSError("listener closed")


def test_serve_loop_serves_every_connection_and_stops_on_listener_close(monkeypatch):
    served: list = []
    monkeypatch.setattr(remote_l3_worker, "_serve_connection", lambda conn: served.append(conn))

    conns = [_FakeConn(), _FakeConn(), _FakeConn()]
    remote_l3_worker._serve_loop(cast(socket.socket, _FakeServer(conns)))

    # Every connection was handled (loop survived each) and the loop exited
    # cleanly when accept() reported the listener closed; each conn was closed.
    assert served == conns
    assert all(c.closed for c in conns)


def _serve_bounded(listener, n):
    for _ in range(n):
        conn, _addr = listener.accept()
        with conn:
            remote_l3_worker._serve_connection(conn)


def _bounded_daemon(listener, n):
    thread = threading.Thread(target=_serve_bounded, args=(listener, n), daemon=True)
    thread.start()
    return thread


def _new_listener():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    return listener, listener.getsockname()[1]


def _send_manifest(sock):
    data = json.dumps({"any": "manifest"}).encode("utf-8")
    sock.sendall(struct.pack("<I", len(data)) + data)


def test_serve_real_socket_isolates_dead_parent_then_serves_next(monkeypatch):
    procs: list = []
    reclaimed: list = []
    first_conn: dict = {"conn": None}
    real_send = remote_l3_worker._send_json

    def fake_start(manifest):
        proc = _FakeProc()
        procs.append(proc)
        return {"ok": True, "pid": proc.pid}, proc

    def flaky_send(conn, payload):
        # Every send on the first accepted connection raises, so the pre-fix
        # success-reply-then-error-reply pair both failed and escaped the loop.
        if first_conn["conn"] is None:
            first_conn["conn"] = conn
        if conn is first_conn["conn"]:
            raise BrokenPipeError("parent disconnected before reply")
        real_send(conn, payload)

    monkeypatch.setattr(remote_l3_worker, "_start_session", fake_start)
    monkeypatch.setattr(remote_l3_worker, "_wait_or_kill_runner", lambda p, **kw: reclaimed.append(p))
    monkeypatch.setattr(remote_l3_worker, "_reap_session_runner", lambda p: None)
    monkeypatch.setattr(remote_l3_worker, "_send_json", flaky_send)

    listener, port = _new_listener()
    server_thread = _bounded_daemon(listener, 2)
    replies: dict = {}
    try:
        c1 = socket.create_connection(("127.0.0.1", port), timeout=5.0)
        c1.settimeout(5.0)
        try:
            _send_manifest(c1)
            with contextlib.suppress(OSError, EOFError):
                replies["c1"] = remote_l3_worker._read_json(c1)
        finally:
            c1.close()

        c2 = socket.create_connection(("127.0.0.1", port), timeout=5.0)
        c2.settimeout(5.0)
        try:
            _send_manifest(c2)
            replies["c2"] = remote_l3_worker._read_json(c2)
        finally:
            c2.close()
    finally:
        server_thread.join(timeout=5.0)
        listener.close()

    assert not server_thread.is_alive()
    assert replies.get("c2", {}).get("ok") is True  # second parent still served
    assert "c1" not in replies  # first parent got no reply (send failed)
    assert reclaimed == [procs[0]]  # only the undelivered first runner reclaimed


def test_serve_real_socket_survives_truncated_frame_then_serves_next(monkeypatch):
    procs: list = []

    def fake_start(manifest):
        proc = _FakeProc()
        procs.append(proc)
        return {"ok": True, "pid": proc.pid}, proc

    monkeypatch.setattr(remote_l3_worker, "_start_session", fake_start)
    monkeypatch.setattr(remote_l3_worker, "_reap_session_runner", lambda p: None)

    listener, port = _new_listener()
    server_thread = _bounded_daemon(listener, 2)
    reply = None
    try:
        # c1: a truncated frame (length prefix promises more than is sent), then
        # close — the daemon's real _read_json hits EOF.
        c1 = socket.create_connection(("127.0.0.1", port), timeout=5.0)
        try:
            c1.sendall(struct.pack("<I", 64) + b"{partial")
        finally:
            c1.close()

        c2 = socket.create_connection(("127.0.0.1", port), timeout=5.0)
        c2.settimeout(5.0)
        try:
            _send_manifest(c2)
            reply = remote_l3_worker._read_json(c2)
        finally:
            c2.close()
    finally:
        server_thread.join(timeout=5.0)
        listener.close()

    assert not server_thread.is_alive()
    assert reply is not None and reply["ok"] is True  # next connection served after EOF
