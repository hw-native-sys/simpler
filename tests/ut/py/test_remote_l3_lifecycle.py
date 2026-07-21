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
            remote_l3_worker._read_runner_ready(ready_r, 0.01)
    finally:
        os.close(ready_r)
        os.close(ready_w)


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

    def fake_read_ready(fd, timeout_s):
        raise TimeoutError("ready timeout")

    monkeypatch.setattr(remote_l3_worker.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(remote_l3_worker, "_read_runner_ready", fake_read_ready)

    with pytest.raises(TimeoutError):
        remote_l3_worker._start_session(_manifest())

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

    reply, proc = remote_l3_worker._start_session(_manifest())

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

    reply, proc = remote_l3_worker._start_session(_manifest())

    # A failed handshake is killed+reaped exactly once inside _start_session, so
    # the caller gets no runner to reclaim.
    assert reply["ok"] is False
    assert proc is None
    assert reclaimed == [fake_proc]


def test_run_session_bounds_post_ready_command_accept(monkeypatch):
    class FakeWorker:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def init(self, *args, **kwargs):
            pass

        def close(self):
            self.closed = True

    class FakeCommandSock:
        def __init__(self):
            self.timeout = None
            self.closed = False

        def getsockname(self):
            return ("127.0.0.1", 12345)

        def settimeout(self, timeout):
            self.timeout = timeout

        def accept(self):
            if self.timeout is None:
                raise AssertionError("command accept has no timeout")
            raise socket.timeout("command attach timed out")

        def close(self):
            self.closed = True

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
        assert remote_l3_session.run_session(_manifest(), ready_w) == 1
        assert command_sock.timeout == 0.01
    finally:
        os.close(ready_r)


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
