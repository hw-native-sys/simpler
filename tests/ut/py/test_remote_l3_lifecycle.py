# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import socket
import threading

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
            self.killed = False
            self.wait_calls = 0

        def poll(self):
            return -9 if self.killed else None

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            self.wait_calls += 1
            return -9 if self.killed else 0

    fake_proc = FakePopen()

    def fake_popen(*args, **kwargs):
        return fake_proc

    def fake_read_ready(fd, timeout_s):
        raise TimeoutError("ready timeout")

    monkeypatch.setattr(remote_l3_worker.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(remote_l3_worker, "_read_runner_ready", fake_read_ready)

    with pytest.raises(TimeoutError):
        remote_l3_worker._start_session(_manifest())

    assert fake_proc.killed
    assert fake_proc.wait_calls >= 1


def test_start_session_reaps_successful_runner(monkeypatch):
    class FakePopen:
        pid = 12345

        def __init__(self, *args, **kwargs):
            self.wait_calls = 0

        def wait(self, timeout=None):
            self.wait_calls += 1
            return 0

    class FakeThread:
        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self):
            self.target(*self.args)

    fake_proc = FakePopen()

    monkeypatch.setattr(remote_l3_worker.subprocess, "Popen", lambda *args, **kwargs: fake_proc)
    monkeypatch.setattr(remote_l3_worker, "_read_runner_ready", lambda fd, timeout_s: {"ok": True})
    monkeypatch.setattr(remote_l3_worker.threading, "Thread", FakeThread)

    reply = remote_l3_worker._start_session(_manifest())

    assert reply["ok"] is True
    assert reply["pid"] == fake_proc.pid
    assert fake_proc.wait_calls == 1


def test_run_session_bounds_post_ready_command_accept(monkeypatch):
    class FakeWorker:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def init(self):
            pass

        def _start_hierarchical(self):
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

    remote_l3_session._health_loop(sock, stop, session_id=1, worker_id=0)

    assert sock.closed
    assert conn.closed
