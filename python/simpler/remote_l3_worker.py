# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Remote L3 control daemon."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import select
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any


def _read_exact(sock: socket.socket, n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise EOFError("remote daemon socket closed")
        data.extend(chunk)
    return bytes(data)


def _read_json(sock: socket.socket) -> dict[str, Any]:
    size = struct.unpack("<I", _read_exact(sock, 4))[0]
    if size > 16 * 1024 * 1024:
        raise ValueError("remote daemon manifest exceeds maximum")
    return json.loads(_read_exact(sock, size).decode("utf-8"))


def _send_json(sock: socket.socket, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    sock.sendall(struct.pack("<I", len(data)) + data)


def _validate_manifest(manifest: dict[str, Any]) -> None:
    required = ["session_id", "worker_id", "parent_worker_level", "remote_worker_level", "platform", "transport"]
    for key in required:
        if key not in manifest:
            raise ValueError(f"manifest missing {key}")
    if int(manifest["session_id"]) == 0:
        raise ValueError("manifest session_id must be non-zero")
    if int(manifest["worker_id"]) < 0:
        raise ValueError("manifest worker_id must be non-negative")
    if int(manifest["remote_worker_level"]) != 3:
        raise ValueError("manifest remote_worker_level must be 3")
    if not str(manifest["platform"]):
        raise ValueError("manifest platform must be non-empty")
    if str(manifest["transport"]) != "sim":
        raise ValueError("only sim transport is accepted by simpler-remote-worker")


def _session_timeout_s(manifest: dict[str, Any]) -> float:
    timeout_s = float(manifest.get("session_timeout_s", 30.0))
    if not (timeout_s > 0 and math.isfinite(timeout_s)):
        raise ValueError("manifest session_timeout_s must be a positive finite number of seconds")
    return timeout_s


def _startup_remaining_s(manifest: dict[str, Any]) -> float:
    # The parent's remaining slice of the single root startup budget bounds how
    # long the runner may take to bring up its subtree. Absent only from a
    # pre-P0.3 parent; fall back to the runtime command timeout then. The
    # fallback is evaluated only when the key is absent, so a valid
    # startup_remaining_s is not held hostage to an invalid session_timeout_s.
    if "startup_remaining_s" not in manifest:
        return _session_timeout_s(manifest)
    remaining_s = float(manifest["startup_remaining_s"])
    if not (remaining_s > 0 and math.isfinite(remaining_s)):
        raise ValueError("manifest startup_remaining_s must be a positive finite number of seconds")
    return remaining_s


def _read_runner_ready(fd: int, timeout_s: float) -> dict[str, Any]:
    chunks = bytearray()
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("session runner did not send ready payload before timeout")
        readable, _writable, _error = select.select([fd], [], [], remaining)
        if not readable:
            raise TimeoutError("session runner did not send ready payload before timeout")
        b = os.read(fd, 1)
        if not b:
            break
        if b == b"\n":
            break
        chunks.extend(b)
    if not chunks:
        raise RuntimeError("session runner exited before sending ready payload")
    return json.loads(bytes(chunks).decode("utf-8"))


def _wait_or_kill_runner(proc: subprocess.Popen[Any], *, timeout_s: float = 5.0) -> None:
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    # Cooperative: SIGTERM lets the runner close its inner Worker and reap its
    # own L3->L2 subtree (unlinking nested shms) before it exits.
    with contextlib.suppress(OSError):
        proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    # Hard backstop: SIGKILL the whole runner process group. The runner is a
    # session leader (start_new_session) and its inner chip/sub children inherit
    # its group, so this reaps the entire subtree rather than orphaning it.
    with contextlib.suppress(OSError, ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(OSError):
        proc.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=timeout_s)


def _reap_session_runner(proc: subprocess.Popen[Any]) -> None:
    try:
        proc.wait()
    except BaseException:  # noqa: BLE001
        pass


def _start_session(manifest: dict[str, Any]) -> tuple[dict[str, Any], subprocess.Popen[Any] | None]:
    # Returns (reply, proc). proc is the live, ready runner handle when the reply
    # is ok — the caller owns it: hand it to a background reaper once the reply
    # send to the parent succeeds, or reclaim it if the send raises. proc is None
    # when no runner survives (a failed handshake has already been killed and
    # reaped here), so a failed send then leaves nothing to reclaim. A successful
    # send only means the bytes were queued locally, not that the parent read
    # them; unobserved receipt would need an ACK / lease, which sim does not have.
    _validate_manifest(manifest)
    # Both numeric timeouts are validated before any spawn resource (ready pipe,
    # manifest tempfile, runner Popen) exists: the runner is never launched only
    # to die on an invalid session_timeout_s or startup_remaining_s.
    _session_timeout_s(manifest)
    # The runner must publish ready within the parent's remaining startup budget,
    # not a fresh full command timeout.
    timeout_s = _startup_remaining_s(manifest)
    ready_r, ready_w = os.pipe()
    manifest_path = ""
    proc: subprocess.Popen[Any] | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", prefix="simpler-remote-l3-", suffix=".json", delete=False
        ) as f:
            manifest_path = f.name
            json.dump(manifest, f, sort_keys=True)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "simpler.remote_l3_session",
                "--manifest",
                manifest_path,
                "--ready-fd",
                str(ready_w),
            ],
            pass_fds=(ready_w,),
            close_fds=True,
            # Own session/group so the daemon can killpg the whole runner
            # subtree (runner + eagerly-forked inner L3->L2 children).
            start_new_session=True,
        )
        os.close(ready_w)
        ready_w = -1
        try:
            ready = _read_runner_ready(ready_r, timeout_s)
        except BaseException:
            _wait_or_kill_runner(proc)
            raise
        ready["pid"] = int(proc.pid)
        if not ready.get("ok", False):
            _wait_or_kill_runner(proc)
            return ready, None
        return ready, proc
    finally:
        if ready_w >= 0:
            try:
                os.close(ready_w)
            except OSError:
                pass
        try:
            os.close(ready_r)
        except OSError:
            pass
        if manifest_path:
            try:
                os.unlink(manifest_path)
            except OSError:
                pass


def _serve_connection(conn: socket.socket) -> None:
    # One parent connection, fully isolated: any Exception on this socket affects
    # only this session, never the shared accept loop. KeyboardInterrupt /
    # SystemExit are BaseException, not Exception, so they propagate out for a
    # clean daemon shutdown instead of turning into a spurious error reply. A live
    # runner is disposed of exactly once on every exit path — handed to the reaper
    # only once its reply send returns, otherwise reclaimed by the finally — so no
    # session runner is orphaned (until its own timeout) or double-reaped, even
    # when the send fails, the reaper cannot be launched, or a control exception
    # unwinds mid-connection.
    proc: subprocess.Popen[Any] | None = None
    try:
        try:
            manifest = _read_json(conn)
            reply, proc = _start_session(manifest)
        except Exception as exc:  # noqa: BLE001
            # Handshake failed (bad manifest, runner start error); _start_session
            # already reaped any runner it started. Best-effort error reply,
            # swallowing only a send that itself fails on a dead parent.
            with contextlib.suppress(OSError):
                _send_json(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return
        try:
            _send_json(conn, reply)
            if proc is not None:
                threading.Thread(target=_reap_session_runner, args=(proc,), daemon=True).start()
                proc = None
        except Exception:  # noqa: BLE001
            # Local send raised (parent gone) or the reaper thread could not be
            # started; the finally reclaims the still-owned runner.
            return
    finally:
        if proc is not None:
            _wait_or_kill_runner(proc)


def _serve_loop(server: socket.socket) -> None:
    while True:
        try:
            conn, _addr = server.accept()
        except OSError:
            # Listener closed (shutdown); stop accepting.
            return
        with conn:
            _serve_connection(conn)


def serve(host: str, port: int) -> int:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen()
    try:
        _serve_loop(server)
    finally:
        server.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    ns = parser.parse_args(argv)
    return serve(ns.host, ns.port)


if __name__ == "__main__":
    sys.exit(main())
