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
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
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


def _read_runner_ready(fd: int) -> dict[str, Any]:
    chunks = bytearray()
    while True:
        b = os.read(fd, 1)
        if not b:
            break
        if b == b"\n":
            break
        chunks.extend(b)
    if not chunks:
        raise RuntimeError("session runner exited before sending ready payload")
    return json.loads(bytes(chunks).decode("utf-8"))


def _start_session(manifest: dict[str, Any]) -> dict[str, Any]:
    _validate_manifest(manifest)
    ready_r, ready_w = os.pipe()
    manifest_path = ""
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
        )
        os.close(ready_w)
        ready_w = -1
        ready = _read_runner_ready(ready_r)
        ready["pid"] = int(proc.pid)
        if not ready.get("ok", False):
            proc.wait(timeout=5)
        return ready
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


def serve(host: str, port: int) -> int:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen()
    try:
        while True:
            conn, _addr = server.accept()
            with conn:
                try:
                    manifest = _read_json(conn)
                    _send_json(conn, _start_session(manifest))
                except BaseException as exc:  # noqa: BLE001
                    _send_json(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        server.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    ns = parser.parse_args(argv)
    return serve(ns.host, ns.port)


if __name__ == "__main__":
    sys.exit(main())
