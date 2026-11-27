# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run an MPI Direct CPU smoke with real rank 0, prefill and decode processes."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

_HOST = "127.0.0.1"
_PREFIX = "SIMPLER_MPI_DIRECT_"
_HEADER = struct.Struct("!I")
_TIMEOUT_S = 25
_REQUEST = bytes(range(256)) * 1024


def _write(sock, **message):
    data = json.dumps(message).encode()
    sock.sendall(_HEADER.pack(len(data)) + data)


def _read(stream):
    (length,) = _HEADER.unpack(stream.read(_HEADER.size))
    return json.loads(stream.read(length))


def _connect(port, token, expected_rank):
    deadline = time.monotonic() + _TIMEOUT_S
    while True:
        try:
            sock = socket.create_connection((_HOST, port), timeout=2)
            break
        except ConnectionRefusedError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)
    sock.settimeout(_TIMEOUT_S)
    stream = sock.makefile("rb")
    _write(sock, type="hello", token=token)
    assert _read(stream) == {"type": "hello", "rank": expected_rank}
    return sock, stream


def _echo(listener):
    with listener:
        data, _ = listener.accept()
        with data, data.makefile("rb") as incoming:
            request = incoming.read()
            assert request == _REQUEST
            data.sendall(request[::-1])


def _probe():
    peers = json.loads(os.environ[_PREFIX + "PEERS"])
    with socket.create_connection((_HOST, peers["d0"]), timeout=_TIMEOUT_S) as data, data.makefile("rb") as response:
        data.sendall(_REQUEST[:11])
        data.sendall(_REQUEST[11:])
        data.shutdown(socket.SHUT_WR)
        assert response.read() == _REQUEST[::-1]


def _worker():
    role = os.environ[_PREFIX + "ROLE"]
    if role == "decode":
        listener = socket.socket()
        listener.bind((_HOST, int(os.environ[_PREFIX + "BIND_PORT"])))
        listener.listen()
        threading.Thread(target=_echo, args=(listener,), daemon=True).start()
    rank = int(os.environ[_PREFIX + "RANK"])
    sock, stream = _connect(int(os.environ[_PREFIX + "PORT"]), os.environ[_PREFIX + "TOKEN"], rank)
    with sock, stream:
        _write(sock, type="status", state="ready", payload={"boot": f"boot-{rank}"})
        while True:
            frame = _read(stream)
            if frame["type"] == "shutdown":
                return
            assert frame["type"] == "broadcast"
            if frame["payload"] == {"probe": True}:
                assert role == "prefill"
                _probe()
            _write(sock, type="status", state="ready", payload={"observed_broadcast": frame["id"]})


def _ports():
    with socket.socket() as router, socket.socket() as decode:
        for sock in (router, decode):
            sock.bind((_HOST, 0))
        return router.getsockname()[1], decode.getsockname()[1]


def _exercise(sock, stream):
    expected = {1: ("p0", "prefill"), 2: ("d0", "decode")}
    ready = {}
    while len(ready) < 2:
        status = _read(stream)
        assert status["type"] == "status", status
        assert (status["instance_id"], status["role"]) == expected[status["rank"]], status
        if status["state"] == "ready":
            ready[status["rank"]] = status["payload"]["boot"]
    assert ready == {1: "boot-1", 2: "boot-2"}

    for scope, ranks in (("all", [1, 2]), ("prefill", [1]), ("decode", [2])):
        payload = {"probe": True} if scope == "prefill" else {}
        _write(sock, type="broadcast", id=scope, scope=scope, payload=payload)
        observed = []
        acked = False
        while not acked or len(observed) < len(ranks):
            message = _read(stream)
            if message["type"] == "response":
                assert message == {"type": "response", "id": scope, "ok": True, "payload": {"ranks": ranks}}, message
                acked = True
            else:
                assert message["state"] == "ready" and message["payload"]["observed_broadcast"] == scope, message
                observed.append(message["rank"])
        assert sorted(observed) == ranks

    _write(sock, type="shutdown", id="stop")
    exited = []
    while True:
        message = _read(stream)
        if message["type"] == "response":
            assert message == {"type": "response", "id": "stop", "ok": True}, message
            break
        if message["state"] == "exited":
            assert message["payload"]["returncode"] == 0, message
            exited.append(message["rank"])
    assert exited == [1, 2], exited


def _supervise(mpirun):
    router_port, bind_port = _ports()
    worker = [sys.executable, str(Path(__file__).resolve()), "--worker"]
    config = {
        "controller_host": _HOST,
        "router_port": router_port,
        "startup_timeout_s": _TIMEOUT_S,
        "session_timeout_s": 5,
        "services": [
            {"id": "p0", "role": "prefill", "host": _HOST, "argv": worker},
            {"id": "d0", "role": "decode", "host": _HOST, "bind_port": bind_port, "argv": worker},
        ],
    }
    token = secrets.token_urlsafe(32)
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "service.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "simpler.mpi_direct_supervisor",
                "--service-config",
                str(config_path),
                "--mpirun",
                mpirun,
            ],
            env={**os.environ, _PREFIX + "ROUTER_TOKEN": token},
        )
        try:
            sock, stream = _connect(router_port, token, 0)
            with sock, stream:
                _exercise(sock, stream)
        finally:
            exit_code = proc.wait(timeout=_TIMEOUT_S + 15)
        assert exit_code == 0, exit_code
    print("PASS 3 MPI ranks; P→D and D→P byte streams; all/P/D broadcasts; ordered P→D shutdown")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    opts = parser.parse_args()
    if opts.worker:
        _worker()
    else:
        _supervise(opts.mpirun)
