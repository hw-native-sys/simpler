# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Rank-role runtime for direct MPI control between L4 and real L3 workers."""

from __future__ import annotations

import argparse
import base64
import contextlib
import importlib
import json
import os
import runpy
import socket
import struct
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from .mpi_direct_protocol import MPI_DIRECT_GATE_MAX_BYTES, MPI_DIRECT_STARTUP_TOKEN_ENV, MpiDirectTag
from .mpi_direct_topology import MpiDirectExecutorSpec, MpiDirectTopology, load_runtime_manifest_data
from .remote_l3_limits import FRAME_HEADER_BYTES, MAX_FRAME_BYTES

COMMAND_REQUEST_TAG = int(MpiDirectTag.COMMAND_REQUEST)
COMMAND_REPLY_TAG = int(MpiDirectTag.COMMAND_REPLY)
HEALTH_TAG = int(MpiDirectTag.HEALTH)
LIFECYCLE_TAG = int(MpiDirectTag.LIFECYCLE)
_MPI_POLL_INTERVAL_S = 0.001
_MPI_GATE_RETRY_INTERVAL_S = 0.01


def _launcher_rank() -> int:
    names = (
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "PMIX_RANK",
        "MV2_COMM_WORLD_RANK",
        "SLURM_PROCID",
    )
    values = {name: int(os.environ[name]) for name in names if name in os.environ}
    if not values:
        raise RuntimeError(f"MPI launcher did not provide a pre-init rank in any of {names}")
    if len(set(values.values())) != 1:
        raise RuntimeError(f"MPI launcher rank variables disagree: {values}")
    return next(iter(values.values()))


def _open_fds() -> tuple[int, ...]:
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.is_dir():
        fd_dir = Path("/dev/fd")
    if not fd_dir.is_dir():
        raise RuntimeError("direct MPI executor requires /proc/self/fd or /dev/fd for launcher FD isolation")
    return tuple(sorted(int(path.name) for path in fd_dir.iterdir() if path.name.isdigit() and int(path.name) >= 3))


def _import_mpi():
    try:
        import mpi4py  # noqa: PLC0415

        mpi4py.rc.initialize = False
        mpi4py.rc.finalize = False
        mpi4py.rc.thread_level = "serialized"
        from mpi4py import MPI  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("direct MPI runtime requires the optional mpi4py dependency") from exc
    return MPI


def _init_mpi(MPI, expected_rank: int, expected_world_size: int):
    if MPI.Is_initialized():
        raise RuntimeError("MPI was initialized before simpler.mpi_direct_runtime called MPI.Init_thread")
    provided = MPI.Init_thread(required=MPI.THREAD_SERIALIZED)
    if provided < MPI.THREAD_SERIALIZED:
        raise RuntimeError(f"MPI_THREAD_SERIALIZED is required, but MPI provided thread level {provided}")
    world = MPI.COMM_WORLD
    world.Set_errhandler(MPI.ERRORS_RETURN)
    if int(world.Get_rank()) != expected_rank:
        raise RuntimeError("pre-init launcher rank does not match MPI_COMM_WORLD rank")
    if int(world.Get_size()) != expected_world_size:
        raise RuntimeError("MPI_COMM_WORLD size does not match the static topology")
    return world


def _gate_send(sock: socket.socket, message: dict[str, object]) -> None:
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    if len(payload) > MPI_DIRECT_GATE_MAX_BYTES:
        raise ValueError("MPI startup gate message is too large")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def _gate_recv(sock: socket.socket) -> dict[str, object]:
    header = b""
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            raise ConnectionError("MPI startup gate closed before response")
        header += chunk
    length = struct.unpack("!I", header)[0]
    if length <= 0 or length > MPI_DIRECT_GATE_MAX_BYTES:
        raise ValueError("invalid MPI startup gate message length")
    payload = bytearray()
    while len(payload) < length:
        chunk = sock.recv(length - len(payload))
        if not chunk:
            raise ConnectionError("MPI startup gate closed while reading response")
        payload.extend(chunk)
    message = json.loads(payload)
    if not isinstance(message, dict):
        raise ValueError("MPI startup gate response must be an object")
    return message


def _pre_mpi_gate(
    host: str,
    port: int,
    token: str,
    rank: int,
    timeout_s: float,
    error: BaseException | None = None,
) -> None:
    deadline = time.monotonic() + float(timeout_s)
    last_error: BaseException | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("MPI startup gate connection timed out") from last_error
        try:
            with socket.create_connection((host, int(port)), timeout=min(remaining, 1.0)) as sock:
                sock.settimeout(max(1.0, remaining))
                _gate_send(
                    sock,
                    {
                        "token": token,
                        "rank": int(rank),
                        "state": "failed" if error is not None else "ready",
                        "error": "" if error is None else f"{type(error).__name__}: {error}",
                    },
                )
                if error is not None:
                    return
                response = _gate_recv(sock)
                if response.get("token") != token:
                    raise RuntimeError("MPI startup gate token mismatch")
                state = response.get("state")
                if state != "go_mpi":
                    raise RuntimeError(str(response.get("error") or "MPI startup gate rejected rank"))
                return
        except (OSError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            retry_remaining = deadline - time.monotonic()
            if retry_remaining > 0:
                time.sleep(min(_MPI_GATE_RETRY_INTERVAL_S, retry_remaining))
            continue


class _ControllerProgress:
    def __init__(self, MPI, world, hub) -> None:
        self._MPI = MPI
        self._world = world
        self._hub = hub
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="simpler-mpi-direct-progress", daemon=True)
        self._error: BaseException | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout_s: float) -> None:
        self._stop.set()
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            self._hub.fail("MPI progress thread did not drain before shutdown deadline")
            raise TimeoutError("MPI progress thread did not stop")
        if self._error is not None:
            raise RuntimeError("MPI progress failed") from self._error

    def _receive_one(self) -> bool:
        status = self._MPI.Status()
        message = self._world.improbe(
            source=self._MPI.ANY_SOURCE,
            tag=self._MPI.ANY_TAG,
            status=status,
        )
        if message is None:
            return False
        count = int(status.Get_count(self._MPI.BYTE))
        if count < FRAME_HEADER_BYTES or count > MAX_FRAME_BYTES:
            raise RuntimeError(f"inbound MPI frame length {count} is outside the SLR3 bounds")
        frame = bytearray(count)
        message.Recv([frame, self._MPI.BYTE])
        self._hub.deliver(int(status.Get_source()), int(status.Get_tag()), bytes(frame))
        return True

    def _run(self) -> None:
        in_flight: list[tuple[Any, int, bytes]] = []
        try:
            while True:
                did_work = False
                remaining: list[tuple[Any, int, bytes]] = []
                for request, ticket, frame in in_flight:
                    if request.Test():
                        self._hub.complete_outbound(ticket)
                        did_work = True
                    else:
                        remaining.append((request, ticket, frame))
                in_flight = remaining

                outbound = self._hub.poll_outbound(0.0)
                if outbound is not None:
                    ticket, target_rank, tag, frame = outbound
                    frame = bytes(frame)
                    request = self._world.Isend(
                        [frame, self._MPI.BYTE],
                        dest=int(target_rank),
                        tag=int(tag),
                    )
                    in_flight.append((request, int(ticket), frame))
                    did_work = True

                did_work = self._receive_one() or did_work
                if self._stop.is_set() and not in_flight and self._hub.pending_frame_bytes == 0:
                    return
                if not did_work:
                    time.sleep(_MPI_POLL_INTERVAL_S)
        except BaseException as exc:  # noqa: BLE001
            self._error = exc
            with contextlib.suppress(BaseException):
                self._hub.fail(f"MPI progress failure: {type(exc).__name__}: {exc}")


class MpiDirectControllerContext:
    def __init__(self, MPI, world, topology: MpiDirectTopology, session_id: int) -> None:
        from _task_interface import _MpiDirectTransportHub  # noqa: PLC0415

        self.topology = topology
        self.session_id = int(session_id)
        self.cluster_id = f"mpi-direct-{self.session_id:016x}"
        self._hub = _MpiDirectTransportHub(topology.max_pending_frame_bytes)
        for spec in topology.executors:
            self._hub.register_route(spec.worker_id, spec.rank, self.session_id, spec.comm_profile)
        self._progress = _ControllerProgress(MPI, world, self._hub)
        self._workers: list[Any] = []
        self._closed = False
        self._progress.start()

    def attach(self, worker) -> tuple[int, ...]:
        from .worker import _MpiDirectWorkerSpec  # noqa: PLC0415

        if self._closed:
            raise RuntimeError("MPI direct controller context is closed")
        if worker.level < 4:
            raise TypeError("MPI direct controller requires Worker(level>=4)")
        if self._workers:
            raise RuntimeError("MPI direct controller context supports exactly one attached L4 Worker")
        worker._global_cluster_id = self.cluster_id  # noqa: SLF001 -- context owns the direct-MPI topology
        worker_ids = []
        for executor in self.topology.executors:
            worker_id = worker._add_mpi_direct_worker(
                _MpiDirectWorkerSpec(
                    worker_id=executor.worker_id,
                    mpi_rank=executor.rank,
                    session_id=self.session_id,
                    host=executor.host,
                    comm_profile=executor.comm_profile,
                    platform=executor.platform,
                    runtime=executor.runtime,
                    device_ids=executor.device_ids,
                    global_device_ranks=executor.global_device_ranks,
                    hub=self._hub,
                    attach_timeout_s=self.topology.startup_timeout_s,
                    runtime_timeout_s=self.topology.session_timeout_s,
                )
            )
            worker_ids.append(worker_id)
        self._workers.append(worker)
        return tuple(worker_ids)

    def create_worker(self, **config):
        from .worker import Worker  # noqa: PLC0415

        worker = Worker(level=4, **config)
        self.attach(worker)
        return worker

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close_errors: list[BaseException] = []
        for worker in reversed(self._workers):
            try:
                worker.close()
            except BaseException as exc:  # noqa: BLE001
                close_errors.append(exc)
        try:
            self._progress.stop(self.topology.session_timeout_s)
        finally:
            self._hub.close()
        if close_errors:
            raise RuntimeError(f"failed to close {len(close_errors)} attached L4 Worker(s)") from close_errors[0]


class _ExecutorFrameSocket:
    def __init__(self, MPI, world, spec: MpiDirectExecutorSpec, session_id: int, heartbeat_interval_s: float) -> None:
        self._MPI = MPI
        self._world = world
        self._spec = spec
        self._session_id = int(session_id)
        self._heartbeat_interval_s = float(heartbeat_interval_s)
        self._mpi_mu = threading.Lock()
        self._receive_buffer = bytearray()
        self._health_stop = threading.Event()
        self._health_thread: threading.Thread | None = None
        self._health_error: BaseException | None = None
        self._health_sequence = 0

    def _send(self, data: bytes, tag: int) -> None:
        with self._mpi_mu:
            request = self._world.Isend([data, self._MPI.BYTE], dest=0, tag=tag)
            request.Wait()

    def _start_health(self) -> None:
        if self._health_thread is not None:
            return
        self._health_thread = threading.Thread(target=self._health_loop, name="simpler-mpi-direct-health", daemon=True)
        self._health_thread.start()

    def _health_loop(self) -> None:
        from .remote_l3_protocol import FrameHeader, FrameType, encode_frame  # noqa: PLC0415

        try:
            while not self._health_stop.wait(self._heartbeat_interval_s):
                self._health_sequence += 1
                frame = encode_frame(
                    FrameHeader(
                        FrameType.HEALTH,
                        self._session_id,
                        self._spec.worker_id,
                        self._health_sequence,
                    ),
                    b"",
                )
                self._send(frame, HEALTH_TAG)
        except BaseException as exc:  # noqa: BLE001
            self._health_error = exc

    def sendall(self, data: bytes) -> None:
        from .remote_l3_protocol import FrameType, decode_frame  # noqa: PLC0415

        raw = bytes(data)
        frame = decode_frame(raw)
        if frame.header.session_id != self._session_id or frame.header.worker_id != self._spec.worker_id:
            raise RuntimeError("executor outbound SLR3 frame identity mismatch")
        if frame.header.frame_type in (FrameType.COMPLETION, FrameType.CONTROL_REPLY):
            tag = COMMAND_REPLY_TAG
        elif frame.header.frame_type == FrameType.HELLO:
            tag = LIFECYCLE_TAG
        elif frame.header.frame_type == FrameType.HEALTH:
            tag = HEALTH_TAG
        else:
            raise RuntimeError(f"executor cannot send SLR3 frame type {frame.header.frame_type.name}")
        self._send(raw, tag)
        if frame.header.frame_type == FrameType.HELLO:
            self._start_health()

    def _receive_message(self) -> bytes:
        from .remote_l3_protocol import FrameType, decode_frame  # noqa: PLC0415

        while True:
            if self._health_error is not None:
                raise RuntimeError("executor heartbeat failed") from self._health_error
            status = self._MPI.Status()
            with self._mpi_mu:
                message = self._world.improbe(source=0, tag=self._MPI.ANY_TAG, status=status)
                if message is not None:
                    count = int(status.Get_count(self._MPI.BYTE))
                    if count < FRAME_HEADER_BYTES or count > MAX_FRAME_BYTES:
                        raise RuntimeError(f"controller MPI frame length {count} is outside the SLR3 bounds")
                    frame = bytearray(count)
                    message.Recv([frame, self._MPI.BYTE])
            if message is None:
                time.sleep(_MPI_POLL_INTERVAL_S)
                continue
            tag = int(status.Get_tag())
            decoded = decode_frame(bytes(frame))
            expected_tag = LIFECYCLE_TAG if decoded.header.frame_type == FrameType.SHUTDOWN else COMMAND_REQUEST_TAG
            if tag != expected_tag:
                raise RuntimeError("controller MPI tag does not match SLR3 request type")
            if decoded.header.session_id != self._session_id or decoded.header.worker_id != self._spec.worker_id:
                raise RuntimeError("controller SLR3 frame identity mismatch")
            return bytes(frame)

    def recv(self, size: int) -> bytes:
        if size < 0:
            raise ValueError("recv size must be non-negative")
        if not self._receive_buffer:
            self._receive_buffer.extend(self._receive_message())
        result = bytes(self._receive_buffer[:size])
        del self._receive_buffer[:size]
        return result

    def close(self) -> None:
        self._health_stop.set()
        if self._health_thread is not None:
            self._health_thread.join(self._heartbeat_interval_s + 1.0)
            if self._health_thread.is_alive():
                raise RuntimeError("executor heartbeat thread did not stop")
        if self._health_error is not None:
            raise RuntimeError("executor heartbeat failed") from self._health_error


def _load_controller(target: str) -> Callable[[MpiDirectControllerContext], Any]:
    path = Path(target)
    if path.is_file():
        namespace = runpy.run_path(str(path.resolve()), run_name="simpler_mpi_direct_controller")
        callback = namespace.get("main")
    else:
        if ":" not in target:
            raise ValueError("controller must be a Python file or module:callable")
        module_name, qualname = target.split(":", 1)
        callback = importlib.import_module(module_name)
        for part in qualname.split("."):
            callback = getattr(callback, part)
    if not callable(callback):
        raise TypeError("controller target must resolve to a callable main(context)")
    return callback


def _run_controller(MPI, world, topology: MpiDirectTopology, session_id: int, target: str) -> None:
    context = MpiDirectControllerContext(MPI, world, topology, session_id)
    try:
        _load_controller(target)(context)
    finally:
        context.close()


def _run_executor(MPI, world, topology: MpiDirectTopology, session_id: int, spec, worker) -> None:
    from .remote_l3_session import _run_command_loop  # noqa: PLC0415

    channel = _ExecutorFrameSocket(MPI, world, spec, session_id, topology.heartbeat_interval_s)
    manifest = {
        "session_id": session_id,
        "worker_id": spec.worker_id,
        "transport": spec.comm_profile,
        "comm_profile": spec.comm_profile,
        "cluster_id": f"mpi-direct-{session_id:016x}",
        "session_timeout_s": topology.session_timeout_s,
        "heartbeat_interval_s": topology.heartbeat_interval_s,
        "node_rank": spec.rank - 1,
        "node_count": len(topology.executors),
        "platform": spec.platform,
        "runtime": spec.runtime,
        "device_ids": list(spec.device_ids),
        "num_sub_workers": spec.num_sub_workers,
        "global_device_ranks": list(spec.global_device_ranks),
    }
    try:
        _run_command_loop(channel, manifest, worker, {}, {})  # type: ignore[arg-type]
    finally:
        try:
            channel.close()
        finally:
            worker.close()


def run_runtime(  # noqa: PLR0912 -- startup gate, role dispatch, and MPI teardown are one ordered lifecycle
    topology_path: str | None,
    session_id: int | None,
    controller: str,
    *,
    manifest_json: str | None = None,
    startup_host: str | None = None,
    startup_port: int | None = None,
) -> int:
    if manifest_json is not None:
        try:
            data = json.loads(base64.urlsafe_b64decode(manifest_json.encode("ascii")))
        except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid inline MPI direct runtime manifest") from exc
        topology, manifest_session_id = load_runtime_manifest_data(data)
        if session_id is not None and int(session_id) != manifest_session_id:
            raise ValueError("session_id does not match inline runtime manifest")
        session_id = manifest_session_id
    elif topology_path is not None and session_id is not None:
        topology = MpiDirectTopology.load(topology_path)
        session_id = int(session_id)
    else:
        raise ValueError("runtime requires either topology+session-id or manifest-json")
    session_id = int(session_id)
    if session_id == 0:
        raise ValueError("session_id must be non-zero")
    rank = _launcher_rank()
    if rank < 0 or rank >= topology.world_size:
        raise ValueError("launcher rank is outside the topology world")
    worker = None
    startup_error: BaseException | None = None
    try:
        if rank != 0:
            from .worker import Worker  # noqa: PLC0415

            spec = topology.executor_for_rank(rank)
            launcher_fds = _open_fds()
            worker = Worker(
                level=3,
                platform=spec.platform,
                runtime=spec.runtime,
                device_ids=spec.device_ids,
                num_sub_workers=spec.num_sub_workers,
                comm_profile=spec.comm_profile,
                global_device_ranks=spec.global_device_ranks,
                startup_timeout_s=topology.startup_timeout_s,
                fork_child_close_fds=launcher_fds,
            )
            worker.init()
    except BaseException as exc:
        startup_error = exc
        if worker is not None:
            with contextlib.suppress(BaseException):
                worker.close()
    if startup_host is not None or startup_port is not None:
        startup_token = os.environ.get(MPI_DIRECT_STARTUP_TOKEN_ENV)
        if not (startup_host and startup_port and startup_token):
            raise ValueError(
                f"startup gate requires host, port, and {MPI_DIRECT_STARTUP_TOKEN_ENV} environment variable"
            )
        _pre_mpi_gate(startup_host, int(startup_port), startup_token, rank, topology.startup_timeout_s, startup_error)
    if startup_error is not None:
        raise startup_error

    MPI = _import_mpi()
    world = None
    try:
        world = _init_mpi(MPI, rank, topology.world_size)
        if rank == 0:
            _run_controller(MPI, world, topology, session_id, controller)
        else:
            assert worker is not None
            _run_executor(MPI, world, topology, session_id, topology.executor_for_rank(rank), worker)
        return 0
    except BaseException:
        print(
            f"[mpi-direct rank={rank} host={socket.gethostname()}] fatal error after MPI initialization",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        with contextlib.suppress(BaseException):
            (world if world is not None else MPI.COMM_WORLD).Abort(1)
        raise
    finally:
        if MPI.Is_initialized() and not MPI.Is_finalized():
            MPI.Finalize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology")
    parser.add_argument("--manifest-json")
    parser.add_argument("--session-id", type=int)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--startup-host")
    parser.add_argument("--startup-port", type=int)
    ns = parser.parse_args(argv)
    return run_runtime(
        ns.topology,
        ns.session_id,
        ns.controller,
        manifest_json=ns.manifest_json,
        startup_host=ns.startup_host,
        startup_port=ns.startup_port,
    )


if __name__ == "__main__":
    raise SystemExit(main())
