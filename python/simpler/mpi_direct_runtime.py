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
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from .mpi_direct_cleanup import MpiDirectCleanupReport, _RankCleanupReporter, cleanup_failure
from .mpi_direct_protocol import (
    MPI_DIRECT_STARTUP_TOKEN_ENV,
    MpiDirectTag,
    _assert_native_wire_constants,
    _gate_recv,
    _gate_send,
)
from .mpi_direct_topology import MpiDirectExecutorSpec, MpiDirectTopology, load_runtime_manifest_data
from .remote_l3_limits import FRAME_HEADER_BYTES, MAX_FRAME_BYTES

COMMAND_REQUEST_TAG = int(MpiDirectTag.COMMAND_REQUEST)
COMMAND_REPLY_TAG = int(MpiDirectTag.COMMAND_REPLY)
HEALTH_TAG = int(MpiDirectTag.HEALTH)
LIFECYCLE_TAG = int(MpiDirectTag.LIFECYCLE)
_MPI_GATE_RETRY_INTERVAL_S = 0.01


class _MpiDirectCleanupError(RuntimeError):
    pass


def _launcher_rank() -> int:
    names = (
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "PMIX_RANK",
        "MV2_COMM_WORLD_RANK",
    )
    values = {name: int(os.environ[name]) for name in names if name in os.environ}
    if not values:
        if "SLURM_PROCID" in os.environ:
            return int(os.environ["SLURM_PROCID"])
        raise RuntimeError(f"MPI launcher did not provide a pre-init rank in any of {(*names, 'SLURM_PROCID')}")
    if len(set(values.values())) != 1:
        raise RuntimeError(f"MPI launcher rank variables disagree: {values}")
    return next(iter(values.values()))


def _open_fds() -> tuple[tuple[int, tuple[int, int, int]], ...]:
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.is_dir():
        fd_dir = Path("/dev/fd")
    if not fd_dir.is_dir():
        raise RuntimeError("direct MPI executor requires /proc/self/fd or /dev/fd for launcher FD isolation")
    result = []
    for path in fd_dir.iterdir():
        if not path.name.isdigit() or int(path.name) < 3:
            continue
        fd = int(path.name)
        try:
            stat = os.fstat(fd)
        except OSError:
            continue
        result.append((fd, (stat.st_dev, stat.st_ino, stat.st_mode)))
    return tuple(sorted(result))


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
                response = _gate_recv(sock, deadline)
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


def _receive_frame(MPI, world, source: int):
    status = MPI.Status()
    message = world.improbe(source=source, tag=MPI.ANY_TAG, status=status)
    if message is None:
        return None
    count = int(status.Get_count(MPI.BYTE))
    if count < FRAME_HEADER_BYTES or count > MAX_FRAME_BYTES:
        raise RuntimeError(f"MPI frame length {count} is outside the SLR3 bounds")
    frame = bytearray(count)
    message.Recv([frame, MPI.BYTE])
    return status, bytes(frame)


class _ControllerProgress:
    def __init__(self, MPI, world, hub, health_timeout_s: float = 30.0) -> None:
        self._MPI = MPI
        self._world = world
        self._hub = hub
        self._health_timeout_s = health_timeout_s
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
            # The progress thread may still own MPI_THREAD_SERIALIZED calls.
            # Rank exit delegates cleanup to the launcher without another MPI call.
            os._exit(1)
        if self._error is not None:
            raise RuntimeError("MPI progress failed") from self._error

    def _run(self) -> None:
        in_flight: list[tuple[Any, int, bytes]] = []
        shutdown_pending: dict[int, bytes] = {}
        try:
            while True:
                remaining: list[tuple[Any, int, bytes]] = []
                for request, ticket, frame in in_flight:
                    if request.Test():
                        self._hub.complete_outbound(ticket)
                    else:
                        remaining.append((request, ticket, frame))
                in_flight = remaining

                outbound = self._hub.poll_outbound(0.0)
                if outbound is not None:
                    ticket, target_rank, tag, frame = outbound
                    frame = bytes(frame)
                    if int(tag.value) == LIFECYCLE_TAG:
                        shutdown_pending[int(target_rank)] = frame
                    request = self._world.Isend(
                        [frame, self._MPI.BYTE],
                        dest=int(target_rank),
                        tag=int(tag.value),
                    )
                    in_flight.append((request, int(ticket), frame))

                received = _receive_frame(self._MPI, self._world, self._MPI.ANY_SOURCE)
                if received is not None:
                    status, frame = received
                    source, tag = int(status.Get_source()), int(status.Get_tag())
                    if tag == LIFECYCLE_TAG and shutdown_pending.get(source) == frame:
                        # The exact shutdown echo is the last message from this rank,
                        # after its serialized heartbeat sender has stopped.
                        del shutdown_pending[source]
                    else:
                        self._hub.deliver(source, tag, frame)
                self._hub.check_health(self._health_timeout_s)
                if (
                    self._stop.is_set()
                    and not in_flight
                    and not shutdown_pending
                    and self._hub.pending_frame_bytes == 0
                ):
                    return
        except BaseException as exc:  # noqa: BLE001
            self._error = exc
            with contextlib.suppress(BaseException):
                self._hub.fail(f"MPI progress failure: {type(exc).__name__}: {exc}")
            try:
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                self._world.Abort(1)
            finally:
                os._exit(1)


class MpiDirectControllerContext:
    def __init__(
        self, MPI, world, topology: MpiDirectTopology, session_id: int, reporter: _RankCleanupReporter | None = None
    ) -> None:
        from _task_interface import _MpiDirectTransportHub  # noqa: PLC0415

        self.topology = topology
        self.session_id = int(session_id)
        self.cluster_id = f"mpi-direct-{self.session_id:016x}"
        self._hub = _MpiDirectTransportHub(topology.max_pending_frame_bytes)
        for spec in topology.executors:
            self._hub.register_route(spec.worker_id, spec.rank, self.session_id, spec.comm_profile)
        self._progress = _ControllerProgress(MPI, world, self._hub, topology.effective_health_timeout_s)
        self._workers: list[Any] = []
        self._closed = False
        self._cleanup_reporter = reporter
        self._progress.start()

    def attach(self, worker) -> tuple[int, ...]:
        if self._closed:
            raise RuntimeError("MPI direct controller context is closed")
        if worker.level < 4:
            raise TypeError("MPI direct controller requires Worker(level>=4)")
        if self._workers:
            raise RuntimeError("MPI direct controller context supports exactly one attached L4 Worker")
        worker._global_cluster_id = self.cluster_id  # noqa: SLF001 -- context owns the direct-MPI topology
        worker_ids = worker._configure_mpi_direct(self.topology, self.session_id, self._hub)  # noqa: SLF001
        self._workers.append(worker)
        return worker_ids

    def create_worker(self, **config):
        from .worker import Worker  # noqa: PLC0415

        worker = Worker(level=4, **config)
        self.attach(worker)
        return worker

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._cleanup_reporter is not None:
            self._cleanup_reporter.begin(command="not_observed")
        close_errors: list[BaseException] = []
        transport_failed = False
        for worker in reversed(self._workers):
            try:
                worker.close()
            except BaseException as exc:  # noqa: BLE001
                close_errors.append(exc)
        if not self._hub.all_routes_shutdown:
            transport_failed = True
            error = RuntimeError("MPI executors were not sent lifecycle shutdown")
            close_errors.append(error)
            self._hub.fail(str(error))
        try:
            self._progress.stop(self.topology.session_timeout_s)
        except BaseException as exc:  # noqa: BLE001
            transport_failed = True
            close_errors.append(exc)
        finally:
            try:
                self._hub.close()
            except BaseException as exc:  # noqa: BLE001
                transport_failed = True
                close_errors.append(exc)
            if self._cleanup_reporter is not None:
                self._cleanup_reporter.finish([cleanup_failure("L4 close", exc) for exc in close_errors])
        if close_errors:
            error_type = RuntimeError if transport_failed else _MpiDirectCleanupError
            raise error_type(f"L4 close encountered {len(close_errors)} failure(s)") from close_errors[0]


class _ExecutorFrameSocket:
    def __init__(
        self,
        MPI,
        world,
        spec: MpiDirectExecutorSpec,
        session_id: int,
        heartbeat_interval_s: float,
        reporter: _RankCleanupReporter | None = None,
    ) -> None:
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
        self._cleanup_reporter = reporter
        self.shutdown_observed = False
        self._shutdown_frame: bytes | None = None

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
            with self._mpi_mu:
                received = _receive_frame(self._MPI, self._world, 0)
            if received is None:
                continue
            status, frame = received
            tag = int(status.Get_tag())
            decoded = decode_frame(frame)
            expected_tag = LIFECYCLE_TAG if decoded.header.frame_type == FrameType.SHUTDOWN else COMMAND_REQUEST_TAG
            if tag != expected_tag:
                raise RuntimeError("controller MPI tag does not match SLR3 request type")
            if decoded.header.session_id != self._session_id or decoded.header.worker_id != self._spec.worker_id:
                raise RuntimeError("controller SLR3 frame identity mismatch")
            if decoded.header.frame_type == FrameType.SHUTDOWN:
                self.shutdown_observed = True
                self._shutdown_frame = frame
                if self._cleanup_reporter is not None:
                    self._cleanup_reporter.begin(command="observed")
            return frame

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
        if self._shutdown_frame is not None:
            self._send(self._shutdown_frame, LIFECYCLE_TAG)
            self._shutdown_frame = None


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


def _run_controller(
    MPI, world, topology: MpiDirectTopology, session_id: int, target: str, reporter: _RankCleanupReporter | None = None
) -> None:
    context = MpiDirectControllerContext(MPI, world, topology, session_id, reporter)
    try:
        _load_controller(target)(context)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        with contextlib.suppress(BaseException):
            context.close()
        raise
    else:
        context.close()


def _run_executor(
    MPI, world, topology: MpiDirectTopology, session_id: int, spec, worker, reporter: _RankCleanupReporter | None = None
) -> int:
    from .remote_l3_session import _run_command_loop  # noqa: PLC0415

    channel = _ExecutorFrameSocket(MPI, world, spec, session_id, topology.heartbeat_interval_s, reporter)
    manifest = {
        **spec.to_dict(),
        "session_id": session_id,
        "transport": spec.comm_profile,
        "cluster_id": f"mpi-direct-{session_id:016x}",
        "session_timeout_s": topology.session_timeout_s,
        "heartbeat_interval_s": topology.heartbeat_interval_s,
        "node_rank": spec.rank - 1,
        "node_count": len(topology.executors),
    }
    cleanup_errors: list[BaseException] = []
    fatal_error: BaseException | None = None
    try:
        _run_command_loop(channel, manifest, worker, {}, {}, cleanup_errors=cleanup_errors)  # type: ignore[arg-type]
    except BaseException as exc:  # noqa: BLE001
        fatal_error = exc
    failures = [cleanup_failure("command-loop buffer release", exc) for exc in cleanup_errors]
    if fatal_error is None:
        fatal_error = next((exc for exc in cleanup_errors if not isinstance(exc, Exception)), None)
    try:
        channel.close()
    except BaseException as exc:  # noqa: BLE001
        failures.append(cleanup_failure("MPI channel close", exc))
        if fatal_error is None:
            fatal_error = exc
    try:
        worker.close()
    except BaseException as exc:  # noqa: BLE001
        failures.append(cleanup_failure("L3 Worker.close", exc))
        if fatal_error is None and not isinstance(exc, Exception):
            fatal_error = exc
    if reporter is not None:
        reporter.finish(failures)
    if fatal_error is not None:
        raise fatal_error
    return 1 if failures else 0


def run_runtime(
    manifest_json: str,
    controller: str,
    *,
    startup_host: str,
    startup_port: int,
) -> int:
    try:
        data = json.loads(base64.urlsafe_b64decode(manifest_json.encode("ascii")))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid inline MPI direct runtime manifest") from exc
    topology, session_id = load_runtime_manifest_data(data)
    startup_token = os.environ.get(MPI_DIRECT_STARTUP_TOKEN_ENV)
    if not (startup_host and startup_port and startup_token):
        raise ValueError(f"startup gate requires host, port, and {MPI_DIRECT_STARTUP_TOKEN_ENV} environment variable")
    rank = _launcher_rank()
    if rank < 0 or rank >= topology.world_size:
        raise ValueError("launcher rank is outside the topology world")
    worker = None
    startup_error: BaseException | None = None
    try:
        _assert_native_wire_constants()
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
    _pre_mpi_gate(startup_host, startup_port, startup_token, rank, topology.startup_timeout_s, startup_error)
    if startup_error is not None:
        raise startup_error

    MPI = _import_mpi()
    world = None
    report_settings = data.get("cleanup_report")
    reporter = None
    if report_settings is not None:
        spec = topology.executor_for_rank(rank) if rank else None
        reporter = _RankCleanupReporter(
            startup_host,
            int(report_settings["port"]),
            startup_token,
            MpiDirectCleanupReport(
                str(report_settings["job_id"]),
                session_id,
                rank,
                None if spec is None else spec.worker_id,
                "L4" if rank == 0 else "L3",
            ),
        )
    try:
        world = _init_mpi(MPI, rank, topology.world_size)
        if rank == 0:
            _run_controller(MPI, world, topology, session_id, controller, reporter)
        else:
            assert worker is not None
            return _run_executor(MPI, world, topology, session_id, topology.executor_for_rank(rank), worker, reporter)
        return 0
    except _MpiDirectCleanupError:
        return 1
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
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--startup-host", required=True)
    parser.add_argument("--startup-port", type=int, required=True)
    return run_runtime(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
