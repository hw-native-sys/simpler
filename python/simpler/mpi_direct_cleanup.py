# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded, process-reported Direct MPI close observations outside the MPI world."""

from __future__ import annotations

import contextlib
import json
import socket
import threading
import time
from dataclasses import asdict, dataclass, replace

from .mpi_direct_protocol import _gate_recv, _gate_send
from .mpi_direct_topology import MpiDirectTopology

CLEANUP_SCHEMA = 1
MAX_REPORT_BYTES = 32 * 1024
MAX_FAILURES = 8
MAX_FAILURE_BYTES = 512
REPORT_TIMEOUT_S = 1.0


def cleanup_failure(operation: str, error: BaseException) -> str:
    message = f"{operation}: {type(error).__name__}: {error}"
    return message.encode("utf-8", errors="replace")[:MAX_FAILURE_BYTES].decode("utf-8", errors="ignore")


@dataclass(frozen=True)
class MpiDirectCleanupReport:
    job_id: str
    session_id: int
    rank: int
    worker_id: int | None
    role: str
    sequence: int = 0
    command: str = "unknown"
    cleanup: str = "unknown"
    failures: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"schema": CLEANUP_SCHEMA, **asdict(self)}


def _decode_cleanup_report(payload: object) -> MpiDirectCleanupReport:
    if len(json.dumps(payload).encode("utf-8")) > MAX_REPORT_BYTES:
        raise ValueError("MPI cleanup report is too large")
    fields = set(MpiDirectCleanupReport.__dataclass_fields__) | {"schema"}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("incomplete or unsupported MPI cleanup report")
    if type(payload["schema"]) is not int or payload["schema"] != CLEANUP_SCHEMA:
        raise ValueError("unsupported MPI cleanup schema")
    for name in ("rank", "session_id", "sequence"):
        if type(payload[name]) is not int:
            raise ValueError(f"invalid MPI cleanup {name}")
    if payload["worker_id"] is not None and type(payload["worker_id"]) is not int:
        raise ValueError("invalid MPI cleanup worker_id")
    failures = payload["failures"]
    if not isinstance(failures, (list, tuple)) or len(failures) > MAX_FAILURES:
        raise ValueError("invalid MPI cleanup failures")
    if any(not isinstance(item, str) or len(item.encode("utf-8")) > MAX_FAILURE_BYTES for item in failures):
        raise ValueError("invalid MPI cleanup failure detail")
    report = MpiDirectCleanupReport(
        **{name: payload[name] for name in fields - {"schema", "failures"}}, failures=tuple(failures)
    )
    if report.command not in ("observed", "not_observed"):
        raise ValueError("invalid MPI shutdown command observation")
    if report.sequence == 1:
        if report.cleanup != "unknown" or failures:
            raise ValueError("command acknowledgement cannot assert cleanup")
    elif report.sequence == 2:
        if report.cleanup not in ("succeeded", "failed") or (report.cleanup == "failed") != bool(failures):
            raise ValueError("invalid MPI cleanup outcome")
    else:
        raise ValueError("invalid MPI cleanup sequence")
    return report


class MpiDirectCleanupLedger:
    """Missing ranks and incomplete closes retain UNKNOWN; exit is never self-reported."""

    def __init__(self, topology: MpiDirectTopology, job_id: str, session_id: int) -> None:
        self._reports = {0: MpiDirectCleanupReport(job_id, session_id, 0, None, "L4")}
        self._reports.update(
            (spec.rank, MpiDirectCleanupReport(job_id, session_id, spec.rank, spec.worker_id, "L3"))
            for spec in topology.executors
        )
        self._lock = threading.Lock()

    def snapshot(self) -> dict[int, MpiDirectCleanupReport]:
        with self._lock:
            return dict(self._reports)

    def accept(self, payload: object) -> bool:
        report = _decode_cleanup_report(payload)
        with self._lock:
            previous = self._reports.get(report.rank)
            if previous is None or any(
                getattr(report, key) != getattr(previous, key)
                for key in ("job_id", "session_id", "rank", "worker_id", "role")
            ):
                raise ValueError("MPI cleanup identity mismatch")
            if report == previous:
                return False
            if report.sequence <= previous.sequence or (previous.sequence and report.command != previous.command):
                raise ValueError("stale or conflicting MPI cleanup report")
            self._reports[report.rank] = report
            return True


class _CleanupReceiver:
    def __init__(self, host: str, token: str, ledger: MpiDirectCleanupLedger) -> None:
        self._token, self._ledger = token, ledger
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self._listener.bind((host, 0))
            self._listener.listen()
            self._listener.settimeout(0.1)
        except BaseException:
            self._listener.close()
            raise
        self.port = int(self._listener.getsockname()[1])
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="simpler-mpi-cleanup", daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        self._thread.join(REPORT_TIMEOUT_S + 0.5)
        self._listener.close()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                peer, _address = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set() or self._listener.fileno() == -1:
                    return
                self._stop.wait(0.01)
                continue
            with peer:
                try:
                    deadline = time.monotonic() + REPORT_TIMEOUT_S
                    message = _gate_recv(peer, deadline)
                    if message.get("token") != self._token:
                        continue
                    self._ledger.accept(message["report"])
                    peer.settimeout(max(0.001, deadline - time.monotonic()))
                    _gate_send(peer, {"accepted": True})
                except (OSError, ValueError, TypeError, KeyError, RecursionError):
                    continue


class _RankCleanupReporter:
    def __init__(self, host: str, port: int, token: str, identity: MpiDirectCleanupReport) -> None:
        self._port, self._token = port, token
        # Name resolution belongs to startup, outside the close-path socket deadline.
        try:
            self._address: str | None = socket.gethostbyname(host)
        except OSError:
            self._address = None
        self._identity = identity
        self._command = "not_observed"

    def _send(self, report: MpiDirectCleanupReport) -> bool:
        if self._address is None:
            return False
        deadline = time.monotonic() + REPORT_TIMEOUT_S
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as peer:
                peer.settimeout(REPORT_TIMEOUT_S)
                peer.connect((self._address, self._port))
                peer.settimeout(max(0.001, deadline - time.monotonic()))
                _gate_send(peer, {"token": self._token, "report": report.to_dict()})
                return _gate_recv(peer, deadline).get("accepted") is True
        except (OSError, ValueError, ConnectionError):
            return False

    def begin(self, *, command: str) -> None:
        self._command = command
        self._send(replace(self._identity, sequence=1, command=command))

    def finish(self, failures: list[str]) -> None:
        self._send(
            replace(
                self._identity,
                sequence=2,
                command=self._command,
                cleanup="failed" if failures else "succeeded",
                failures=tuple(failures[:MAX_FAILURES]),
            )
        )


def _print_cleanup_reports(ledger: MpiDirectCleanupLedger) -> None:
    # Launcher status and each rank's process exit are separate external observations.
    for report in ledger.snapshot().values():
        with contextlib.suppress(OSError, ValueError):
            print(json.dumps({"mpi_direct_cleanup": report.to_dict(), "process_exit": "unknown"}), flush=True)
