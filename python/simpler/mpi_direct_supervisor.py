# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""External process owner for a direct-MPI L4 job."""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
from itertools import groupby
from typing import Any

from .mpi_direct_cleanup import (
    MpiDirectCleanupLedger,
    MpiDirectCleanupReport,
    _CleanupReceiver,
    _print_cleanup_reports,
)
from .mpi_direct_protocol import MPI_DIRECT_STARTUP_TOKEN_ENV, _gate_recv, _gate_send
from .mpi_direct_topology import MpiDirectTopology

_EXPORTED_ENV_VARS = (
    "PATH",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "PYTHONNOUSERSITE",
    "PYTHONUNBUFFERED",
    "ASCEND_HOME_PATH",
    "ASCEND_OPP_PATH",
    MPI_DIRECT_STARTUP_TOKEN_ENV,
)


class _StartupGateRankFailure(RuntimeError):
    pass


def _host_slots(topology: MpiDirectTopology) -> tuple[tuple[str, int], ...]:
    return tuple((host, sum(1 for _ in ranks)) for host, ranks in groupby(topology.hosts))


def _detect_launcher_family(mpirun_path: str) -> str:
    output_parts = []
    for flag in ("--version", "-info"):
        try:
            result = subprocess.run(
                [mpirun_path, flag],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            output_parts.append(f"{flag}: {exc}")
            continue
        output_parts.append(result.stdout)
        lowered = result.stdout.lower()
        if "open mpi" in lowered or "openrte" in lowered:
            return "openmpi"
        if "hydra" in lowered or "mpich" in lowered:
            return "mpich"
    detail = "\n".join(part.strip() for part in output_parts if part.strip())
    raise RuntimeError(
        f"cannot identify MPI launcher {mpirun_path!r}; pass --launcher-family openmpi or mpich"
        + (f"\n{detail}" if detail else "")
    )


def _mpi_vendor_family(vendor: str) -> str:
    lowered = vendor.lower()
    if "open mpi" in lowered or "openmpi" in lowered:
        return "openmpi"
    if "mpich" in lowered or "mvapich" in lowered or "intel(r) mpi" in lowered or "intel mpi" in lowered:
        return "mpich"
    raise RuntimeError(f"unsupported mpi4py MPI vendor {vendor!r}")


def _detect_mpi4py_family(python_executable: str | None = None) -> tuple[str, str]:
    script = (
        "import mpi4py; "
        "mpi4py.rc.initialize = False; "
        "mpi4py.rc.finalize = False; "
        "from mpi4py import MPI; "
        "print(MPI.get_vendor()[0])"
    )
    try:
        result = subprocess.run(
            [python_executable or sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"cannot inspect mpi4py with {python_executable or sys.executable!r}: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"cannot import mpi4py with {python_executable or sys.executable!r}" + (f"\n{detail}" if detail else "")
        )
    vendor = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if not vendor:
        raise RuntimeError(f"mpi4py did not report its MPI vendor with {python_executable or sys.executable!r}")
    return _mpi_vendor_family(vendor), vendor


@contextlib.contextmanager
def _launcher_hostfile(topology: MpiDirectTopology, launcher_family: str):
    if launcher_family != "mpich":
        yield None
        return
    with tempfile.NamedTemporaryFile(mode="w", prefix="simpler-mpi-hosts-", suffix=".txt", encoding="utf-8") as stream:
        for host, slots in _host_slots(topology):
            stream.write(f"{host}:{slots}\n")
        stream.flush()
        yield stream.name


def _family_launcher_args(topology: MpiDirectTopology, launcher_family: str, hostfile_path: str | None) -> list[str]:
    if launcher_family == "mpich":
        if not hostfile_path:
            raise ValueError("MPICH/Hydra launch requires a local hostfile")
        args = ["-f", hostfile_path]
        for name in _EXPORTED_ENV_VARS:
            if name == MPI_DIRECT_STARTUP_TOKEN_ENV or name in os.environ:
                if name == MPI_DIRECT_STARTUP_TOKEN_ENV:
                    args.extend(("-genvlist", name))
                else:
                    args.extend(("-genv", name, os.environ[name]))
        return args
    if launcher_family == "openmpi":
        host_spec = ",".join(f"{host}:{slots}" for host, slots in _host_slots(topology))
        args = ["--host", host_spec, "--map-by", "slot", "--bind-to", "none"]
        if os.geteuid() == 0:
            args.append("--allow-run-as-root")
        for name in _EXPORTED_ENV_VARS:
            if name == MPI_DIRECT_STARTUP_TOKEN_ENV or name in os.environ:
                args.extend(("-x", name))
        return args
    raise ValueError(f"unsupported MPI launcher family {launcher_family!r}")


def _build_command(  # noqa: PLR0913 -- launcher construction mirrors the complete MPI CLI surface
    topology: MpiDirectTopology,
    *,
    mpirun_path: str,
    controller: str,
    launcher_family: str,
    hostfile_path: str | None = None,
    manifest_json: str,
    python_executable: str | None = None,
    startup_port: int,
) -> list[str]:
    return [
        mpirun_path,
        *_family_launcher_args(topology, launcher_family, hostfile_path),
        *topology.launcher_args,
        "-np",
        str(topology.world_size),
        python_executable or sys.executable,
        "-m",
        "simpler.mpi_direct_runtime",
        "--manifest-json",
        manifest_json,
        "--controller",
        controller,
        "--startup-host",
        topology.controller_host,
        "--startup-port",
        str(startup_port),
    ]


def _startup_gate(
    topology: MpiDirectTopology,
    token: str,
    listener: socket.socket,
    proc: subprocess.Popen[Any],
) -> None:
    deadline = time.monotonic() + topology.startup_timeout_s
    peers: dict[int, socket.socket] = {}
    try:
        while len(peers) < topology.world_size:
            if proc.poll() is not None:
                raise RuntimeError(f"MPI job exited before startup gate completed (status {proc.returncode})")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("MPI startup gate timed out waiting for all ranks")
            listener.settimeout(min(0.25, remaining))
            try:
                peer, _address = listener.accept()
            except socket.timeout:
                continue
            try:
                message = _gate_recv(peer, min(deadline, time.monotonic() + 5.0))
                if message.get("token") != token:
                    peer.close()
                    continue
                try:
                    rank = int(message.get("rank", -1))  # type: ignore[arg-type]
                except (TypeError, ValueError) as exc:
                    raise _StartupGateRankFailure("invalid MPI startup rank") from exc
                if rank < 0 or rank >= topology.world_size or rank in peers:
                    raise _StartupGateRankFailure(f"invalid or duplicate MPI startup rank {rank}")
                if message.get("state") == "failed":
                    raise _StartupGateRankFailure(
                        f"rank {rank} failed before MPI initialization: {message.get('error', '')}"
                    )
                if message.get("state") != "ready":
                    raise _StartupGateRankFailure(f"rank {rank} sent invalid startup state")
                peers[rank] = peer
            except _StartupGateRankFailure:
                with contextlib.suppress(OSError):
                    peer.close()
                raise
            except (OSError, ValueError):
                with contextlib.suppress(OSError):
                    peer.close()
                continue
        for rank in sorted(peers):
            _gate_send(peers[rank], {"token": token, "state": "go_mpi"})
    finally:
        for peer in peers.values():
            with contextlib.suppress(OSError):
                peer.close()
        listener.close()


def _terminate_job(proc: subprocess.Popen[Any], timeout_s: float = 5.0) -> None:
    if proc.poll() is not None:
        proc.wait()
        return
    with contextlib.suppress(OSError, ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(OSError, ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=timeout_s)


@contextlib.contextmanager
def _job_termination_handlers():
    previous = {}

    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    try:
        for signum in (signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.signal(signum, terminate)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def run_supervisor(
    topology_path: str,
    controller: str,
    *,
    mpirun_path: str = "mpirun",
    launcher_family: str = "auto",
    python_executable: str | None = None,
    cleanup_reports: dict[int, MpiDirectCleanupReport] | None = None,
) -> int:
    topology = MpiDirectTopology.load(topology_path)
    if launcher_family == "auto":
        launcher_family = _detect_launcher_family(mpirun_path)
    mpi4py_family, mpi4py_vendor = _detect_mpi4py_family(python_executable)
    if mpi4py_family != launcher_family:
        raise RuntimeError(
            f"MPI launcher family is {launcher_family}, but mpi4py is linked to "
            f"{mpi4py_vendor} ({mpi4py_family}); use a matching launcher or rebuild "
            "mpi4py with the launcher's MPI compiler"
        )
    session_id = secrets.randbits(64) or 1
    gate_token = secrets.token_urlsafe(32)
    job_id = secrets.token_hex(16)
    ledger = MpiDirectCleanupLedger(topology, job_id, session_id)
    try:
        with _job_termination_handlers(), _CleanupReceiver(topology.controller_host, gate_token, ledger) as receiver:
            manifest = topology.runtime_manifest(session_id)
            manifest["cleanup_report"] = {"job_id": job_id, "port": receiver.port}
            manifest_json = base64.urlsafe_b64encode(
                json.dumps(manifest, separators=(",", ":")).encode("utf-8")
            ).decode("ascii")
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((topology.controller_host, 0))
            listener.listen(topology.world_size)
            gate_port = int(listener.getsockname()[1])
            proc: subprocess.Popen[Any] | None = None
            try:
                with _launcher_hostfile(topology, launcher_family) as hostfile_path:
                    command = _build_command(
                        topology,
                        mpirun_path=mpirun_path,
                        controller=controller,
                        launcher_family=launcher_family,
                        hostfile_path=hostfile_path,
                        manifest_json=manifest_json,
                        python_executable=python_executable,
                        startup_port=gate_port,
                    )
                    proc = subprocess.Popen(
                        command, start_new_session=True, env={**os.environ, MPI_DIRECT_STARTUP_TOKEN_ENV: gate_token}
                    )
                    _startup_gate(topology, gate_token, listener, proc)
                    return int(proc.wait())
            except BaseException:
                with contextlib.suppress(OSError):
                    listener.close()
                if proc is not None:
                    _terminate_job(proc)
                raise
    finally:
        if cleanup_reports is not None:
            cleanup_reports.update(ledger.snapshot())
        _print_cleanup_reports(ledger)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--controller", required=True, help="Python file with main(context), or module:callable")
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--launcher-family", choices=("auto", "openmpi", "mpich"), default="auto")
    parser.add_argument("--python", dest="python_executable", help="Python executable available on every MPI host")
    ns = parser.parse_args(argv)
    return run_supervisor(
        ns.topology,
        ns.controller,
        mpirun_path=ns.mpirun,
        launcher_family=ns.launcher_family,
        python_executable=ns.python_executable,
    )


if __name__ == "__main__":
    raise SystemExit(main())
