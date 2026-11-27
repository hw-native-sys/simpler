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
import hmac
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from itertools import groupby
from pathlib import Path
from typing import Any

from .mpi_direct_bootstrap import encode_startup, send_startup
from .mpi_direct_cleanup import (
    MpiDirectCleanupLedger,
    MpiDirectCleanupReport,
    _CleanupReceiver,
    _print_cleanup_reports,
)
from .mpi_direct_protocol import MPI_DIRECT_STARTUP_TOKEN_ENV, _gate_recv, _gate_send
from .mpi_direct_service import ROUTER_TOKEN_ENV, ServiceConfig, _read_frame, _write_frame
from .mpi_direct_service_relay import close_socket, listen
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


def _launcher_environment(startup_token: str) -> dict[str, str]:
    inherited = dict(os.environ)
    inherited.pop(MPI_DIRECT_STARTUP_TOKEN_ENV, None)
    # MPICH 5.0.2 skips the last inherited environment entry in -genvlist lookups.
    return {MPI_DIRECT_STARTUP_TOKEN_ENV: startup_token, **inherited}


def _host_slots(topology: MpiDirectTopology | ServiceConfig) -> tuple[tuple[str, int], ...]:
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
def _launcher_hostfile(topology: MpiDirectTopology | ServiceConfig, launcher_family: str):
    if launcher_family != "mpich":
        yield None
        return
    with tempfile.TemporaryDirectory(prefix="simpler-mpi-hosts-") as directory:
        hostfile = Path(directory) / "hosts.txt"
        hostfile.write_text("".join(f"{host}:{slots}\n" for host, slots in _host_slots(topology)), encoding="utf-8")
        yield str(hostfile)


def _family_launcher_args(
    topology: MpiDirectTopology | ServiceConfig, launcher_family: str, hostfile_path: str | None
) -> list[str]:
    if launcher_family == "mpich":
        if not hostfile_path:
            raise ValueError("MPICH/Hydra launch requires a local hostfile")
        args = ["-f", hostfile_path]
        if isinstance(topology, ServiceConfig) and os.name == "nt" and len(set(topology.hosts)) == 1:
            args = ["-localonly"]
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
        if hasattr(os, "geteuid") and os.geteuid() == 0:
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


def _drain_failed_startup(
    topology: MpiDirectTopology | ServiceConfig,
    token: str,
    listener: socket.socket,
    proc: subprocess.Popen[Any],
    peers: dict[int, socket.socket],
    failure: _StartupGateRankFailure,
) -> None:
    deadline = time.monotonic() + topology.session_timeout_s

    def reject(peer: socket.socket) -> None:
        remaining = max(0.001, deadline - time.monotonic())
        with contextlib.suppress(OSError):
            peer.settimeout(remaining)
            _gate_send(
                peer, {"token": token, "state": "failed", "error": str(failure)[:4096], "cleanup_timeout_s": remaining}
            )

    for peer in peers.values():
        reject(peer)
    while len(peers) < topology.world_size:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or proc.poll() is not None:
            return
        listener.settimeout(min(0.25, remaining))
        try:
            peer, _address = listener.accept()
        except socket.timeout:
            continue
        try:
            message = _gate_recv(peer, min(deadline, time.monotonic() + 5.0))
            rank = message.get("rank")
            if message.get("token") == token and type(rank) is int and 0 <= rank < topology.world_size:
                if rank not in peers and message.get("state") in ("ready", "failed"):
                    peers[rank] = peer
                reject(peer)
        except (OSError, ValueError):
            pass
        finally:
            if peer not in peers.values():
                peer.close()

    acknowledged = 0
    for peer in peers.values():
        with contextlib.suppress(OSError, ValueError):
            message = _gate_recv(peer, deadline)
            if message.get("token") == token and message.get("state") == "cleaned":
                acknowledged += 1
    if acknowledged < len(peers):
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))


def _startup_gate(  # noqa: PLR0912 -- one owner validates bootstrap, readiness and rank failure messages
    topology: MpiDirectTopology | ServiceConfig,
    token: str,
    listener: socket.socket,
    proc: subprocess.Popen[Any],
    startup_payloads: Mapping[int, bytes] | None = None,
) -> None:
    deadline = time.monotonic() + topology.startup_timeout_s
    peers: dict[int, socket.socket] = {}
    bootstrapped: set[int] = set()
    failure = None
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
                if message.get("state") == "bootstrap":
                    if startup_payloads is None or rank in bootstrapped:
                        raise _StartupGateRankFailure(f"unexpected bootstrap request from rank {rank}")
                    _gate_send(peer, {"token": token, "state": "bootstrap", "timeout_s": deadline - time.monotonic()})
                    send_startup(peer, startup_payloads.get(rank, encode_startup({})), deadline)
                    bootstrapped.add(rank)
                    peer.close()
                    continue
                if message.get("state") == "failed":
                    peers[rank] = peer
                    failure = _StartupGateRankFailure(
                        f"rank {rank} failed before MPI initialization: {message.get('error', '')}"
                    )
                    break
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
        if failure is not None:
            _drain_failed_startup(topology, token, listener, proc, peers, failure)
            raise failure
        for rank in sorted(peers):
            _gate_send(peers[rank], {"token": token, "state": "go_mpi"})
    except BaseException as exc:
        if failure is None:
            for peer in peers.values():
                with contextlib.suppress(OSError):
                    _gate_send(peer, {"token": token, "state": "failed", "error": str(exc)[:4096]})
        raise
    finally:
        for peer in peers.values():
            with contextlib.suppress(OSError):
                peer.close()
        listener.close()


def _terminate_job(proc: subprocess.Popen[Any], timeout_s: float = 5.0) -> None:
    if proc.poll() is not None:
        proc.wait()
        return
    if os.name == "nt":
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_s,
                check=False,
            )
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=timeout_s)
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
def _job_termination_handlers(stop=None):
    previous = {}

    def terminate(signum, _frame):
        if stop is not None:
            stop.set()
            return
        raise SystemExit(128 + signum)

    try:
        managed = (signal.SIGINT, getattr(signal, "SIGBREAK", None)) if stop is not None else ()
        for signum in (signal.SIGTERM, getattr(signal, "SIGHUP", None), *managed):
            if signum is None:
                continue
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
    startup_by_rank: Mapping[int, dict] | None = None,
) -> int:
    """Own a static MPI job and optionally bootstrap each rank before MPI init.

    Records may contain startup_artifacts (path-free byte descriptions),
    inner_l3_worker and remote_task_dispatcher registry entries, a startup_hook
    manifest, and controller parameters. Each encoded record is limited to
    256 MiB and travels over the authenticated startup socket, never argv.
    """
    topology = MpiDirectTopology.load(topology_path)
    startup_payloads = None
    if startup_by_rank is not None:
        if any(type(rank) is not int or not 0 <= rank < topology.world_size for rank in startup_by_rank):
            raise ValueError("startup_by_rank contains a rank outside the static topology")
        startup_payloads = {rank: encode_startup(record) for rank, record in startup_by_rank.items()}
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
            manifest["bootstrap"] = startup_payloads is not None
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
                    proc = subprocess.Popen(command, start_new_session=True, env=_launcher_environment(gate_token))
                    _startup_gate(topology, gate_token, listener, proc, startup_payloads)
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


def _wait_coordinated(proc, topology, listener, admin, stop):
    """Own mpirun and deliver a cooperative stop to rank zero."""
    listener.settimeout(0.2)
    deadline = time.monotonic() + topology.startup_timeout_s
    while proc.poll() is None:
        if time.monotonic() >= deadline:
            raise TimeoutError("rank zero did not connect to the Supervisor")
        try:
            peer, _ = listener.accept()
        except socket.timeout:
            continue
        with peer:
            peer.settimeout(5)
            try:
                hello = _read_frame(peer)
            except (OSError, ValueError):
                continue
            token = hello.get("token")
            if (
                hello.get("type") != "hello"
                or not isinstance(token, str)
                or not hmac.compare_digest(token.encode(), admin["token"].encode())
            ):
                continue
            while proc.poll() is None:
                if stop.wait(0.2):
                    try:
                        _write_frame(peer, {"type": "shutdown"})
                    except OSError:
                        # Rank zero treats loss of the admin connection as a shutdown request.
                        close_socket(peer)
                    try:
                        return proc.wait(
                            timeout=topology.coordination["drain_timeout_s"] + 2 * topology.session_timeout_s + 45
                        )
                    except subprocess.TimeoutExpired as exc:
                        raise TimeoutError("MPI shutdown unconfirmed by the Supervisor") from exc
            return proc.returncode
    return proc.returncode


def run_service_supervisor(
    config_path: str,
    *,
    mpirun_path: str = "mpirun",
    launcher_family: str = "auto",
    python_executable: str | None = None,
) -> int:
    """Launch and own an MPI job whose service ranks each run one external process."""
    topology = ServiceConfig.load(config_path)
    router_token = os.environ.get(ROUTER_TOKEN_ENV)
    if not router_token:
        raise ValueError(f"{ROUTER_TOKEN_ENV} must contain the Router's job token")
    if launcher_family == "auto":
        launcher_family = _detect_launcher_family(mpirun_path)
    mpi4py_family, mpi4py_vendor = _detect_mpi4py_family(python_executable)
    if mpi4py_family != launcher_family:
        raise RuntimeError(f"MPI launcher family {launcher_family} does not match mpi4py vendor {mpi4py_vendor}")

    token = secrets.token_urlsafe(32)
    coordination = topology.coordination
    stop = threading.Event() if coordination is not None else None
    with contextlib.ExitStack() as stack:
        stack.enter_context(_job_termination_handlers(stop))
        listener = stack.enter_context(socket.socket())
        admin_listener = admin = None
        if coordination is not None:
            admin_listener = stack.enter_context(listen())
            admin = {"port": admin_listener.getsockname()[1], "token": secrets.token_urlsafe(32)}
        payloads = {rank: encode_startup(record) for rank, record in topology.records(router_token, admin).items()}
        listener.bind((topology.controller_host, 0))
        listener.listen(topology.world_size)
        with _launcher_hostfile(topology, launcher_family) as hostfile:
            command = [
                mpirun_path,
                *_family_launcher_args(topology, launcher_family, hostfile),
                *topology.launcher_args,
                "-np",
                str(topology.world_size),
                python_executable or sys.executable,
                "-m",
                "simpler.mpi_direct_service",
                "--startup-host",
                topology.controller_host,
                "--startup-port",
                str(listener.getsockname()[1]),
                "--startup-timeout",
                str(topology.startup_timeout_s),
            ]
            env = _launcher_environment(token)
            env.pop(ROUTER_TOKEN_ENV, None)
            proc = subprocess.Popen(command, start_new_session=os.name != "nt", env=env)
            try:
                _startup_gate(topology, token, listener, proc, payloads)
                if admin_listener is not None:
                    return _wait_coordinated(proc, topology, admin_listener, admin, stop)
                return proc.wait()
            except BaseException:
                _terminate_job(proc)
                raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--topology")
    modes.add_argument("--service-config", help="JSON configuration for external P/D service ranks")
    parser.add_argument("--controller", help="Python file with main(context), or module:callable")
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--launcher-family", choices=("auto", "openmpi", "mpich"), default="auto")
    parser.add_argument("--python", dest="python_executable", help="Python executable available on every MPI host")
    ns = parser.parse_args(argv)
    if ns.service_config:
        if ns.controller is not None:
            parser.error("--controller applies only to --topology")
        return run_service_supervisor(
            ns.service_config,
            mpirun_path=ns.mpirun,
            launcher_family=ns.launcher_family,
            python_executable=ns.python_executable,
        )
    if ns.controller is None:
        parser.error("--topology requires --controller")
    return run_supervisor(
        ns.topology,
        ns.controller,
        mpirun_path=ns.mpirun,
        launcher_family=ns.launcher_family,
        python_executable=ns.python_executable,
    )


if __name__ == "__main__":
    raise SystemExit(main())
