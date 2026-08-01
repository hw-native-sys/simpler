# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MPI-launched Remote L3 session runner.

Each mpirun rank runs one ordinary Remote L3 session. The L4 parent still owns
the task/control lane over the existing RemoteL3 socket transport; MPI is used
only when the full mpirun group participates in one Global CommDomain prepare.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import struct
import sys
import time
from typing import Any

from .global_comm_domain import (
    GlobalDomainCommand,
    GlobalDomainPhase,
    GlobalDomainReleaseCommand,
    decode_descriptor_table,
    encode_descriptor_table,
    validate_descriptor_table,
)
from .remote_l3_session import _format_remote_error, run_session
from .worker import Worker


class MpiGlobalDomainExchange:
    """Descriptor exchange hook for a full mpirun-launched L3 group."""

    def __init__(self, comm: Any, *, group_worker_ids: tuple[int, ...], timeout_s: float) -> None:
        self._comm = comm
        self._rank = int(comm.Get_rank())
        self._group_worker_ids = tuple(int(worker_id) for worker_id in group_worker_ids)
        self._group_worker_id_set = set(self._group_worker_ids)
        self._timeout_s = float(timeout_s)
        if not (self._timeout_s > 0 and math.isfinite(self._timeout_s)):
            raise ValueError("MPI Global CommDomain timeout must be a positive finite number")

    def _allgather(self, payload: Any, *, operation: str, on_timeout) -> list[Any]:
        request = self._comm.iallgather(payload)
        deadline = time.monotonic() + self._timeout_s
        while True:
            complete, gathered = request.test()
            if complete:
                return list(gathered)
            if time.monotonic() >= deadline:
                on_timeout()
                try:
                    self._comm.Abort(1)
                except BaseException as exc:  # noqa: BLE001
                    raise TimeoutError(f"MPI Global CommDomain {operation} timed out") from exc
                raise TimeoutError(f"MPI Global CommDomain {operation} timed out")

    def prepare_import(self, command: GlobalDomainCommand, inner_worker: Worker, worker_id: int) -> bytes | None:
        if command.phase is not GlobalDomainPhase.PREPARE_EXPORT:
            return None
        if {int(member.node_worker_id) for member in command.members} != self._group_worker_id_set:
            return None

        ok = True
        error_message = ""
        local_payload = b""
        release_command = GlobalDomainReleaseCommand(command.domain_id, command.generation)

        def release_local() -> None:
            inner_worker._release_global_domain_node(  # noqa: SLF001
                release_command,
                suppress_errors=True,
            )

        try:
            descriptors = inner_worker._prepare_global_domain_node(command, int(worker_id))  # noqa: SLF001
            local_payload = encode_descriptor_table(descriptors)
        except BaseException as exc:  # noqa: BLE001
            release_local()
            ok = False
            error_message = _format_remote_error(
                f"mpi global domain prepare rank={self._rank} worker_id={worker_id}",
                exc,
            )

        try:
            gathered = self._allgather(
                (self._rank, ok, error_message, local_payload),
                operation="prepare",
                on_timeout=release_local,
            )
        except BaseException:
            release_local()
            raise
        errors = [(rank, message) for rank, rank_ok, message, _payload in gathered if not rank_ok]
        if errors:
            if ok:
                release_local()
            rank, message = errors[0]
            raise RuntimeError(f"MPI Global CommDomain prepare failed on rank {rank}: {message}")

        try:
            descriptor_by_rank = {}
            for _rank, _ok, _message, payload in sorted(gathered, key=lambda item: int(item[0])):
                for descriptor in decode_descriptor_table(bytes(payload)):
                    if descriptor.domain_rank in descriptor_by_rank:
                        raise RuntimeError("MPI Global CommDomain exchange returned a duplicate domain rank")
                    descriptor_by_rank[descriptor.domain_rank] = descriptor
            descriptors = tuple(descriptor_by_rank[rank] for rank in range(len(command.members)))
            validate_descriptor_table(descriptors, rank_count=len(command.members), profile=command.profile)
        except BaseException:
            release_local()
            raise

        import_command = GlobalDomainCommand(
            phase=GlobalDomainPhase.IMPORT,
            domain_id=command.domain_id,
            generation=command.generation,
            name=command.name,
            profile=command.profile,
            window_size=command.window_size,
            members=command.members,
            buffers=command.buffers,
            descriptors=descriptors,
        )
        import_ok = True
        import_error = ""
        try:
            inner_worker._import_global_domain_node(import_command, int(worker_id))  # noqa: SLF001
        except BaseException as exc:  # noqa: BLE001
            release_local()
            import_ok = False
            import_error = _format_remote_error(
                f"mpi global domain import rank={self._rank} worker_id={worker_id}",
                exc,
            )

        try:
            statuses = self._allgather(
                (self._rank, import_ok, import_error),
                operation="import",
                on_timeout=release_local,
            )
        except BaseException:
            release_local()
            raise
        import_errors = [(rank, message) for rank, rank_ok, message in statuses if not rank_ok]
        if import_errors:
            release_local()
            rank, message = import_errors[0]
            raise RuntimeError(f"MPI Global CommDomain import failed on rank {rank}: {message}")
        return encode_descriptor_table(descriptors)


def _write_ready_file(ready_dir: str, rank: int, payload: dict[str, Any]) -> None:
    os.makedirs(ready_dir, exist_ok=True)
    final_path = os.path.join(ready_dir, f"rank-{int(rank)}.json")
    tmp_path = f"{final_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, final_path)


def _send_ready_tcp(host: str, port: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    with socket.create_connection((host, int(port)), timeout=10.0) as sock:
        sock.sendall(struct.pack("<I", len(data)) + data)


def _raise_keyboard_interrupt(_signum, _frame):
    raise KeyboardInterrupt


def _load_group_manifest_from_rank0(comm: Any, manifest_path: str) -> dict[str, Any]:
    rank = int(comm.Get_rank())
    if rank == 0:
        try:
            with open(manifest_path, encoding="utf-8") as f:
                payload = (True, json.load(f))
        except BaseException as exc:  # noqa: BLE001
            payload = (False, f"{type(exc).__name__}: {exc}")
    else:
        payload = None
    ok, value = comm.bcast(payload, root=0)
    if not ok:
        raise RuntimeError(f"MPI L3 rank0 failed to read group manifest {manifest_path!r}: {value}")
    if not isinstance(value, dict):
        raise ValueError("MPI L3 group manifest broadcast returned a non-object payload")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-manifest", required=True)
    ns = parser.parse_args(argv)

    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        from mpi4py import MPI  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("simpler.mpi_l3_session requires mpi4py inside the mpirun Python environment") from exc

    comm = MPI.COMM_WORLD
    rank = int(comm.Get_rank())
    world_size = int(comm.Get_size())

    group_manifest = _load_group_manifest_from_rank0(comm, ns.group_manifest)
    rank_manifests = group_manifest.get("rank_manifests")
    if not isinstance(rank_manifests, list):
        raise ValueError("MPI L3 group manifest requires a rank_manifests list")
    if len(rank_manifests) != world_size:
        raise ValueError("MPI L3 group manifest world size does not match MPI_COMM_WORLD")
    manifest = dict(rank_manifests[rank])
    group_worker_ids = tuple(int(x) for x in group_manifest.get("worker_ids", ()))
    if len(group_worker_ids) != world_size:
        raise ValueError("MPI L3 group manifest worker_ids must match MPI_COMM_WORLD size")

    ready_dir = str(group_manifest["ready_dir"])
    ready_host = str(group_manifest.get("ready_host") or "")
    ready_port = int(group_manifest.get("ready_port") or 0)
    ready_token = str(group_manifest.get("ready_token") or "")

    def ready_writer(payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload["mpi_rank"] = rank
        if ready_host:
            payload["ready_token"] = ready_token
            _send_ready_tcp(ready_host, ready_port, payload)
        else:
            _write_ready_file(ready_dir, rank, payload)

    exchange = MpiGlobalDomainExchange(
        comm,
        group_worker_ids=group_worker_ids,
        timeout_s=float(manifest["session_timeout_s"]),
    )
    return run_session(
        manifest,
        None,
        ready_writer=ready_writer,
        global_domain_prepare_import=exchange.prepare_import,
    )


if __name__ == "__main__":
    sys.exit(main())
