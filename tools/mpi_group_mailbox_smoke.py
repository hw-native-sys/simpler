#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device-free two-rank integration smoke for the named mailbox and MPI dispatch loop."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

from simpler.mpi_group_mailbox import (
    MailboxGroupState,
    MailboxOpcode,
    MailboxRequestState,
    MailboxTarget,
    MpiGroupError,
    MpiGroupMailbox,
    MpiRankError,
    open_rank_mailbox,
)


def _wait_until(predicate, *, deadline: float, label: str) -> None:
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {label}")


def _worker(manifest_path: str) -> int:  # noqa: PLR0912 -- mirrors the ordered rank dispatcher state machine
    from mpi4py import MPI  # noqa: PLC0415

    comm = MPI.COMM_WORLD.Dup()
    rank = int(comm.Get_rank())
    if int(comm.Get_size()) != 2:
        raise RuntimeError("MPI mailbox smoke requires exactly two ranks")
    if rank == 0:
        with open(manifest_path, encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    else:
        manifest = None
    manifest = comm.bcast(manifest, root=0)
    mailbox = None
    try:
        ready = comm.allgather(rank)
        if ready != [0, 1]:
            raise RuntimeError(f"unexpected readiness result: {ready}")
        mailbox = open_rank_mailbox(manifest, rank=rank)
        if rank == 0:
            assert mailbox is not None
            mailbox.publish_ready()
        last_sequence = 0
        while True:
            if rank == 0:
                assert mailbox is not None
                _wait_until(
                    lambda: mailbox.request_state
                    in (MailboxRequestState.REQUEST_READY, MailboxRequestState.SHUTDOWN_READY),
                    deadline=time.monotonic() + 10.0,
                    label="mailbox request",
                )
                request = mailbox.accept_request(last_sequence_id=last_sequence)
                last_sequence = request.sequence_id
            else:
                request = None
            request = comm.bcast(request, root=0)
            if request.opcode is MailboxOpcode.SHUTDOWN:
                gathered = comm.gather((rank, True, b"", None), root=0)
                if rank == 0:
                    assert gathered is not None
                    assert mailbox is not None
                    mailbox.complete_shutdown(sequence_id=request.sequence_id)
                    mailbox.publish_closed()
                break

            payload = request.payloads[rank] if request.target is MailboxTarget.PER_RANK else request.payloads[0]
            local_error = None
            local_result = b""
            try:
                command = json.loads(payload.decode())
                if command.get("fail_rank") == rank:
                    raise ValueError(f"injected failure on rank {rank}")
                local_result = json.dumps(
                    {
                        "rank": rank,
                        "input": int(command["value"]),
                        "result": int(command["value"]) * (rank + 2),
                    },
                    sort_keys=True,
                ).encode()
            except BaseException as exc:  # noqa: BLE001
                local_error = MpiRankError(rank, type(exc).__name__, str(exc))
            gathered = comm.gather((rank, True, local_result, local_error), root=0)
            if rank == 0:
                assert mailbox is not None
                errors = tuple(item[3] for item in gathered if item[3] is not None)
                if errors:
                    mailbox.fail_request(sequence_id=request.sequence_id, errors=errors, terminal=False)
                else:
                    mailbox.complete_request(
                        sequence_id=request.sequence_id,
                        payloads=tuple(item[2] for item in gathered),
                    )
        return 0
    finally:
        if mailbox is not None:
            mailbox.close()
        comm.Free()


def _parent(mpirun: str, python: str) -> int:
    mailbox = MpiGroupMailbox.create(world_size=2)
    process = None
    with tempfile.TemporaryDirectory(prefix="simpler-mpi-mailbox-smoke-") as temp_dir:
        manifest_path = os.path.join(temp_dir, "mailbox.json")
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(mailbox.manifest(), manifest_file)
        cmd = [mpirun, "-np", "2", python, os.path.abspath(__file__), "--worker", manifest_path]
        try:
            process = subprocess.Popen(cmd, start_new_session=True)
            deadline = time.monotonic() + 20.0
            _wait_until(
                lambda: mailbox.group_state is not MailboxGroupState.INITIALIZING or process.poll() is not None,
                deadline=deadline,
                label="MPI mailbox READY",
            )
            if process.poll() is not None:
                raise RuntimeError(f"mpirun exited during startup with status {process.returncode}")
            if mailbox.group_state is not MailboxGroupState.READY:
                raise RuntimeError(f"MPI mailbox failed during startup: {mailbox.terminal_reason()}")

            mailbox.write_request(
                sequence_id=1,
                opcode=MailboxOpcode.TASK,
                target=MailboxTarget.PER_RANK,
                target_rank=-1,
                payloads=(b'{"value": 7}', b'{"value": 11}'),
            )
            _wait_until(
                lambda: mailbox.request_state in (MailboxRequestState.TASK_DONE, MailboxRequestState.TASK_FAILED),
                deadline=time.monotonic() + 10.0,
                label="per-rank task",
            )
            result = mailbox.read_result(sequence_id=1)
            decoded = tuple(json.loads(payload.decode()) for payload in result.payloads)
            assert decoded == (
                {"input": 7, "rank": 0, "result": 14},
                {"input": 11, "rank": 1, "result": 33},
            )
            print(f"[mpi-mailbox-local] per-rank results: {decoded}")

            mailbox.write_request(
                sequence_id=2,
                opcode=MailboxOpcode.TASK,
                target=MailboxTarget.GROUP,
                target_rank=-1,
                payloads=(b'{"value": 5, "fail_rank": 1}',),
            )
            _wait_until(
                lambda: mailbox.request_state in (MailboxRequestState.TASK_DONE, MailboxRequestState.TASK_FAILED),
                deadline=time.monotonic() + 10.0,
                label="rank failure",
            )
            try:
                mailbox.read_result(sequence_id=2)
            except MpiGroupError as exc:
                if "rank 1" not in str(exc):
                    raise
                print(f"[mpi-mailbox-local] expected rank failure: {exc}")
            else:
                raise AssertionError("rank-1 failure unexpectedly succeeded")

            mailbox.write_request(
                sequence_id=3,
                opcode=MailboxOpcode.SHUTDOWN,
                target=MailboxTarget.GROUP,
                target_rank=-1,
                payloads=(b"shutdown",),
            )
            _wait_until(
                lambda: mailbox.request_state is MailboxRequestState.SHUTDOWN_DONE,
                deadline=time.monotonic() + 10.0,
                label="shutdown",
            )
            process.wait(timeout=10.0)
            if process.returncode != 0:
                raise RuntimeError(f"mpirun returned status {process.returncode}")
            print("[mpi-mailbox-local] PASS: READY, one per-rank request, ranked failure, and shutdown")
            return 0
        finally:
            if process is not None and process.poll() is None:
                if os.name == "posix":
                    os.killpg(process.pid, 9)
                else:
                    process.kill()
                process.wait(timeout=5.0)
            mailbox.close(unlink=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--worker")
    args = parser.parse_args()
    if args.worker:
        return _worker(args.worker)
    return _parent(args.mpirun, args.python)


if __name__ == "__main__":
    sys.exit(main())
