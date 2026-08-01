#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device-free integration smoke for the real L4->mailbox->MPI L3 command path."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from multiprocessing.shared_memory import SharedMemory

from simpler.task_interface import CallConfig, TaskArgs
from simpler.worker import MpiL3GroupSpec, RemoteCallable, Worker


def _args(value: int) -> TaskArgs:
    args = TaskArgs()
    args.add_scalar(value)
    return args


def _assert_unlinked(name: str) -> None:
    try:
        reopened = SharedMemory(name=name, create=False)
    except FileNotFoundError:
        return
    reopened.close()
    raise AssertionError(f"MPI L3 mailbox {name!r} remained after shutdown")


def run(args: argparse.Namespace) -> int:  # noqa: PLR0915 -- integration lifecycle stays visible in one function
    with tempfile.TemporaryDirectory(prefix="simpler-mpi-l3-group-smoke-") as output_dir:
        os.environ["SIMPLER_MPI_SMOKE_DIR"] = output_dir
        worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=args.timeout_s)
        worker_ids = worker.add_mpirun_worker_group(
            MpiL3GroupSpec(
                hosts=("localhost", "localhost"),
                platform="a2a3sim",
                device_ids_by_rank=((0,), (1,)),
                num_sub_workers_by_rank=(0, 0),
                global_device_ranks_by_rank=((0,), (1,)),
                mpirun_path=args.mpirun,
                mpirun_args=(
                    "--oversubscribe",
                    "--mca",
                    "btl",
                    "self,vader,tcp",
                    "--mca",
                    "pml",
                    "ob1",
                ),
                python_executable=args.python,
            )
        )
        callback = worker.register(
            RemoteCallable("simpler.mpi_group_smoke:record_rank_value"),
            workers=list(worker_ids),
        )
        mailbox_name = ""
        manifest_path = ""
        process = None
        try:
            worker.init()
            group = worker._mpi_l3_groups[0]
            assert group.mailbox is not None
            assert group.manifest_path is not None
            mailbox_name = group.mailbox.name
            manifest_path = group.manifest_path
            process = group.process
            if worker._remote_sessions:
                raise AssertionError("MPI L3 group unexpectedly created a Simpler TCP session")
            with open(manifest_path, encoding="utf-8") as manifest_file:
                manifest_text = manifest_file.read()
            for forbidden in ("command_port", "health_port", "listen_host", "connect_host"):
                if forbidden in manifest_text:
                    raise AssertionError(f"MPI group manifest unexpectedly contains {forbidden}")

            def submit_values(orch, _run_args, cfg):
                orch.submit_next_level_group(
                    callback,
                    [_args(101), _args(202)],
                    cfg,
                    workers=list(worker_ids),
                )

            worker.run(submit_values, args=None, config=CallConfig())
            observed = []
            for rank in range(2):
                with open(os.path.join(output_dir, f"rank-{rank}.json"), encoding="utf-8") as output_file:
                    observed.append(json.load(output_file))
            if observed != [{"rank": 0, "value": 101}, {"rank": 1, "value": 202}]:
                raise AssertionError(f"unexpected per-rank callback results: {observed}")
            print(f"[mpi-l3-group-local] per-rank results: {observed}")

            def submit_failure(orch, _run_args, cfg):
                orch.submit_next_level_group(
                    callback,
                    [_args(303), _args(0xFFFF)],
                    cfg,
                    workers=list(worker_ids),
                )

            try:
                worker.run(submit_failure, args=None, config=CallConfig())
            except RuntimeError as exc:
                if "rank 1" not in str(exc):
                    raise
                print(f"[mpi-l3-group-local] expected rank failure: {exc}")
            else:
                raise AssertionError("rank-1 callback failure unexpectedly succeeded")

            worker.run(submit_values, args=None, config=CallConfig())
            print("[mpi-l3-group-local] PASS: real endpoint batching, ranked failure, reuse, and no Simpler TCP")
            return 0
        finally:
            worker.close()
            if process is not None and process.poll() is None:
                raise AssertionError("mpirun remained alive after MPI L3 group shutdown")
            if mailbox_name:
                _assert_unlinked(mailbox_name)
            if manifest_path and os.path.exists(manifest_path):
                raise AssertionError(f"MPI L3 group manifest remained after shutdown: {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
