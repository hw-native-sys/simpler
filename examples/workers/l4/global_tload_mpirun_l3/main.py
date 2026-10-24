#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run a two-machine mpirun-launched L3 group through a Global CommDomain peer TLOAD.

One L4 parent (this process) registers a two-rank ``MpiL3GroupSpec`` — rank 0
on this machine, rank 1 on the peer — and launches both L3 workers through a
single parent-owned ``mpirun``. Rank 0 must be this machine: only it can read
the parent-written group manifest, which it then broadcasts over MPI. Rank 0
marks the group's named shared-memory mailbox READY once every rank reports
in over MPI, the parent attaches the group as mailbox-backed remote L3
endpoints and dispatches one TLOAD task per NPU device on each machine. The
Global CommDomain (``a3-fabric-v1`` on real A3 devices) spans every device of
both ranks, so it is built over the MPI collective descriptor-exchange path
(full-group members; no L4 import fanout), and each device's AIV kernel reads
every peer window through the fabric and sums it — cross-machine link setup,
communication, and compute in one pass.

The kernels, the chip callable, and the rank-side orchestration are shared
with ``examples/workers/l4/global_tload_mixed_l3``; this example differs only
in how the two L3 ranks come up.

``mpirun`` executes one identical command line on both machines, so
``--python`` must name an interpreter path that is valid on BOTH: point it at
a per-machine launcher script installed at one shared absolute path, which
sources the CANN environment, enters that machine's copy of this source tree
(the rank imports this package's orchestration module by name), and execs
that machine's ``.venv`` python. ``mpirun`` (MPICH/Hydra or Open MPI) and
``mpi4py`` must be installed on both machines; the default ``--host h1,h2``
placement puts rank 0 on the first host.
"""

from __future__ import annotations

import argparse
import contextlib
import struct

from simpler.session_cleanup import CleanupOutcome, ShutdownCommand
from simpler.task_interface import (
    CallConfig,
    CommBufferSpec,
    GlobalCommDomainHandle,
    TaskArgs,
)
from simpler.worker import MpiL3GroupSpec, RemoteCallable, Worker

from examples.workers.l4.global_tload_mixed_l3.main import (
    COUNT,
    FLOAT_NBYTES,
    REMOTE_ORCH_TARGET,
    WINDOW_SIZE,
    _add_digest_scalars,
    _build_tload_callable,
    _expected_values,
    _input_values,
)

NRANKS = 2


def _parse_devices(value: str, *, label: str) -> tuple[int, ...]:
    device_ids = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not device_ids or any(device_id < 0 for device_id in device_ids):
        raise ValueError(f"{label} device list must be non-empty non-negative ids")
    return device_ids


def _check_cleanup_ledger(worker: Worker, node_ids: tuple[int, ...]) -> None:
    """Check that closing the group left L4 a usable release record for rank 0.

    This is the one case that carries the record end to end on hardware: rank 0
    writes it beside the group manifest and L4 reads it back after reaping
    `mpirun`. The other ranks share no filesystem with L4 and report
    `cleanup=UNKNOWN` by construction, so only rank 0 — `node_ids` is in rank
    order — can be checked. Runs after `close()`, which is what populates the
    ledger.

    Two things are deliberately not required.

    The release *outcome* need not be SUCCEEDED. A rank whose release fails is a
    defect in whatever failed, not in the ledger, and the ledger reports it
    rather than raising; demanding success here would make this example fail for
    a reason it does not test. What it does require is an answer: UNKNOWN means
    the record never arrived.

    `command` is not required to be OBSERVED either, because no shutdown command
    reaches this route. `MpiGroupMailboxChannel::shutdown()` runs only from
    `RemoteL3Endpoint::request_progress_stop()`, which a healthy close never
    calls, so the mailbox SHUTDOWN opcode is never sent and the group is reaped
    by the SIGTERM backstop instead. The command is printed rather than asserted;
    it reads `observed` once that delivery exists.
    """
    reports = worker.session_cleanup_reports()
    rank0_worker_id = node_ids[0]
    rank0 = reports.get(rank0_worker_id)
    if rank0 is None:
        raise AssertionError(
            f"no cleanup ledger row for rank 0 (worker {rank0_worker_id}); ledger holds {sorted(reports)}"
        )
    if not rank0.committed:
        raise AssertionError(f"rank 0 left no cleanup record L4 could read: {rank0.describe()}")
    if rank0.command is ShutdownCommand.UNKNOWN:
        raise AssertionError(f"rank 0 reported no shutdown-command outcome: {rank0.describe()}")
    if rank0.cleanup is CleanupOutcome.UNKNOWN:
        raise AssertionError(f"rank 0 reported no release outcome: {rank0.describe()}")
    print(f"global_tload_mpirun_l3 cleanup ledger: {rank0.describe()}")


def run(  # noqa: PLR0913, PLR0915 -- mirrors the CLI surface; one linear two-run flow like the mixed_l3 sibling
    *,
    local_host: str,
    remote_host: str,
    python_executable: str,
    local_devices: str,
    remote_devices: str,
    platform: str = "a2a3",
    runtime: str = "tensormap_and_ringbuffer",
    comm_profile: str = "a3-fabric-v1",
    session_timeout: float = 120.0,
    mpirun_path: str = "mpirun",
) -> int:
    local_ids = _parse_devices(local_devices, label="local")
    remote_ids = _parse_devices(remote_devices, label="remote")
    device_ids_by_rank = (local_ids, remote_ids)
    # One domain member per NPU device on either machine, in (rank, device)
    # order; the member count is what the TLOAD kernel sums across.
    member_count = len(local_ids) + len(remote_ids)
    global_ranks_by_rank = (
        tuple(range(len(local_ids))),
        tuple(range(len(local_ids), member_count)),
    )

    worker: Worker | None = None
    domain_handle: GlobalCommDomainHandle | None = None
    parent_keepalive: list[TaskArgs] = []
    node_ids: tuple[int, ...] = ()
    # The ledger check below runs only for a workload that got as far as its
    # golden comparison: on any earlier failure the group never reached a
    # graceful shutdown, so its record would say nothing about this code.
    workload_passed = False
    try:
        chip_callable = _build_tload_callable(platform, runtime)
        worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=session_timeout)
        node_ids = worker.add_mpirun_worker_group(
            MpiL3GroupSpec(
                hosts=(local_host, remote_host),
                platform=platform,
                runtime=runtime,
                device_ids_by_rank=device_ids_by_rank,
                comm_profile=comm_profile,
                global_device_ranks_by_rank=global_ranks_by_rank,
                mpirun_path=mpirun_path,
                # Hydra and Open MPI both propagate the parent's cwd to every
                # rank and fail the launch when it is missing on a remote
                # machine. Any directory that exists everywhere does here: the
                # --python launcher enters the real tree itself.
                mpirun_args=("-wdir", "/tmp"),
                python_executable=python_executable,
            )
        )
        chip_handle = worker.register(chip_callable)
        rank_handle = worker.register(RemoteCallable(REMOTE_ORCH_TARGET), workers=list(node_ids))
        worker.init()

        members = tuple(
            (node_id, local_index)
            for node_id, device_ids in zip(node_ids, device_ids_by_rank)
            for local_index in range(len(device_ids))
        )

        def build_and_run(orch, _args, cfg):
            nonlocal domain_handle
            domain = orch.allocate_global_domain(
                name="global-tload-mpirun-l3",
                members=members,
                window_size=WINDOW_SIZE,
                buffers=(
                    CommBufferSpec("input", "float32", COUNT, COUNT * FLOAT_NBYTES),
                    CommBufferSpec("result", "float32", COUNT, COUNT * FLOAT_NBYTES),
                ),
                retain_after_run=True,
            )
            for domain_rank in range(member_count):
                orch.copy_to_global_domain(
                    domain,
                    domain_rank,
                    struct.pack(f"<{COUNT}f", *_input_values(domain_rank)),
                    buffer="input",
                )
            for node_id, local_index in members:
                task_args = TaskArgs()
                task_args.add_scalar(domain.domain_id)
                task_args.add_scalar(local_index)
                _add_digest_scalars(task_args, chip_handle.digest)
                parent_keepalive.append(task_args)
                orch.submit_next_level(rank_handle, task_args, cfg, worker=node_id)
            # One full-group batched dispatch: every rank's task travels in a
            # single PER_RANK mailbox envelope. It re-runs TLOAD on each
            # rank's first device, rewriting the same summed values, so the
            # golden check below also proves the batched path executed.
            group_args = []
            for node_id in node_ids:
                task_args = TaskArgs()
                task_args.add_scalar(domain.domain_id)
                task_args.add_scalar(0)
                _add_digest_scalars(task_args, chip_handle.digest)
                parent_keepalive.append(task_args)
                group_args.append(task_args)
            orch.submit_next_level_group(rank_handle, group_args, cfg, workers=list(node_ids))
            domain_handle = domain

        worker.run(build_and_run, args=None, config=CallConfig())
        if domain_handle is None:
            raise RuntimeError("the Global CommDomain was not allocated")
        observed: list[tuple[float, ...]] = []

        def read_and_release(orch, _args, _cfg):
            domain = orch.get_domain("global-tload-mpirun-l3")
            try:
                for domain_rank in range(member_count):
                    raw = orch.copy_from_global_domain(domain, domain_rank, COUNT * FLOAT_NBYTES, buffer="result")
                    observed.append(tuple(float(value) for value in struct.unpack(f"<{COUNT}f", raw)))
            finally:
                domain.release()

        worker.run(read_and_release, args=None, config=CallConfig())

        expected = _expected_values(member_count)
        failed_ranks: list[int] = []
        for domain_rank, result in enumerate(observed):
            max_diff = max(abs(actual - wanted) for actual, wanted in zip(result, expected))
            node_id, local_index = members[domain_rank]
            print(
                f"[global-tload-mpirun-l3] domain_rank={domain_rank} "
                f"node={node_id} device_index={local_index} max_diff={max_diff:.3e}"
            )
            if max_diff > 1e-3:
                failed_ranks.append(domain_rank)
        if failed_ranks:
            raise AssertionError(f"peer TLOAD golden mismatch on domain rank(s) {failed_ranks}")

        print("global_tload_mpirun_l3 passed")
        workload_passed = True
        return 0
    finally:
        parent_keepalive.clear()
        if domain_handle is not None and not domain_handle.freed:
            with contextlib.suppress(Exception):
                domain_handle.release()
        if worker is not None:
            # close() is what populates the ledger, so the check follows it.
            worker.close()
            if workload_passed:
                _check_cleanup_ledger(worker, node_ids)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-host", required=True, help="this machine's numeric IP; rank 0 and READY bind here")
    parser.add_argument("--remote-host", required=True, help="peer machine's numeric IP; rank 1 runs there")
    parser.add_argument(
        "--python",
        dest="python_executable",
        required=True,
        help="interpreter path valid on BOTH machines (per-machine launcher at one shared absolute path)",
    )
    parser.add_argument("--local-devices", default="0", help="rank 0's NPU device ids, comma separated")
    parser.add_argument("--remote-devices", default="0", help="rank 1's NPU device ids, comma separated")
    parser.add_argument("--platform", default="a2a3")
    parser.add_argument("--runtime", default="tensormap_and_ringbuffer")
    parser.add_argument("--comm-profile", default="a3-fabric-v1")
    parser.add_argument("--session-timeout", type=float, default=120.0)
    parser.add_argument("--mpirun-path", default="mpirun")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return run(
        local_host=args.local_host,
        remote_host=args.remote_host,
        python_executable=args.python_executable,
        local_devices=args.local_devices,
        remote_devices=args.remote_devices,
        platform=args.platform,
        runtime=args.runtime,
        comm_profile=args.comm_profile,
        session_timeout=args.session_timeout,
        mpirun_path=args.mpirun_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
