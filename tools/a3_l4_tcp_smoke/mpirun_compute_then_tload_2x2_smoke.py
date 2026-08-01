#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the mailbox->MPI two-L3/four-L2 A3 compute + peer-TLOAD smoke on servers 37 and 35."""

from __future__ import annotations

import argparse
import os
import struct
import sys
from multiprocessing.shared_memory import SharedMemory

from simpler.task_interface import ArgDirection, CallConfig, ChipCallable, CommBufferSpec, CoreCallable, TaskArgs
from simpler.worker import MpiL3GroupSpec, RemoteCallable, Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_ADD_AIV = os.path.join(HERE, "kernels", "aiv", "local_add_kernel.cpp")
LOCAL_ADD_ORCH = os.path.join(HERE, "kernels", "orchestration", "local_add_orch.cpp")
GLOBAL_TLOAD_AIV = os.path.join(HERE, "kernels", "aiv", "global_tload_kernel.cpp")
GLOBAL_TLOAD_ORCH = os.path.join(HERE, "kernels", "orchestration", "global_tload_orch.cpp")
COUNT = 256
FLOAT_BYTES = 4
WINDOW_SIZE = 4096


def _csv_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if len(values) != 2:
        raise ValueError("this smoke requires exactly two devices on each server")
    return values


def _csv_strings(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if len(values) != 2:
        raise ValueError("this smoke requires exactly two RoCE addresses on each server")
    return values


def _compile_aiv(compiler: KernelCompiler, platform: str, runtime: str, source: str) -> bytes:
    include_dirs = compiler.get_orchestration_include_dirs(runtime)
    kernel_include_dirs = list(include_dirs) + [str(compiler.project_root / "src" / "common")]
    binary = compiler.compile_incore(
        source_path=source,
        core_type="aiv",
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=kernel_include_dirs,
    )
    return binary if platform.endswith("sim") else extract_text_section(binary)


def _build_chip_callable(
    platform: str,
    runtime: str,
    kernel_source: str,
    orchestration_source: str,
    signature: list[ArgDirection],
    func_name: str,
    config_name: str,
) -> ChipCallable:
    compiler = KernelCompiler(platform=platform)
    core = CoreCallable.build(
        signature=signature,
        binary=_compile_aiv(compiler, platform, runtime, kernel_source),
    )
    return ChipCallable.build(
        signature=signature,
        func_name=func_name,
        config_name=config_name,
        binary=compiler.compile_orchestration(runtime_name=runtime, source_path=orchestration_source),
        children=[(0, core)],
    )


def _digest_scalars(digest: bytes) -> tuple[int, ...]:
    if len(digest) != 32:
        raise ValueError("callable digest must be 32 bytes")
    return tuple(int.from_bytes(digest[offset : offset + 8], "little") for offset in range(0, 32, 8))


def _rank_args(domain_id: int, local_worker_count: int, digest: bytes) -> TaskArgs:
    args = TaskArgs()
    args.add_scalar(domain_id)
    args.add_scalar(local_worker_count)
    for value in _digest_scalars(digest):
        args.add_scalar(value)
    return args


def _lhs(rank: int) -> tuple[float, ...]:
    return tuple(float(rank * 100 + index) for index in range(COUNT))


def _rhs(rank: int) -> tuple[float, ...]:
    return tuple(float(rank * 10 + 2 * index) for index in range(COUNT))


def _compute_expected(rank: int) -> tuple[float, ...]:
    return tuple(a + b for a, b in zip(_lhs(rank), _rhs(rank), strict=True))


def _tload_expected(rank_count: int) -> tuple[float, ...]:
    by_rank = tuple(_compute_expected(rank) for rank in range(rank_count))
    return tuple(sum(values[index] for values in by_rank) for index in range(COUNT))


def _unpack(raw: bytes) -> tuple[float, ...]:
    return tuple(float(value) for value in struct.unpack(f"<{COUNT}f", raw))


def _max_diff(actual: tuple[float, ...], expected: tuple[float, ...]) -> float:
    return max(abs(a - b) for a, b in zip(actual, expected, strict=True))


def _assert_mailbox_unlinked(name: str) -> None:
    try:
        reopened = SharedMemory(name=name, create=False)
    except FileNotFoundError:
        return
    reopened.close()
    raise AssertionError(f"MPI group shared-memory object {name!r} remained after shutdown")


def run(args: argparse.Namespace) -> int:
    devices37 = _csv_ints(args.devices_37)
    devices35 = _csv_ints(args.devices_35)
    roce37 = _csv_strings(args.roce_37)
    roce35 = _csv_strings(args.roce_35)
    print(f"[mpi-mailbox-2x2] rank0/L3 server37={args.host_37}, devices={devices37}, roce={roce37}")
    print(f"[mpi-mailbox-2x2] rank1/L3 server35={args.host_35}, devices={devices35}, roce={roce35}")

    compute_callable = _build_chip_callable(
        args.platform,
        args.runtime,
        LOCAL_ADD_AIV,
        LOCAL_ADD_ORCH,
        [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
        "local_add_orchestration",
        "local_add_orchestration_config",
    )
    tload_callable = _build_chip_callable(
        args.platform,
        args.runtime,
        GLOBAL_TLOAD_AIV,
        GLOBAL_TLOAD_ORCH,
        [ArgDirection.IN, ArgDirection.OUT],
        "global_tload_orchestration",
        "global_tload_orchestration_config",
    )
    mpirun_args = tuple(args.mpirun_arg) if args.mpirun_arg else ("--map-by", "ppr:1:node")
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=args.timeout_s)
    node37, node35 = worker.add_mpirun_worker_group(
        MpiL3GroupSpec(
            hosts=(f"{args.host_37}:1", f"{args.host_35}:1"),
            platform=args.platform,
            device_ids_by_rank=(devices37, devices35),
            runtime=args.runtime,
            comm_profile="a3-fabric-v1",
            global_device_ranks_by_rank=((0, 1), (2, 3)),
            mpirun_path=args.mpirun_path,
            mpirun_args=mpirun_args,
            python_executable=args.python_executable,
        )
    )
    compute_handle = worker.register(compute_callable)
    tload_handle = worker.register(tload_callable)
    rank_compute_handle = worker.register(
        RemoteCallable("simpler.global_comm_smoke:remote_compute_group_orch"),
        workers=[node37, node35],
    )
    rank_tload_handle = worker.register(
        RemoteCallable("simpler.global_comm_smoke:remote_rank_group_orch"),
        workers=[node37, node35],
    )
    targets = ((node37, 0), (node37, 1), (node35, 0), (node35, 1))
    captured: dict[str, object] = {}
    compute_results: list[tuple[float, ...]] = []
    tload_results: list[tuple[float, ...]] = []
    mailbox_name = ""
    manifest_path = ""
    mpirun_process = None
    try:
        worker.init()
        group = worker._mpi_l3_groups[0]
        assert group.mailbox is not None
        mailbox_name = group.mailbox.name
        manifest_path = group.manifest_path
        mpirun_process = group.process
        if worker._remote_sessions:
            raise AssertionError("MPI group unexpectedly created a Simpler TCP Remote L3 session")
        print(f"[mpi-mailbox-2x2] mailbox={mailbox_name}; Simpler TCP sessions=0")

        def compute_phase(orch, _args, cfg):
            domain = orch.allocate_global_domain(
                name="a3-mpi-mailbox-2x2-compute-tload",
                members=targets,
                window_size=WINDOW_SIZE,
                buffers=(
                    CommBufferSpec("lhs", "float32", COUNT, COUNT * FLOAT_BYTES),
                    CommBufferSpec("rhs", "float32", COUNT, COUNT * FLOAT_BYTES),
                    CommBufferSpec("input", "float32", COUNT, COUNT * FLOAT_BYTES),
                    CommBufferSpec("result", "float32", COUNT, COUNT * FLOAT_BYTES),
                ),
                retain_after_run=True,
            )
            for global_rank in range(4):
                orch.copy_to_global_domain(
                    domain,
                    global_rank,
                    struct.pack(f"<{COUNT}f", *_lhs(global_rank)),
                    buffer="lhs",
                )
                orch.copy_to_global_domain(
                    domain,
                    global_rank,
                    struct.pack(f"<{COUNT}f", *_rhs(global_rank)),
                    buffer="rhs",
                )
            orch.submit_next_level_group(
                rank_compute_handle,
                [
                    _rank_args(domain.domain_id, 2, compute_handle.digest),
                    _rank_args(domain.domain_id, 2, compute_handle.digest),
                ],
                cfg,
                workers=[node37, node35],
            )
            captured["domain"] = domain

        worker.run(compute_phase, args=None, config=CallConfig())
        domain = captured["domain"]

        def tload_phase(orch, _args, cfg):
            for global_rank in range(4):
                compute_results.append(
                    _unpack(
                        orch.copy_from_global_domain(
                            domain,
                            global_rank,
                            COUNT * FLOAT_BYTES,
                            buffer="input",
                        )
                    )
                )
            orch.submit_next_level_group(
                rank_tload_handle,
                [
                    _rank_args(domain.domain_id, 2, tload_handle.digest),
                    _rank_args(domain.domain_id, 2, tload_handle.digest),
                ],
                cfg,
                workers=[node37, node35],
            )

        worker.run(tload_phase, args=None, config=CallConfig())

        def verify_phase(orch, _args, _cfg):
            try:
                for global_rank in range(4):
                    tload_results.append(
                        _unpack(
                            orch.copy_from_global_domain(
                                domain,
                                global_rank,
                                COUNT * FLOAT_BYTES,
                                buffer="result",
                            )
                        )
                    )
            finally:
                domain.release()

        worker.run(verify_phase, args=None, config=CallConfig())
        for global_rank, observed in enumerate(compute_results):
            diff = _max_diff(observed, _compute_expected(global_rank))
            print(f"[mpi-mailbox-2x2] compute global_rank={global_rank} max_diff={diff:.3e}")
            if diff > 1e-5:
                raise AssertionError(f"compute mismatch on global rank {global_rank}: max_diff={diff}")
        expected = _tload_expected(4)
        for global_rank, observed in enumerate(tload_results):
            diff = _max_diff(observed, expected)
            print(f"[mpi-mailbox-2x2] TLOAD global_rank={global_rank} max_diff={diff:.3e}")
            if diff > 1e-3:
                raise AssertionError(f"TLOAD mismatch on global rank {global_rank}: max_diff={diff}")
        print("[mpi-mailbox-2x2] PASS: one per-rank mailbox task drove both MPI ranks and all four L2 devices")
        return 0
    finally:
        worker.close()
        if mpirun_process is not None and mpirun_process.poll() is None:
            raise AssertionError("mpirun remained alive after Worker.close()")
        if mailbox_name:
            _assert_mailbox_unlinked(mailbox_name)
        if manifest_path and os.path.exists(manifest_path):
            raise AssertionError(f"MPI group manifest remained after shutdown: {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-37", required=True)
    parser.add_argument("--host-35", required=True)
    parser.add_argument("--roce-37", required=True)
    parser.add_argument("--roce-35", required=True)
    parser.add_argument("--devices-37", default="0,1")
    parser.add_argument("--devices-35", default="0,1")
    parser.add_argument("--platform", default="a2a3")
    parser.add_argument("--runtime", default="tensormap_and_ringbuffer")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--mpirun-path", default="mpirun")
    parser.add_argument("--mpirun-arg", action="append", default=[])
    parser.add_argument("--python-executable", default=sys.executable)
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
