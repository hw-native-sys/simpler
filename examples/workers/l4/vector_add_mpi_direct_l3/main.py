#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run one L4 MPI rank with local-host and remote-host real L3 MPI ranks."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from simpler.mpi_direct_supervisor import run_supervisor
from simpler.remote_l3_session import get_inner_handle
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    CoreCallable,
    DataType,
    RemoteTensorRef,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import RemoteCallable

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

CONTROLLER_TARGET = "examples.workers.l4.vector_add_mpi_direct_l3.main:controller"
REMOTE_ORCH_TARGET = "examples.workers.l4.vector_add_mpi_direct_l3.main:remote_l3_vector_orch"
ELEMENTS = 128 * 128
NBYTES = ELEMENTS * ctypes.sizeof(ctypes.c_float)
FloatArray = ctypes.c_float * ELEMENTS
_REMOTE_KEEPALIVE: list[TaskArgs] = []


def _parse_devices(value: str, *, label: str) -> tuple[int, ...]:
    device_ids = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not device_ids or any(device_id < 0 for device_id in device_ids):
        raise ValueError(f"{label} device list must contain non-negative ids")
    if len(set(device_ids)) != len(device_ids):
        raise ValueError(f"{label} device list must not contain duplicates")
    return device_ids


def _digest_from_args(args: TaskArgs) -> bytes:
    return b"".join(int(args.scalar(index)).to_bytes(8, "little", signed=False) for index in range(4))


def remote_l3_vector_orch(orch, args: TaskArgs, cfg: CallConfig) -> None:
    chip_handle = get_inner_handle(_digest_from_args(args).hex())
    chip_args = TaskArgs()
    chip_args.add_tensor(args.tensor(0), TensorArgType.INPUT)
    chip_args.add_tensor(args.tensor(1), TensorArgType.INPUT)
    chip_args.add_tensor(args.tensor(2), TensorArgType.OUTPUT_EXISTING)
    _REMOTE_KEEPALIVE[:] = [chip_args]
    orch.submit_next_level(chip_handle, chip_args, cfg, worker=0)


def _build_chip_callable(platform: str, runtime: str) -> ChipCallable:
    root = Path(__file__).resolve().parents[4]
    kernels = root / "examples" / "workers" / "l3" / "multi_chip_dispatch" / "kernels"
    compiler = KernelCompiler(platform=platform)
    kernel = compiler.compile_incore(
        source_path=str(kernels / "aiv" / "vector_add_kernel.cpp"),
        core_type="aiv",
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=compiler.get_orchestration_include_dirs(runtime),
    )
    orchestration = compiler.compile_orchestration(
        runtime_name=runtime,
        source_path=str(kernels / "orchestration" / "vector_add_orch.cpp"),
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
        func_name="vector_add_orchestration",
        binary=orchestration,
        children=[
            (
                0,
                CoreCallable.build(
                    signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
                    binary=extract_text_section(kernel),
                ),
            )
        ],
    )


def _array(value: float) -> Any:
    return FloatArray(*([value] * ELEMENTS))


def _task_args(handles, digest: bytes) -> TaskArgs:
    args = TaskArgs()
    for index, handle in enumerate(handles):
        direction = TensorArgType.OUTPUT_EXISTING if index == 2 else TensorArgType.INPUT
        args.add_tensor(RemoteTensorRef(handle, shape=(ELEMENTS,), dtype=DataType.FLOAT32), direction)
    for offset in range(0, len(digest), 8):
        args.add_scalar(int.from_bytes(digest[offset : offset + 8], "little", signed=False))
    return args


def controller(context) -> None:
    """Rank-0 controller invoked by ``simpler.mpi_direct_runtime``."""
    executors = context.topology.executors
    if len(executors) != 2:
        raise ValueError("vector_add_mpi_direct_l3 requires exactly two L3 executor ranks")
    worker_ids = tuple(spec.worker_id for spec in executors)
    worker = context.create_worker(
        num_sub_workers=0,
        startup_timeout_s=context.topology.startup_timeout_s,
        remote_session_timeout_s=context.topology.session_timeout_s,
    )
    allocations = []
    keepalive: list[TaskArgs] = []
    try:
        first = executors[0]
        chip_handle = worker.register(_build_chip_callable(first.platform, first.runtime))
        remote_handle = worker.register(RemoteCallable(REMOTE_ORCH_TARGET), workers=list(worker_ids))
        worker.init()

        digest = bytes(chip_handle.digest)
        expected: dict[int, tuple[Any, float]] = {}
        task_inputs: dict[int, list] = {}
        for index, worker_id in enumerate(worker_ids):
            lhs = float(2 + index * 4)
            rhs = float(3 + index * 4)
            handles = [worker.remote_malloc(worker=worker_id, nbytes=NBYTES) for _ in range(3)]
            allocations.extend(handles)
            worker.remote_copy_to(handles[0], _array(lhs), NBYTES)
            worker.remote_copy_to(handles[1], _array(rhs), NBYTES)
            worker.remote_copy_to(handles[2], _array(0.0), NBYTES)
            task_inputs[worker_id] = handles
            expected[worker_id] = (_array(0.0), lhs + rhs)

        members = tuple(
            (spec.worker_id, local_index) for spec in executors for local_index in range(len(spec.device_ids))
        )

        def parent_orch(orch, _args, cfg):
            orch.allocate_global_domain(
                name="vector-add-mpi-direct-l3",
                members=members,
                window_size=1024 * 1024,
            )
            local_args = [_task_args(task_inputs[worker_id], digest) for worker_id in worker_ids]
            keepalive[:] = local_args
            for worker_id, task_args in zip(worker_ids, local_args):
                orch.submit_next_level(remote_handle, task_args, cfg, worker=worker_id)

        config = CallConfig()
        config.aicpu_thread_num = 2
        worker.run(parent_orch, config=config)

        for worker_id, handles in task_inputs.items():
            output, wanted = expected[worker_id]
            worker.remote_copy_from(handles[2], output, NBYTES)
            max_diff = max(abs(float(output[index]) - wanted) for index in range(ELEMENTS))
            if max_diff > 1e-5:
                raise AssertionError(f"worker {worker_id} vector result mismatch: max_diff={max_diff}")
        print("vector_add_mpi_direct_l3 passed")
    finally:
        keepalive.clear()
        for handle in reversed(allocations):
            with contextlib.suppress(Exception):
                worker.remote_free(handle)


def run(  # noqa: PLR0913 -- mirrors the two-host CLI surface used by the other L4 examples
    *,
    local_host: str,
    remote_host: str,
    python_executable: str,
    local_devices: str,
    remote_devices: str,
    platform: str = "a2a3",
    runtime: str = "tensormap_and_ringbuffer",
    comm_profile: str = "a3-fabric-v1",
    startup_timeout: float = 180.0,
    session_timeout: float = 120.0,
    mpirun_path: str = "mpirun",
    launcher_family: str = "auto",
) -> int:
    local_ids = _parse_devices(local_devices, label="local")
    remote_ids = _parse_devices(remote_devices, label="remote")
    local_global_ranks = tuple(range(len(local_ids)))
    remote_global_ranks = tuple(range(len(local_ids), len(local_ids) + len(remote_ids)))
    topology = {
        "controller_rank": 0,
        "controller_host": local_host,
        "startup_timeout_s": float(startup_timeout),
        "session_timeout_s": float(session_timeout),
        "heartbeat_interval_s": 1.0,
        "max_pending_frame_bytes": 64 * 1024 * 1024,
        "launcher_args": ["-wdir", "/tmp"],
        "executor_ranks": [
            {
                "rank": 1,
                "worker_id": 0,
                "host": local_host,
                "platform": platform,
                "runtime": runtime,
                "device_ids": list(local_ids),
                "global_device_ranks": list(local_global_ranks),
                "num_sub_workers": 0,
                "comm_profile": comm_profile,
            },
            {
                "rank": 2,
                "worker_id": 1,
                "host": remote_host,
                "platform": platform,
                "runtime": runtime,
                "device_ids": list(remote_ids),
                "global_device_ranks": list(remote_global_ranks),
                "num_sub_workers": 0,
                "comm_profile": comm_profile,
            },
        ],
    }
    fd, topology_path = tempfile.mkstemp(prefix="simpler-mpi-direct-", suffix=".json", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(topology, stream, sort_keys=True)
            stream.write("\n")
        return run_supervisor(
            topology_path,
            CONTROLLER_TARGET,
            mpirun_path=mpirun_path,
            launcher_family=launcher_family,
            python_executable=python_executable,
        )
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(topology_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-host", required=True, help="L4/rank-0 machine address")
    parser.add_argument("--remote-host", required=True, help="peer L3 machine address")
    parser.add_argument("--python", dest="python_executable", required=True)
    parser.add_argument("--local-devices", default="0,1")
    parser.add_argument("--remote-devices", default="0,1")
    parser.add_argument("--platform", default="a2a3")
    parser.add_argument("--runtime", default="tensormap_and_ringbuffer")
    parser.add_argument("--comm-profile", default="a3-fabric-v1")
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--session-timeout", type=float, default=120.0)
    parser.add_argument("--mpirun-path", default="mpirun")
    parser.add_argument("--launcher-family", choices=("auto", "openmpi", "mpich"), default="auto")
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
        startup_timeout=args.startup_timeout,
        session_timeout=args.session_timeout,
        mpirun_path=args.mpirun_path,
        launcher_family=args.launcher_family,
    )


if __name__ == "__main__":
    raise SystemExit(main())
