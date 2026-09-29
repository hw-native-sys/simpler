#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compute on both L2s first, then run a Global CommDomain peer TLOAD in the same Worker lifetime."""

from __future__ import annotations

import argparse
import contextlib
import struct
from pathlib import Path

from simpler.callable_identity import CallableHandle
from simpler.remote_l3_protocol import HOST_TCP_TRANSPORT_PROFILE
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    CommBufferSpec,
    CoreCallable,
    DataType,
    GlobalCommDomainHandle,
    TaskArgs,
    Tensor,
    TensorArgType,
)
from simpler.worker import RemoteCallable, RemoteWorkerSpec, Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

REMOTE_COMPUTE_TARGET = "examples.workers.l4.compute_then_tload_mixed_l3.main:remote_compute_orch"
REMOTE_TLOAD_TARGET = "examples.workers.l4.compute_then_tload_mixed_l3.main:remote_tload_orch"
COUNT = 256
FLOAT_NBYTES = 4
WINDOW_SIZE = 4096
RANK_LABELS = ("local", "remote")
_LOCAL_COMPUTE_HANDLE: CallableHandle | None = None
_LOCAL_TLOAD_HANDLE: CallableHandle | None = None
_LOCAL_KEEPALIVE: list[TaskArgs] = []
_REMOTE_KEEPALIVE: list[TaskArgs] = []


def _digest_from_scalars(args: TaskArgs, start: int) -> bytes:
    return b"".join(int(args.scalar(start + index)).to_bytes(8, "little") for index in range(4))


def _window_tensor(context, buffer_name: str):
    return Tensor(context.buffers[buffer_name], shapes=(COUNT,), dtype=DataType.FLOAT32)


def _submit_compute_task(orch, chip_handle: CallableHandle, args: TaskArgs, cfg: CallConfig) -> TaskArgs:
    domain_id = int(args.scalar(0))
    local_worker_id = int(args.scalar(1))
    context = orch.get_global_domain(domain_id)[local_worker_id]

    chip_args = TaskArgs()
    chip_args.add_tensor(_window_tensor(context, "lhs"), TensorArgType.INPUT)
    chip_args.add_tensor(_window_tensor(context, "rhs"), TensorArgType.INPUT)
    chip_args.add_tensor(_window_tensor(context, "input"), TensorArgType.OUTPUT_EXISTING)
    orch.submit_next_level(chip_handle, chip_args, cfg, worker=local_worker_id)
    return chip_args


def _submit_tload_task(orch, chip_handle: CallableHandle, args: TaskArgs, cfg: CallConfig) -> TaskArgs:
    domain_id = int(args.scalar(0))
    local_worker_id = int(args.scalar(1))
    context = orch.get_global_domain(domain_id)[local_worker_id]

    chip_args = TaskArgs()
    chip_args.add_tensor(_window_tensor(context, "input"), TensorArgType.INPUT)
    chip_args.add_tensor(_window_tensor(context, "result"), TensorArgType.OUTPUT_EXISTING)
    chip_args.add_scalar(context.domain_size)
    chip_args.add_scalar(context.device_ctx)
    orch.submit_next_level(chip_handle, chip_args, cfg, worker=local_worker_id)
    return chip_args


def local_compute_orch(orch, args: TaskArgs, cfg: CallConfig) -> None:
    """Submit the vector-add chip task from the forked local L3 worker."""
    if _LOCAL_COMPUTE_HANDLE is None:
        raise RuntimeError("local L3 compute handle was not installed before fork")
    if args.scalar_count() != 2:
        raise ValueError("local compute task expects domain_id and local_worker_id")
    _LOCAL_KEEPALIVE.append(_submit_compute_task(orch, _LOCAL_COMPUTE_HANDLE, args, cfg))


def local_tload_orch(orch, args: TaskArgs, cfg: CallConfig) -> None:
    """Submit the peer-TLOAD chip task from the forked local L3 worker."""
    if _LOCAL_TLOAD_HANDLE is None:
        raise RuntimeError("local L3 TLOAD handle was not installed before fork")
    if args.scalar_count() != 2:
        raise ValueError("local TLOAD task expects domain_id and local_worker_id")
    _LOCAL_KEEPALIVE.append(_submit_tload_task(orch, _LOCAL_TLOAD_HANDLE, args, cfg))


def remote_compute_orch(orch, args: TaskArgs, cfg: CallConfig) -> None:
    """Submit the vector-add chip task from the daemon-started remote L3 worker."""
    from simpler.remote_l3_session import get_inner_handle  # noqa: PLC0415

    if args.scalar_count() != 6:
        raise ValueError("remote compute task expects domain_id, local_worker_id, and four digest scalars")
    chip_handle = get_inner_handle(_digest_from_scalars(args, 2).hex())
    _REMOTE_KEEPALIVE.append(_submit_compute_task(orch, chip_handle, args, cfg))


def remote_tload_orch(orch, args: TaskArgs, cfg: CallConfig) -> None:
    """Submit the peer-TLOAD chip task from the daemon-started remote L3 worker."""
    from simpler.remote_l3_session import get_inner_handle  # noqa: PLC0415

    if args.scalar_count() != 6:
        raise ValueError("remote TLOAD task expects domain_id, local_worker_id, and four digest scalars")
    chip_handle = get_inner_handle(_digest_from_scalars(args, 2).hex())
    _REMOTE_KEEPALIVE.append(_submit_tload_task(orch, chip_handle, args, cfg))


def _build_chip_callable(
    *,
    platform: str,
    runtime: str,
    kernel_name: str,
    orch_name: str,
    signature: list[ArgDirection],
    func_name: str,
    config_name: str,
) -> ChipCallable:
    kernels = Path(__file__).resolve().parent / "kernels"
    compiler = KernelCompiler(platform=platform)
    include_dirs = list(compiler.get_orchestration_include_dirs(runtime))
    include_dirs.append(str(compiler.project_root / "src" / "common"))
    kernel_binary = compiler.compile_incore(
        source_path=str(kernels / "aiv" / kernel_name),
        core_type="aiv",
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=include_dirs,
    )
    if not platform.endswith("sim"):
        kernel_binary = extract_text_section(kernel_binary)
    orch_binary = compiler.compile_orchestration(
        runtime_name=runtime,
        source_path=str(kernels / "orchestration" / orch_name),
    )
    core = CoreCallable.build(signature=signature, binary=kernel_binary)
    return ChipCallable.build(
        signature=signature,
        func_name=func_name,
        config_name=config_name,
        binary=orch_binary,
        children=[(0, core)],
    )


def _build_compute_callable(platform: str, runtime: str) -> ChipCallable:
    return _build_chip_callable(
        platform=platform,
        runtime=runtime,
        kernel_name="local_add_kernel.cpp",
        orch_name="local_add_orch.cpp",
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
        func_name="local_add_orchestration",
        config_name="local_add_orchestration_config",
    )


def _build_tload_callable(platform: str, runtime: str) -> ChipCallable:
    return _build_chip_callable(
        platform=platform,
        runtime=runtime,
        kernel_name="global_tload_kernel.cpp",
        orch_name="global_tload_orch.cpp",
        signature=[ArgDirection.IN, ArgDirection.OUT],
        func_name="global_tload_orchestration",
        config_name="global_tload_orchestration_config",
    )


def _add_digest_scalars(task_args: TaskArgs, digest: bytes) -> None:
    if len(digest) != 32:
        raise ValueError("inner chip callable digest must be 32 bytes")
    for offset in range(0, 32, 8):
        task_args.add_scalar(int.from_bytes(digest[offset : offset + 8], "little"))


def _parse_first_device(value: str, *, label: str) -> int:
    device_ids = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not device_ids or device_ids[0] < 0:
        raise ValueError(f"{label} device list must start with a non-negative device id")
    return device_ids[0]


def _lhs_values(rank: int) -> tuple[float, ...]:
    return tuple(float(rank * 100 + index) for index in range(COUNT))


def _rhs_values(rank: int) -> tuple[float, ...]:
    return tuple(float(rank * 10 + 2 * index) for index in range(COUNT))


def _expected_compute(rank: int) -> tuple[float, ...]:
    return tuple(lhs + rhs for lhs, rhs in zip(_lhs_values(rank), _rhs_values(rank)))


def _expected_communication(rank_count: int) -> tuple[float, ...]:
    rank_results = tuple(_expected_compute(rank) for rank in range(rank_count))
    return tuple(sum(rank_result[index] for rank_result in rank_results) for index in range(COUNT))


def _unpack_floats(raw: bytes) -> tuple[float, ...]:
    return tuple(float(value) for value in struct.unpack(f"<{COUNT}f", raw))


def _max_diff(actual: tuple[float, ...], expected: tuple[float, ...]) -> float:
    return max(abs(observed - wanted) for observed, wanted in zip(actual, expected))


def _local_args(domain) -> TaskArgs:
    args = TaskArgs()
    args.add_scalar(domain.domain_id)
    args.add_scalar(0)
    return args


def _remote_args(domain, digest: bytes) -> TaskArgs:
    args = _local_args(domain)
    _add_digest_scalars(args, digest)
    return args


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", required=True, help="remote L3 daemon endpoint, HOST:PORT")
    parser.add_argument("--local-devices", default="0", help="device list whose first id the forked local L3 owns")
    parser.add_argument("--remote-devices", default="0", help="device list whose first id the daemon L3 owns")
    parser.add_argument("--platform", default="a2a3")
    parser.add_argument("--runtime", default="tensormap_and_ringbuffer")
    parser.add_argument("--comm-profile", default="a3-fabric-v1")
    parser.add_argument("--session-timeout", type=float, default=120.0)
    parser.add_argument("--session-listen-host", default="0.0.0.0")
    return parser.parse_args()


def run(  # noqa: PLR0915 -- one linear two-phase scenario; splitting it would scatter the phase ordering this example exists to show
    *,
    remote: str,
    local_devices: str,
    remote_devices: str,
    platform: str = "a2a3",
    runtime: str = "tensormap_and_ringbuffer",
    comm_profile: str = "a3-fabric-v1",
    session_timeout: float = 120.0,
    session_listen_host: str = "0.0.0.0",  # noqa: S104 - Remote peer callbacks need a reachable listener.
) -> int:
    # The local L3 is a fork of this process, so its orchestration functions
    # reach the handles only through module state; locals would not survive
    # into the child.
    global _LOCAL_COMPUTE_HANDLE, _LOCAL_TLOAD_HANDLE  # noqa: PLW0603

    local_device = _parse_first_device(local_devices, label="local")
    remote_device = _parse_first_device(remote_devices, label="remote")

    local_l3: Worker | None = None
    local_l3_attached = False
    worker: Worker | None = None
    domain_handle: GlobalCommDomainHandle | None = None
    parent_keepalive: list[TaskArgs] = []
    try:
        compute_callable = _build_compute_callable(platform, runtime)
        tload_callable = _build_tload_callable(platform, runtime)
        local_l3 = Worker(
            level=3,
            device_ids=[local_device],
            num_sub_workers=0,
            platform=platform,
            runtime=runtime,
            comm_profile=comm_profile,
            global_device_ranks=(0,),
        )
        _LOCAL_COMPUTE_HANDLE = local_l3.register(compute_callable)
        _LOCAL_TLOAD_HANDLE = local_l3.register(tload_callable)

        worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=session_timeout)
        local_node = worker.add_worker(local_l3)
        local_l3_attached = True
        remote_node = worker.add_remote_worker(
            RemoteWorkerSpec(
                endpoint=remote,
                platform=platform,
                runtime=runtime,
                device_ids=(remote_device,),
                transport=HOST_TCP_TRANSPORT_PROFILE,
                comm_profile=comm_profile,
                global_device_ranks=(1,),
                session_listen_host=session_listen_host,
                allow_wildcard_session_bind=True,
            )
        )
        compute_handle = worker.register(compute_callable)
        tload_handle = worker.register(tload_callable)
        local_compute_handle = worker.register(local_compute_orch)
        local_tload_handle = worker.register(local_tload_orch)
        remote_compute_handle = worker.register(RemoteCallable(REMOTE_COMPUTE_TARGET), workers=[remote_node])
        remote_tload_handle = worker.register(RemoteCallable(REMOTE_TLOAD_TARGET), workers=[remote_node])
        worker.init()

        def compute_phase(orch, _args, cfg):
            nonlocal domain_handle
            domain = orch.allocate_global_domain(
                name="compute-then-tload-mixed-l3",
                members=((local_node, 0), (remote_node, 0)),
                window_size=WINDOW_SIZE,
                buffers=(
                    CommBufferSpec("lhs", "float32", COUNT, COUNT * FLOAT_NBYTES),
                    CommBufferSpec("rhs", "float32", COUNT, COUNT * FLOAT_NBYTES),
                    CommBufferSpec("input", "float32", COUNT, COUNT * FLOAT_NBYTES),
                    CommBufferSpec("result", "float32", COUNT, COUNT * FLOAT_NBYTES),
                ),
                retain_after_run=True,
            )
            for rank in range(len(RANK_LABELS)):
                orch.copy_to_global_domain(domain, rank, struct.pack(f"<{COUNT}f", *_lhs_values(rank)), buffer="lhs")
                orch.copy_to_global_domain(domain, rank, struct.pack(f"<{COUNT}f", *_rhs_values(rank)), buffer="rhs")
            local_task = _local_args(domain)
            remote_task = _remote_args(domain, compute_handle.digest)
            parent_keepalive[:] = [local_task, remote_task]
            orch.submit_next_level(local_compute_handle, local_task, cfg, worker=local_node)
            orch.submit_next_level(remote_compute_handle, remote_task, cfg, worker=remote_node)
            domain_handle = domain

        # run() drains the DAG before returning, so the compute tasks are
        # complete on both L2s before the communication phase reads their
        # outputs or issues any peer TLOAD.
        worker.run(compute_phase, args=None, config=CallConfig())
        if domain_handle is None:
            raise RuntimeError("the Global CommDomain was not allocated")
        domain = domain_handle
        observed_compute: list[tuple[float, ...]] = []
        observed_communication: list[tuple[float, ...]] = []

        def communication_phase(orch, _args, cfg):
            for rank in range(len(RANK_LABELS)):
                raw = orch.copy_from_global_domain(domain, rank, COUNT * FLOAT_NBYTES, buffer="input")
                observed_compute.append(_unpack_floats(raw))
            local_task = _local_args(domain)
            remote_task = _remote_args(domain, tload_handle.digest)
            parent_keepalive[:] = [local_task, remote_task]
            orch.submit_next_level(local_tload_handle, local_task, cfg, worker=local_node)
            orch.submit_next_level(remote_tload_handle, remote_task, cfg, worker=remote_node)

        worker.run(communication_phase, args=None, config=CallConfig())

        def verify_phase(orch, _args, _cfg):
            try:
                for rank in range(len(RANK_LABELS)):
                    raw = orch.copy_from_global_domain(domain, rank, COUNT * FLOAT_NBYTES, buffer="result")
                    observed_communication.append(_unpack_floats(raw))
            finally:
                domain.release()

        worker.run(verify_phase, args=None, config=CallConfig())

        failed: list[str] = []
        for rank, result in enumerate(observed_compute):
            max_diff = _max_diff(result, _expected_compute(rank))
            print(f"[compute-then-tload-mixed-l3] compute {RANK_LABELS[rank]} rank={rank} max_diff={max_diff:.3e}")
            if max_diff > 1e-5:
                failed.append(f"compute rank {rank}")

        expected_communication = _expected_communication(len(RANK_LABELS))
        for rank, result in enumerate(observed_communication):
            max_diff = _max_diff(result, expected_communication)
            print(
                f"[compute-then-tload-mixed-l3] communication {RANK_LABELS[rank]} rank={rank} max_diff={max_diff:.3e}"
            )
            if max_diff > 1e-3:
                failed.append(f"communication rank {rank}")
        if failed:
            raise AssertionError(f"golden mismatch: {', '.join(failed)}")

        print("compute_then_tload_mixed_l3 passed")
        return 0
    finally:
        parent_keepalive.clear()
        _LOCAL_KEEPALIVE.clear()
        _REMOTE_KEEPALIVE.clear()
        if domain_handle is not None and not domain_handle.freed:
            with contextlib.suppress(Exception):
                domain_handle.release()
        try:
            if worker is not None:
                worker.close()
        finally:
            if local_l3 is not None and not local_l3_attached:
                with contextlib.suppress(Exception):
                    local_l3.close()


def main() -> int:
    args = _parse_args()
    return run(
        remote=args.remote,
        local_devices=args.local_devices,
        remote_devices=args.remote_devices,
        platform=args.platform,
        runtime=args.runtime,
        comm_profile=args.comm_profile,
        session_timeout=args.session_timeout,
        session_listen_host=args.session_listen_host,
    )


if __name__ == "__main__":
    raise SystemExit(main())
