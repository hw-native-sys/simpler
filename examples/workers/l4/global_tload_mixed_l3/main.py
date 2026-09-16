#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run one local L3 and one remote L3 rank through a Global CommDomain peer TLOAD."""

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
    TensorArgType,
)
from simpler.worker import RemoteCallable, RemoteWorkerSpec, Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

REMOTE_ORCH_TARGET = "examples.workers.l4.global_tload_mixed_l3.main:remote_rank_orch"
COUNT = 256
FLOAT_NBYTES = 4
WINDOW_SIZE = 4096
RANK_LABELS = ("local", "remote")
_LOCAL_CHIP_HANDLE: CallableHandle | None = None
_LOCAL_KEEPALIVE: list[TaskArgs] = []
_REMOTE_KEEPALIVE: list[TaskArgs] = []


def _digest_from_scalars(args: TaskArgs, start: int) -> bytes:
    return b"".join(int(args.scalar(start + index)).to_bytes(8, "little") for index in range(4))


def _submit_tload_task(orch, chip_handle: CallableHandle, args: TaskArgs, cfg: CallConfig) -> TaskArgs:
    domain_id = int(args.scalar(0))
    local_worker_id = int(args.scalar(1))
    context = orch.get_global_domain(domain_id)[local_worker_id]

    chip_args = TaskArgs()
    chip_args.add_tensor(
        context.buffers["input"].tensor((COUNT,), DataType.FLOAT32),
        TensorArgType.INPUT,
    )
    chip_args.add_tensor(
        context.buffers["result"].tensor((COUNT,), DataType.FLOAT32),
        TensorArgType.OUTPUT_EXISTING,
    )
    chip_args.add_scalar(context.domain_size)
    chip_args.add_scalar(context.device_ctx)
    orch.submit_next_level(chip_handle, chip_args, cfg, worker=local_worker_id)
    return chip_args


def local_rank_orch(orch, args: TaskArgs, cfg: CallConfig) -> None:
    """Submit the peer-TLOAD chip task from the forked local L3 worker."""
    if _LOCAL_CHIP_HANDLE is None:
        raise RuntimeError("local L3 chip handle was not installed before fork")
    if args.scalar_count() != 2:
        raise ValueError("local TLOAD task expects domain_id and local_worker_id")
    _LOCAL_KEEPALIVE[:] = [_submit_tload_task(orch, _LOCAL_CHIP_HANDLE, args, cfg)]


def remote_rank_orch(orch, args: TaskArgs, cfg: CallConfig) -> None:
    """Submit the peer-TLOAD chip task from the daemon-started remote L3 worker."""
    from simpler.remote_l3_session import get_inner_handle  # noqa: PLC0415

    if args.scalar_count() != 6:
        raise ValueError("remote TLOAD task expects domain_id, local_worker_id, and four digest scalars")
    chip_handle = get_inner_handle(_digest_from_scalars(args, 2).hex())
    _REMOTE_KEEPALIVE[:] = [_submit_tload_task(orch, chip_handle, args, cfg)]


def _build_tload_callable(platform: str, runtime: str) -> ChipCallable:
    kernels = Path(__file__).resolve().parent / "kernels"
    compiler = KernelCompiler(platform=platform)
    include_dirs = list(compiler.get_orchestration_include_dirs(runtime))
    include_dirs.append(str(compiler.project_root / "src" / "common"))
    kernel_binary = compiler.compile_incore(
        source_path=str(kernels / "aiv" / "global_tload_kernel.cpp"),
        core_type="aiv",
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=include_dirs,
    )
    if not platform.endswith("sim"):
        kernel_binary = extract_text_section(kernel_binary)
    orch_binary = compiler.compile_orchestration(
        runtime_name=runtime,
        source_path=str(kernels / "orchestration" / "global_tload_orch.cpp"),
    )
    core = CoreCallable.build(signature=[ArgDirection.IN, ArgDirection.OUT], binary=kernel_binary)
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT],
        func_name="global_tload_orchestration",
        config_name="global_tload_orchestration_config",
        binary=orch_binary,
        children=[(0, core)],
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


def _input_values(rank: int) -> tuple[float, ...]:
    return tuple(float(rank * 100 + index) for index in range(COUNT))


def _expected_values(rank_count: int) -> tuple[float, ...]:
    rank_bias = 100 * rank_count * (rank_count - 1) // 2
    return tuple(float(rank_count * index + rank_bias) for index in range(COUNT))


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
    parser.add_argument(
        "--enable-chip-swimlane",
        nargs="?",
        const=4,
        default=0,
        type=int,
        metavar="PERF_LEVEL",
        help="Capture a chip swimlane on every rank this run reaches, the peer machine's included. "
        "Bare flag=level 4 (full). Needs --output-prefix to have somewhere to land.",
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Directory this run's diagnostic artifacts go under: each rank's chip_swimlane_records.json, "
        "and this process's host.<pid>.log. Without one the host log stays on stderr and no capture is written.",
    )
    return parser.parse_args()


def run(
    *,
    remote: str,
    local_devices: str,
    remote_devices: str,
    platform: str = "a2a3",
    runtime: str = "tensormap_and_ringbuffer",
    comm_profile: str = "a3-fabric-v1",
    session_timeout: float = 120.0,
    session_listen_host: str = "0.0.0.0",  # noqa: S104 - Remote peer callbacks need a reachable listener.
    enable_chip_swimlane: int = 0,
    output_prefix: str = "",
) -> int:
    # The local L3 is a fork of this process, so its orchestration function
    # reaches the handle only through module state; a local would not survive
    # into the child.
    global _LOCAL_CHIP_HANDLE  # noqa: PLW0603

    def call_config() -> CallConfig:
        """The config every run this process issues carries.

        The diagnostics ride it down each dispatch, and the remote L3 decodes
        these same fields off the wire, so one flag covers both machines rather
        than needing the peer to be configured on its own.
        """
        cfg = CallConfig()
        if enable_chip_swimlane:
            cfg.enable_chip_swimlane = enable_chip_swimlane
        if output_prefix:
            cfg.output_prefix = output_prefix
        return cfg

    local_device = _parse_first_device(local_devices, label="local")
    remote_device = _parse_first_device(remote_devices, label="remote")

    local_l3: Worker | None = None
    local_l3_attached = False
    worker: Worker | None = None
    domain_handle: GlobalCommDomainHandle | None = None
    parent_keepalive: list[TaskArgs] = []
    try:
        chip_callable = _build_tload_callable(platform, runtime)
        local_l3 = Worker(
            level=3,
            device_ids=[local_device],
            num_sub_workers=0,
            platform=platform,
            runtime=runtime,
            comm_profile=comm_profile,
            global_device_ranks=(0,),
        )
        _LOCAL_CHIP_HANDLE = local_l3.register(chip_callable)

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
        chip_handle = worker.register(chip_callable)
        local_handle = worker.register(local_rank_orch)
        remote_handle = worker.register(RemoteCallable(REMOTE_ORCH_TARGET), workers=[remote_node])
        worker.init()

        def build_and_run(orch, _args, cfg):
            nonlocal domain_handle
            domain = orch.allocate_global_domain(
                name="global-tload-mixed-l3",
                members=((local_node, 0), (remote_node, 0)),
                window_size=WINDOW_SIZE,
                buffers=(
                    CommBufferSpec("input", "float32", COUNT, COUNT * FLOAT_NBYTES),
                    CommBufferSpec("result", "float32", COUNT, COUNT * FLOAT_NBYTES),
                ),
                retain_after_run=True,
            )
            for rank in range(len(RANK_LABELS)):
                orch.copy_to_global_domain(
                    domain,
                    rank,
                    struct.pack(f"<{COUNT}f", *_input_values(rank)),
                    buffer="input",
                )
            local_args = TaskArgs()
            local_args.add_scalar(domain.domain_id)
            local_args.add_scalar(0)
            remote_args = TaskArgs()
            remote_args.add_scalar(domain.domain_id)
            remote_args.add_scalar(0)
            _add_digest_scalars(remote_args, chip_handle.digest)
            parent_keepalive[:] = [local_args, remote_args]
            orch.submit_next_level(local_handle, local_args, cfg, worker=local_node)
            orch.submit_next_level(remote_handle, remote_args, cfg, worker=remote_node)
            domain_handle = domain

        worker.run(build_and_run, args=None, config=call_config())
        if domain_handle is None:
            raise RuntimeError("the Global CommDomain was not allocated")
        domain = domain_handle
        observed: list[tuple[float, ...]] = []

        def read_and_release(orch, _args, _cfg):
            try:
                for rank in range(len(RANK_LABELS)):
                    raw = orch.copy_from_global_domain(domain, rank, COUNT * FLOAT_NBYTES, buffer="result")
                    observed.append(tuple(float(value) for value in struct.unpack(f"<{COUNT}f", raw)))
            finally:
                domain.release()

        worker.run(read_and_release, args=None, config=call_config())

        expected = _expected_values(len(RANK_LABELS))
        failed_ranks: list[int] = []
        for rank, result in enumerate(observed):
            max_diff = max(abs(actual - wanted) for actual, wanted in zip(result, expected))
            print(f"[global-tload-mixed-l3] {RANK_LABELS[rank]} rank={rank} max_diff={max_diff:.3e}")
            if max_diff > 1e-3:
                failed_ranks.append(rank)
        if failed_ranks:
            raise AssertionError(f"peer TLOAD golden mismatch on rank(s) {failed_ranks}")

        print("global_tload_mixed_l3 passed")
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
        enable_chip_swimlane=args.enable_chip_swimlane,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    raise SystemExit(main())
