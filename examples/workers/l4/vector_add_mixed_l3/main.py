#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run L4 -> one local L3 and one remote L3, each executing a two-NPU vector group."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
from pathlib import Path
from typing import Any

from simpler.callable_identity import CallableHandle
from simpler.remote_l3_protocol import HOST_TCP_TRANSPORT_PROFILE
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    CoreCallable,
    DataType,
    RemoteBufferHandle,
    RemoteTensorRef,
    TaskArgs,
    Tensor,
    TensorArgType,
)
from simpler.worker import RemoteCallable, RemoteWorkerSpec, Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

REMOTE_ORCH_TARGET = "examples.workers.l4.vector_add_mixed_l3.main:remote_l3_group_orch"
ELEMENTS = 128 * 128
FLOAT_NBYTES = ctypes.sizeof(ctypes.c_float)
TENSOR_NBYTES = ELEMENTS * FLOAT_NBYTES
TENSOR_COUNT = 6
FloatArray = ctypes.c_float * ELEMENTS
_LOCAL_GROUP_KEEPALIVE: list[TaskArgs] = []
_LOCAL_CHIP_HANDLE: CallableHandle | None = None
_REMOTE_GROUP_KEEPALIVE: list[TaskArgs] = []


def _digest_from_scalars(args: TaskArgs) -> bytes:
    return b"".join(int(args.scalar(index)).to_bytes(8, "little") for index in range(4))


def _submit_two_chip_group(orch, chip_handle: CallableHandle, args: TaskArgs, cfg: CallConfig) -> list[TaskArgs]:
    if args.tensor_count() != TENSOR_COUNT or args.scalar_count() != 4:
        raise ValueError("vector_add_mixed_l3 group task expects six tensors and four digest scalars")

    chip_args0 = TaskArgs()
    chip_args0.add_tensor(args.tensor(0), TensorArgType.INPUT)
    chip_args0.add_tensor(args.tensor(1), TensorArgType.INPUT)
    chip_args0.add_tensor(args.tensor(2), TensorArgType.OUTPUT_EXISTING)

    chip_args1 = TaskArgs()
    chip_args1.add_tensor(args.tensor(3), TensorArgType.INPUT)
    chip_args1.add_tensor(args.tensor(4), TensorArgType.INPUT)
    chip_args1.add_tensor(args.tensor(5), TensorArgType.OUTPUT_EXISTING)

    group_args = [chip_args0, chip_args1]
    orch.submit_next_level_group(chip_handle, group_args, cfg, workers=[0, 1])
    return group_args


def local_l3_group_orch(orch, args: TaskArgs, cfg: CallConfig) -> None:
    """Submit two local chip tasks from the forked local L3 worker."""
    if _LOCAL_CHIP_HANDLE is None:
        raise RuntimeError("local L3 chip handle was not installed before fork")
    _LOCAL_GROUP_KEEPALIVE[:] = _submit_two_chip_group(orch, _LOCAL_CHIP_HANDLE, args, cfg)


def remote_l3_group_orch(orch, args: TaskArgs, cfg: CallConfig) -> None:
    """Submit two local chip tasks from the daemon-started remote L3 worker."""
    from simpler.remote_l3_session import get_inner_handle  # noqa: PLC0415

    chip_handle = get_inner_handle(_digest_from_scalars(args).hex())
    _REMOTE_GROUP_KEEPALIVE[:] = _submit_two_chip_group(orch, chip_handle, args, cfg)


def _build_vector_chip_callable(platform: str, runtime: str) -> ChipCallable:
    kernels = Path(__file__).resolve().parent / "kernels"
    orch_source = kernels / "orchestration" / "vector_add_mixed_l3_orchestration.cpp"
    aiv_sources = (
        kernels / "aiv" / "kernel_add.cpp",
        kernels / "aiv" / "kernel_add_scalar.cpp",
        kernels / "aiv" / "kernel_mul.cpp",
    )

    compiler = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    include_dirs = compiler.get_orchestration_include_dirs(runtime)
    include_dirs = list(include_dirs) + [str(compiler.project_root / "src" / "common")]

    def compile_aiv(source: Path) -> bytes:
        binary = compiler.compile_incore(
            source_path=str(source),
            core_type="aiv",
            pto_isa_root=pto_isa_root,
            extra_include_dirs=include_dirs,
        )
        return binary if platform.endswith("sim") else extract_text_section(binary)

    children = (
        (
            0,
            CoreCallable.build(
                signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
                binary=compile_aiv(aiv_sources[0]),
            ),
        ),
        (
            1,
            CoreCallable.build(
                signature=[ArgDirection.IN, ArgDirection.OUT],
                binary=compile_aiv(aiv_sources[1]),
            ),
        ),
        (
            2,
            CoreCallable.build(
                signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
                binary=compile_aiv(aiv_sources[2]),
            ),
        ),
    )
    orch_binary = compiler.compile_orchestration(runtime_name=runtime, source_path=str(orch_source))
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
        func_name="aicpu_orchestration_entry",
        config_name="aicpu_orchestration_config",
        binary=orch_binary,
        children=list(children),
    )


def _add_digest_scalars(task_args: TaskArgs, digest: bytes) -> None:
    if len(digest) != 32:
        raise ValueError("inner chip callable digest must be 32 bytes")
    for offset in range(0, 32, 8):
        task_args.add_scalar(int.from_bytes(digest[offset : offset + 8], "little"))


def _parse_device_ids(value: str, *, label: str) -> tuple[int, int]:
    device_ids = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(device_ids) != 2:
        raise ValueError(f"{label} L3 group requires exactly two device ids")
    if any(device_id < 0 for device_id in device_ids) or len(set(device_ids)) != 2:
        raise ValueError(f"{label} device ids must be distinct and non-negative")
    return device_ids


def _fill_array(array: Any, value: float) -> None:
    for index in range(ELEMENTS):
        array[index] = value


def _make_array(value: float) -> Any:
    array = FloatArray()
    _fill_array(array, value)
    return array


def _expected(lhs: float, rhs: float) -> float:
    summed = lhs + rhs
    return (summed + 1.0) * (summed + 2.0) + summed


def _make_local_group(
    worker: Worker,
    values: tuple[float, float, float, float],
) -> tuple[list[Any], TaskArgs, dict[str, tuple[Any, float]]]:
    """Six owner Buffers on ``worker`` plus the wire ``TaskArgs`` naming them.

    ``worker`` must be initialized: ``create_buffer`` needs a forked child to reach the backing, and
    the local L3 that consumes these tensors is that child. The views alias the buffers' shm, so every
    one of them must be dropped before ``worker.close()`` releases the backings.
    """
    a0_value, b0_value, a1_value, b1_value = values
    initial_values = (a0_value, b0_value, 0.0, a1_value, b1_value, 0.0)
    views: list[Any] = []
    args = TaskArgs()
    for index, value in enumerate(initial_values):
        handle = worker.create_buffer(TENSOR_NBYTES)
        shm = handle.shm
        assert shm is not None
        buf = shm.buf
        assert buf is not None
        view = FloatArray.from_buffer(buf)
        _fill_array(view, value)
        views.append(view)
        tag = TensorArgType.OUTPUT_EXISTING if index in (2, 5) else TensorArgType.INPUT
        args.add_tensor(Tensor(handle, shapes=(ELEMENTS,), dtype=DataType.FLOAT32), tag)
    return (
        views,
        args,
        {
            "f0": (views[2], _expected(a0_value, b0_value)),
            "f1": (views[5], _expected(a1_value, b1_value)),
        },
    )


def _make_remote_group_args(handles: list[RemoteBufferHandle], digest: bytes) -> TaskArgs:
    if len(handles) != TENSOR_COUNT:
        raise ValueError("remote L3 group requires six remote buffers")
    args = TaskArgs()
    for index, handle in enumerate(handles):
        tag = TensorArgType.OUTPUT_EXISTING if index in (2, 5) else TensorArgType.INPUT
        args.add_tensor(RemoteTensorRef(handle, shape=(ELEMENTS,), dtype=DataType.FLOAT32), tag)
    _add_digest_scalars(args, digest)
    return args


def _check_outputs(worker_label: str, output_map: dict[str, tuple[Any, float]]) -> None:
    """Compare each output against its golden value.

    The local outputs alias owner Buffers, so the iteration that binds them lives in its own frame:
    a name still holding one when ``worker.close()`` runs blocks the shm release.
    """
    for name, (output_array, expected) in output_map.items():
        max_diff = max(abs(float(output_array[index]) - expected) for index in range(ELEMENTS))
        print(f"[vector-add-mixed-l3] {worker_label} output={name} max_diff={max_diff:.3e}")
        if max_diff > 1e-4:
            raise AssertionError(f"{worker_label} {name} golden mismatch: max_diff={max_diff}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", required=True, help="remote L3 daemon endpoint, HOST:PORT")
    parser.add_argument("--local-devices", default="0,1", help="two local device ids owned by the forked local L3")
    parser.add_argument("--remote-devices", default="0,1", help="two remote device ids owned by the daemon L3")
    parser.add_argument("--platform", default="a2a3")
    parser.add_argument("--runtime", default="tensormap_and_ringbuffer")
    parser.add_argument("--session-timeout", type=float, default=120.0)
    parser.add_argument("--session-listen-host", default="0.0.0.0")
    return parser.parse_args()


def run(
    *,
    remote: str,
    local_devices: str,
    remote_devices: str,
    platform: str = "a2a3",
    runtime: str = "tensormap_and_ringbuffer",
    session_timeout: float = 120.0,
    session_listen_host: str = "0.0.0.0",  # noqa: S104 - Remote peer callbacks need a reachable listener.
) -> int:
    # The local L3 is a fork of this process, so its orchestration function
    # reaches the handle only through module state; a local would not survive
    # into the child.
    global _LOCAL_CHIP_HANDLE  # noqa: PLW0603

    local_device_ids = _parse_device_ids(local_devices, label="local")
    remote_device_ids = _parse_device_ids(remote_devices, label="remote")

    local_l3: Worker | None = None
    local_l3_attached = False
    worker: Worker | None = None
    remote_buffers: list[RemoteBufferHandle] = []
    local_views: list[Any] = []
    local_outputs: dict[str, tuple[Any, float]] = {}
    parent_keepalive: list[TaskArgs] = []
    try:
        chip_callable = _build_vector_chip_callable(platform, runtime)
        local_l3 = Worker(level=3, platform=platform, runtime=runtime, device_ids=local_device_ids)
        _LOCAL_CHIP_HANDLE = local_l3.register(chip_callable)

        worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=session_timeout)
        local_worker = worker.add_worker(local_l3)
        local_l3_attached = True
        remote_worker = worker.add_remote_worker(
            RemoteWorkerSpec(
                endpoint=remote,
                platform=platform,
                runtime=runtime,
                device_ids=remote_device_ids,
                transport=HOST_TCP_TRANSPORT_PROFILE,
                session_listen_host=session_listen_host,
                allow_wildcard_session_bind=True,
            )
        )
        chip_handle = worker.register(chip_callable)
        local_handle = worker.register(local_l3_group_orch)
        remote_handle = worker.register(RemoteCallable(REMOTE_ORCH_TARGET), workers=[remote_worker])

        worker.init()

        local_views, local_args, local_outputs = _make_local_group(worker, (2.0, 3.0, 4.0, 5.0))
        _add_digest_scalars(local_args, chip_handle.digest)

        remote_handles = [worker.remote_malloc(worker=remote_worker, nbytes=TENSOR_NBYTES) for _ in range(TENSOR_COUNT)]
        remote_buffers.extend(remote_handles)
        remote_inputs = (
            _make_array(6.0),
            _make_array(7.0),
            _make_array(0.0),
            _make_array(8.0),
            _make_array(9.0),
            _make_array(0.0),
        )
        # zip() truncates to the shorter side, which would leave a remote input
        # silently uninitialised if TENSOR_COUNT and this tuple ever disagree.
        # zip(strict=True) says the same thing in one word but needs 3.10, and
        # the network1 runners are on 3.9.
        if len(remote_inputs) != TENSOR_COUNT:
            raise AssertionError(f"expected {TENSOR_COUNT} remote inputs, got {len(remote_inputs)}")
        for handle, array in zip(remote_handles, remote_inputs):
            worker.remote_copy_to(handle, array, TENSOR_NBYTES)
        remote_outputs = {
            "f0": (_make_array(0.0), _expected(6.0, 7.0)),
            "f1": (_make_array(0.0), _expected(8.0, 9.0)),
        }

        def parent_orch(orch, _args, cfg):
            remote_args = _make_remote_group_args(remote_handles, chip_handle.digest)
            parent_keepalive[:] = [local_args, remote_args]
            orch.submit_next_level(local_handle, local_args, cfg, worker=local_worker)
            orch.submit_next_level(remote_handle, remote_args, cfg, worker=remote_worker)

        config = CallConfig()
        worker.run(parent_orch, args=None, config=config)

        worker.remote_copy_from(remote_handles[2], remote_outputs["f0"][0], TENSOR_NBYTES)
        worker.remote_copy_from(remote_handles[5], remote_outputs["f1"][0], TENSOR_NBYTES)

        _check_outputs("local", local_outputs)
        _check_outputs("remote", remote_outputs)

        print(
            "vector_add_mixed_l3 passed: "
            f"local[devices={local_devices}], remote={remote}[devices={remote_devices}], "
            f"elements={ELEMENTS}"
        )
        return 0
    finally:
        parent_keepalive.clear()
        _LOCAL_GROUP_KEEPALIVE.clear()
        _REMOTE_GROUP_KEEPALIVE.clear()
        if worker is not None:
            for handle in reversed(remote_buffers):
                try:
                    worker.remote_free(handle)
                except Exception as exc:  # noqa: BLE001
                    # The network1 job diagnoses this example from stdout alone, so a
                    # leaked peer buffer has to name itself here or leave no trace.
                    print(f"[vector-add-mixed-l3] remote_free failed: {exc}")
        # close() unlinks the owner Buffers, which fails while any view still aliases their shm.
        local_outputs.clear()
        local_views.clear()
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
        session_timeout=args.session_timeout,
        session_listen_host=args.session_listen_host,
    )


if __name__ == "__main__":
    raise SystemExit(main())
