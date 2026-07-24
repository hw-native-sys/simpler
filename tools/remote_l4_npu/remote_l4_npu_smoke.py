#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run L4 -> two remote L3 workers, each executing a two-NPU vector group."""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
from typing import Any

from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    CoreCallable,
    DataType,
    RemoteBufferHandle,
    RemoteTensorRef,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import RemoteCallable, RemoteWorkerSpec, Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

REMOTE_ORCH_TARGET = "tools.remote_l4_npu.remote_l4_npu_smoke:remote_l3_group_orch"
ELEMENTS = 128 * 128
FLOAT_NBYTES = ctypes.sizeof(ctypes.c_float)
TENSOR_COUNT = 6
FloatArray = ctypes.c_float * ELEMENTS
_REMOTE_GROUP_KEEPALIVE: list[TaskArgs] = []


def _digest_from_scalars(args: TaskArgs) -> bytes:
    return b"".join(int(args.scalar(index)).to_bytes(8, "little") for index in range(4))


def remote_l3_group_orch(orch, args: TaskArgs, cfg: CallConfig) -> None:
    """Rebuild the two local chip tasks and submit them as one L3 group."""
    from simpler.remote_l3_session import get_inner_handle  # noqa: PLC0415

    if args.tensor_count() != TENSOR_COUNT or args.scalar_count() != 4:
        raise ValueError("remote L3 group task expects six tensors and four digest scalars")
    chip_handle = get_inner_handle(_digest_from_scalars(args).hex())

    chip_args0 = TaskArgs()
    chip_args0.add_tensor(args.tensor(0), TensorArgType.INPUT)
    chip_args0.add_tensor(args.tensor(1), TensorArgType.INPUT)
    chip_args0.add_tensor(args.tensor(2), TensorArgType.OUTPUT_EXISTING)

    chip_args1 = TaskArgs()
    chip_args1.add_tensor(args.tensor(3), TensorArgType.INPUT)
    chip_args1.add_tensor(args.tensor(4), TensorArgType.INPUT)
    chip_args1.add_tensor(args.tensor(5), TensorArgType.OUTPUT_EXISTING)

    # The inner Worker drains after this callback returns, so retain the rebuilt
    # TaskArgs until that drain has completed.
    _REMOTE_GROUP_KEEPALIVE[:] = [chip_args0, chip_args1]
    orch.submit_next_level_group(chip_handle, [chip_args0, chip_args1], cfg, workers=[0, 1])


def _build_vector_chip_callable(platform: str, runtime: str) -> ChipCallable:
    root = Path(__file__).resolve().parents[2]
    kernels = root / "examples" / "a2a3" / "tensormap_and_ringbuffer" / "vector_example" / "kernels"
    orch_source = kernels / "orchestration" / "example_orchestration.cpp"
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


def _parse_device_ids(value: str) -> tuple[int, int]:
    device_ids = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(device_ids) != 2:
        raise ValueError("each remote L3 group requires exactly two device ids")
    if any(device_id < 0 for device_id in device_ids) or len(set(device_ids)) != 2:
        raise ValueError("device ids must be distinct and non-negative")
    return device_ids


def _make_array(value: float) -> Any:
    array = FloatArray()
    for index in range(ELEMENTS):
        array[index] = value
    return array


def _expected(lhs: float, rhs: float) -> float:
    summed = lhs + rhs
    return (summed + 1.0) * (summed + 2.0) + summed


def _make_remote_group_args(handles: list[RemoteBufferHandle], digest: bytes) -> TaskArgs:
    if len(handles) != TENSOR_COUNT:
        raise ValueError("remote L3 group requires six remote buffers")
    args = TaskArgs()
    for index, handle in enumerate(handles):
        tag = TensorArgType.OUTPUT_EXISTING if index in (2, 5) else TensorArgType.INPUT
        args.add_tensor(RemoteTensorRef(handle, shape=(ELEMENTS,), dtype=DataType.FLOAT32), tag)
    _add_digest_scalars(args, digest)
    return args


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine-a", required=True, help="machine A daemon endpoint, HOST:PORT")
    parser.add_argument("--machine-b", required=True, help="machine B daemon endpoint, HOST:PORT")
    parser.add_argument("--machine-a-devices", default="0,1")
    parser.add_argument("--machine-b-devices", default="0,1")
    parser.add_argument("--platform", default="a2a3")
    parser.add_argument("--runtime", default="tensormap_and_ringbuffer")
    parser.add_argument("--session-timeout", type=float, default=120.0)
    parser.add_argument("--session-listen-host", default="0.0.0.0")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    machine_a_devices = _parse_device_ids(args.machine_a_devices)
    machine_b_devices = _parse_device_ids(args.machine_b_devices)

    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=args.session_timeout)
    remote_buffers: list[RemoteBufferHandle] = []
    parent_keepalive: list[TaskArgs] = []
    try:
        worker_a = worker.add_remote_worker(
            RemoteWorkerSpec(
                endpoint=args.machine_a,
                platform=args.platform,
                runtime=args.runtime,
                device_ids=machine_a_devices,
                transport="sim",
                session_listen_host=args.session_listen_host,
                allow_wildcard_session_bind=True,
            )
        )
        worker_b = worker.add_remote_worker(
            RemoteWorkerSpec(
                endpoint=args.machine_b,
                platform=args.platform,
                runtime=args.runtime,
                device_ids=machine_b_devices,
                transport="sim",
                session_listen_host=args.session_listen_host,
                allow_wildcard_session_bind=True,
            )
        )
        chip_handle = worker.register(_build_vector_chip_callable(args.platform, args.runtime))
        remote_handle = worker.register(RemoteCallable(REMOTE_ORCH_TARGET), workers=[worker_a, worker_b])
        worker.init()

        tensor_nbytes = ELEMENTS * FLOAT_NBYTES
        group_values = {
            worker_a: (2.0, 3.0, 4.0, 5.0),
            worker_b: (6.0, 7.0, 8.0, 9.0),
        }
        group_handles: dict[int, list[RemoteBufferHandle]] = {}
        output_arrays: dict[int, dict[str, tuple[Any, float]]] = {}
        for worker_id, (a0_value, b0_value, a1_value, b1_value) in group_values.items():
            handles = [worker.remote_malloc(worker=worker_id, nbytes=tensor_nbytes) for _ in range(TENSOR_COUNT)]
            remote_buffers.extend(handles)
            group_handles[worker_id] = handles
            initial_arrays = (
                _make_array(a0_value),
                _make_array(b0_value),
                _make_array(0.0),
                _make_array(a1_value),
                _make_array(b1_value),
                _make_array(0.0),
            )
            for handle, array in zip(handles, initial_arrays, strict=True):
                worker.remote_copy_to(handle, array, tensor_nbytes)
            output_arrays[worker_id] = {
                "f0": (_make_array(0.0), _expected(a0_value, b0_value)),
                "f1": (_make_array(0.0), _expected(a1_value, b1_value)),
            }

        def parent_orch(orch, _args, cfg):
            args_a = _make_remote_group_args(group_handles[worker_a], chip_handle.digest)
            args_b = _make_remote_group_args(group_handles[worker_b], chip_handle.digest)
            parent_keepalive[:] = [args_a, args_b]
            orch.submit_next_level(remote_handle, args_a, cfg, worker=worker_a)
            orch.submit_next_level(remote_handle, args_b, cfg, worker=worker_b)

        config = CallConfig()
        config.block_dim = 3
        config.aicpu_thread_num = 4
        worker.run(parent_orch, args=None, config=config)

        for worker_id, handles in group_handles.items():
            worker.remote_copy_from(handles[2], output_arrays[worker_id]["f0"][0], tensor_nbytes)
            worker.remote_copy_from(handles[5], output_arrays[worker_id]["f1"][0], tensor_nbytes)

        for worker_id, outputs in output_arrays.items():
            for name, (array, expected) in outputs.items():
                max_diff = max(abs(float(array[index]) - expected) for index in range(ELEMENTS))
                print(f"[remote-l4-group] worker={worker_id} output={name} max_diff={max_diff:.3e}")
                if max_diff > 1e-4:
                    raise AssertionError(f"worker {worker_id} {name} golden mismatch: max_diff={max_diff}")

        print(
            "remote L4 L3-group NPU smoke passed: "
            f"{args.machine_a}[devices={args.machine_a_devices}], "
            f"{args.machine_b}[devices={args.machine_b_devices}], "
            f"elements={ELEMENTS}"
        )
        return 0
    finally:
        parent_keepalive.clear()
        for handle in reversed(remote_buffers):
            try:
                worker.remote_free(handle)
            except Exception:  # noqa: BLE001
                pass
        worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
