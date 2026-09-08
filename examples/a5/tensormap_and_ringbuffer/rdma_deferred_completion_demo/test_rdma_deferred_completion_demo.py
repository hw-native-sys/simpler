#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real A5 RDMA deferred completion smoke test."""

from __future__ import annotations

import argparse
import os

import pytest
import torch
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    CommBufferSpec,
    CoreCallable,
    DataType,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root
from simpler_setup.torch_interop import make_tensor_arg

HERE = os.path.dirname(os.path.abspath(__file__))
DTYPE_NBYTES = 4
STATUS_WORDS = 8
RDMA_DATA_OFFSET_NBYTES = 64 * 4
CASES = (64, 16384)


def rdma_workspace_enabled() -> bool:
    return os.environ.get("SIMPLER_ENABLE_PTO_RDMA_WORKSPACE", "").strip().upper() in {"1", "ON", "TRUE", "YES", "Y"}


def parse_device_range(spec: str) -> list[int]:
    if "," in spec:
        return [int(x) for x in spec.split(",") if x]
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        return list(range(lo, hi + 1))
    return [int(spec)]


def build_chip_callable(platform: str) -> ChipCallable:
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root()
    include_dirs = kc.get_orchestration_include_dirs(runtime)
    extra_includes = list(include_dirs) + [str(kc.project_root / "src" / "common")]

    children = []
    for func_id, rel, signature in [
        (
            0,
            "kernels/aiv/kernel_rdma_deferred_completion_tget.cpp",
            [ArgDirection.IN, ArgDirection.OUT, ArgDirection.OUT, ArgDirection.IN, ArgDirection.IN, ArgDirection.IN],
        ),
        (
            1,
            "kernels/aiv/kernel_rdma_deferred_completion_tput.cpp",
            [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT, ArgDirection.IN, ArgDirection.IN, ArgDirection.IN],
        ),
        (
            2,
            "kernels/aiv/kernel_rdma_deferred_completion_consumer.cpp",
            [
                ArgDirection.IN,
                ArgDirection.IN,
                ArgDirection.IN,
                ArgDirection.IN,
                ArgDirection.IN,
                ArgDirection.IN,
                ArgDirection.IN,
                ArgDirection.OUT,
                ArgDirection.IN,
                ArgDirection.IN,
            ],
        ),
    ]:
        kernel = kc.compile_incore(
            source_path=os.path.join(HERE, rel),
            core_type="aiv",
            pto_isa_root=pto_isa_root,
            extra_include_dirs=extra_includes,
        )
        if not platform.endswith("sim"):
            kernel = extract_text_section(kernel)
        children.append((func_id, CoreCallable.build(signature=signature, binary=kernel)))

    orch = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "kernels/orchestration/rdma_deferred_completion_orch.cpp"),
        extra_include_dirs=[str(kc.project_root / "src" / "common")],
    )
    return ChipCallable.build(
        signature=[
            ArgDirection.IN,
            ArgDirection.INOUT,
            ArgDirection.INOUT,
            ArgDirection.OUT,
            ArgDirection.IN,
            ArgDirection.IN,
        ],
        func_name="rdma_deferred_completion_orchestration",
        config_name="rdma_deferred_completion_orchestration_config",
        binary=orch,
        children=children,
    )


def _send_pattern(rank: int, count: int) -> torch.Tensor:
    return torch.tensor([float(rank * 100000 + i) for i in range(count)], dtype=torch.float32).share_memory_()


def _zero_float(count: int) -> torch.Tensor:
    return torch.zeros(count, dtype=torch.float32).share_memory_()


def _zero_i32(count: int) -> torch.Tensor:
    return torch.zeros(count, dtype=torch.int32).share_memory_()


def _status_i32(count: int) -> torch.Tensor:
    return torch.full((count,), -1, dtype=torch.int32).share_memory_()


def _make_case_buffers(elem_count: int, nranks: int):
    tget_elems = nranks * elem_count
    tput_elems = (nranks + 1) * elem_count
    return {
        "send_host": [_send_pattern(rank, elem_count) for rank in range(nranks)],
        "tget_zero": [_zero_float(tget_elems) for _ in range(nranks)],
        "tput_zero": [_zero_float(tput_elems) for _ in range(nranks)],
        "status": [_status_i32(STATUS_WORDS) for _ in range(nranks)],
    }


def _run_case_on_worker(worker: Worker, chip_handle, elem_count: int, nranks: int, case_buffers) -> bool:
    if nranks != 2:
        raise ValueError(f"rdma_deferred_completion_demo needs exactly 2 devices, got {nranks}")

    send_nbytes = elem_count * DTYPE_NBYTES
    tget_elems = nranks * elem_count
    tget_nbytes = tget_elems * DTYPE_NBYTES
    tput_elems = (nranks + 1) * elem_count
    tput_nbytes = tput_elems * DTYPE_NBYTES
    window_size = max(RDMA_DATA_OFFSET_NBYTES + send_nbytes + tget_nbytes + tput_nbytes, 4 * 1024 * 1024)

    send_host = case_buffers["send_host"]
    tget_zero = case_buffers["tget_zero"]
    tput_zero = case_buffers["tput_zero"]
    status = case_buffers["status"]

    # In L3 distributed mode, host buffers for copy_to must be Worker Buffer handles
    # backed by POSIX shared memory (worker.create_buffer).
    send_buf = [worker.create_buffer(send_nbytes) for _ in range(nranks)]
    tget_buf = [worker.create_buffer(tget_nbytes) for _ in range(nranks)]
    tput_buf = [worker.create_buffer(tput_nbytes) for _ in range(nranks)]
    for r in range(nranks):
        s_shm, tg_shm, tp_shm = send_buf[r].shm, tget_buf[r].shm, tput_buf[r].shm
        assert s_shm is not None and s_shm.buf is not None
        assert tg_shm is not None and tg_shm.buf is not None
        assert tp_shm is not None and tp_shm.buf is not None
        torch.frombuffer(s_shm.buf, dtype=torch.float32, count=elem_count).copy_(send_host[r])
        torch.frombuffer(tg_shm.buf, dtype=torch.float32, count=tget_elems).copy_(tget_zero[r])
        torch.frombuffer(tp_shm.buf, dtype=torch.float32, count=tput_elems).copy_(tput_zero[r])

    def orch_fn(orch, _args, cfg):
        with orch.allocate_domain(
            name=f"rdma_deferred_completion_{elem_count}",
            workers=list(range(nranks)),
            window_size=window_size,
            buffers=[
                CommBufferSpec(
                    name="rdma_reserved",
                    dtype="int32",
                    count=RDMA_DATA_OFFSET_NBYTES // 4,
                    nbytes=RDMA_DATA_OFFSET_NBYTES,
                ),
                CommBufferSpec(name="send", dtype="float32", count=elem_count, nbytes=send_nbytes),
                CommBufferSpec(name="tget_recv", dtype="float32", count=tget_elems, nbytes=tget_nbytes),
                CommBufferSpec(name="tput_recv", dtype="float32", count=tput_elems, nbytes=tput_nbytes),
            ],
        ) as handle:
            for rank in range(nranks):
                domain = handle[rank]
                orch.copy_to(domain.buffers["send"], send_buf[rank])
                orch.copy_to(domain.buffers["tget_recv"], tget_buf[rank])
                orch.copy_to(domain.buffers["tput_recv"], tput_buf[rank])

            for rank in range(nranks):
                domain = handle[rank]
                args = TaskArgs()
                args.add_tensor(
                    domain.buffers["send"].tensor((elem_count,), DataType.FLOAT32),
                    TensorArgType.INPUT,
                )
                args.add_tensor(
                    domain.buffers["tget_recv"].tensor((tget_elems,), DataType.FLOAT32),
                    TensorArgType.INOUT,
                )
                args.add_tensor(
                    domain.buffers["tput_recv"].tensor((tput_elems,), DataType.FLOAT32),
                    TensorArgType.INOUT,
                )
                args.add_tensor(make_tensor_arg(worker, status[rank]), TensorArgType.OUTPUT_EXISTING)
                args.add_scalar(domain.device_ctx)
                args.add_scalar(elem_count)
                orch.submit_next_level(chip_handle, args, cfg, worker=rank)

    try:
        worker.run(orch_fn, args=None, config=CallConfig())
    except RuntimeError:
        _print_status(elem_count, status)
        raise
    finally:
        for r in range(nranks):
            worker.release_buffer(send_buf[r])
            worker.release_buffer(tget_buf[r])
            worker.release_buffer(tput_buf[r])

    return _print_status(elem_count, status)


def run_case(platform: str, device_ids: list[int], elem_count: int, *, build: bool = False) -> bool:
    if platform != "a5":
        raise ValueError("rdma_deferred_completion_demo requires platform 'a5'; a5sim cannot validate real RDMA")
    if len(device_ids) != 2:
        raise ValueError(f"rdma_deferred_completion_demo needs exactly 2 devices, got {device_ids}")

    case_buffers = _make_case_buffers(elem_count, len(device_ids))
    chip_callable = build_chip_callable(platform)
    worker = Worker(
        level=3,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=device_ids,
        num_sub_workers=0,
        build=build,
    )
    chip_handle = worker.register(chip_callable)
    try:
        worker.init()
        return _run_case_on_worker(worker, chip_handle, elem_count, len(device_ids), case_buffers)
    finally:
        worker.close()


def _print_status(elem_count: int, status: list[torch.Tensor]) -> bool:
    ok = True
    for rank, rank_status in enumerate(status):
        words = [int(x) for x in rank_status.tolist()]
        print(f"[rdma_deferred_completion_demo] count={elem_count} rank={rank} status={words}")
        ok = ok and words[0] == 0 and words[1] == elem_count
    return ok


def run(platform: str = "a5", device_ids: list[int] | None = None, *, build: bool = False, repeat: int = 1) -> int:
    if device_ids is None:
        device_ids = [0, 1]
    if platform != "a5":
        raise ValueError("rdma_deferred_completion_demo requires platform 'a5'; a5sim cannot validate real RDMA")
    if len(device_ids) != 2:
        raise ValueError(f"rdma_deferred_completion_demo needs exactly 2 devices, got {device_ids}")
    if repeat < 1:
        raise ValueError(f"rdma_deferred_completion_demo repeat must be >= 1, got {repeat}")
    ok = True
    for iteration in range(repeat):
        if repeat > 1:
            print(f"[rdma_deferred_completion_demo] iteration={iteration + 1}/{repeat}")
        for elem_count in CASES:
            ok = run_case(platform, device_ids, elem_count, build=build) and ok
    return 0 if ok else 1


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a5"])
@pytest.mark.device_count(2)
def test_rdma_deferred_completion_demo(st_platform, st_device_ids) -> None:
    if not rdma_workspace_enabled():
        pytest.skip("rdma_deferred_completion_demo requires SIMPLER_ENABLE_PTO_RDMA_WORKSPACE=ON")
    assert run(st_platform, [int(st_device_ids[0]), int(st_device_ids[1])]) == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a5")
    parser.add_argument("-d", "--device", default="0-1")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    return run(args.platform, parse_device_range(args.device), build=args.build, repeat=args.repeat)


if __name__ == "__main__":
    raise SystemExit(main())
