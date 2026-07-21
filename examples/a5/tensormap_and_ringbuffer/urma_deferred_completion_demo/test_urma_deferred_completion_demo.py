#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""URMA deferred completion smoke test for onboard a5.

Each rank stages its input inside the HCCL/URMA communication window. The
producer TGET_ASYNCs the peer rank's input into local ``out`` and registers the
URMA AsyncEvent through the deferred completion path. The consumer depends on
that producer output and writes ``result = out + 1``. Correct ``out`` and
``result`` validate both URMA completion polling and deferred-release dependency
handling.
"""

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
    Tensor,
    TensorArgType,
)
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root
from simpler_setup.torch_interop import make_tensor_arg

HERE = os.path.dirname(os.path.abspath(__file__))
N = 128 * 128
DTYPE_NBYTES = 4
URMA_DATA_OFFSET_NBYTES = 64 * 4
_URMA_WORKSPACE_ENV = "SIMPLER_ENABLE_PTO_URMA_WORKSPACE"
_WORKSPACE_TRUTHY = {"1", "ON", "TRUE", "YES"}


def _urma_workspace_enabled() -> bool:
    return os.environ.get(_URMA_WORKSPACE_ENV, "").upper() in _WORKSPACE_TRUTHY


def _require_urma_workspace_enabled() -> None:
    if _urma_workspace_enabled():
        return
    raise RuntimeError(
        "urma_deferred_completion_demo requires host runtime built with "
        f"{_URMA_WORKSPACE_ENV}=ON; set it before rebuilding simpler."
    )


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
            "kernels/aiv/kernel_urma_tget_async.cpp",
            [ArgDirection.IN, ArgDirection.OUT, ArgDirection.IN],
        ),
        (
            1,
            "kernels/aiv/kernel_consumer.cpp",
            [ArgDirection.IN, ArgDirection.OUT],
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
        source_path=os.path.join(HERE, "kernels/orchestration/urma_deferred_completion_orch.cpp"),
        extra_include_dirs=[str(kc.project_root / "src" / "common")],
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.OUT, ArgDirection.IN],
        func_name="urma_deferred_completion_orchestration",
        config_name="urma_deferred_completion_orchestration_config",
        binary=orch,
        children=children,
    )


def run(platform: str = "a5", device_ids: list[int] | None = None) -> int:
    _require_urma_workspace_enabled()
    if device_ids is None:
        device_ids = [0, 1]
    nranks = len(device_ids)
    if nranks != 2:
        raise ValueError(f"urma_deferred_completion_demo needs exactly 2 devices, got {device_ids}")
    if platform != "a5":
        raise ValueError("urma_deferred_completion_demo requires onboard a5 hardware")

    input_nbytes = N * DTYPE_NBYTES
    window_size = max(URMA_DATA_OFFSET_NBYTES + input_nbytes, 4 * 1024 * 1024)

    # `inputs` must live in shared memory: `orch.copy_to` stages each rank's
    # data into its HCCL window from the forked chip child, which reads `src`
    # out of its own address space.
    inputs = [
        torch.tensor([float(rank * 1000 + (i % 251)) / 10.0 for i in range(N)], dtype=torch.float32).share_memory_()
        for rank in range(nranks)
    ]
    out = [torch.zeros(N, dtype=torch.float32).share_memory_() for _ in range(nranks)]
    result = [torch.zeros(N, dtype=torch.float32).share_memory_() for _ in range(nranks)]

    chip_callable = build_chip_callable(platform)
    worker = Worker(
        level=3,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=device_ids,
        num_sub_workers=0,
    )
    chip_cid = worker.register(chip_callable)
    try:
        worker.init()

        def orch_fn(orch, _args, cfg):
            with orch.allocate_domain(
                name="urma_deferred_completion",
                workers=list(range(nranks)),
                window_size=window_size,
                buffers=[
                    CommBufferSpec(
                        name="urma_reserved",
                        dtype="int32",
                        count=URMA_DATA_OFFSET_NBYTES // 4,
                        nbytes=URMA_DATA_OFFSET_NBYTES,
                    ),
                    CommBufferSpec(name="input_window", dtype="float32", count=N, nbytes=input_nbytes),
                ],
            ) as handle:
                # Stage every rank's input window before submitting any kernel:
                # each producer TGET_ASYNCs the *peer* rank's window, so all
                # windows must hold real data before execution begins.
                for rank in range(nranks):
                    orch.copy_to(
                        rank,
                        dst=handle[rank].buffer_ptrs["input_window"],
                        src=inputs[rank].data_ptr(),
                        size=input_nbytes,
                    )
                for rank in range(nranks):
                    domain = handle[rank]
                    args = TaskArgs()
                    args.add_tensor(
                        Tensor.make(
                            data=domain.buffer_ptrs["input_window"],
                            shapes=(N,),
                            dtype=DataType.FLOAT32,
                            child_memory=True,
                        ),
                        TensorArgType.INPUT,
                    )
                    args.add_tensor(make_tensor_arg(out[rank]), TensorArgType.OUTPUT_EXISTING)
                    args.add_tensor(make_tensor_arg(result[rank]), TensorArgType.OUTPUT_EXISTING)
                    args.add_scalar(domain.device_ctx)
                    orch.submit_next_level(chip_cid, args, cfg, worker=rank)

        worker.run(orch_fn, args=None, config=CallConfig())

        ok = True
        for rank in range(nranks):
            peer = 1 - rank
            expected_out = inputs[peer]
            expected_result = expected_out + 1.0
            max_out = float(torch.max(torch.abs(out[rank] - expected_out)))
            max_result = float(torch.max(torch.abs(result[rank] - expected_result)))
            print(f"[urma_deferred_completion_demo] rank {rank}: max_out={max_out:.3e} max_result={max_result:.3e}")
            ok = ok and max_out <= 1e-3 and max_result <= 1e-3
        return 0 if ok else 1
    finally:
        worker.close()


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(2)
@pytest.mark.skipif(
    not _urma_workspace_enabled(),
    reason="URMA workspace overlay not enabled (set SIMPLER_ENABLE_PTO_URMA_WORKSPACE=ON to run). "
    "See docs/a5-sdma-overlay.md (#1315).",
)
def test_urma_deferred_completion_demo(st_device_ids, st_platform) -> None:
    assert run(st_platform, [int(d) for d in st_device_ids]) == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a5")
    parser.add_argument("-d", "--device", default="0-1")
    args = parser.parse_args()
    return run(args.platform, parse_device_range(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
