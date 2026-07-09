#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""URMA-shaped deferred completion smoke test for a5sim."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for path in (str(REPO_ROOT), str(REPO_ROOT / "python")):
    if path not in sys.path:
        sys.path.insert(0, path)
sys.meta_path = [finder for finder in sys.meta_path if type(finder).__module__ != "_simpler_editable"]

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
N = 1024
FAKE_WORKSPACE_NBYTES = 128 * 1024


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
    for func_id, rel in [
        (0, "kernels/aiv/kernel_urma_tget.cpp"),
        (1, "kernels/aiv/kernel_urma_tput.cpp"),
        (2, "kernels/aiv/kernel_urma_complete.cpp"),
        (3, "kernels/aiv/kernel_urma_consumer.cpp"),
        (4, "kernels/aiv/kernel_urma_reset.cpp"),
    ]:
        kernel = kc.compile_incore(
            source_path=os.path.join(HERE, rel),
            core_type="aiv",
            pto_isa_root=pto_isa_root,
            extra_include_dirs=extra_includes,
        )
        if not platform.endswith("sim"):
            kernel = extract_text_section(kernel)
        children.append(
            (
                func_id,
                CoreCallable.build(
                    signature=[
                        ArgDirection.IN,
                        ArgDirection.INOUT,
                        ArgDirection.OUT,
                        ArgDirection.OUT,
                        ArgDirection.IN,
                    ],
                    binary=kernel,
                ),
            )
        )

    orch = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "kernels/orchestration/urma_async_completion_orch.cpp"),
        extra_include_dirs=[str(kc.project_root / "src" / "common")],
    )
    return ChipCallable.build(
        signature=[
            ArgDirection.IN,
            ArgDirection.INOUT,
            ArgDirection.OUT,
            ArgDirection.OUT,
            ArgDirection.IN,
        ],
        func_name="urma_async_completion_orchestration",
        binary=orch,
        children=children,
    )


def run(platform: str = "a5sim", device_ids: list[int] | None = None) -> int:
    if device_ids is None:
        device_ids = [0, 1]
    nranks = len(device_ids)
    if nranks != 2:
        raise ValueError(f"urma_async_completion_demo needs exactly 2 devices, got {device_ids}")

    src = [
        torch.tensor([float(rank * 1000 + i) for i in range(N)], dtype=torch.float32).share_memory_()
        for rank in range(nranks)
    ]
    scratch = [torch.zeros(N, dtype=torch.float32).share_memory_() for _ in range(nranks)]
    tget_result = [torch.zeros(N, dtype=torch.float32).share_memory_() for _ in range(nranks)]
    tput_result = [torch.zeros(N, dtype=torch.float32).share_memory_() for _ in range(nranks)]

    chip_callable = build_chip_callable(platform)
    worker = Worker(
        level=3,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=device_ids,
        num_sub_workers=0,
    )
    chip_handle = worker.register(chip_callable)
    try:
        worker.init()

        def orch_fn(orch, _args, cfg):
            with orch.allocate_domain(
                name="default",
                workers=list(range(nranks)),
                window_size=FAKE_WORKSPACE_NBYTES,
                buffers=[
                    CommBufferSpec(
                        name="urma_workspace",
                        dtype="uint8",
                        count=FAKE_WORKSPACE_NBYTES,
                        nbytes=FAKE_WORKSPACE_NBYTES,
                    )
                ],
            ) as handle:
                for rank in range(nranks):
                    domain = handle[rank]
                    args = TaskArgs()
                    args.add_tensor(make_tensor_arg(src[rank]), TensorArgType.INPUT)
                    args.add_tensor(make_tensor_arg(scratch[rank]), TensorArgType.INOUT)
                    args.add_tensor(make_tensor_arg(tget_result[rank]), TensorArgType.OUTPUT_EXISTING)
                    args.add_tensor(make_tensor_arg(tput_result[rank]), TensorArgType.OUTPUT_EXISTING)
                    args.add_tensor(
                        Tensor.make(
                            data=domain.buffer_ptrs["urma_workspace"],
                            shapes=(FAKE_WORKSPACE_NBYTES,),
                            dtype=DataType.UINT8,
                            child_memory=True,
                        ),
                        TensorArgType.INPUT,
                    )
                    args.add_scalar(domain.device_ctx)
                    orch.submit_next_level(chip_handle, args, cfg, worker=rank)

        worker.run(orch_fn, args=None, config=CallConfig())

        ok = True
        for rank in range(nranks):
            max_tget = float(torch.max(torch.abs(tget_result[rank] - src[rank])))
            max_tput = float(torch.max(torch.abs(tput_result[rank] - src[rank])))
            print(f"[urma_async_completion_demo] rank {rank}: max_tget={max_tget:.3e} max_tput={max_tput:.3e}")
            ok = ok and max_tget <= 1e-6 and max_tput <= 1e-6
        return 0 if ok else 1
    finally:
        worker.close()


def test_urma_async_completion_demo() -> None:
    assert run("a5sim", [0, 1]) == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a5sim")
    parser.add_argument("-d", "--device", default="0-1")
    args = parser.parse_args()
    return run(args.platform, parse_device_range(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
