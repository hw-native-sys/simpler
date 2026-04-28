#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Notification counter + deferred completion smoke test for onboard a5."""

from __future__ import annotations

import argparse
import os
import tempfile

import torch
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipBootstrapConfig,
    ChipBufferSpec,
    ChipCallable,
    ChipCommBootstrapConfig,
    ChipContext,
    ContinuousTensor,
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
N = 128 * 128


def parse_device_range(spec: str) -> list[int]:
    if "," in spec:
        return [int(x) for x in spec.split(",") if x]
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        return list(range(lo, hi + 1))
    return [int(spec)]


def build_chip_callable(platform: str, pto_isa_commit: str | None, clone_protocol: str) -> ChipCallable:
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root(commit=pto_isa_commit, clone_protocol=clone_protocol)
    include_dirs = kc.get_orchestration_include_dirs(runtime)
    extra_includes = list(include_dirs) + [str(kc.project_root / "src" / "common")]

    children = []
    for func_id, rel in [
        (0, "kernels/aiv/kernel_producer_notify.cpp"),
        (1, "kernels/aiv/kernel_consumer.cpp"),
        (2, "kernels/aiv/kernel_notify_wait.cpp"),
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
                    signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.OUT, ArgDirection.IN],
                    binary=kernel,
                ),
            )
        )

    orch = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "kernels/orchestration/async_notify_orchestration.cpp"),
        extra_include_dirs=[str(kc.project_root / "src" / "common")],
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.OUT, ArgDirection.IN],
        func_name="async_notify_orchestration",
        binary=orch,
        children=children,
    )


def run(platform: str = "a5", device_ids: list[int] | None = None, pto_isa_commit: str | None = None) -> int:
    if device_ids is None:
        device_ids = [0, 1]
    nranks = len(device_ids)
    if nranks != 2:
        raise ValueError(f"async_notify_demo needs exactly 2 devices, got {device_ids}")

    rootinfo_path = os.path.join(tempfile.gettempdir(), f"pto_async_notify_rootinfo_{os.getpid()}.bin")
    try:
        os.unlink(rootinfo_path)
    except FileNotFoundError:
        pass

    inp = [
        torch.tensor([float(i % 251) / 10.0 for i in range(N)], dtype=torch.float32).share_memory_()
        for _ in range(nranks)
    ]
    out = [torch.zeros(N, dtype=torch.float32).share_memory_() for _ in range(nranks)]
    result = [torch.zeros(N, dtype=torch.float32).share_memory_() for _ in range(nranks)]

    cfgs = [
        ChipBootstrapConfig(
            comm=ChipCommBootstrapConfig(rank=rank, nranks=nranks, rootinfo_path=rootinfo_path, window_size=4 * 1024),
            buffers=[ChipBufferSpec(name="notify_counter", dtype="int32", count=1, nbytes=4)],
        )
        for rank in range(nranks)
    ]

    chip_callable = build_chip_callable(platform, pto_isa_commit, "https")
    worker = Worker(
        level=3,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=device_ids,
        num_sub_workers=0,
        chip_bootstrap_configs=cfgs,
    )
    try:
        worker.init()
        contexts: list[ChipContext] = worker.chip_contexts

        def orch_fn(orch, _args, cfg):
            for rank, ctx in enumerate(contexts):
                args = TaskArgs()
                args.add_tensor(make_tensor_arg(inp[rank]), TensorArgType.INPUT)
                args.add_tensor(make_tensor_arg(out[rank]), TensorArgType.OUTPUT_EXISTING)
                args.add_tensor(make_tensor_arg(result[rank]), TensorArgType.OUTPUT_EXISTING)
                args.add_tensor(
                    ContinuousTensor.make(
                        data=ctx.buffer_ptrs["notify_counter"],
                        shapes=(1,),
                        dtype=DataType.INT32,
                        child_memory=True,
                    ),
                    TensorArgType.INPUT,
                )
                args.add_scalar(ctx.device_ctx)
                orch.submit_next_level(chip_callable, args, cfg, worker=rank)

        worker.run(orch_fn, args=None, config=CallConfig())

        ok = True
        for rank in range(nranks):
            expected_out = inp[rank] * 2.0
            expected_result = expected_out + 1.0
            max_out = float(torch.max(torch.abs(out[rank] - expected_out)))
            max_result = float(torch.max(torch.abs(result[rank] - expected_result)))
            print(f"[async_notify_demo] rank {rank}: max_out={max_out:.3e} max_result={max_result:.3e}")
            ok = ok and max_out <= 1e-3 and max_result <= 1e-3
        return 0 if ok else 1
    finally:
        worker.close()
        try:
            os.unlink(rootinfo_path)
        except FileNotFoundError:
            pass


def test_async_notify_demo() -> None:
    assert run("a5", [0, 1]) == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a5")
    parser.add_argument("-d", "--device", default="0-1")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--pto-isa-commit", default=None)
    args = parser.parse_args()
    return run(args.platform, parse_device_range(args.device), args.pto_isa_commit)


if __name__ == "__main__":
    raise SystemExit(main())
