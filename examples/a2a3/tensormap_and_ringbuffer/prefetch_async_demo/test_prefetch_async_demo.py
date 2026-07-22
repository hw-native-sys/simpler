#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-card TPREFETCH_ASYNC smoke test for onboard a2a3.

Exercises the runtime-injected SDMA workspace: a Worker created with
``enable_sdma=True`` provisions the PTO-ISA async-SDMA workspace once at init and
injects its address into every kernel's GlobalContext, so the kernel obtains it
via ``get_dma_workspace(args, DMA_WORKSPACE_SDMA)`` -- no workspace is threaded
as a user arg. A Worker without ``enable_sdma`` creates no SDMA streams and its
kernels read a zero workspace address.
The kernel prefetches ``in`` into L2, waits on the returned event, then copies
``in`` to ``out``.

The prefetch is a pure cache hint that changes no value, so ``out == in``
bit-exactly is the property under test -- together with the event wait actually
completing rather than hanging. The device log (``[SDMA] Created 48 STARS
streams OK``) confirms the real SDMA path ran rather than the skip branch.

Unlike the SDMA completion demo this needs no comm domain: the workspace is a
runtime-owned per-device resource, so a single device is enough.
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
    ChipStorageTaskArgs,
    CoreCallable,
)
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root
from simpler_setup.torch_interop import make_tensor_arg

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = "tensormap_and_ringbuffer"
N = 128


def build_chip_callable(platform: str) -> ChipCallable:
    kc = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    include_dirs = kc.get_orchestration_include_dirs(RUNTIME)
    extra_includes = list(include_dirs) + [str(kc.project_root / "src" / "common")]

    # in (IN), out (OUT) -- the SDMA workspace is injected, not an arg.
    signature = [ArgDirection.IN, ArgDirection.OUT]

    kernel = kc.compile_incore(
        source_path=os.path.join(HERE, "kernels/aiv/kernel_prefetch_copy.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=extra_includes,
    )
    if not platform.endswith("sim"):
        kernel = extract_text_section(kernel)
    children = [
        (
            0,
            CoreCallable.build(
                signature=signature,
                binary=kernel,
            ),
        )
    ]

    orch = kc.compile_orchestration(
        runtime_name=RUNTIME,
        source_path=os.path.join(HERE, "kernels/orchestration/prefetch_async_orch.cpp"),
        extra_include_dirs=[str(kc.project_root / "src" / "common")],
    )
    return ChipCallable.build(
        signature=signature,
        func_name="prefetch_async_orchestration",
        binary=orch,
        children=children,
    )


def run(platform: str = "a2a3", device_id: int = 0) -> int:
    if platform.endswith("sim"):
        raise ValueError("prefetch_async_demo requires onboard hardware")

    src = torch.arange(N, dtype=torch.float32) / 8.0
    out = torch.full((N,), -1.0, dtype=torch.float32)

    chip_callable = build_chip_callable(platform)
    worker = Worker(level=2, platform=platform, runtime=RUNTIME, device_id=device_id, enable_sdma=True)
    worker.init()
    try:
        handle = worker.register(chip_callable)
        args = ChipStorageTaskArgs()
        args.add_tensor(make_tensor_arg(src))
        args.add_tensor(make_tensor_arg(out))
        worker.run(handle, args, CallConfig())
    finally:
        worker.close()

    if not torch.equal(out, src):
        bad = int((out != src).sum().item())
        first = int((out != src).nonzero()[0].item())
        print(
            f"[ERROR] prefetch_async_demo mismatch count={bad}, first={first}, "
            f"got={float(out[first])}, expect={float(src[first])}"
        )
        return 1
    print("[INFO] prefetch_async_demo: out matches src after injected SDMA prefetch + copy")
    return 0


@pytest.mark.platforms(["a2a3"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(1)
def test_prefetch_async_demo(st_platform, st_device_ids) -> None:
    """Prefetching a GM region then copying it leaves the data bit-exact."""
    assert run(platform=st_platform, device_id=int(st_device_ids[0])) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="a2a3")
    parser.add_argument("--device", type=int, default=0)
    cli = parser.parse_args()
    raise SystemExit(run(platform=cli.platform, device_id=cli.device))
