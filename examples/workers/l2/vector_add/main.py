#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L2 Worker API demo — compile one AIV kernel, run it, verify against numpy.

Pipeline (what the @scene_test framework normally does for you):

    .cpp sources ──[KernelCompiler]──► binaries
           │                              │
           ▼                              ▼
    CoreCallable.build(kernel bytes)   ChipCallable.build(orch bytes + children)
                                          │
    host arrays ──[worker.malloc + copy_to]──►  device buffers
                                          │
                                          ▼
                              worker.run(chip_callable, task_args, cfg)
                                          │
    device result ──[worker.copy_from]──► host array ──[numpy compare]

The code below walks through each stage explicitly so you can see what the
``@scene_test`` decorator hides.

Run:
    python examples/workers/l2/vector_add/main.py -p a2a3sim -d 0
"""

import argparse
import os
import sys

import numpy as np
from simpler.task_interface import (
    ArgDirection,
    ChipCallable,
    ChipCallConfig,
    ChipStorageTaskArgs,
    ContinuousTensor,
    CoreCallable,
    DataType,
)
from simpler.worker import Worker

from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

HERE = os.path.dirname(os.path.abspath(__file__))

# Matches the tile geometry hard-coded in vector_add_kernel.cpp (128x128 float32).
# The kernel assumes tensors are exactly this shape; no boundary handling.
N_ROWS = 128
N_COLS = 128
N_ELEMS = N_ROWS * N_COLS
NBYTES = N_ELEMS * np.dtype(np.float32).itemsize


def parse_args() -> argparse.Namespace:
    """Same CLI shape as every example under ``examples/workers/``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--platform", required=True, choices=["a2a3sim", "a2a3", "a5sim", "a5"])
    parser.add_argument("-d", "--device", type=int, default=0)
    return parser.parse_args()


def build_chip_callable(platform: str) -> ChipCallable:
    """Compile orchestration + kernel sources, wrap into a ChipCallable.

    This is a thin wrapper around ``KernelCompiler`` that reads the CALLABLE
    structure you'd see in a ``@scene_test`` class and returns the opaque
    handle Worker.run(...) expects. In production code this lives inside
    ``simpler_setup/scene_test.py`` (see ``_compile_chip_callable_from_spec``);
    we inline a minimal version here so the flow is visible.
    """
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"

    # 1. Compile the AIV kernel source to a .o. ``pto_isa_root`` is the
    # sibling header repo; ``ensure_pto_isa_root()`` clones it on first run
    # (see docs/getting-started.md). Uses HTTPS clone by default; set
    # PTO_ISA_ROOT to skip the clone and point at an existing checkout.
    pto_isa_root = ensure_pto_isa_root(clone_protocol="https")
    include_dirs = kc.get_orchestration_include_dirs(runtime)
    kernel_bytes = kc.compile_incore(
        source_path=os.path.join(HERE, "kernels/aiv/vector_add_kernel.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=include_dirs,
    )

    # On real hardware (not sim) the .o needs its .text section extracted
    # before being wrapped into a CoreCallable. Skip this on sim builds.
    if not platform.endswith("sim"):
        from simpler_setup.elf_parser import extract_text_section  # noqa: PLC0415

        kernel_bytes = extract_text_section(kernel_bytes)

    # 2. Compile the orchestration source to a .so (host-loadable).
    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "kernels/orchestration/vector_add_orch.cpp"),
    )

    # 3. Wrap the kernel bytes as a CoreCallable with its tensor-arg signature
    # (a, b are inputs; out is output). ``func_id=0`` matches the first
    # (and only) ``pto2_rt_submit_aiv_task(0, ...)`` in vector_add_orch.cpp.
    core_callable = CoreCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
        binary=kernel_bytes,
    )

    # 4. Wrap everything into a ChipCallable. The orchestration function name
    # must match the ``__attribute__((visibility("default")))`` symbol in
    # vector_add_orch.cpp.
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
        func_name="vector_add_orchestration",
        binary=orch_bytes,
        children=[(0, core_callable)],
    )


def run(worker: Worker, chip_callable: ChipCallable) -> None:
    """Allocate device memory, copy inputs, execute, copy outputs back, verify."""
    # --- 1. Prepare host arrays ---
    rng = np.random.default_rng(seed=42)
    host_a = rng.standard_normal((N_ROWS, N_COLS), dtype=np.float32)
    host_b = rng.standard_normal((N_ROWS, N_COLS), dtype=np.float32)
    expected = host_a + host_b
    host_out = np.zeros((N_ROWS, N_COLS), dtype=np.float32)

    # --- 2. Allocate device buffers + H2D copy ---
    # malloc returns a uint64 device pointer. copy_to takes (dst_dev, src_host, nbytes).
    dev_a = worker.malloc(NBYTES)
    dev_b = worker.malloc(NBYTES)
    dev_out = worker.malloc(NBYTES)
    worker.copy_to(dev_a, host_a.ctypes.data, NBYTES)
    worker.copy_to(dev_b, host_b.ctypes.data, NBYTES)

    # --- 3. Build TaskArgs describing the tensors visible to the orchestration ---
    # Each tensor is a ContinuousTensor(data_ptr, shape, dtype). Order must
    # match the ``signature`` list in the ChipCallable (IN, IN, OUT).
    args = ChipStorageTaskArgs()
    args.add_tensor(ContinuousTensor.make(dev_a, (N_ROWS, N_COLS), DataType.FLOAT32))
    args.add_tensor(ContinuousTensor.make(dev_b, (N_ROWS, N_COLS), DataType.FLOAT32))
    args.add_tensor(ContinuousTensor.make(dev_out, (N_ROWS, N_COLS), DataType.FLOAT32))

    # --- 4. Run. ChipCallConfig() defaults are fine for this kernel. ---
    config = ChipCallConfig()
    print("[vector_add] running on device...")
    worker.run(chip_callable, args, config)

    # --- 5. D2H copy back + verify ---
    worker.copy_from(host_out.ctypes.data, dev_out, NBYTES)

    # --- 6. Free device buffers. Order doesn't matter, but leaking is bad. ---
    worker.free(dev_a)
    worker.free(dev_b)
    worker.free(dev_out)

    max_diff = float(np.max(np.abs(host_out - expected)))
    print(f"[vector_add] max |host_out - expected| = {max_diff:.3e}")
    np.testing.assert_allclose(host_out, expected, rtol=1e-5, atol=1e-5)
    print("[vector_add] golden check PASSED ✅")


def main() -> int:
    args = parse_args()
    worker = Worker(
        level=2,
        platform=args.platform,
        runtime="tensormap_and_ringbuffer",
        device_id=args.device,
    )

    print(f"[vector_add] compiling kernels for {args.platform}...")
    chip_callable = build_chip_callable(args.platform)
    print(f"[vector_add] compiled. binary_size={chip_callable.binary_size} bytes")

    print(f"[vector_add] init worker (device={args.device})...")
    worker.init()
    try:
        run(worker, chip_callable)
    finally:
        worker.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
