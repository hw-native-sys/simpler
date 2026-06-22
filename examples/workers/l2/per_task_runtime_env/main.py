#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L2 Worker API demo — per-task ring sizing via ``CallConfig.runtime_env``.

Runs the same vector_add kernel three times on one L2 Worker, each time with a
different ``CallConfig.runtime_env`` (ring buffer sizing). Ring sizing is a
per-run knob carried on ``CallConfig`` — no process-wide ``PTO2_RING_*`` env
export needed, and each ``worker.run`` binds its ring buffers from the config
it was handed.

    runtime_env fields (0 / unset => fall back to env var / compile default):
      ring_task_window   power of 2, >= 4
      ring_heap          bytes per ring, power of 2, >= 1024
      ring_dep_pool      4 .. INT32_MAX
    Precedence: runtime_env field > PTO2_RING_* env var > compile-time default.

See ../vector_add/main.py for the full L2 lifecycle walk-through; this example
reuses that kernel verbatim and only varies the per-run ring configuration.

Run:
    python examples/workers/l2/per_task_runtime_env/main.py -p a2a3sim -d 0
"""

import argparse
import os
import sys
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    ChipStorageTaskArgs,
    CoreCallable,
    DataType,
    Tensor,
)
from simpler.worker import Worker

from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

HERE = os.path.dirname(os.path.abspath(__file__))
# Reuse the sibling vector_add kernel verbatim — this example only varies ring sizing.
VECTOR_ADD_KERNELS = os.path.join(HERE, "..", "vector_add", "kernels")

N_ROWS = 128
N_COLS = 128
N_ELEMS = N_ROWS * N_COLS
NBYTES = N_ELEMS * 4  # float32

# (label, runtime_env dict or None). None => no override; falls back to the
# PTO2_RING_* env var / compile-time default. Same kernel + same inputs run
# under every sizing, so all three produce identical (correct) output.
RING_CONFIGS = [
    ("small_ring", {"ring_task_window": 16, "ring_heap": 1 * 1024 * 1024, "ring_dep_pool": 64}),
    ("large_ring", {"ring_task_window": 128, "ring_heap": 8 * 1024 * 1024, "ring_dep_pool": 256}),
    ("env_or_default", None),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--platform", required=True, choices=["a2a3sim", "a2a3"])
    parser.add_argument("-d", "--device", type=int, default=0)
    return parser.parse_args()


def build_chip_callable(platform: str) -> ChipCallable:
    """Compile the reused vector_add sources into a ChipCallable.

    Identical to ../vector_add/main.py::build_chip_callable except the kernel
    sources are read from the sibling vector_add example.
    """
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root(clone_protocol="https")
    include_dirs = kc.get_orchestration_include_dirs(runtime)

    kernel_bytes = kc.compile_incore(
        source_path=os.path.join(VECTOR_ADD_KERNELS, "aiv", "vector_add_kernel.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=include_dirs,
    )
    if not platform.endswith("sim"):
        from simpler_setup.elf_parser import extract_text_section  # noqa: PLC0415

        kernel_bytes = extract_text_section(kernel_bytes)

    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(VECTOR_ADD_KERNELS, "orchestration", "vector_add_orch.cpp"),
    )
    core_callable = CoreCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
        binary=kernel_bytes,
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
        func_name="vector_add_orchestration",
        binary=orch_bytes,
        children=[(0, core_callable)],
    )


def _make_config(ring: Optional[dict]) -> CallConfig:
    """Build a CallConfig, attaching this run's ring sizing under runtime_env."""
    cfg = CallConfig()
    if ring is not None:
        cfg.runtime_env.ring_task_window = ring["ring_task_window"]
        cfg.runtime_env.ring_heap = ring["ring_heap"]
        cfg.runtime_env.ring_dep_pool = ring["ring_dep_pool"]
    return cfg


def _run_one(worker: Worker, chip_handle, label: str, ring: Optional[dict]) -> None:
    """One malloc → copy → run(config) → readback → verify cycle."""
    torch.manual_seed(42)
    host_a = torch.randn(N_ROWS, N_COLS, dtype=torch.float32)
    host_b = torch.randn(N_ROWS, N_COLS, dtype=torch.float32)
    expected = host_a + host_b
    host_out = torch.zeros(N_ROWS, N_COLS, dtype=torch.float32)

    dev_a = worker.malloc(NBYTES)
    dev_b = worker.malloc(NBYTES)
    dev_out = worker.malloc(NBYTES)
    worker.copy_to(dev_a, host_a.data_ptr(), NBYTES)
    worker.copy_to(dev_b, host_b.data_ptr(), NBYTES)

    args = ChipStorageTaskArgs()
    args.add_tensor(Tensor.make(dev_a, (N_ROWS, N_COLS), DataType.FLOAT32))
    args.add_tensor(Tensor.make(dev_b, (N_ROWS, N_COLS), DataType.FLOAT32))
    args.add_tensor(Tensor.make(dev_out, (N_ROWS, N_COLS), DataType.FLOAT32))

    config = _make_config(ring)
    print(f"[per_task_runtime_env] run '{label}': runtime_env={config.runtime_env!r}")
    worker.run(chip_handle, args, config)

    worker.copy_from(host_out.data_ptr(), dev_out, NBYTES)
    worker.free(dev_a)
    worker.free(dev_b)
    worker.free(dev_out)

    assert torch.allclose(host_out, expected, rtol=1e-5, atol=1e-5), f"{label} result mismatch"
    print(f"[per_task_runtime_env] '{label}' golden check PASSED")


def run(platform: str, device_id: int) -> int:
    """Core logic — callable from both CLI and pytest."""
    worker = Worker(
        level=2,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_id=device_id,
    )

    print(f"[per_task_runtime_env] compiling kernels for {platform}...")
    chip_callable = build_chip_callable(platform)
    chip_handle = worker.register(chip_callable)

    print(f"[per_task_runtime_env] init worker (device={device_id})...")
    worker.init()
    try:
        for label, ring in RING_CONFIGS:
            _run_one(worker, chip_handle, label, ring)
    finally:
        worker.close()
    print("[per_task_runtime_env] all ring configurations PASSED ✅")
    return 0


def main() -> int:
    args = parse_args()
    return run(args.platform, args.device)


if __name__ == "__main__":
    sys.exit(main())
