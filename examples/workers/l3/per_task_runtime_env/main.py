#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3 Worker API demo — one orchestration dispatches several L2 tasks, each
sized with its OWN ring buffers.

This is the headline use case for ``CallConfig.runtime_env``: an L3 fans out
several heterogeneous L2 tasks in one launch, and each L2 needs a different
ring footprint (a heavy task wants a big heap / wide window; a light one is
fine with the default). Before this knob, all L2 tasks in one L3 launch shared
the process-wide ``PTO2_RING_*`` env and could not be sized independently.

The demo dispatches three L2 tasks: two use the scalar form (one ring value
broadcast to every ring), and a third passes 4-entry lists for
``ring_task_window`` / ``ring_heap`` / ``ring_dep_pool`` to size each of the four
scope-depth rings independently.

The key line is inside ``orch_fn``: each ``submit_next_level`` gets its OWN
``CallConfig`` whose ``runtime_env`` is set per task. That per-task config
travels through the mailbox to the chip child, so each L2 binds its ring
buffers from its own values.

Both L2s run the same vector_add kernel (reused from ../../l2/vector_add), so
golden validation confirms each ring sizing produces correct output.

Run:
    python examples/workers/l3/per_task_runtime_env/main.py -p a2a3sim -d 0
"""

import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    CoreCallable,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import Worker

from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root
from simpler_setup.torch_interop import make_tensor_arg

HERE = os.path.dirname(os.path.abspath(__file__))
# Reuse the L2 vector_add kernel verbatim — this example only varies ring sizing.
VECTOR_ADD_KERNELS = os.path.join(HERE, "..", "..", "l2", "vector_add", "kernels")

N_ROWS = 128
N_COLS = 128

# RuntimeEnv keys an L2 spec may carry. Each takes a scalar (broadcast to every
# ring) or a 4-entry list (one value per scope-depth ring 0..3).
RING_FIELDS = ("ring_task_window", "ring_heap", "ring_dep_pool")

# One entry per L2 task dispatched by the orchestration. Each carries its own
# ring sizing; the inputs differ so the golden checks are independent. The first
# two tasks use the scalar form (one value broadcast to every ring); the third
# passes 4-entry lists to size each scope-depth ring independently.
L2_TASKS = [
    {
        "label": "l2_scalar_small",
        "a": 2.0,
        "b": 3.0,
        "ring_task_window": 16,
        "ring_heap": 1 * 1024 * 1024,
        "ring_dep_pool": 64,
    },
    {
        "label": "l2_scalar_large",
        "a": 5.0,
        "b": 7.0,
        "ring_task_window": 128,
        "ring_heap": 8 * 1024 * 1024,
        "ring_dep_pool": 256,
    },
    {
        "label": "l2_per_ring",
        "a": 1.0,
        "b": 4.0,
        "ring_task_window": [128, 64, 32, 16],
        "ring_heap": [8 * 1024 * 1024, 4 * 1024 * 1024, 2 * 1024 * 1024, 1 * 1024 * 1024],
        "ring_dep_pool": [256, 128, 64, 64],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--platform", required=True, choices=["a2a3sim", "a2a3"])
    parser.add_argument("-d", "--device", type=int, default=0, help="Single device id; the L2 tasks run serially.")
    return parser.parse_args()


def build_chip_callable(platform: str) -> ChipCallable:
    """Compile the reused vector_add sources into a ChipCallable.

    See ../../l2/vector_add/main.py for the lifecycle walk-through.
    """
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root()
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


def _l2_config(base: CallConfig, spec: dict) -> CallConfig:
    """Derive this L2 task's CallConfig from the orchestration's base config,
    overriding only runtime_env.

    runtime_env lives on CallConfig alongside the diagnostics flags
    (enable_scope_stats, output_prefix, ...). Each submit_next_level needs its
    own CallConfig to size its rings independently, but it must preserve any
    framework-injected fields on the base config — otherwise harness-driven
    diagnostics (--enable-scope-stats, etc.) silently collect nothing for the
    child L2. So copy those over and override only the ring sizing.
    """
    cfg = CallConfig()
    cfg.block_dim = base.block_dim
    cfg.aicpu_thread_num = base.aicpu_thread_num
    cfg.enable_l2_swimlane = base.enable_l2_swimlane
    cfg.enable_dump_tensor = base.enable_dump_tensor
    cfg.enable_pmu = base.enable_pmu
    cfg.enable_dep_gen = base.enable_dep_gen
    cfg.enable_scope_stats = base.enable_scope_stats
    cfg.output_prefix = base.output_prefix
    for key in RING_FIELDS:
        if key in spec:
            setattr(cfg.runtime_env, key, spec[key])
    return cfg


def run(platform: str, device_id: int) -> int:
    """Core logic — callable from both CLI and pytest."""
    print(f"[per_task_runtime_env] device={device_id}, {len(L2_TASKS)} L2 tasks")

    # Shared-memory tensors so the forked chip child writes results the parent
    # sees in place (no explicit copy_back). One (a, b, out) triple per L2.
    host_a = [torch.full((N_ROWS, N_COLS), t["a"], dtype=torch.float32).share_memory_() for t in L2_TASKS]
    host_b = [torch.full((N_ROWS, N_COLS), t["b"], dtype=torch.float32).share_memory_() for t in L2_TASKS]
    host_out = [torch.zeros(N_ROWS, N_COLS, dtype=torch.float32).share_memory_() for _ in L2_TASKS]
    expected = [a + b for a, b in zip(host_a, host_b)]

    worker = Worker(
        level=3,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=[device_id],
        num_sub_workers=0,
    )

    print(f"[per_task_runtime_env] compiling kernels for {platform}...")
    chip_callable = build_chip_callable(platform)
    chip_handle = worker.register(chip_callable)

    print("[per_task_runtime_env] init worker...")
    worker.init()
    try:

        def orch_fn(orch, _args, _cfg):
            # One chip task per L2 spec, each with its OWN runtime_env. This is
            # the per-task ring sizing #1025 enables: heterogeneous L2s in one
            # launch, each binding its own rings.
            for i, spec in enumerate(L2_TASKS):
                chip_args = TaskArgs()
                chip_args.add_tensor(make_tensor_arg(host_a[i]), TensorArgType.INPUT)
                chip_args.add_tensor(make_tensor_arg(host_b[i]), TensorArgType.INPUT)
                chip_args.add_tensor(make_tensor_arg(host_out[i]), TensorArgType.OUTPUT_EXISTING)
                cfg = _l2_config(_cfg, spec)
                print(f"[per_task_runtime_env] submit '{spec['label']}': runtime_env={cfg.runtime_env!r}")
                orch.submit_next_level(chip_handle, chip_args, cfg)

        print(f"[per_task_runtime_env] running DAG ({len(L2_TASKS)} L2 tasks, distinct rings)...")
        worker.run(orch_fn, args=None, config=CallConfig())

        for i, spec in enumerate(L2_TASKS):
            max_diff = float(torch.max(torch.abs(host_out[i] - expected[i])))
            print(f"[per_task_runtime_env] {spec['label']}: max |out - expected| = {max_diff:.3e}")
            assert torch.allclose(host_out[i], expected[i], rtol=1e-5, atol=1e-5), f"{spec['label']} mismatch"

        print("[per_task_runtime_env] all golden checks PASSED ✅")
    finally:
        worker.close()
    return 0


def main() -> int:
    cli = parse_args()
    return run(cli.platform, cli.device)


if __name__ == "__main__":
    sys.exit(main())
