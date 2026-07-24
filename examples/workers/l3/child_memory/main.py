#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3 Worker API demo — orch-allocated weight, child_memory tensor, two kernel calls.

Pattern (the "load weights once, run kernel many times" idiom):

  * ``orch.malloc(worker_id=0, nbytes)`` — allocate a buffer that lives on
    the chip child for as long as the chip is alive.
  * ``orch.copy_to(worker_id=0, dev, host, n)`` — H2D upload of the weight.
  * ``Tensor.make(dev_ptr, shape, dtype, child_memory=True)`` —
    wrap the worker pointer as a tensor that the runtime treats as
    *already on device*. ``init_runtime_impl`` skips malloc + H2D copy
    for these and does not record them in ``tensor_pairs``, so the buffer
    is **not** freed at the end of the task — it stays live for the next
    invocation.
  * Submit two kernel tasks pinned to the same worker, both reading the
    same weight tensor. The second invocation proves the weight survived
    the first task's teardown.

Primitives introduced over ``multi_chip_dispatch`` (see that example for the
basics of L3 / orch / submit_next_level):

  * ``orch.malloc / orch.copy_to`` — control-plane device-memory ops that
    forward to the chip child via mailbox IPC.
  * ``Tensor.make(..., child_memory=True)`` — opt-out of the
    runtime's auto-malloc + auto-free for a tensor whose lifetime you
    manage yourself.

Reuses the kernels from ``examples/a2a3/tensormap_and_ringbuffer/vector_example/``
(``f = (a + w + 1) * (a + w + 2) + (a + w)``) — only a2a3 kernels exist for
this orchestration, so the example is a2a3-only on purpose.

Run:
    python examples/workers/l3/child_memory/main.py -p a2a3sim -d 0
    python examples/workers/l3/child_memory/main.py -p a2a3   -d 0
"""

import argparse
import os
import sys

# See the equivalent comment in ../multi_chip_dispatch/main.py — Linux is unaffected.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    CoreCallable,
    DataType,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import Worker

from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS_DIR = os.path.normpath(os.path.join(HERE, "../../../a2a3/tensormap_and_ringbuffer/vector_example/kernels"))

SIZE = 128 * 128
NBYTES = SIZE * 4  # float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--platform", required=True, choices=["a2a3sim", "a2a3"])
    parser.add_argument("-d", "--device", type=int, default=0)
    return parser.parse_args()


def build_chip_callable(platform: str) -> ChipCallable:
    """Compile the vector_example orchestration + 3 AIV kernels into one ChipCallable."""
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root()
    include_dirs = kc.get_orchestration_include_dirs(runtime)

    # Three AIV kernels — func_id matches the rt_submit_aiv_task(N, ...) calls in
    # example_orchestration.cpp. Order doesn't matter; only func_id matters.
    kernel_specs = [
        (0, "kernel_add.cpp", [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT]),
        (1, "kernel_add_scalar.cpp", [ArgDirection.IN, ArgDirection.OUT]),
        (2, "kernel_mul.cpp", [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT]),
    ]
    children = []
    for func_id, fname, sig in kernel_specs:
        bytes_ = kc.compile_incore(
            source_path=os.path.join(KERNELS_DIR, "aiv", fname),
            core_type="aiv",
            pto_isa_root=pto_isa_root,
            extra_include_dirs=include_dirs,
        )
        if not platform.endswith("sim"):
            from simpler_setup.elf_parser import extract_text_section  # noqa: PLC0415

            bytes_ = extract_text_section(bytes_)
        children.append((func_id, CoreCallable.build(signature=sig, binary=bytes_)))

    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(KERNELS_DIR, "orchestration", "example_orchestration.cpp"),
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
        func_name="aicpu_orchestration_entry",
        binary=orch_bytes,
        children=children,
    )


def run(platform: str, device_id: int) -> int:
    """Core logic — callable from both CLI and pytest."""
    print(f"[child_memory] platform={platform} device={device_id}")

    # --- 1. The H2D source weight. share_memory_() makes data_ptr() valid in the chip child, where
    # orch.copy_to reads it. Inputs/outputs are handle-backed shared buffers (create_buffer, below).
    torch.manual_seed(0)
    host_w = torch.full((SIZE,), 3.0, dtype=torch.float32).share_memory_()

    # --- 2. Worker(level=3, ...) — single chip, no Python sub-workers. We
    # still need level=3 (not level=2) because orch.malloc + a device (DEVICE_MALLOC)
    # task arg require the L3 Orchestrator to reach into the chip child via mailbox.
    worker = Worker(
        level=3,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=[device_id],
        num_sub_workers=0,
    )

    print(f"[child_memory] compiling kernels for {platform}...")
    chip_callable = build_chip_callable(platform)
    chip_handle = worker.register(chip_callable)

    print("[child_memory] init worker...")
    worker.init()

    f32 = DataType.FLOAT32.value
    a = f1 = f2 = None
    try:
        # Born-shared input/output buffers (post-fork, attached into the chip child). torch is used
        # only here at the run boundary to fill the input and read the outputs.
        def _view(handle):
            shm = handle.shm
            assert shm is not None
            return torch.frombuffer(shm.buf, dtype=torch.float32, count=SIZE)

        a_h = worker.create_buffer(NBYTES)
        f1_h = worker.create_buffer(NBYTES)
        f2_h = worker.create_buffer(NBYTES)
        a = _view(a_h)
        f1 = _view(f1_h)
        f2 = _view(f2_h)
        a.fill_(2.0)
        f1.zero_()
        f2.zero_()
        s = a + host_w
        expected = (s + 1) * (s + 2) + s

        def orch_fn(orch, _args, cfg):
            # Allocate the weight on chip 0 as a DEVICE_MALLOC handle (kind4; successor of
            # orch.malloc + child_memory=True). alloc_child_tensor rtMallocs inside the chip child's
            # bound device context and wraps the pointer as a handle owned by this worker — NOT
            # auto-freed at end-of-task, so both kernel calls share the same live weight (reclaimed at
            # worker.close()). Its .base is the device pointer, the copy_to destination.
            w_h = orch.alloc_child_tensor(worker_id=0, shapes=(SIZE,), dtype=DataType.FLOAT32)
            orch.copy_to(worker_id=0, dst=w_h.base, src=host_w.data_ptr(), size=NBYTES)

            for out_h in (f1_h, f2_h):
                ta = TaskArgs()
                ta.add_ref(a_h.ref(shapes=(SIZE,), dtype=f32), TensorArgType.INPUT)
                ta.add_ref(w_h.ref(shapes=(SIZE,), dtype=f32), TensorArgType.INPUT)
                ta.add_ref(out_h.ref(shapes=(SIZE,), dtype=f32), TensorArgType.OUTPUT_EXISTING)
                orch.submit_next_level(chip_handle, ta, cfg, worker=0)

        print("[child_memory] running DAG (1 malloc + 1 copy_to + 2 kernel tasks)...")
        worker.run(orch_fn, args=None, config=CallConfig())

        # --- 3. Verify — both outputs must equal the same golden, and they must agree with each other
        # (proves the second invocation read the same weight, not freed/garbage memory).
        for tag, got in (("f1", f1), ("f2", f2)):
            max_diff = float(torch.max(torch.abs(got - expected)))
            print(f"[child_memory] {tag}: max |got - expected| = {max_diff:.3e}")
            assert torch.allclose(got, expected, rtol=1e-5, atol=1e-5), f"{tag} mismatch"
        assert torch.equal(f1, f2), "f1 and f2 diverged — weight buffer was not preserved"
        print("[child_memory] golden + cross-invocation checks PASSED")
    finally:
        a = f1 = f2 = None  # drop views before close unlinks the shm
        worker.close()
    return 0


def main() -> int:
    cli = parse_args()
    return run(cli.platform, cli.device)


if __name__ == "__main__":
    sys.exit(main())
