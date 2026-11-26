#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3 Worker API demo — one orchestration, two chips, one SubWorker.

Dispatches the same vector_add ChipCallable to two chips (each with its own
input tensors), then runs a Python SubWorker that depends on the chip outputs.

Primitives introduced over L2 (see ../../l2/vector_add/main.py for the L2 version):

  * torch.share_memory_() tensors       — visible to forked chip children, IS the data plane
  * TaskArgs + TensorArgType tags        — INPUT / OUTPUT_EXISTING drive DAG deps
  * Worker.register(python_fn)           — register Python callable runnable as a sub task
  * Worker(level=3, device_ids=[...], num_sub_workers=N) — multi-chip + sub fork-and-serve
  * orch.submit_next_level(cb, args, cfg, worker=i)  — submit a chip task
  * orch.submit_sub(handle, args)        — submit a Python sub task

Run:
    python examples/workers/l3/multi_chip_dispatch/main.py -p a2a3sim -d 0-1
"""

import argparse
import ctypes
import os
import sys

# Workaround for the duplicate-libomp abort when homebrew numpy and pip torch
# coexist in one macOS process. Harmless on Linux. Must be set before
# ``import torch``. See docs/troubleshooting/macos-libomp-collision.md.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    CoreCallable,
    DataType,
    RemoteTensorRef,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import RemoteChipSpec, Worker

from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root
from simpler_setup.torch_interop import make_tensor_arg

HERE = os.path.dirname(os.path.abspath(__file__))

N_ROWS = 128
N_COLS = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--platform", required=True, choices=["a2a3sim", "a2a3", "a5sim", "a5"])
    parser.add_argument("-d", "--device", default="0-1", help="Device range, e.g. '0-1' or '4-5'. Two chips required.")
    parser.add_argument("--remote-endpoint", help="Remote daemon IPv4 address and port, e.g. 192.0.2.20:19073")
    parser.add_argument("--remote-device", default="0-1", help="Device range on the remote host")
    return parser.parse_args()


def parse_device_range(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        ids = list(range(lo, hi + 1))
    else:
        ids = [int(spec)]
    if len(ids) != 2:
        raise ValueError(f"multi_chip_dispatch needs exactly 2 devices, got {ids}")
    return ids


def build_chip_callable(platform: str) -> ChipCallable:
    """Same as L2 vector_add — see ../../l2/vector_add/main.py for the walk-through."""
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root()
    include_dirs = kc.get_orchestration_include_dirs(runtime)

    kernel_bytes = kc.compile_incore(
        source_path=os.path.join(HERE, "kernels/aiv/vector_add_kernel.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=include_dirs,
    )
    if not platform.endswith("sim"):
        from simpler_setup.elf_parser import extract_text_section  # noqa: PLC0415

        kernel_bytes = extract_text_section(kernel_bytes)

    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "kernels/orchestration/vector_add_orch.cpp"),
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


def run(platform: str, device_ids: list[int]) -> int:
    """Core logic — callable from both CLI and pytest."""
    print(f"[multi_chip_dispatch] devices={device_ids}")

    # --- 1. Allocate shared-memory tensors (visible to forked chip processes).
    # ``share_memory_()`` moves the storage into an mmap region, so when the
    # chip process writes to ``host_out[i]`` the parent's tensor sees the
    # update immediately — no explicit copy_back needed.
    torch.manual_seed(42)
    host_a = [torch.randn(N_ROWS, N_COLS, dtype=torch.float32).share_memory_() for _ in device_ids]
    host_b = [torch.randn(N_ROWS, N_COLS, dtype=torch.float32).share_memory_() for _ in device_ids]
    host_out = [torch.zeros(N_ROWS, N_COLS, dtype=torch.float32).share_memory_() for _ in device_ids]
    expected = [a + b for a, b in zip(host_a, host_b)]

    # --- 2. Worker(level=3, ...) construction. No fork / no ACL yet.
    worker = Worker(
        level=3,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=device_ids,
        num_sub_workers=1,
    )

    # --- 3. Register the Python SubWorker callable BEFORE init().
    # Register before init() so startup can seed child registries and pre-warm
    # chip callables before the first DAG dispatch.
    def subworker(sub_args: TaskArgs) -> None:
        # sub_args carries the tensors we tagged INPUT on submit_sub below.
        # In this demo we just print confirmation; a real app might write
        # results to disk or trigger the next pipeline stage.
        print(f"[multi_chip_dispatch] subworker fired (received {sub_args.tensor_count()} tensor refs) ✅")

    sub_handle = worker.register(subworker)

    # --- 4. Compile the ChipCallable once, reused on both chips.
    print(f"[multi_chip_dispatch] compiling kernels for {platform}...")
    chip_callable = build_chip_callable(platform)

    # Register the ChipCallable so submit_next_level takes a handle.
    chip_handle = worker.register(chip_callable)

    # --- 5. init() forks chip + sub child processes, starts C++ scheduler.
    print("[multi_chip_dispatch] init worker...")
    worker.init()

    try:
        # --- 6. Define the orchestration function. It runs in the parent
        # process once per worker.run() call. Each ``orch.submit_*`` adds a
        # node to the DAG; the scheduler (running on C++ threads) actually
        # dispatches the nodes to child processes.
        def orch_fn(orch, _args, cfg):
            # Submit one chip task per device. ``worker=i`` pins the task to
            # the i-th ChipWorker. TensorArgType drives dependency tracking:
            #   INPUT           = "must be ready before task starts"
            #   OUTPUT_EXISTING = "task writes to this pre-allocated tensor"
            for i in range(len(device_ids)):
                chip_args = TaskArgs()
                chip_args.add_tensor(make_tensor_arg(worker, host_a[i]), TensorArgType.INPUT)
                chip_args.add_tensor(make_tensor_arg(worker, host_b[i]), TensorArgType.INPUT)
                chip_args.add_tensor(make_tensor_arg(worker, host_out[i]), TensorArgType.OUTPUT_EXISTING)
                orch.submit_next_level(chip_handle, chip_args, cfg, worker=i)

            # Sub task that depends on both chip outputs. Tagging the two
            # host_out[i] tensors INPUT tells the scheduler to wait for
            # both chip tasks to finish before running the sub.
            sub_args = TaskArgs()
            for i in range(len(device_ids)):
                sub_args.add_tensor(make_tensor_arg(worker, host_out[i]), TensorArgType.INPUT)
            orch.submit_sub(sub_handle, sub_args)

        # --- 7. Run the DAG. Worker.run() opens a scope, invokes orch_fn,
        # drains the DAG to completion, then closes the scope.
        print("[multi_chip_dispatch] running DAG (2 chip tasks + 1 sub)...")
        worker.run(orch_fn, args=None, config=CallConfig())

        # --- 8. Verify outputs. ``host_out`` was written in-place by the
        # chip processes via shared memory; no explicit copy needed.
        for i in range(len(device_ids)):
            max_diff = float(torch.max(torch.abs(host_out[i] - expected[i])))
            print(f"[multi_chip_dispatch] chip {i}: max |out - expected| = {max_diff:.3e}")
            assert torch.allclose(host_out[i], expected[i], rtol=1e-5, atol=1e-5), f"chip {i} result mismatch"

        print("[multi_chip_dispatch] all golden checks PASSED ✅")
    finally:
        # close() shuts down sub + chip child processes, unlinks shared memory.
        worker.close()
    return 0


def run_remote(platform: str, device_ids: list[int], endpoint: str, remote_device_ids: list[int]) -> int:
    """Validate one L3 dispatching directly to local and remote single-card runtimes."""
    worker = Worker(
        level=3, platform=platform, runtime="tensormap_and_ringbuffer", device_ids=device_ids, num_sub_workers=0
    )
    remote_ids = [
        worker.add_remote_chip(RemoteChipSpec(endpoint=endpoint, platform=platform, device_id=device_id))
        for device_id in remote_device_ids
    ]
    callable_obj = build_chip_callable(platform)
    chip = worker.register(callable_obj)
    startup_alias = worker.register(callable_obj)
    worker.init()
    remote_outputs = []
    try:
        # Exercise dynamic registration as well as the startup registry snapshot.
        dynamic_chip = worker.register(callable_obj)
        count = len(device_ids) + len(remote_ids)
        shape = (N_ROWS, N_COLS)
        nbytes = N_ROWS * N_COLS * 4
        local_outputs, expected, task_args = [], [], []
        for rank in range(count):
            a = torch.arange(N_ROWS * N_COLS, dtype=torch.float32).reshape(shape) + rank
            b = torch.full(shape, rank + 0.5, dtype=torch.float32)
            expected.append(a + b)
            args = TaskArgs()
            if rank < len(device_ids):
                inputs = [worker.create_buffer(nbytes) for _ in range(2)]
                output = worker.create_buffer(nbytes)
                local_outputs.append(output)
                for buffer, value in zip(inputs, (a, b)):
                    ctypes.memmove(buffer.base, value.numpy().tobytes(), nbytes)
                    args.add_tensor(buffer.tensor(shape, DataType.FLOAT32), TensorArgType.INPUT)
                args.add_tensor(output.tensor(shape, DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
            else:
                output = worker.remote_malloc(worker=remote_ids[rank - len(device_ids)], nbytes=nbytes)
                remote_outputs.append(output)
                args.add_tensor(RemoteTensorRef.host_inline(a.numpy().tobytes(), shape=shape, dtype=DataType.FLOAT32))
                args.add_tensor(RemoteTensorRef.host_inline(b.numpy().tobytes(), shape=shape, dtype=DataType.FLOAT32))
                args.add_tensor(
                    RemoteTensorRef(output, shape=shape, dtype=DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING
                )
            task_args.append(args)

        targets = list(range(len(device_ids))) + remote_ids
        active_chip = dynamic_chip

        def dispatch(orch, _args, config):
            orch.submit_next_level_group(active_chip, task_args, config, workers=targets)

        poison = torch.full(shape, -1.0, dtype=torch.float32)
        poison_bytes = poison.numpy().tobytes()
        for epoch in range(3):
            if epoch == 1:
                worker.unregister(dynamic_chip)
                active_chip = startup_alias
            elif epoch == 2:
                worker.unregister(startup_alias)
                active_chip = chip
            for output in local_outputs:
                ctypes.memmove(output.base, poison_bytes, nbytes)
            for output in remote_outputs:
                worker.remote_copy_to(output, poison.data_ptr(), nbytes)
            worker.run(dispatch, TaskArgs(), CallConfig())
            actual = [
                torch.frombuffer(bytearray(ctypes.string_at(output.base, nbytes)), dtype=torch.float32).reshape(shape)
                for output in local_outputs
            ]
            for output in remote_outputs:
                result = torch.empty(shape, dtype=torch.float32)
                worker.remote_copy_from(output, result.data_ptr(), nbytes)
                actual.append(result)
            for target, result, golden in zip(targets, actual, expected):
                torch.testing.assert_close(result, golden, rtol=1e-5, atol=1e-5)
                print(f"[remote_chip_dispatch] epoch={epoch} worker={target}: PASS")
        for output in remote_outputs:
            worker.remote_free(output)
        worker.unregister(chip)
    finally:
        worker.close()
    return 0


def main() -> int:
    cli = parse_args()
    if cli.remote_endpoint:
        return run_remote(
            cli.platform, parse_device_range(cli.device), cli.remote_endpoint, parse_device_range(cli.remote_device)
        )
    return run(cli.platform, parse_device_range(cli.device))


if __name__ == "__main__":
    sys.exit(main())
