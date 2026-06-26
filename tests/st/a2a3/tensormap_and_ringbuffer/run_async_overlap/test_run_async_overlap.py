#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hardware acceptance for L3 DAG run_async/register_async overlap.

Run through task-submit, for example:

    task-submit --device auto --device-num 1 --run \
      "python tests/st/a2a3/tensormap_and_ringbuffer/run_async_overlap/test_run_async_overlap.py \
       --platform a2a3 --device \\$TASK_DEVICE"
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402
from simpler.task_interface import (  # noqa: E402
    ArgDirection,
    CallConfig,
    ChipCallable,
    CoreCallable,
    DataType,
    TaskArgs,
    Tensor,
    TensorArgType,
)
from simpler.worker import Worker  # noqa: E402

from simpler_setup.elf_parser import extract_text_section  # noqa: E402
from simpler_setup.kernel_compiler import KernelCompiler  # noqa: E402
from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: E402
from simpler_setup.torch_interop import make_tensor_arg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VECTOR_EXAMPLE = os.path.abspath(os.path.join(HERE, "../../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example"))
RUNTIME = "tensormap_and_ringbuffer"
N_ROWS = 128
N_COLS = 128
N_ELEMS = N_ROWS * N_COLS
NBYTES = N_ELEMS * 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--repeat-count", type=int, default=2000)
    parser.add_argument("--skip-register", action="store_true")
    parser.add_argument("--dag-baseline", action="store_true")
    return parser.parse_args()


def _kernel_compiler(platform: str):
    kc = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root(clone_protocol="https")
    include_dirs = kc.get_orchestration_include_dirs(RUNTIME)
    return kc, pto_isa_root, include_dirs


def build_repeat_vector_add_callable(platform: str) -> ChipCallable:
    kc, pto_isa_root, include_dirs = _kernel_compiler(platform)
    kernel_bytes = kc.compile_incore(
        source_path=os.path.join(VECTOR_EXAMPLE, "kernels/aiv/kernel_add.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=include_dirs,
    )
    if not platform.endswith("sim"):
        kernel_bytes = extract_text_section(kernel_bytes)
    orch_bytes = kc.compile_orchestration(
        runtime_name=RUNTIME,
        source_path=os.path.join(HERE, "kernels/orchestration/repeat_vector_add_orch.cpp"),
    )
    core_callable = CoreCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
        binary=kernel_bytes,
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
        func_name="repeat_vector_add_orchestration",
        config_name="repeat_vector_add_orchestration_config",
        binary=orch_bytes,
        children=[(0, core_callable)],
    )


def build_vector_add_callable(platform: str) -> ChipCallable:
    kc, pto_isa_root, include_dirs = _kernel_compiler(platform)
    kernel_bytes = kc.compile_incore(
        source_path=os.path.join(VECTOR_EXAMPLE, "kernels/aiv/kernel_add.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=include_dirs,
    )
    if not platform.endswith("sim"):
        kernel_bytes = extract_text_section(kernel_bytes)
    orch_bytes = kc.compile_orchestration(
        runtime_name=RUNTIME,
        source_path=os.path.join(HERE, "kernels/orchestration/simple_vector_add_orch.cpp"),
    )
    core_callable = CoreCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
        binary=kernel_bytes,
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
        func_name="simple_vector_add_orchestration",
        config_name="simple_vector_add_orchestration_config",
        binary=orch_bytes,
        children=[(0, core_callable)],
    )


def make_host_data():
    host_a = torch.full((N_ELEMS,), 2.0, dtype=torch.float32).share_memory_()
    host_w = torch.full((N_ELEMS,), 3.0, dtype=torch.float32).share_memory_()
    host_out = torch.zeros(N_ELEMS, dtype=torch.float32).share_memory_()
    expected = host_a + host_w
    return host_a, host_w, host_out, expected


def make_vector_args(host_a: torch.Tensor, host_out: torch.Tensor, dev_w: int):
    w_dev = Tensor.make(dev_w, (N_ELEMS,), DataType.FLOAT32, child_memory=True)
    args = TaskArgs()
    args.add_tensor(make_tensor_arg(host_a), TensorArgType.INPUT)
    args.add_tensor(w_dev, TensorArgType.INPUT)
    args.add_tensor(make_tensor_arg(host_out), TensorArgType.OUTPUT_EXISTING)
    return args


def run(platform: str, device: int, repeat_count: int, *, skip_register: bool = False, dag_baseline: bool = False) -> None:
    repeat_callable = None if skip_register else build_repeat_vector_add_callable(platform)
    vector_callable = build_vector_add_callable(platform)

    worker = Worker(level=3, platform=platform, runtime=RUNTIME, device_ids=[device], num_sub_workers=0)
    repeat_handle = worker.register(repeat_callable) if repeat_callable is not None else None
    skip_vector_handle = worker.register(vector_callable) if skip_register else None
    host_a, host_w, host_out, expected = make_host_data()
    worker.init()

    dev_w: int | None = None
    try:
        dev_w = worker.malloc(NBYTES, worker_id=0)
        worker.copy_to(dev_w, host_w.data_ptr(), NBYTES, worker_id=0)
        vec_args = make_vector_args(host_a, host_out, dev_w)
        repeat_args = TaskArgs()
        for i in range(vec_args.tensor_count()):
            repeat_args.add_tensor(vec_args.tensor(i))
        repeat_args.add_scalar(int(repeat_count))
        cfg = CallConfig()

        submit_ns = time.perf_counter_ns()
        c1_handle = skip_vector_handle if skip_register else repeat_handle
        assert c1_handle is not None
        c1_args = vec_args if skip_register else repeat_args
        if dag_baseline:
            worker.run(lambda orch, _args, _cfg: orch.submit_next_level(c1_handle, c1_args, cfg, worker=0))
            max_diff = float(torch.max(torch.abs(host_out - expected)))
            print(f"[run_async_overlap] dag_baseline max_diff={max_diff:.3e}")
            assert torch.allclose(host_out, expected, rtol=1e-5, atol=1e-5)
            return
        def run_c1(orch, _args, _cfg):
            orch.submit_next_level(c1_handle, c1_args, cfg, worker=0)

        run_handle = worker.run_async(run_c1)
        if skip_register:
            repeat_timing = run_handle.wait()
            max_diff = float(torch.max(torch.abs(host_out - expected)))
            print(
                "[run_async_overlap] "
                f"skip_register repeat_host_us={repeat_timing.host_wall_us:.1f} max_diff={max_diff:.3e}"
            )
            assert torch.allclose(host_out, expected, rtol=1e-5, atol=1e-5)
            return
        time.sleep(0.005)

        reg_start_ns = time.perf_counter_ns()
        vector_pending = worker.register_async(vector_callable)
        vector_handle = vector_pending.wait()
        reg_done_ns = time.perf_counter_ns()

        repeat_timing = run_handle.wait()
        run_done_ns = time.perf_counter_ns()

        register_wait_ns = reg_done_ns - reg_start_ns
        run_total_ns = run_done_ns - submit_ns
        sequential_estimate_ns = repeat_timing.host_wall_ns + register_wait_ns
        print(
            "[run_async_overlap] "
            f"register_wait_us={register_wait_ns / 1000.0:.1f} "
            f"run_total_us={run_total_ns / 1000.0:.1f} "
            f"repeat_host_us={repeat_timing.host_wall_us:.1f} "
            f"sequential_estimate_us={sequential_estimate_ns / 1000.0:.1f}"
        )
        if run_total_ns >= sequential_estimate_ns * 0.85:
            raise AssertionError(
                "run/register did not overlap enough: "
                f"run_total_ns={run_total_ns}, sequential_estimate_ns={sequential_estimate_ns}"
            )

        worker.run(lambda orch, _args, _cfg: orch.submit_next_level(vector_handle, vec_args, cfg, worker=0))
        max_diff = float(torch.max(torch.abs(host_out - expected)))
        print(f"[run_async_overlap] vector_add max_diff={max_diff:.3e}")
        assert torch.allclose(host_out, expected, rtol=1e-5, atol=1e-5)
    finally:
        if dev_w is not None:
            worker.free(dev_w, worker_id=0)
        worker.close()


def main() -> int:
    args = parse_args()
    run(args.platform, args.device, args.repeat_count, skip_register=args.skip_register, dag_baseline=args.dag_baseline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
