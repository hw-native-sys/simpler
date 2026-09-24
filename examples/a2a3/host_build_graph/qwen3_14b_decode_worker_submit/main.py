#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the Qwen3-14B decode callable through level-3 Worker.submit."""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch
from simpler.task_interface import ArgDirection, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.parallel_scheduler import device_range_to_list
from simpler_setup.scene_test import compile_chip_callable_spec
from simpler_setup.torch_interop import torch_dtype_to_datatype

HERE = Path(__file__).resolve().parent
TMR_DRIVER_PATH = HERE.parents[1] / "tensormap_and_ringbuffer/qwen3_14b_decode/main.py"
HBG_ORCHESTRATION = HERE.parents[1] / "host_build_graph/qwen3_14b_decode/kernels/orchestration/decode_fwd_layers.cpp"
RUNTIME = "host_build_graph"

_DTYPE_BY_NAME = {
    "BFLOAT16": torch.bfloat16,
    "FLOAT32": torch.float32,
    "INT32": torch.int32,
}
_TAG_BY_DIRECTION = {
    ArgDirection.IN: TensorArgType.INPUT,
    ArgDirection.OUT: TensorArgType.OUTPUT_EXISTING,
    ArgDirection.INOUT: TensorArgType.INOUT,
}


def _load_module(path: Path, name: str):
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Qwen driver at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _base_driver():
    return _load_module(TMR_DRIVER_PATH, "_qwen3_14b_worker_submit_base")


def _buffer_view(buffer, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    if buffer.shm is None:
        raise RuntimeError("HOST buffer is closed")
    return torch.frombuffer(buffer.shm.buf, dtype=dtype, count=math.prod(shape)).reshape(shape)


def _copy_to_buffer(buffer, tensor: torch.Tensor) -> None:
    _buffer_view(buffer, tuple(tensor.shape), tensor.dtype).copy_(tensor)


def _build_host_buffers(worker: Worker, base, *, depth: int, seed: int, seq_len: int):
    specs = base.param_specs(base.N_LAYERS)
    signature = base.TestQwen314BDecode.CALLABLE["orchestration"]["signature"]
    if len(specs) != len(signature):
        raise ValueError("Qwen parameter count does not match the orchestration signature")
    mutable_names = {spec.name for spec, direction in zip(specs, signature) if direction != ArgDirection.IN}
    common: dict[str, object] = {}
    per_run: list[dict[str, object]] = [dict() for _ in range(depth)]
    for name, tensor in base.param_tensors(seed=seed, seq_len=seq_len, n_layers=base.N_LAYERS):
        targets = per_run if name in mutable_names else [common]
        for target in targets:
            buffer = worker.create_buffer(tensor.numel() * tensor.element_size())
            _copy_to_buffer(buffer, tensor)
            target[name] = buffer
        del tensor
    return specs, signature, common, per_run


def _task_args(specs, signature, common, run_buffers):
    args = TaskArgs()
    for spec, direction in zip(specs, signature):
        buffer = run_buffers.get(spec.name, common.get(spec.name))
        if buffer is None:
            raise RuntimeError(f"missing HOST buffer for {spec.name}")
        dtype = _DTYPE_BY_NAME[spec.dtype]
        args.add_tensor(
            buffer.tensor(tuple(spec.shape), torch_dtype_to_datatype(dtype).value),
            _TAG_BY_DIRECTION[direction],
        )
    return args


def _compare_run(base, buffers, golden) -> None:
    for name in ("out", "k_cache", "v_cache"):
        spec = next(item for item in base.param_specs(base.N_LAYERS) if item.name == name)
        actual = _buffer_view(buffers[name], spec.shape, _DTYPE_BY_NAME[spec.dtype]).clone()
        expected = getattr(golden, name)
        if not torch.allclose(actual, expected, rtol=base.TestQwen314BDecode.RTOL, atol=base.TestQwen314BDecode.ATOL):
            diff = (actual.float() - expected.float()).abs().max().item()
            raise AssertionError(f"Qwen {name} mismatch: max_diff={diff}")


def run(device_ids, *, depth: int, seed: int, seq_len: int, skip_golden: bool, compile_only: bool = False) -> int:
    if depth not in (1, 2):
        raise ValueError("depth must be 1 or 2")
    if not device_ids and not compile_only:
        raise ValueError("one device is required")
    base = _base_driver()
    spec = base._chip_spec(HBG_ORCHESTRATION, "aicpu_orchestration_entry")
    cache_key = base.l3_compile_cache_key(
        "examples.qwen3_14b_decode_worker_submit",
        f"a2a3:{RUNTIME}:aicpu_orchestration_entry",
        spec["name"],
        "a2a3",
        RUNTIME,
    )
    chip = compile_chip_callable_spec(spec, "a2a3", RUNTIME, cache_key)
    if compile_only:
        return 0

    golden = None if skip_golden else base._decode_generate_inputs(seed=seed, seq_len=seq_len, n_layers=base.N_LAYERS)
    if golden is not None:
        base._decode_golden(golden, n_layers=base.N_LAYERS)

    worker = Worker(
        level=3,
        platform="a2a3",
        runtime=RUNTIME,
        device_ids=[int(device_ids[0])],
        num_sub_workers=0,
        launch_depth=depth,
    )
    chip_handle = worker.register(chip)
    worker.init()
    common = {}
    per_run = []
    try:
        specs, signature, common, per_run = _build_host_buffers(worker, base, depth=depth, seed=seed, seq_len=seq_len)
        config = base._build_config(
            {},
            enable_chip_swimlane=0,
            dump_args=0,
            enable_pmu=0,
            enable_dep_gen=False,
            enable_scope_stats=False,
            output_prefix="",
        )
        handles = []
        for run_index in range(depth):
            args = _task_args(specs, signature, common, per_run[run_index])

            def submit_one(orch, _args, cfg, task_args=args):
                orch.submit_next_level(chip_handle, task_args, cfg, worker=0)

            handles.append(worker.submit(submit_one, args=None, config=config))
        for handle in handles:
            handle.wait()
        if golden is not None:
            for run_buffers in per_run:
                _compare_run(base, run_buffers, golden)
    finally:
        worker.close()
    print(f"[qwen-worker-submit] PASSED depth={depth} device={device_ids[0]}", flush=True)
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("-d", "--device", default="0")
    parser.add_argument("--depth", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seq-len", type=int, default=3500)
    parser.add_argument("--skip-golden", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return run(
        device_range_to_list(args.device),
        depth=args.depth,
        seed=args.seed,
        seq_len=args.seq_len,
        skip_golden=args.skip_golden,
        compile_only=args.compile_only,
    )


if __name__ == "__main__":
    sys.exit(main())
