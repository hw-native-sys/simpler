# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from simpler.task_interface import CallConfig, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.torch_interop import torch_dtype_to_datatype

BATCH = 16
HEADS = 8
PAGE = 128
HEAD_DIM = 128
LAYERS = 40
PADDED_VOCAB = 152064
HIDDEN = 5120


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _view(buffer, shape, dtype):
    if buffer.shm is None:
        raise RuntimeError("host buffer has no shared backing")
    return torch.frombuffer(buffer.shm.buf, dtype=dtype, count=int(torch.tensor(shape).prod())).reshape(shape)


def _host_buffer(worker, tensor: torch.Tensor):
    out = worker.create_buffer(tensor.numel() * tensor.element_size())
    _view(out, tuple(tensor.shape), tensor.dtype).copy_(tensor)
    return out


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "bfloat16": torch.bfloat16, "int32": torch.int32}[name]


def _shape(name: str, shape: list[int], *, pages: int) -> tuple[int, ...]:
    if all(value >= 0 for value in shape):
        return tuple(shape)
    base = name.split("__ssa_", 1)[0]
    return {
        "seq_lens": (BATCH,),
        "block_table": (BATCH * 32,),
        "slot_mapping": (BATCH,),
        "rope_cos": (4096, HEAD_DIM),
        "rope_sin": (4096, HEAD_DIM),
        "k_cache": (LAYERS * pages * HEADS * PAGE, HEAD_DIM),
        "v_cache": (LAYERS * pages * HEADS * PAGE, HEAD_DIM),
        "out": (BATCH, PADDED_VOCAB),
        "sampled_ids_in": (BATCH, 8),
        "sampled_ids": (BATCH, 8),
        "next_hidden": (BATCH, HIDDEN),
    }[base]


def _make_kv(fixture, kind: str, pages: int) -> torch.Tensor:
    rows_per_layer = pages * HEADS * PAGE
    result = torch.zeros((LAYERS * rows_per_layer, HEAD_DIM), dtype=torch.bfloat16)
    for layer, path, tensor_name in fixture.iter_kv_shards(kind):
        shard = load_file(str(path), device="cpu")[tensor_name]
        start = layer * rows_per_layer
        result[start : start + rows_per_layer].view(pages, HEADS, PAGE, HEAD_DIM).index_copy_(
            0, fixture.metadata["used_page_ids"].to(torch.long), shard
        )
    return result


def _bind(specs, buffers, *, directions, cache_pages):
    args = TaskArgs()
    for spec in specs:
        name = spec["name"].split("__ssa_", 1)[0]
        dtype = _dtype(spec["dtype"])
        shape = _shape(name, spec["shape"], pages=cache_pages)
        args.add_tensor(buffers[name].tensor(shape, torch_dtype_to_datatype(dtype).value), directions[name])
    return args


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    case = args.repo / "examples/a2a3/host_build_graph/qwen3_14b_serving_effective"
    fixture_mod = _load(case / "fixture.py", "qwen_real_fixture")
    weights_mod = _load(case / "weights.py", "qwen_real_weights")
    bridge = _load(case / "callable_bridge.py", "qwen_real_bridge")
    fixture = fixture_mod.load_fixture(args.fixture)
    fixture.verify_external_inputs(args.model)
    golden = fixture.load_golden()
    inspected = bridge.inspect_artifact(args.artifact)
    chip = inspected.compile()
    metadata = json.loads((args.artifact / "distributed_meta.json").read_text())
    specs = metadata["params"]
    directions = {
        p["name"].split("__ssa_", 1)[0]: (
            TensorArgType.INPUT if p["direction"] == "In" else TensorArgType.OUTPUT_EXISTING
        )
        for p in specs
    }
    pages = int(fixture.manifest["physical_layout"]["num_pages"])
    host_values = {}
    for name, tensor in weights_mod.iter_kernel_weights(args.model):
        host_values[name] = tensor
    cos, sin = weights_mod.rope_tables(args.model)
    host_values["rope_cos"], host_values["rope_sin"] = cos, sin
    host_values["k_cache"] = _make_kv(fixture, "key", pages)
    host_values["v_cache"] = _make_kv(fixture, "value", pages)
    host_values["seq_lens"] = fixture.metadata["seq_lens_after_first_token"].clone()
    host_values["block_table"] = fixture.metadata["block_table"].reshape(-1).clone()
    host_values["slot_mapping"] = fixture.metadata["next_slot_mapping"].clone()
    host_values["out"] = torch.zeros((BATCH, PADDED_VOCAB), dtype=torch.float32)
    host_values["sampled_ids_in"] = torch.zeros((BATCH, 8), dtype=torch.int32)
    host_values["sampled_ids"] = torch.zeros((BATCH, 8), dtype=torch.int32)
    host_values["next_hidden"] = torch.zeros((BATCH, HIDDEN), dtype=torch.bfloat16)

    worker = Worker(
        level=3,
        platform="a2a3",
        runtime="host_build_graph",
        device_ids=[args.device],
        num_sub_workers=0,
        launch_depth=1,
    )
    chip_handle = worker.register(chip)
    worker.init()
    host_buffers = {}
    device_buffers = {}
    try:
        # L3 create_buffer values are host-visible shared memory. The generated Qwen
        # artifact consumes device addresses, so allocate one child-device buffer per
        # ABI argument and transfer the initial values before the first dispatch.
        host_buffers = {name: _host_buffer(worker, tensor) for name, tensor in host_values.items()}
        adapter_mod = _load(case / "standalone_adapter.py", "qwen_real_adapter")
        adapter = adapter_mod.StandaloneDecodeAdapter(fixture, golden)
        config = CallConfig()
        config.enable_dep_gen = False
        config.enable_chip_swimlane = 0

        def ensure_device_buffers(orch):
            if device_buffers:
                return
            for spec in specs:
                name = spec["name"].split("__ssa_", 1)[0]
                dtype = _dtype(spec["dtype"])
                shape = _shape(name, spec["shape"], pages=pages)
                device = orch.alloc_child_tensor(0, shape, torch_dtype_to_datatype(dtype).value)
                orch.copy_to(device, host_buffers[name])
                device_buffers[name] = device

        dynamic_names = ("sampled_ids_in", "seq_lens", "block_table", "slot_mapping", "sampled_ids")
        for _ in range(args.steps):
            step = adapter.next_step()
            _view(host_buffers["sampled_ids_in"], (BATCH, 8), torch.int32).zero_()
            _view(host_buffers["sampled_ids_in"], (BATCH, 8), torch.int32)[:, 0].copy_(step.input_token_ids)
            _view(host_buffers["seq_lens"], (BATCH,), torch.int32).copy_(step.seq_lens)
            _view(host_buffers["slot_mapping"], (BATCH,), torch.int32).copy_(step.slot_mapping)
            _view(host_buffers["block_table"], (BATCH * 32,), torch.int32).copy_(step.block_table.reshape(-1))
            _view(host_buffers["sampled_ids"], (BATCH, 8), torch.int32).zero_()
            if device_buffers:
                for name in dynamic_names:
                    worker.copy_to(device_buffers[name], host_buffers[name])

            task_args = None

            def submit_next_level(orch, _args, cfg):
                nonlocal task_args
                ensure_device_buffers(orch)
                task_args = _bind(specs, device_buffers, directions=directions, cache_pages=pages)
                orch.submit_next_level(chip_handle, task_args, cfg, worker=0)

            handle = worker.submit(submit_next_level, args=None, config=config)
            handle.wait()
            worker.copy_from(host_buffers["sampled_ids"], device_buffers["sampled_ids"])
            sampled = _view(host_buffers["sampled_ids"], (BATCH, 8), torch.int32).clone()
            adapter.complete_step(sampled)
            print(
                f"step={step.index} token0={int(sampled[0, 0])} expected0={int(step.expected_output_token_ids[0])}",
                flush=True,
            )
    finally:
        # close() also tears down child allocations after the child exits; explicit
        # release is best effort so a poisoned device does not mask the root failure.
        for device in reversed(list(device_buffers.values())):
            try:
                worker.free(device)
            except Exception:
                pass
        worker.close()
    print(f"PASS real Worker.submit steps={adapter.completed_steps} device={args.device}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
