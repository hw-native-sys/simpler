# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fixed-shape Qwen decode over caller-owned torch storage."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import torch
from weights import HEAD_DIM, HIDDEN, PADDED_VOCAB, CheckpointReader
from weights import NUM_LAYERS as LAYERS

BATCH, HEADS, PAGE = 16, 8, 128
_GENERATIONS = itertools.count(1)
PROMPT_SHA256 = "8db4a9a6897a5ae237fd3806ce39b4d49a549f8faf2881d5ae5740d5e6be6bb9"


def validate_prompt_ids(ids):
    if not isinstance(ids, list) or len(ids) != 3338 or any(type(value) is not int or value < 0 for value in ids):
        raise ValueError("prompt must contain exactly 3338 token IDs")
    if hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest() != PROMPT_SHA256:
        raise ValueError("prompt token IDs differ from the qualified workload")


class PackedLinear(torch.nn.Module):
    """Native prefill uses the same input-major weights as the decode callable."""

    def __init__(self, *weights):
        super().__init__()
        for index, weight in enumerate(weights):
            self.register_buffer(f"weight_{index}", weight, persistent=False)

    def forward(self, hidden):
        parts = [torch.matmul(hidden, weight) for weight in self.buffers()]
        return (parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)), None


@torch.no_grad()
def pack_model_weights(model):
    layers = model.model.layers
    packed = {}
    attention = layers[0].self_attn
    intermediate = layers[0].mlp.gate_up_proj.weight.shape[0] // 2
    groups = (
        ("self_attn", "qkv_proj", ("wq", "wk", "wv"), (attention.q_size, attention.kv_size, attention.kv_size)),
        ("self_attn", "o_proj", ("wo",), (HIDDEN,)),
        ("mlp", "gate_up_proj", ("w_gate", "w_up"), (intermediate, intermediate)),
        ("mlp", "down_proj", ("w_down",), (HIDDEN,)),
    )
    for parent_name, projection_name, names, sizes in groups:
        for part, name in enumerate(names):
            start, count = sum(sizes[:part]), sizes[part]
            weight = getattr(getattr(layers[0], parent_name), projection_name).weight
            packed[name] = weight.new_empty((LAYERS * weight.shape[1], count))
            for index, layer in enumerate(layers):
                weight = getattr(getattr(layer, parent_name), projection_name).weight
                packed[name].view(LAYERS, -1, count)[index].copy_(weight[start : start + count].T)
            del weight
        for index, layer in enumerate(layers):
            views = [packed[name].view(LAYERS, -1, count)[index] for name, count in zip(names, sizes, strict=True)]
            setattr(getattr(layer, parent_name), projection_name, PackedLinear(*views))
        # A projection group is released before conversion advances to the next group.
        torch.npu.synchronize()
        torch.npu.empty_cache()
    for name, module, repeat_padding in (
        ("embed_weight", model.model.embed_tokens, False),
        ("lm_head_weight", model.lm_head, True),
    ):
        value = module.weight[: model.config.vocab_size]
        missing = PADDED_VOCAB - value.shape[0]
        if missing < 0:
            raise ValueError("vocabulary exceeds the decode ABI")
        padding = value[:1].expand(missing, -1) if repeat_padding else value.new_zeros((missing, HIDDEN))
        packed[name] = torch.cat((value, padding)).contiguous()
        module.weight = torch.nn.Parameter(packed[name], requires_grad=False)
    reader = CheckpointReader(Path(model.vllm_config.model_config.model))
    for name, suffix in (
        ("input_rms_weight", "input_layernorm.weight"),
        ("post_rms_weight", "post_attention_layernorm.weight"),
        ("q_norm_weight", "self_attn.q_norm.weight"),
        ("k_norm_weight", "self_attn.k_norm.weight"),
    ):
        packed[name] = torch.cat(
            [reader.load(f"model.layers.{i}.{suffix}").view(1, -1).float() for i in range(LAYERS)]
        ).to(value.device)
    packed["final_norm_weight"] = reader.load("model.norm.weight").view(1, -1).float().to(value.device)
    return packed


def allocate_layer_caches(layer_names, *, pages, device):
    if len(layer_names) != LAYERS or len(set(layer_names)) != LAYERS or pages < BATCH * 28:
        raise ValueError("fixed Qwen decode requires 40 layers and at least 448 pages")
    caches = tuple(
        torch.zeros((LAYERS, pages, PAGE, HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device) for _ in range(2)
    )
    raw = {
        name: tuple(cache[index].view(torch.int8).flatten() for cache in caches)
        for index, name in enumerate(layer_names)
    }
    return caches, raw


class TorchDecode:
    def __init__(self, *, artifact: Path, packed, caches, device):
        from callable_bridge import inspect_artifact  # noqa: PLC0415
        from real_worker_submit import _dtype, _shape  # noqa: PLC0415
        from simpler.task_interface import CallConfig  # noqa: PLC0415

        self.artifact = inspect_artifact(artifact)
        if "sampled_ids_host" in self.artifact.parameter_names:
            raise ValueError("torch eager decode requires the 25-argument DEVICE-output ABI")
        self.device = torch.device(device)
        self.caches = caches
        self.worker = None
        self.run = None
        self.sources = []
        self.failed = False
        self.closed = False
        self.cohort = None
        self.positions = None
        self.blocks = None
        self.steps = 0
        self.config = CallConfig()
        self.config.enable_dep_gen = False
        self.config.enable_chip_swimlane = 0
        pages = caches[0].shape[1]
        expected = (LAYERS, pages, PAGE, HEADS, HEAD_DIM)
        for cache in caches:
            if tuple(cache.shape) != expected or not cache.is_contiguous() or cache.dtype != torch.bfloat16:
                raise ValueError("KV backing must be contiguous BF16 layer/BSND storage")
        self.values = {**packed, "k_cache": caches[0].view(-1, HEAD_DIM), "v_cache": caches[1].view(-1, HEAD_DIM)}
        specs = json.loads((artifact / "distributed_meta.json").read_text())["params"]
        for spec in specs:
            name = spec["name"].split("__ssa_", 1)[0]
            shape, dtype = _shape(name, spec["shape"], pages=pages), _dtype(spec["dtype"])
            if name not in self.values:
                self.values[name] = torch.zeros(shape, dtype=dtype, device=self.device)
            value = self.values[name]
            if tuple(value.shape) != shape or value.dtype != dtype or not value.is_contiguous():
                raise ValueError(f"tensor does not match callable ABI: {name}")
            if value.device != self.device:
                raise ValueError(f"tensor is on another device: {name}")

    def _initialize(self):
        from simpler.buffer import Buffer  # noqa: PLC0415
        from simpler.task_interface import ArgDirection, TaskArgs, Tensor, TensorArgType  # noqa: PLC0415
        from simpler.worker import Worker  # noqa: PLC0415

        from simpler_setup.torch_interop import torch_dtype_to_datatype  # noqa: PLC0415

        callable_ = self.artifact.compile()
        self.worker = Worker(level=2, device_id=self.device.index, platform="a2a3", runtime="host_build_graph")
        self.callable = self.worker.register(callable_)
        self.worker.init(borrowed_context_generation=next(_GENERATIONS))
        self.args = TaskArgs()
        signature = self.artifact.callable_spec["orchestration"]["signature"]
        for name, direction in zip(self.artifact.parameter_names, signature, strict=True):
            value = self.values[name]
            source = Buffer.wrap(
                address=value.data_ptr(),
                nbytes=value.numel() * value.element_size(),
                location=self.worker.device_location,
            )
            self.sources.append(source)
            access = TensorArgType.INPUT if direction == ArgDirection.IN else TensorArgType.OUTPUT_EXISTING
            if direction == ArgDirection.INOUT or name in ("k_cache", "v_cache"):
                access = TensorArgType.INOUT
            self.args.add_tensor(Tensor(source, shapes=value.shape, dtype=torch_dtype_to_datatype(value.dtype)), access)

    def reset_cohort(self):
        if self.failed or self.closed:
            raise RuntimeError("failed decode backend cannot admit another cohort")
        self.cohort = None
        self.positions = None
        self.blocks = None

    def __call__(self, input_ids, positions, metadata, request_ids):
        if self.failed or self.closed:
            raise RuntimeError("failed decode backend cannot admit another step")
        if tuple(input_ids.shape) != (BATCH,) or tuple(positions.shape) != (BATCH,):
            raise ValueError("fixed decode requires exactly 16 one-token rows")
        tokens = input_ids.tolist()
        if len(set(tokens)) != 1:
            raise ValueError("fixed decode requires clone-request token rows")
        identity = tuple(request_ids)
        if len(identity) != BATCH or len(set(identity)) != BATCH:
            raise ValueError("decode requires 16 distinct request identities")
        current = tuple(int(value) for value in positions.tolist())
        if self.cohort is None:
            if current != (3338,) * BATCH:
                raise ValueError("decode must follow the supported 3338-token prefill")
        elif (
            self.positions is None or identity != self.cohort or current != tuple(value + 1 for value in self.positions)
        ):
            raise ValueError("request replacement or nonconsecutive decode requires a drained cohort boundary")
        if max(current) >= 3338 + 127:
            raise ValueError("fixed decode exceeds the qualified generation range")
        table = metadata.block_tables
        if tuple(table.shape) != (BATCH, 32):
            raise ValueError("decode requires a 16 by 32 block table")
        lengths = tuple(int(value) for value in metadata.seq_lens.tolist())
        if lengths != tuple(value + 1 for value in current):
            raise ValueError("KV lengths do not include exactly the current decode token")
        rows = table.tolist()
        blocks = tuple(tuple(row[: (position + PAGE) // PAGE]) for row, position in zip(rows, current, strict=True))
        used = [page for row in blocks for page in row]
        pages = self.caches[0].shape[1]
        if len(set(used)) != len(used) or any(page < 0 or page >= pages for page in used):
            raise ValueError("request KV pages must be independent and within the backing capacity")
        if self.blocks is not None and any(
            row[: len(old)] != old for row, old in zip(blocks, self.blocks, strict=True)
        ):
            raise ValueError("live request KV pages cannot be replaced")
        slots = tuple(int(value) for value in metadata.slot_mapping.tolist())
        expected_slots = tuple(
            row[position // PAGE] * PAGE + position % PAGE for row, position in zip(rows, current, strict=True)
        )
        if slots != expected_slots:
            raise ValueError("slot mapping disagrees with request positions and KV pages")
        stream = torch.npu.current_stream(self.device)
        self.values["sampled_ids_in"][:, 0].copy_(input_ids)
        self.values["seq_lens"].copy_(metadata.seq_lens)
        self.values["block_table"].view(BATCH, 32).copy_(table)
        self.values["slot_mapping"].copy_(metadata.slot_mapping)
        # Flush the framework emission queue and finish prior consumers before reusing storage.
        stream.synchronize()
        try:
            if self.worker is None:
                self._initialize()
            assert self.worker is not None
            self.run = self.worker.submit(self.callable, self.args, self.config, caller_stream=stream.npu_stream)
            self.run.wait()
        except BaseException:
            self.failed = True
            raise
        self.cohort, self.positions = identity, current
        self.blocks = blocks
        self.steps += 1
        return self.values["out"]

    def close(self):
        if self.closed:
            return
        torch.npu.current_stream(self.device).synchronize()
        if self.run is not None:
            self.run.wait()
        if self.worker is not None:
            self.worker.close()
        for source in self.sources:
            source.close()
        self.closed = True
