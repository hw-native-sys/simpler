# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Share packed projection storage between native vLLM prefill and Simpler decode."""

from pathlib import Path

import torch
from weights import HIDDEN, PADDED_VOCAB, CheckpointReader, _stack_layers
from weights import NUM_LAYERS as LAYERS


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
        if len(names) != len(sizes):
            raise ValueError("projection names and sizes differ in length")
        for part, name in enumerate(names):
            start, count = sum(sizes[:part]), sizes[part]
            weight = getattr(getattr(layers[0], parent_name), projection_name).weight
            packed[name] = weight.new_empty((LAYERS * weight.shape[1], count))
            for index, layer in enumerate(layers):
                weight = getattr(getattr(layer, parent_name), projection_name).weight
                packed[name].view(LAYERS, -1, count)[index].copy_(weight[start : start + count].T)
            del weight
        for index, layer in enumerate(layers):
            views = [packed[name].view(LAYERS, -1, count)[index] for name, count in zip(names, sizes)]
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
    device = value.device
    del value, padding
    reader = CheckpointReader(Path(model.vllm_config.model_config.model))
    for name, suffix in (
        ("input_rms_weight", "input_layernorm.weight"),
        ("post_rms_weight", "post_attention_layernorm.weight"),
        ("q_norm_weight", "self_attn.q_norm.weight"),
        ("k_norm_weight", "self_attn.k_norm.weight"),
    ):
        packed[name] = _stack_layers(reader, suffix, dtype=torch.float32, transpose=False).to(device)
    packed["final_norm_weight"] = reader.load("model.norm.weight").view(1, -1).float().to(device)
    return packed
