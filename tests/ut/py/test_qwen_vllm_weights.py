# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

CASE = Path(__file__).resolve().parents[3] / "examples/a2a3/host_build_graph/qwen3_14b_serving_effective"


@pytest.fixture
def packing(monkeypatch):
    monkeypatch.syspath_prepend(str(CASE))
    spec = importlib.util.spec_from_file_location("qwen_vllm_weights_test", CASE / "vllm_weights.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_prefill_linear_borrows_input_major_weights(packing):
    bank = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
    other = torch.arange(3 * 2, dtype=torch.float32).reshape(3, 2)
    linear = packing.PackedLinear(bank[1], other)
    inputs = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    expected = torch.cat(
        [torch.nn.functional.linear(inputs, bank[1].T), torch.nn.functional.linear(inputs, other.T)], -1
    )
    actual, bias = linear(inputs)
    torch.testing.assert_close(actual, expected)
    assert bias is None
    assert linear.weight_0.untyped_storage().data_ptr() == bank.untyped_storage().data_ptr()
    bank[1].zero_()
    assert torch.count_nonzero(linear(inputs)[0][:, :5]) == 0


def test_pack_model_weights_reuses_loaded_projection_storage(packing, monkeypatch):
    class Projection(torch.nn.Module):
        def __init__(self, rows, columns, value):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.full((rows, columns), value, dtype=torch.bfloat16))

    class Attention(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.q_size, self.kv_size = 4, 2
            self.qkv_proj = Projection(8, 3, value)
            self.o_proj = Projection(3, 3, value + 1)

    class Mlp(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.gate_up_proj = Projection(10, 3, value + 2)
            self.down_proj = Projection(3, 5, value + 3)

    class Layer(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.self_attn, self.mlp = Attention(value), Mlp(value)

    class Embedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((6, 3), dtype=torch.bfloat16))

    class Reader:
        def __init__(self, _):
            pass

        def load(self, name):
            width = 2 if name.endswith((".q_norm.weight", ".k_norm.weight")) else 3
            return torch.ones(width)

    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=torch.nn.ModuleList([Layer(index) for index in range(40)]), embed_tokens=Embedding()
        ),
        lm_head=Embedding(),
        config=SimpleNamespace(vocab_size=6),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(model="checkpoint")),
    )
    monkeypatch.setattr(packing, "HIDDEN", 3)
    monkeypatch.setattr(packing, "PADDED_VOCAB", 7)
    monkeypatch.setattr(packing, "CheckpointReader", Reader)
    monkeypatch.setattr(
        packing.torch,
        "npu",
        SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None),
        raising=False,
    )

    packed = packing.pack_model_weights(model)
    assert packed["wq"].shape == (40 * 3, 4)
    assert packed["w_down"].shape == (40 * 5, 3)
    assert packed["embed_weight"].shape == (7, 3)
    assert all(not value.requires_grad for value in packed.values())
    assert all(isinstance(layer.self_attn.qkv_proj, packing.PackedLinear) for layer in model.model.layers)
    torch.testing.assert_close(model.model.layers[7].self_attn.qkv_proj.weight_0, packed["wq"].view(40, 3, 4)[7])
