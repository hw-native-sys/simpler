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
def adapter(monkeypatch):
    monkeypatch.syspath_prepend(str(CASE))
    spec = importlib.util.spec_from_file_location("qwen_torch_decode_test", CASE / "torch_decode.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_prefill_linear_borrows_input_major_weights(adapter):
    bank = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
    other = torch.arange(3 * 2, dtype=torch.float32).reshape(3, 2)
    linear = adapter.PackedLinear(bank[1], other)
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


def test_pack_model_weights_reuses_loaded_projection_storage(adapter, monkeypatch):
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
            width = 2 if name.endswith(('.q_norm.weight', '.k_norm.weight')) else 3
            return torch.ones(width)

    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=torch.nn.ModuleList([Layer(index) for index in range(40)]), embed_tokens=Embedding()
        ),
        lm_head=Embedding(),
        config=SimpleNamespace(vocab_size=6),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(model="checkpoint")),
    )
    monkeypatch.setattr(adapter, "HIDDEN", 3)
    monkeypatch.setattr(adapter, "PADDED_VOCAB", 7)
    monkeypatch.setattr(adapter, "CheckpointReader", Reader)
    monkeypatch.setattr(
        adapter.torch,
        "npu",
        SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None),
        raising=False,
    )

    packed = adapter.pack_model_weights(model)
    assert packed["wq"].shape == (40 * 3, 4)
    assert packed["w_down"].shape == (40 * 5, 3)
    assert packed["embed_weight"].shape == (7, 3)
    assert all(isinstance(layer.self_attn.qkv_proj, adapter.PackedLinear) for layer in model.model.layers)
    torch.testing.assert_close(model.model.layers[7].self_attn.qkv_proj.weight_0, packed["wq"].view(40, 3, 4)[7])


def test_kv_views_share_backing_and_isolate_layers(adapter, monkeypatch):
    for name, value in {"LAYERS": 2, "BATCH": 2, "PAGE": 2, "HEADS": 2, "HEAD_DIM": 4}.items():
        monkeypatch.setattr(adapter, name, value)
    caches, raw = adapter.allocate_layer_caches(["first", "second"], pages=56, device="cpu")
    raw["second"][0].view(torch.bfloat16)[0] = 7
    assert caches[0][1, 0, 0, 0, 0] == 7
    assert torch.count_nonzero(caches[0][0]) == 0
    assert torch.count_nonzero(caches[1]) == 0
    assert raw["first"][0].untyped_storage().data_ptr() == raw["second"][0].untyped_storage().data_ptr()
    with pytest.raises(ValueError, match="40 layers"):
        adapter.allocate_layer_caches(["same", "same"], pages=56, device="cpu")


@pytest.fixture
def backend(adapter, monkeypatch):
    calls = []
    stream = SimpleNamespace(npu_stream=123, synchronize=lambda: calls.append("framework-drained"))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_stream=lambda device: stream), raising=False)

    class Run:
        def wait(self):
            calls.append("retired")

    class Worker:
        def submit(self, *args, caller_stream):
            assert caller_stream == 123
            calls.append("submitted")
            return Run()

        def close(self):
            calls.append("worker-closed")

    obj = adapter.TorchDecode.__new__(adapter.TorchDecode)
    obj.failed, obj.cohort, obj.positions, obj.blocks, obj.steps = False, None, None, None, 0
    obj.closed = False
    obj.caches = [torch.empty((40, 512), device="meta")]
    obj.device, obj.worker, obj.run = torch.device("cpu"), Worker(), None
    obj.callable, obj.args, obj.config = object(), object(), object()
    obj.sources = [SimpleNamespace(close=lambda: calls.append("source-closed"))]
    obj.values = {
        "sampled_ids_in": torch.zeros((16, 8), dtype=torch.int32),
        "seq_lens": torch.zeros(16, dtype=torch.int32),
        "block_table": torch.zeros(16 * 32, dtype=torch.int32),
        "slot_mapping": torch.zeros(16, dtype=torch.int32),
        "out": torch.arange(16 * 7, dtype=torch.float32).reshape(16, 7),
    }
    metadata = SimpleNamespace(
        seq_lens=torch.full((16,), 3339, dtype=torch.int32),
        block_tables=torch.arange(16 * 32, dtype=torch.int32).reshape(16, 32),
        slot_mapping=torch.arange(16, dtype=torch.int32) * 32 * 128 + 3338,
    )
    return obj, metadata, calls


def test_real_sampled_input_and_ordering(backend):
    obj, metadata, calls = backend
    identities = [f"request-{index}" for index in range(16)]
    ids = torch.full((16,), 32313)
    result = obj(ids, torch.full((16,), 3338), metadata, identities)
    assert result is obj.values["out"]
    torch.testing.assert_close(obj.values["sampled_ids_in"][:, 0], ids.int())
    assert calls == ["framework-drained", "submitted", "retired"]
    actual_sample = result.argmax(-1)
    metadata.seq_lens += 1
    metadata.slot_mapping += 1
    obj(actual_sample, torch.full((16,), 3339), metadata, identities)
    torch.testing.assert_close(obj.values["sampled_ids_in"][:, 0], actual_sample.int())
    assert obj.steps == 2
    obj.close()
    assert calls[-3:] == ["retired", "worker-closed", "source-closed"]


@pytest.mark.parametrize(
    "change",
    ["identity", "position", "batch", "duplicate", "table", "range", "shared-page", "moved-page", "length", "slot"],
)
def test_unsupported_transition_refused_before_submission(backend, change):
    obj, metadata, calls = backend
    identities = list(range(16))
    obj(torch.ones(16), torch.full((16,), 3338), metadata, identities)
    calls.clear()
    positions = torch.full((16,), 3339)
    metadata.seq_lens += 1
    metadata.slot_mapping += 1
    if change == "identity":
        identities[0] = 99
    elif change == "position":
        positions[0] += 1
    elif change == "batch":
        positions = positions[:15]
    elif change == "duplicate":
        identities[0] = identities[1]
    elif change == "table":
        metadata.block_tables = metadata.block_tables[:, :31]
    elif change == "range":
        positions.fill_(3465)
        obj.positions = (3464,) * 16
    elif change == "shared-page":
        metadata.block_tables[1, 0] = metadata.block_tables[0, 0]
    elif change == "moved-page":
        metadata.block_tables[0, 0] = 511
    elif change == "length":
        metadata.seq_lens[0] -= 1
    elif change == "slot":
        metadata.slot_mapping[0] += 128
    with pytest.raises(ValueError):
        obj(torch.ones(16), positions, metadata, identities)
    assert calls == []
    assert obj.steps == 1


def test_failed_run_retains_sources_and_stops_admission(backend):
    obj, metadata, calls = backend

    def fail():
        calls.append("wait-failed")
        raise RuntimeError("completion unproven")

    obj.worker.submit = lambda *args, **kwargs: SimpleNamespace(wait=fail)
    with pytest.raises(RuntimeError, match="unproven"):
        obj(torch.ones(16), torch.full((16,), 3338), metadata, range(16))
    assert obj.failed
    with pytest.raises(RuntimeError, match="failed decode"):
        obj.reset_cohort()
    with pytest.raises(RuntimeError, match="failed decode"):
        obj(torch.ones(16), torch.full((16,), 3338), metadata, range(16))
    with pytest.raises(RuntimeError, match="unproven"):
        obj.close()
    assert "source-closed" not in calls and "worker-closed" not in calls
