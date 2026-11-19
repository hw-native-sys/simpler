# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

CASE = Path(__file__).resolve().parents[3] / "examples/a2a3/host_build_graph/qwen3_14b_serving_effective"


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.syspath_prepend(str(CASE))

    class Qwen(torch.nn.Module):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__()
            self.vllm_config = vllm_config

        def forward(self, *args):
            return "native-prefill"

        def compute_logits(self, value):
            return value + 1

    forward_context = SimpleNamespace(attn_metadata=None)
    exports = {
        "vllm.forward_context": {"get_forward_context": lambda: forward_context},
        "vllm.model_executor.models": {"ModelRegistry": object()},
        "vllm.model_executor.models.qwen3": {"Qwen3ForCausalLM": Qwen},
        "vllm_ascend.worker.model_runner_v1": {"NPUModelRunner": object},
        "vllm_ascend.worker.worker": {"NPUWorker": object},
    }
    for name, values in exports.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("qwen_vllm_eager_test", CASE / "vllm_eager.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "validate_prompt_ids", lambda _: None)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "0.20.2" if name == "vllm" else "0.20.2rc1")
    return module, forward_context


@pytest.fixture
def config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="qwen3", num_hidden_layers=40, hidden_size=5120, num_key_value_heads=8
            ),
            dtype=torch.bfloat16,
            quantization=None,
            enforce_eager=True,
            max_model_len=4096,
            enable_sleep_mode=False,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=False, async_scheduling=False, max_num_seqs=16),
        speculative_config=None,
        kv_transfer_config=None,
        lora_config=None,
        additional_config={"simpler": {"artifact": "artifact"}},
    )


@pytest.mark.parametrize("requests", [1, 16])
def test_fixed_contract_accepted(adapter, config, requests):
    module, _ = adapter
    config.scheduler_config.max_num_seqs = requests
    module.validate_config(config)
    parameters = inspect.signature(module.SimplerQwen3ForCausalLM.__init__).parameters
    assert {"vllm_config", "prefix"} <= parameters.keys()


@pytest.mark.parametrize(
    ("section", "name", "value"),
    [
        ("model_config", "dtype", torch.float16),
        ("model_config", "enforce_eager", False),
        ("model_config", "enable_sleep_mode", True),
        ("parallel_config", "tensor_parallel_size", 2),
        ("cache_config", "enable_prefix_caching", True),
        ("scheduler_config", "enable_chunked_prefill", True),
        ("scheduler_config", "async_scheduling", True),
        ("scheduler_config", "max_num_seqs", 8),
    ],
)
def test_unsupported_config_refused(adapter, config, section, name, value):
    module, _ = adapter
    setattr(getattr(config, section), name, value)
    with pytest.raises(ValueError):
        module.validate_config(config)


@pytest.mark.parametrize("requests", [1, 16])
def test_framework_logits_and_native_prefill_routes(adapter, config, requests):
    module, context = adapter
    config.scheduler_config.max_num_seqs = requests
    model = module.SimplerQwen3ForCausalLM(vllm_config=config, prefix="")
    model.config = SimpleNamespace(vocab_size=3)
    logits = torch.arange(10).reshape(2, 5)
    model.simpler_logits = True
    torch.testing.assert_close(model.compute_logits(logits), logits[:, :3])
    assert not model.simpler_logits
    torch.testing.assert_close(model.compute_logits(logits), logits + 1)
    assert model.forward(None, None) == "native-prefill"
    reset = []
    model.simpler = SimpleNamespace(close=lambda: reset.append(True))
    context.attn_metadata = {"attention": SimpleNamespace(num_prefills=requests, num_decodes=0)}
    assert model.forward(torch.ones(requests * 3338, dtype=torch.int64), None) == "native-prefill"
    assert reset == [True]
    assert model.simpler is None
    context.attn_metadata["attention"].num_decodes = 1
    with pytest.raises(ValueError, match="mixed"):
        model.forward(None, None)


def test_runner_preserves_each_layers_kv_views(adapter):
    module, _ = adapter
    runner = module.SimplerModelRunner()
    names = [f"model.layers.{i}.self_attn.attn" for i in range(40)]
    specs = {
        name: SimpleNamespace(block_size=128, num_kv_heads=8, head_size=128, dtype=torch.bfloat16) for name in names
    }
    runner._get_layer_kv_cache_specs = lambda _: specs
    runner.model = SimpleNamespace()
    runner.device = "meta"
    configuration = SimpleNamespace(
        kv_cache_tensors=[SimpleNamespace(shared_by=[name], size=449 * 2 * 128 * 8 * 128 * 2) for name in names]
    )
    raw = runner._allocate_kv_cache_tensors(configuration)
    assert list(raw) == names
    assert runner.model.simpler_caches[0].shape == (40, 449, 128, 8, 128)
    assert raw[names[0]][0].shape == (449 * 128 * 8 * 128 * 2,)
    configuration.kv_cache_tensors[0].shared_by.append(names[1])
    with pytest.raises(ValueError, match="independent"):
        runner._allocate_kv_cache_tensors(configuration)


@pytest.mark.parametrize("stored_rows", [15, 16, 17])
def test_cohort_decode_checks_private_pages_positions_and_slots(adapter, stored_rows):
    module, _ = adapter
    obj = module.VllmDecode.__new__(module.VllmDecode)
    obj.steps, obj.position, obj.blocks = 0, None, None
    obj.caches = [torch.empty((40, 449), device="meta")]
    table = torch.full((16, 32), -1, dtype=torch.int32)
    table[:, :28] = torch.arange(448).view(16, 28)
    metadata = SimpleNamespace(
        block_tables=table, seq_lens=torch.full((16,), 3339, dtype=torch.int32), slot_mapping=table[:, 26] * 128 + 10
    )
    ids = torch.ones(16, dtype=torch.int64)
    positions = torch.full((16,), 3338, dtype=torch.int64)
    _, result, position, blocks = obj._prepare_single_request(ids, positions, metadata, tuple(range(16)))
    assert result.block_tables.shape == (16, 28) and position == 3338 and len(blocks) == 16
    obj.steps, obj.position, obj.cohort = 1, position, tuple(range(16))
    obj.blocks = (blocks + blocks[:1])[:stored_rows]
    positions += 1
    metadata.seq_lens += 1
    metadata.slot_mapping += 1
    if stored_rows == 16:
        obj._prepare_single_request(ids, positions, metadata, tuple(range(16)))
    else:
        with pytest.raises(ValueError, match="pages"):
            obj._prepare_single_request(ids, positions, metadata, tuple(range(16)))
    metadata.block_tables[1, 0] = metadata.block_tables[0, 0]
    with pytest.raises(ValueError, match="private"):
        obj._prepare_single_request(ids, positions, metadata, tuple(range(16)))


def test_rope_uses_the_stage0_length(adapter, monkeypatch, config):
    module, _ = adapter
    model = module.SimplerQwen3ForCausalLM(vllm_config=config)
    config.model_config.model = "checkpoint"
    monkeypatch.setattr(module, "pack_model_weights", lambda _: {})
    cos = torch.ones((4096, 128))
    sin = torch.zeros_like(cos)
    monkeypatch.setattr(module, "rope_tables", lambda _: (cos, sin))
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(synchronize=lambda _: None, empty_cache=lambda: None), raising=False
    )
    model.initialize_packed_weights("cpu")
    assert model.simpler_weights["rope_cos"].shape == (3466, 128)
    assert model.simpler_weights["rope_sin"].shape == (3466, 128)


@pytest.mark.parametrize("requests", [1, 16])
def test_decode_maps_only_allocated_native_pages(adapter, config, requests):
    module, context = adapter
    config.scheduler_config.max_num_seqs = requests
    model = module.SimplerQwen3ForCausalLM(vllm_config=config)
    model.simpler_request_ids = tuple(range(requests))
    captured = []

    def decode(ids, positions, metadata, identities):
        captured.append(metadata.block_tables)
        assert identities == tuple(range(requests))
        return torch.ones((requests, 4))

    model.simpler = decode
    table = torch.zeros((requests, 32), dtype=torch.int32)
    table[:, :27] = torch.arange(requests * 27).view(requests, 27)
    context.attn_metadata = {
        "attention": SimpleNamespace(
            num_prefills=0,
            num_decodes=requests,
            block_tables=table,
            seq_lens=torch.full((requests,), 3339),
            slot_mapping=table[:, 26] * 128 + 10,
        )
    }
    out = model.forward(torch.ones(requests, dtype=torch.int64), torch.full((requests,), 3338))
    assert out.shape == (requests, 4) and model.simpler_steps == 1
    assert torch.equal(captured[0][:, :27], table[:, :27])
    assert captured[0].shape == (requests, 27 if requests == 1 else 28)
    if requests == 16:
        assert bool((captured[0][:, 27] == -1).all())


@pytest.mark.parametrize("configured,live", [(16, 1), (16, 15), (1, 16)])
def test_decode_refuses_partial_or_wrong_sized_group_before_dispatch(adapter, config, configured, live):
    module, context = adapter
    config.scheduler_config.max_num_seqs = configured
    model = module.SimplerQwen3ForCausalLM(vllm_config=config)
    model.simpler_request_ids = tuple(range(live))

    def unexpected_dispatch(*args):
        raise AssertionError("unsupported group reached decode")

    model.simpler = unexpected_dispatch
    context.attn_metadata = {
        "attention": SimpleNamespace(
            num_prefills=0,
            num_decodes=live,
            block_tables=torch.arange(live * 32, dtype=torch.int32).reshape(live, 32),
            seq_lens=torch.full((live,), 3339),
            slot_mapping=torch.zeros(live, dtype=torch.int32),
        )
    }
    with pytest.raises(ValueError, match="configured"):
        model.forward(torch.ones(live, dtype=torch.int64), torch.full((live,), 3338))
    assert model.simpler_steps == 0
