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


def test_fixed_contract_accepted(adapter, config):
    module, _ = adapter
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


def test_framework_logits_and_native_prefill_routes(adapter):
    module, context = adapter
    model = module.SimplerQwen3ForCausalLM(vllm_config=SimpleNamespace(), prefix="")
    model.config = SimpleNamespace(vocab_size=3)
    logits = torch.arange(10).reshape(2, 5)
    model.simpler_logits = True
    torch.testing.assert_close(model.compute_logits(logits), logits[:, :3])
    assert not model.simpler_logits
    torch.testing.assert_close(model.compute_logits(logits), logits + 1)
    assert model.forward(None, None) == "native-prefill"
    reset = []
    model.simpler = SimpleNamespace(reset_cohort=lambda: reset.append(True))
    context.attn_metadata = {"attention": SimpleNamespace(num_prefills=16, num_decodes=0)}
    assert model.forward(torch.ones(16 * 3338, dtype=torch.int64), None) == "native-prefill"
    assert reset == [True]
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
