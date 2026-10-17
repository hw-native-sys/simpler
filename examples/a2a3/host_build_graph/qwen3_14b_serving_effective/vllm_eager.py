# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B eager worker for vLLM 0.20.2 / vLLM-Ascend 0.20.2rc1."""

from __future__ import annotations

import importlib.metadata
import re
from contextlib import contextmanager
from pathlib import Path

import torch
from torch_decode import TorchDecode, allocate_layer_caches, pack_model_weights, validate_prompt_ids
from vllm.forward_context import get_forward_context
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.worker import NPUWorker
from weights import rope_tables


def validate_config(config):
    for package, expected in (("vllm", "0.20.2"), ("vllm-ascend", "0.20.2rc1")):
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise ValueError(f"{package} {actual} is outside the fixed eager contract ({expected})")
    model = config.model_config
    hf = model.hf_config
    expected = {"model_type": "qwen3", "num_hidden_layers": 40, "hidden_size": 5120, "num_key_value_heads": 8}
    if any(getattr(hf, key, None) != value for key, value in expected.items()):
        raise ValueError("eager backend requires unquantized Qwen3-14B geometry")
    parallel = config.parallel_config
    if any(
        getattr(parallel, name) != 1
        for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size")
    ):
        raise ValueError("eager backend requires single-device execution")
    if model.dtype != torch.bfloat16 or model.quantization is not None or config.speculative_config is not None:
        raise ValueError("eager backend requires BF16 without quantization or speculative decoding")
    if config.lora_config is not None or model.enable_sleep_mode:
        raise ValueError("fixed eager requires stable weights and context throughout its lifetime")
    if not model.enforce_eager or model.max_model_len != 4096:
        raise ValueError("eager backend requires enforce_eager and max_model_len=4096")
    if config.cache_config.enable_prefix_caching or config.kv_transfer_config is not None:
        raise ValueError("eager backend requires private local KV pages")
    if config.scheduler_config.enable_chunked_prefill or config.scheduler_config.async_scheduling:
        raise ValueError("eager backend requires unchunked prefill and synchronous scheduling")
    if config.scheduler_config.max_num_seqs != 16:
        raise ValueError("eager backend requires max_num_seqs=16")
    if (config.additional_config or {}).get("simpler", {}).get("artifact") is None:
        raise ValueError("additional_config.simpler.artifact is required")


class SimplerQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.simpler = None
        self.simpler_caches = None
        self.simpler_request_ids = ()
        self.simpler_logits = False

    def initialize_packed_weights(self, device):
        root = Path(self.vllm_config.model_config.model)
        self.simpler_weights = pack_model_weights(self)
        cos, sin = rope_tables(root)
        self.simpler_weights.update(rope_cos=cos.to(device), rope_sin=sin.to(device))
        torch.npu.synchronize(device)
        torch.npu.empty_cache()

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        context = get_forward_context()
        metadata = context.attn_metadata
        self.simpler_logits = False
        if metadata is None:
            return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)
        metadata = next(iter(metadata.values()))
        if metadata.num_prefills:
            if metadata.num_decodes:
                raise ValueError("mixed prefill/decode is outside the fixed eager contract")
            if metadata.num_prefills != 16 or input_ids is None or input_ids.numel() != 16 * 3338:
                raise ValueError("prefill requires 16 complete 3338-token requests")
            prompts = input_ids.reshape(16, 3338)
            if not torch.equal(prompts, prompts[:1].expand_as(prompts)):
                raise ValueError("fixed eager requires clone-request prompts")
            validate_prompt_ids(prompts[0].tolist())
            if self.simpler is not None:
                self.simpler.reset_cohort()
            return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)
        if intermediate_tensors is not None or inputs_embeds is not None:
            raise ValueError("fixed decode requires token IDs without pipeline inputs")
        if self.simpler is None:
            self.simpler = TorchDecode(
                artifact=Path(self.vllm_config.additional_config["simpler"]["artifact"]),
                packed=self.simpler_weights,
                caches=self.simpler_caches,
                device=input_ids.device,
            )
        result = self.simpler(input_ids, positions, metadata, self.simpler_request_ids)
        self.simpler_logits = True
        return result

    def compute_logits(self, hidden_states):
        if self.simpler_logits:
            self.simpler_logits = False
            return hidden_states[:, : self.config.vocab_size].contiguous()
        return super().compute_logits(hidden_states)


class SimplerModelRunner(NPUModelRunner):
    def load_model(self):
        ModelRegistry.register_model("Qwen3ForCausalLM", SimplerQwen3ForCausalLM)
        super().load_model()
        self.model.initialize_packed_weights(self.device)

    def _allocate_kv_cache_tensors(self, kv_cache_config):
        specs = self._get_layer_kv_cache_specs(kv_cache_config)
        names = sorted(specs, key=lambda name: int(re.search(r"\.layers\.(\d+)\.", name).group(1)))
        if len(names) != 40 or [int(re.search(r"\.layers\.(\d+)\.", name).group(1)) for name in names] != list(
            range(40)
        ):
            raise ValueError("fixed eager requires exactly one attention cache per Qwen layer")
        for spec in specs.values():
            if (spec.block_size, spec.num_kv_heads, spec.head_size, spec.dtype) != (128, 8, 128, torch.bfloat16):
                raise ValueError("KV specification differs from the BSND decode ABI")
        tensors = kv_cache_config.kv_cache_tensors
        if len(tensors) != 40 or any(len(item.shared_by) != 1 for item in tensors):
            raise ValueError("fixed eager requires independent per-layer KV allocations")
        if len({item.size for item in tensors}) != 1:
            raise ValueError("KV layers must have equal capacity")
        page_bytes = 2 * 128 * 8 * 128 * 2
        if tensors[0].size % page_bytes:
            raise ValueError("KV allocation is not page aligned")
        caches, raw = allocate_layer_caches(names, pages=tensors[0].size // page_bytes, device=self.device)
        self.model.simpler_caches = caches
        return raw

    def _model_forward(self, *args, **kwargs):
        if self.input_batch.num_reqs and not self.input_batch.sampling_metadata.all_greedy:
            raise ValueError("fixed eager requires greedy sampling")
        self.model.simpler_request_ids = tuple(self.input_batch.req_ids)
        return super()._model_forward(*args, **kwargs)


class SimplerWorker(NPUWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        validate_config(self.vllm_config)
        self.simpler_failed = False

    def init_device(self):
        from vllm.v1.worker.workspace import init_workspace_manager  # noqa: PLC0415

        self.device = self._init_device()
        init_workspace_manager(self.device, 1)
        self.model_runner = SimplerModelRunner(self.vllm_config, self.device)

    @contextmanager
    def _execution(self):
        if self.simpler_failed:
            raise RuntimeError("failed eager worker cannot admit another invocation")
        try:
            yield
        except BaseException:
            self.simpler_failed = True
            model = self.model_runner.model
            if model is not None and model.simpler is not None:
                model.simpler.failed = True
            raise

    def execute_model(self, *args, **kwargs):
        with self._execution():
            return super().execute_model(*args, **kwargs)

    def sample_tokens(self, *args, **kwargs):
        with self._execution():
            return super().sample_tokens(*args, **kwargs)

    def shutdown(self):
        self.close_simpler()
        super().shutdown()

    def close_simpler(self):
        model = self.model_runner.model
        if model is not None and model.simpler is not None:
            model.simpler.close()

    def simpler_stats(self):
        backend = self.model_runner.model.simpler
        return {"steps": 0 if backend is None else backend.steps, "failed": backend is not None and backend.failed}
