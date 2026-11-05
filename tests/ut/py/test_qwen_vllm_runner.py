# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Engine runner preserves generation and shutdown outcomes."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

CASE = Path(__file__).resolve().parents[3] / "examples/a2a3/host_build_graph/qwen3_14b_serving_effective"


@pytest.mark.parametrize("requests", [1, 16])
@pytest.mark.parametrize("failure", [None, "generate", "close", "both"])
def test_runner_receipt_preserves_primary_and_always_shuts_down(monkeypatch, tmp_path, requests, failure):
    monkeypatch.syspath_prepend(str(CASE))
    for name in ("ASCEND_RT_VISIBLE_DEVICES", "VLLM_USE_V2_MODEL_RUNNER", "PYTHONPATH"):
        monkeypatch.setenv(name, "")
    spec = importlib.util.spec_from_file_location("qwen_vllm_runner_test", CASE / "run_vllm_eager.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    errors = [RuntimeError("generation failed"), RuntimeError("close failed")]
    events = []

    class LLM:
        def __init__(self, **kwargs):
            assert kwargs["max_num_seqs"] == requests
            self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda: events.append("shutdown")))

        def sleep(self, **kwargs):
            pass

        def enqueue(self, prompts, *args, **kwargs):
            assert len(prompts) == requests

        def wake_up(self, **kwargs):
            pass

        def wait_for_completion(self, **kwargs):
            if failure in ("generate", "both"):
                raise errors[0]
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1, 2])]) for _ in range(requests)]

        def collective_rpc(self, method):
            if method == "simpler_stats":
                return [{"steps": 1, "failed": False}]
            events.append("close")
            if failure in ("close", "both"):
                raise errors[1]

    vllm = ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams = LLM, lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setattr(importlib.import_module("torch_decode"), "validate_prompt_ids", lambda _: None)
    monkeypatch.setattr(
        importlib.import_module("validate_real_inputs"), "validate_checkpoint", lambda _: {"model": "hash"}
    )
    prompt = tmp_path / "ids.json"
    prompt.write_text("[1]")
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--model",
            str(tmp_path),
            "--artifact",
            str(tmp_path),
            "--prompt-ids",
            str(prompt),
            "--device",
            "0",
            "--max-tokens",
            "2",
            "--requests",
            str(requests),
            "--output",
            str(output),
        ],
    )
    if failure is not None:
        with pytest.raises(RuntimeError) as caught:
            module.main()
        assert caught.value is errors[0 if failure in ("generate", "both") else 1]
    else:
        assert module.main() == 0
    receipt = json.loads(output.read_text())
    assert events == ["close", "shutdown"]
    assert receipt["closed"] and receipt["status"] == ("passed" if failure is None else "failed")
    assert receipt["model_sha256"] == {"model": "hash"}
    if failure in ("generate", "both"):
        assert receipt["error"] == "RuntimeError: generation failed"
    if failure in ("close", "both"):
        assert receipt["cleanup_error"] == "RuntimeError: close failed"
