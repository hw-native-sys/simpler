# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real-prefill mapping, complete numerical evidence and failure receipts."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

CASE = Path(__file__).resolve().parents[3] / "examples/a2a3/host_build_graph/qwen3_14b_serving_effective"


@pytest.fixture
def validation(monkeypatch):
    monkeypatch.syspath_prepend(str(CASE))
    spec = importlib.util.spec_from_file_location("qwen_torch_numerics_test", CASE / "validate_torch_eager.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def complete_report(validation):
    names = [f"{kind}_{layer:02d}" for layer in range(validation.LAYERS) for kind in ("key", "value")]
    state = {
        "tokens": [7] * 128,
        "token_equal": True,
        "decode_dispatches": 127,
        "closed": True,
        "failed": False,
        "page_boundary_covered": True,
        "caller_stream_after_close": True,
        "prefill_and_tail": [
            {"tensor": name, "prefix_exact": True, "tail_zero": True, "finite": True} for name in names
        ],
    }
    common = copy.deepcopy(state)
    common.update(appended_kv={name: {"passed": True} for name in names}, logits=[{"passed": True}] * 127)
    return {"common_prefix": common, "free_generation": state}


@pytest.mark.parametrize("missing", ["common_prefix", "free_generation", "logits", "kv", "prefix", "tokens", "steps"])
def test_gate_rejects_missing_or_truncated_evidence(validation, complete_report, missing):
    assert validation.validation_passed(complete_report)
    common = complete_report["common_prefix"]
    if missing in ("common_prefix", "free_generation"):
        del complete_report[missing]
    elif missing == "logits":
        common["logits"].pop()
    elif missing == "kv":
        common["appended_kv"].pop("value_39")
    elif missing == "prefix":
        common["prefill_and_tail"].pop()
    elif missing == "tokens":
        complete_report["free_generation"]["tokens"].pop()
    else:
        common["decode_dispatches"] = 126
    assert not validation.validation_passed(complete_report)


@pytest.mark.parametrize("mode", ["common_prefix", "free_generation"])
@pytest.mark.parametrize("field", ["token_equal", "closed", "caller_stream_after_close", "page_boundary_covered"])
def test_gate_requires_both_modes_and_complete_lifecycle(validation, complete_report, mode, field):
    complete_report[mode][field] = False
    assert not validation.validation_passed(complete_report)


def test_gate_rejects_duplicate_layer_evidence_and_recorded_error(validation, complete_report):
    original = copy.deepcopy(complete_report)
    complete_report["common_prefix"]["prefill_and_tail"][-1] = complete_report["common_prefix"]["prefill_and_tail"][0]
    assert not validation.validation_passed(complete_report)
    original["common_prefix"]["error"] = "incomplete run"
    assert not validation.validation_passed(original)


def test_kv_gate_checks_each_step_and_rejects_shape_mismatch(validation):
    expected = torch.tensor([[[[10000.0, 0.0], [1.0, 0.0]]]])
    actual = torch.tensor([[[[10000.0, 0.0], [0.0, 1.0]]]])
    assert validation.compare(actual, expected)["passed"]
    result = validation.compare_kv(actual, expected)
    assert not result["passed"] and result["first_out_of_tolerance_decode"] == 2
    with pytest.raises(ValueError, match="shape"):
        validation.compare_kv(actual[:, :, :1], expected)


@pytest.fixture
def decode_case(validation, monkeypatch, tmp_path):
    for name, value in {"LAYERS": 2, "HEADS": 1, "HEAD_DIM": 1, "PADDED_VOCAB": 8, "VOCAB": 4}.items():
        monkeypatch.setattr(validation, name, value)
    stream = SimpleNamespace(npu_stream=123, synchronize=lambda: None)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_stream=lambda _: stream), raising=False)
    state = SimpleNamespace(primary=None, cleanup=None, replay=False, backend=None, fail_after=0)

    class Backend:
        def __init__(self, **kwargs):
            self.steps, self.closed, self.failed = 0, False, False
            self.cleanup_error = None
            self.output = torch.zeros((1, 8))
            state.backend = self

        def __call__(self, token, position, metadata, identities):
            if state.primary and self.steps == state.fail_after:
                self.failed = True
                raise state.primary
            self.steps += 1
            self.output[0, 3] = self.steps
            return self.output

        def close(self):
            self.cleanup_error = state.cleanup
            if state.cleanup:
                raise state.primary if state.replay else state.cleanup
            self.closed = True

    monkeypatch.setattr(validation, "TorchDecode", Backend)
    monkeypatch.setattr(validation, "allocate_layer_caches", lambda **kwargs: ())
    monkeypatch.setattr(validation, "load_prefill_kv", lambda *args: None)
    names = [f"{kind}_{layer:02d}" for layer in range(2) for kind in ("key", "value")]
    prefill = {name: torch.zeros((1, 1, 3338, 1), dtype=torch.bfloat16) for name in names}
    golden = {name: torch.zeros((1, 1, 127, 1), dtype=torch.bfloat16) for name in names}
    golden.update(tokens=torch.full((128,), 3, dtype=torch.int64), logits=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 128))
    records = [{"tensor": name, "prefix_exact": True, "tail_zero": True, "finite": True} for name in names]
    monkeypatch.setattr(validation, "inspect_cache", lambda *args: (records, {n: golden[n] for n in names}))
    return SimpleNamespace(
        state=state,
        kwargs=dict(
            artifact=tmp_path,
            packed={},
            prefill=prefill,
            golden=golden,
            device=torch.device("cpu"),
            output=tmp_path / "actual.safetensors",
            forced=True,
            receipt={},
        ),
    )


@pytest.mark.parametrize("forced", [True, False])
def test_decode_counts_prefill_token_and_snapshots_reusable_logits(validation, decode_case, forced):
    decode_case.kwargs["forced"] = forced
    result = validation.run_decode(**decode_case.kwargs)
    assert result["decode_dispatches"] == 127 and result["tokens"] == [3] * 128
    assert [row["position"] for row in result["positions"]] == list(range(3338, 3465))
    assert result["page_boundary_covered"] and result["closed"] and result["caller_stream_after_close"]
    logits = load_file(str(decode_case.kwargs["output"]))["logits"]
    assert torch.equal(logits[:, 3], torch.arange(1, 128, dtype=torch.float32))


@pytest.mark.parametrize("replay", [False, True, "close-first"])
def test_decode_preserves_primary_and_independent_cleanup(validation, decode_case, replay):
    primary, cleanup = RuntimeError("execution failed"), OSError("cleanup failed")
    decode_case.state.primary, decode_case.state.cleanup, decode_case.state.replay = primary, cleanup, replay
    if replay == "close-first":
        decode_case.state.primary = None
        inspect = validation.inspect_cache

        def finish(*args):
            if args[-1]:
                decode_case.state.primary = primary
            return inspect(*args)

        validation.inspect_cache = finish
    with pytest.raises(RuntimeError) as caught:
        validation.run_decode(**decode_case.kwargs)
    assert caught.value is primary
    receipt = decode_case.kwargs["receipt"]
    assert receipt["error"] == "RuntimeError: execution failed"
    assert receipt["cleanup_error"] == "OSError: cleanup failed"
    assert not receipt["closed"] and receipt["decode_dispatches"] == (127 if replay == "close-first" else 0)


@pytest.mark.parametrize("error", [RuntimeError("execution failed"), TimeoutError("completion unknown")])
def test_decode_failure_saves_only_completed_cpu_logits(validation, decode_case, error):
    decode_case.state.primary, decode_case.state.fail_after = error, 1
    with pytest.raises(type(error)) as caught:
        validation.run_decode(**decode_case.kwargs)
    assert caught.value is error
    values = load_file(str(decode_case.kwargs["output"]))
    assert set(values) == {"logits"} and values["logits"].shape == (1, 4)
    assert decode_case.kwargs["receipt"]["decode_dispatches"] == 1


def test_close_only_error_and_diagnostic_failure_cannot_pass(validation, decode_case, monkeypatch):
    cleanup = OSError("cleanup failed")
    decode_case.state.cleanup = cleanup

    def fail_save(*args):
        raise OSError("diagnostic storage failed")

    monkeypatch.setattr(validation, "save_file", fail_save)
    with pytest.raises(OSError) as caught:
        validation.run_decode(**decode_case.kwargs)
    assert caught.value is cleanup
    receipt = decode_case.kwargs["receipt"]
    assert receipt["cleanup_error"] == "OSError: cleanup failed"
    assert receipt["diagnostic_error"] == "OSError: diagnostic storage failed"


@pytest.mark.parametrize("invalid", ["layer", "steps", "logits", "token-range", "nonfinite"])
def test_reference_refuses_incomplete_or_invalid_tensors(validation, decode_case, invalid):
    prefill, golden = decode_case.kwargs["prefill"], decode_case.kwargs["golden"]
    validation.validate_reference(prefill, golden)
    if invalid == "layer":
        del golden["value_01"]
    elif invalid == "steps":
        golden["value_01"] = golden["value_01"][:, :, :126]
    elif invalid == "logits":
        golden["logits"] = golden["logits"][:127]
    elif invalid == "token-range":
        golden["tokens"][0] = 4
    else:
        golden["logits"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        validation.validate_reference(prefill, golden)


def test_reference_snapshots_prefill_and_reused_logits(validation, monkeypatch, tmp_path):
    for name, value in {"LAYERS": 2, "HEADS": 1, "HEAD_DIM": 1, "VOCAB": 4, "TOKENS": 4}.items():
        monkeypatch.setattr(validation, name, value)

    class Model:
        def __init__(self):
            self.length, self.calls = 0, 0
            self.logits = torch.zeros(1, 1, 4)

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, tokens, **kwargs):
            self.length += tokens.shape[1]
            self.calls += 1
            self.logits[0, 0, 3] = self.calls
            value = torch.full((1, 1, self.length, 1), self.calls, dtype=torch.bfloat16)
            cache = SimpleNamespace(to_legacy_cache=lambda: ((value, value),) * 2)
            return SimpleNamespace(logits=self.logits, past_key_values=cache)

    model = Model()
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *args, **kwargs: model)),
    )
    monkeypatch.setattr(validation.importlib.metadata, "version", lambda _: "4.57.1")
    prefill, golden = validation.generate_reference(tmp_path, [1, 2], tmp_path, torch.device("cpu"))
    assert model.calls == 4 and golden["tokens"].tolist() == [3] * 4
    assert golden["logits"][:, 3].tolist() == [1, 2, 3, 4]
    assert all(value.shape == (1, 1, 2, 1) and bool((value == 1).all()) for value in prefill.values())
    assert all(golden[name].shape == (1, 1, 3, 1) for name in validation.kv_names())
    assert set(load_file(str(tmp_path / "prefill.safetensors"))) == set(validation.kv_names())


@pytest.mark.parametrize("invalid", [None, "missing-aux", "tampered", "escape", "duplicate", "revision", "single-file"])
def test_input_guards_authenticate_consumed_assets(validation, monkeypatch, tmp_path, invalid):
    contents = {"config.json": b"config", "weights.safetensors": b"weights", "tokenizer_config.json": b"aux"}
    for name, value in contents.items():
        (tmp_path / name).write_bytes(value)
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}
    monkeypatch.setattr(
        validation,
        "validate_checkpoint",
        lambda _: {name: hashes[name] for name in ("config.json", "weights.safetensors")},
    )
    manifest = {
        "revision": validation.MODEL_REVISION,
        "small_files": {name: {"sha256": hashes[name]} for name in ("config.json", "tokenizer_config.json")},
        "shards": [{"path": "weights.safetensors", "sha256": hashes["weights.safetensors"]}],
    }
    if invalid == "missing-aux":
        del manifest["small_files"]["tokenizer_config.json"]
    elif invalid == "tampered":
        (tmp_path / "tokenizer_config.json").write_text("changed")
    elif invalid == "escape":
        manifest["small_files"]["../outside"] = {"sha256": "digest"}
    elif invalid == "duplicate":
        manifest["shards"].append(manifest["shards"][0])
    elif invalid == "revision":
        manifest["revision"] = "other"
    elif invalid == "single-file":
        (tmp_path / "model.safetensors").write_bytes(b"alternate weights")
    path = tmp_path.parent / (tmp_path.name + ".json")
    path.write_text(json.dumps(manifest))
    if invalid:
        with pytest.raises(ValueError):
            validation.verify_inputs(tmp_path, path)
    else:
        assert validation.verify_inputs(tmp_path, path)["files"] == hashes


def test_main_records_preflight_error_before_device_work(validation, monkeypatch, tmp_path):
    prompt = tmp_path / "prompt.json"
    prompt.write_text("[]")
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validator",
            "--artifact",
            str(tmp_path),
            "--model",
            str(tmp_path),
            "--prompt-ids",
            str(prompt),
            "--input-manifest",
            str(tmp_path / "inputs.json"),
            "--device",
            "0",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="3338"):
        validation.main()
    receipt = json.loads((output / "result.json").read_text())
    assert receipt["status"] == "failed" and "3338" in receipt["error"]


def test_receipt_storage_failure_preserves_primary(validation, monkeypatch, tmp_path):
    prompt = tmp_path / "prompt.json"
    prompt.write_text("[]")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validator",
            "--artifact",
            str(tmp_path),
            "--model",
            str(tmp_path),
            "--prompt-ids",
            str(prompt),
            "--input-manifest",
            str(tmp_path / "inputs.json"),
            "--device",
            "0",
            "--output",
            str(tmp_path / "run"),
        ],
    )
    primary = RuntimeError("invalid input")

    def fail(*args, **kwargs):
        raise primary

    def fail_write(*args, **kwargs):
        raise OSError("receipt storage failed")

    monkeypatch.setattr(validation, "validate_prompt_ids", fail)
    monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises(RuntimeError) as caught:
        validation.main()
    assert caught.value is primary
