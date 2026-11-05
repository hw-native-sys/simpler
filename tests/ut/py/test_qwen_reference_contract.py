# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

CASE = Path(__file__).resolve().parents[3] / "examples/a2a3/host_build_graph/qwen3_14b_serving_effective"
spec = importlib.util.spec_from_file_location("qwen_reference_builder", CASE / "build_hbg_artifact.py")
assert spec is not None and spec.loader is not None
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_artifact_rejects_unrecognized_loop_instead_of_certifying_flat(tmp_path):
    child = tmp_path / "next_levels/decode_fwd"
    (child / "orchestration").mkdir(parents=True)
    (child / "kernel_config.py").write_text('\t"runtime": "tensormap_and_ringbuffer",\n')
    (child / "orchestration/decode_fwd.cpp").write_text(
        "void broken() { for (int layer_idx = 0; layer_idx < 40; ++layer_idx) {} }"
    )
    with pytest.raises(RuntimeError, match="cannot locate generated 40-layer loop"):
        builder._adapt_child_callable(tmp_path, 25)


def _reference(tmp_path):
    fixture_spec = importlib.util.spec_from_file_location("qwen_reference_fixture", CASE / "reference_fixture.py")
    assert fixture_spec is not None and fixture_spec.loader is not None
    module = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(module)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "qwen-reference-logical-kv-v1",
                "layers": 40,
                "kv_heads": 8,
                "head_dim": 128,
                "dtype": "bfloat16",
                "layout": "B,H,S,D",
                "prompt_tokens": 127,
            }
        )
    )
    (tmp_path / "validation.json").write_text(json.dumps({"reference_roundtrip_passed": True, "decode_steps": 2}))
    save_file({"first_token_id": torch.tensor([10])}, str(tmp_path / "metadata.safetensors"))
    save_file(
        {"decode_input_token_ids": torch.tensor([10, 11]), "decode_output_token_ids": torch.tensor([11, 12])},
        str(tmp_path / "decode.safetensors"),
    )
    values = torch.arange(8 * 127 * 128).reshape(1, 8, 127, 128).to(torch.bfloat16)
    save_file({"key": values, "value": -values}, str(tmp_path / "layer_00.safetensors"))
    (tmp_path / "SHA256SUMS").write_text(
        "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in sorted(tmp_path.iterdir()))
    )
    return module.ReferenceFixture, values


def test_reference_pages_cross_boundary_and_preserve_independent_rows(tmp_path):
    fixture_type, original = _reference(tmp_path)
    fixture = fixture_type(tmp_path)
    assert fixture.step(0)["seq_lens"][0] == 128
    assert fixture.step(1)["seq_lens"][0] == 129
    assert fixture.step(0)["slot_mapping"][0] == 127
    assert fixture.step(1)["slot_mapping"][0] == 128
    physical = dict(fixture.layer(0))["key"]
    for row in (0, 15):
        pages = fixture.block_table[row, :2].long()
        restored = physical[pages].permute(2, 0, 1, 3).reshape(1, 8, 256, 128)
        assert torch.equal(restored[:, :, :127], original)
        assert not restored[:, :, 127:].count_nonzero()
    physical[0].zero_()
    assert physical[int(fixture.block_table[1, 0])].count_nonzero()
    decoded = fixture.adapter_fixture()
    assert decoded.load_golden()["decode_input_token_ids"].shape == (2, 16)


def test_reference_rejects_changed_payload(tmp_path):
    fixture_type, _ = _reference(tmp_path)
    with (tmp_path / "layer_00.safetensors").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        fixture_type(tmp_path)


def _validator():
    validator_spec = importlib.util.spec_from_file_location("qwen_input_validator", CASE / "validate_real_inputs.py")
    assert validator_spec is not None and validator_spec.loader is not None
    module = importlib.util.module_from_spec(validator_spec)
    validator_spec.loader.exec_module(module)
    return module


def _model_dir(tmp_path):
    """A checkpoint whose four required files exist and carry the geometry the validator demands."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"num_hidden_layers": 40, "hidden_size": 5120}))
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    (model_dir / "tokenizer.json").write_text("{}")
    (model_dir / "generation_config.json").write_text(json.dumps({"do_sample": False}))
    return model_dir


def test_the_tokenizer_is_pinned_to_one_digest():
    """The prompt contract and the checkpoint file list name the same tokenizer.json.

    Two pins on one file: a re-pin that updates one and not the other leaves the
    validator accepting a checkpoint whose tokenizer the prompt pin rejects.
    """
    validator = _validator()
    assert validator.EXPECTED_PROMPT["tokenizer_sha256"] == validator.EXPECTED_MODEL_FILES["tokenizer.json"]


def test_every_pinned_checkpoint_file_is_compared(tmp_path, monkeypatch):
    """A mismatch in any one required file fails, and a missing file fails.

    Every entry in the pin set is compared, so no required file can pass unchecked —
    which is what a lookup keyed on anything other than the file's own name allows
    when the key it derives is absent.
    """
    validator = _validator()
    model_dir = _model_dir(tmp_path)
    pinned = {
        name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest() for name in validator.EXPECTED_MODEL_FILES
    }
    monkeypatch.setattr(validator, "EXPECTED_MODEL_FILES", pinned)
    assert validator.validate_model(model_dir) == pinned

    for name in pinned:
        with (model_dir / name).open("ab") as stream:
            stream.write(b" ")
        with pytest.raises(ValueError, match=f"model checksum mismatch: {name}"):
            validator.validate_model(model_dir)
        (model_dir / name).write_bytes((model_dir / name).read_bytes()[:-1])

    (model_dir / "generation_config.json").unlink()
    with pytest.raises(FileNotFoundError):
        validator.validate_model(model_dir)


def test_a_prompt_that_is_not_the_pinned_one_is_refused(tmp_path, monkeypatch):
    """The frozen prompt is enforced against the contract computed from the file."""
    validator = _validator()
    model_dir = _model_dir(tmp_path)
    monkeypatch.setattr(
        validator,
        "EXPECTED_MODEL_FILES",
        {name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest() for name in validator.EXPECTED_MODEL_FILES},
    )
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("not the frozen prompt")

    monkeypatch.setattr(validator, "prompt_contract", lambda *_: dict(validator.EXPECTED_PROMPT, prompt_token_count=7))
    with pytest.raises(ValueError, match="prompt/tokenizer contract mismatch"):
        validator.validate_inputs(model_dir=model_dir, prompt=prompt, fixture=None, artifact=None)

    monkeypatch.setattr(validator, "prompt_contract", lambda *_: dict(validator.EXPECTED_PROMPT))
    report = validator.validate_inputs(model_dir=model_dir, prompt=prompt, fixture=None, artifact=None)
    assert report["prompt"] == validator.EXPECTED_PROMPT
    assert set(report["model_checksums"]) == set(validator.EXPECTED_MODEL_FILES)


def test_checkpoint_verifies_all_consumed_shards_against_pinned_digests(tmp_path, monkeypatch):
    validator = _validator()
    model = _model_dir(tmp_path)
    shards = {"part-a.safetensors": b"first", "part-b.safetensors": b"second"}
    for name, payload in shards.items():
        (model / name).write_bytes(payload)
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "part-a.safetensors", "b": "part-b.safetensors"}})
    )
    pinned = {name: hashlib.sha256((model / name).read_bytes()).hexdigest() for name in validator.EXPECTED_MODEL_FILES}
    monkeypatch.setattr(validator, "EXPECTED_MODEL_FILES", pinned)
    monkeypatch.setattr(
        validator,
        "EXPECTED_SHARD_FILES",
        {name: hashlib.sha256(payload).hexdigest() for name, payload in shards.items()},
    )
    observed = validator.validate_checkpoint(model)
    assert set(observed) == set(pinned) | set(shards)
    (model / "part-b.safetensors").write_bytes(b"replaced")
    with pytest.raises(ValueError, match="part-b"):
        validator.validate_checkpoint(model)
    del validator.EXPECTED_SHARD_FILES["part-a.safetensors"]
    with pytest.raises(ValueError, match="unpinned"):
        validator.validate_checkpoint(model)
