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
