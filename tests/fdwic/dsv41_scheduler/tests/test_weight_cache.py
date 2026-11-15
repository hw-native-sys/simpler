# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native model fixture caching preserves bytes and dynamic control inputs."""

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.manual


@dataclass
class InputSpec:
    name: str
    shape: list
    dtype: object
    init_value: object = None
    resident: object = "stacked"


class ModelFixture:
    def build_moe_specs(self, counts, *, initialize):
        return [
            InputSpec(
                "routed_w1",
                [3],
                torch.uint8,
                (lambda: torch.tensor([2, 5, 9], dtype=torch.uint8)) if initialize else None,
            ),
            InputSpec("shared_w1", [2], torch.float32, (lambda: torch.tensor([1.5, -2.5])) if initialize else None),
            InputSpec("num_tokens", [2], torch.int32, lambda: torch.tensor(counts)),
            InputSpec("force_topk", [2], torch.int32, lambda: torch.tensor(counts).flip(0)),
        ]


def load_cache():
    path = Path(__file__).resolve().parents[1] / "production/weight_cache.py"
    spec = importlib.util.spec_from_file_location("weight_cache", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cached_weight_specs


def test_cached_weights_are_exact_and_dynamic_inputs_are_fresh(tmp_path):
    cached = load_cache()
    original, cold = cached(ModelFixture(), (0, 6), "model-a", tmp_path)
    assert not cold["hit"]
    original[0].init_value()[0] = 0
    warm, hit = cached(ModelFixture(), (6, 0), "model-a", tmp_path)
    assert hit["hit"] and hit["key"] == cold["key"] and hit["bytes"] == cold["bytes"]
    assert warm[0].init_value().tolist() == [2, 5, 9]
    assert warm[1].init_value().tolist() == [1.5, -2.5]
    assert warm[2].init_value().tolist() == [6, 0]
    assert warm[3].init_value().tolist() == [0, 6]
    warm[1].init_value()[0] = 99
    assert warm[1].init_value().tolist() == [1.5, -2.5]


def test_new_model_revision_uses_another_fixture(tmp_path):
    cached = load_cache()
    _, old = cached(ModelFixture(), (0, 6), "model-a", tmp_path)
    _, new = cached(ModelFixture(), (0, 6), "model-b", tmp_path)
    assert not new["hit"] and old["key"] != new["key"]


def test_truncated_weights_are_rejected_before_deserialization(tmp_path):
    cached = load_cache()
    _, info = cached(ModelFixture(), (0, 6), "model-a", tmp_path)
    payload = tmp_path / info["key"] / "weights.pt"
    payload.write_bytes(payload.read_bytes()[:-1])
    with pytest.raises(ValueError, match="failed identity/size verification"):
        cached(ModelFixture(), (6, 0), "model-a", tmp_path)
