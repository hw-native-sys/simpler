# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host KV storage lifetime during resident request initialization."""

import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

CASE = Path(__file__).resolve().parents[3] / "examples/a2a3/host_build_graph/qwen3_14b_serving_effective"
sys.path.insert(0, str(CASE))
import reference_worker_submit as consumer  # noqa: E402


@pytest.fixture
def resident_upload(monkeypatch):
    for name in ("BATCH", "HEADS", "PAGE", "HIDDEN", "PADDED_VOCAB"):
        monkeypatch.setattr(consumer, name, 1)
    monkeypatch.setattr(consumer, "HEAD_DIM", 2)
    monkeypatch.setattr(
        consumer, "_host_buffer", lambda _worker, tensor: SimpleNamespace(tensor=tensor.clone(), close=lambda: None)
    )
    references = []
    generated_layers = []
    peak_live = 0

    def layer_values(layer):
        nonlocal peak_live
        generated_layers.append(layer)
        for index, kind in enumerate(("key", "value")):
            physical = torch.full((1, 1, 1, 2), layer * 2 + index, dtype=torch.bfloat16)
            references.append(weakref.ref(physical))
            peak_live = max(peak_live, sum(ref() is not None for ref in references))
            yield kind, physical

    fixture = SimpleNamespace(num_pages=1, layer=layer_values)
    uploads = []

    def copy_to(device, staging, *, dst_offset=None):
        if dst_offset is not None:
            uploads.append((device, dst_offset, staging.tensor.flatten().tolist()))

    worker = SimpleNamespace(alloc_child_tensor=lambda *_args: object(), copy_to=copy_to)

    def allocate(**kwargs):
        devices = {}
        consumer.allocate_resident(worker, devices, {}, fixture, None, shared_devices={}, **kwargs)
        expected = [
            (devices[name], layer * 4, [float(layer * 2 + index)] * 2)
            for layer in range(consumer.LAYERS)
            for index, name in enumerate(("k_cache", "v_cache"))
        ]
        assert uploads == expected
        uploads.clear()

    return SimpleNamespace(
        allocate=allocate,
        references=references,
        generated_layers=generated_layers,
        peak_live=lambda: peak_live,
    )


def test_default_resident_upload_releases_completed_kv_layers(resident_upload):
    resident_upload.allocate()

    assert resident_upload.generated_layers == list(range(consumer.LAYERS))
    assert resident_upload.peak_live() <= 2
    assert all(ref() is None for ref in resident_upload.references)


def test_explicit_kv_cache_reuses_layers_until_caller_releases_them(resident_upload):
    cache = {}
    resident_upload.allocate(layer_cache=cache)
    resident_upload.allocate(layer_cache=cache)

    assert resident_upload.generated_layers == list(range(consumer.LAYERS))
    assert len(cache) == consumer.LAYERS
    assert all(ref() is not None for ref in resident_upload.references)
    cache.clear()
    assert all(ref() is None for ref in resident_upload.references)
