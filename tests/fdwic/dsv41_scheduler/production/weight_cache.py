#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Runtime-independent native MoE weight fixtures with identity and layout checks."""

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import replace
from functools import partial
from pathlib import Path

import torch


def weight_layout(specs):
    return {
        spec.name: {"shape": list(spec.shape), "dtype": str(spec.dtype), "resident": spec.resident}
        for spec in specs
        if spec.name.startswith(("routed_w", "shared_w"))
    }


def cached_weight_specs(moe, counts, source_commit, cache_root):
    """Only immutable weights are cached; counts, routing and outputs remain freshly constructed."""
    specs = moe.build_moe_specs(counts, initialize=False)
    layout = weight_layout(specs)
    if not layout:
        raise ValueError("Native MoE has no weight specifications")
    identity = {"source_commit": source_commit, "torch_version": str(torch.__version__), "weights": layout}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    directory = cache_root / key
    with (cache_root / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        hit = directory.exists()
        if not hit:
            generated = moe.build_moe_specs(counts, initialize=True)
            if weight_layout(generated) != layout:
                raise ValueError("Initialized MoE weight layout differs from its declared layout")
            weights = {
                spec.name: spec.init_value().to(dtype=spec.dtype).contiguous()
                for spec in generated
                if spec.name in layout
            }
            staging = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=cache_root))
            try:
                payload = staging / "weights.pt"
                torch.save(weights, payload)
                (staging / "manifest.json").write_text(
                    json.dumps({"identity": identity, "bytes": payload.stat().st_size}, indent=2) + "\n"
                )
                os.rename(staging, directory)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        manifest = json.loads((directory / "manifest.json").read_text())
        payload = directory / "weights.pt"
        if manifest["identity"] != identity or payload.stat().st_size != manifest["bytes"]:
            raise ValueError(f"Native MoE weight cache failed identity/size verification: {directory}")
        loaded = torch.load(payload, weights_only=True, mmap=True, map_location="cpu")
        if set(loaded) != set(layout):
            raise ValueError("Native MoE weight cache has unexpected tensor names")
        for spec in specs:
            if spec.name not in layout:
                continue
            value = loaded[spec.name]
            if list(value.shape) != list(spec.shape) or value.dtype != spec.dtype:
                raise ValueError(f"Native MoE weight cache has invalid tensor layout: {spec.name}")
            if not hit and not torch.equal(value.view(torch.uint8), weights[spec.name].view(torch.uint8)):
                raise ValueError(f"Native MoE weight serialization changed bytes: {spec.name}")
        specs = [
            replace(spec, init_value=partial(torch.clone, loaded[spec.name])) if spec.name in loaded else spec
            for spec in specs
        ]
    return specs, {"hit": hit, "key": key, "bytes": payload.stat().st_size}
