#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate the external model, prompt, fixture, and generated Qwen artifact inputs.

The model and prompt this qualification is frozen against live outside the
repository, so their content is the only thing that can be pinned here. The
digests below are those pins, and this module is their only reader: it compares
them before any device work, so a checkpoint or prompt that is not the frozen
one fails here rather than producing numbers against the wrong inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

# The frozen prompt, in the form `prompt_contract` returns so the two compare whole.
EXPECTED_PROMPT: dict[str, Any] = {
    "prompt_sha256": "6b3e99d56ed58d58d98f149be7a68d2e454df2447414ac51f3f46da92adb9dfd",
    "prompt_token_count": 3338,
    "prompt_token_ids_sha256": "8db4a9a6897a5ae237fd3806ce39b4d49a549f8faf2881d5ae5740d5e6be6bb9",
    "tokenizer_sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
}

# Every checkpoint file this qualification reads, and the digest it must have.
# A required file with no digest here is a failure rather than an unchecked pass.
EXPECTED_MODEL_FILES: dict[str, str] = {
    "config.json": "e73c3664ca09b10a673fef0c22e8a6b456201d49bd4713c9691f775720e8857a",
    "model.safetensors.index.json": "62d7ad35757bae5e7baa452cb1483178b7daa50e869e923226b8da10871f7ebc",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
}

EXPECTED_SHARD_FILES = {
    "model-00001-of-00008.safetensors": "e942bdbdf08857d16a8fef7d1dae9fceabeb4e84def6043485fe2f6f085dab0e",
    "model-00002-of-00008.safetensors": "f7c9c6eee628f5ad831d2d1d292e120505e5fcadeb38f88b4d3c4cb86306ccf9",
    "model-00003-of-00008.safetensors": "dfb8c5df9404b41ad6ae74e8b6b367135f017b4467b884cf71b17c71954f18a9",
    "model-00004-of-00008.safetensors": "eab286fec759e3e59ab228621aefa0fef14ed56039e06f959e67257d5af7604d",
    "model-00005-of-00008.safetensors": "97f0dc2992e59da95c466eff6f4fd0c8335843bbc36ed5c913a6f5150748c0e6",
    "model-00006-of-00008.safetensors": "9e8e76a013cd5e253865b792991e0b410f869b136b3c500079b531b09198e99e",
    "model-00007-of-00008.safetensors": "0aee70ee6e91dc00d818804fb47f124d13ee4ad5b4a64553e09dbf9391cd5750",
    "model-00008-of-00008.safetensors": "0d6b92296e326d39bbbaeb32c3ec454ac606da843d4c8ffa8edf010b62b8c9e0",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_contract(prompt: Path, tokenizer: Path) -> dict[str, Any]:
    from tokenizers import Tokenizer  # noqa: PLC0415

    text = prompt.read_text(encoding="utf-8")
    ids = Tokenizer.from_file(str(tokenizer)).encode(text).ids
    token_payload = json.dumps(ids, separators=(",", ":")).encode("utf-8")
    return {
        "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "prompt_token_count": len(ids),
        "prompt_token_ids_sha256": hashlib.sha256(token_payload).hexdigest(),
        "tokenizer_sha256": sha256_file(tokenizer),
    }


def validate_model(model_dir: Path) -> dict[str, str]:
    observed = {}
    for name, expected_hash in EXPECTED_MODEL_FILES.items():
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[name] = sha256_file(path)
        if observed[name] != expected_hash:
            raise ValueError(f"model checksum mismatch: {name}")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if int(config.get("num_hidden_layers", 0)) != 40 or int(config.get("hidden_size", 0)) != 5120:
        raise ValueError("checkpoint geometry is not Qwen3-14B decode geometry")
    return observed


def validate_checkpoint(model_dir: Path) -> dict[str, str]:
    """Verify pinned metadata and every shard named by the verified weight index."""
    observed = validate_model(model_dir)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    for name in sorted(set(index["weight_map"].values())):
        if name not in EXPECTED_SHARD_FILES:
            raise ValueError(f"unpinned checkpoint shard: {name}")
        observed[name] = sha256_file(model_dir / name)
        if observed[name] != EXPECTED_SHARD_FILES[name]:
            raise ValueError(f"checkpoint checksum mismatch: {name}")
    return observed


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_inputs(*, model_dir: Path, prompt: Path, fixture: Path | None, artifact: Path | None):
    observed_prompt = prompt_contract(prompt, model_dir / "tokenizer.json")
    if observed_prompt != EXPECTED_PROMPT:
        raise ValueError(f"prompt/tokenizer contract mismatch: {observed_prompt}")
    model_hashes = validate_model(model_dir)
    report = {"prompt": observed_prompt, "model_checksums": model_hashes}
    if fixture is not None:
        fixture_module = _load_module("_qwen_fixture_validator", Path(__file__).with_name("fixture.py"))
        loaded = fixture_module.load_fixture(fixture)
        report["fixture"] = {
            "schema": loaded.manifest["schema"],
            "metadata_only": loaded.metadata_only,
            "decode_dispatches_remaining": loaded.manifest["decode_dispatches_remaining"],
        }
    if artifact is not None:
        bridge = _load_module("_qwen_artifact_bridge_validator", Path(__file__).with_name("callable_bridge.py"))
        inspected = bridge.inspect_artifact(artifact)
        report["artifact"] = {
            "parameter_count": len(inspected.parameter_names),
            "parameter_names": list(inspected.parameter_names),
            "runtime_config": inspected.runtime_config,
        }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args(argv)
    report = validate_inputs(
        model_dir=args.model_dir,
        prompt=args.prompt,
        fixture=args.fixture,
        artifact=args.artifact,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
