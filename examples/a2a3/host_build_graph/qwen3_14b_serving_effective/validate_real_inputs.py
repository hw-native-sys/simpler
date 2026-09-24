#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate the external model, prompt, fixture, and generated Qwen artifact inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


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


def validate_model(model_dir: Path, expected: dict[str, str]) -> dict[str, str]:
    required = ("config.json", "model.safetensors.index.json", "tokenizer.json", "generation_config.json")
    observed = {}
    for name in required:
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[name] = sha256_file(path)
        expected_hash = expected.get(f"model_{name.replace('.', '_')}_sha256")
        if expected_hash is not None and observed[name] != expected_hash:
            raise ValueError(f"model checksum mismatch: {name}")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if int(config.get("num_hidden_layers", 0)) != 40 or int(config.get("hidden_size", 0)) != 5120:
        raise ValueError("checkpoint geometry is not Qwen3-14B decode geometry")
    return observed


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_inputs(manifest_path: Path, *, model_dir: Path, prompt: Path, fixture: Path | None, artifact: Path | None):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_prompt = manifest["prompt"]
    observed_prompt = prompt_contract(prompt, model_dir / "tokenizer.json")
    if observed_prompt != expected_prompt:
        raise ValueError(f"prompt/tokenizer contract mismatch: {observed_prompt}")
    model_hashes = validate_model(model_dir, manifest.get("model_checksums", {}))
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args(argv)
    report = validate_inputs(
        args.manifest,
        model_dir=args.model_dir,
        prompt=args.prompt,
        fixture=args.fixture,
        artifact=args.artifact,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
