# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real vLLM eager -> Simpler Worker smoke for the fixed Qwen3-14B contract."""

from __future__ import annotations

import importlib.metadata
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.requires_hardware,
    pytest.mark.platforms(["a2a3"]),
    pytest.mark.test_environment("vllm_torch_npu"),
]

REQUESTS = 16
TOKENS = 2


def _atb_home() -> Path:
    candidates = []
    atb_env = os.environ.get("ATB_ENV")
    if atb_env:
        candidates.append(Path(atb_env).resolve().parent)
    for name in ("ATB_HOME_PATH", "ATB_HOME"):
        value = os.environ.get(name)
        if value:
            candidates.append(Path(value))
    candidates.append(Path("/usr/local/Ascend/nnal/atb/latest"))
    for base in candidates:
        for candidate in (base, base / "latest", *base.parents):
            if (candidate / "version.info").is_file():
                return candidate
    return candidates[0]


def test_target_versions_are_pinned():
    assert importlib.metadata.version("torch") == "2.10.0+cpu"
    assert importlib.metadata.version("torch-npu") == "2.10.0"
    assert importlib.metadata.version("vllm") == "0.20.2"
    assert importlib.metadata.version("vllm-ascend") == "0.20.2rc1"
    assert importlib.metadata.version("triton-ascend") == "3.2.1"
    assert importlib.metadata.version("numpy") == "1.26.4"
    version_file = _atb_home() / "version.info"
    assert version_file.is_file(), f"ATB version.info not found under {_atb_home()}"
    version = version_file.read_text()
    match = re.search(r"Ascend-cann-atb\s*:\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)", version)
    assert match is not None, version
    assert match.group(1).startswith("9.0"), f"NNAL/ATB 9.0.0 is required; found {match.group(1)}"


@pytest.mark.manual
def test_native_vllm_short_request(native_result):
    data, _ = native_result
    assert data["backend"] == "native"
    assert data["simpler"] is None
    assert len(data["rounds"]) == 1
    tokens = data["rounds"][0]["tokens"]
    assert len(tokens) == REQUESTS
    assert all(len(row) == TOKENS for row in tokens)


@pytest.mark.manual
def test_simpler_decode_smoke(run_backend, native_result, vllm_paths, tmp_path):
    native_data, reference = native_result
    data, _ = run_backend(vllm_paths, tmp_path, "simpler", reference)
    assert data["backend"] == "simpler"
    assert data["simpler"] == [{"steps": TOKENS - 1, "failed": False}]
    assert len(data["rounds"]) == 1
    assert data["rounds"][0]["tokens"] == native_data["rounds"][0]["tokens"]
    assert all(len(row) == TOKENS for row in data["rounds"][0]["tokens"])


@pytest.mark.manual
def test_native_single_active_request(single_active_native_result):
    data, _ = single_active_native_result
    assert data["backend"] == "native"
    assert data["requests"] == 1
    assert data["simpler"] is None
    assert len(data["rounds"]) == 1
    tokens = data["rounds"][0]["tokens"]
    assert len(tokens) == 1
    assert all(len(row) == TOKENS for row in tokens)


@pytest.mark.manual
def test_simpler_single_active_decode(run_single_active_backend, single_active_native_result, vllm_paths, tmp_path):
    native_data, reference = single_active_native_result
    data, _ = run_single_active_backend(vllm_paths, tmp_path, "simpler", reference)
    assert data["backend"] == "simpler"
    assert data["requests"] == 1
    assert data["simpler"] == [{"steps": TOKENS - 1, "failed": False}]
    assert len(data["rounds"]) == 1
    assert data["rounds"][0]["tokens"] == native_data["rounds"][0]["tokens"]
    assert all(len(row) == TOKENS for row in data["rounds"][0]["tokens"])


def test_simpler_backend_refuses_missing_artifact(adapter_runner, tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(adapter_runner),
            "--model",
            str(tmp_path / "model"),
            "--prompt-ids",
            str(tmp_path / "prompt-ids.json"),
            "--backend",
            "simpler",
            "--device",
            "0",
            "--output",
            str(tmp_path / "simpler.json"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={**os.environ},
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "--artifact is required" in result.stdout + result.stderr
