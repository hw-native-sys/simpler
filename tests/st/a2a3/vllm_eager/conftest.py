# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fixtures for the real vLLM -> Simpler eager integration smoke."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_ADAPTER_CASE = Path("examples/a2a3/host_build_graph/qwen3_14b_serving_effective")


@dataclass(frozen=True)
class VllmPaths:
    adapter_root: Path
    model: Path
    artifact: Path | None
    prompt_ids: Path

    @property
    def runner(self) -> Path:
        return self.adapter_root / "run_vllm_eager.py"


def _required(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} is required for the vLLM eager integration smoke")
    return Path(value)


@pytest.fixture(scope="session")
def adapter_runner() -> Path:
    root = Path(os.environ.get("VLLM_ADAPTER_ROOT") or _ROOT)
    candidates = [root / "run_vllm_eager.py", root / _ADAPTER_CASE / "run_vllm_eager.py"]
    runner = next((candidate for candidate in candidates if candidate.is_file()), None)
    if runner is None:
        tried = ", ".join(str(candidate) for candidate in candidates)
        pytest.fail(
            f"vLLM adapter runner not found under {root} (tried {tried}); set VLLM_ADAPTER_ROOT to the "
            "pinned #2605 checkout or its qwen3_14b_serving_effective directory"
        )
    return runner


@pytest.fixture(scope="session")
def vllm_paths(adapter_runner) -> VllmPaths:
    artifact = os.environ.get("VLLM_ARTIFACT")
    paths = VllmPaths(
        adapter_root=adapter_runner.parent,
        model=_required("VLLM_MODEL_DIR"),
        artifact=Path(artifact) if artifact else None,
        prompt_ids=_required("VLLM_PROMPT_IDS"),
    )
    checks = (("VLLM_MODEL_DIR", paths.model), ("VLLM_PROMPT_IDS", paths.prompt_ids))
    if paths.artifact is not None:
        checks += (("VLLM_ARTIFACT", paths.artifact),)
    for name, path in checks:
        if not path.exists():
            pytest.fail(f"{name} does not exist: {path}")
    return paths


def _device(request) -> str:
    return str(request.config.getoption("--device")).split("-")[0].split(",")[0]


def _run_backend(request, paths: VllmPaths, directory: Path, backend: str, reference: Path | None = None):
    output = directory / f"{backend}.json"
    command = [
        sys.executable,
        "-u",
        str(paths.runner),
        "--model",
        str(paths.model),
        "--prompt-ids",
        str(paths.prompt_ids),
        "--backend",
        backend,
        "--device",
        _device(request),
        "--max-tokens",
        "2",
        "--rounds",
        "1",
        "--output",
        str(output),
    ]
    if backend == "simpler":
        if paths.artifact is None:
            pytest.fail("VLLM_ARTIFACT is required for the Simpler eager backend")
        command += ["--artifact", str(paths.artifact)]
    if reference is not None:
        command += ["--reference", str(reference)]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
        env={
            **os.environ,
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
            "TASK_QUEUE_ENABLE": os.environ.get("TASK_QUEUE_ENABLE", "0"),
        },
    )
    log = directory / f"{backend}.log"
    log.write_text(result.stdout + result.stderr)
    print(f"VLLM_{backend.upper()}_LOG {log}", flush=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.is_file(), result.stdout + result.stderr
    return json.loads(output.read_text()), output


@pytest.fixture(scope="session")
def native_result(request, vllm_paths, tmp_path_factory):
    return _run_backend(request, vllm_paths, tmp_path_factory.mktemp("vllm-native"), "native")


@pytest.fixture
def run_backend(request):
    def _run(paths: VllmPaths, directory: Path, backend: str, reference: Path | None = None):
        return _run_backend(request, paths, directory, backend, reference)

    return _run
