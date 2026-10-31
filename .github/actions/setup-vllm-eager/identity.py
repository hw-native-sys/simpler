# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Record the actual vLLM-Ascend and runtime modules inside the exclusive CI task."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import _task_interface
import torch
import torch_npu
import vllm
import vllm_ascend

from simpler_setup.test_environments import preflight

# Import eagerly so identity fails before the exclusive task if a backend is
# missing; the tuple also makes the imported modules intentional.
_REQUIRED_MODULES = (torch, torch_npu, vllm, vllm_ascend)

_DISTRIBUTIONS = (
    "torch",
    "torch-npu",
    "vllm",
    "vllm-ascend",
    "triton-ascend",
    "numpy",
    "opencv-python-headless",
)


def _sha256(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def _module(name):
    module = __import__(name)
    return _sha256(module.__file__)


def _distribution_identity(name):
    distribution = importlib.metadata.distribution(name)
    record = {"version": distribution.version}
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        try:
            record["direct_url"] = json.loads(direct_url)
        except json.JSONDecodeError:
            record["direct_url"] = direct_url
    return record


def _optional_file(path):
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_file():
        return _sha256(candidate)
    if candidate.is_dir():
        for name in ("manifest.json", "config.json", "model.safetensors.index.json"):
            target = candidate / name
            if target.is_file():
                return _sha256(target)
    return {"path": str(candidate), "sha256": None}


def _atb_home():
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
    return candidates[0] if candidates else None


def _atb_version(home):
    if home is None:
        return None
    version_file = home / "version.info"
    return version_file.read_text().strip() if version_file.is_file() else None


def main():
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert _task_interface.__build_commit__ == source, (_task_interface.__build_commit__, source)
    environment = preflight("vllm_torch_npu")
    atb_home = _atb_home()
    record = {
        "source": source,
        "python": sys.version,
        "executable": sys.executable,
        "machine": platform.machine(),
        "environment": environment,
        "modules": {name: _module(name) for name in ("torch", "torch_npu", "vllm", "vllm_ascend", "_task_interface")},
        "distributions": {name: _distribution_identity(name) for name in _DISTRIBUTIONS if _is_installed(name)},
        "setup": {
            "cann_home": os.environ.get("ASCEND_HOME_PATH"),
            "atb_home": str(atb_home) if atb_home else None,
            "atb_version": _atb_version(atb_home),
            "vllm_source_dir": os.environ.get("VLLM_SOURCE_DIR"),
            "vllm_ascend_source_dir": os.environ.get("VLLM_ASCEND_SOURCE_DIR"),
        },
        "inputs": {
            "model": _optional_file(os.environ.get("VLLM_MODEL_DIR")),
            "artifact": _optional_file(os.environ.get("VLLM_ARTIFACT")),
            "prompt_ids": _optional_file(os.environ.get("VLLM_PROMPT_IDS")),
        },
    }
    destination = Path("outputs/vllm-eager-ci/identity.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print("VLLM_CI_RUNTIME " + json.dumps(record), flush=True)


def _is_installed(name):
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


if __name__ == "__main__":
    main()
