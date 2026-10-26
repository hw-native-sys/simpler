#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Install the pinned vLLM-Ascend 0.20.2rc1 A3 environment for issue #2715.
#
# Mirrors vllm-ascend Dockerfile.a3 at 367b8e62 while making the exact
# distribution versions visible to the adapter's checks:
#   * vLLM source install: VLLM_TARGET_DEVICE=empty + VLLM_VERSION_OVERRIDE=0.20.2
#   * vllm-ascend source install: SETUPTOOLS_SCM_PRETEND_VERSION=0.20.2rc1
#   * triton-ascend 3.2.1: Huawei Ascend mirror, numpy 1.26.4
#
# The caller must source CANN 9.0.0 and NNAL/ATB before invoking this script.
set -euo pipefail

ROOT=${GITHUB_WORKSPACE:-${ROOT:-$(pwd)}}
SRC_ROOT=${VLLM_SOURCE_ROOT:-${RUNNER_TEMP:-/tmp}/vllm-src}
VENV=${VLLM_VENV:-$ROOT/.venv-vllm}
PYTHON=${VLLM_PYTHON:-python3}
COMPILE_CUSTOM_KERNELS=${COMPILE_CUSTOM_KERNELS:-1}
MAX_JOBS=${MAX_JOBS:-32}
VLLM_SRC=${VLLM_SOURCE_DIR:-$SRC_ROOT/vllm-v0.20.2}
VLLM_ASCEND_SRC=${VLLM_ASCEND_SOURCE_DIR:-$SRC_ROOT/vllm-ascend-v0.20.2rc1}
VLLM_SHA=bc150f50299199599673614f80d12a196f377655
VLLM_ASCEND_SHA=367b8e62da799870a7476ce34f5f7658589a8aad
HW_PYPI=https://mirrors.huaweicloud.com/ascend/repos/pypi
CPU_INDEX=https://download.pytorch.org/whl/cpu/
REPORT=$ROOT/outputs/vllm-eager-ci

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export MAX_JOBS
export SOC_VERSION=${SOC_VERSION:-ascend910_9391}
export TORCH_DEVICE_BACKEND_AUTOLOAD=${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}

mkdir -p "$REPORT"
stage=init
trap 'rc=$?; echo "VLLM_ENV_FAILED stage=$stage rc=$rc at $(date -Is)" | tee -a "$REPORT/environment-status.txt"; exit $rc' ERR
log() { echo "VLLM_ENV_STAGE $* $(date -Is)" | tee -a "$REPORT/environment-status.txt"; }

log init
if ! command -v clang >/dev/null 2>&1 || ! command -v clang++ >/dev/null 2>&1; then
  if [ "$COMPILE_CUSTOM_KERNELS" = "1" ]; then
    echo "ERROR: COMPILE_CUSTOM_KERNELS=1 requires clang and clang++ (official Dockerfile installs clang-15)" >&2
    exit 1
  fi
  echo "WARN: clang/clang++ missing; COMPILE_CUSTOM_KERNELS=0 only validates imports" \
    | tee -a "$REPORT/environment-status.txt"
fi

stage=venv
log venv
if [ ! -x "$VENV/bin/python" ]; then "$PYTHON" -m venv "$VENV"; fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
if ! python -c 'import ssl' >/dev/null 2>&1; then
  echo "ERROR: the selected Python lacks the ssl module; use a Python with HTTPS support" >&2
  exit 1
fi
python -m pip install --upgrade pip setuptools wheel
# vLLM's build-system declares setuptools>=77.0.3,<81.0.0; the build below
# runs without isolation against this venv.
python -m pip install 'setuptools>=77.0.3,<81.0.0' 'wheel'

stage=torch
log torch
python -m pip install --extra-index-url "$CPU_INDEX" \
  'torch==2.10.0+cpu' 'torchvision==0.25.0' 'torchaudio==2.10.0'
# torch-npu requires torch==2.10.0; install it without a resolver pass so the
# CPU local version is not replaced by a PyPI build.
python -m pip install --no-deps 'torch-npu==2.10.0'

stage=build_deps
log build_deps
python -m pip install 'cmake>=3.26' 'ninja' 'packaging' 'pybind11' 'setuptools>=64' 'setuptools-scm>=8' 'wheel'

stage=vllm_clone
log vllm_clone
if [ ! -d "$VLLM_SRC/.git" ]; then
  mkdir -p "$(dirname "$VLLM_SRC")"
  git clone --depth 1 -b v0.20.2 https://github.com/vllm-project/vllm.git "$VLLM_SRC"
fi
test "$(git -C "$VLLM_SRC" rev-parse HEAD)" = "$VLLM_SHA"

stage=vllm_install
log vllm_install
VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=0.20.2 \
  python -m pip install --no-build-isolation -e "$VLLM_SRC[audio]" --extra-index-url "$CPU_INDEX"
python -m pip uninstall -y triton || true

stage=vllm_ascend_clone
log vllm_ascend_clone
if [ ! -d "$VLLM_ASCEND_SRC/.git" ]; then
  mkdir -p "$(dirname "$VLLM_ASCEND_SRC")"
  git clone --depth 1 -b v0.20.2rc1 https://github.com/vllm-project/vllm-ascend.git "$VLLM_ASCEND_SRC"
fi
test "$(git -C "$VLLM_ASCEND_SRC" rev-parse HEAD)" = "$VLLM_ASCEND_SHA"

stage=vllm_ascend_deps
log vllm_ascend_deps
export PIP_EXTRA_INDEX_URL="$HW_PYPI"
python -m pip install \
  'cmake>=3.26' 'pybind11' 'setuptools>=64' 'setuptools-scm>=8' \
  'attrs' 'decorator' 'einops' 'googleapis-common-protos' 'packaging' \
  'pyyaml' 'scipy' 'pandas' 'pandas-stubs' 'psutil' 'wheel' 'regex' \
  'msgpack' 'quart' 'numba' 'xgrammar>=0.1.30' 'fastapi<0.124.0' \
  'compressed_tensors>=0.11.0' 'arctic-inference==0.1.1' 'transformers==5.5.3'
python -m pip install 'memfabric_hybrid' 'memcache_hybrid' || \
  echo "WARN: optional memfabric/memcache unavailable" | tee -a "$REPORT/environment-status.txt"

stage=triton_ascend
log triton_ascend
python -m pip uninstall -y triton triton-ascend || true
# opencv 4.13 declares numpy>=2; the downgrade is deliberate for triton-ascend.
python -m pip install --no-deps 'numpy==1.26.4'
python -m pip install --no-deps 'triton-ascend==3.2.1' --extra-index-url "$HW_PYPI"

stage=opencv
log opencv
# vLLM requires opencv>=4.13; opencv 4.13 metadata requires numpy>=2.
# Keep numpy 1.26.4 for triton-ascend and verify cv2 actually imports.
python -m pip install --no-deps 'opencv-python-headless==4.13.0.92' || true
if ! python -c 'import cv2' >/dev/null 2>&1; then
  python -m pip install --no-deps 'opencv-python-headless==4.11.0.86'
fi
python -c 'import cv2; print("cv2", cv2.__version__, cv2.__file__)'

stage=vllm_ascend_install
log vllm_ascend_install
SETUPTOOLS_SCM_PRETEND_VERSION=0.20.2rc1 COMPILE_CUSTOM_KERNELS="$COMPILE_CUSTOM_KERNELS" \
  python -m pip install --no-build-isolation -e "$VLLM_ASCEND_SRC" --no-deps

stage=freeze
log freeze
python -m pip freeze --all | tee "$REPORT/freeze.txt"
python -m pip check > "$REPORT/pip-check.txt" 2>&1 || true
python - "$REPORT" <<'PY'
import importlib
import json
import platform
import sys
from pathlib import Path

root = Path(sys.argv[1])
modules = ["torch", "torch_npu", "vllm", "vllm_ascend", "cv2", "numpy"]
record = {"python": sys.version, "machine": platform.machine(), "modules": {}}
for name in modules:
    try:
        module = importlib.import_module(name)
        record["modules"][name] = {
            "version": getattr(module, "__version__", None),
            "file": getattr(module, "__file__", None),
        }
    except Exception as exc:  # noqa: BLE001
        record["modules"][name] = {"error": repr(exc)}
try:
    import importlib.metadata as metadata

    for distribution in ("torch", "torch-npu", "vllm", "vllm-ascend", "triton-ascend", "numpy", "opencv-python-headless"):
        record.setdefault("distributions", {})[distribution] = metadata.version(distribution)
except Exception as exc:  # noqa: BLE001
    record["distribution_error"] = repr(exc)
(root / "versions.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY

stage=done
log done
echo "OK $(date -Is)" | tee -a "$REPORT/environment-status.txt"
