#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Verify all 5 install paths x 2 entry points are green.
#
# Each mode runs from a fully clean state (uninstall + wipe build artifacts) so
# leftover binaries from a previous mode cannot mask a regression in the next.
# Slow but reliable. CI calls this script directly; docs/python-packaging.md
# documents it. Run from the repo root inside an activated venv.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: activate a venv first (source .venv/bin/activate)" >&2
    exit 1
fi

# macOS libomp collision (homebrew numpy + pip torch) — silence here so the
# smoke check never aborts on it; conftest.py sets the same env for pytest
# entry points. See docs/macos-libomp-collision.md.
export KMP_DUPLICATE_LIB_OK=TRUE

PACKAGING_SMOKE_DIR="$(mktemp -d)"
trap 'rm -rf "${PACKAGING_SMOKE_DIR}"' EXIT

# ---------------------------------------------------------------------------
# Reset to a fully clean state — what every mode runs into.
# ---------------------------------------------------------------------------
wipe_state() {
    pip uninstall -y simpler >/dev/null 2>&1 || true
    rm -rf build/ python/_task_interface*.so
}

# ---------------------------------------------------------------------------
# Smoke check: import surface + each user entry point's argparse. Run outside
# the repository so source files cannot shadow an incomplete wheel install.
# Tests packaging only, not functionality. Functional tests live in pytest.
# ---------------------------------------------------------------------------
smoke() {
    local mode="$1"
    echo "::group::[${mode}] import surface"
    (
        cd "${PACKAGING_SMOKE_DIR}"
        python -c "
import os

import simpler, simpler_setup
from simpler.worker import Worker
from simpler.task_interface import ChipWorker
from simpler.orchestrator import Orchestrator
from simpler_setup.runtime_builder import RuntimeBuilder
from simpler_setup.runtime_compiler import RuntimeCompiler
from simpler_setup.environment import PROJECT_ROOT
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.elf_parser import extract_text_section
from simpler_setup.platform_info import parse_platform, discover_runtimes
from simpler_setup.scene_test import SceneTestCase, scene_test
from simpler_setup.goldens.paged_attention import generate_inputs, compute_golden
print('simpler:', simpler.__file__)
print('simpler_setup:', simpler_setup.__file__)
# Verify shipped kernel-side helpers are reachable on the incore include path.
# A wheel that misses these data files would fall through to a cryptic kernel
# compilation error; this catches it at smoke time.
inc_dirs = KernelCompiler('a2a3sim').get_incore_include_dirs()
for rel in ('pipe_sync.h', os.path.join('common', 'dma_workspace.h')):
    assert any(os.path.isfile(os.path.join(d, rel)) for d in inc_dirs), \
        'incore helper not shipped: ' + rel + '; include dirs: ' + repr(inc_dirs)
print('incore helpers OK:', inc_dirs)
# RuntimeCompiler passes this directory to every host-side CMake configure.
for name in ('host_log_sources.cmake', 'profiling_config.cmake', 'sanitizers.cmake'):
    path = PROJECT_ROOT / 'cmake' / name
    assert path.is_file(), 'shared CMake module not shipped: ' + str(path)
print('shared CMake modules OK:', PROJECT_ROOT / 'cmake')
"
    )
    echo "::endgroup::"
    echo "::group::[${mode}] standalone test_*.py --help"
    (
        cd "${PACKAGING_SMOKE_DIR}"
        python "${REPO_ROOT}/tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_unroll/test_paged_attention_unroll.py" \
            --help >/dev/null
    )
    echo "::endgroup::"
    echo "smoke[${mode}] OK"
}

# ---------------------------------------------------------------------------
# Verify required deps meet the same minimums as pyproject.toml. This catches a
# persistent self-hosted venv whose unversioned packages would otherwise be
# accepted by the no-build-isolation and cmake-direct modes.
# ---------------------------------------------------------------------------
python - <<'PY' || {
import importlib.metadata
import re
import shutil
import subprocess

from packaging.requirements import Requirement
from packaging.version import Version

import cmake  # noqa: E402,F401
import nanobind  # noqa: E402,F401
import pytest  # noqa: E402,F401
import scikit_build_core  # noqa: E402,F401
import torch  # noqa: E402,F401

required = (
    "scikit-build-core>=0.10.0",
    "nanobind>=2.0.0",
    "cmake>=3.15",
    "pytest>=6.0",
    "torch>=2.3",
)
for requirement_text in required:
    requirement = Requirement(requirement_text)
    installed = importlib.metadata.version(requirement.name)
    if not requirement.specifier.contains(installed, prereleases=True):
        raise RuntimeError(f"{requirement.name} {installed} does not satisfy {requirement.specifier}")
    print(f"{requirement.name}: {installed}")

cmake_path = shutil.which("cmake")
if cmake_path is None:
    raise RuntimeError("cmake executable not found on PATH")
cmake_version_output = subprocess.run(
    [cmake_path, "--version"], check=True, capture_output=True, text=True
).stdout.splitlines()[0]
match = re.fullmatch(r"cmake version (\S+)", cmake_version_output)
if match is None or Version(match.group(1)) < Version("3.15"):
    raise RuntimeError(f"unsupported CMake executable: {cmake_version_output}")
print(f"cmake executable: {cmake_path} ({cmake_version_output})")

ninja_path = shutil.which("ninja")
if ninja_path is not None:
    ninja_version = subprocess.run(
        [ninja_path, "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    print(f"ninja executable: {ninja_path} ({ninja_version})")
else:
    print("ninja executable: not found; CMake will select another generator")
PY
    echo "ERROR: venv missing or has unsupported packaging dependencies. Install with:" >&2
    echo "  pip install 'scikit-build-core>=0.10.0' 'nanobind>=2.0.0' 'cmake>=3.15' 'ninja>=1.11' 'pytest>=6.0' 'torch>=2.3'" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Mode 1: pip install .
# ---------------------------------------------------------------------------
echo "===== Mode 1: pip install . ====="
wipe_state
pip install .
smoke "pip install ."

# ---------------------------------------------------------------------------
# Mode 2: targeted install followed by a default --no-build-isolation install.
# The second install deliberately reuses the CMake build directory: targeted
# platform selection must not leak into an ordinary package install.
# ---------------------------------------------------------------------------
echo "===== Mode 2: targeted install -> default install ====="
wipe_state
pip install --no-build-isolation \
    --config-settings=build.targets=build_package_a2a3sim .
test -f build/lib/a2a3/sim/host_build_graph/libhost_runtime.so
test ! -e build/lib/a5/sim
pip install --no-build-isolation .
test -f build/lib/a5/sim/host_build_graph/libhost_runtime.so
smoke "targeted -> default --no-build-isolation install"

# ---------------------------------------------------------------------------
# Mode 3: pip install -e .
# ---------------------------------------------------------------------------
echo "===== Mode 3: pip install -e . ====="
wipe_state
pip install -e .
smoke "pip install -e ."

# ---------------------------------------------------------------------------
# Mode 4: pip install --no-build-isolation -e .
# ---------------------------------------------------------------------------
echo "===== Mode 4: pip install --no-build-isolation -e . ====="
wipe_state
pip install --no-build-isolation -e .
smoke "pip install --no-build-isolation -e ."

# ---------------------------------------------------------------------------
# Mode 5: cmake + PYTHONPATH (no pip install at all)
# ---------------------------------------------------------------------------
echo "===== Mode 5: cmake + PYTHONPATH ====="
wipe_state
cmake -S . -B build/cmake_only \
      -Dnanobind_DIR=$(python -c 'import nanobind; print(nanobind.cmake_dir())')
cmake --build build/cmake_only
PYTHONPATH=$(pwd):$(pwd)/python smoke "cmake + PYTHONPATH"

echo
echo "===== ALL 5 MODES PASSED ====="
