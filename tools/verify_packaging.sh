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

# ---------------------------------------------------------------------------
# Reset to a fully clean state — what every mode runs into.
# ---------------------------------------------------------------------------
wipe_state() {
    pip uninstall -y simpler >/dev/null 2>&1 || true
    rm -rf build/ python/_task_interface*.so
}

# ---------------------------------------------------------------------------
# Smoke check: import surface + each user entry point's argparse.
# Tests packaging only, not functionality. Functional tests live in pytest.
# ---------------------------------------------------------------------------
smoke() {
    local mode="$1"
    echo "::group::[${mode}] import surface"
    python -c "
import os
import simpler, simpler_setup
from simpler.worker import Worker
from simpler.task_interface import ChipWorker
from simpler.orchestrator import Orchestrator
from simpler_setup.runtime_builder import RuntimeBuilder
from simpler_setup.runtime_compiler import RuntimeCompiler
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
for d in inc_dirs:
    h = os.path.join(d, 'pipe_sync.h')
    assert os.path.isfile(h), 'incore helper not shipped: ' + h
print('incore helpers OK:', inc_dirs)
"
    echo "::endgroup::"
    echo "::group::[${mode}] standalone test_*.py --help"
    python tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_unroll/test_paged_attention_unroll.py --help >/dev/null
    echo "::endgroup::"
    echo "smoke[${mode}] OK"
}

# ---------------------------------------------------------------------------
# Verify required deps are present. Build deps: --no-build-isolation modes
# need scikit-build-core/nanobind/cmake; cmake-direct mode needs nanobind for
# find_package(). Runtime deps: smoke imports simpler_setup.goldens.paged_attention
# which imports torch, and pytest is used by one of the smoke checks.
# ---------------------------------------------------------------------------
python -c "import scikit_build_core, nanobind, cmake, torch, pytest" 2>/dev/null || {
    echo "ERROR: venv missing required deps. Install with:" >&2
    echo "  pip install scikit-build-core nanobind cmake pytest torch" >&2
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
# Mode 2: pip install --no-build-isolation .
# ---------------------------------------------------------------------------
echo "===== Mode 2: pip install --no-build-isolation . ====="
wipe_state
pip install --no-build-isolation .
smoke "pip install --no-build-isolation ."

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
