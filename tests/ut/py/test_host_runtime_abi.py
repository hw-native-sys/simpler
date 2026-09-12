# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Uniform host-runtime pipeline-symbol contract tests."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_NEWLY_REQUIRED_PIPELINE_SYMBOLS = {
    "get_arena_bank_gm_heap_base_ctx",
    "get_pipeline_contract",
    "get_retained_temp_addr_ctx",
    "supports_concurrent_native_prepare_ctx",
}
# Kernel-mode lifecycle surface: every host-runtime component exports all of
# these; components without kernel-mode support export validating stubs that
# report unsupported rather than omitting the symbols.
_KERNEL_MODE_SYMBOLS = {
    "simpler_kernel_mode_init",
    "simpler_kernel_mode_launch",
    "simpler_kernel_mode_prepare_callable",
    "simpler_kernel_mode_supported",
}
_REMOVED_AMBIENT_SELECTION_SYMBOLS = {
    "select_arena_bank_ctx",
    "select_pipeline_slot_ctx",
    "set_native_run_identity_ctx",
    "set_task_accepted_state_ctx",
}

_SIM_CASES = [
    pytest.param(arch, "sim", runtime, id=f"{arch}-sim-{runtime}")
    for arch in ("a2a3", "a5")
    for runtime in ("host_build_graph", "tensormap_and_ringbuffer")
]
_ONBOARD_CASES = [
    pytest.param(
        arch,
        "onboard",
        runtime,
        id=f"{arch}-onboard-{runtime}",
        marks=[pytest.mark.requires_hardware, pytest.mark.platforms([arch])],
    )
    for arch in ("a2a3", "a5")
    for runtime in ("host_build_graph", "tensormap_and_ringbuffer")
]


def _defined_external_symbols(path: Path) -> set[str]:
    if sys.platform == "darwin":
        command = ["nm", "-gU", str(path)]
    else:
        command = ["nm", "-D", "--defined-only", str(path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    symbols = {line.split()[-1] for line in result.stdout.splitlines() if line.split()}
    if sys.platform == "darwin":
        return {symbol.removeprefix("_") for symbol in symbols}
    return symbols


@pytest.mark.parametrize(
    ("arch", "variant", "runtime"),
    _SIM_CASES + _ONBOARD_CASES,
)
def test_host_runtime_exports_required_pipeline_symbols(arch: str, variant: str, runtime: str):
    runtime_dir = _PROJECT_ROOT / "build" / "lib" / arch / variant / runtime
    runtime_libraries = tuple(runtime_dir.glob("libhost_runtime.*"))
    assert len(runtime_libraries) == 1, runtime_dir

    runtime_path = runtime_libraries[0]
    symbols = _defined_external_symbols(runtime_path)

    assert _NEWLY_REQUIRED_PIPELINE_SYMBOLS <= symbols, sorted(_NEWLY_REQUIRED_PIPELINE_SYMBOLS - symbols)
    assert _KERNEL_MODE_SYMBOLS <= symbols, sorted(_KERNEL_MODE_SYMBOLS - symbols)
    assert symbols.isdisjoint(_REMOVED_AMBIENT_SELECTION_SYMBOLS), sorted(symbols & _REMOVED_AMBIENT_SELECTION_SYMBOLS)


@pytest.mark.parametrize(("language", "standard", "compiler_name"), [("c", "c11", "cc"), ("c++", "c++17", "c++")])
@pytest.mark.parametrize(
    "headers",
    [
        ("execution_mode.h",),
        ("kernel_invocation_header.h",),
        ("execution_mode.h", "kernel_invocation_header.h"),
        ("kernel_invocation_header.h", "execution_mode.h"),
    ],
)
def test_execution_mode_headers_are_self_contained(language: str, standard: str, compiler_name: str, headers: tuple):
    compiler = shutil.which(compiler_name)
    assert compiler is not None, f"Header ABI tests require {compiler_name}"
    includes = "\n".join(f'#include "{header}"' for header in headers)
    source = includes + "\nSimplerExecutionMode mode = SIMPLER_MODE_KERNEL;\n"
    assertion = "_Static_assert" if language == "c" else "static_assert"
    source += f'{assertion}(SIMPLER_MODE_PROGRAM == 0 && SIMPLER_MODE_KERNEL == 1, "mode values");\n'
    result = subprocess.run(
        [
            compiler,
            "-x",
            language,
            f"-std={standard}",
            "-I",
            str(_PROJECT_ROOT / "src/common/task_interface"),
            "-fsyntax-only",
            "-",
        ],
        input=source,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_pipeline_contract_has_no_kernel_lifecycle_or_invocation_dependency():
    compiler = shutil.which("c++")
    assert compiler is not None, "Header dependency test requires c++"
    command = [compiler, "-x", "c++", "-std=c++17"]
    for directory in ("src/common", "src/common/task_interface", "src/common/platform/include"):
        command.extend(["-I", str(_PROJECT_ROOT / directory)])
    source = '#include "worker/pipeline_contract.h"\n'
    result = subprocess.run(command + ["-fsyntax-only", "-"], input=source, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    dependencies = subprocess.run(command + ["-M", "-"], input=source, capture_output=True, text=True, check=False)
    assert dependencies.returncode == 0, dependencies.stderr
    assert "execution_mode.h" in dependencies.stdout
    for header in ("kernel_invocation_header.h", "kernel_execution_state.h", "execution_mode_latch.h"):
        assert header not in dependencies.stdout
