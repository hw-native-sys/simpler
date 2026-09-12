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
