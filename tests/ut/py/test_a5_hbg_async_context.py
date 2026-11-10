# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A5 HBG kernel context ABI with real PTO CPU headers; no device required."""

import shutil
import subprocess
from pathlib import Path

import pytest

from simpler_setup.pto_isa import ensure_pto_isa_root

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def async_context_binary(tmp_path_factory):
    compiler = shutil.which("g++-15")
    if compiler is None:
        pytest.skip("A5 kernel context fixture requires the PTO CPU compiler g++-15")
    assert compiler is not None
    build = tmp_path_factory.mktemp("a5_hbg_async_context")
    binary = build / "context"
    includes = [
        ROOT / "src/a5/runtime/host_build_graph/runtime",
        ROOT / "src/a5/runtime/host_build_graph/common",
        ROOT / "src/a5/runtime/host_build_graph/host",
        ROOT / "src/a5/platform/include",
        ROOT / "src/common/platform/include",
        ROOT / "src/common/task_interface",
        ROOT / "src/common/log/include",
        ROOT / "src/common/host_build_graph",
        ROOT / "src/common",
        Path(ensure_pto_isa_root()) / "include",
    ]
    result = subprocess.run(
        [
            compiler,
            "-std=c++23",
            "-D__CPU_SIM",
            "-DPTO_CPU_MAX_THREADS=1",
            "-O1",
            *[f"-I{path}" for path in includes],
            str(ROOT / "tests/ut/py/support/a5_hbg_async_context.cpp"),
            str(ROOT / "src/a5/runtime/host_build_graph/host/async_completion_plan.cpp"),
            "-o",
            str(binary),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.parametrize("scenario", range(37))
def test_kernel_context_preserves_resident_and_legacy_boundaries(async_context_binary, scenario):
    result = subprocess.run([str(async_context_binary), str(scenario)], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
