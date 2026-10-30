# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The real A3 simulator executors must drain skipped runs without dispatch."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("runtime", ["host_build_graph", "tensormap_and_ringbuffer"])
def test_actual_executor_skip_drains_without_dispatch(runtime, tmp_path):
    """Use built simulator DSOs, with synthetic cores answering real registers."""
    library = _ROOT / "build/lib/a2a3/sim" / runtime / "libaicpu_kernel.so"
    compiler = shutil.which("c++")
    if not library.exists() or compiler is None:
        pytest.skip("requires built A3 simulator runtime and a C++ compiler")
    source = _ROOT / "tests/ut/py/support/native_execution_skip_sim_probe.cpp"
    includes = [
        f"src/a2a3/runtime/{runtime}/runtime",
        f"src/a2a3/runtime/{runtime}/common",
        "src/a2a3/platform/include",
        "src/common/platform/include",
        "src/common/platform/sim/aicpu",
        "src/common/task_interface",
        "src/common/log/include",
        "src/common",
        f"src/common/{runtime}",
    ]
    binary = tmp_path / "skip_probe"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-pthread",
            *(["-DSIMPLER_TEST_TMR=1"] if runtime == "tensormap_and_ringbuffer" else []),
            *[f"-I{_ROOT / path}" for path in includes],
            str(source),
            str(library),
            f"-Wl,-rpath,{library.parent}",
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=30)
    assert result.stdout.count("rc=-71 dispatched=0 drained=3 terminal=-71") == 3
    # The descriptor intentionally lacks an arena/callable. If the frozen Skip
    # is ignored, the ordinary path cannot reproduce the original error. This
    # negative control proves the probe distinguishes that path.
    result = subprocess.run(
        [str(binary)],
        check=False,
        env={**os.environ, "SIMPLER_TEST_DISABLE_SKIP": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2

    if runtime == "tensormap_and_ringbuffer":
        assert "negative-control scheduler=0 orchestrator=-1 drained=3 terminal=-1 source=ThreadRc" in result.stdout
