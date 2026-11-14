# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SDMA adapter admission and failure results at the external SDK boundary."""

import shutil
import subprocess
from pathlib import Path

import pytest

from simpler_setup.pto_isa import ensure_pto_isa_root

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def sdma_submission_binary(tmp_path_factory):
    compiler = shutil.which("g++-15")
    if compiler is None:
        pytest.skip("SDMA adapter fixture requires the PTO CPU compiler g++-15")
    build = tmp_path_factory.mktemp("a2a3_tmr_sdma_submission")
    binary = build / "submission"
    includes = [
        ROOT / "src/a2a3/runtime/tensormap_and_ringbuffer/runtime",
        ROOT / "src/a2a3/runtime/tensormap_and_ringbuffer/common",
        ROOT / "src/a2a3/platform/include",
        ROOT / "src/common/platform/include",
        ROOT / "src/common/task_interface",
        ROOT / "src/common/log/include",
        ROOT / "src/common/tensormap_and_ringbuffer",
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
            str(ROOT / "tests/ut/py/support/a2a3_tmr_sdma_submission.cpp"),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.parametrize("put", [0, 1], ids=["get", "put"])
@pytest.mark.parametrize(
    "scenario",
    [
        "full",
        "partial",
        "invalid_count",
        "exact",
        "prior_error",
        "missing_entries",
        "post_failed",
        "sync_failed",
        "wrong_engine",
        "too_many_records",
        "empty",
        "sync_empty",
        "missing_count",
        "session_failed",
        "failed_with_records",
        "invalid_handle",
        "missing_records",
        "sdk_rejected",
        "sync_sdk_rejected",
    ],
)
def test_sdma_submission_preserves_tracking_and_failure(sdma_submission_binary, scenario, put):
    result = subprocess.run(
        [str(sdma_submission_binary), scenario, str(put)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
