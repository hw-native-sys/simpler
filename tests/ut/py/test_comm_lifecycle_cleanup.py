# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fault injection for the C++ lifecycle test's rank cleanup, without a device."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def lifecycle_rank_binary(tmp_path_factory):
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann"))
    if not (cann / "include/acl/acl.h").is_file():
        pytest.skip("lifecycle rank fixture requires CANN headers, no device")
    binary = tmp_path_factory.mktemp("comm_lifecycle_rank") / "rank_cleanup"
    result = subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-UNDEBUG",
            '-DPTO_HOST_RUNTIME_LIB_PATH="unused-host-runtime"',
            f"-I{cann}/include",
            f"-I{ROOT}/src/common",
            f"-I{ROOT}/src/common/platform/include",
            str(ROOT / "tests/ut/py/support/comm_lifecycle_rank_host.cpp"),
            "-o",
            str(binary),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.parametrize(
    "scenario",
    ["success", "barrier_unconfirmed", "alloc_unconfirmed", "destroy_unconfirmed", "barrier_error", "destroy_error"],
)
def test_rank_preserves_first_failure_and_skips_cleanup_when_unconfirmed(lifecycle_rank_binary, scenario):
    subprocess.run([str(lifecycle_rank_binary), scenario], check=True, timeout=5)
