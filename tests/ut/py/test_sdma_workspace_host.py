# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Production A3 SDMA provider ownership under initialization failures (no device)."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def sdma_workspace_binary(tmp_path_factory):
    build = tmp_path_factory.mktemp("sdma_workspace_host")
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    if not (cann / "include/acl/acl.h").exists():
        pytest.skip("A3 SDMA host fixture requires CANN headers (no device)")
    sdma = build / "pto/comm/async/sdma/sdma_workspace_manager.hpp"
    sdma.parent.mkdir(parents=True)
    # The C++ fixture defines the fault-injecting manager before including the provider.
    sdma.write_text("#pragma once\n")
    log = build / "common/unified_log.h"
    log.parent.mkdir()
    log.write_text(
        "#pragma once\n"
        + "".join(f"#define LOG_{level}(...) ((void)0)\n" for level in ("ERROR", "WARN", "INFO", "DEBUG"))
    )
    includes = [
        build,
        cann / "include",
        cann / "pkg_inc/base",
        ROOT / "src/common",
        ROOT / "src/common/worker",
        ROOT / "src/common/platform/include",
        ROOT / "src/common/task_interface",
        ROOT / "src/a2a3/platform/include",
        ROOT / "src/a2a3/platform/onboard/host",
    ]
    binary = build / "sdma_workspace_host"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-UNDEBUG",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wl,--gc-sections",
            *[f"-I{path}" for path in includes],
            str(ROOT / "tests/ut/py/support/sdma_workspace_host.cpp"),
            "-ldl",
            "-pthread",
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


@pytest.mark.parametrize(
    "scenario",
    [
        "before_bind",
        "throw_before_bind",
        "after_bind",
        "throw_after_bind",
        "zero_size",
        "allocation_failure",
        "success",
        "before_bind_free_failure",
        "release_free_failure",
    ],
)
def test_sdma_workspace_ownership(sdma_workspace_binary, scenario):
    subprocess.run([str(sdma_workspace_binary), scenario], check=True, timeout=5)
