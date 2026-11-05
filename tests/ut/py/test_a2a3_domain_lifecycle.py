# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Production A2/A3 Fabric and IPC initialization rollback with stateful ACL mocks."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def a2a3_lifecycle_binary(tmp_path_factory):
    build = tmp_path_factory.mktemp("a2a3_lifecycle")
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    if not (cann / "include/acl/acl.h").exists():
        pytest.skip("A2/A3 host fixture requires CANN headers (no device)")
    sdma = build / "pto/comm/async/sdma/sdma_workspace_manager.hpp"
    sdma.parent.mkdir(parents=True)
    sdma.write_text((ROOT / "tests/ut/py/support/unused_sdma_workspace_manager.h").read_text())
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
    binary = build / "a2a3_lifecycle"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-UNDEBUG",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wl,--gc-sections",
            *[f"-I{p}" for p in includes],
            str(ROOT / "tests/ut/py/support/a2a3_domain_lifecycle_host.cpp"),
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


@pytest.mark.parametrize("transport", ["fabric", "ipc"])
@pytest.mark.parametrize(
    "scenario", ["success", "late_import", "early_get_device", "peer_import", "peer_reserve", "peer_map"]
)
def test_a2a3_domain_lifecycle(a2a3_lifecycle_binary, transport, scenario):
    subprocess.run([str(a2a3_lifecycle_binary), transport, scenario], check=True, timeout=5)


def test_a2a3_fallback_stops_when_cleanup_reports_an_error_after_consumption(a2a3_lifecycle_binary):
    subprocess.run([str(a2a3_lifecycle_binary), "ipc", "fallback_cleanup"], check=True, timeout=5)


@pytest.mark.parametrize("transport", ["fabric", "ipc"])
def test_a2a3_mapped_owner_needs_no_explicit_access_grant(a2a3_lifecycle_binary, transport):
    subprocess.run([str(a2a3_lifecycle_binary), transport, "map_grants_access"], check=True, timeout=5)
