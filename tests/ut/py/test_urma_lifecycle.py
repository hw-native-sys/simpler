# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real pinned URMA manager and production host cleanup with stateful HCCL mocks."""

import os
import subprocess
from pathlib import Path

import pytest

from simpler_setup.pto_isa import ensure_pto_isa_root

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def urma_lifecycle_binary(tmp_path_factory):
    build = tmp_path_factory.mktemp("urma_lifecycle")
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann"))
    channel = cann / "include/hccl/hccl_channel.h"
    if not channel.exists() or "HcclChannelAcquireWithConfig" not in channel.read_text():
        pytest.skip("real URMA manager fixture requires current A5 CANN headers (no device)")
    isa = Path(ensure_pto_isa_root())
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
        isa / "include",
        isa / "pkg_inc",
        cann / "include",
        cann / "pkg_inc/base",
        ROOT / "src/common",
        ROOT / "src/common/platform/include",
        ROOT / "src/common/task_interface",
        ROOT / "src/a5/platform/include",
        ROOT / "src/a5/platform/onboard/host",
    ]
    binary = build / "urma_lifecycle"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-UNDEBUG",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wl,--gc-sections",
            *[f"-I{p}" for p in includes],
            str(ROOT / "tests/ut/py/support/urma_lifecycle_host.cpp"),
            "-ldl",
            "-lrt",
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
        "notify_registration",
        "channel_acquire",
        "destroy_failure",
        "context_upload",
        "late_import",
        "early_get_device",
        "peer_import",
        "peer_reserve",
        "peer_map",
        "peer_access",
        "success",
        "undrained",
        "cqe_pending",
        "base_destroy",
        "abort_unpublished",
        "release_destroy_failure",
        "finalize_live",
        "finalize_undrained",
    ],
)
def test_urma_lifecycle(urma_lifecycle_binary, scenario):
    subprocess.run(
        [str(urma_lifecycle_binary), scenario],
        check=True,
        timeout=5 if scenario == "abort_unpublished" else 20,
    )
