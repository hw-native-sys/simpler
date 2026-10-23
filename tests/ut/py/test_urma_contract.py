# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile the adapter against the pin-resolved PTO declarations, without hardware."""

import os
import subprocess
from pathlib import Path

import pytest

from simpler_setup.pto_isa import ensure_pto_isa_root

ROOT = Path(__file__).resolve().parents[3]


def test_urma_workspace_matches_pinned_pto(tmp_path):
    pto = ensure_pto_isa_root()
    source = tmp_path / "abi.cpp"
    source.write_text('#include "common/urma_workspace_contract.h"\nint main() {}\n')
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-fsyntax-only",
            f"-I{tmp_path}",
            f"-I{pto}/include",
            f"-I{ROOT}/src/a5/platform/include",
            str(source),
        ],
        check=True,
    )


@pytest.fixture(scope="module")
def urma_host_lifecycle_binary(tmp_path_factory):
    build = tmp_path_factory.mktemp("urma_host")
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann"))
    if not (cann / "include/hccl/hccl_comm.h").exists():
        pytest.skip("native host lifecycle fixture requires CANN headers (no device)")
    pto = ensure_pto_isa_root()
    manager = build / "pto/comm/async/urma/urma_workspace_manager.hpp"
    manager.parent.mkdir(parents=True)
    manager.write_text(r"""
#pragma once
#include <cassert>
#include <vector>
#include "pto/comm/async/urma/urma_types.hpp"
extern int live_comms, finalized;
extern std::vector<int> order;
namespace pto::comm::urma {
class UrmaWorkspaceManager {
public:
    ~UrmaWorkspaceManager() { assert(live_comms == 0); ++finalized; order.push_back(3); }
    template <typename... Args> bool Init(Args...) { return false; }
    void *GetWorkspaceAddr() const { return nullptr; }
};
}
""")
    log = build / "common/unified_log.h"
    log.parent.mkdir()
    log.write_text(
        "#pragma once\n"
        + "".join(f"#define LOG_{level}(...) ((void)0)\n" for level in ("ERROR", "WARN", "INFO", "DEBUG"))
    )
    binary = build / "lifecycle"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wl,--gc-sections",
            "-DSIMPLER_ENABLE_PTO_URMA_WORKSPACE=1",
            '-DSIMPLER_PTO_ISA_BUILD_COMMIT="test"',
            f"-I{build}",
            f"-I{cann}/include",
            f"-I{cann}/pkg_inc/base",
            f"-I{pto}/include",
            f"-I{ROOT}/src/a5/platform/include",
            f"-I{ROOT}/src/common/platform/include",
            f"-I{ROOT}/src/common",
            f"-I{ROOT}/src/common/task_interface",
            f"-I{ROOT}/src/a5/platform/onboard/host",
            str(ROOT / "tests/ut/py/support/urma_host_lifecycle.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
    )
    return binary


@pytest.mark.parametrize(
    "scenario",
    [
        "normal",
        "domains",
        "retired_domains",
        "init_failure",
        "domain_init_failure",
        "handshake_failure",
        "stop_failure",
        "drain_failure",
        "destroy_failure",
        "child_destroy_failure",
    ],
)
def test_urma_host_lifecycle(urma_host_lifecycle_binary, scenario):
    subprocess.run([str(urma_host_lifecycle_binary), scenario], check=True, timeout=15)
