# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real A5 Host allocation/IPC/abort code with CPU VMM and workspace-owner mocks.

No device or DMA is used. These tests qualify the allocation ownership protocol,
not public SDK completion records, metadata visibility, or engine behavior.
"""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def sdma_owner_actor(tmp_path_factory):
    build = tmp_path_factory.mktemp("sdma_owner_actor")
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    if not (cann / "include/acl/acl.h").exists():
        pytest.skip("Host IPC actor needs CANN headers, but no device")
    stubs = {
        "pto/comm/async/sdma/sdma_workspace_manager.hpp": """
#pragma once
#include <cstdint>
namespace pto::comm::sdma {
class SdmaWorkspaceManager {
    void* address_ = nullptr; uint64_t bytes_ = 0;
public:
    static int GetWorkspaceSize(uint64_t* bytes) { *bytes = 4096; return 0; }
    bool Init(void* address, uint64_t bytes) { address_ = address; bytes_ = bytes; return true; }
    void* GetWorkspaceAddr() const { return address_; }
    uint64_t GetWorkspaceSize() const { return bytes_; }
};
}
""",
        "pto/comm/async/urma/urma_workspace_manager.hpp": """
#pragma once
#include <cstdint>
#include <stdexcept>
namespace pto::comm::urma {
class UrmaWorkspaceManager {
public:
    static int GetWorkspaceSize(uint32_t, uint64_t*) { throw std::logic_error("URMA outside actor"); }
    template<class... Args> bool Init(Args...) { throw std::logic_error("URMA outside actor"); }
    void* GetWorkspaceAddr() const { return nullptr; }
    uint64_t GetWorkspaceSize() const { return 0; }
};
}
""",
        "pto/comm/async/urma/urma_types.hpp": """
#pragma once
#include <cstdint>
// Unreachable URMA compatibility declarations. This actor never requests URMA.
namespace pto::comm::urma {
enum class UrmaLayout { PER_PEER };
struct UrmaInfo { UrmaLayout layout; uint32_t rankCount, rowsPerPeer, jettyCount; uint64_t sqPtr, scqPtr; };
struct UrmaWQCtx { uint64_t headAddr, tailAddr; uint32_t submittedWqeCount; };
struct UrmaCqCtx { uint64_t tailAddr; };
}
""",
        "common/unified_log.h": "#pragma once\n"
        + "".join(f"#define LOG_{level}(...) ((void)0)\n" for level in ("ERROR", "WARN", "INFO", "DEBUG")),
    }
    for name, content in stubs.items():
        path = build / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    includes = [
        build,
        cann / "include",
        cann / "pkg_inc/base",
        ROOT / "src/common",
        ROOT / "src/common/worker",
        ROOT / "src/common/platform/include",
        ROOT / "src/common/task_interface",
        ROOT / "src/a5/platform/include",
        ROOT / "src/a5/platform/onboard/host",
    ]
    binary = build / "sdma_owner_actor"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-UNDEBUG",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wl,--gc-sections",
            *[f"-I{path}" for path in includes],
            str(ROOT / "tests/ut/py/support/a5_sdma_owner_host.cpp"),
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


@pytest.mark.parametrize("late_rank", [0, 1])
@pytest.mark.parametrize("scenario", ["legacy_conflict", "q_conflict", "request_mismatch", "mixed_entry"])
def test_actual_announcements_converge_before_p2p_on_rejected_owner(sdma_owner_actor, scenario, late_rank):
    subprocess.run([str(sdma_owner_actor), scenario, str(late_rank)], check=True, timeout=9)


@pytest.mark.parametrize("scenario", ["managed", "legacy"])
def test_workspace_configuration_survives_actual_domain_release(sdma_owner_actor, scenario):
    subprocess.run([str(sdma_owner_actor), scenario, "0"], check=True, timeout=9)


@pytest.mark.parametrize("late_rank", [0, 1])
def test_unpublished_failed_context_does_not_fix_a_partition(sdma_owner_actor, late_rank):
    subprocess.run([str(sdma_owner_actor), "upload_failure", str(late_rank)], check=True, timeout=9)
