# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Domain configuration and production A5 host resource ownership (no device)."""

import os
import subprocess
from pathlib import Path

import pytest
from simpler.comm_domain import DomainConfig

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("engines,mask", [((), 0), ([], 0), (["sdma"], 1), (["urma", "sdma"], 3), (["rdma"], 4)])
def test_domain_workspace_selection_is_part_of_configuration(engines, mask):
    config = DomainConfig.capture([0, 1], 4096, [], True, (), engines)
    assert config.workspace_mask() == mask
    if engines:
        reordered = DomainConfig.capture([0, 1], 4096, [], True, (), list(reversed(engines)))
        assert not config.difference(reordered)
    synchronous = DomainConfig.capture([0, 1], 4096, [], True, ())
    assert synchronous.workspace_engines == () and synchronous.workspace_mask() == 0
    if engines:
        assert config.difference(synchronous).startswith("workspace_engines:")
    else:
        assert not config.difference(synchronous)


@pytest.mark.parametrize("engines", [["unknown"], ["sdma", "sdma"], ["rdma", "sdma"], ["rdma", "urma"]])
def test_invalid_workspace_requests_are_rejected_before_allocation(engines):
    with pytest.raises(ValueError):
        DomainConfig.capture([0], 4096, [], False, (), engines)


@pytest.mark.parametrize("engines", ["sdma", None])
def test_workspace_names_require_a_sequence(engines):
    with pytest.raises(TypeError, match="sequence"):
        DomainConfig.capture([0], 4096, [], False, (), engines)


@pytest.mark.parametrize("urma,rdma", [(False, False), (True, False), (False, True), (True, True)])
def test_a5_host_options_allow_combined_builds(tmp_path, urma, rdma):
    source = (ROOT / "src/a5/platform/onboard/host/CMakeLists.txt").read_text()
    selection = source[: source.index("# Pin-resolved checkout path")]
    (tmp_path / "CMakeLists.txt").write_text(
        selection + '\nfile(WRITE "${CMAKE_BINARY_DIR}/selected" '
        '"${SIMPLER_ENABLE_PTO_SDMA_WORKSPACE};${SIMPLER_ENABLE_PTO_URMA_WORKSPACE};'
        '${SIMPLER_ENABLE_PTO_RDMA_WORKSPACE}")\n'
    )
    subprocess.run(
        [
            "cmake",
            "-S",
            str(tmp_path),
            "-B",
            str(tmp_path / "build"),
            f"-DSIMPLER_CMAKE_DIR={ROOT / 'cmake'}",
            f"-DCUSTOM_INCLUDE_DIRS={ROOT / 'src/common'}",
            f"-DSIMPLER_ENABLE_PTO_URMA_WORKSPACE={'ON' if urma else 'OFF'}",
            f"-DSIMPLER_ENABLE_PTO_RDMA_WORKSPACE={'ON' if rdma else 'OFF'}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (tmp_path / "build/selected").read_text() == f"ON;{'ON' if urma else 'OFF'};{'ON' if rdma else 'OFF'}"


@pytest.fixture(scope="module", params=[(False, False), (True, False), (False, True), (True, True)])
def host_resource_binary(request, tmp_path_factory):
    build = tmp_path_factory.mktemp("comm_workspace_host")
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann"))
    if not (cann / "include/hccl/hccl_comm.h").exists():
        pytest.skip("production host ownership fixture requires CANN headers, no device")
    for engine in ("sdma", "urma", "rdma"):
        header = build / f"pto/comm/async/{engine}/{engine}_workspace_manager.hpp"
        header.parent.mkdir(parents=True)
        header.write_text('#include "comm_workspace_managers.h"\n')
    log = build / "common/unified_log.h"
    log.parent.mkdir()
    log.write_text(
        "#pragma once\n"
        + "".join(f"#define LOG_{level}(...) ((void)0)\n" for level in ("ERROR", "WARN", "INFO", "DEBUG"))
    )
    urma, rdma = request.param
    defines = ["-DSIMPLER_ENABLE_PTO_SDMA_WORKSPACE=1"]
    if urma:
        defines.append("-DSIMPLER_ENABLE_PTO_URMA_WORKSPACE=1")
    if rdma:
        defines.append("-DSIMPLER_ENABLE_PTO_RDMA_WORKSPACE=1")
    binary = build / "host_resources"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-UNDEBUG",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wl,--gc-sections",
            *defines,
            f"-I{build}",
            f"-I{cann}/include",
            f"-I{cann}/pkg_inc/base",
            f"-I{ROOT}/src/a5/platform/include",
            f"-I{ROOT}/src/common/platform/include",
            f"-I{ROOT}/src/common",
            f"-I{ROOT}/src/common/task_interface",
            f"-I{ROOT}/src/a5/platform/onboard/host",
            f"-I{ROOT}/tests/ut/py/support",
            str(ROOT / "tests/ut/py/support/comm_workspaces_host.cpp"),
            "-ldl",
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


def test_workspace_domains_keep_resources_isolated_and_rollback_failures(host_resource_binary):
    subprocess.run([str(host_resource_binary)], check=True, timeout=20)


@pytest.mark.parametrize("kind", ["CommWorkspaceKind::COUNT", "static_cast<CommWorkspaceKind>(UINT32_MAX)"])
def test_invalid_static_workspace_kind_is_rejected_by_compiler(tmp_path, kind):
    source = tmp_path / "invalid_workspace.cpp"
    source.write_text(
        '#include "platform_comm/comm_context.h"\n'
        f"int main() {{ CommContext ctx{{}}; return comm_workspace<{kind}>(ctx).addr; }}\n"
    )
    result = subprocess.run(
        ["c++", "-std=c++17", f"-I{ROOT / 'src/common'}", "-c", str(source), "-o", str(tmp_path / "invalid.o")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "invalid communication workspace kind" in result.stderr
