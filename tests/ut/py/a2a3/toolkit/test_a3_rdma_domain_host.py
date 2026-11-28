# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Production RDMA owner and bootstrap with substituted external ACL/PTO providers; no device."""

import os
import subprocess
from pathlib import Path

import pytest

from simpler_setup.pto_isa import ensure_pto_isa_root

ROOT = Path(__file__).resolve().parents[5]

pytestmark = [pytest.mark.requires_toolkit, pytest.mark.platforms(["a2a3"])]


@pytest.fixture(scope="module")
def rdma_domain_binary(tmp_path_factory):
    build = tmp_path_factory.mktemp("a3_rdma_domain")
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    if not (cann / "include/acl/acl.h").is_file():
        pytest.skip("resource ownership fixture needs CANN declarations, no device")
    replacements = {
        "pto/comm/async/sdma/sdma_workspace_manager.hpp": "unused_sdma_workspace_manager.h",
        "communication/host/comm_context.hpp": "a3_rdma_domain_provider.h",
    }
    for target, source in replacements.items():
        header = build / target
        header.parent.mkdir(parents=True, exist_ok=True)
        header.write_text((ROOT / "tests/ut/py/support" / source).read_text())
    (build / "pto/pto-inst.hpp").write_text("#pragma once\n")
    log = build / "common/unified_log.h"
    log.parent.mkdir()
    log.write_text(
        "#pragma once\n"
        + "".join(f"#define LOG_{level}(...) ((void)0)\n" for level in ("ERROR", "WARN", "INFO", "DEBUG"))
    )
    pto = Path(ensure_pto_isa_root())
    includes = [
        build,
        pto / "include",
        pto / "pkg_inc",
        pto / "kernels/include",
        cann / "include",
        cann / "pkg_inc/base",
        ROOT / "src/common",
        ROOT / "src/common/worker",
        ROOT / "src/common/platform/include",
        ROOT / "src/common/task_interface",
        ROOT / "src/a2a3/platform/include",
        ROOT / "src/a2a3/platform/onboard/host",
    ]
    binary = build / "a3_rdma_domain"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-UNDEBUG",
            "-DSIMPLER_ENABLE_PTO_RDMA_WORKSPACE",
            "-DPTO_NPU_ARCH_A2A3",
            "-DPTO_COMM_WORKSPACE_RDMA_SUPPORTED=0",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wl,--gc-sections",
            *[f"-I{path}" for path in includes],
            str(ROOT / "tests/ut/py/support/a3_rdma_domain_host.cpp"),
            "-ldl",
            "-pthread",
            "-lrt",
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


@pytest.mark.parametrize(
    "scenario", ["success", "initialization", "cleanup", "gm_cleanup", "ctx_cleanup", "port", "exchange"]
)
def test_rdma_resource_ownership(rdma_domain_binary, scenario):
    subprocess.run([str(rdma_domain_binary), scenario], check=True, timeout=15)
