# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A3 communication allocation views through the production host backend."""

import os
import subprocess
from pathlib import Path

import pytest

from simpler_setup.pto_isa import ensure_pto_isa_root

ROOT = Path(__file__).resolve().parents[5]

pytestmark = [pytest.mark.requires_toolkit, pytest.mark.platforms(["a2a3"])]


def test_a3_domain_query_attests_owned_views_without_device(tmp_path):
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann"))
    if not (cann / "include/hccl/hccl_comm.h").exists():
        pytest.skip("native A3 metadata query fixture requires CANN headers, no device")
    log = tmp_path / "common/unified_log.h"
    log.parent.mkdir()
    log.write_text(
        "#pragma once\n"
        + "".join(f"#define LOG_{level}(...) ((void)0)\n" for level in ("ERROR", "WARN", "INFO", "DEBUG"))
    )
    isa = Path(ensure_pto_isa_root())
    binary = tmp_path / "a3_domain_views"
    compile_result = subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-UNDEBUG",
            "-DPTO_COMM_WORKSPACE_SDMA_SUPPORTED=1",
            "-DPTO_COMM_WORKSPACE_URMA_SUPPORTED=0",
            "-DPTO_COMM_WORKSPACE_RDMA_SUPPORTED=0",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wl,--gc-sections",
            f"-I{tmp_path}",
            f"-I{isa}/include",
            f"-I{cann}/include",
            f"-I{cann}/pkg_inc/base",
            f"-I{ROOT}/src/a2a3/platform/include",
            f"-I{ROOT}/src/common/platform/include",
            f"-I{ROOT}/src/common",
            f"-I{ROOT}/src/common/task_interface",
            f"-I{ROOT}/src/common/worker",
            f"-I{ROOT}/src/a2a3/platform/onboard/host",
            str(Path(__file__).with_name("a3_domain_allocation_views.cpp")),
            "-ldl",
            "-pthread",
            "-o",
            str(binary),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    subprocess.run([str(binary)], check=True, timeout=20)
