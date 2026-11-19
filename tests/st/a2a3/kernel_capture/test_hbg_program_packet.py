# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Observe Program's HBG runtime descriptor and execution image at the RTS launch boundary."""

import os

import pytest

from tests.st.a2a3.kernel_capture.test_hbg_capture import _run_capture_probe, build_probe

pytestmark = [
    pytest.mark.requires_hardware,
    pytest.mark.platforms(["a2a3", "a5"]),
    pytest.mark.runtime("host_build_graph"),
]


def test_hbg_program_inline_packet(request, tmp_path):
    binaries = build_probe(tmp_path / "build", platform=request.config.getoption("--platform"))
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    result = _run_capture_probe([binaries[0], device, "16", *binaries[1:]], tmp_path, dict(os.environ))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "stage=program-inline-packet-with-references rc=0" in result.stdout
    assert result.stdout.count("stage=program-output rc=0") == 16
