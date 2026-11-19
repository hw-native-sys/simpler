# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real L3 Host DDR MAP examples, using devices reserved by the test runner."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a5"])
@pytest.mark.device_count(2)
@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("example", ["run.py", "run_mx.py"])
def test_l3_host_ddr_map(st_device_ids, staged, example):
    devices = ",".join(str(value) for value in st_device_ids)
    path = Path(__file__).resolve().parents[3] / "examples/a5/tensormap_and_ringbuffer/host_ddr_map" / example
    command = [sys.executable, str(path), "--devices", devices]
    if staged:
        command.append("--staged")
    subprocess.run(command, check=True, timeout=300)
