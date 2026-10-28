# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Actual A5 local Host mapping, gather and dependent consumer acceptance."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a5"])
@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
def test_local_host_ddr_gather(runtime, request):
    path = Path(__file__).resolve().parents[3] / "examples/a5/tensormap_and_ringbuffer/host_ddr_gather/run.py"
    device = int(str(request.config.getoption("--device")).split("-")[0].split(",")[0])
    # Runtime modules share the process's ACL context; isolate different runtimes.
    subprocess.run([sys.executable, str(path), "--device", str(device), "--runtime", runtime], check=True, timeout=180)


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a5"])
def test_local_host_mxfp8_with_device_scales(request):
    path = Path(__file__).resolve().parents[3] / "examples/a5/tensormap_and_ringbuffer/host_ddr_gather/run_mx.py"
    device = int(str(request.config.getoption("--device")).split("-")[0].split(",")[0])
    subprocess.run([sys.executable, str(path), "--device", str(device)], check=True, timeout=180)
