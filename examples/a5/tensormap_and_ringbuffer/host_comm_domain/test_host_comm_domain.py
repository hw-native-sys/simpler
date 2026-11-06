#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hardware coverage of retained Host-source A5 domains and Device-domain coexistence."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest


def _driver():
    name = "_a5_host_comm_domain_driver"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("run.py"))
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load the Host communication-domain example")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.mark.platforms(["a5"])
@pytest.mark.manual(["a5"])
@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="A5 PCIe CI Host-source URMA ACK timeout; DT validation passes (issue #2922)",
)
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(2)
@pytest.mark.parametrize("staged", [False, True], ids=["serial", "staged"])
@pytest.mark.parametrize("aiv_blocks", [1, 4], ids=["single_aiv", "multi_aiv"])
def test_host_comm_domain(st_platform, st_device_ids, staged, aiv_blocks):
    assert st_platform == "a5"
    _driver().run(tuple(int(device) for device in st_device_ids[:2]), staged=staged, aiv_blocks=aiv_blocks)
