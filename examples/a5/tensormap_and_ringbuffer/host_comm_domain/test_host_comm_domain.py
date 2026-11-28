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

import subprocess
import sys
from pathlib import Path

import pytest

from examples.a5.tensormap_and_ringbuffer.host_comm_domain.host_comm_callable import GATHER, RUNTIME
from simpler_setup import SceneTestCase, scene_test

# UB and HCCS both run the transfer checks. Only UB has been verified on hardware.
_PCIE_SKIP = "Cross-rank URMA reads of Host memory hit an ACK timeout with a PCIe Host-Device link"
pytestmark = pytest.mark.coverage_skip(platforms=["a5"], reason=_PCIE_SKIP)


@scene_test(level=3, runtime=RUNTIME)
class TestHostCommDomain(SceneTestCase):
    CALLABLE = {"callables": [GATHER]}
    CASES = [{"name": "host_comm_domain", "platforms": ["a5"], "config": {"device_count": 2}}]

    @pytest.mark.device_count(2)
    @pytest.mark.parametrize(
        "staged", [pytest.param(False, marks=pytest.mark.manual(["a5"])), True], ids=["serial", "staged"]
    )
    @pytest.mark.parametrize(
        "aiv_blocks", [pytest.param(1, marks=pytest.mark.manual(["a5"])), 4], ids=["single_aiv", "multi_aiv"]
    )
    def test_run(self, st_platform, st_device_ids, staged, aiv_blocks, request):
        if not self._matching_cases(st_platform, request):
            pytest.skip("Host communication domain case was not selected")
        devices = tuple(map(int, st_device_ids))
        assert len(devices) == len(set(devices)) == 2, f"expected two distinct devices, got {devices}"
        # The L3 worker forks its children, so the pytest parent must stay free of ACL state.
        probe = subprocess.run(
            [sys.executable, Path(__file__).with_name("probe_host_link.py"), *map(str, devices)],
            timeout=30,
            check=False,
        )
        if probe.returncode == 77:
            pytest.skip(_PCIE_SKIP)
        probe.check_returncode()
        from .run import run  # noqa: PLC0415 -- only the transfer path needs the worker

        compiled = self._compile_l3_callables(st_platform)["gather"]
        run(devices, compiled=compiled, staged=staged, aiv_blocks=aiv_blocks)
