#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Read retained Host tables on two A5 ranks using the HBG calendar."""

import pytest

from examples.a5.tensormap_and_ringbuffer.host_comm_domain import test_host_comm_domain as host_scene
from simpler_setup import scene_test

pytestmark = host_scene.pytestmark


@scene_test(level=3, runtime="host_build_graph")
class TestHbgHostGather(host_scene.TestHostCommDomain):
    CASES = [{"name": "hbg_host_gather", "platforms": ["a5"], "manual": ["a5"], "config": {"device_count": 2}}]


if __name__ == "__main__":
    # These lifecycle cases own their Workers across several Runs and use pytest's topology probe.
    import sys

    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
