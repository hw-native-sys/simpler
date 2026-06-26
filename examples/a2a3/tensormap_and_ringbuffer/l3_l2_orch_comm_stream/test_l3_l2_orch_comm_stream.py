#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Example entry for the L3-L2 in-flight orchestration stream."""

from __future__ import annotations

import pytest

from examples.a2a3.tensormap_and_ringbuffer.l3_l2_orch_comm_stream.l3_l2_orch_comm_stream import (
    run_closed_loop_stream,
)


@pytest.mark.platforms(["a2a3sim", "a2a3"])
@pytest.mark.device_count(1)
@pytest.mark.runtime("tensormap_and_ringbuffer")
def test_l3_l2_orch_comm_stream(st_platform, st_device_ids):
    run_closed_loop_stream(st_platform, int(st_device_ids[0]))
