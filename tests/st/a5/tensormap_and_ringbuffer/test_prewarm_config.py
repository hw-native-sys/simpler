#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Regression test for runtime-arena prewarm during L2 worker initialization."""

import pytest
from simpler.task_interface import CallConfig
from simpler.worker import Worker

_RUNTIME = "tensormap_and_ringbuffer"


@pytest.mark.platforms(["a5sim"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_l2_init_with_prewarm_config(st_platform, st_device_ids):
    config = CallConfig()
    config.runtime_env.ring_task_window = 64

    worker = Worker(
        level=2,
        device_id=int(st_device_ids[0]),
        platform=st_platform,
        runtime=_RUNTIME,
    )
    try:
        worker.init(prewarm_config=config)
    finally:
        worker.close()
