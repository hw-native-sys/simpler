#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Focused SPSC reference scene: full ring, descriptor wrap, and a second run."""

from __future__ import annotations

import pytest

from .test_worker_chip_message_queue import run_worker_chip_message_queue_example

# The example kernel keeps four inputs outstanding. Depth 4 fills that window,
# so the STOP descriptor reuses slot 0. A second run uses a fresh Region.
_REFERENCE_DEPTH = 4


@pytest.mark.platforms(["a2a3sim", "a2a3", "a5sim", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime("tensormap_and_ringbuffer")
def test_spsc_reference_capacity_wrap_and_reuse(st_platform, st_device_ids):
    run_worker_chip_message_queue_example(
        st_platform, int(st_device_ids[0]), depth=_REFERENCE_DEPTH, repeats=2, prefill=True
    )
