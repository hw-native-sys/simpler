#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Negative ST for explicit orchestration fatal reporting."""

import pytest
from _task_interface import ChipStorageTaskArgs

from simpler_setup import SceneTestCase, scene_test


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class _ExplicitFatal(SceneTestCase):
    __test__ = False
    CALLABLE = {
        "orchestration": {
            "source": "explicit_fatal/kernels/orchestration/explicit_fatal_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [],
        },
        "incores": [],
    }


@pytest.mark.platforms(["a2a3sim"])
@pytest.mark.device_count(1)
def test_explicit_fatal_reports(st_platform, st_device_ids, monkeypatch):
    monkeypatch.setenv("PTO_LOG_LEVEL", "error")

    callable_obj = _ExplicitFatal.compile_chip_callable(st_platform)
    worker = _ExplicitFatal._create_worker(st_platform, st_device_ids[0])

    try:
        with pytest.raises(RuntimeError, match=r"run_runtime failed with code -9"):
            worker.run(callable_obj, ChipStorageTaskArgs(), block_dim=24, aicpu_thread_num=4)
    finally:
        worker.finalize()
