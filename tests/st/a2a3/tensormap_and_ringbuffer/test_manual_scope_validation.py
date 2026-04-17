#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Negative ST for manual scope validation."""

from _task_interface import ChipStorageTaskArgs
import pytest

from simpler_setup import SceneTestCase, scene_test


def _make_case(orch_source: str):
    @scene_test(level=2, runtime="tensormap_and_ringbuffer")
    class _ManualScopeValidation(SceneTestCase):
        __test__ = False
        CALLABLE = {
            "orchestration": {
                "source": f"manual_scope_validation/kernels/orchestration/{orch_source}",
                "function_name": "aicpu_orchestration_entry",
                "signature": [],
            },
            "incores": [
                {
                    "func_id": 0,
                    "source": "manual_scope_validation/kernels/aiv/aiv_noop.cpp",
                    "core_type": "aiv",
                    "signature": [],
                }
            ],
        }

    return _ManualScopeValidation


@pytest.mark.platforms(["a2a3sim"])
@pytest.mark.device_count(1)
@pytest.mark.parametrize(
    "orch_source",
    [
        "outside_scope_add_dep.cpp",
        "nested_manual_scope.cpp",
        "foreign_manual_dep.cpp",
    ],
)
def test_manual_scope_invalid_usage_reports_invalid_args(st_platform, st_device_ids, monkeypatch, orch_source):
    monkeypatch.setenv("PTO_LOG_LEVEL", "error")

    case = _make_case(orch_source)
    callable_obj = case.compile_chip_callable(st_platform)
    worker = case._create_worker(st_platform, st_device_ids[0])

    try:
        with pytest.raises(RuntimeError, match=r"run_runtime failed with code -5"):
            worker.run(callable_obj, ChipStorageTaskArgs(), block_dim=24, aicpu_thread_num=4)
    finally:
        worker.finalize()
