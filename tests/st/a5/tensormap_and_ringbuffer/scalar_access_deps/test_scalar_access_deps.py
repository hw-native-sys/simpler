#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Declared task dependencies for get/set_tensor_data (#2084).

Both positive cases run their producer in a manual scope, so the TensorMap holds
no entry for the tensor and the accessors can only order against the task ids
the orchestration declares.
"""

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

# Values written by kernel_write_five.cpp and by the WAR case's set_tensor_data.
PRODUCER_VALUE = 5.0
OVERWRITE_VALUE = 9.0

_ORCHESTRATION = {
    "source": "kernels/orchestration/scalar_access_deps_orch.cpp",
    "function_name": "aicpu_orchestration_entry",
    "signature": [D.INOUT, D.INOUT, D.INOUT],
}

_INCORES = [
    {
        "func_id": 0,
        "name": "WRITE_FIVE",
        "source": "kernels/aiv/kernel_write_five.cpp",
        "core_type": "aiv",
        "signature": [D.INOUT],
    },
    {
        "func_id": 1,
        "name": "COPY_FIRST",
        "source": "kernels/aiv/kernel_copy_first.cpp",
        "core_type": "aiv",
        "signature": [D.IN, D.INOUT],
    },
]


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestScalarAccessDeps(SceneTestCase):
    """get/set_tensor_data order against caller-declared task ids."""

    ATOL = 0
    RTOL = 0

    CALLABLE = {"orchestration": _ORCHESTRATION, "incores": _INCORES}

    CASES = [
        {
            "name": "read_manual_scope_producer",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 2},
            "params": {"case": 1},
        },
        {
            "name": "write_after_manual_scope_readers",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 2},
            "params": {"case": 2},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("x", torch.zeros(1, dtype=torch.float32)),
            TensorArg("y", torch.zeros(1, dtype=torch.float32)),
            TensorArg("w", torch.zeros(1, dtype=torch.float32)),
            Scalar("case", int(params["case"])),
        )

    def compute_golden(self, args, params):
        if params["case"] == 1:
            # The read observed the manual-scope producer's value.
            args.x.fill_(PRODUCER_VALUE)
            args.y.fill_(PRODUCER_VALUE)
        else:
            # The write waited for the producer and its reader: X holds the
            # orchestration value, while W holds what the reader saw before it.
            args.x.fill_(OVERWRITE_VALUE)
            args.w.fill_(PRODUCER_VALUE)


class _InvalidDepMixin:
    """Shared setup for the negative classes: the run must fail on its deps."""

    ATOL = 0
    RTOL = 0

    CALLABLE = {"orchestration": _ORCHESTRATION, "incores": _INCORES}

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("x", torch.zeros(1, dtype=torch.float32)),
            TensorArg("y", torch.zeros(1, dtype=torch.float32)),
            TensorArg("w", torch.zeros(1, dtype=torch.float32)),
            Scalar("case", int(params["case"])),
        )

    def compute_golden(self, args, params):
        pass

    def test_run(self, st_platform, st_worker, request):
        with pytest.raises(RuntimeError, match=r"INVALID_ARGS|code -5"):
            super().test_run(st_platform, st_worker, request)


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestScalarAccessDepsUnsubmitted(_InvalidDepMixin, SceneTestCase):
    """A declared task id below no submitted task is INVALID_ARGS."""

    CASES = [
        {
            "name": "unsubmitted_task_id",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 2},
            "params": {"case": 3},
        },
    ]


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestScalarAccessDepsNull(_InvalidDepMixin, SceneTestCase):
    """A null deps array with a non-zero count is INVALID_ARGS."""

    CASES = [
        {
            "name": "null_deps_with_count",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 2},
            "params": {"case": 4},
        },
    ]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
