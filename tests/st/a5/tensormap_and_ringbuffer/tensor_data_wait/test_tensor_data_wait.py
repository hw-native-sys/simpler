#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A slow scalar producer can finish while independent tasks keep scheduling active."""

import ctypes

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestTensorDataWait(SceneTestCase):
    ATOL = 0
    RTOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/tensor_wait_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aiv/finite_producer.cpp",
                "core_type": "aiv",
                "signature": [D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": name,
            "platforms": ["a5sim"],
            "params": {"delay_ms": delay_ms},
            # Scalar reads block orchestration; keep a separate scheduler thread.
            "config": {"aicpu_thread_num": 2},
        }
        for name, delay_ms in [("short_producer", 2000), ("slow_producer", 18000)]
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("output", torch.zeros(1, dtype=torch.int32)),
            Scalar("delay_ms", ctypes.c_int64(params["delay_ms"])),
        )

    def compute_golden(self, args, params):
        args.output.fill_(7)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
