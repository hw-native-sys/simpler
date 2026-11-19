#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""HBG runtime-owned SDMA workspace injection and prefetch/copy completion.

Simulation checks address injection; onboard execution drives the SDMA engine.
"""

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


@pytest.mark.sdma
@scene_test(level=2, runtime="host_build_graph")
class TestHbgPrefetchAsyncDemo(SceneTestCase):
    RTOL = 0
    ATOL = 0
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/prefetch_async_orch.cpp",
            "function_name": "prefetch_async_orchestration",
            "signature": [D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "../../tensormap_and_ringbuffer/prefetch_async_demo/kernels/aiv/kernel_prefetch_copy.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
        ],
    }
    CASES = [{"name": "prefetch_copy", "platforms": ["a2a3sim", "a2a3"], "params": {}}]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("src", torch.arange(128, dtype=torch.float32) / 8.0),
            TensorArg("out", torch.full((128,), -1.0, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        args.out[:] = args.src


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
