#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""available_aicore_counts: rt_available_cluster_count() returns the run's AIC cluster count.

The orchestration reads the this-run MIX cluster (= AIC) count through the ops
table and writes it into a single int32 output tensor; the host compares it
against the count the a2a3 sim platform exposes. A regression that returns the
host placeholder, 0, or a stale ratio fails the comparison.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

# a2a3sim exposes a fixed AIC cluster count (cores_total_num / 3); pinned from
# the value the orchestration surfaces on this platform.
EXPECTED_AIC_COUNT = 1


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestAvailableAicoreCounts(SceneTestCase):
    """rt_available_cluster_count() surfaces the this-run AIC cluster count."""

    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/available_aicore_counts_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [],
    }

    CASES = [
        {
            "name": "Default",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 2, "block_dim": 1},
            "params": {},
        },
    ]

    def generate_args(self, params):
        out = torch.zeros((1,), dtype=torch.int32)
        return TaskArgsBuilder(Tensor("out", out))

    def compute_golden(self, args, params):
        args.out[0] = EXPECTED_AIC_COUNT


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
