#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Only the last block of an SPMD task releases what depends on it.

A block completes the way any dispatch does, and the scheduler has to tell
"this dispatch finished" from "this task finished" -- the second is true only
once every block has reported. Getting that wrong releases successors onto data
their producer has not finished writing.

Nothing else in the suite can see it. The unit tests pin the counter, which is
the mechanism, but not the data a kernel actually reads; and every other scene
test's golden depends only on each task's own output, so a successor that ran
too early still computes the right answer from values that happened to be there
already.

Two things make the window real here. The consumer reads *every* block's slot,
so one unwritten slot is enough to change its answer; and every block but the
first pays a delay, so a consumer released when the first finished starts while
the rest are still running. Without the delay all blocks land at once and a
premature release is indistinguishable from a correct one.
"""

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

FLOATS_PER_CACHE_LINE = 16
BLOCK_NUM = 8
# Long enough that a block still holds its core while a successor released on
# the first block's completion would be reading. Short enough not to dominate
# the suite.
REPEATS = 200000
# What an unwritten slot holds. Negative so that no partial sum of block
# indices can reach the golden by accident: every missing block subtracts at
# least one more than the largest index could add.
UNWRITTEN = -1.0


@pytest.mark.scheduler_mode("resident")
@scene_test(level=2, runtime="host_build_graph")
class TestSpmdLastBlockReleases(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_last_block_releases_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.IN, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aic/kernel_block_slow.cpp",
                "core_type": "aic",
                "signature": [D.INOUT, D.IN],
            },
            {
                "func_id": 1,
                "source": "kernels/aic/kernel_sum_blocks.cpp",
                "core_type": "aic",
                "signature": [D.INOUT, D.IN],
            },
        ],
    }

    CASES = [
        {
            "name": "sum_of_8_blocks",
            "platforms": ["a5sim", "a5"],
            "params": {"blocks": BLOCK_NUM, "repeats": REPEATS},
        },
    ]

    def generate_args(self, params):
        # One slot per block, plus one for the consumer's sum.
        slots = (params["blocks"] + 1) * FLOATS_PER_CACHE_LINE
        return TaskArgsBuilder(
            TensorArg("out", torch.full((slots,), UNWRITTEN, dtype=torch.float32)),
            Scalar("blocks", params["blocks"]),
            Scalar("repeats", params["repeats"]),
        )

    def compute_golden(self, args, params):
        blocks = params["blocks"]
        for block in range(blocks):
            args.out[block * FLOATS_PER_CACHE_LINE] = float(block)
        args.out[blocks * FLOATS_PER_CACHE_LINE] = float(sum(range(blocks)))


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
