#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A Mix block closes on its last subtask, not its first.

Aggregating a Mix SPMD task is two levels deep, and the levels are easy to
collapse. Within a block, the claiming AIC tracks its subtasks locally and the
block is done when all of them have reported. Across blocks, a counter the task
owns reaches the task's width and only then is the task done. Count subtasks
where blocks are meant and a task of N blocks reports done after N subtasks --
a third of the way through, for a block of three.

spmd_last_block_releases makes the same shape of assertion for an AIC-only
task, where every dispatch is a whole block and only the outer level exists.
What is untested without this case is the inner one, and it is only observable
when a block's subtasks finish far apart: here the AIC half spins while the two
AIV halves return at once, so a block's first report and its last are a long
way from each other, and a consumer released on the wrong one lands in between.
"""

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

FLOATS_PER_CACHE_LINE = 16
SLOTS_PER_BLOCK = 2
BLOCK_NUM = 6
REPEATS = 200000
# Negative so a sum missing any slot cannot land on the golden by accident.
UNWRITTEN = -1.0


@pytest.mark.scheduler_mode("resident")
@scene_test(level=2, runtime="host_build_graph")
class TestSpmdMixBlockClosesLast(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_mix_block_closes_last_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.IN, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aic/kernel_block_slow_half.cpp",
                "core_type": "aic",
                "signature": [D.INOUT, D.IN],
            },
            {
                "func_id": 1,
                "source": "kernels/aiv/kernel_block_fast_half.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.IN],
            },
            {
                "func_id": 2,
                "source": "kernels/aic/kernel_sum_halves.cpp",
                "core_type": "aic",
                "signature": [D.INOUT, D.IN],
            },
        ],
    }

    CASES = [
        {
            "name": "mix_blocks_6",
            "platforms": ["a5sim", "a5"],
            "params": {"blocks": BLOCK_NUM, "repeats": REPEATS},
        },
    ]

    def generate_args(self, params):
        # Two slots per block, plus one for the consumer's sum.
        slots = (params["blocks"] * SLOTS_PER_BLOCK + 1) * FLOATS_PER_CACHE_LINE
        return TaskArgsBuilder(
            TensorArg("out", torch.full((slots,), UNWRITTEN, dtype=torch.float32)),
            Scalar("blocks", params["blocks"]),
            Scalar("repeats", params["repeats"]),
        )

    def compute_golden(self, args, params):
        blocks = params["blocks"]
        total = 0.0
        for block in range(blocks):
            for slot in range(SLOTS_PER_BLOCK):
                args.out[(block * SLOTS_PER_BLOCK + slot) * FLOATS_PER_CACHE_LINE] = float(block)
                total += float(block)
        args.out[blocks * SLOTS_PER_BLOCK * FLOATS_PER_CACHE_LINE] = total


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
