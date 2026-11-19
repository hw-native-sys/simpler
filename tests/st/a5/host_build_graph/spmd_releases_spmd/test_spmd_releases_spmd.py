#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""An SPMD task released by another SPMD task's last block.

spmd_last_block_releases makes the producer SPMD and the consumer a single
block, which leaves the successor on the ordinary path: a completing core may
hand it straight to the dispatch slot it just freed. An SPMD successor may not
be handed anywhere -- no core owns a task every core claims blocks from -- so it
has to be pushed into the shared queue instead, by the same code that would
otherwise direct-fill. A chain of SPMD stages is the only shape that exercises
that branch.

It is also the only shape that empties the shared queue between every pair of
stages. One stage is in it at a time, and the next arrives at a queue every
core has already observed empty, so a stage boundary is a full re-observation
rather than a step along a list that was there all along.

What it does *not* reach is the handover race the head's identity check exists
for. That one needs the successor to be in the queue while the cores that lost
the last block are still advancing past the task they exhausted, and in a chain
the successor is not pushed until that last block has finished running -- long
after those cores have advanced. spmd_mixed_waves keeps several SPMD tasks
queued at once and is what covers it; disabling the identity check leaves this
test green.

What makes a premature release visible: each block sums *every* slot of the
previous stage, so one sentinel changes its output, and its output is what the
next stage sums -- the corruption propagates to the golden rather than being
overwritten. The spin is proportional to the block index, so a stage's blocks
finish spread out and a stage declared done on any block but its last leaves
the next stage reading slots still held by running blocks.
"""

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

FLOATS_PER_CACHE_LINE = 16
# Under the 8 AIC cores a5sim provides, so a stage's blocks all run at once and
# the stage lasts as long as its slowest block rather than a queue of them.
BLOCK_NUM = 6
STAGE_NUM = 4
# Scaled by block index inside the kernel, so the longest block of a stage pays
# (BLOCK_NUM - 1) times this. Long enough that a stage released early is still
# running when the next one reads it.
REPEATS = 30000
# What an unwritten slot holds. Negative, so a sum missing any slot falls below
# every sum of real values and cannot reach the golden by accident.
UNWRITTEN = -1.0


@pytest.mark.scheduler_mode("resident")
@scene_test(level=2, runtime="host_build_graph")
class TestSpmdReleasesSpmd(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_releases_spmd_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.IN, D.IN, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aic/kernel_chain_stage.cpp",
                "core_type": "aic",
                "signature": [D.INOUT, D.IN, D.IN, D.IN],
            },
        ],
    }

    CASES = [
        {
            "name": "chain_of_4_spmd_stages",
            "platforms": ["a5sim", "a5"],
            "params": {"blocks": BLOCK_NUM, "stages": STAGE_NUM, "repeats": REPEATS},
        },
    ]

    def generate_args(self, params):
        slots = params["stages"] * params["blocks"] * FLOATS_PER_CACHE_LINE
        return TaskArgsBuilder(
            TensorArg("out", torch.full((slots,), UNWRITTEN, dtype=torch.float32)),
            Scalar("blocks", params["blocks"]),
            Scalar("stages", params["stages"]),
            Scalar("repeats", params["repeats"]),
        )

    def compute_golden(self, args, params):
        blocks = params["blocks"]
        previous_sum = 0.0
        for stage in range(params["stages"]):
            values = [previous_sum + float(block) for block in range(blocks)]
            for block, value in enumerate(values):
                args.out[(stage * blocks + block) * FLOATS_PER_CACHE_LINE] = value
            previous_sum = sum(values)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
