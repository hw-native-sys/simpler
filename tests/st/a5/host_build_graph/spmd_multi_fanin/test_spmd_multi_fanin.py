#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""An SPMD task with four predecessors, resolved from three different places.

Every other SPMD case here has a consumer of fanin one, which cannot tell "the
successor is queued when its fanin reaches zero" from "the successor is queued
whenever a predecessor finishes". With four predecessors the two differ three
times per run, and what they differ by is not a wrong number: a task pushed
into the shared queue twice is in a linked list twice, and the second push
writes a link the first one's reader is following.

The four are deliberately unlike each other, because the push is reached from
code that is not shared. A single-block AIC task and a single-block AIV task
resolve from the ordinary completion path. The two SPMD producers resolve from
the block-counting one, and they are different widths so neither width is the
only one tried. Their AIC and AIV halves also mean the resolving core is not
always of a type that could have run the consumer, which is what decides
whether the successor is offered to a dispatch slot or pushed.

The repeat counts rise 1:2:3:4 across the producers, so the four completions
land far apart and a consumer released on an early one has a wide window in
which to read a sentinel.
"""

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

FLOATS_PER_CACHE_LINE = 16
SPMD_AIC_BLOCKS = 4
SPMD_AIV_BLOCKS = 3
CONSUMER_BLOCKS = 6
# Scaled 1:2:3:4 inside the orchestration, so the slowest producer pays four
# times this and the four completions do not arrive together.
REPEATS = 40000
# Mirrors the value bases in the orchestration. A producer writes its base plus
# its block index.
SINGLE_AIC_VALUE_BASE = 1
SINGLE_AIV_VALUE_BASE = 2
SPMD_AIC_VALUE_BASE = 10
SPMD_AIV_VALUE_BASE = 20
# Negative, so any slot the consumer reads before its producer wrote drags the
# sum below every reachable total.
UNWRITTEN = -1.0


@pytest.mark.scheduler_mode("resident")
@scene_test(level=2, runtime="host_build_graph")
class TestSpmdMultiFanin(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_multi_fanin_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.IN, D.IN, D.IN, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aic/kernel_producer.cpp",
                "core_type": "aic",
                "signature": [D.INOUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 1,
                "source": "kernels/aiv/kernel_producer.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 2,
                "source": "kernels/aic/kernel_consumer.cpp",
                "core_type": "aic",
                "signature": [D.INOUT, D.IN, D.IN],
            },
        ],
    }

    CASES = [
        {
            "name": "fanin_4_into_spmd",
            "platforms": ["a5sim", "a5"],
            "params": {
                "aic_blocks": SPMD_AIC_BLOCKS,
                "aiv_blocks": SPMD_AIV_BLOCKS,
                "consumer_blocks": CONSUMER_BLOCKS,
                "repeats": REPEATS,
            },
        },
    ]

    @staticmethod
    def _producer_values(params):
        """Slot index -> value, in the order the orchestration lays them out."""
        values = [float(SINGLE_AIC_VALUE_BASE), float(SINGLE_AIV_VALUE_BASE)]
        values += [float(SPMD_AIC_VALUE_BASE + block) for block in range(params["aic_blocks"])]
        values += [float(SPMD_AIV_VALUE_BASE + block) for block in range(params["aiv_blocks"])]
        return values

    def generate_args(self, params):
        slots = len(self._producer_values(params)) + params["consumer_blocks"]
        return TaskArgsBuilder(
            TensorArg("out", torch.full((slots * FLOATS_PER_CACHE_LINE,), UNWRITTEN, dtype=torch.float32)),
            Scalar("aic_blocks", params["aic_blocks"]),
            Scalar("aiv_blocks", params["aiv_blocks"]),
            Scalar("consumer_blocks", params["consumer_blocks"]),
            Scalar("repeats", params["repeats"]),
        )

    def compute_golden(self, args, params):
        values = self._producer_values(params)
        for slot, value in enumerate(values):
            args.out[slot * FLOATS_PER_CACHE_LINE] = value
        total = sum(values)
        for block in range(params["consumer_blocks"]):
            args.out[(len(values) + block) * FLOATS_PER_CACHE_LINE] = total + float(block)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
