#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Waves of mixed work, every wave released at once and repeated.

One task ready at a time says little about a scheduler. These waves put many
ready at the same instant -- out of a single wake-list walk -- and do it again
every round, so the transition the SPMD queue sits on is exercised repeatedly
rather than once.

A wave's slots cycle through a palette of eight kinds: ordinary and SPMD, AIC,
AIV and Mix, with sync-start cohorts among them. The combinations worth having
fall out of composing that rather than being written by hand -- a Mix SPMD task
ready beside an AIC SPMD one, SPMD work ready beside ordinary work, a cohort
assembling while unrelated tasks compete for the same cores. Only one SPMD task
is claimed at a time by construction, so a wave of them is also a queue of
them, and what this asks is whether they all come back.

Block counts are sized by the orchestration from the topology it finds, which
is why the expected output cannot mention them: the same recipe yields
different widths on an eight-cluster box and a thirty-two-cluster one. Only
block 0 of each task marks the output, so the result depends on the recipe
alone. How many blocks actually ran, and where, is what spmd_multiblock_aic and
spmd_sync_start_mix_aiv assert; a successor released before its producer
finished is what spmd_last_block_releases asserts. This case is deliberately
blind to both -- its golden is order-independent, which is the property that
lets the graph be this large and this varied.
"""

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

FLOATS_PER_CACHE_LINE = 16
SLOTS_PER_TASK = 4
KIND_COUNT = 8

# Mirrors Kind in the orchestration. Only which core types a kind occupies
# matters here; its block count does not, because the expected output does not
# depend on one.
AIC_ORDINARY, AIV_ORDINARY, MIX_ORDINARY = 0, 1, 2
AIC_SPMD, AIV_SPMD, MIX_SPMD = 3, 4, 5
AIC_COHORT, MIX_COHORT = 6, 7

_WRITES_AIC = frozenset({AIC_ORDINARY, MIX_ORDINARY, AIC_SPMD, MIX_SPMD, AIC_COHORT, MIX_COHORT})
_WRITES_AIV = frozenset({AIV_ORDINARY, MIX_ORDINARY, AIV_SPMD, MIX_SPMD, MIX_COHORT})


def _plan(rounds, width, seed):
    """The tasks the orchestration submits, in the order it submits them.

    Kept in step with the orchestration by hand. The alternative -- having the
    device report what it built -- would make the expectation a copy of the
    behaviour under test.
    """
    plan = [AIC_ORDINARY]  # the task the first wave hangs off
    for round_index in range(rounds):
        for slot in range(width):
            plan.append((seed + round_index * width + slot) % KIND_COUNT)
        plan.append(AIC_ORDINARY)  # the join that ends the round
    return plan


@pytest.mark.scheduler_mode("resident")
@scene_test(level=2, runtime="host_build_graph")
class TestSpmdMixedWaves(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_mixed_waves_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.IN, D.IN, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aic/kernel_mark_aic.cpp",
                "core_type": "aic",
                "signature": [D.INOUT, D.IN],
            },
            {
                "func_id": 1,
                "source": "kernels/aiv/kernel_mark_aiv.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.IN],
            },
        ],
    }

    CASES = [
        # Small enough for the default gate: every kind appears, and the seed
        # shifts which kind lands where so the palette is not always in the
        # same order relative to a wave boundary.
        {
            "name": "waves_4x8",
            "platforms": ["a5sim", "a5"],
            "params": {"rounds": 4, "width": 8, "seed": 0},
        },
        {
            "name": "waves_4x8_shifted",
            "platforms": ["a5sim", "a5"],
            "params": {"rounds": 4, "width": 8, "seed": 3},
        },
        # The stress tier: enough rounds that the scheduler sees the same
        # transition hundreds of times, and wide enough that a wave does not
        # fit the machine. Manual because it is a scale case, not a gate.
        {
            "name": "waves_40x24_stress",
            "platforms": ["a5sim", "a5"],
            "params": {"rounds": 40, "width": 24, "seed": 1},
            "manual": True,
            "config": {"runtime_env": {"ring_task_window": 4096}},
        },
    ]

    def generate_args(self, params):
        plan = _plan(params["rounds"], params["width"], params["seed"])
        slots = len(plan) * SLOTS_PER_TASK * FLOATS_PER_CACHE_LINE
        return TaskArgsBuilder(
            TensorArg("out", torch.zeros(slots, dtype=torch.float32)),
            Scalar("rounds", params["rounds"]),
            Scalar("width", params["width"]),
            Scalar("seed", params["seed"]),
        )

    def compute_golden(self, args, params):
        plan = _plan(params["rounds"], params["width"], params["seed"])
        for index, kind in enumerate(plan):
            base = index * SLOTS_PER_TASK * FLOATS_PER_CACHE_LINE
            if kind in _WRITES_AIC:
                args.out[base] = float(index)
            if kind in _WRITES_AIV:
                args.out[base + FLOATS_PER_CACHE_LINE] = float(index)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
