#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sliding-window moving average over runtime-allocated outputs.

``out[i] = mean(base[i], out[i-1], ..., out[i-WINDOW])``

Dense-fanin coverage for the orchestrator and scheduler. Every ``out[i]`` is a
runtime-allocated OUTPUT, so a later task reading it takes a Step-A creator edge
(``DEP_WAIT | DEP_RETAIN``) from ``owner_task_id``. The whole graph therefore
comes from dependency discovery — no explicit dependency is ever added.

Each task carries WINDOW producer edges where the rest of the scene-test corpus
carries at most 1.5, which is what this case exists to exercise:

- per-task fanin degree of 16 against a corpus maximum of 1.49
- producer fanout degree of 16, so the completion-time fanout walk is O(16)
  rather than O(1)
- dependency-pool occupancy proportional to the window rather than to a chain

Only ``i-1`` is non-transitive: ``T_{i-1}`` already reaches ``T_{i-2}`` through
``T_{i-WINDOW}``, so WINDOW-1 of every task's edges are transitively redundant,
all at submission distance <= WINDOW.

Predecessors that do not exist yet (``i < WINDOW``) are padded with ``base[i]``,
an external tensor with no owner, which contributes no edge.
"""

import ctypes

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

# Must match WINDOW in sliding_window_orch.cpp and kernel_window_avg.cpp.
WINDOW = 16
TILE_ELEMS = 128 * 128


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSlidingWindowDeps(SceneTestCase):
    """Dense creator-edge fanin from a sliding-window recurrence."""

    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/sliding_window_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "WINDOW_AVG",
                "source": "kernels/aiv/kernel_window_avg.cpp",
                "core_type": "aiv",
                # WINDOW + 1 inputs (base slice + the previous WINDOW outputs), 1 output.
                "signature": [D.IN] * (WINDOW + 1) + [D.OUT],
            },
            {
                "func_id": 1,
                "name": "EMIT",
                "source": "kernels/aiv/kernel_emit.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "default",
            "manual": ["a2a3"],
            "platforms": ["a2a3sim", "a2a3"],
            "params": {"steps": 24, "tiles": 1},
        },
        {
            "name": "Dense16",
            "platforms": ["a2a3"],
            "params": {"steps": 1000, "tiles": 1},
            "manual": True,
        },
    ]

    def generate_args(self, params):
        steps = params["steps"]
        tiles = params["tiles"]
        step_elems = tiles * TILE_ELEMS

        torch.manual_seed(42)
        base = torch.randn(steps, step_elems, dtype=torch.float32)
        result = torch.zeros(step_elems, dtype=torch.float32)

        return TaskArgsBuilder(
            TensorArg("base", base.flatten()),
            TensorArg("result", result),
            Scalar("steps", ctypes.c_int64(steps)),
        )

    def compute_golden(self, args, params):
        steps = params["steps"]
        tiles = params["tiles"]
        step_elems = tiles * TILE_ELEMS

        base = args.base.reshape(steps, step_elems)
        result = args.result

        outs = []
        for i in range(steps):
            acc = base[i].clone()
            for k in range(1, WINDOW + 1):
                j = i - k
                acc += outs[j] if j >= 0 else base[i]
            outs.append(acc / (WINDOW + 1))

        result.copy_(outs[-1])


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
