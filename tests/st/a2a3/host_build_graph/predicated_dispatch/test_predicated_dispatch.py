#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""predicated_dispatch: a dispatch predicate is evaluated by the scheduler at the
dispatch point (delayed evaluation), not in orchestration.

The predicate value is produced by a prior task; the scheduler reads it only
when the predicated task becomes ready (its producer done), so orchestration
never stalls on it. Chain:

  gate_producer writes gate[0] = gate_value (INT32)
  x_producer    writes X[0]   = 42.0
  clobber       would write X[0] = 999.0, but carries set_predicate(gate[0] > 0)
                and depends on gate_producer; the scheduler reads gate[0] at the
                dispatch point and dispatches only if gate[0] > 0.
  consumer      copies X[0] -> Y[0]

  case=1 (gate_value = 0): predicate FALSE -> clobber NOT dispatched -> X stays
         42.0 -> Y = 42.0. Proves non-dispatch AND that the retired task still
         unlocks the consumer.
  case=2 (gate_value = 1): predicate TRUE -> clobber dispatched -> X = 999.0 ->
         Y = 999.0. Proves the dispatch path is taken when the predicate holds.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test

SENTINEL = 42.0
POISON = 999.0  # what the clobber writes if the predicate lets it dispatch
INIT_VAL = -1.0


@scene_test(level=2, runtime="host_build_graph")
class TestPredicatedDispatch(SceneTestCase):
    """predicated_dispatch: scheduler evaluates the dispatch predicate at dispatch time."""

    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/predicated_dispatch_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT, D.INOUT],  # X, Y, gate
        },
        "incores": [
            {
                "func_id": 0,
                "name": "WRITE_CONST",
                "source": "kernels/aic/kernel_write_const.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "name": "COPY_FIRST",
                "source": "kernels/aic/kernel_copy_first.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.INOUT],
            },
            {
                "func_id": 2,
                "name": "CLOBBER",
                "source": "kernels/aic/kernel_clobber.cpp",
                "core_type": "aic",
                # Body of the predicated task; runs only if the predicate holds.
                "signature": [D.INOUT],
            },
            {
                "func_id": 3,
                "name": "WRITE_GATE",
                "source": "kernels/aic/kernel_write_gate.cpp",
                "core_type": "aic",
                # One INOUT tensor (gate); the gate value rides as a trailing scalar.
                "signature": [D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "PredicateFalseSkips",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 2, "block_dim": 1},
            "params": {"case": 1},
        },
        {
            "name": "PredicateTrueDispatches",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 2, "block_dim": 1},
            "params": {"case": 2},
        },
    ]

    def generate_args(self, params):
        x = torch.full((16,), INIT_VAL, dtype=torch.float32)
        y = torch.full((16,), INIT_VAL, dtype=torch.float32)
        gate = torch.full((16,), -1, dtype=torch.int32)
        return TaskArgsBuilder(
            Tensor("x", x),
            Tensor("y", y),
            Tensor("gate", gate),
            Scalar("case", int(params["case"])),
        )

    def compute_golden(self, args, params):
        gate_value = 0 if params["case"] == 1 else 1
        args.gate[0] = gate_value
        # x_producer writes 42.0; the clobber (X = 999.0) runs only if gate > 0.
        x_final = SENTINEL if gate_value == 0 else POISON
        args.x[0] = x_final
        args.y[0] = x_final


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
