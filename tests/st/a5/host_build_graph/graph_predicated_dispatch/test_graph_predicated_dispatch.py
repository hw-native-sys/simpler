#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""graph_predicated_dispatch: a Graph carries dispatch predicates across replays.

One fixed Graph body holds three predicated tasks, one per operand source
(boundary tensor, Graph-internal tensor, view of a boundary tensor). The body is
invoked four times: invocation 0 records the Definition, the rest replay it as
outer Graph tasks.

Each invocation supplies its own gate buffer and its own gate scalar:

  invocation | gate[0] | internal | gate[view] | dispatched         | X     | Y
  0          | 0       | 0        | 1          | consumer           | 42.0  | 42.0
  1          | 1       | 0        | 1          | clobber + consumer | 999.0 | 999.0
  2          | 0       | 1        | 1          | alt + consumer     | 555.0 | 555.0
  3          | 0       | 0        | 0          | none               | 42.0  | INIT

An operand address frozen at record time would replay invocation 0's gates, so
every invocation would take invocation 0's branch.

The consumer's operand is a view of the gate starting at GATE_VIEW_BASE, indexed
at element 1 — the only case where the address stacks a boundary base, a view
offset and an element offset. Only gate[GATE_VIEW_BASE + 1] carries its value, so
dropping any one of the three reads an element that is 0 everywhere and leaves Y
unwritten where the table says it must be copied.
"""

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

ELEMS = 16
GATE_ELEMS = 16
GATE_VIEW_BASE = 4  # must match the orchestration
GATE_VIEW_INDEX = GATE_VIEW_BASE + 1
SENTINEL = 42.0
POISON = 999.0  # written by the task predicated on the boundary gate
POISON_ALT = 555.0  # written by the task predicated on the Graph-internal gate
INIT_VAL = -1.0

# Per invocation: (boundary gate[0], internal gate scalar, gate[GATE_VIEW_INDEX]).
GATES = ((0, 0, 1), (1, 0, 1), (0, 1, 1), (0, 0, 0))


@pytest.mark.scheduler_mode("legacy")
@scene_test(level=2, runtime="host_build_graph")
class TestGraphPredicatedDispatchA5(SceneTestCase):
    """graph_predicated_dispatch: predicates recorded into a Graph resolve per replay."""

    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/graph_predicated_dispatch_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            # x0..x3, y0..y3, gate0..gate3
            "signature": [D.INOUT] * 8 + [D.IN] * 4,
        },
        "incores": [
            {
                "func_id": 0,
                "name": "WRITE_CONST",
                "source": "../predicated_dispatch/kernels/aic/kernel_write_const.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "name": "COPY_FIRST",
                "source": "../predicated_dispatch/kernels/aic/kernel_copy_first.cpp",
                "core_type": "aic",
                # Predicated on a view of the boundary gate.
                "signature": [D.IN, D.INOUT],
            },
            {
                "func_id": 2,
                "name": "CLOBBER",
                "source": "../predicated_dispatch/kernels/aic/kernel_clobber.cpp",
                "core_type": "aic",
                # Predicated on the boundary gate.
                "signature": [D.INOUT],
            },
            {
                "func_id": 3,
                "name": "WRITE_GATE",
                "source": "../predicated_dispatch/kernels/aic/kernel_write_gate.cpp",
                "core_type": "aic",
                # Produces the Graph-internal gate; the value rides as a trailing scalar.
                "signature": [D.OUT],
            },
            {
                "func_id": 4,
                "name": "CLOBBER_ALT",
                "source": "kernels/aic/kernel_clobber_alt.cpp",
                "core_type": "aic",
                # Predicated on the Graph-internal gate.
                "signature": [D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "PredicateAcrossGraphReplays",
            "platforms": ["a5sim", "a5"],
            "params": {},
        },
    ]

    def generate_args(self, params):
        del params
        args = []
        for i in range(len(GATES)):
            args.append(TensorArg(f"x{i}", torch.full((ELEMS,), INIT_VAL, dtype=torch.float32)))
        for i in range(len(GATES)):
            args.append(TensorArg(f"y{i}", torch.full((ELEMS,), INIT_VAL, dtype=torch.float32)))
        for i, (boundary_gate, _, view_gate) in enumerate(GATES):
            gate = torch.zeros((GATE_ELEMS,), dtype=torch.int32)
            gate[0] = boundary_gate
            gate[GATE_VIEW_INDEX] = view_gate
            args.append(TensorArg(f"gate{i}", gate))
        args.extend(Scalar(f"internal_gate{i}", internal) for i, (_, internal, _) in enumerate(GATES))
        return TaskArgsBuilder(*args)

    def compute_golden(self, args, params):
        del params
        for i, (boundary_gate, internal_gate, view_gate) in enumerate(GATES):
            if internal_gate > 0:
                x_final = POISON_ALT
            elif boundary_gate > 0:
                x_final = POISON
            else:
                x_final = SENTINEL
            getattr(args, f"x{i}")[0] = x_final
            # The consumer is itself predicated: suppressed, it never copies.
            if view_gate > 0:
                getattr(args, f"y{i}")[0] = x_final


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
