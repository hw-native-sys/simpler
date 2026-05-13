#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""dummy_task: verify dep-only tasks block consumers and never run a kernel.

The orchestration submits one of three scenes, controlled by params["case"]:

  case=1 (single dummy via tensormap INOUT):
    producer writes X[0]=42.0 -> dummy_T INOUTs X -> consumer copies X to Y.
    Y[0] must equal 42.0. If dummy somehow ran a kernel it would zero or
    corrupt the buffer; the value 42.0 in Y proves both ordering and the
    no-op nature of dummy_task.

  case=2 (long dummy chain):
    Same as case 1, but with LONG_CHAIN_DUMMIES dummies between producer
    and consumer. Looks after the dummy_ready_queue + dispatch-loop drain
    when several dummies sit on the critical path back-to-back.

  case=3 (explicit add_dep barrier):
    Two independent producers (writing X and W); a dummy_T uses
    add_dep(A, B) as a many-to-one barrier; the consumer add_deps on dummy
    and reads X. Verifies dummy_task participates in explicit_dep wiring.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test

SENTINEL = 42.0
INIT_VAL = -1.0  # so unmodified Y is distinguishable from the sentinel


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestDummyTask(SceneTestCase):
    """dummy_task: dep-only tasks must block consumers and never run a kernel."""

    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/dummy_task_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "WRITE_CONST",
                "source": "kernels/aic/kernel_write_const.cpp",
                "core_type": "aic",
            },
            {
                "func_id": 1,
                "name": "COPY_FIRST",
                "source": "kernels/aic/kernel_copy_first.cpp",
                "core_type": "aic",
            },
        ],
    }

    CASES = [
        {
            "name": "SingleDummyAutoDep",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 2, "block_dim": 1},
            "params": {"case": 1},
        },
        {
            "name": "LongDummyChain",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 2, "block_dim": 1},
            "params": {"case": 2},
        },
        {
            "name": "DummyExplicitDepBarrier",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 2, "block_dim": 1},
            "params": {"case": 3},
        },
    ]

    def generate_args(self, params):
        x = torch.full((16,), INIT_VAL, dtype=torch.float32)
        y = torch.full((16,), INIT_VAL, dtype=torch.float32)
        w = torch.full((16,), INIT_VAL, dtype=torch.float32)
        return TaskArgsBuilder(
            Tensor("x", x),
            Tensor("y", y),
            Tensor("w", w),
            Scalar("case", int(params["case"])),
        )

    def compute_golden(self, args, params):
        # The producer (kernel_write_const) writes 42.0 to X[0]; the consumer
        # (kernel_copy_first) copies X[0] -> Y[0]. Any dummy_task in the chain
        # is a pure barrier and does NOT touch the buffer, so X[0] / Y[0]
        # must equal SENTINEL on the host side regardless of case.
        args.x[0] = SENTINEL
        args.y[0] = SENTINEL
        if params["case"] == 3:
            # case 3 has a second producer writing W
            args.w[0] = SENTINEL


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
