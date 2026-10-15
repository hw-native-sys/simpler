#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-build-graph dump args example: f = (a + b) + 1.

Demonstrates the two dump-args metadata registration APIs:
  Task 0 (add):                add_task() + set_tensor_info_to_task()
  Task 1 (add_scalar_inplace): add_task_with_tensor_info()

Those are orchestration APIs and they work here; the capture they register for does not. This
runtime's route is the AICore Scheduler, where the on-core schedulers both dispatch and complete,
so it has no per-task argument producer — `args_dump_capture_supported_impl` answers 0 and
`simpler_prepare_run` refuses an `enable_dump_args` request. **So this case asserts the computation
only, and `--dump-args` fails every a5 `host_build_graph` scene rather than collecting here.** The
capture template is the a2a3 mirror, `tests/st/a2a3/host_build_graph/dump_args`; the refusal is
asserted by `tests/st/a5/host_build_graph/test_dfx_joined_launch_attribution.py`.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


@scene_test(level=2, runtime="host_build_graph")
class TestDumpArgsExampleA5(SceneTestCase):
    """f = (a + b) + 1, where a=2.0, b=3.0 -> f=6.0."""

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/dump_args_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": "kernels/aiv/kernel_add_scalar_inplace.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "default",
            "platforms": ["a5sim", "a5"],
            "params": {},
        },
    ]

    def generate_args(self, params):
        SIZE = 128 * 128
        return TaskArgsBuilder(
            TensorArg("a", torch.full((SIZE,), 2.0, dtype=torch.float32)),
            TensorArg("b", torch.full((SIZE,), 3.0, dtype=torch.float32)),
            TensorArg("f", torch.zeros(SIZE, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        args.f[:] = (args.a + args.b) + 1


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
