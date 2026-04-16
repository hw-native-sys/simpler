#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3 group task — 2 ChipWorkers (process-isolated) on 1 DAG node.

Each chip runs the same kernel with its own args (different tensors).
A downstream SubTask depends on the group output.
Verifies: fork+shm process isolation, 2-chip concurrent execution,
group completion aggregation, downstream dependency waits for group.
"""

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import TaskArgs, TensorArgType, make_tensor_arg

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

KERNELS_BASE = "../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"


def verify(args):
    """SubCallable — runs after group completes."""


def _chip_args(in_a, in_b, out_f) -> TaskArgs:
    """Build per-chip TaskArgs with INPUT/INPUT/OUTPUT_EXISTING tags."""
    a = TaskArgs()
    a.add_tensor(make_tensor_arg(in_a), TensorArgType.INPUT)
    a.add_tensor(make_tensor_arg(in_b), TensorArgType.INPUT)
    a.add_tensor(make_tensor_arg(out_f), TensorArgType.OUTPUT_EXISTING)
    return a


def run_dag(orch, callables, task_args, config):
    """L3 orchestration: group of 2 chips → SubTask dependency."""
    args0 = _chip_args(task_args.a0, task_args.b0, task_args.f0)
    args1 = _chip_args(task_args.a1, task_args.b1, task_args.f1)
    callables.keep(args0, args1)  # prevent GC before drain

    orch.submit_next_level_group(callables.vector_kernel, [args0, args1], config)

    # SubTask depends on both group outputs (f0, f1) — tag both as INPUT.
    sub_args = TaskArgs()
    sub_args.add_tensor(make_tensor_arg(task_args.f0), TensorArgType.INPUT)
    sub_args.add_tensor(make_tensor_arg(task_args.f1), TensorArgType.INPUT)
    orch.submit_sub(callables.verify, sub_args)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestL3Group(SceneTestCase):
    """L3: Group of 2 ChipWorkers as 1 DAG node, SubTask depends on group."""

    CALLABLE = {
        "orchestration": run_dag,
        "callables": [
            {
                "name": "vector_kernel",
                "orchestration": {
                    "source": f"{KERNELS_BASE}/orchestration/example_orchestration.cpp",
                    "function_name": "aicpu_orchestration_entry",
                    "signature": [D.IN, D.IN, D.OUT],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": f"{KERNELS_BASE}/aiv/kernel_add.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.IN, D.OUT],
                    },
                    {
                        "func_id": 1,
                        "source": f"{KERNELS_BASE}/aiv/kernel_add_scalar.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT],
                    },
                    {
                        "func_id": 2,
                        "source": f"{KERNELS_BASE}/aiv/kernel_mul.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.IN, D.OUT],
                    },
                ],
            },
            {"name": "verify", "callable": verify},
        ],
    }

    CASES = [
        {
            "name": "default",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"device_count": 2, "num_sub_workers": 1, "block_dim": 3, "aicpu_thread_num": 4},
            "params": {},
        },
    ]

    def generate_args(self, params):
        SIZE = 128 * 128
        return TaskArgsBuilder(
            Tensor("a0", torch.full((SIZE,), 2.0, dtype=torch.float32).share_memory_()),
            Tensor("b0", torch.full((SIZE,), 3.0, dtype=torch.float32).share_memory_()),
            Tensor("f0", torch.zeros(SIZE, dtype=torch.float32).share_memory_()),
            Tensor("a1", torch.full((SIZE,), 2.0, dtype=torch.float32).share_memory_()),
            Tensor("b1", torch.full((SIZE,), 3.0, dtype=torch.float32).share_memory_()),
            Tensor("f1", torch.zeros(SIZE, dtype=torch.float32).share_memory_()),
        )

    def compute_golden(self, args, params):
        args.f0[:] = (args.a0 + args.b0 + 1) * (args.a0 + args.b0 + 2) + (args.a0 + args.b0)
        args.f1[:] = (args.a1 + args.b1 + 1) * (args.a1 + args.b1 + 2) + (args.a1 + args.b1)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
