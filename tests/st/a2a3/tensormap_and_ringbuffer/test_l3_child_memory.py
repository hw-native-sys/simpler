#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3 child_memory — orch.malloc weight on worker, invoke kernel twice.

Allocates a weight buffer on worker 0 via ``orch.malloc()``, uploads
host data with ``orch.copy_to()``, then creates a ``ContinuousTensor``
with ``child_memory=True``.  Both kernel invocations pin to ``worker=0``.

The second invocation proves the weight was not freed after the first
task — ``init_runtime_impl`` skips malloc + H2D copy for child_memory
tensors and does not record them in tensor_pairs.
"""

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import ContinuousTensor, DataType, TaskArgs, TensorArgType

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, make_tensor_arg, scene_test

KERNELS_BASE = "../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"


def run_dag(orch, callables, task_args, config):
    """L3 orchestration: malloc weight on worker 0, run kernel twice."""
    SIZE = task_args.a.numel()
    nbytes = SIZE * 4  # float32

    # Allocate on worker 0 and upload weight data.
    # task_args.w is share_memory_() so its data_ptr is valid in the child.
    dev_ptr = orch.malloc(worker_id=0, size=nbytes)
    orch.copy_to(worker_id=0, dst=dev_ptr, src=task_args.w.data_ptr(), size=nbytes)

    # Build child_memory tensor from the worker pointer
    w_dev = ContinuousTensor.make(dev_ptr, (SIZE,), DataType.FLOAT32, child_memory=True)

    # Run 1: f1 = kernel(a, w_dev)
    args1 = TaskArgs()
    args1.add_tensor(make_tensor_arg(task_args.a), TensorArgType.INPUT)
    args1.add_tensor(w_dev, TensorArgType.INPUT)
    args1.add_tensor(make_tensor_arg(task_args.f1), TensorArgType.OUTPUT_EXISTING)
    callables.keep(args1)
    orch.submit_next_level(callables.vector_kernel, args1, config, worker=0)

    # Run 2: f2 = kernel(a, w_dev) — same weight, same worker
    args2 = TaskArgs()
    args2.add_tensor(make_tensor_arg(task_args.a), TensorArgType.INPUT)
    args2.add_tensor(w_dev, TensorArgType.INPUT)
    args2.add_tensor(make_tensor_arg(task_args.f2), TensorArgType.OUTPUT_EXISTING)
    callables.keep(args2)
    orch.submit_next_level(callables.vector_kernel, args2, config, worker=0)

    # dev_ptr is freed by DeviceRunner::finalize on worker shutdown


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestL3ChildMemory(SceneTestCase):
    """L3: orch.malloc weight, child_memory, two kernel invocations with worker affinity."""

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
        ],
    }

    CASES = [
        {
            "name": "default",
            "platforms": ["a2a3sim"],
            "config": {"device_count": 1, "num_sub_workers": 0, "block_dim": 3, "aicpu_thread_num": 4},
            "params": {},
        },
    ]

    def generate_args(self, params):
        SIZE = 128 * 128
        return TaskArgsBuilder(
            Tensor("a", torch.full((SIZE,), 2.0, dtype=torch.float32).share_memory_()),
            Tensor("w", torch.full((SIZE,), 3.0, dtype=torch.float32).share_memory_()),
            Tensor("f1", torch.zeros(SIZE, dtype=torch.float32).share_memory_()),
            Tensor("f2", torch.zeros(SIZE, dtype=torch.float32).share_memory_()),
        )

    def compute_golden(self, args, params):
        # vector_example: f = (a + b + 1) * (a + b + 2) + (a + b)
        s = args.a + args.w
        expected = (s + 1) * (s + 2) + s
        args.f1[:] = expected
        args.f2[:] = expected


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
