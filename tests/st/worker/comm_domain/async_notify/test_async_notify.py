#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cross-architecture notification-counter deferred-completion test."""

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import ChipTensor, CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.torch_interop import make_tensor_arg

N = 128 * 128
NRANKS = 2


def async_notify_orch_fn(orch, callables, task_args, config):
    with orch.allocate_domain(
        name="default",
        workers=list(range(NRANKS)),
        window_size=4 * 1024,
        buffers=[CommBufferSpec(name="notify_counter", dtype="int32", count=1, nbytes=4)],
    ) as handle:
        for rank in range(NRANKS):
            domain = handle[rank]
            args = TaskArgs()
            args.add_tensor(make_tensor_arg(getattr(task_args, f"in_{rank}")), TensorArgType.INPUT)
            args.add_tensor(make_tensor_arg(getattr(task_args, f"out_{rank}")), TensorArgType.OUTPUT_EXISTING)
            args.add_tensor(make_tensor_arg(getattr(task_args, f"result_{rank}")), TensorArgType.OUTPUT_EXISTING)
            args.add_tensor(
                ChipTensor.make(
                    data=domain.buffer_ptrs["notify_counter"],
                    shapes=(1,),
                    dtype=DataType.INT32,
                    child_memory=True,
                ),
                TensorArgType.INPUT,
            )
            args.add_scalar(domain.device_ctx)
            orch.submit_next_level(callables.async_notify, args, config, worker=rank)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestAsyncNotifyDemo(SceneTestCase):
    CALLABLE = {
        "orchestration": async_notify_orch_fn,
        "callables": [
            {
                "name": "async_notify",
                "orchestration": {
                    "source": "kernels/orchestration/async_notify_orchestration.cpp",
                    "function_name": "async_notify_orchestration",
                    "signature": [D.IN, D.OUT, D.OUT, D.IN],
                },
                "incores": [
                    {
                        "func_id": func_id,
                        "source": source,
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT, D.OUT, D.IN],
                    }
                    for func_id, source in enumerate(
                        [
                            "kernels/aiv/kernel_producer_notify.cpp",
                            "kernels/aiv/kernel_consumer.cpp",
                            "kernels/aiv/kernel_notify_wait.cpp",
                        ]
                    )
                ],
            }
        ],
    }
    CASES = [
        {
            "name": "notification_counter",
            "platforms": ["a2a3", "a5"],
            "config": {"device_count": NRANKS, "num_sub_workers": 0},
            "params": {},
        }
    ]
    RTOL = 0.0
    ATOL = 1e-3

    def generate_args(self, params):
        specs = []
        for rank in range(NRANKS):
            inp = torch.tensor([float(i % 251) / 10.0 for i in range(N)], dtype=torch.float32)
            specs.extend(
                [
                    Tensor(f"in_{rank}", inp),
                    Tensor(f"out_{rank}", torch.zeros(N, dtype=torch.float32)),
                    Tensor(f"result_{rank}", torch.zeros(N, dtype=torch.float32)),
                ]
            )
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        for rank in range(NRANKS):
            expected_out = getattr(args, f"in_{rank}") * 2.0
            getattr(args, f"out_{rank}").copy_(expected_out)
            getattr(args, f"result_{rank}").copy_(expected_out + 1.0)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
