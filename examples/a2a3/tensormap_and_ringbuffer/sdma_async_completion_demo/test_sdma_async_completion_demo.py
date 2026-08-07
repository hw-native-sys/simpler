#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SDMA deferred-completion smoke test for onboard a2a3.

Each rank stages its input inside the HCCL window. The deferred producer fetches
the peer rank's input into local ``out`` and registers its async event. The
consumer depends on that output and writes ``result = out + 1``.
"""

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import ChipTensor, CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.torch_interop import make_tensor_arg

N = 128 * 128
NRANKS = 2
DTYPE_NBYTES = 4


def sdma_async_completion_orch_fn(orch, callables, task_args, config):
    input_nbytes = N * DTYPE_NBYTES
    with orch.allocate_domain(
        name="default",
        workers=list(range(NRANKS)),
        window_size=max(input_nbytes, 4 * 1024),
        buffers=[CommBufferSpec(name="input_window", dtype="float32", count=N, nbytes=input_nbytes)],
    ) as handle:
        for rank in range(NRANKS):
            orch.copy_to(
                rank,
                dst=handle[rank].buffer_ptrs["input_window"],
                src=getattr(task_args, f"in_{rank}").data_ptr(),
                size=input_nbytes,
            )
        for rank in range(NRANKS):
            domain = handle[rank]
            args = TaskArgs()
            args.add_tensor(
                ChipTensor.make(
                    data=domain.buffer_ptrs["input_window"],
                    shapes=(N,),
                    dtype=DataType.FLOAT32,
                    child_memory=True,
                ),
                TensorArgType.INPUT,
            )
            args.add_tensor(make_tensor_arg(getattr(task_args, f"out_{rank}")), TensorArgType.OUTPUT_EXISTING)
            args.add_tensor(make_tensor_arg(getattr(task_args, f"result_{rank}")), TensorArgType.OUTPUT_EXISTING)
            args.add_scalar(domain.device_ctx)
            orch.submit_next_level(callables.sdma_async_completion, args, config, worker=rank)


@pytest.mark.sdma
@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestSdmaAsyncCompletionDemo(SceneTestCase):
    CALLABLE = {
        "orchestration": sdma_async_completion_orch_fn,
        "callables": [
            {
                "name": "sdma_async_completion",
                "orchestration": {
                    "source": "kernels/orchestration/sdma_async_completion_orch.cpp",
                    "function_name": "sdma_async_completion_orchestration",
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
                            "kernels/aiv/kernel_sdma_tget_async.cpp",
                            "kernels/aiv/kernel_consumer.cpp",
                        ]
                    )
                ],
            }
        ],
    }
    CASES = [
        {
            "name": "peer_fetch",
            "platforms": ["a2a3"],
            "config": {"device_count": NRANKS, "num_sub_workers": 0},
            "params": {},
        }
    ]
    RTOL = 0.0
    ATOL = 1e-3

    def generate_args(self, params):
        specs = []
        for rank in range(NRANKS):
            inp = torch.tensor([float(rank * 1000 + (i % 251)) / 10.0 for i in range(N)], dtype=torch.float32)
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
            expected_out = getattr(args, f"in_{1 - rank}")
            getattr(args, f"out_{rank}").copy_(expected_out)
            getattr(args, f"result_{rank}").copy_(expected_out + 1.0)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
