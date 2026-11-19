#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cross-architecture deferred-completion communication test."""

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _rehosted_ref_for

N = 128 * 128
NRANKS = 2
DTYPE_NBYTES = 4


def deferred_notify_orch_fn(orch, callables, task_args, config):
    mailbox_nbytes = N * DTYPE_NBYTES
    with orch.allocate_domain(
        name="default",
        workers=list(range(NRANKS)),
        window_size=max(mailbox_nbytes + 64, 4 * 1024),
        buffers=[
            CommBufferSpec(name="mailbox", dtype="float32", count=N, nbytes=mailbox_nbytes),
            CommBufferSpec(name="notify_counter", dtype="int32", count=1, nbytes=64),
        ],
    ) as handle:
        for rank in range(NRANKS):
            domain = handle[rank]
            args = TaskArgs()
            args.add_tensor(_rehosted_ref_for(task_args, getattr(task_args, f"partial_{rank}")), TensorArgType.INPUT)
            args.add_tensor(domain.buffers["mailbox"].tensor((N,), DataType.FLOAT32), TensorArgType.INOUT)
            args.add_tensor(
                _rehosted_ref_for(task_args, getattr(task_args, f"result_{rank}")), TensorArgType.OUTPUT_EXISTING
            )
            args.add_tensor(domain.buffers["notify_counter"].tensor((1,), DataType.INT32), TensorArgType.INPUT)
            args.add_scalar(domain.device_ctx)
            orch.submit_next_level(callables.deferred_notify, args, config, worker=rank)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestDeferredNotifyDemo(SceneTestCase):
    CALLABLE = {
        "orchestration": deferred_notify_orch_fn,
        "callables": [
            {
                "name": "deferred_notify",
                "orchestration": {
                    "source": "kernels/orchestration/deferred_notify_orch.cpp",
                    "function_name": "deferred_notify_orchestration",
                    "signature": [D.IN, D.INOUT, D.OUT, D.IN],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": "kernels/aiv/kernel_producer.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.INOUT, D.OUT, D.SCALAR, D.SCALAR],
                    },
                    {
                        "func_id": 1,
                        "source": "kernels/aiv/kernel_consumer.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.IN, D.OUT, D.SCALAR],
                    },
                    {
                        "func_id": 2,
                        "source": "kernels/aiv/kernel_notify_wait.cpp",
                        "core_type": "aiv",
                        "signature": [D.OUT, D.SCALAR, D.SCALAR],
                    },
                ],
            }
        ],
    }
    CASES = [
        {
            "name": "peer_notification",
            "platforms": ["a2a3", "a2a3sim", "a5sim"],
            "manual": ["a2a3sim"],
            "config": {"device_count": NRANKS, "num_sub_workers": 0},
            "params": {},
        }
    ]
    RTOL = 0.0
    ATOL = 1e-6

    def generate_args(self, params):
        specs = []
        for rank in range(NRANKS):
            specs.extend(
                [
                    TensorArg(f"partial_{rank}", torch.full((N,), float(rank + 1), dtype=torch.float32)),
                    TensorArg(f"result_{rank}", torch.zeros(N, dtype=torch.float32)),
                ]
            )
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        for rank in range(NRANKS):
            getattr(args, f"result_{rank}").copy_(getattr(args, f"partial_{(rank + 1) % NRANKS}"))


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
