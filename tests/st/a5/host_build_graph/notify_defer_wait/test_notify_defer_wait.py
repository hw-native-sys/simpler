# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A5 HBG adaptation of the shared A2/A3 remote-store/deferred-notify scenario."""

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _rehosted_buffer_for, _rehosted_ref_for


def notify_orchestration(orch, callables, task_args, config):
    ranks = int(task_args.ranks)
    count = int(task_args.elements)
    with orch.allocate_domain(
        name="notifications",
        workers=list(range(ranks)),
        window_size=ranks * (count * 4 + 128) + 4096,
        buffers=[
            CommBufferSpec(name="mailbox", dtype="float32", count=ranks * count, nbytes=ranks * count * 4),
            CommBufferSpec(name="counters", dtype="int32", count=ranks * 32, nbytes=ranks * 128),
        ],
    ) as domain:
        for rank in range(ranks):
            orch.copy_to(domain[rank].buffers["mailbox"], _rehosted_buffer_for(task_args, task_args.mailbox_seed))
            orch.copy_to(domain[rank].buffers["counters"], _rehosted_buffer_for(task_args, task_args.counter_seed))
        for rank in range(ranks):
            args = TaskArgs()
            args.add_tensor(_rehosted_ref_for(task_args, getattr(task_args, f"in_{rank}")), TensorArgType.INPUT)
            args.add_tensor(
                domain[rank].buffers["mailbox"].tensor((ranks * count,), DataType.FLOAT32), TensorArgType.INOUT
            )
            args.add_tensor(domain[rank].buffers["counters"].tensor((ranks * 32,), DataType.INT32), TensorArgType.INOUT)
            args.add_tensor(
                _rehosted_ref_for(task_args, getattr(task_args, f"result_{rank}")), TensorArgType.OUTPUT_EXISTING
            )
            args.add_scalar(domain.device_context(rank))
            args.add_scalar(int(task_args.delta))
            orch.submit_next_level(callables.notify, args, config, worker=rank)


class _NotifyBase(SceneTestCase):
    RTOL = 0
    ATOL = 1e-5
    CALLABLE = {
        "orchestration": notify_orchestration,
        "callables": [
            {
                "name": "notify",
                "orchestration": {
                    "source": "kernels/orchestration/notify.cpp",
                    "function_name": "aicpu_orchestration_entry",
                    "signature": [D.IN, D.INOUT, D.INOUT, D.OUT, D.SCALAR, D.SCALAR],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": "kernels/notify.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.INOUT, D.INOUT, D.OUT, D.SCALAR, D.SCALAR],
                    }
                ],
            }
        ],
    }

    def generate_args(self, params):
        ranks, count = params["ranks"], params["elements"]
        specs = []
        for rank in range(ranks):
            specs += [
                TensorArg(f"in_{rank}", (torch.arange(count, dtype=torch.float32) % 251) / 10 + rank * 1000),
                TensorArg(f"result_{rank}", torch.zeros(ranks * count)),
            ]
        return TaskArgsBuilder(
            *specs,
            TensorArg("mailbox_seed", torch.full((ranks * count,), -17.0)),
            TensorArg("counter_seed", torch.zeros(ranks * 32, dtype=torch.int32)),
            Scalar("ranks", ranks),
            Scalar("elements", count),
            Scalar("delta", params["delta"]),
        )

    def compute_golden(self, args, params):
        ranks, count = params["ranks"], params["elements"]
        for rank in range(ranks):
            target = getattr(args, f"result_{rank}").view(ranks, count)
            for peer in range(ranks):
                if peer == rank:
                    target[peer].fill_(-17.0)
                else:
                    target[peer].copy_(getattr(args, f"in_{peer}") + 1)


def cases(ranks):
    return [
        {
            "name": name,
            "platforms": ["a5"],
            "manual": True,
            "config": {"device_count": ranks, "num_sub_workers": 0},
            "params": {"ranks": ranks, "elements": 16384, "delta": delta},
        }
        for name, delta in [("c_minus_one", -1), ("c", 0), ("c_plus_one", 1)]
    ]


@scene_test(level=3, runtime="host_build_graph")
class TestNotifyDeferWaitTwoRanks(_NotifyBase):
    CASES = cases(2)


@scene_test(level=3, runtime="host_build_graph")
class TestNotifyDeferWaitFourRanks(_NotifyBase):
    CASES = cases(4)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
