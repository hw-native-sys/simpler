# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A5 HBG adaptation of the existing SDMA peer-fetch deferred example."""

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _rehosted_buffer_for, _rehosted_ref_for


def peer_fetch_orchestration(orch, callables, task_args, config):
    ranks = int(task_args.ranks)
    count = int(task_args.elements)
    nbytes = count * 4
    with orch.allocate_domain(
        workspace_engines=("sdma",),
        name="peer_fetch",
        workers=list(range(ranks)),
        window_size=max(nbytes, 4096),
        buffers=[CommBufferSpec(name="input", dtype="float32", count=count, nbytes=nbytes)],
    ) as domain:
        for rank in range(ranks):
            orch.copy_to(
                domain[rank].buffers["input"], _rehosted_buffer_for(task_args, getattr(task_args, f"in_{rank}"))
            )
        for rank in range(ranks):
            args = TaskArgs()
            args.add_tensor(domain[rank].buffers["input"].tensor((count,), DataType.FLOAT32), TensorArgType.INPUT)
            args.add_tensor(_rehosted_ref_for(task_args, getattr(task_args, f"out_{rank}")), TensorArgType.INOUT)
            args.add_tensor(
                _rehosted_ref_for(task_args, getattr(task_args, f"result_{rank}")), TensorArgType.OUTPUT_EXISTING
            )
            args.add_scalar(domain.device_context(rank, workspace_engines=("sdma",)))
            args.add_scalar(int(task_args.transfer_elements))
            args.add_scalar(int(task_args.local_put))
            orch.submit_next_level(callables.peer_fetch, args, config, worker=rank)


class _PeerFetchBase(SceneTestCase):
    RTOL = 0
    ATOL = 1e-3
    CALLABLE = {
        "orchestration": peer_fetch_orchestration,
        "callables": [
            {
                "name": "peer_fetch",
                "orchestration": {
                    "source": "kernels/orchestration/peer_fetch.cpp",
                    "function_name": "aicpu_orchestration_entry",
                    "signature": [D.IN, D.INOUT, D.OUT, D.SCALAR, D.SCALAR, D.SCALAR],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": "kernels/aiv/peer_fetch.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT, D.SCALAR, D.SCALAR, D.SCALAR],
                    },
                    {
                        "func_id": 1,
                        "source": "kernels/aiv/consumer.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT],
                    },
                ],
            }
        ],
    }

    def generate_args(self, params):
        count = params["elements"]
        specs = []
        for rank in range(params["ranks"]):
            values = (torch.arange(count, dtype=torch.float32) % 251 + rank * 1000) / 10
            specs += [
                TensorArg(f"in_{rank}", values),
                TensorArg(f"out_{rank}", torch.full((count,), -17.0)),
                TensorArg(f"result_{rank}", torch.full((count,), -99.0)),
            ]
        return TaskArgsBuilder(
            *specs,
            Scalar("ranks", params["ranks"]),
            Scalar("elements", count),
            Scalar("transfer_elements", 0 if params.get("zero") else count),
            Scalar("local_put", int(params.get("local_put", False))),
        )

    def compute_golden(self, args, params):
        for rank in range(params["ranks"]):
            target = getattr(args, f"out_{rank}")
            if not params.get("zero"):
                source_rank = rank if params.get("local_put") else (rank + 1) % params["ranks"]
                target.copy_(getattr(args, f"in_{source_rank}"))
            getattr(args, f"result_{rank}").copy_(target + 1)


def cases(ranks):
    return [
        {
            "name": name,
            "platforms": ["a5"],
            "config": {"device_count": ranks, "num_sub_workers": 0},
            "params": {"ranks": ranks, "elements": elements, "zero": zero, "local_put": local_put},
        }
        # Local GM-to-GM TPUT checks the SDK's MTE fallback and downstream
        # visibility. These cases do not claim remote PUT/notification coverage.
        for name, elements, zero, local_put in [
            ("peer_fetch", 16384, False, False),
            ("zero_work", 16384, True, False),
            ("large_peer_fetch", 1048576, False, False),
            ("local_put_fallback", 16384, False, True),
            ("local_put_fallback_large", 1048576, False, True),
            ("local_put_zero_work", 16384, True, True),
        ]
    ]


@pytest.mark.sdma(worker_workspace=False)
@scene_test(level=3, runtime="host_build_graph")
class TestSdmaDeferWaitTwoRanks(_PeerFetchBase):
    CASES = cases(2)


@pytest.mark.sdma(worker_workspace=False)
@scene_test(level=3, runtime="host_build_graph")
class TestSdmaDeferWaitFourRanks(_PeerFetchBase):
    CASES = cases(4)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
