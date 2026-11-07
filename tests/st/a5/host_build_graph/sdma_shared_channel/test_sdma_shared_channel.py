# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Paired AIV submitters must preserve unique SDMA posts and every deferred condition."""

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
            args.add_tensor(_rehosted_ref_for(task_args, getattr(task_args, f"stamps_{rank}")), TensorArgType.INOUT)
            args.add_scalar(domain.device_context(rank, workspace_engines=("sdma",)))
            args.add_scalar(int(task_args.transfer_elements))
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
                    "signature": [D.IN, D.INOUT, D.OUT, D.INOUT, D.SCALAR, D.SCALAR],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": "kernels/aiv/peer_fetch.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT, D.INOUT, D.SCALAR, D.SCALAR],
                    },
                    {
                        "func_id": 1,
                        "source": "../sdma_defer_wait/kernels/aiv/consumer.cpp",
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
                TensorArg(f"stamps_{rank}", torch.zeros(1024, dtype=torch.int64)),
            ]
        return TaskArgsBuilder(
            *specs,
            Scalar("ranks", params["ranks"]),
            Scalar("elements", count),
            Scalar("transfer_elements", 0 if params.get("zero") else params.get("transfer_elements", count)),
        )

    def compute_golden(self, args, params):
        for rank in range(params["ranks"]):
            target = getattr(args, f"out_{rank}")
            if not params.get("zero"):
                target.copy_(getattr(args, f"in_{(rank + 1) % params['ranks']}"))
            getattr(args, f"result_{rank}").copy_(target + 1)

    def compare_outputs(self, test_args, golden_args, output_names, params):
        super().compare_outputs(
            test_args, golden_args, [n for n in output_names if not n.startswith("stamps_")], params
        )
        for rank in range(params["ranks"]):
            stamps = getattr(test_args, f"stamps_{rank}")
            assert int(stamps[0]) == 16 and int(stamps[16]) == 16, stamps[:32].tolist()
            assert int(stamps[1]) == int(stamps[17]), "The two AIVs did not select the same SDK channel group"
            assert int(stamps[2]) == 16 and int(stamps[18]) == 16, f"lost registration count: {stamps[:32].tolist()}"
            ids = [int(stamps[32 + iteration * 32 + lane * 16]) for iteration in range(16) for lane in range(2)]
            print("SDMA_PAIRED_POST_IDS", rank, ids, flush=True)
            # A reused workspace can resume its committed sequence. The batch
            # must own 32 distinct consecutive posts, regardless of its start.
            ordered = sorted(ids)
            assert ordered[0] > 0 and ordered == list(range(ordered[0], ordered[0] + 32)), (
                f"rank {rank}: duplicate/lost SDMA post IDs: {ids}"
            )
            for lane in range(2):
                posts = ids[lane::2]
                assert all(left < right for left, right in zip(posts, posts[1:])), (
                    f"rank {rank}, lane {lane}: SDMA post IDs regressed: {posts}"
                )


def cases(ranks):
    return [
        {
            "name": name,
            "platforms": ["a5"],
            "config": {"device_count": ranks, "num_sub_workers": 0},
            "params": {"ranks": ranks, "elements": elements, "transfer_elements": 16384},
        }
        for name, elements in [
            ("peer_fetch", 16384),
            ("peer_fetch_per_submit_payload", 16 * 16384),
        ]
    ]


@pytest.mark.sdma(worker_workspace=False)
@scene_test(level=3, runtime="host_build_graph")
class TestSdmaSharedChannelTwoRanks(_PeerFetchBase):
    CASES = cases(2)


@pytest.mark.sdma(worker_workspace=False)
@scene_test(level=3, runtime="host_build_graph")
class TestSdmaSharedChannelFourRanks(_PeerFetchBase):
    CASES = cases(4)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
