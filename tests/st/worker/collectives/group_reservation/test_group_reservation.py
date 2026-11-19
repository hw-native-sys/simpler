#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end coverage for overlapping NEXT_LEVEL group reservations."""

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CommBufferSpec, TaskArgs, TensorArgType

from simpler_setup import SceneTestCase, TaskArgsBuilder, scene_test
from simpler_setup import TensorArg as STensor
from simpler_setup.scene_test import _rehosted_ref_for

FIRST_GROUP = 0
RELEASE_FIRST_GROUP = 1
SECOND_GROUP = 2
CHECK_SECOND_GROUP = 3


def group_reservation_orch_fn(orch, callables, task_args, config):
    chip = callables.group_reservation

    with orch.allocate_domain(
        name="reservation",
        workers=[0, 1, 2],
        window_size=4 * 1024,
        buffers=[CommBufferSpec(name="state", dtype="int32", count=2, nbytes=64)],
    ) as handle:

        def make_args(worker_id, output_name, operation):
            domain = handle[worker_id]
            args = TaskArgs()
            args.add_tensor(
                _rehosted_ref_for(task_args, getattr(task_args, output_name)), TensorArgType.OUTPUT_EXISTING
            )
            args.add_scalar(operation)
            args.add_scalar(domain.buffers["state"].base)
            args.add_scalar(domain.device_ctx)
            return args

        orch.submit_next_level_group(
            chip,
            [
                make_args(0, "first_group_0", FIRST_GROUP),
                make_args(1, "first_group_1", FIRST_GROUP),
            ],
            config,
            workers=[0, 1],
        )
        orch.submit_next_level_group(
            chip,
            [
                make_args(1, "second_group_1", SECOND_GROUP),
                make_args(2, "second_group_2", SECOND_GROUP),
            ],
            config,
            workers=[1, 2],
        )
        orch.submit_next_level(
            chip,
            make_args(0, "unrelated_single_0", RELEASE_FIRST_GROUP),
            config,
            worker=0,
        )
        orch.submit_next_level(
            chip,
            make_args(2, "reserved_single_2", CHECK_SECOND_GROUP),
            config,
            worker=2,
        )


_CALLABLE = {
    "orchestration": group_reservation_orch_fn,
    "callables": [
        {
            "name": "group_reservation",
            "orchestration": {
                "source": "kernels/orchestration/group_reservation_orch.cpp",
                "function_name": "group_reservation_orchestration",
                "config_name": "group_reservation_orchestration_config",
                "signature": [D.OUT],
            },
            "incores": [
                {
                    "func_id": 0,
                    "source": "kernels/aiv/group_reservation_kernel.cpp",
                    "core_type": "aiv",
                    "signature": [D.OUT],
                }
            ],
        }
    ],
}


def _make_args():
    names = [
        "first_group_0",
        "first_group_1",
        "second_group_1",
        "second_group_2",
        "unrelated_single_0",
        "reserved_single_2",
    ]
    return TaskArgsBuilder(*(STensor(name, torch.zeros(1, dtype=torch.int32).share_memory_()) for name in names))


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestConsecutiveGroupReservation(SceneTestCase):
    CALLABLE = _CALLABLE
    CASES = [
        {
            "name": "overlapping_targets",
            "platforms": ["a2a3sim", "a2a3", "a5sim"],
            "manual": ["a2a3sim", "a5sim"],
            "config": {"device_count": 3},
            "params": {},
        }
    ]

    def generate_args(self, params):
        return _make_args()

    def compute_golden(self, args, params):
        args.first_group_0.fill_(10)
        args.first_group_1.fill_(11)
        args.second_group_1.fill_(31)
        args.second_group_2.fill_(32)
        args.unrelated_single_0.fill_(20)
        args.reserved_single_2.fill_(1)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
