# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise the production completion reader on both core types at integer boundaries."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

COUNTERS = [(value, expected) for value in [-(2**31), -1, 0, 1, 2**31 - 1] for expected in [0, 1, 2**31 - 1]]
POSTS = [
    (0, 1),
    (1, 1),
    (2**32 - 1, 2**32),
    (2**32, 2**32),
    (2**63 - 1, 2**63),
    (2**63, 2**63),
    (2**63 + 1, 2**63),
    (2**64 - 2, 2**64 - 1),
    (2**64 - 1, 2**64 - 1),
]


@scene_test(level=2, runtime="host_build_graph")
class TestDeferWaitObserver(SceneTestCase):
    RTOL = 0
    ATOL = 0
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/observer_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT, D.SCALAR],
        },
        "incores": [
            {"func_id": index, "source": "kernels/observer.cpp", "core_type": core, "signature": [D.IN, D.IN, D.OUT]}
            for index, core in enumerate(["aic", "aiv"])
        ],
    }
    CASES = [
        {"name": core, "platforms": ["a5sim", "a5"], "params": {"core": index}}
        for index, core in enumerate(["aic", "aiv"])
    ]

    def generate_args(self, params):
        # INT64 transport preserves the high-bit patterns; the SDK record reader
        # interprets them as uint64, while COUNTER uses signed INT32 comparison.
        post_bits = [[word if word < 2**63 else word - 2**64 for word in row] for row in POSTS]
        return TaskArgsBuilder(
            TensorArg("counters", torch.tensor(COUNTERS, dtype=torch.int32)),
            TensorArg("posts", torch.tensor(post_bits, dtype=torch.int64)),
            TensorArg("out", torch.full((len(COUNTERS) + len(POSTS),), -9, dtype=torch.int32)),
            Scalar("core", params["core"]),
        )

    def compute_golden(self, args, params):
        expected = [int(value >= threshold) for value, threshold in COUNTERS + POSTS]
        args.out.copy_(torch.tensor(expected, dtype=torch.int32))


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
