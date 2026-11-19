# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Resident regression: an ordinary AIV completion must not lose its sole Mix successor."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

if __package__:
    from .test_spmd_contract import TestHbgSpmdContract as _SpmdContract
else:
    from test_spmd_contract import TestHbgSpmdContract as _SpmdContract

_RECORD_WORDS = 16


@scene_test(level=2, runtime="host_build_graph")
class TestAivMixHandoff(_SpmdContract):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/aiv_mix_handoff_orchestration.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT, D.INOUT],
        },
        "incores": [dict(entry) for entry in _SpmdContract.CALLABLE["incores"]],
    }
    CASES = [
        {
            "name": f"aiv_to_mix_mask{mask}_{'pair' if tasks == 1 else 'chain17'}",
            "platforms": ["a5", "a5sim"],
            "params": {"blocks": 1, "mask": mask, "tasks": tasks, "tail_delay": 4096},
        }
        for mask in (3, 5, 6, 7)
        for tasks in (1, 17)
    ]

    def generate_args(self, params):
        records = params["tasks"] * (params["blocks"] * 3 + 1) * _RECORD_WORDS
        return TaskArgsBuilder(
            TensorArg("records", torch.zeros(records, dtype=torch.int64)),
            TensorArg("summary", torch.zeros(_RECORD_WORDS, dtype=torch.int64)),
            TensorArg("root_records", torch.zeros(records, dtype=torch.int64)),
            Scalar("blocks", params["blocks"]),
            Scalar("mask", params["mask"]),
            Scalar("tasks", params["tasks"]),
            Scalar("tail_delay", params["tail_delay"]),
        )

    def compute_golden(self, args, params):
        super().compute_golden(args, params)
        roots = args.root_records.view(params["tasks"], 4, _RECORD_WORDS)
        for task in range(params["tasks"]):
            roots[task, 1, 0] = 1
            roots[task, 1, 8:12] = torch.tensor([1, 1, 2, task + 1])

    def compare_outputs(self, test_args, golden_args, output_names, params):
        guards = test_args.root_records.view(params["tasks"], 4, _RECORD_WORDS)[:, -1, :]
        assert torch.count_nonzero(guards[:, :2]).item() == 0, "AIV predecessor has missing or invalid context"
        # The inherited hook checks fresh resident provenance and every output word,
        # including root visits, inactive slots, metadata, and both sets of guards.
        super().compare_outputs(test_args, golden_args, output_names, params)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
