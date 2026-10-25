# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A later ready B must execute while earlier A is still waiting for its actual predecessor P."""

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import TensorTransfer

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

if __package__:
    from .test_ordered_execution import TestHbgOrderedExecution as _OrderedExecution
    from .test_ordered_execution import make_ordered_plan
    from .test_sync_ordered_execution import TestHbgSyncOrderedExecution as _SyncOrderedExecution
    from .test_sync_ordered_execution import check_sync_guards, fill_sync_guards
else:
    from test_ordered_execution import TestHbgOrderedExecution as _OrderedExecution
    from test_ordered_execution import make_ordered_plan
    from test_sync_ordered_execution import TestHbgSyncOrderedExecution as _SyncOrderedExecution
    from test_sync_ordered_execution import check_sync_guards, fill_sync_guards


@scene_test(level=2, runtime="host_build_graph")
class TestHbgReadyOrderExecution(_OrderedExecution):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/ready_order_orchestration.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.IN, D.IN, D.IN, D.IN],
        },
        "incores": [dict(entry) for entry in _SyncOrderedExecution.CALLABLE["incores"]],
    }
    CASES = [
        {
            "name": f"{platform}_a{mask}_b{other}_sync{flags[0]}{flags[1]}",
            "platforms": [platform],
            "params": {
                "scenario": "ready_before",
                "mask": mask,
                "other_mask": other,
                "clusters": clusters,
                "sync_flags": flags,
            },
        }
        for platform, clusters in (("a5sim", 8), ("a5", 32))
        for mask in range(1, 8)
        for other in (1, 2, 4)
        for flags in ((0, 0), (0, 1), (1, 0), (1, 1))
    ]

    def generate_args(self, params):
        # P is outside B's participant set. Waiting for B cannot occupy a
        # position B needs to enter, even if P starts after B is admitted.
        predecessor_mask = 2 if params["other_mask"] == 1 else 1
        shapes = [(1, predecessor_mask)] + [
            (params["clusters"] * (2 if mask in (2, 4) else 1) + (0 if sync else 1), mask)
            for mask, sync in zip((params["mask"], params["other_mask"]), params["sync_flags"], strict=True)
        ]
        plan, records = make_ordered_plan({**params, "scenario": "overlap", "shapes": shapes, "tail_delay": 0})
        plan[1, 5:7] = 1
        plan[1, 8] = plan[1, 20] = 0
        return TaskArgsBuilder(
            TensorArg("records", torch.zeros(records * 16, dtype=torch.int64)),
            TensorArg("plan", plan),
            TensorArg("host_plan", plan.clone(), transfer=TensorTransfer.NONE),
            TensorArg(
                "sync_flags", torch.tensor((0, *params["sync_flags"]), dtype=torch.int64), transfer=TensorTransfer.NONE
            ),
            TensorArg("wait_peers", torch.tensor((2, 2, 1), dtype=torch.int64), transfer=TensorTransfer.NONE),
        )

    def compute_golden(self, args, params):
        super().compute_golden(args, params)
        fill_sync_guards(args, wait_for_peer=True)

    def compare_outputs(self, test_args, golden_args, output_names, params):
        intervals = check_sync_guards(test_args, golden_args, wait_for_peer=True)
        witness = dict(
            masks=golden_args.plan[:, 1].tolist(), sync_flags=golden_args.sync_flags.tolist(), events=intervals
        )
        print(f"[READY_ORDER] {witness}")
        predecessor, earlier, ready = intervals
        assert ready[0] < predecessor[3], "B did not enter while A was blocked on P"
        assert earlier[0] < ready[3], "A did not enter before independent B finished"
        super().compare_outputs(test_args, golden_args, output_names, params)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
