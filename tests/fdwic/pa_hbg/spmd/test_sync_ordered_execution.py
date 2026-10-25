# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Heterogeneous sync overlap, dependencies and ordinary waves; event ordinals are not performance measurements."""

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import TensorTransfer

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

if __package__:
    from .test_ordered_execution import TestHbgOrderedExecution as _OrderedExecution
    from .test_ordered_execution import make_ordered_plan
else:
    from test_ordered_execution import TestHbgOrderedExecution as _OrderedExecution
    from test_ordered_execution import make_ordered_plan


def _case(platform, clusters, scenario, mask, other, flags=(1, 1)):
    return {
        "name": f"{platform}_{scenario}_mask{mask}_to{other}_sync{flags[0]}{flags[1]}",
        "platforms": [platform],
        "params": {"scenario": scenario, "mask": mask, "other_mask": other, "clusters": clusters, "sync_flags": flags},
    }


def fill_sync_guards(args, *, wait_for_peer):
    for task, (n, mask, base, *_) in enumerate(args.plan.tolist()):
        args.records.view(-1, 16)[base + n * 3, 4] = n * mask.bit_count()
        args.records.view(-1, 16)[base + n * 3, 6] = mask.bit_count() if wait_for_peer else 0
        if args.sync_flags[task].item():
            args.records.view(-1, 16)[base + n * 3, 2] = n * mask.bit_count()


def check_sync_guards(test_args, golden_args, *, wait_for_peer):
    rows = test_args.records.view(-1, 16)
    intervals = []
    for task, (n, mask, base, *_) in enumerate(golden_args.plan.tolist()):
        guard = rows[base + 3 * n]
        expected = n * mask.bit_count() if golden_args.sync_flags[task].item() else 0
        assert guard[2].item() == expected, "missing or duplicate sync arrival"
        assert guard[3].item() == 0, "sync kernel rendezvous timed out"
        assert guard[4].item() == n * mask.bit_count(), "missing or duplicate peer entry"
        assert guard[5].item() == 0, "independent peer entry timed out"
        assert guard[6].item() == (mask.bit_count() if wait_for_peer else 0), (
            "tail peer-wait branch was not exercised exactly once per active lane"
        )
        indices = [base + 3 * block + slot for block in range(n) for slot in range(3) if mask & (1 << slot)]
        entered, returned = rows[indices, 13], rows[indices, 14]
        if expected:
            assert entered.max().item() < returned.min().item(), "sync return precedes all task entries"
        intervals.append((entered.min().item(), entered.max().item(), returned.min().item(), returned.max().item()))
    return intervals


@scene_test(level=2, runtime="host_build_graph")
class TestHbgSyncOrderedExecution(_OrderedExecution):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/sync_ordered_orchestration.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.IN, D.IN, D.IN],
        },
        "incores": [
            {
                "func_id": slot,
                "name": f"sync_ordered_slot_{slot}",
                "source": f"kernels/sync_ordered_slot_{slot}.cpp",
                "core_type": "aic" if slot == 0 else "aiv",
                "signature": [D.INOUT, D.IN, D.IN],
            }
            for slot in range(3)
        ],
    }
    CASES = [
        _case(platform, clusters, scenario, mask, other)
        for platform, clusters in (("a5sim", 8), ("a5", 32))
        for scenario in ("overlap", "edge")
        for mask in range(1, 8)
        for other in range(1, 8)
    ] + [
        _case(platform, clusters, "overlap", mask, mask, flags)
        for platform, clusters in (("a5sim", 8), ("a5", 32))
        for mask in range(1, 8)
        for flags in ((1, 0), (0, 1))
    ]

    def generate_args(self, params):
        flags = params["sync_flags"]
        shapes = [
            (params["clusters"] * (2 if mask in (2, 4) else 1) + (0 if sync else 1), mask)
            for mask, sync in zip((params["mask"], params["other_mask"]), flags, strict=True)
        ]
        plan, records = make_ordered_plan({**params, "shapes": shapes, "tail_delay": 8192})
        return TaskArgsBuilder(
            TensorArg("records", torch.zeros(records * 16, dtype=torch.int64)),
            TensorArg("plan", plan),
            TensorArg("host_plan", plan.clone(), transfer=TensorTransfer.NONE),
            TensorArg("sync_flags", torch.tensor(flags, dtype=torch.int64), transfer=TensorTransfer.NONE),
            Scalar("wait_for_peer", int(params["scenario"] == "overlap")),
        )

    def compute_golden(self, args, params):
        super().compute_golden(args, params)
        fill_sync_guards(args, wait_for_peer=params["scenario"] == "overlap")

    def compare_outputs(self, test_args, golden_args, output_names, params):
        intervals = check_sync_guards(test_args, golden_args, wait_for_peer=params["scenario"] == "overlap")
        witness = dict(
            scenario=params["scenario"],
            masks=(params["mask"], params["other_mask"]),
            sync_flags=params["sync_flags"],
            events=intervals,
        )
        print(f"[SYNC_ORDER] {witness}")
        super().compare_outputs(test_args, golden_args, output_names, params)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
