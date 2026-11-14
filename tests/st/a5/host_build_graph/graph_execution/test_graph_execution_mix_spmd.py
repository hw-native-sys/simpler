#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Graph Execution preserves MIX active slots and multi-block SPMD metadata."""

import json
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

FLOATS_PER_CACHE_LINE = 16
SLOTS_PER_BLOCK = 3
MAX_CLUSTERS = 36
TOTAL_FLOATS = MAX_CLUSTERS * SLOTS_PER_BLOCK * FLOATS_PER_CACHE_LINE


@pytest.mark.scheduler_mode("resident")
@scene_test(level=2, runtime="host_build_graph")
class TestGraphExecutionMixSpmdHostBuildGraphA5(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/graph_execution_mix_spmd_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT, D.INOUT, D.SCALAR],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aic/kernel_spmd_mix.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "source": "kernels/aiv/kernel_spmd_mix.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 2,
                "source": "kernels/aiv/kernel_spmd_mix.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "record_then_replay_mix_spmd",
            "platforms": ["a5sim", "a5"],
            "params": {},
        },
        {
            "name": "partial_sync_2",
            "platforms": ["a5sim", "a5"],
            "params": {"blocks": 2},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("blocks_1", torch.zeros(TOTAL_FLOATS, dtype=torch.float32)),
            TensorArg("blocks_2", torch.zeros(TOTAL_FLOATS, dtype=torch.float32)),
            TensorArg("blocks_3", torch.zeros(TOTAL_FLOATS, dtype=torch.float32)),
            Scalar("blocks", params.get("blocks", 0)),
        )

    def compute_golden(self, args, params):
        # The exact whole-device cluster count is platform-provided. Validate
        # it from the first execution and require both Graph replays to match.
        pass

    def compare_outputs(self, test_args, golden_args, output_names, params):
        outputs = [test_args.blocks_1, test_args.blocks_2, test_args.blocks_3]
        assert torch.equal(outputs[0], outputs[1])
        assert torch.equal(outputs[0], outputs[2])

        cache_line_heads = outputs[0].reshape(-1, FLOATS_PER_CACHE_LINE)[:, 0]
        nonzero = torch.nonzero(cache_line_heads, as_tuple=False)
        assert nonzero.numel() > 0, "SPMD Graph did not execute any non-zero block"
        cluster_count = int(cache_line_heads.max().item()) + 1
        assert 1 < cluster_count <= MAX_CLUSTERS
        if params.get("blocks"):
            assert cluster_count == params["blocks"], "partial cohort did not execute its requested block count"

        expected = torch.zeros_like(cache_line_heads)
        for block_idx in range(cluster_count):
            begin = block_idx * SLOTS_PER_BLOCK
            expected[begin : begin + SLOTS_PER_BLOCK] = float(block_idx)
        assert torch.equal(cache_line_heads, expected)

    def _run_and_validate_l2(self, worker, callable_obj, case, **kwargs):
        """Check Graph instance membership and execution coverage in L4 captures."""
        super()._run_and_validate_l2(worker, callable_obj, case, **kwargs)
        if kwargs.get("enable_chip_swimlane", 0) < 4:
            return
        captures = list(Path(kwargs["output_prefix"]).rglob("chip_swimlane_records.json"))
        assert len(captures) == 1, captures
        protocol = json.loads(captures[0].read_text())["aicore_protocol"]
        assert protocol["schema"] == "hbg-native-protocol-v23"
        cohorts = [task for task in protocol["tasks"] if task["active_mask"] == 7 and task["flags"] & 16]
        assert len(cohorts) == 3, "every Graph instance must contain its own MIX sync cohort"
        requested = case["params"].get("blocks", 0)
        expected = requested or protocol["cluster_count"]
        assert all(task["block_num"] == expected for task in cohorts)
        expected_executions = {(block, lane) for block in range(expected) for lane in range(SLOTS_PER_BLOCK)}
        for task in cohorts:
            executions = [row for row in protocol["executions"] if row["task_ref"] == task["task_ref"]]
            assert len(executions) == len(expected_executions)
            assert {(row["block_idx"], row["subtask_slot"]) for row in executions} == expected_executions
        if requested:
            assert 1 < requested < protocol["cluster_count"], "the selected cohort was not partial"


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
