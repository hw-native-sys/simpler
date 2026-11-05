#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent AIV work alongside residual MIX and a waiting sync-start MIX.

Golden output checks lane routing and completion. Scheduler CPU tests check the
progress contract; a chip swimlane capture can additionally check actual overlap.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

MAX_BLOCKS = 384
FLOATS_PER_CL = 16
MIX_FLOATS = MAX_BLOCKS * 3 * FLOATS_PER_CL


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestMixSpareLane(SceneTestCase):
    RTOL = 0
    ATOL = 0
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/mix_spare_lane_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT] * 5,
        },
        "incores": [
            {
                "func_id": 0,
                "name": "MIX_AIC",
                "source": "../spmd_sync_start_mix_spill/kernels/aic/kernel_spmd_mix_slow.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "name": "MIX_AIV",
                "source": "../spmd_sync_start_mix_spill/kernels/aiv/kernel_spmd_mix_slow.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 2,
                "name": "INDEPENDENT_AIV",
                "source": "../spmd_sync_start_mix_spill/kernels/aiv/kernel_spmd_write_slow.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }
    CASES = [
        {
            "name": f"{mode}_mask{mask}",
            "platforms": ["a5sim", "a5"],
            "params": {"mask": mask, "sync": int(mode == "Sync")},
        }
        for mode in ("Residual", "Sync")
        for mask in (3, 5, 7)
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("producer", torch.full((MIX_FLOATS,), -1.0)),
            TensorArg("cohort", torch.full((MIX_FLOATS,), -1.0)),
            TensorArg("vector", torch.full((MAX_BLOCKS * FLOATS_PER_CL,), -1.0)),
            TensorArg("trigger", torch.full((3 * FLOATS_PER_CL,), -1.0)),
            TensorArg("layout", torch.zeros(4, dtype=torch.int32)),
            Scalar("mask", params["mask"]),
            Scalar("sync", params["sync"]),
            Scalar("spin", 200000),
        )

    def compute_golden(self, args, params):
        pass  # The runtime reports physical widths in layout.

    def compare_outputs(self, test_args, golden_args, output_names, params):
        clusters, producer_blocks, cohort_blocks, vector_blocks = map(int, test_args.layout)
        assert 2 <= clusters <= MAX_BLOCKS // 4
        assert producer_blocks == (clusters - 1 if params["sync"] else 4 * clusters)
        assert cohort_blocks == (clusters if params["sync"] else 0)
        assert vector_blocks == clusters
        for name, blocks in (("producer", producer_blocks), ("cohort", cohort_blocks)):
            expected = torch.full((MIX_FLOATS,), -1.0)
            for block in range(blocks):
                for slot in range(3):
                    if params["mask"] & (1 << slot):
                        value = block if params["sync"] or name == "cohort" else 0
                        expected[(block * 3 + slot) * FLOATS_PER_CL] = float(value)
            assert torch.equal(getattr(test_args, name), expected), name
        expected_vector = torch.full((MAX_BLOCKS * FLOATS_PER_CL,), -1.0)
        for block in range(vector_blocks):
            expected_vector[block * FLOATS_PER_CL] = float(block)
        assert torch.equal(test_args.vector, expected_vector), "independent vector"
        expected_trigger = torch.full((3 * FLOATS_PER_CL,), -1.0)
        if params["sync"]:
            expected_trigger[0] = 0.0
        assert torch.equal(test_args.trigger, expected_trigger), "trigger"


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
