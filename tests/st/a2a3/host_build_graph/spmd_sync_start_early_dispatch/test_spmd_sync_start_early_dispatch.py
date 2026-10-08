#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A wide flagged producer feeds a MIX sync_start early-dispatch consumer (hbg).

Exercises host_build_graph's publish-list detector end to end: the host marks
the consumer a candidate, its publish-list rescan fires when the producer's
last block publishes, the gated cohort pre-stages, and the rendezvous releases
every block together once the producer completes. The consumer must not stage
until every producer block range has reserved a core slot; otherwise it
occupies every AIC slot and strands the producer's unclaimed remainder.

The consumer cohort is exactly this run's cluster count: it must occupy every
AIC slot, and require_sync_start needs every block of it co-resident. The
orchestration reports the width it used and the golden is rebuilt from it. The
producer stays wider than the device on purpose — it carries no sync_start.
Kernels are shared with the tensormap_and_ringbuffer twin of this scene.
"""

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import TensorTransfer

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

FLOATS_PER_CACHE_LINE = 16
PRODUCER_BLOCKS = 50
SYNC_BASE_CL = PRODUCER_BLOCKS
# Widest the platform allows: one consumer block per cluster.
MAX_CLUSTERS = 24
TOTAL_CL = SYNC_BASE_CL + MAX_CLUSTERS * 3

TMR_TWIN = "../../tensormap_and_ringbuffer"


@scene_test(level=2, runtime="host_build_graph")
class TestSpmdSyncStartEarlyDispatchHbg(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_sync_start_early_dispatch_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "SPMD_WRITE_AIC",
                "source": f"{TMR_TWIN}/spmd_sync_start_early_dispatch/kernels/aiv/kernel_spmd_write_slow.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "name": "SPMD_MIX_AIC",
                "source": f"{TMR_TWIN}/spmd_multiblock_mix/kernels/aic/kernel_spmd_mix.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 2,
                "name": "SPMD_MIX_AIV0",
                "source": f"{TMR_TWIN}/spmd_multiblock_mix/kernels/aiv/kernel_spmd_mix.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 3,
                "name": "SPMD_MIX_AIV1",
                "source": f"{TMR_TWIN}/spmd_multiblock_mix/kernels/aiv/kernel_spmd_mix.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "Case1",
            "platforms": ["a2a3sim", "a2a3"],
            "params": {},
        }
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("output", torch.zeros(TOTAL_CL * FLOATS_PER_CACHE_LINE, dtype=torch.float32)),
            TensorArg("layout", torch.zeros(1, dtype=torch.int32), transfer=TensorTransfer.NONE),
        )

    def compute_golden(self, args, params):
        # Both outputs are checked against the reported wave cap in compare_outputs.
        pass

    def compare_outputs(self, test_args, golden_args, output_names, params):
        sync_blocks = int(test_args.layout[0])
        assert 1 <= sync_blocks <= MAX_CLUSTERS, f"consumer width {sync_blocks} outside [1, {MAX_CLUSTERS}]"
        expected = torch.zeros(TOTAL_CL, dtype=torch.float32)
        for block_idx in range(PRODUCER_BLOCKS):
            expected[block_idx] = float(block_idx)
        for block_idx in range(sync_blocks):
            for slot in range(3):
                expected[SYNC_BASE_CL + block_idx * 3 + slot] = float(block_idx)
        actual = test_args.output.reshape(TOTAL_CL, FLOATS_PER_CACHE_LINE)[:, 0]
        assert torch.equal(actual, expected), f"slots disagree with the reported consumer width {sync_blocks}"


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
