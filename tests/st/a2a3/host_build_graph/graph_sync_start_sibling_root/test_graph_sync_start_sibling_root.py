#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A sync_start body root shares one shell release with an ordinary sibling root.

`stage_graph_roots_early` stages every qualifying root from a single shell
release, so a `require_sync_start` root that needs the whole device can find its
cores held by a sibling waiting for its own doorbell. The cohort then takes the
global drain, and the drain can only be satisfied once those cores come free —
which happens only once the scheduler loop reaches `activate_graph_task` and
routes the body's roots.

The neighbouring scene `../graph_sync_start_early_dispatch/` keeps exactly one
gated-stageable root in its body and so never exercises that interaction.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

FLOATS_PER_CACHE_LINE = 16
SLOTS_PER_BLOCK = 3
# Widest the platform allows: one cohort block per cluster.
MAX_CLUSTERS = 24

# Must mirror the orchestration's cache-line layout.
SEED_BASE_CL = 0
SEED_BLOCKS = 8
SYNC_BASE_CL = 8
TAIL_BASE_CL = 80  # SYNC_BASE_CL + MAX_CLUSTERS * SLOTS_PER_BLOCK
TAIL_BLOCKS = 2
TOTAL_CL = TAIL_BASE_CL + TAIL_BLOCKS
SIBLING_BLOCKS = 4
# No kernel writes this, so an unwritten cache line stays distinguishable from a
# block that legitimately wrote 0.0.
UNWRITTEN = -1.0

TMR_TWIN = "../../tensormap_and_ringbuffer"
GRAPH_TWIN = "../graph_execution"


@scene_test(level=2, runtime="host_build_graph")
class TestGraphSyncStartSiblingRootHbg(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/graph_sync_start_sibling_root_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT, D.INOUT],
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
                "source": f"{GRAPH_TWIN}/kernels/aic/kernel_spmd_mix.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 2,
                "name": "SPMD_MIX_AIV0",
                "source": f"{GRAPH_TWIN}/kernels/aiv/kernel_spmd_mix.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 3,
                "name": "SPMD_MIX_AIV1",
                "source": f"{GRAPH_TWIN}/kernels/aiv/kernel_spmd_mix.cpp",
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
            TensorArg(
                "output",
                torch.full((TOTAL_CL * FLOATS_PER_CACHE_LINE,), UNWRITTEN, dtype=torch.float32),
            ),
            TensorArg(
                "sibling_output",
                torch.full((SIBLING_BLOCKS * FLOATS_PER_CACHE_LINE,), UNWRITTEN, dtype=torch.float32),
            ),
            TensorArg("layout", torch.zeros(1, dtype=torch.int32)),
        )

    def compute_golden(self, args, params):
        # The cohort's width is this run's cluster count, which only the device
        # knows. The orchestration reports it through `layout` and
        # compare_outputs builds the expectation from that.
        pass

    def compare_outputs(self, test_args, golden_args, output_names, params):
        lines = test_args.output.reshape(TOTAL_CL, FLOATS_PER_CACHE_LINE)[:, 0]
        sibling = test_args.sibling_output.reshape(SIBLING_BLOCKS, FLOATS_PER_CACHE_LINE)[:, 0]

        # Reported by the orchestration from the same call the body launched
        # with, so a cohort that ran short fails below instead of redefining the
        # width it is checked against.
        cohort_blocks = int(test_args.layout[0])
        assert 1 <= cohort_blocks <= MAX_CLUSTERS, f"cohort width {cohort_blocks} outside [1, {MAX_CLUSTERS}]"

        expected = torch.full((TOTAL_CL,), UNWRITTEN, dtype=torch.float32)
        for block_idx in range(SEED_BLOCKS):
            expected[SEED_BASE_CL + block_idx] = float(block_idx)
        for block_idx in range(cohort_blocks):
            for slot in range(SLOTS_PER_BLOCK):
                expected[SYNC_BASE_CL + block_idx * SLOTS_PER_BLOCK + slot] = float(block_idx)
        for block_idx in range(TAIL_BLOCKS):
            expected[TAIL_BASE_CL + block_idx] = float(block_idx)

        assert torch.equal(lines, expected), (
            f"output disagrees with the reported cohort width {cohort_blocks}; "
            "an unrung cohort, a short cohort or an unordered tail shows up here"
        )

        expected_sibling = torch.arange(SIBLING_BLOCKS, dtype=torch.float32)
        assert torch.equal(sibling, expected_sibling), "the sibling root did not complete"


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
