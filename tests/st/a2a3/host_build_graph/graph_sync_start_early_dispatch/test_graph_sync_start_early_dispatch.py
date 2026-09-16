#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A sync_start root inside a Graph body reaches early dispatch end to end (hbg).

An early-released Graph shell stages its body's roots, so a root carrying
`require_sync_start` reaches `early_sync_start_queue` and the rendezvous through
the same code a top-level candidate uses. The neighbouring scenes leave that
path uncovered: `../spmd_sync_start_early_dispatch/` submits everything outside
a Graph, and `../graph_execution/` puts a `sync_start` task in a body under
shells that have no producer, so those shells never qualify as early-dispatch
candidates and never stage a root.

The tail task carries one half of the assertion. It writes `output` after the
cohort does, so a cohort that never rang surfaces as a scheduler timeout rather
than as a silently wrong pass.

The other half is the oracle. The cohort's width is reported through `layout`
from the same `rt_available_cluster_count()` call the body launches with, and
`output` starts at a sentinel no kernel writes, so a block that did not run is
distinguishable from one that wrote zero and a short cohort fails instead of
narrowing the expectation to match itself.

The body carries exactly one gated-stageable root. One shell release stages every
qualifying root at once, so a second, ordinary root holds gated cores until its
own doorbell rings; the sync cohort then cannot fit and enters the global drain,
which parks the scheduler threads that would have rung the sibling (#2256).
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
# No kernel writes this, so an unwritten cache line stays distinguishable from a
# block that legitimately wrote 0.0.
UNWRITTEN = -1.0

TMR_TWIN = "../../tensormap_and_ringbuffer"
GRAPH_TWIN = "../graph_execution"


@scene_test(level=2, runtime="host_build_graph")
class TestGraphSyncStartEarlyDispatchHbg(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/graph_sync_start_early_dispatch_orch.cpp",
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
            TensorArg("layout", torch.zeros(1, dtype=torch.int32)),
        )

    def compute_golden(self, args, params):
        # The cohort's width is this run's cluster count, which only the device
        # knows. The orchestration reports it through `layout` and
        # compare_outputs builds the expectation from that.
        pass

    def compare_outputs(self, test_args, golden_args, output_names, params):
        lines = test_args.output.reshape(TOTAL_CL, FLOATS_PER_CACHE_LINE)[:, 0]

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


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
