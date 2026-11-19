#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""sync_start MIX per-core pending-spill: a flagged AIV producer occupies every AIV core
(and spins), leaving every AIC core idle. The require_sync_start MIX consumer then pre-stages
with EVERY cluster mixed — AIC on an idle running slot, both AIVs on the producer's busy cores'
gated pending slots. Exercises the rendezvous seed/mask counting on the MIX per-core split path
(stage_sync_start_cores to_pending=true, mix_cluster_idle_core_count=1/cluster + Case 3.3 promote for
every pending AIV). A counting mismatch stalls the rendezvous -> gated cores never launch ->
allocator deadlock.

The producer spans every AIV core and the consumer every cluster, so both
widths are the device's own counts — a run always takes the whole device, and
that width differs between sim and silicon. The orchestration reports the two
widths in `layout` and the golden is rebuilt from them.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

FLOATS_PER_CACHE_LINE = 16
SLOTS_PER_BLOCK = 3  # MIX consumer block writes 3 cache lines: AIC slot 0, AIV0 slot 1, AIV1 slot 2
# Widest the platform allows: every AIV core (2 per cluster) for the producer,
# every cluster for the MIX consumer.
MAX_CLUSTERS = 24
MAX_AIV = MAX_CLUSTERS * 2
MAX_TOTAL_CL = MAX_AIV + MAX_CLUSTERS * SLOTS_PER_BLOCK


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSpmdSyncStartMixSpill(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_sync_start_mix_spill_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "SPMD_MIX_AIC",
                "source": "../../../common/kernels/spmd_mix/aic/kernel_spmd_mix_slow.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "name": "SPMD_MIX_AIV0",
                "source": "../../../common/kernels/spmd_mix/aiv/kernel_spmd_mix_slow.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 2,
                "name": "SPMD_MIX_AIV1",
                "source": "../../../common/kernels/spmd_mix/aiv/kernel_spmd_mix_slow.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 3,
                "name": "SPMD_WRITE_AIV",
                "source": "../../../common/kernels/spmd_mix/aiv/kernel_spmd_write_slow.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }

    # Three AICPU threads provide two schedulers. The full-device cohorts span
    # both ownership partitions, covering cross-scheduler pending-slot staging.
    CASES = [
        {
            "name": "Case1",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 3},
            "params": {},
        }
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("output", torch.zeros(MAX_TOTAL_CL * FLOATS_PER_CACHE_LINE, dtype=torch.float32)),
            TensorArg("layout", torch.zeros(2, dtype=torch.int32)),
        )

    def compute_golden(self, args, params):
        # Both outputs are checked against the reported layout in compare_outputs.
        pass

    def compare_outputs(self, test_args, golden_args, output_names, params):
        producer_blocks, consumer_blocks = (int(v) for v in test_args.layout)
        assert producer_blocks == consumer_blocks * 2, (
            f"producer {producer_blocks} is not 2 AIV per cluster of {consumer_blocks}"
        )
        assert producer_blocks + consumer_blocks * SLOTS_PER_BLOCK <= MAX_TOTAL_CL, (
            f"layout ({producer_blocks}, {consumer_blocks}) overflows {MAX_TOTAL_CL} cache lines"
        )
        expected = torch.zeros(MAX_TOTAL_CL, dtype=torch.float32)
        # AIV producer: 1 cache line per block.
        for block_idx in range(producer_blocks):
            expected[block_idx] = float(block_idx)
        # MIX consumer: 3 cache lines per block (AIC slot 0, AIV0 slot 1, AIV1 slot 2).
        for block_idx in range(consumer_blocks):
            for slot in range(SLOTS_PER_BLOCK):
                expected[producer_blocks + block_idx * SLOTS_PER_BLOCK + slot] = float(block_idx)
        actual = test_args.output.reshape(MAX_TOTAL_CL, FLOATS_PER_CACHE_LINE)[:, 0]
        assert torch.equal(actual, expected), (
            f"slots disagree with layout (producer={producer_blocks}, consumer={consumer_blocks})"
        )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
