#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""sync_start MIX per-core pending-spill: a flagged AIV producer occupies all 48 AIV cores (and
spins), leaving the 24 AIC cores idle. The require_sync_start MIX consumer then pre-stages with
EVERY cluster mixed — AIC on an idle running slot, both AIVs on the producer's busy cores' gated
pending slots. Exercises the rendezvous seed/mask counting on the MIX per-core split path
(drain_stage_cores to_pending=true, mix_cluster_idle_core_count=1/cluster + Case 3.3 promote for
the 48 pending AIVs). A counting mismatch stalls the rendezvous -> gated cores never launch ->
allocator deadlock."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

FLOATS_PER_CACHE_LINE = 16
SLOTS_PER_BLOCK = 3  # MIX consumer block writes 3 cache lines: AIC slot 0, AIV0 slot 1, AIV1 slot 2
PRODUCER_BLOCKS = 48  # AIV producer: 1 cache line per block, base_cl 0
PRODUCER_BASE_CL = 0
CONSUMER_BLOCKS = 24  # MIX consumer: 3 cache lines per block
CONSUMER_BASE_CL = 48
TOTAL_CL = 120  # 48 (producer) + 24*3 (consumer)


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSpmdSyncStartMixSpill(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_sync_start_mix_spill_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "SPMD_MIX_AIC",
                "source": "kernels/aic/kernel_spmd_mix_slow.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "name": "SPMD_MIX_AIV0",
                "source": "kernels/aiv/kernel_spmd_mix_slow.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 2,
                "name": "SPMD_MIX_AIV1",
                "source": "kernels/aiv/kernel_spmd_mix_slow.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 3,
                "name": "SPMD_WRITE_AIV",
                "source": "kernels/aiv/kernel_spmd_write_slow.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "Case1",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 3, "block_dim": 24},
            "params": {},
        }
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(Tensor("output", torch.zeros(TOTAL_CL * FLOATS_PER_CACHE_LINE, dtype=torch.float32)))

    def compute_golden(self, args, params):
        out = args.output
        # AIV producer: 1 cache line per block.
        for block_idx in range(PRODUCER_BLOCKS):
            out[(PRODUCER_BASE_CL + block_idx) * FLOATS_PER_CACHE_LINE] = float(block_idx)
        # MIX consumer: 3 cache lines per block (AIC slot 0, AIV0 slot 1, AIV1 slot 2).
        for block_idx in range(CONSUMER_BLOCKS):
            for slot in range(SLOTS_PER_BLOCK):
                out[(CONSUMER_BASE_CL + block_idx * SLOTS_PER_BLOCK + slot) * FLOATS_PER_CACHE_LINE] = float(block_idx)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
