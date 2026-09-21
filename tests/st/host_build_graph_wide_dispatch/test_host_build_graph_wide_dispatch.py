#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Wide host-build-graph dispatch: AIV cohort plus sync-start and normal MIX placement.

Sizes every cohort from ``rt_available_cluster_count`` / ``rt_available_aiv_count``, so
coverage follows the device the case runs on. On the sim platforms this file is marked
for, that is ``SIM_AUTO_BLOCKDIM`` (8) clusters — 24 core-state bits, entirely within the
low 64 of ``CoreTracker::BitStates``. **This case therefore does not reach a cluster
offset above bit 63**; the >63-bit MIX selection path is covered by the arch-parity unit
test ``tests/ut/cpp/common/host_build_graph/test_hbg_core_tracker.cpp``, which drives ``CoreTracker``
directly at ``MAX_CLUSTERS``. What this case does cover is that MIX placement, sync-start
drain, and the AIV cohort agree on block-to-core mapping when a single scheduler thread
(``aicpu_thread_num=2`` leaves one scheduler plus one resolution thread) owns every cluster.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

FLOATS_PER_CACHE_LINE = 16
SLOTS_PER_MIX_BLOCK = 3
# Buffer capacity ceiling, not an expected count: the orchestration sizes cohorts from the
# runtime and writes back the actual geometry, which compare_outputs reads. a5 has the
# larger PLATFORM_MAX_BLOCKDIM of the two arches, so it bounds both.
MAX_CLUSTERS = 36
MAX_AIV = MAX_CLUSTERS * 2
MAX_TOTAL_CL = MAX_AIV + 2 * MAX_CLUSTERS * SLOTS_PER_MIX_BLOCK


@scene_test(level=2, runtime="host_build_graph")
class TestHostBuildGraphWideDispatch(SceneTestCase):
    """Exercise wide AIV plus sync-start and normal MIX placement with one scheduler."""

    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/host_build_graph_wide_dispatch_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "SPMD_MIX_AIC",
                "source": "../a2a3/tensormap_and_ringbuffer/spmd_sync_start_mix_spill/"
                "kernels/aic/kernel_spmd_mix_slow.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "name": "SPMD_MIX_AIV0",
                "source": "../a2a3/tensormap_and_ringbuffer/spmd_sync_start_mix_spill/"
                "kernels/aiv/kernel_spmd_mix_slow.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 2,
                "name": "SPMD_MIX_AIV1",
                "source": "../a2a3/tensormap_and_ringbuffer/spmd_sync_start_mix_spill/"
                "kernels/aiv/kernel_spmd_mix_slow.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 3,
                "name": "SPMD_WRITE_AIV",
                "source": "../a2a3/tensormap_and_ringbuffer/spmd_sync_start_mix_spill/"
                "kernels/aiv/kernel_spmd_write_slow.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }

    # Two AICPU threads leave one scheduler, which owns every cluster used by
    # the AIV and MIX placement assertions.
    CASES = [
        {
            "name": "TwoThreads",
            "platforms": ["a2a3sim", "a5sim"],
            "manual": ["a2a3sim", "a5sim"],
            "config": {"aicpu_thread_num": 2},
            "params": {},
        }
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("output", torch.zeros(MAX_TOTAL_CL * FLOATS_PER_CACHE_LINE, dtype=torch.float32)),
            TensorArg("layout", torch.zeros(2, dtype=torch.int32)),
        )

    def compute_golden(self, args, params):
        pass

    def compare_outputs(self, test_args, golden_args, output_names, params):
        aiv_blocks, mix_blocks = (int(value) for value in test_args.layout)
        assert aiv_blocks == mix_blocks * 2
        used_cache_lines = aiv_blocks + 2 * mix_blocks * SLOTS_PER_MIX_BLOCK
        assert used_cache_lines <= MAX_TOTAL_CL

        expected = torch.zeros(MAX_TOTAL_CL, dtype=torch.float32)
        for block_idx in range(aiv_blocks):
            expected[block_idx] = float(block_idx)

        sync_base = aiv_blocks
        normal_base = sync_base + mix_blocks * SLOTS_PER_MIX_BLOCK
        for base in (sync_base, normal_base):
            for block_idx in range(mix_blocks):
                for slot in range(SLOTS_PER_MIX_BLOCK):
                    expected[base + block_idx * SLOTS_PER_MIX_BLOCK + slot] = float(block_idx)

        actual = test_args.output.reshape(MAX_TOTAL_CL, FLOATS_PER_CACHE_LINE)[:, 0]
        assert torch.equal(actual, expected)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
