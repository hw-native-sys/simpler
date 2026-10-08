#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""available_aicore_counts: the runtime's core counts are real and spendable.

The orchestration reports what `rt_available_*_count()` gave it in `shape` and
spends it: a MIX cohort of exactly `cluster_count` blocks, each writing its
block index into `blocks`.

An **over**-reported count fails: the cohort asks for `require_sync_start`,
so it needs every block co-resident and the deadlock guard fires on device.

An **under**-reported count is NOT detected. Catching one needs an expectation
the host holds independently of what the device said, and the only such handle
was a pinned `block_dim` — a knob that no longer exists now that a run always
takes the whole device. Everything here derives from the reported number, so a
count that is too small is self-consistent: fewer blocks launch, fewer slots are
expected, and the tail is zero on both sides.

host_build_graph is where the counts are hardest to get right: its
orchestrator runs on the host, inside the bind, so it reads `worker_count`
off `Runtime` rather than the AICPU's handshake result. Publishing the core
geometry any later than that hands host orchestration a zero, which the Pinned
case catches (0 != 4) while leaving the `require_sync_start` guard it feeds
silently disabled.
"""

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import TensorTransfer

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

FLOATS_PER_CACHE_LINE = 16
SLOTS_PER_BLOCK = 3
# Cover the largest cohort the platform can launch (PLATFORM_MAX_BLOCKDIM).
MAX_CLUSTERS = 24
AIV_PER_CLUSTER = 2
TOTAL_CL = MAX_CLUSTERS * SLOTS_PER_BLOCK


@scene_test(level=2, runtime="host_build_graph")
class TestAvailableAicoreCounts(SceneTestCase):
    """rt_available_cluster_count() / rt_available_aiv_count() report a spendable width."""

    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/available_aicore_counts_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "SPMD_MIX_AIC",
                "source": "kernels/aic/kernel_spmd_mix.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "name": "SPMD_MIX_AIV0",
                "source": "kernels/aiv/kernel_spmd_mix.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 2,
                "name": "SPMD_MIX_AIV1",
                "source": "kernels/aiv/kernel_spmd_mix.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "Default",
            "platforms": ["a2a3sim", "a2a3"],
            "manual": ["a2a3sim"],
            "params": {},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("blocks", torch.zeros(TOTAL_CL * FLOATS_PER_CACHE_LINE, dtype=torch.float32)),
            TensorArg("shape", torch.zeros(2, dtype=torch.int32), transfer=TensorTransfer.NONE),
        )

    def compute_golden(self, args, params):
        # Both outputs are checked against the reported width in
        # compare_outputs; nothing here is host-computable.
        pass

    def compare_outputs(self, test_args, golden_args, output_names, params):
        clusters = int(test_args.shape[0])
        aivs = int(test_args.shape[1])
        assert 1 <= clusters <= MAX_CLUSTERS, f"cluster_count {clusters} outside [1, {MAX_CLUSTERS}]"
        assert aivs == clusters * AIV_PER_CLUSTER, f"aiv_count {aivs} != {clusters} * {AIV_PER_CLUSTER}"

        blocks = test_args.blocks.reshape(TOTAL_CL, FLOATS_PER_CACHE_LINE)[:, 0]
        expected = torch.zeros(TOTAL_CL, dtype=torch.float32)
        for block_idx in range(clusters):
            for slot in range(SLOTS_PER_BLOCK):
                expected[block_idx * SLOTS_PER_BLOCK + slot] = float(block_idx)
        assert torch.equal(blocks, expected), (
            f"block slots disagree with the reported cluster_count {clusters}: "
            f"got {blocks.tolist()}, expected {expected.tolist()}"
        )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
