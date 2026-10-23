#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A multi-block task keeps the whole graph on the legacy AICPU scheduler.

`scheduler_resident_v0_task_shape_supported` admits a task only with
`logical_block_num == 1`, so one `set_block_num(2)` task sends the entire graph
to the legacy scheduler with `LEGACY_UNSUPPORTED_SHAPE`. Work that keeps SPMD on
the legacy path depends on that, and the scheduler_mode marker is what proves it
still happens rather than inferring it from a correct result -- legacy and
resident both compute the same answer.

This duplicates `single_root::TestHbgSingleAicSpmdLegacyFallback` on purpose.
That class inherits from `TestHbgSingleAicRoot` and overrides only the
orchestration, and when the parent class runs first the subclass binds the
parent's single-block graph instead of its own: it reports resident, so it
guards nothing in a full-suite run while still passing. This class inherits from
nothing, so no sibling can substitute its graph. See KNOWN_ISSUES.md for the
underlying defect; when that is fixed, the two are redundant and this one can go.
"""

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


@pytest.mark.scheduler_mode("legacy")
@scene_test(level=2, runtime="host_build_graph")
class TestSpmdLegacyFallback(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/spmd_legacy_fallback_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "../../tensormap_and_ringbuffer/dummy_task/kernels/aic/kernel_write_const.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            }
        ],
    }

    CASES = [
        {
            "name": "aic_root_block_num_2",
            "platforms": ["a5sim", "a5"],
            "params": {},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(TensorArg("out", torch.zeros(16, dtype=torch.float32)))

    def compute_golden(self, args, params):
        args.out[0] = 42.0


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
