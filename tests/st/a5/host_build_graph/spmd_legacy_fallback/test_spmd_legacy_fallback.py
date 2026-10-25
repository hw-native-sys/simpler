#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pure AIC N=2 is resident; sync_start N=2 requires explicit legacy selection."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


@scene_test(level=2, runtime="host_build_graph")
class TestSpmdShapeSelection(SceneTestCase):
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
            "params": {"require_sync_start": False},
        },
        {
            "name": "aic_root_sync_start_block_num_2",
            "platforms": ["a5sim", "a5"],
            "params": {"require_sync_start": True},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("out", torch.zeros(16, dtype=torch.float32)),
            Scalar("require_sync_start", int(params["require_sync_start"])),
        )

    def compute_golden(self, args, params):
        args.out[0] = 42.0

    def _run_and_validate(self, worker, *args, **kwargs):
        assert not kwargs.get("skip_golden", False), "shape selection requires output and fresh-mode validation"
        self._selection_worker = worker
        self._selection_publish_count = worker.scheduler_mode_publish_count
        super()._run_and_validate(worker, *args, **kwargs)

    def compare_outputs(self, test_args, golden_args, output_names, params):
        count = self._selection_worker.scheduler_mode_publish_count
        assert count > self._selection_publish_count, "no fresh scheduler selection for this invocation"
        mode = self._selection_worker.scheduler_runtime_mode
        expected_modes = (4,) if params["require_sync_start"] else (1, 2)
        assert mode in expected_modes, f"expected scheduler mode {expected_modes}, invocation mode={mode}"
        self._selection_publish_count = count
        super().compare_outputs(test_args, golden_args, output_names, params)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
