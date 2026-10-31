#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Correctness smoke for independent MIX/pure streams and the full-mask control.

Numerical checks cover execution and sentinels; they do not assert temporal overlap.
Placement policy is checked by the production-scheduler unit tests.
"""

import ctypes

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import TensorTransfer

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

MAX_BLOCKS = 36 * 8
FLOATS_PER_LINE = 16
KERNEL_ROOT = "../common/kernels/spmd_mix/"


class SpareLanesBase(SceneTestCase):
    RTOL = 0
    ATOL = 0
    CALLABLE = {
        "orchestration": {
            "source": "kernels/spare_lanes_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": KERNEL_ROOT + "aic/kernel_spmd_mix_slow.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "source": KERNEL_ROOT + "aiv/kernel_spmd_mix_slow.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 2,
                "source": KERNEL_ROOT + "aiv/kernel_spmd_mix_slow.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 3,
                "source": KERNEL_ROOT + "aiv/kernel_spmd_write_slow.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 4,
                "source": KERNEL_ROOT + "aiv/kernel_spmd_write_slow.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
        ],
    }
    # A fixed thread budget keeps multiple schedulers competing over MIX and pure queues.
    CASES = [
        {
            "name": f"mask{mask}",
            "platforms": ["a2a3sim", "a2a3", "a5sim", "a5"],
            "config": {"aicpu_thread_num": 3},
            "params": {"mask": mask},
        }
        for mask in [3, 5, 6, 7]
    ]

    def generate_args(self, params):
        layout_options = {} if self._st_runtime == "tensormap_and_ringbuffer" else {"transfer": TensorTransfer.NONE}
        return TaskArgsBuilder(
            TensorArg("mixed", torch.zeros(MAX_BLOCKS * 3 * FLOATS_PER_LINE, dtype=torch.float32)),
            TensorArg("pure", torch.zeros(MAX_BLOCKS * FLOATS_PER_LINE, dtype=torch.float32)),
            TensorArg("layout", torch.zeros(1, dtype=torch.int32), **layout_options),
            Scalar("mask", ctypes.c_int64(params["mask"])),
        )

    def compute_golden(self, args, params):
        pass

    def compare_outputs(self, test_args, golden_args, output_names, params):
        blocks = int(test_args.layout[0])
        assert 0 < blocks <= MAX_BLOCKS
        mixed = torch.zeros(MAX_BLOCKS * 3, dtype=torch.float32)
        pure = torch.zeros(MAX_BLOCKS, dtype=torch.float32)
        for block in range(blocks):
            pure[block] = float(block)
            for lane in range(3):
                if params["mask"] & (1 << lane):
                    mixed[3 * block + lane] = float(block)
        torch.testing.assert_close(test_args.mixed.reshape(-1, FLOATS_PER_LINE)[:, 0], mixed, rtol=0, atol=0)
        torch.testing.assert_close(test_args.pure.reshape(-1, FLOATS_PER_LINE)[:, 0], pure, rtol=0, atol=0)


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSpareLanesTmr(SpareLanesBase):
    pass


@scene_test(level=2, runtime="host_build_graph")
class TestSpareLanesHbg(SpareLanesBase):
    pass


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
