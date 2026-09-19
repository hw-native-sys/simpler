# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""FFTS kernel hooks through real DeviceRunner loading and mixed-task dispatch."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


@scene_test(level=2, runtime="host_build_graph")
class TestSimFftsDispatch(SceneTestCase):
    """Two clusters exchange payloads through independently loaded AIC/AIV kernels."""

    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/ffts_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT],
        },
        "incores": [
            {"func_id": 0, "source": "kernels/aic/ffts.cpp", "core_type": "aic", "signature": [D.INOUT, D.INOUT]},
            {"func_id": 1, "source": "kernels/aiv/ffts.cpp", "core_type": "aiv", "signature": [D.INOUT, D.INOUT]},
            {"func_id": 2, "source": "kernels/aiv/ffts.cpp", "core_type": "aiv", "signature": [D.INOUT, D.INOUT]},
        ],
    }

    CASES = [{"name": "two_clusters", "platforms": ["a2a3sim"], "params": {}}]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("scratch", torch.zeros(2, 4, dtype=torch.int32)),
            TensorArg("output", torch.zeros(2, 7, 2, dtype=torch.int32)),
        )

    def compute_golden(self, args, params):
        for block in range(2):
            for epoch in range(1, 8):
                value = block * 100 + epoch
                for lane in range(2):
                    args.output[block, epoch - 1, lane] = value * 3 + lane
            args.scratch[block] = torch.tensor([value, value, value * 2, value * 3], dtype=torch.int32)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
