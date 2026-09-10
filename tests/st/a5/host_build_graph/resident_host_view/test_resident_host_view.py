# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host reads observe device updates from preceding resident rounds."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


@scene_test(level=2, runtime="host_build_graph")
class TestResidentHostView(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/resident_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [{"func_id": 0, "source": "kernels/aiv/increment.cpp", "core_type": "aiv", "signature": [D.INOUT]}],
    }
    CASES = [{"name": "refresh", "platforms": ["a5sim", "a5"], "params": {}}]

    def generate_args(self, params):
        return TaskArgsBuilder(TensorArg("state", torch.zeros(128 * 128), child_memory=True, host_view=True))

    def compute_golden(self, args, params):
        args.state[0] += 2
        args.state.add_(1)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
