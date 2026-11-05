#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Batch-4 paged attention example using the current host-build-graph runtime."""

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import TensorTransfer

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.goldens.paged_attention import compute_golden as _pa_compute_golden
from simpler_setup.goldens.paged_attention import generate_inputs as _pa_generate_inputs

_KERNEL_DIR = "../../st/a5/host_build_graph/paged_attention_unroll/kernels"


@scene_test(level=2, runtime="host_build_graph")
class TestPaHbgB4(SceneTestCase):
    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/paged_attention_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "QK",
                "source": f"{_KERNEL_DIR}/aic/aic_qk_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "name": "SF",
                "source": f"{_KERNEL_DIR}/aiv/aiv_softmax_prepare.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT, D.OUT, D.OUT],
            },
            {
                "func_id": 2,
                "name": "PV",
                "source": f"{_KERNEL_DIR}/aic/aic_pv_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 3,
                "name": "UP",
                "source": f"{_KERNEL_DIR}/aiv/aiv_online_update.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.IN, D.INOUT, D.INOUT, D.INOUT, D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "B4",
            "platforms": ["a5", "a5sim"],
            "params": {
                "batch": 4,
                "num_heads": 16,
                "kv_head_num": 1,
                "head_dim": 128,
                "block_size": 128,
                "context_len": 8192,
                "max_model_len": 32768,
                "dtype": "bfloat16",
            },
        },
    ]

    def generate_args(self, params):
        params = {**params, "variant": "paged_attention_unroll"}
        result = _pa_generate_inputs(params)
        specs = []
        for name, value in result:
            if isinstance(value, torch.Tensor):
                specs.append(TensorArg(name, value))
            else:
                specs.append(Scalar(name, value))
        tensors = [s for s in specs if isinstance(s, TensorArg)]
        scalars = [s for s in specs if isinstance(s, Scalar)]
        by_name = {s.name: s.value for s in tensors}
        tensors.append(TensorArg("host_context_lens", by_name["context_lens"].clone(), transfer=TensorTransfer.NONE))
        return TaskArgsBuilder(*tensors, *scalars)

    def compute_golden(self, args, params):
        tensors = {s.name: s.value for s in args.specs if isinstance(s, TensorArg)}
        _pa_compute_golden(tensors, params)
        for s in args.specs:
            if isinstance(s, TensorArg) and s.name in tensors:
                getattr(args, s.name)[:] = tensors[s.name]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
