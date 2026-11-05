#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise reused Softmax UB buffers with nonuniform block maxima."""

import ctypes

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


@scene_test(level=2, runtime="host_build_graph")
class TestSoftmaxPrepare(SceneTestCase):
    # All expected probabilities are BF16 values; these sums fit exactly in FP32.
    RTOL = 0.0
    ATOL = 0.0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/softmax_prepare.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.OUT, D.OUT, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aiv/aiv_softmax_prepare.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT, D.OUT, D.OUT],
            }
        ],
    }

    CASES = [
        {
            "name": f"m{m}_blocks{blocks}_valid{valid}_peak{peak}",
            "platforms": ["a2a3"],
            "params": {"m": m, "n": n, "blocks": blocks, "valid": valid, "peak": peak},
        }
        for m, n in ((16, 128), (64, 64))
        for blocks in (1, 2, 3, 4, 65)
        for valid in (1, n - 1, n)
        for peak in dict.fromkeys((None, 0, blocks // 2, blocks - 1))
    ]

    def generate_args(self, params):
        m, n, blocks = params["m"], params["n"], params["blocks"]
        scores = torch.zeros((blocks, m, n), dtype=torch.float32)
        if params["peak"] is None:
            scores.fill_(1.0)
        else:
            scores[params["peak"]].fill_(1.0)
        # GM is block-major. The descriptor's leading dimension selects the
        # kernel's tile shape; its remaining extent covers the packed blocks.
        return TaskArgsBuilder(
            TensorArg("sij", scores.reshape(m, blocks * n)),
            TensorArg("pij", torch.empty((m, blocks * n), dtype=torch.bfloat16)),
            TensorArg("mij", torch.empty(m, dtype=torch.float32)),
            TensorArg("lij", torch.empty(m, dtype=torch.float32)),
            Scalar("scale", ctypes.c_float(1.0)),
            Scalar("blocks", blocks),
            Scalar("valid", params["valid"]),
        )

    def compute_golden(self, args, params):
        m, n, blocks = params["m"], params["n"], params["blocks"]
        expected = torch.exp(args.sij.reshape(blocks, m, n) - 1.0).to(torch.bfloat16)
        expected[-1, :, params["valid"] :] = 0
        args.pij.copy_(expected.reshape(m, blocks * n))
        args.mij.fill_(1.0)
        args.lij.copy_(expected.float().sum((0, 2)))


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
