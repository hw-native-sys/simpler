# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Golden script for diamond test (fork-join topology).

Diamond DAG: A -> {B_0, B_1, ..., B_{W-1}} -> D

  seed(0.0) -> [Source A] -> a_out(1.0) -> [Branch B_0] -> b_out_0(2.0) -.
                                         -> [Branch B_1] -> b_out_1(2.0) -+-> [Merge D] -> result(1.0)
                                         -> ...                            |
                                         -> [Branch B_{W-1}] -> ...       -'

Source A and branches each add 1.0 to their input. Merge D increments
result from 0.0 to 1.0 (using the noop kernel, which only touches result).

Three branch modes:
  mode=0: All AIV branches
  mode=1: All AIC branches
  mode=2: Mixed AIC+AIV branches (even=AIC, odd=AIV)

Cases sweep branch width (2/4/8/15) x mode (AIV/AIC/mixed).

Args layout: [seed, result, width, mode]
"""

import ctypes

import torch

__outputs__ = ["result"]

RTOL = 1e-5
ATOL = 1e-5

ALL_CASES = {
    # All-AIV branches
    "W2_AIV": {"width": 2, "mode": 0},
    "W4_AIV": {"width": 4, "mode": 0},
    "W8_AIV": {"width": 8, "mode": 0},
    "W15_AIV": {"width": 15, "mode": 0},
    # All-AIC branches
    "W2_AIC": {"width": 2, "mode": 1},
    "W4_AIC": {"width": 4, "mode": 1},
    "W8_AIC": {"width": 8, "mode": 1},
    "W15_AIC": {"width": 15, "mode": 1},
    # Mixed AIC+AIV branches
    "W2_Mixed": {"width": 2, "mode": 2},
    "W4_Mixed": {"width": 4, "mode": 2},
    "W8_Mixed": {"width": 8, "mode": 2},
    "W15_Mixed": {"width": 15, "mode": 2},
}

DEFAULT_CASE = "W15_AIV"


def generate_inputs(params: dict) -> list:
    width = params["width"]
    mode = params["mode"]

    seed = torch.zeros(1, dtype=torch.float32)
    result = torch.zeros(1, dtype=torch.float32)

    return [
        ("seed", seed),
        ("result", result),
        ("width", ctypes.c_int64(width)),
        ("mode", ctypes.c_int64(mode)),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    result = torch.as_tensor(tensors["result"])
    # Merge D increments result from 0.0 to 1.0
    result[0] = 1.0
