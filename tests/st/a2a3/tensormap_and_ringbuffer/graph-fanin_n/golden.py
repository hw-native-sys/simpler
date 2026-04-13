# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Golden script for fanin_N test (convergence barrier topology).

N independent producer tasks converge into 1 barrier task:
  seed(0.0) -> [Producer_0] -> prod_out_0 -.
  seed(0.0) -> [Producer_1] -> prod_out_1 -+-> [Barrier] -> result = 1.0
  ...                                      |
  seed(0.0) -> [Producer_{N-1}] -> prod_out_{N-1} -'

Each producer writes to an independent runtime tensor (no inter-producer deps).
The barrier task depends on all N producer outputs (via INPUT args) and writes
to the result tensor (INOUT), adding 1.0.

Tests: dependency convergence overhead — how efficiently the runtime tracks
N predecessors for a single barrier task.

Cases sweep fan-in width: 2, 4, 8, 15.

Args layout: [seed, result, fanin_width]
"""

import ctypes

import torch

__outputs__ = ["result"]

RTOL = 1e-5
ATOL = 1e-5

ALL_CASES = {
    "Fanin2": {"fanin_width": 2},
    "Fanin4": {"fanin_width": 4},
    "Fanin8": {"fanin_width": 8},
    "Fanin15": {"fanin_width": 15},
}

DEFAULT_CASE = "Fanin15"


def generate_inputs(params: dict) -> list:
    fanin_width = params["fanin_width"]

    seed = torch.zeros(1, dtype=torch.float32)
    result = torch.zeros(1, dtype=torch.float32)

    return [
        ("seed", seed),
        ("result", result),
        ("fanin_width", ctypes.c_int64(fanin_width)),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    result = torch.as_tensor(tensors["result"])
    # Barrier increments result from 0.0 to 1.0
    result[0] = 1.0
