# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Golden script for aicpu_build_graph example.

Computation:
    f = (a + b + 1) * (a + b + 2)
    where a=2.0, b=3.0, so f=42.0

Args layout: [a, b, f]  — shape/dtype/size in ContinuousTensor metadata
"""

import torch

__outputs__ = ["f"]

RTOL = 1e-5
ATOL = 1e-5


def generate_inputs(params: dict) -> list:
    ROWS = 128
    COLS = 128
    SIZE = ROWS * COLS

    a = torch.full((SIZE,), 2.0, dtype=torch.float32)
    b = torch.full((SIZE,), 3.0, dtype=torch.float32)
    f = torch.zeros(SIZE, dtype=torch.float32)

    return [
        ("a", a),
        ("b", b),
        ("f", f),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    a = torch.as_tensor(tensors["a"])
    b = torch.as_tensor(tensors["b"])
    tensors["f"][:] = (a + b + 1) * (a + b + 2)
