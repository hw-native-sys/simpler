# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import ctypes

import torch


ALL_CASES = {
    "NestedManualScope": {"mode": 1},
    "ManualGetTensorData": {"mode": 2},
    "ManualSetTensorData": {"mode": 3},
    "ManualSelfDependency": {"mode": 4},
    "NestedAutoInsideManual": {"mode": 5},
}

DEFAULT_CASE = "NestedManualScope"
__outputs__ = ["tensor"]


def generate_inputs(params: dict) -> list:
    tensor = torch.arange(16, dtype=torch.float32)
    return [
        ("tensor", tensor),
        ("mode", ctypes.c_uint64(params["mode"])),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    del tensors, params
