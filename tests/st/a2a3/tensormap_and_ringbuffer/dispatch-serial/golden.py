# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Golden script for dispatch_throughput test.

Measures scheduler throughput by submitting N noop tasks serially.
Each task increments a counter, so the final output equals N (or N/2
for each core type in AIC+AIV mode).

Cases sweep across task counts and core types:
  Case1:  100 AIC-only tasks
  Case2:  500 AIC-only tasks
  Case3: 1000 AIC-only tasks
  Case4: 2000 AIC-only tasks
  Case5:  100 AIV-only tasks
  Case6:  500 AIV-only tasks
  Case7: 1000 AIV-only tasks
  Case8: 2000 AIV-only tasks
  Case9:  100 AIC+AIV alternating tasks
  Case10: 500 AIC+AIV alternating tasks
  Case11:1000 AIC+AIV alternating tasks
  Case12:2000 AIC+AIV alternating tasks

Args layout: [out_aic, out_aiv, num_tasks, mode]
"""

import ctypes

import torch

__outputs__ = ["out_aic", "out_aiv"]

RTOL = 1e-3
ATOL = 1e-1  # Accumulated float additions may drift slightly

ALL_CASES = {
    # AIC-only (mode=0)
    "Case1": {"num_tasks": 100, "mode": 0},
    "Case2": {"num_tasks": 500, "mode": 0},
    "Case3": {"num_tasks": 1000, "mode": 0},
    "Case4": {"num_tasks": 2000, "mode": 0},
    # AIV-only (mode=1)
    "Case5": {"num_tasks": 100, "mode": 1},
    "Case6": {"num_tasks": 500, "mode": 1},
    "Case7": {"num_tasks": 1000, "mode": 1},
    "Case8": {"num_tasks": 2000, "mode": 1},
    # AIC+AIV alternating (mode=2)
    "Case9": {"num_tasks": 100, "mode": 2},
    "Case10": {"num_tasks": 500, "mode": 2},
    "Case11": {"num_tasks": 1000, "mode": 2},
    "Case12": {"num_tasks": 2000, "mode": 2},
}

DEFAULT_CASE = "Case2"


def generate_inputs(params: dict) -> list:
    num_tasks = params["num_tasks"]
    mode = params["mode"]

    out_aic = torch.zeros(1, dtype=torch.float32)
    out_aiv = torch.zeros(1, dtype=torch.float32)

    return [
        ("out_aic", out_aic),
        ("out_aiv", out_aiv),
        ("num_tasks", ctypes.c_int64(num_tasks)),
        ("mode", ctypes.c_int64(mode)),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    num_tasks = params["num_tasks"]
    mode = params["mode"]

    out_aic = torch.as_tensor(tensors["out_aic"])
    out_aiv = torch.as_tensor(tensors["out_aiv"])

    if mode == 0:
        # AIC-only: all N tasks increment out_aic
        out_aic[0] = float(num_tasks)
    elif mode == 1:
        # AIV-only: all N tasks increment out_aiv
        out_aiv[0] = float(num_tasks)
    elif mode == 2:
        # AIC+AIV alternating: even tasks → AIC, odd tasks → AIV
        aic_count = (num_tasks + 1) // 2
        aiv_count = num_tasks // 2
        out_aic[0] = float(aic_count)
        out_aiv[0] = float(aiv_count)
