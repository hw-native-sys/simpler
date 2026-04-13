# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Golden script for task_scaling test.

Measures dispatch overhead vs task count. Submits N independent noop tasks,
each writing 1.0 to a separate cache-line-aligned slot. Output tensor is
padded so each task's slot sits on its own cache line (stride = 16 float32
elements = 64 bytes), avoiding false sharing across non-coherent AICore L1
caches.

Cases parameterize task count (100→2000) and core type:
  AIC-only sweep:  100, 500, 1000, 2000 tasks
  AIV-only sweep:  100, 500, 1000, 2000 tasks
  AIC+AIV sweep:   100, 500, 1000, 2000 tasks

Args layout: [output, num_tasks, mode]
"""

import ctypes

import torch

__outputs__ = ["output"]

RTOL = 1e-5
ATOL = 1e-5

# Each task writes to a separate cache line to avoid false sharing
# across non-coherent AICore L1 caches (64B = 16 float32 elements).
CACHE_LINE_ELEMS = 16

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

    output = torch.zeros(num_tasks * CACHE_LINE_ELEMS, dtype=torch.float32)

    return [
        ("output", output),
        ("num_tasks", ctypes.c_int64(num_tasks)),
        ("mode", ctypes.c_int64(mode)),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    num_tasks = params["num_tasks"]
    output = torch.as_tensor(tensors["output"])

    # Each independent task writes 1.0 to its cache-line-aligned slot
    for i in range(num_tasks):
        output[i * CACHE_LINE_ELEMS] = 1.0
