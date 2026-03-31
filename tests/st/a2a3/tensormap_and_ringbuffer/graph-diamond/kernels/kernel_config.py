# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Kernel configuration for diamond test (tensormap_and_ringbuffer).

Fork-join topology: A -> {B_0, ..., B_{W-1}} -> D.
Supports all-AIV, all-AIC, and mixed AIC+AIV branch modes.

Kernels:
  func_id=0: kernel_inc_aic  (AIC) - reads input, writes output = input + 1.0
  func_id=1: kernel_inc_aiv  (AIV) - reads input, writes output = input + 1.0
  func_id=2: kernel_noop_aiv (AIV) - increments INOUT by 1.0 (merge kernel)
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent
_CHAIN_KERNELS = _KERNELS_ROOT / ".." / ".." / "graph-chain_n" / "kernels"

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "diamond_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {
        "func_id": 0,
        "name": "INC_AIC",
        "source": str(_KERNELS_ROOT / "aic" / "kernel_inc_aic.cpp"),
        "core_type": "aic",
    },
    {
        "func_id": 1,
        "name": "INC_AIV",
        "source": str(_CHAIN_KERNELS / "aiv" / "kernel_inc_aiv.cpp"),
        "core_type": "aiv",
    },
    {
        "func_id": 2,
        "name": "NOOP_AIV",
        "source": str(_CHAIN_KERNELS / "aiv" / "kernel_noop_aiv.cpp"),
        "core_type": "aiv",
    },
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "orch_thread_num": 1,
    "block_dim": 24,
}
