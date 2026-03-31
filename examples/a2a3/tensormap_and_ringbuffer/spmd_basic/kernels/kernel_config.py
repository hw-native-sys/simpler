# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Kernel configuration for SPMD basic test (tensormap_and_ringbuffer Runtime).

Submits a single MIX task (AIC + AIV0 + AIV1) so all three sub_block_id
values are exercised in one dispatch.
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "spmd_basic_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {
        "func_id": 0,
        "name": "SPMD_READ_AIC",
        "source": str(_KERNELS_ROOT / "aic" / "kernel_spmd_read.cpp"),
        "core_type": "aic",
    },
    {
        "func_id": 1,
        "name": "SPMD_READ_AIV0",
        "source": str(_KERNELS_ROOT / "aiv" / "kernel_spmd_read.cpp"),
        "core_type": "aiv",
    },
    {
        "func_id": 2,
        "name": "SPMD_READ_AIV1",
        "source": str(_KERNELS_ROOT / "aiv" / "kernel_spmd_read.cpp"),
        "core_type": "aiv",
    },
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "orch_thread_num": 1,
    "block_dim": 24,
}
