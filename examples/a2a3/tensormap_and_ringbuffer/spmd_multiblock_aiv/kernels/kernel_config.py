# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Kernel configuration for SPMD multi-block AIV test (tensormap_and_ringbuffer Runtime).

Submits a single AIV task with block_num=4 so each block writes its
block_idx at a distinct cacheline-aligned offset.
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "spmd_multiblock_aiv_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {
        "func_id": 0,
        "name": "SPMD_WRITE_AIV",
        "source": str(_KERNELS_ROOT / "aiv" / "kernel_spmd_write.cpp"),
        "core_type": "aiv",
    },
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "orch_thread_num": 1,
    "block_dim": 24,
}
