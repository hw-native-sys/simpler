# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from pathlib import Path

from task_interface import ArgDirection as D  # pyright: ignore[reportAttributeAccessIssue]

_ROOT = Path(__file__).parent
_BASE_KERNELS = _ROOT.parent.parent / "benchmark_bgemm" / "kernels"

ORCHESTRATION = {
    "source": str(_ROOT / "orchestration" / "bgemm_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
    "signature": [D.IN, D.IN, D.OUT, D.IN],
}

KERNELS = [
    {
        "func_id": 0,
        "name": "GEMM",
        "source": str(_BASE_KERNELS / "aic" / "kernel_gemm_tile.cpp"),
        "core_type": "aic",
        "signature": [D.IN, D.IN, D.OUT],
    },
    {
        "func_id": 1,
        "name": "ADD",
        "source": str(_BASE_KERNELS / "aiv" / "kernel_tile_add.cpp"),
        "core_type": "aiv",
        "signature": [D.INOUT, D.IN],
    },
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 24,
}
