# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Kernel and Orchestration Configuration

Defines the kernels and orchestration function used by the matmul example.
Supports both hardware (a2a3) and simulation (a2a3sim) platforms.
"""

from pathlib import Path

from task_interface import ArgDirection as D  # pyright: ignore[reportAttributeAccessIssue]

_KERNELS_ROOT = Path(__file__).parent

# Orchestration config
ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "matmul_orch.cpp"),
    "function_name": "build_matmul_graph",
    "signature": [D.IN, D.IN, D.IN, D.OUT],
}

# Kernel configs
# func_id mapping:
#   0: kernel_log_sqrt (TLOG + TSQRT) - AIV
#   1: kernel_matmul   (TMATMUL)      - AIC
#   2: kernel_add_exp  (TADD + TEXP)  - AIV
KERNELS = [
    {
        "func_id": 0,
        "source": str(_KERNELS_ROOT / "aiv" / "kernel_log_sqrt.cpp"),
        "core_type": "aiv",
        "signature": [D.IN, D.OUT],
    },
    {
        "func_id": 1,
        "source": str(_KERNELS_ROOT / "aic" / "kernel_matmul.cpp"),
        "core_type": "aic",
        "signature": [D.IN, D.IN, D.OUT],
    },
    {
        "func_id": 2,
        "source": str(_KERNELS_ROOT / "aiv" / "kernel_add_exp.cpp"),
        "core_type": "aiv",
        "signature": [D.IN, D.IN, D.OUT],
    },
]

# Runtime configuration
RUNTIME_CONFIG = {
    "runtime": "host_build_graph",
    "aicpu_thread_num": 3,
    "block_dim": 3,
}
