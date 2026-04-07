# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Paged Attention Kernel and Orchestration Configuration

Defines the kernels and orchestration function for paged attention
with AIC/AIV subgraph splitting:

AIC Kernels (Matrix Multiplication):
  - aic_qk_matmul: Q @ K^T computation
  - aic_pv_matmul: P @ V computation

AIV Kernels (Vector Operations):
  - aiv_softmax_prepare: scale, rowmax, exp, rowsum
  - aiv_online_update: online softmax accumulation + fused normalization
"""

from pathlib import Path

from task_interface import ArgDirection as D  # pyright: ignore[reportAttributeAccessIssue]

_KERNELS_ROOT = Path(__file__).parent

# Orchestration config
ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "paged_attention_orch.cpp"),
    "function_name": "build_paged_attention_graph",
    "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT],
}

# Kernel configs
KERNELS = [
    # AIC kernels (matrix multiplication using Cube unit)
    {
        "func_id": 0,
        "name": "QK",
        "source": str(_KERNELS_ROOT / "aic" / "aic_qk_matmul.cpp"),
        "core_type": "aic",
        "signature": [D.IN, D.IN, D.OUT],
    },
    {
        "func_id": 2,
        "name": "PV",
        "source": str(_KERNELS_ROOT / "aic" / "aic_pv_matmul.cpp"),
        "core_type": "aic",
        "signature": [D.IN, D.IN, D.OUT],
    },
    # AIV kernels (vector operations)
    {
        "func_id": 1,
        "name": "SF",
        "source": str(_KERNELS_ROOT / "aiv" / "aiv_softmax_prepare.cpp"),
        "core_type": "aiv",
        "signature": [D.IN, D.OUT, D.OUT, D.OUT],
    },
    {
        "func_id": 3,
        "name": "UP",
        "source": str(_KERNELS_ROOT / "aiv" / "aiv_online_update.cpp"),
        "core_type": "aiv",
        "signature": [D.IN, D.IN, D.IN, D.INOUT, D.INOUT, D.INOUT, D.INOUT],
    },
]

# Runtime configuration
RUNTIME_CONFIG = {
    "runtime": "host_build_graph",
    "aicpu_thread_num": 3,
    "block_dim": 3,
}
