# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Paged Attention — aicpu_build_graph Runtime

Kernels and orchestration config for paged attention (per-block version).
Uses explicit add_dependency for task ordering, scope-end batch publish.

AIC Kernels (Cube):
  - aic_qk_matmul: Q @ K^T computation
  - aic_pv_matmul: P @ V computation
  - aic_hub: placeholder hub task

AIV Kernels (Vector):
  - aiv_softmax_prepare: scale, rowmax, exp, rowsum
  - aiv_online_update: online softmax accumulation + fused normalization
  - aiv_hub: zero-initialize accumulators
"""

from pathlib import Path

from task_interface import ArgDirection as D  # pyright: ignore[reportAttributeAccessIssue]

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "paged_attention_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
    "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT],
}

KERNELS = [
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
    {
        "func_id": 4,
        "name": "AIC_HUB",
        "source": str(_KERNELS_ROOT / "aic" / "aic_hub.cpp"),
        "core_type": "aic",
        "signature": [],
    },
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
    {
        "func_id": 5,
        "name": "AIV_HUB",
        "source": str(_KERNELS_ROOT / "aiv" / "aiv_hub.cpp"),
        "core_type": "aiv",
        "signature": [],
    },
]

RUNTIME_CONFIG = {
    "runtime": "aicpu_build_graph",
    "aicpu_thread_num": 4,
    "block_dim": 24,
}
