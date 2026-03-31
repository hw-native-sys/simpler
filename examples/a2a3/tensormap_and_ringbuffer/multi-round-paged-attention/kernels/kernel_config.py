# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Multi-Round Paged Attention Configuration

Reuses kernel sources from the paged_attention example with 10 rounds
for benchmarking multi-round execution.
"""

from pathlib import Path

from task_interface import ArgDirection as D  # pyright: ignore[reportAttributeAccessIssue]

_KERNELS_ROOT = Path(__file__).parent
_PA_KERNELS = _KERNELS_ROOT.parent.parent / "paged_attention" / "kernels"

# Orchestration config — reuse paged_attention orchestration source
ORCHESTRATION = {
    "source": str(_PA_KERNELS / "orchestration" / "paged_attention_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
    "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT],
}

# Kernel configs — reuse paged_attention kernel sources
KERNELS = [
    {
        "func_id": 0,
        "name": "QK",
        "source": str(_PA_KERNELS / "aic" / "aic_qk_matmul.cpp"),
        "core_type": "aic",
        "signature": [D.IN, D.IN, D.OUT],
    },
    {
        "func_id": 2,
        "name": "PV",
        "source": str(_PA_KERNELS / "aic" / "aic_pv_matmul.cpp"),
        "core_type": "aic",
        "signature": [D.IN, D.IN, D.OUT],
    },
    {
        "func_id": 4,
        "name": "AIC_HUB",
        "source": str(_PA_KERNELS / "aic" / "aic_hub.cpp"),
        "core_type": "aic",
        "signature": [],
    },
    {
        "func_id": 1,
        "name": "SF",
        "source": str(_PA_KERNELS / "aiv" / "aiv_softmax_prepare.cpp"),
        "core_type": "aiv",
        "signature": [D.IN, D.OUT, D.OUT, D.OUT],
    },
    {
        "func_id": 3,
        "name": "UP",
        "source": str(_PA_KERNELS / "aiv" / "aiv_online_update.cpp"),
        "core_type": "aiv",
        "signature": [D.IN, D.IN, D.IN, D.INOUT, D.INOUT, D.INOUT, D.INOUT],
    },
    {
        "func_id": 5,
        "name": "AIV_HUB",
        "source": str(_PA_KERNELS / "aiv" / "aiv_hub.cpp"),
        "core_type": "aiv",
        "signature": [],
    },
]

# Runtime configuration
RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 24,
    "rounds": 10,
}
