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
_BASE_KERNELS = _ROOT.parent.parent / "batch_paged_attention" / "kernels"

ORCHESTRATION = {
    "source": str(_ROOT / "orchestration" / "paged_attention_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
    "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT],
}

KERNELS = [
    {
        "func_id": 0,
        "name": "QK",
        "source": str(_BASE_KERNELS / "aic" / "aic_qk_matmul.cpp"),
        "core_type": "aic",
        "signature": [D.IN, D.IN, D.OUT],
    },
    {
        "func_id": 2,
        "name": "PV",
        "source": str(_BASE_KERNELS / "aic" / "aic_pv_matmul.cpp"),
        "core_type": "aic",
        "signature": [D.IN, D.IN, D.OUT],
    },
    {
        "func_id": 1,
        "name": "SF",
        "source": str(_BASE_KERNELS / "aiv" / "aiv_softmax_prepare.cpp"),
        "core_type": "aiv",
        "signature": [D.IN, D.OUT, D.OUT, D.OUT],
    },
    {
        "func_id": 3,
        "name": "UP",
        "source": str(_BASE_KERNELS / "aiv" / "aiv_online_update.cpp"),
        "core_type": "aiv",
        "signature": [D.IN, D.IN, D.IN, D.INOUT, D.INOUT, D.INOUT, D.INOUT],
    },
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 24,
}
