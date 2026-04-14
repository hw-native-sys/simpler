# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
SPMD Paged Attention with TPUSH/TPOP (Combined AIC+AIV MixedKernels)

Single MixedKernels task per invocation. AIC handles QK/PV matmul,
AIV handles online softmax and update. Data flows via TPUSH/TPOP pipes:
  - sij pipe (C2V): QK scores
  - pij pipe (V2C): softmax probabilities
  - oi  pipe (C2V): PV output

The same source is compiled twice: once for AIC (__DAV_CUBE__) and
once for AIV (__DAV_VEC__), using if constexpr dispatch.
"""

from pathlib import Path

from simpler.task_interface import ArgDirection as D  # pyright: ignore[reportAttributeAccessIssue]

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "spmd_paged_attention_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {
        "func_id": 0,
        "name": "PA_AIC",
        "source": str(_KERNELS_ROOT / "mix" / "paged_attention_parallel.cpp"),
        "core_type": "aic",
    },
    {
        "func_id": 1,
        "name": "PA_AIV",
        "source": str(_KERNELS_ROOT / "mix" / "paged_attention_parallel.cpp"),
        "core_type": "aiv",
    },
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 24,
}
