# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
SPMD Paged Attention Kernel and Orchestration Configuration (fixed block_num=24)

Uses SPMD parallelism with a fixed hardware block_num of 24.
total_logical_blocks = batch * q_loop work items are distributed across
the 24 hardware blocks via stride loops inside each kernel.
q_tile adapts to num_heads: q_tile = min(num_heads, MAX_Q_TILE).

Softmax and online-update run as MIX tasks (AIC idle + AIV0 + AIV1), with the
two AIVs splitting the q_tile query rows via get_sub_block_id().

AIC Kernels (Matrix Multiplication):
  - aic_qk_matmul: Q @ K^T (SPMD across batch*q_loop)
  - aic_pv_matmul: P @ V (SPMD across batch*q_loop)
  - aic_hub: no-op, occupies the AIC slot of softmax/update MIX tasks

AIV Kernels (Vector Operations):
  - aiv_softmax_prepare: scale, rowmax, exp, rowsum on sub-tile (q_tile/2 rows)
  - aiv_online_update: online softmax accumulation + normalization on sub-tile
  - aiv_hub: no-op, used to allocate persistent accumulators
"""

from pathlib import Path

from simpler.task_interface import ArgDirection as D  # pyright: ignore[reportAttributeAccessIssue]

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "spmd_paged_attention_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    # AIC kernels (matrix multiplication using Cube unit)
    {
        "func_id": 0,
        "name": "SPMD_QK",
        "source": str(_KERNELS_ROOT / "aic" / "aic_qk_matmul.cpp"),
        "core_type": "aic",
    },
    {
        "func_id": 1,
        "name": "SPMD_PV",
        "source": str(_KERNELS_ROOT / "aic" / "aic_pv_matmul.cpp"),
        "core_type": "aic",
    },
    {
        "func_id": 2,
        "name": "AIC_HUB",
        "source": str(_KERNELS_ROOT / "aic" / "aic_hub.cpp"),
        "core_type": "aic",
    },
    # AIV kernels (vector operations)
    {
        "func_id": 3,
        "name": "SPMD_SF",
        "source": str(_KERNELS_ROOT / "aiv" / "aiv_softmax_prepare.cpp"),
        "core_type": "aiv",
    },
    {
        "func_id": 4,
        "name": "SPMD_UP",
        "source": str(_KERNELS_ROOT / "aiv" / "aiv_online_update.cpp"),
        "core_type": "aiv",
    },
    {
        "func_id": 5,
        "name": "AIV_HUB",
        "source": str(_KERNELS_ROOT / "aiv" / "aiv_hub.cpp"),
        "core_type": "aiv",
    },
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 24,
}
