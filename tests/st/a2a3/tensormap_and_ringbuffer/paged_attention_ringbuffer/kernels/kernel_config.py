# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Paged Attention Ring Buffer Stress Test

Reuses paged_attention kernels and orchestration with deliberately small
ring buffer sizes to exercise and guard the ring buffer rotation logic.

The orchestration uses an inner PTO2_SCOPE per block, allowing per-block
ring resources to be reclaimed. Combined with small ring sizes, this
stresses the back-pressure and reclamation paths.

Environment overrides:
  PTO2_RING_TASK_WINDOW = 128   (vs default 65536, 8x smaller than prev 1024)
  PTO2_RING_HEAP        = 256KB (vs default 1GB,   4x smaller than prev 1MB)
  PTO2_RING_DEP_POOL    = 256   (vs default 65536, 4x smaller than prev 1024)
"""

from pathlib import Path

from simpler.task_interface import ArgDirection as D  # pyright: ignore[reportAttributeAccessIssue]

# Point to paged_attention's kernel sources (no duplication)
_PA_KERNELS = Path(__file__).parent / ".." / ".." / "paged_attention" / "kernels"

ORCHESTRATION = {
    "source": str(_PA_KERNELS / "orchestration" / "paged_attention_orch.cpp"),
    "function_name": "build_paged_attention_graph",
    "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT],
}

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
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 24,
}

# Small ring buffer sizes — see module docstring for rationale.
RUNTIME_ENV = {
    "PTO2_RING_TASK_WINDOW": "128",
    "PTO2_RING_HEAP": "262144",
    "PTO2_RING_DEP_POOL": "256",
}
