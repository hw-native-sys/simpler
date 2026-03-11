"""
Kernel configuration for mixed AIC+AIV example (tensormap_and_ringbuffer Runtime).

Covers all 5 resource shapes:
  - AIC_ONLY:   standalone matmul
  - AIV_X1:     standalone add
  - AIV_X2:     add (AIV0) + mul (AIV1)
  - AIC_AIV_X1: matmul (AIC) + add (AIV0)
  - AIC_AIV_X2: matmul (AIC) + add (AIV0) + mul (AIV1)
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "mixed_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {
        "func_id": 0,
        "name": "MATMUL",
        "source": str(_KERNELS_ROOT / "aic" / "kernel_matmul.cpp"),
        "core_type": "aic",
    },
    {
        "func_id": 1,
        "name": "ADD",
        "source": str(_KERNELS_ROOT / "aiv" / "kernel_add.cpp"),
        "core_type": "aiv",
    },
    {
        "func_id": 2,
        "name": "MUL",
        "source": str(_KERNELS_ROOT / "aiv" / "kernel_mul.cpp"),
        "core_type": "aiv",
    },
    {
        "func_id": 3,
        "name": "ADD_STANDALONE",
        "source": str(_KERNELS_ROOT / "aiv" / "kernel_add_standalone.cpp"),
        "core_type": "aiv",
    },
    {
        "func_id": 4,
        "name": "MUL_STANDALONE",
        "source": str(_KERNELS_ROOT / "aiv" / "kernel_mul_standalone.cpp"),
        "core_type": "aiv",
    },
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 3,
}
