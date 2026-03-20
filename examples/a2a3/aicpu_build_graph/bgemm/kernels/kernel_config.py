"""
Kernel configuration for BGEMM (aicpu_build_graph Runtime).

Cube core (AIC) for matrix multiplication, Vector core (AIV) for element-wise addition.
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "bgemm_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {
        "func_id": 0,
        "source": str(_KERNELS_ROOT / "aic" / "kernel_gemm_tile.cpp"),
        "core_type": "aic",
    },
    {
        "func_id": 1,
        "source": str(_KERNELS_ROOT / "aiv" / "kernel_tile_add.cpp"),
        "core_type": "aiv",
    },
]

RUNTIME_CONFIG = {
    "runtime": "aicpu_build_graph",
    "aicpu_thread_num": 4,
    "block_dim": 3,
}
