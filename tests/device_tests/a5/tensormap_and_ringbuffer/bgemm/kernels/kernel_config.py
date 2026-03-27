"""
Kernel configuration for BGEMM (tensormap_and_ringbuffer Runtime).

Cube core (AIC) for matrix multiplication, Vector core (AIV) for accumulation.
Uses TPUSH/TPOP for cube-to-vector data transfer (bypasses GM).
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
        "name": "GEMM",
        "source": str(_KERNELS_ROOT / "mix" / "kernel_bgemm.cpp"),
        "core_type": "aic",
    },
    {
        "func_id": 1,
        "name": "ADD",
        "source": str(_KERNELS_ROOT / "mix" / "kernel_bgemm.cpp"),
        "core_type": "aiv",
    },
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 3,
}
