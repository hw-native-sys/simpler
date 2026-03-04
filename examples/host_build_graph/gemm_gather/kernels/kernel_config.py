"""
Kernel and orchestration configuration for gemm_gather (a2a3).

Two kernels: GEMM (AIC), Gather (AIV).
Dependency: compute first, then communication (t0 -> t1).
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "gemm_gather_orch.cpp"),
    "function_name": "build_gemm_gather_graph",
}

KERNELS = [
    {"func_id": 0, "name": "GEMM", "source": str(_KERNELS_ROOT / "aic" / "kernel_gemm.cpp"), "core_type": "aic"},
    {"func_id": 1, "name": "Gather", "source": str(_KERNELS_ROOT / "aiv" / "kernel_gather.cpp"), "core_type": "aiv"},
]

RUNTIME_CONFIG = {
    "runtime": "host_build_graph",
    "aicpu_thread_num": 3,
    "block_dim": 3,
}
