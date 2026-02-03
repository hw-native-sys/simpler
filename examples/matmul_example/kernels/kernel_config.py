"""
Kernel and Orchestration Configuration

Defines the kernels and orchestration function used by the matmul example.
Supports both hardware (a2a3) and simulation (a2a3sim) platforms.
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

# Orchestration config
ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "matmul_orch.cpp"),
    "function_name": "build_matmul_graph",
}

# Kernel configs
# func_id mapping:
#   0: kernel_log_sqrt (TLOG + TSQRT) - AIV
#   1: kernel_matmul   (TMATMUL)      - AIC
#   2: kernel_add_exp  (TADD + TEXP)  - AIV
KERNELS = [
    {"func_id": 0, "source": str(_KERNELS_ROOT / "aiv" / "kernel_log_sqrt.cpp"), "core_type": "aiv"},
    {"func_id": 1, "source": str(_KERNELS_ROOT / "aic" / "kernel_matmul.cpp"),   "core_type": "aic"},
    {"func_id": 2, "source": str(_KERNELS_ROOT / "aiv" / "kernel_add_exp.cpp"),  "core_type": "aiv"},
]
