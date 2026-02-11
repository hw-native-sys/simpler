"""
Paged Attention Kernel and Orchestration Configuration (aicpu_build_graph)

Uses the aicpu_build_graph runtime where the AICPU device builds the task
graph via a dlopen'd orchestration plugin, while scheduler threads execute
tasks concurrently.

Kernels are identical to the host_build_graph version:

AIC Kernels (Matrix Multiplication):
  - aic_qk_matmul: Q @ K^T computation
  - aic_pv_matmul: P @ V computation

AIV Kernels (Vector Operations):
  - aiv_softmax_prepare: scale, rowmax, exp, rowsum
  - aiv_online_update: online softmax accumulation + fused normalization
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

RUNTIME_CONFIG = {
    "runtime": "aicpu_build_graph",
    # 1 AICPU thread builds tasks while 3 AICPU threads schedule/execute.
    "aicpu_thread_num": 4,
    "block_dim": 24,
}

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "paged_attention_orch.cpp"),
    "function_name": "orchestration",
}

# Runtime behavior knobs.
#
# `RUNTIME_ENV` is applied both during runtime compilation and during runtime
# initialization (host `dlopen()` + orchestrator call).
#
# PTO_AICPU_BUILD_GRAPH_BUILD_MODE = "1" (concurrent build||schedule, default)
# PTO_AICPU_BUILD_GRAPH_BUILD_MODE = "0" (sequential build->schedule)
RUNTIME_ENV = {
    "PTO_AICPU_BUILD_GRAPH_BUILD_MODE": "1",
}

# Kernel configs (same func_ids and core_types as host_build_graph version)
KERNELS = [
    # AIC kernels (matrix multiplication using Cube unit)
    {"func_id": 0, "name": "QK", "source": str(_KERNELS_ROOT / "aic" / "aic_qk_matmul.cpp"),       "core_type": "aic"},
    {"func_id": 2, "name": "PV", "source": str(_KERNELS_ROOT / "aic" / "aic_pv_matmul.cpp"),       "core_type": "aic"},
    # AIV kernels (vector operations)
    {"func_id": 1, "name": "SF", "source": str(_KERNELS_ROOT / "aiv" / "aiv_softmax_prepare.cpp"), "core_type": "aiv"},
    {"func_id": 3, "name": "UP", "source": str(_KERNELS_ROOT / "aiv" / "aiv_online_update.cpp"),   "core_type": "aiv"},
]
