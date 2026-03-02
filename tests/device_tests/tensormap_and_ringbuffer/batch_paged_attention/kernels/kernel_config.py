"""
Batch Paged Attention Kernel and Orchestration Configuration

Defines the kernels and orchestration function for batched paged attention
with AIC/AIV subgraph splitting:

AIC Kernels (Matrix Multiplication):
  - aic_qk_matmul: Q @ K^T computation (batched)
  - aic_pv_matmul: P @ V computation (batched)

AIV Kernels (Vector Operations):
  - aiv_softmax_prepare: scale, rowmax, exp, rowsum (batched)
  - aiv_online_update: online softmax accumulation + fused normalization (batched)
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

# Orchestration config
ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "paged_attention_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

# Kernel configs
KERNELS = [
    # AIC kernels (matrix multiplication using Cube unit)
    {"func_id": 0, "name": "QK", "source": str(_KERNELS_ROOT / "aic" / "aic_qk_matmul.cpp"),       "core_type": "aic"},
    {"func_id": 2, "name": "PV", "source": str(_KERNELS_ROOT / "aic" / "aic_pv_matmul.cpp"),       "core_type": "aic"},
    {"func_id": 4, "name": "AIC_HUB", "source": str(_KERNELS_ROOT / "aic" / "aic_hub.cpp"),       "core_type": "aic"},
    # AIV kernels (vector operations)
    {"func_id": 1, "name": "SF", "source": str(_KERNELS_ROOT / "aiv" / "aiv_softmax_prepare.cpp"), "core_type": "aiv"},
    {"func_id": 3, "name": "UP", "source": str(_KERNELS_ROOT / "aiv" / "aiv_online_update.cpp"),   "core_type": "aiv"},
    {"func_id": 5, "name": "AIV_HUB", "source": str(_KERNELS_ROOT / "aiv" / "aiv_hub.cpp"),       "core_type": "aiv"},
]

# Runtime configuration
RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 24,
}
