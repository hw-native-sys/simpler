"""
Paged Attention – Task Ring Stress Test

Reuses the same kernels and orchestration as paged_attention but configures
small ring buffers via RUNTIME_ENV to exercise:
  - CAS-based watermark advancement (slot reuse with task_window=16)
  - Heap ring wrapping (total allocations > 1MB heap)
  - Dependency pool wrapping (256 entries)
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent
_EXISTING_KERNELS = _KERNELS_ROOT.parent.parent / "paged_attention" / "kernels"

# Orchestration config (reuse existing source)
ORCHESTRATION = {
    "source": str(_EXISTING_KERNELS / "orchestration" / "paged_attention_orch.cpp"),
    "function_name": "build_paged_attention_graph",
}

# Kernel configs (reuse existing sources)
KERNELS = [
    # AIC kernels (matrix multiplication using Cube unit)
    {"func_id": 0, "name": "QK", "source": str(_EXISTING_KERNELS / "aic" / "aic_qk_matmul.cpp"), "core_type": "aic"},
    {"func_id": 2, "name": "PV", "source": str(_EXISTING_KERNELS / "aic" / "aic_pv_matmul.cpp"), "core_type": "aic"},
    {"func_id": 4, "name": "AIC_HUB", "source": str(_EXISTING_KERNELS / "aic" / "aic_hub.cpp"), "core_type": "aic"},
    # AIV kernels (vector operations)
    {"func_id": 1, "name": "SF", "source": str(_EXISTING_KERNELS / "aiv" / "aiv_softmax_prepare.cpp"), "core_type": "aiv"},
    {"func_id": 3, "name": "UP", "source": str(_EXISTING_KERNELS / "aiv" / "aiv_online_update.cpp"), "core_type": "aiv"},
    {"func_id": 5, "name": "AIV_HUB", "source": str(_EXISTING_KERNELS / "aiv" / "aiv_hub.cpp"), "core_type": "aiv"},
]

# Runtime configuration
RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 24,
}

# Small ring buffer sizes to stress task-ring slot reuse and heap wrapping
RUNTIME_ENV = {
    "PTO2_RING_TASK_WINDOW": "16",   # 16 slots (default 65536) - heavy slot reuse
    "PTO2_RING_HEAP": "1048576",     # 1MB (default 1GB) - heap ring wrapping
    "PTO2_RING_DEP_POOL": "256",     # 256 entries (default 65536) - dep pool wrapping
}
