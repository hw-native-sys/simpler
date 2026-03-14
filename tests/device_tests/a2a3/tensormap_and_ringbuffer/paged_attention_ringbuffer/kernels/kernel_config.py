"""
Paged Attention Ring Buffer Stress Test

Reuses paged_attention kernels and orchestration with deliberately small
ring buffer sizes to exercise and guard the ring buffer rotation logic.

Environment overrides:
  PTO2_RING_TASK_WINDOW = 1024  (vs default 65536)
  PTO2_RING_HEAP        = 1MB   (vs default 1GB)
  PTO2_RING_DEP_POOL    = 1024  (vs default 65536)
"""

from pathlib import Path

# Point to paged_attention's kernel sources (no duplication)
_PA_KERNELS = Path(__file__).parent / ".." / ".." / "paged_attention" / "kernels"

ORCHESTRATION = {
    "source": str(_PA_KERNELS / "orchestration" / "paged_attention_orch.cpp"),
    "function_name": "build_paged_attention_graph",
}

KERNELS = [
    {"func_id": 0, "name": "QK", "source": str(_PA_KERNELS / "aic" / "aic_qk_matmul.cpp"), "core_type": "aic"},
    {"func_id": 2, "name": "PV", "source": str(_PA_KERNELS / "aic" / "aic_pv_matmul.cpp"), "core_type": "aic"},
    {"func_id": 4, "name": "AIC_HUB", "source": str(_PA_KERNELS / "aic" / "aic_hub.cpp"), "core_type": "aic"},
    {"func_id": 1, "name": "SF", "source": str(_PA_KERNELS / "aiv" / "aiv_softmax_prepare.cpp"), "core_type": "aiv"},
    {"func_id": 3, "name": "UP", "source": str(_PA_KERNELS / "aiv" / "aiv_online_update.cpp"), "core_type": "aiv"},
    {"func_id": 5, "name": "AIV_HUB", "source": str(_PA_KERNELS / "aiv" / "aiv_hub.cpp"), "core_type": "aiv"},
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 24,
}

# Small ring buffer sizes to stress rotation/reclamation
RUNTIME_ENV = {
    "PTO2_RING_TASK_WINDOW": "1024",
    "PTO2_RING_HEAP": "1048576",
    "PTO2_RING_DEP_POOL": "1024",
}
