"""
Kernel and Orchestration Configuration

This example uses the aicpu_build_graph runtime with PTO2 ring buffer infrastructure:
- Device-side orchestration with explicit dependency management
- Intermediate tensors allocated from HeapRing
- Tasks batch-published at scope_end
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "orchestration.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {"func_id": 0, "source": str(_KERNELS_ROOT / "aiv" / "kernel_add.cpp"), "core_type": "aiv"},
    {"func_id": 1, "source": str(_KERNELS_ROOT / "aiv" / "kernel_add_scalar.cpp"), "core_type": "aiv"},
    {"func_id": 2, "source": str(_KERNELS_ROOT / "aiv" / "kernel_mul.cpp"), "core_type": "aiv"},
]

RUNTIME_CONFIG = {
    "runtime": "aicpu_build_graph",
    "aicpu_thread_num": 4,
    "block_dim": 3,
}
