"""
Kernel configuration for scalar data dependency test (tensormap_and_ringbuffer).

Tests GetTensorData, SetTensorData, and add_inout with initial value.

Kernels:
  func_id=0: kernel_add (AIV) - element-wise tensor addition (128x128)
  func_id=1: kernel_noop (AIV) - empty kernel for allocation trigger
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "scalar_data_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {
        "func_id": 0,
        "source": str(_KERNELS_ROOT / "aiv" / "kernel_add.cpp"),
        "core_type": "aiv",
    },
    {
        "func_id": 1,
        "source": str(_KERNELS_ROOT / "aiv" / "kernel_noop.cpp"),
        "core_type": "aiv",
    },
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 3,
}
