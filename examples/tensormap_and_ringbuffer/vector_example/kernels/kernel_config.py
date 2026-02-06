"""
Tensormap and Ringbuffer Kernel and Orchestration Configuration

Defines the kernels and orchestration function used by the tensormap_and_ringbuffer example.
Supports both hardware (a2a3) and simulation (a2a3sim) platforms.

This runtime uses device-side orchestration (AICPU thread 3), so the orchestration
function is compiled into the AICPU binary rather than loaded as a separate SO.
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

# Orchestration config for tensormap_and_ringbuffer (device-side orchestration)
# The orchestration function is linked into the AICPU binary
ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "example_orchestration.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

# Kernel configs
# These are the same kernels as host_build_graph/vector_example
KERNELS = [
    {"func_id": 0, "source": str(_KERNELS_ROOT / "aiv" / "kernel_add.cpp"),        "core_type": "aiv"},
    {"func_id": 1, "source": str(_KERNELS_ROOT / "aiv" / "kernel_add_scalar.cpp"), "core_type": "aiv"},
    {"func_id": 2, "source": str(_KERNELS_ROOT / "aiv" / "kernel_mul.cpp"),        "core_type": "aiv"},
]

# Runtime configuration for tensormap_and_ringbuffer
# This runtime requires 4 AICPU threads (3 schedulers + 1 orchestrator on thread 3)
RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 3,
}
