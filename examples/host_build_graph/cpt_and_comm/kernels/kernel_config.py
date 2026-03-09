"""
Kernel configuration for cpt_and_comm (compute then communicate).

Flow: GEMM -> WindowMemCopyIn -> CommBarrier -> TGATHER -> WindowMemCopyOut (root only).
CommBarrier uses TNOTIFY/TWAIT for device-side cross-rank synchronization.
Requires HCCL (multi-card), PTO_COMM_ISA_ROOT pointing to pto-comm-isa for comm headers.
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "cpt_and_comm_orch.cpp"),
    "function_name": "build_cpt_and_comm_graph",
}

KERNELS = [
    {"func_id": 0, "name": "GEMM", "source": str(_KERNELS_ROOT / "aic" / "kernel_gemm_tile.cpp"), "core_type": "aic"},
    {"func_id": 1, "name": "WindowMemCopyIn", "source": str(_KERNELS_ROOT / "aiv" / "window_memcopy_in.cpp"), "core_type": "aiv"},
    {"func_id": 2, "name": "Gather", "source": str(_KERNELS_ROOT / "aiv" / "gather_kernel.cpp"), "core_type": "aiv"},
    {"func_id": 3, "name": "WindowMemCopyOut", "source": str(_KERNELS_ROOT / "aiv" / "window_memcopy_out.cpp"), "core_type": "aiv"},
    {"func_id": 4, "name": "CommBarrier", "source": str(_KERNELS_ROOT / "aiv" / "comm_barrier_kernel.cpp"), "core_type": "aiv"},
]

RUNTIME_CONFIG = {
    "runtime": "host_build_graph",
    "n_devices": 2,
    "first_device_id": 0,
    "requires_comm": True,
}
