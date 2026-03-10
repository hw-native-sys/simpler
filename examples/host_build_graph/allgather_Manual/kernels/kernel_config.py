"""
AllGather (Manual): direct RDMA reads for performance comparison.

Flow: WindowMemCopyIn -> CommBarrier -> AllGatherManual (HcclRemotePtr+TLOAD/TSTORE)
-> WindowMemCopyOut -> CommBarrier(post).
No TGATHER, all ranks run in parallel. Requires HCCL, PTO_COMM_ISA_ROOT.
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "allgather_orch.cpp"),
    "function_name": "build_allgather_graph",
}

KERNELS = [
    {"func_id": 0, "name": "WindowMemCopyIn", "source": str(_KERNELS_ROOT / "aiv" / "window_memcopy_in.cpp"), "core_type": "aiv"},
    {"func_id": 1, "name": "AllGatherManual", "source": str(_KERNELS_ROOT / "aiv" / "allgather_manual_kernel.cpp"), "core_type": "aiv"},
    {"func_id": 2, "name": "WindowMemCopyOut", "source": str(_KERNELS_ROOT / "aiv" / "window_memcopy_out.cpp"), "core_type": "aiv"},
    {"func_id": 3, "name": "CommBarrierAll", "source": str(_KERNELS_ROOT / "aiv" / "comm_barrier_all_kernel.cpp"), "core_type": "aiv"},
]

RUNTIME_CONFIG = {
    "runtime": "host_build_graph",
    "n_devices": 2,
    "first_device_id": 0,
    "requires_comm": True,
}
