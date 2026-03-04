"""
Kernel configuration for multi_bgemm (multi-device BGEMM, Host Build Graph Runtime).

Same orchestration and kernels as bgemm; supports running on multiple devices in parallel.
Use --n-devices and --first-device to specify card count and starting device ID.
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "bgemm_orch.cpp"),
    "function_name": "build_bgemm_graph",
}

KERNELS = [
    {
        "func_id": 0,
        "source": str(_KERNELS_ROOT / "aic" / "kernel_gemm_tile.cpp"),
        "core_type": "aic",
    },
    {
        "func_id": 1,
        "source": str(_KERNELS_ROOT / "aiv" / "kernel_tile_add.cpp"),
        "core_type": "aiv",
    },
]

RUNTIME_CONFIG = {
    "runtime": "host_build_graph",
    "n_devices": 2,
    "first_device_id": 0,
}
