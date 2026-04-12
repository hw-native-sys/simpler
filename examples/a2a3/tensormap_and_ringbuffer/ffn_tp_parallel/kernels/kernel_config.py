"""Kernel config for the distributed ffn_tp_parallel example."""

import os
from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent
_platform = os.environ.get("PTO_PLATFORM", "a2a3sim")
_DIST_NRANKS = 2

if _platform != "a2a3":
    raise RuntimeError("ffn_tp_parallel currently requires PTO_PLATFORM=a2a3")

KERNELS = [
    {
        "func_id": 0,
        "name": "LOCAL_LINEAR",
        "source": str(_KERNELS_ROOT / "aic" / "kernel_local_linear.cpp"),
        "core_type": "aic",
    },
    {
        "func_id": 1,
        "name": "ALLREDUCE_SUM",
        "source": str(_KERNELS_ROOT / "aiv" / "kernel_allreduce_sum.cpp"),
        "core_type": "aiv",
    },
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "orch_thread_num": 1,
    "block_dim": 3,
    "rounds": 1,
}

DISTRIBUTED_CONFIG = {
    "nranks": _DIST_NRANKS,
    "root": 0,
    "win_sync_prefix": 256,
    "buffers": [
        {"name": "x_shard", "dtype": "float32", "count": 64 * 64, "placement": "device"},
        {"name": "w_shard", "dtype": "float32", "count": 64 * 64, "placement": "device"},
        {"name": "partial_local", "dtype": "float32", "count": 64 * 64, "placement": "device"},
        {"name": "partial_window", "dtype": "float32", "count": _DIST_NRANKS * 64 * 64, "placement": "window"},
        {"name": "y", "dtype": "float32", "count": 64 * 64, "placement": "device"},
        {"name": "notify_counter", "dtype": "int32", "count": 1, "placement": "window"},
    ],
    "inputs": ["x_shard", "w_shard", "partial_window", "y", "notify_counter"],
    "outputs": ["y"],
}

DISTRIBUTED_HOST_ORCH = {
    "source": str(_KERNELS_ROOT / "orchestration" / "host_orch.py"),
    "function_name": "distributed_orch",
}
