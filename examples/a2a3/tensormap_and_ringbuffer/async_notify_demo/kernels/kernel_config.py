"""
Async Notify Demo - Kernel and Orchestration Configuration

Two hardware cards use TNOTIFY(AtomicAdd) for inter-rank notification.
The consumer is launch-gated on the local notification counter >= 1.
"""

import os
from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent
_platform = os.environ.get("PTO_PLATFORM", "a2a3sim")

if _platform != "a2a3":
    raise RuntimeError("async_notify_demo currently requires PTO_PLATFORM=a2a3")

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "async_notify_orchestration.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {"func_id": 0, "source": str(_KERNELS_ROOT / "aiv" / "kernel_producer_notify.cpp"), "core_type": "aiv"},
    {"func_id": 1, "source": str(_KERNELS_ROOT / "aiv" / "kernel_consumer.cpp"), "core_type": "aiv"},
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "orch_thread_num": 1,
    "block_dim": 3,
    "rounds": 1,
}

DISTRIBUTED_CONFIG = {
    "nranks": 2,
    "root": 0,
    "win_sync_prefix": 256,
    "buffers": [
        {"name": "in", "dtype": "float32", "count": 128 * 128, "placement": "window"},
        {"name": "out", "dtype": "float32", "count": 128 * 128, "placement": "device"},
        {"name": "result", "dtype": "float32", "count": 128 * 128, "placement": "device"},
        {"name": "notify_counter", "dtype": "int32", "count": 1, "placement": "window"},
    ],
    "inputs": ["in", "notify_counter"],
    "outputs": ["out", "result"],
    "args": ["in", "out", "result", "notify_counter", "deviceCtx"],
}
