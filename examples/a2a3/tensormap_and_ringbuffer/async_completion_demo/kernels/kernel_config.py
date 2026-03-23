"""
Async Completion Demo - Kernel and Orchestration Configuration

Dual-mode demonstration:
  Sim mode (a2a3sim):  func_id=0 (simulated producer, complete_in_future=1)
  HW mode  (a2a3):     func_id=2 (TPUT_ASYNC producer, complete_in_future=2)

Both modes share func_id=1 (consumer, run-to-completion).
Orchestration dynamically selects mode based on SDMA workspace availability.
"""

import os
from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "async_demo_orchestration.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

_platform = os.environ.get("PTO_PLATFORM", "a2a3sim")

KERNELS = [
    {"func_id": 0, "source": str(_KERNELS_ROOT / "aiv" / "kernel_producer.cpp"), "core_type": "aiv"},
    {"func_id": 1, "source": str(_KERNELS_ROOT / "aiv" / "kernel_consumer.cpp"), "core_type": "aiv"},
]

if _platform == "a2a3":
    KERNELS.append(
        {"func_id": 2, "source": str(_KERNELS_ROOT / "aiv" / "kernel_producer_async.cpp"), "core_type": "aiv"},
    )

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 3,
    "rounds": 1,
}
