"""
Distributed TREDUCE kernel configuration — tensormap_and_ringbuffer runtime.

Device-side orchestration via PTO2Runtime API. The orchestration function
wraps each arg as a PTOParam (tensor or scalar) and submits a single AIV task.
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "treduce_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {
        "func_id": 0,
        "source": str(_KERNELS_ROOT / "aiv" / "treduce_kernel.cpp"),
        "core_type": "aiv",
    },
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 3,
    "orch_thread_num": 1,
    "rounds": 1,
}

# Distributed layout contract consumed by DistributedCodeRunner/worker:
# - win_sync_prefix reserves a small header at the front of each rank's RDMA
#   window before any placement="window" buffers are laid out.
# - buffers declares runtime allocation metadata:
#   * count is the element count, not byte size.
#   * placement="window": buffer lives in the shared RDMA window and may be
#     accessed by remote ranks.
#   * placement="device": buffer uses regular device_malloc and is local-only.
# - inputs/outputs control which buffers are loaded from .bin files and which
#   are copied back after execution.
# - args defines the orchestration/kernel uint64_t* args order.
DISTRIBUTED_CONFIG = {
    "nranks": 4,
    "root": 0,
    "win_sync_prefix": 256,
    "buffers": [
        # Root rank reads every rank's input through CommRemotePtr(...), so the
        # input buffer must be placed in the shared RDMA window.
        {"name": "input",  "dtype": "float32", "count": 256, "placement": "window"},
        # The output is produced and consumed locally on the root rank only.
        {"name": "output", "dtype": "float32", "count": 256, "placement": "device"},
    ],
    "inputs": ["input"],
    "outputs": ["output"],
    "args": ["input", "output", "nranks", "root", "deviceCtx"],
}
