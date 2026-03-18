"""
Distributed TREDUCE kernel configuration — aicpu_build_graph runtime.

The AICPU orchestration plugin reads args from runtime->orch_args[],
builds the task graph via the aicpu_build_api, and publishes tasks for
the AICPU scheduler threads.
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "treduce_orch.cpp"),
    "function_name": "build_treduce_graph",
}

KERNELS = [
    {
        "func_id": 0,
        "source": str(_KERNELS_ROOT / "aiv" / "treduce_kernel.cpp"),
        "core_type": "aiv",
    },
]

RUNTIME_CONFIG = {
    "runtime": "aicpu_build_graph",
    "aicpu_thread_num": 4,
    "block_dim": 4,
}

RUNTIME_ENV = {
    "PTO_AICPU_BUILD_GRAPH_BUILD_MODE": "1",
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
