"""
Distributed AllReduce kernel configuration.

Multi-card collective allreduce (Sum) across N ranks using PTO comm
instructions. Every rank reads all inputs via RDMA and computes the sum
locally. Communication addresses are set up by the comm_* platform API.

DISTRIBUTED_CONFIG is the "multi-card graph" — it describes buffer layout,
args order, and artifact names. DistributedCodeRunner translates this into
distributed_worker.py CLI arguments.
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "allreduce_orch.cpp"),
    "function_name": "build_allreduce_graph",
}

KERNELS = [
    {
        "func_id": 0,
        "source": str(_KERNELS_ROOT / "aiv" / "allreduce_kernel.cpp"),
        "core_type": "aiv",
    },
]

RUNTIME_CONFIG = {
    "runtime": "host_build_graph",
    "aicpu_thread_num": 1,
    "block_dim": 1,
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
        # Every rank reads all ranks' inputs via CommRemotePtr, so the
        # input buffer must be placed in the shared RDMA window.
        {"name": "input",  "dtype": "float32", "count": 256, "placement": "window"},
        # Each rank writes the reduced sum to its own local output.
        {"name": "output", "dtype": "float32", "count": 256, "placement": "device"},
    ],
    "inputs": ["input"],
    "outputs": ["output"],
    "args": ["input", "output", "nranks", "root", "deviceCtx"],
}
