"""
MOE Dispatch V2 — 8-rank multi-expert example following pypto V2 pattern.

Phases:
  0. Prepare      — route tokens by expertId, pack per-rank staging buffers,
                     write local shmem_data slots, compute per-expert counts
  1. Send         — TPUT_ASYNC data + counts to all peers, TNOTIFY each peer
  2. RecvAssemble — cumsum received counts, assemble expandX, compute expertTokenNums

Window memory layout per rank (shared across ranks via RDMA):
  shmem_data[NUM_EXPERT_SLOTS][NUM_TOKENS][HIDDEN_DIM]  — token data
    Slot index: expert_local_offset * NUM_RANKS + src_rank
  recv_counts[NUM_RANKS][COUNT_PAD]  — per-source-rank counts

Staging layout:
  send_staging[NUM_RANKS][EXPERTS_PER_RANK][NUM_TOKENS][HIDDEN_DIM]
    Indexed by (target_rank, expert_offset)

Requires PTO_PLATFORM=a2a3 (hardware with SDMA support).
"""

import os
from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent
_platform = os.environ.get("PTO_PLATFORM", "a2a3sim")

if _platform != "a2a3":
    raise RuntimeError("moe_dispatch requires PTO_PLATFORM=a2a3")

NUM_TOKENS = 16
HIDDEN_DIM = 128
NUM_RANKS = 8
EXPERTS_PER_RANK = 2
TOTAL_EXPERTS = NUM_RANKS * EXPERTS_PER_RANK
NUM_EXPERT_SLOTS = EXPERTS_PER_RANK * NUM_RANKS
EXPAND_X_ROWS = NUM_TOKENS * NUM_RANKS
COUNT_PAD = 32

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "moe_dispatch_orchestration.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {"func_id": 0, "source": str(_KERNELS_ROOT / "aiv" / "kernel_moe_prepare.cpp"),       "core_type": "aiv"},
    {"func_id": 1, "source": str(_KERNELS_ROOT / "aiv" / "kernel_moe_send_data.cpp"),     "core_type": "aiv"},
    {"func_id": 2, "source": str(_KERNELS_ROOT / "aiv" / "kernel_moe_recv_assemble.cpp"), "core_type": "aiv"},
    {"func_id": 3, "source": str(_KERNELS_ROOT / "aiv" / "kernel_notify_wait.cpp"),       "core_type": "aiv"},
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "orch_thread_num": 1,
    "block_dim": 3,
    "rounds": 1,
}

RUNTIME_ENV = {
    "PTO2_ENABLE_SDMA": "1",
}

STAGING_ELEMS = NUM_RANKS * EXPERTS_PER_RANK * NUM_TOKENS * HIDDEN_DIM

DISTRIBUTED_CONFIG = {
    "nranks": NUM_RANKS,
    "root": 0,
    "win_sync_prefix": 256,
    "buffers": [
        {"name": "tokens",       "dtype": "float32", "count": NUM_TOKENS * HIDDEN_DIM,              "placement": "window"},
        {"name": "expert_ids",   "dtype": "int32",   "count": NUM_TOKENS,                           "placement": "window"},
        {"name": "shmem_data",   "dtype": "float32", "count": NUM_EXPERT_SLOTS * NUM_TOKENS * HIDDEN_DIM, "placement": "window"},
        {"name": "send_staging", "dtype": "float32", "count": STAGING_ELEMS,                        "placement": "window"},
        {"name": "local_counts", "dtype": "int32",   "count": COUNT_PAD,                            "placement": "window"},
        {"name": "send_counts",  "dtype": "int32",   "count": NUM_RANKS * COUNT_PAD,                "placement": "window"},
        {"name": "recv_counts",  "dtype": "int32",   "count": NUM_RANKS * COUNT_PAD,                "placement": "window"},
        {"name": "notify_counter", "dtype": "int32", "count": 1,                                    "placement": "window"},
        {"name": "expand_x",    "dtype": "float32",  "count": EXPAND_X_ROWS * HIDDEN_DIM,           "placement": "device"},
        {"name": "expert_token_nums", "dtype": "int32", "count": EXPERTS_PER_RANK,                  "placement": "device"},
    ],
    "inputs": ["tokens", "expert_ids", "notify_counter"],
    "outputs": ["expert_token_nums", "local_counts"],
    "args": [
        "tokens", "expert_ids", "shmem_data", "send_staging",
        "local_counts", "send_counts", "recv_counts", "notify_counter",
        "expand_x", "expert_token_nums", "deviceCtx",
    ],
}
