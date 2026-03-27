"""
Golden script for MOE Dispatch V2 — 8-rank multi-expert dispatch.

Routing:  expert_ids[i] = i % TOTAL_EXPERTS  (deterministic round-robin)
Tokens:   tokens[i][j] = ((rank * NUM_TOKENS + i) * HIDDEN_DIM + j) / 1e5

Per rank, the prepare kernel partitions tokens:
  - Local expert tokens -> written directly to shmem_data[slot]
  - Remote expert tokens -> packed into send_staging[target_rank][expert_offset]

SendData TPUT_ASYNCs each (peer, expert) staging buffer to the peer's shmem_data.
SendCount TPUT_ASYNCs per-peer counts to the peer's recv_counts + TNOTIFY.
RecvAssemble reads shmem_data + counts after 7 notifications, assembles expandX.

Verified outputs (symmetric across all ranks):
  - expert_token_nums: each local expert receives 1 token from each of 8 ranks = [8, 8]
  - local_counts: each rank has 2 local tokens, 1 per local expert = [1, 1, 0, ...]
"""

NUM_TOKENS = 16
HIDDEN_DIM = 128
NUM_RANKS = 8
EXPERTS_PER_RANK = 2
TOTAL_EXPERTS = NUM_RANKS * EXPERTS_PER_RANK
NUM_EXPERT_SLOTS = EXPERTS_PER_RANK * NUM_RANKS
EXPAND_X_ROWS = NUM_TOKENS * NUM_RANKS
COUNT_PAD = 32

__outputs__ = ["expert_token_nums", "local_counts"]

RTOL = 1e-5
ATOL = 1e-5


def _make_tokens(rank):
    tokens = [0.0] * (NUM_TOKENS * HIDDEN_DIM)
    for i in range(NUM_TOKENS):
        for j in range(HIDDEN_DIM):
            tokens[i * HIDDEN_DIM + j] = float(
                (rank * NUM_TOKENS + i) * HIDDEN_DIM + j) / 100000.0
    return tokens


def _route_expert_ids():
    return [i % TOTAL_EXPERTS for i in range(NUM_TOKENS)]


def generate_distributed_inputs(rank: int, nranks: int, root: int,
                                comm_ctx=None) -> list:
    del root, comm_ctx

    tokens = _make_tokens(rank)
    expert_ids = _route_expert_ids()

    return [
        ("tokens", tokens),
        ("expert_ids", expert_ids),
        ("shmem_data", [0.0] * (NUM_EXPERT_SLOTS * NUM_TOKENS * HIDDEN_DIM)),
        ("send_staging", [0.0] * (NUM_RANKS * EXPERTS_PER_RANK * NUM_TOKENS * HIDDEN_DIM)),
        ("local_counts", [0] * COUNT_PAD),
        ("send_counts", [0] * (NUM_RANKS * COUNT_PAD)),
        ("recv_counts", [0] * (NUM_RANKS * COUNT_PAD)),
        ("notify_counter", [0]),
        ("expand_x", [0.0] * (EXPAND_X_ROWS * HIDDEN_DIM)),
        ("expert_token_nums", [0] * EXPERTS_PER_RANK),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    nranks = params.get("nranks", NUM_RANKS)
    my_rank = params.get("root", 0)

    expert_ids = _route_expert_ids()
    local_expert_start = my_rank * EXPERTS_PER_RANK

    local_counts = tensors["local_counts"]
    for k in range(EXPERTS_PER_RANK):
        local_counts[k] = 0

    for i in range(NUM_TOKENS):
        eid = expert_ids[i]
        target_rank = eid // EXPERTS_PER_RANK
        expert_offset = eid % EXPERTS_PER_RANK

        if target_rank == my_rank:
            local_counts[expert_offset] += 1

    expert_token_nums = tensors["expert_token_nums"]

    for exp_off in range(EXPERTS_PER_RANK):
        expert_total = 0
        for src_rank in range(nranks):
            src_expert_id = local_expert_start + exp_off
            src_expert_ids = _route_expert_ids()

            for i in range(NUM_TOKENS):
                if src_expert_ids[i] == src_expert_id:
                    expert_total += 1

        expert_token_nums[exp_off] = expert_total
