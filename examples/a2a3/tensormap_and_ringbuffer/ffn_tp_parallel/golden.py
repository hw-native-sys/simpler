"""Golden script for the ffn_tp_parallel distributed example."""

import torch

M = 64
K_SHARD = 64
N = 64

__outputs__ = ["y"]
RTOL = 1e-4
ATOL = 1e-4


def _make_rank_inputs(rank: int):
    x = (
        torch.arange(M * K_SHARD, dtype=torch.float32).reshape(M, K_SHARD)
        + torch.tensor(float(rank) * 0.25, dtype=torch.float32)
    ) / 32.0
    w = (
        torch.arange(K_SHARD * N, dtype=torch.float32).reshape(K_SHARD, N)
        + torch.tensor(float(rank + 1) * 0.5, dtype=torch.float32)
    ) / 48.0
    return x, w


def generate_distributed_inputs(rank: int, nranks: int, root: int, comm_ctx=None) -> list:
    del root
    del comm_ctx

    x_shard, w_shard = _make_rank_inputs(rank)
    zeros = torch.zeros(M * N, dtype=torch.float32)
    mailbox = torch.zeros(nranks * M * N, dtype=torch.float32)
    notify_counter = torch.zeros(1, dtype=torch.int32)
    return [
        ("x_shard", x_shard.flatten().tolist()),
        ("w_shard", w_shard.flatten().tolist()),
        ("partial_local", zeros.tolist()),
        ("partial_window", mailbox.tolist()),
        ("y", zeros.tolist()),
        ("notify_counter", notify_counter.tolist()),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    nranks = int(params.get("nranks", 2))
    expected = torch.zeros((M, N), dtype=torch.float32)
    for rank in range(nranks):
        x_shard, w_shard = _make_rank_inputs(rank)
        expected += torch.matmul(x_shard, w_shard)
    tensors["y"][:] = expected.flatten()
