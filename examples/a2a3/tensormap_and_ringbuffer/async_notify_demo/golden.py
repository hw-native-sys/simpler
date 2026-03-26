"""
Golden script for async_notify_demo.

Two hardware ranks each produce `out = in * 2` and TNOTIFY the peer.
The consumer is launch-gated on the local notification counter >= 1.
When the consumer runs, it reads notify_counter (must be 1) and computes
`result = out + notify_counter = in*2 + 1`.
"""

import torch

__outputs__ = ["result", "out"]

RTOL = 1e-5
ATOL = 1e-5


def generate_distributed_inputs(rank: int, nranks: int, root: int,
                                comm_ctx=None) -> list:
    del rank
    del nranks
    del root
    del comm_ctx

    size = 128 * 128
    inp = [float(i % 251) / 10.0 for i in range(size)]
    out = [0.0] * size
    result = [0.0] * size
    notify_counter = [0]

    return [
        ("in", inp),
        ("out", out),
        ("result", result),
        ("notify_counter", notify_counter),
    ]


def generate_inputs(params: dict) -> list:
    del params

    size = 128 * 128
    inp = torch.tensor([float(i % 251) / 10.0 for i in range(size)], dtype=torch.float32)
    out = torch.zeros(size, dtype=torch.float32)
    result = torch.zeros(size, dtype=torch.float32)
    notify_counter = torch.zeros(1, dtype=torch.int32)

    return [
        ("in", inp),
        ("out", out),
        ("result", result),
        ("notify_counter", notify_counter),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    del params

    if "in" in tensors:
        inp = torch.as_tensor(tensors["in"])
        tensors["out"][:] = inp * 2.0
        tensors["result"][:] = tensors["out"] + 1.0
        return

    out = tensors["out"]
    result = tensors["result"]
    for i in range(len(out)):
        value = float(i % 251) / 10.0
        out[i] = value * 2.0
        result[i] = out[i] + 1.0
