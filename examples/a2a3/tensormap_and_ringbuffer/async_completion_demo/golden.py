"""
Golden script for async_completion_demo.

Single-card / sim path keeps the original producer-consumer pipeline:
    producer: out[i] = in[i] * 2.0
    consumer: result[i] = out[i] + 1.0

Hardware 2-card path validates `out` and `result`:
    each rank TGET_ASYNCs the peer rank's `in` into local `out`, then the
    normal consumer computes `result = out + 1`.
"""

import ctypes
import torch

__outputs__ = ["result", "out"]

RTOL = 1e-5
ATOL = 1e-5


def generate_inputs(params: dict) -> list:
    SIZE = 128 * 128

    inp = torch.full((SIZE,), 3.0, dtype=torch.float32)
    out = torch.zeros(SIZE, dtype=torch.float32)
    result = torch.zeros(SIZE, dtype=torch.float32)
    event_handle_output = torch.zeros(4, dtype=torch.int32)

    return [
        ("in", inp),
        ("out", out),
        ("result", result),
        ("event_handle_output", event_handle_output),
        ("size_in", ctypes.c_int64(inp.nbytes)),
        ("size_out", ctypes.c_int64(out.nbytes)),
        ("size_result", ctypes.c_int64(result.nbytes)),
        ("size_event_handle_output", ctypes.c_int64(event_handle_output.nbytes)),
        ("SIZE", ctypes.c_int64(SIZE)),
    ]


def generate_distributed_inputs(rank: int, nranks: int, root: int,
                                comm_ctx=None) -> list:
    del comm_ctx
    del nranks
    del root

    size = 128 * 128
    inp = [float(i % 251) / 10.0 for i in range(size)]
    out = [0.0] * size
    result = [0.0] * size

    return [
        ("in", inp),
        ("out", out),
        ("result", result),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    if "in" in tensors:
        inp = torch.as_tensor(tensors["in"])
        tensors["result"][:] = inp * 2.0 + 1.0
        tensors["out"][:] = inp * 2.0
        return

    out = tensors["out"]
    result = tensors["result"]
    for i in range(len(out)):
        value = float(i % 251) / 10.0
        out[i] = value
        result[i] = value + 1.0
