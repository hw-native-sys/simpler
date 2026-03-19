"""
Golden script for async_completion_demo (dual-mode).

Computation:
    producer: out[i] = in[i] * 2.0   (with deferred completion)
    consumer: result[i] = out[i] + 1.0

    So: result[i] = in[i] * 2.0 + 1.0
    With in = 3.0: result = 7.0

Args layout: [ptr_in, ptr_out, ptr_result, ptr_event_handle_output,
              size_in, size_out, size_result, size_event_handle_output, SIZE]

event_handle_output: 16 bytes — used by the kernel and scheduler for async
  completion signaling. Not compared as test output.
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


def compute_golden(tensors: dict, params: dict) -> None:
    inp = torch.as_tensor(tensors["in"])
    tensors["result"][:] = inp * 2.0 + 1.0
    tensors["out"][:] = inp * 2.0
