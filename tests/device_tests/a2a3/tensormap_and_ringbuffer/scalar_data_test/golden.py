"""
Golden script for scalar data dependency test.

Tests GetTensorData, SetTensorData, and add_inout with initial value.

Computation:
  c = a + b (kernel, internal tensor)
  check[0] = GetTensorData(c, {0})     = a[0]+b[0] = 2.0+0.0 = 2.0
  check[1] = GetTensorData(c, {100})   = a[100]+b[100] = 2.0+100.0 = 102.0
  scalar initialized to 77.0 via add_output(scalar, float_to_u64(77.0f))
  check[2] = GetTensorData(scalar, {0}) = 77.0
  second noop with add_inout(scalar), value preserved
  check[3] = GetTensorData(scalar, {0}) = 77.0
  check[4] = orchestration arithmetic: 2.0 + 77.0 = 79.0
  SetTensorData(scalar, {0}, 42.0), then GetTensorData round-trip
  check[5] = GetTensorData(scalar, {0}) = 42.0
  Orch SetTensorData(d, {0}, 10.0) → kernel_add(d, a) → e[0] = 12.0
  check[6] = GetTensorData(e, {0}) = 12.0
  WAW+WAR: kernel reads c as INPUT, then SetTensorData(c, 88.0) auto-waits
  check[7] = GetTensorData(c, {0}) = 88.0
  External WAR: noop(ext_b as INOUT) → SetTensorData(ext_b, 55.0) auto-waits
  check[8] = GetTensorData(ext_b, {0}) = 55.0 (ext_b[0] restored to 0.0 after)
  result = a + b (kernel, external output)

Args layout: [a, b, result, check]
"""

import torch

__outputs__ = ["result", "check"]

RTOL = 1e-5
ATOL = 1e-5


def generate_inputs(params: dict) -> list:
    SIZE = 128 * 128  # 16384 -- matches kernel_add 128x128 tile

    a = torch.full((SIZE,), 2.0, dtype=torch.float32)
    b = torch.arange(SIZE, dtype=torch.float32)
    result = torch.zeros(SIZE, dtype=torch.float32)
    check = torch.zeros(10, dtype=torch.float32)

    return [
        ("a", a),
        ("b", b),
        ("result", result),
        ("check", check),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    a = torch.as_tensor(tensors["a"])
    b = torch.as_tensor(tensors["b"])

    # result = a + b (computed by kernel_add)
    tensors["result"][:] = a + b

    # check values written by orchestration via SetTensorData
    check = torch.as_tensor(tensors["check"])
    check[0] = 2.0  # GetTensorData(c, {0}): c = a + b, c[0] = 2.0+0.0
    check[1] = 102.0  # GetTensorData(c, {100}): c[100] = 2.0+100.0
    check[2] = 77.0  # add_inout initial value (first use, OUTPUT path)
    check[3] = 77.0  # add_inout second use (INOUT path, value preserved)
    check[4] = 79.0  # orchestration arithmetic: 2.0 + 77.0
    check[5] = 42.0  # Orch set→get round-trip: SetTensorData then GetTensorData
    check[6] = (
        12.0  # Orch→AICore RAW: SetTensorData(d,10.0) + kernel_add(d,a) → 10.0+2.0
    )
    check[7] = (
        88.0  # WAW+WAR: kernel reads c, SetTensorData(c,88.0) auto-waits for consumer
    )
    check[8] = (
        55.0  # External WAR: noop(ext_b INOUT) → SetTensorData(ext_b,55.0) auto-waits
    )
    # check[9] remains 0.0 (sentinel)
