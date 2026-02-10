"""
Golden script for tensormap_and_ringbuffer example.

This script defines the input data generation and expected output computation
for the tensormap_and_ringbuffer example (both a2a3 and a2a3sim platforms).

Computation:
    f = (a + b + 1) * (a + b + 2) + (a + b)
    where a=2.0, b=3.0, so f=47.0

This is the same computation as host_build_graph/vector_example, but uses
device-side orchestration (tensormap_and_ringbuffer runtime).
"""

import numpy as np

# Output tensor names
__outputs__ = ["f"]

# Tensor order for orchestration function arguments
# tensormap_and_ringbuffer args layout: [dev_a, dev_b, dev_f, size_a, size_b, size_f, SIZE]
# Note: intermediate tensors (c, d, e, g) are allocated on-device by the runtime heap
TENSOR_ORDER = ["a", "b", "f"]

# Comparison tolerances
RTOL = 1e-5
ATOL = 1e-5


def generate_inputs(params: dict) -> dict:
    """
    Generate input and output tensors.

    Creates:
    - a: 16384 elements, all 2.0
    - b: 16384 elements, all 3.0
    - f: 16384 elements, zeros (output)

    Returns:
        Dict of numpy arrays with tensor names as keys
    """
    ROWS = 128
    COLS = 128
    SIZE = ROWS * COLS  # 16384 elements

    return {
        "a": np.full(SIZE, 2.0, dtype=np.float32),
        "b": np.full(SIZE, 3.0, dtype=np.float32),
        "f": np.zeros(SIZE, dtype=np.float32),
    }


def compute_golden(tensors: dict, params: dict) -> None:
    """
    Compute expected output in-place.

    f = (a + b + 1) * (a + b + 2) + (a + b)
      = (2 + 3 + 1) * (2 + 3 + 2) + (2 + 3)
      = 6 * 7 + 5
      = 47

    Args:
        tensors: Dict containing all tensors (inputs and outputs)
        params: Parameter dict (unused in this example)
    """
    a = tensors["a"]
    b = tensors["b"]
    tensors["f"][:] = (a + b + 1) * (a + b + 2) + (a + b)
