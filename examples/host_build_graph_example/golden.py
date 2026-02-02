"""
Golden script for host_build_graph example.

This script defines the input data generation and expected output computation
for the host_build_graph example (both a2a3 and a2a3sim platforms).

Computation:
    f = (a + b + 1) * (a + b + 2)
    where a=2.0, b=3.0, so f=42.0
"""

import numpy as np

# Output tensor names (alternatively, use 'out_' prefix convention)
__outputs__ = ["f"]

# Tensor order for orchestration function arguments
# This MUST match the order expected by BuildExampleGraph in example_orch.cpp
# Args layout: [ptr_a, ptr_b, ptr_f, size_a, size_b, size_f, SIZE]
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

    f = (a + b + 1) * (a + b + 2)
      = (2 + 3 + 1) * (2 + 3 + 2)
      = 6 * 7
      = 42

    Args:
        tensors: Dict containing all tensors (inputs and outputs)
        params: Parameter dict (unused in this example)
    """
    a = tensors["a"]
    b = tensors["b"]
    tensors["f"][:] = (a + b + 1) * (a + b + 2)
