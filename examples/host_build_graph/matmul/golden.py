"""
Golden script for matmul example.

This script defines the input data generation and expected output computation
for the matmul example (both a2a3 and a2a3sim platforms).

Computation:
    F = exp(sqrt(log(A)) @ W1 + sqrt(log(A)) @ W2)

    Task graph (diamond topology):
          t0: B = sqrt(log(A))      [AIV, half->half]
         /  \
       t1     t2                    [AIC, half*half->float]
       |       |
    C=B@W1  D=B@W2
         \  /
          t3: F = exp(C + D)        [AIV, float->float]

    where A = e^4 (128x128, float16), W1 = W2 = 1/256 (128x128, float16)
    Result: F = exp(2) ≈ 7.389
"""

import torch

# Output tensor names (alternatively, use 'out_' prefix convention)
__outputs__ = ["f"]

# Tensor order for orchestration function arguments
# This MUST match the order expected by build_matmul_graph in matmul_orch.cpp
# Args layout: [ptr_a, ptr_w1, ptr_w2, ptr_f, size_a, size_w1, size_w2, size_f, SIZE]
TENSOR_ORDER = ["a", "w1", "w2", "f"]

# Comparison tolerances (slightly relaxed for half precision intermediate)
RTOL = 1e-2
ATOL = 1e-2


def generate_inputs(params: dict) -> dict:
    """
    Generate input and output tensors.

    Creates:
    - a:  128x128 matrix, all e^4 ≈ 54.598 (float32, so log(a) = 4)
    - w1: 128x128 matrix, all 1/256 (float16, weight matrix for first matmul)
    - w2: 128x128 matrix, all 1/256 (float16, weight matrix for second matmul)
    - f:  128x128 matrix, zeros (float32, output)

    Returns:
        Dict of torch tensors with tensor names as keys
    """
    ROWS = 128
    COLS = 128
    SIZE = ROWS * COLS  # 16384 elements

    # Input value: e^4 so that log(A) = 4, sqrt(4) = 2
    input_value = torch.exp(torch.tensor(4.0)).item()

    # Weight matrices: 1/256 each, so that after two matmuls and addition:
    #   C = B @ W1: each element = 2 * (1/256) * 128 = 1
    #   D = B @ W2: each element = 2 * (1/256) * 128 = 1
    #   C + D = 2
    #   exp(2) ≈ 7.389
    weight_value = 1.0 / (2 * COLS)

    return {
        "a":  torch.full((SIZE,), input_value, dtype=torch.float16),   # half precision input
        "w1": torch.full((SIZE,), weight_value, dtype=torch.float16),  # half precision weight
        "w2": torch.full((SIZE,), weight_value, dtype=torch.float16),  # half precision weight
        "f":  torch.zeros(SIZE, dtype=torch.float32),               # float output
    }


def compute_golden(tensors: dict, params: dict) -> None:
    """
    Compute expected output in-place.

    F = exp(sqrt(log(A)) @ W1 + sqrt(log(A)) @ W2)

    Step by step:
    1. B = sqrt(log(A)) = sqrt(log(e^4)) = sqrt(4) = 2  (all elements, half precision)
    2. C = B @ W1 (matrix multiplication)
       - B is 128x128 with all 2s (half), W1 is 128x128 with all 1/256 (half)
       - C[i,j] = sum(B[i,k] * W1[k,j]) = 2 * (1/256) * 128 = 1 (float output)
    3. D = B @ W2 (matrix multiplication)
       - Same as C: D[i,j] = 1 (float output)
    4. F = exp(C + D) = exp(1 + 1) = exp(2) ≈ 7.389

    Args:
        tensors: Dict containing all tensors (inputs and outputs)
        params: Parameter dict (unused in this example)
    """
    ROWS = 128
    COLS = 128

    # Use float32 for computation accuracy in golden
    # Convert to torch tensors (handles both array types)
    a  = torch.as_tensor(tensors["a"]).reshape(ROWS, COLS).to(torch.float32)
    w1 = torch.as_tensor(tensors["w1"]).reshape(ROWS, COLS).to(torch.float32)
    w2 = torch.as_tensor(tensors["w2"]).reshape(ROWS, COLS).to(torch.float32)

    # F = exp(sqrt(log(A)) @ W1 + sqrt(log(A)) @ W2)
    b = torch.sqrt(torch.log(a))          # B = sqrt(log(A))
    c = torch.matmul(b, w1)            # C = B @ W1
    d = torch.matmul(b, w2)            # D = B @ W2
    f = torch.exp(c + d)               # F = exp(C + D)

    tensors["f"][:] = f.flatten().to(torch.float32)
