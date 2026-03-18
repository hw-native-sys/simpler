"""
Golden script for distributed TREDUCE.

Each rank r contributes input[i] = i + r * 100 for i in [0, 256).
Root rank reduces (Sum) all inputs.

Expected output on root:
    output[i] = sum_{r=0}^{nranks-1} (i + r * 100)
              = nranks * i + 100 * nranks * (nranks - 1) / 2
"""

TREDUCE_COUNT = 256
NRANKS = 4

__outputs__ = ["output"]

RTOL = 1e-5
ATOL = 1e-5


def generate_distributed_inputs(rank: int, nranks: int, root: int,
                                 comm_ctx=None) -> list:
    """Each rank generates its own input; output is allocated on all ranks."""
    input_data = [float(i + rank * 100) for i in range(TREDUCE_COUNT)]
    output_data = [0.0] * TREDUCE_COUNT
    return [
        ("input", input_data),
        ("output", output_data),
        ("nranks", nranks),
        ("root", root),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    """Compute expected output for the root rank."""
    nranks = params.get("nranks", NRANKS)
    output = tensors["output"]
    for i in range(TREDUCE_COUNT):
        output[i] = float(
            nranks * i + 100 * nranks * (nranks - 1) // 2)
