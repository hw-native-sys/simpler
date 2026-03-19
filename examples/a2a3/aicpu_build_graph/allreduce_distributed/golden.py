"""
Golden script for distributed AllReduce.

Each rank r contributes input[i] = i + r * 100 for i in [0, 256).
Every rank independently reduces (Sum) all inputs, so all ranks
produce the same output.

Expected output (same on every rank):
    output[i] = sum_{r=0}^{nranks-1} (i + r * 100)
              = nranks * i + 100 * nranks * (nranks - 1) / 2
"""

ALLREDUCE_COUNT = 256
NRANKS = 4

__outputs__ = ["output"]

RTOL = 1e-5
ATOL = 1e-5


def generate_distributed_inputs(rank: int, nranks: int, root: int,
                                 comm_ctx=None) -> list:
    """Each rank generates its own input; output is allocated on all ranks."""
    input_data = [float(i + rank * 100) for i in range(ALLREDUCE_COUNT)]
    output_data = [0.0] * ALLREDUCE_COUNT
    return [
        ("input", input_data),
        ("output", output_data),
        ("nranks", nranks),
        ("root", root),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    """Compute expected output — same for every rank."""
    nranks = params.get("nranks", NRANKS)
    output = tensors["output"]
    for i in range(ALLREDUCE_COUNT):
        output[i] = float(
            nranks * i + 100 * nranks * (nranks - 1) // 2)
