"""
Kernel and Orchestration Configuration (Simulation)

Defines the kernels and orchestration function used by the a2a3sim example.
Simulation kernels are plain C++ (no PTO ISA), compiled with g++.
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

# Orchestration config
ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "example_orch.cpp"),
    "function_name": "build_example_graph",
}

# Kernel configs (simulation kernels, compiled with g++)
KERNELS = [
    {"func_id": 0, "source": str(_KERNELS_ROOT / "aiv" / "kernel_add.cpp"),        "core_type": "aiv"},
    {"func_id": 1, "source": str(_KERNELS_ROOT / "aiv" / "kernel_add_scalar.cpp"), "core_type": "aiv"},
    {"func_id": 2, "source": str(_KERNELS_ROOT / "aiv" / "kernel_mul.cpp"),        "core_type": "aiv"},
]
