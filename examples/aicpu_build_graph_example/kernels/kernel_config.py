"""
Kernel and Orchestration Configuration

This example uses the aicpu_build_graph runtime:
- Host orchestration prepares tensors and marshals `runtime->orch_args[]`.
- AICPU builds the task graph on device by dlopen-ing a small `.so` plugin that
  exports `build_graph_aicpu(Runtime*)`.
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

RUNTIME_CONFIG = {
    "runtime": "aicpu_build_graph",
    # Default split: 1 AICPU thread builds tasks while 3 AICPU threads schedule/execute.
    "aicpu_thread_num": 4,
    "block_dim": 24,
}

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "example_orch.cpp"),
    "function_name": "prepare_example_graph",
}

# AICPU-side orchestration plugin (loaded by AICPU via dlopen+dlsym).
AICPU_ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "aicpu" / "build_graph_aicpu.cpp"),
    "function_name": "build_graph_aicpu",
}

#
# Runtime behavior knobs.
#
# `RUNTIME_ENV` is applied both during runtime compilation and during runtime
# initialization (host `dlopen()` + orchestrator call).
#
# For `aicpu_build_graph`, the runtime reads:
#   PTO_AICPU_BUILD_GRAPH_BUILD_MODE = "1" (concurrent build||schedule, default)
#   PTO_AICPU_BUILD_GRAPH_BUILD_MODE = "0" (sequential build->schedule)
RUNTIME_ENV = {
    "PTO_AICPU_BUILD_GRAPH_BUILD_MODE": "1",
}

KERNELS = [
    {"func_id": 0, "source": str(_KERNELS_ROOT / "aiv" / "kernel_add.cpp"), "core_type": "aiv"},
    {"func_id": 1, "source": str(_KERNELS_ROOT / "aiv" / "kernel_add_scalar.cpp"), "core_type": "aiv"},
    {"func_id": 2, "source": str(_KERNELS_ROOT / "aiv" / "kernel_mul.cpp"), "core_type": "aiv"},
]
