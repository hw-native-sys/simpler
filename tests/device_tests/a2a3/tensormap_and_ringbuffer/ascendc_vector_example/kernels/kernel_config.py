"""
AscendC Vector Example — kernel_config.py

Demonstrates integrating a pre-compiled AscendC operator (add_custom) into the
PTO tensormap_and_ringbuffer runtime via wrapper generation + link.

The .o is compiled with AscendC toolchain (ccec --cce-aicore-lang) but adapted
for PTO dispatch:
  - No __global__ attribute (causes hang under PTO subroutine dispatch)
  - No GetBlockNum()/GetBlockIdx() partitioning (PTO dispatches to single cores)
  - Static tiling (constexpr values, no runtime tiling pointer)

Simpler generates a PTO wrapper (kernel_entry), compiles it with PTO flags
(-x cce), and links it with the kernel .o so kernel_entry sits at .text offset 0.

Computation:
  z = x + y          (AscendC add_custom, func_id=0)
  w = z * z          (PTO kernel_mul,     func_id=1)

The orchestration submits two AIV tasks in sequence.
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "ascendc_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {
        "func_id": 0,
        # Pre-compiled AscendC kernel .o with static tiling degeneration
        # (tiling values baked in at compile time, no runtime tiling needed)
        "source": str(_KERNELS_ROOT / "ascendc" / "add_custom.o"),
        "core_type": "aiv",
        "compiler": "ascendc",
        "ascendc_symbol": "add_custom",
        "tensor_args": [
            {"name": "x", "direction": "input"},
            {"name": "y", "direction": "input"},
            {"name": "z", "direction": "output"},
        ],
        "has_workspace": True,
    },
    {
        "func_id": 1,
        "source": str(_KERNELS_ROOT / "aiv" / "kernel_mul.cpp"),
        "core_type": "aiv",
    },
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 3,
    "block_dim": 2,
}
