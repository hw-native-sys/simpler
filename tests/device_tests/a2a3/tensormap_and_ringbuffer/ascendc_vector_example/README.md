# AscendC Vector Example

Demonstrates integrating an AscendC operator into the PTO tensormap_and_ringbuffer runtime.

## Overview

This device test runs two tasks on a single blockdim (2 AIV cores):

1. **t0**: `z = add_custom(x, y)` -- AscendC AddCustom operator (func_id=0)
2. **t1**: `w = mul(z, z)` -- PTO-native kernel_mul (func_id=1)

The AscendC kernel source is compiled directly by simpler's `AscendCCompiler` using the
CANN SDK's ccec compiler with `--cce-aicore-lang` flags, then linked with a generated PTO
wrapper to match PTO's `kernel_entry(int64_t* args)` convention.

## Prerequisites

- Ascend CANN SDK with ccec compiler (hardware platform only, no sim support)

## How to Run

```bash
python examples/scripts/run_example.py \
  -k tests/device_tests/a2a3/tensormap_and_ringbuffer/ascendc_vector_example/kernels \
  -g tests/device_tests/a2a3/tensormap_and_ringbuffer/ascendc_vector_example/golden.py \
  -p a2a3 -d <device_id>
```

Or via CI (device tests only):
```bash
./ci.sh -p a2a3 -d <device_range>
```

## Architecture

```
    AscendC source (.cpp)                PTO wrapper (.cpp)
    uses kernel_operator.h               uses tensor.h
            │                                    │
            ▼                                    ▼
    ccec --cce-aicore-lang              ccec -x cce (PTO-ISA)
    (AscendC toolchain flags)           (PTO toolchain flags)
            │                                    │
            ▼                                    ▼
        kernel.o                            wrapper.o
            │                                    │
            └──────────┬─────────────────────────┘
                       ▼
              ld.lld -r wrapper.o kernel.o
                       │
                       ▼
                  combined.o
                       │
                       ▼
             extract_text_section
                       │
                       ▼
                 kernel binary
                       │
    ┌──────────────────▼───────────────────────────┐
    │  PTO Runtime                                  │
    │  kernel_binaries = [(0, ascendc_bin),          │
    │                      (1, mul_bin)]             │
    │                                               │
    │  Orchestration:                               │
    │    pto2_rt_submit_aiv_task(rt, 0, params)     │
    │    pto2_rt_submit_aiv_task(rt, 1, params)     │
    └───────────────────────────────────────────────┘
```

Key: wrapper.o is listed first in the link command so `kernel_entry` lands
at `.text` offset 0 (PTO dispatch jumps to offset 0).

## Configuration

- `block_dim = 1`: Single blockdim (1 AIC + 2 AIV). Only AIV used.
- `aicpu_thread_num = 3`: 2 schedulers + 1 orchestrator.
- `compiler = "ascendc"`: Tells CodeRunner to use AscendCCompiler for this kernel.

## kernel_config.py Schema for AscendC Kernels

```python
{
    "func_id": 0,
    "source": "path/to/ascendc_kernel.cpp",
    "core_type": "aiv",
    "compiler": "ascendc",
    "ascendc_symbol": "add_custom",    # extern "C" symbol name
    "tensor_args": [                   # ordered tensor descriptors
        {"name": "x", "direction": "input"},
        {"name": "y", "direction": "input"},
        {"name": "z", "direction": "output"},
    ],
    "has_workspace": False,            # workspace pointer (optional)
}
```
