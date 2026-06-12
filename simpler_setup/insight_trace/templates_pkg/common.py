# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

from ..models import TraceConfig, TraceScalarArg, TraceTensorArg

_DTYPE_CPP = {
    "FLOAT16": "DataType::FLOAT16",
    "FLOAT32": "DataType::FLOAT32",
    "BFLOAT16": "DataType::BFLOAT16",
    "INT8": "DataType::INT8",
    "INT16": "DataType::INT16",
    "INT32": "DataType::INT32",
    "INT64": "DataType::INT64",
    "UINT8": "DataType::UINT8",
    "UINT16": "DataType::UINT16",
    "UINT32": "DataType::UINT32",
    "UINT64": "DataType::UINT64",
}

_DTYPE_SIZE = {
    "FLOAT16": 2,
    "FLOAT32": 4,
    "BFLOAT16": 2,
    "INT8": 1,
    "INT16": 2,
    "INT32": 4,
    "INT64": 8,
    "UINT8": 1,
    "UINT16": 2,
    "UINT32": 4,
    "UINT64": 8,
}


def _prologue(config: TraceConfig) -> str:
    arch = config.platform_arch
    return f"""#ifndef __CCE_AICORE__
#define __CCE_AICORE__ {arch.cce_aicore_number}
#endif
#include <cce_aicore_intrinsics.h>
#ifndef {arch.pto_arch_macro}
#define {arch.pto_arch_macro}
#endif
#ifndef EVENT_ID7
#define EVENT_ID7 {arch.prologue_event_id7}
#endif
#ifndef PIPE_FIX
#define PIPE_FIX {arch.prologue_pipe_fix}
#endif
"""


def _arg_to_json(arg: TraceTensorArg | TraceScalarArg) -> dict:
    if isinstance(arg, TraceTensorArg):
        return {"index": arg.index, "kind": "tensor", "dtype": arg.dtype, "shape": list(arg.shape), "name": arg.name}
    return {"index": arg.index, "kind": "scalar", "dtype": arg.dtype, "value": arg.value, "name": arg.name}


def _validate_args(args: tuple[TraceTensorArg | TraceScalarArg, ...]) -> None:
    seen = set()
    reserved = {48, 49}
    for arg in args:
        if arg.index in reserved:
            raise ValueError(f"Arg index {arg.index} is reserved for context pointers (slots 48, 49)")
        if arg.index < 0 or arg.index >= 50:
            raise ValueError(f"Arg index {arg.index} exceeds max slots (50)")
        if arg.index in seen:
            raise ValueError(f"Duplicate arg index {arg.index}")
        seen.add(arg.index)


def _require_kernel(config: TraceConfig):
    if config.kernel_spec is None:
        raise ValueError("simpler backend requires a kernel spec")
    return config.kernel_spec


def _require_spmd_meta(config: TraceConfig):
    if config.spmd_meta is None:
        raise ValueError("SPMD mix kernels require spmd_meta")
    return config.spmd_meta


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))
