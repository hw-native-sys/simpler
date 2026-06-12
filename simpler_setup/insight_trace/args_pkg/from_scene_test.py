# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

import ctypes
import struct

from simpler.task_interface import ArgDirection

from simpler_setup.scene_test import Scalar, Tensor

from ..models import SceneCaseContext, TraceArg, TraceScalarArg, TraceTensorArg

_TORCH_DTYPE_MAP = {
    "torch.float32": "FLOAT32",
    "torch.float16": "FLOAT16",
    "torch.bfloat16": "BFLOAT16",
    "torch.int32": "INT32",
    "torch.int64": "INT64",
    "torch.uint8": "UINT8",
    "torch.int8": "INT8",
    "torch.bool": "BOOL",
}

_CTYPES_DTYPE_MAP = {
    ctypes.c_bool: "BOOL",
    ctypes.c_int8: "INT8",
    ctypes.c_uint8: "UINT8",
    ctypes.c_int16: "INT16",
    ctypes.c_uint16: "UINT16",
    ctypes.c_int32: "INT32",
    ctypes.c_uint32: "UINT32",
    ctypes.c_int64: "INT64",
    ctypes.c_uint64: "UINT64",
    ctypes.c_float: "FLOAT32_BITS",
    ctypes.c_double: "FLOAT64_BITS",
}

_ROLE_MAP = {
    ArgDirection.IN: "input",
    ArgDirection.OUT: "output",
    ArgDirection.INOUT: "inout",
}


def resolve_scene_test_args(context: SceneCaseContext) -> tuple[TraceArg, ...]:
    builder = context.test_class().generate_args(context.case.get("params", {}))
    orch_signature = context.callable_spec.get("orchestration", {}).get("signature")
    if orch_signature is None:
        raise ValueError("No orchestration signature available for generic insight trace arg inference")

    result: list[TraceArg] = []
    tensor_index = 0
    arg_index = 0
    for spec in builder.specs:
        if isinstance(spec, Tensor):
            if tensor_index >= len(orch_signature):
                raise ValueError(
                    f"Tensor '{spec.name}' at index {tensor_index} has no matching orchestration signature entry"
                )
            result.append(
                TraceTensorArg(
                    index=arg_index,
                    name=spec.name,
                    dtype=_tensor_dtype_name(spec.value),
                    shape=tuple(int(dim) for dim in spec.value.shape),
                    role=_ROLE_MAP.get(orch_signature[tensor_index], "input"),
                )
            )
            tensor_index += 1
            arg_index += 1
            continue

        if isinstance(spec, Scalar):
            result.append(
                TraceScalarArg(
                    index=arg_index,
                    name=spec.name,
                    dtype=_scalar_dtype_name(spec.value),
                    value=_scalar_value(spec.value),
                    pack_mode=_scalar_pack_mode(spec.value),
                )
            )
            arg_index += 1
            continue

        raise ValueError(f"Unsupported TaskArgsBuilder spec type: {type(spec).__name__}")

    if tensor_index != len(orch_signature):
        raise ValueError(
            f"Orchestration signature length {len(orch_signature)} does not match tensor count {tensor_index}"
        )
    return tuple(result)


def _tensor_dtype_name(value) -> str:
    key = str(value.dtype)
    try:
        return _TORCH_DTYPE_MAP[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported tensor dtype for insight trace: {key}") from exc


def _scalar_dtype_name(value) -> str:
    if type(value) in _CTYPES_DTYPE_MAP:
        return _CTYPES_DTYPE_MAP[type(value)]
    if isinstance(value, bool):
        return "BOOL"
    if isinstance(value, int):
        return "INT64"
    if isinstance(value, float):
        return "FLOAT32_BITS"
    raise ValueError(f"Unsupported scalar type for insight trace: {type(value).__name__}")


def _scalar_value(value):
    if isinstance(value, ctypes._SimpleCData):
        raw = value.value
    else:
        raw = value
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, float):
        return _f32_bits(raw)
    return raw


def _scalar_pack_mode(value) -> str:
    dtype = _scalar_dtype_name(value)
    if dtype.endswith("_BITS"):
        return "bits"
    return "value"


def _f32_bits(value: float) -> int:
    return struct.unpack("I", struct.pack("f", value))[0]
