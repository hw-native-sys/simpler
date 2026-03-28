# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public Python API for task_interface nanobind bindings.

Re-exports the canonical C++ types (DataType, TaskArgKind, TaskArg, TaskArgArray)
and adds torch-aware convenience helpers for building TaskArg instances from
torch tensors.

Usage:
    from task_interface import DataType, TaskArg, TaskArgArray, make_tensor_arg
"""

from _task_interface import (  # pyright: ignore[reportMissingImports]
    TASK_ARG_MAX_DIMS,
    DataType,
    TaskArg,
    TaskArgArray,
    TaskArgKind,
    get_dtype_name,
    get_element_size,
)

__all__ = [
    "DataType",
    "TaskArgKind",
    "TASK_ARG_MAX_DIMS",
    "get_element_size",
    "get_dtype_name",
    "TaskArg",
    "TaskArgArray",
    "torch_dtype_to_datatype",
    "make_tensor_arg",
    "make_scalar_arg",
]


# Lazy-loaded torch dtype → DataType map (avoids importing torch at module load)
_TORCH_DTYPE_MAP = None


def _ensure_torch_map():
    global _TORCH_DTYPE_MAP  # noqa: PLW0603
    if _TORCH_DTYPE_MAP is not None:
        return
    import torch  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    _TORCH_DTYPE_MAP = {
        torch.float32: DataType.FLOAT32,
        torch.float16: DataType.FLOAT16,
        torch.int32: DataType.INT32,
        torch.int16: DataType.INT16,
        torch.int8: DataType.INT8,
        torch.uint8: DataType.UINT8,
        torch.bfloat16: DataType.BFLOAT16,
        torch.int64: DataType.INT64,
    }


def torch_dtype_to_datatype(dt) -> DataType:
    """Convert a ``torch.dtype`` to a ``DataType`` enum value.

    Raises ``KeyError`` for unsupported dtypes.
    """
    _ensure_torch_map()
    return _TORCH_DTYPE_MAP[dt]  # pyright: ignore[reportOptionalSubscript]


def make_tensor_arg(tensor) -> TaskArg:
    """Create a TENSOR ``TaskArg`` from a torch.Tensor.

    The tensor must be CPU-contiguous. Its ``data_ptr()``, shape, and dtype
    are read and stored in the returned ``TaskArg``.
    """
    _ensure_torch_map()
    dt = _TORCH_DTYPE_MAP.get(tensor.dtype)  # pyright: ignore[reportOptionalMemberAccess]
    if dt is None:
        raise ValueError(f"Unsupported tensor dtype for TaskArg: {tensor.dtype}")
    shapes = tuple(int(s) for s in tensor.shape)
    return TaskArg.make_tensor(tensor.data_ptr(), shapes, dt)


def make_scalar_arg(value) -> TaskArg:
    """Create a SCALAR ``TaskArg``.

    *value* can be a Python int, float, a ctypes scalar (``c_int64``,
    ``c_float``, etc.), or any object convertible to ``int``.

    Python float values are converted to IEEE 754 single precision (32-bit)
    and their bit pattern is zero-extended to uint64. This may cause a loss of
    precision. For double precision, use ``ctypes.c_double``.
    """
    import struct as _struct  # noqa: PLC0415

    if isinstance(value, float):
        bits = _struct.unpack("<I", _struct.pack("<f", value))[0]
        return TaskArg.make_scalar(bits)
    import ctypes as _ct  # noqa: PLC0415

    if isinstance(value, _ct._SimpleCData):
        if isinstance(value, (_ct.c_float, _ct.c_double)):
            uint_type = _ct.c_uint32 if isinstance(value, _ct.c_float) else _ct.c_uint64
            bits = uint_type.from_buffer_copy(value).value
            return TaskArg.make_scalar(bits)
        return TaskArg.make_scalar(int(value.value) & 0xFFFFFFFFFFFFFFFF)
    return TaskArg.make_scalar(int(value) & 0xFFFFFFFFFFFFFFFF)
