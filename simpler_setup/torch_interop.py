# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLW0603, PLC0415
"""Torch integration helpers.

Canonical home for torch-aware helpers that convert ``torch.Tensor`` and
``torch.dtype`` values into the runtime's ``ContinuousTensor`` / ``DataType``
types. These helpers live in ``simpler_setup`` (not ``simpler``) so that the
stable ``simpler`` runtime API can remain torch-free; torch integration is a
setup-time/test-framework concern.

Callers:
    from simpler_setup.torch_interop import make_tensor_arg, torch_dtype_to_datatype

torch is imported lazily inside ``_ensure_torch_map`` so that importing this
module does not force torch onto users who only touch ``simpler_setup`` for
other reasons (e.g. ``RuntimeBuilder``). ``simpler.task_interface`` is also
imported lazily because ``simpler_setup/__init__.py`` is executed during
``pip install`` (via ``build_runtimes.py``), before the ``_task_interface``
nanobind extension is built.

Requires torch >= 2.3 (for ``torch.uint16`` / ``torch.uint32``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simpler.task_interface import ContinuousTensor, DataType

_TORCH_DTYPE_MAP = None


def _ensure_torch_map():
    global _TORCH_DTYPE_MAP
    if _TORCH_DTYPE_MAP is not None:
        return
    import torch  # pyright: ignore[reportMissingImports]
    from simpler.task_interface import DataType

    _TORCH_DTYPE_MAP = {
        torch.float32: DataType.FLOAT32,
        torch.float16: DataType.FLOAT16,
        torch.int32: DataType.INT32,
        torch.int16: DataType.INT16,
        torch.int8: DataType.INT8,
        torch.uint8: DataType.UINT8,
        torch.bfloat16: DataType.BFLOAT16,
        torch.int64: DataType.INT64,
        torch.uint16: DataType.UINT16,
        torch.uint32: DataType.UINT32,
    }


def torch_dtype_to_datatype(dt) -> DataType:
    """Convert a ``torch.dtype`` to a ``DataType`` enum value.

    Raises ``KeyError`` for unsupported dtypes.
    """
    _ensure_torch_map()
    return _TORCH_DTYPE_MAP[dt]  # pyright: ignore[reportOptionalSubscript]


def make_tensor_arg(tensor) -> ContinuousTensor:
    """Create a ``ContinuousTensor`` from a torch.Tensor.

    The tensor must be CPU-contiguous. Its ``data_ptr()``, shape, and dtype
    are read and stored in the returned ``ContinuousTensor``.
    """
    from simpler.task_interface import ContinuousTensor

    _ensure_torch_map()
    dt = _TORCH_DTYPE_MAP.get(tensor.dtype)  # pyright: ignore[reportOptionalMemberAccess]
    if dt is None:
        raise ValueError(f"Unsupported tensor dtype for ContinuousTensor: {tensor.dtype}")
    shapes = tuple(int(s) for s in tensor.shape)
    return ContinuousTensor.make(tensor.data_ptr(), shapes, dt)
