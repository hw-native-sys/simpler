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
``torch.dtype`` values into the runtime's ``Tensor`` / ``ChipTensor`` / ``DataType``
types. These helpers live in ``simpler_setup`` (not ``simpler``) so that the
stable ``simpler`` runtime API can remain torch-free; torch integration is a
setup-time/test-framework concern.

Callers:
    from simpler_setup.torch_interop import make_chip_tensor_arg, torch_dtype_to_datatype

torch is imported lazily inside ``_ensure_torch_map`` so that importing this
module does not force torch onto users who only touch ``simpler_setup`` for
other reasons (e.g. ``RuntimeBuilder``). ``simpler.task_interface`` is also
imported lazily because ``simpler_setup/__init__.py`` is executed during
``pip install`` (via ``build_runtimes.py``), before the ``_task_interface``
nanobind extension is built.

Requires torch >= 2.3 (for ``torch.uint16`` / ``torch.uint32``).
"""

from __future__ import annotations

import itertools
import threading
import weakref
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simpler.task_interface import ChipTensor, DataType

_TORCH_DTYPE_MAP = None
_TORCH_CONTENT_GENERATIONS = itertools.count(1)
_TORCH_CONTENT_STATES = {}
_TORCH_CONTENT_STATES_LOCK = threading.Lock()


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
    # MX low-precision dtypes — A5 only (consumed by the A5-only pto.tquant.mx /
    # tmatmul.mx ops). MX paths use E4M3FN (+ E8M0 scale) and packed E2M1;
    # FP8E5M2 is not registered. float8_e4m3fn needs torch >= 2.1;
    # float8_e8m0fnu needs >= 2.7; float4_e2m1fn_x2 needs a recent torch.
    # Guard with getattr so an older torch silently skips the ones it lacks.
    for _torch_dt_name, _sim_dt in (
        ("float8_e4m3fn", DataType.FP8E4M3FN),  # A5 only
        ("float8_e8m0fnu", DataType.FP8E8M0),  # A5 only
        ("float4_e2m1fn_x2", DataType.FP4E2M1),  # A5 only
    ):
        _torch_dt = getattr(torch, _torch_dt_name, None)
        if _torch_dt is not None:
            _TORCH_DTYPE_MAP[_torch_dt] = _sim_dt


def torch_dtype_to_datatype(dt) -> DataType:
    """Convert a ``torch.dtype`` to a ``DataType`` enum value.

    Raises ``KeyError`` for unsupported dtypes.
    """
    _ensure_torch_map()
    return _TORCH_DTYPE_MAP[dt]  # pyright: ignore[reportOptionalSubscript]


def torch_content_generation(tensor) -> int:
    """Return a nonzero freshness generation for a torch tensor.

    The token is stable for one tensor object while PyTorch's ``_version`` is
    unchanged, and advances for both a new tensor object and a tracked in-place
    mutation. Object identity matters because CPU allocators can reuse a freed
    tensor's data pointer while the new tensor starts again at ``_version == 0``.
    Weak references prevent the identity table from retaining tensors.
    """
    try:
        version = getattr(tensor, "_version", None)
    except (AttributeError, RuntimeError):
        return 0
    if not isinstance(version, int) or version < 0:
        return 0

    tensor_id = id(tensor)

    def _forget_tensor(ref, *, expected_id=tensor_id):
        with _TORCH_CONTENT_STATES_LOCK:
            state = _TORCH_CONTENT_STATES.get(expected_id)
            if state is not None and state[0] is ref:
                del _TORCH_CONTENT_STATES[expected_id]

    with _TORCH_CONTENT_STATES_LOCK:
        state = _TORCH_CONTENT_STATES.get(tensor_id)
        if state is not None and state[0]() is tensor and state[1] == version:
            return state[2]
        try:
            tensor_ref = weakref.ref(tensor, _forget_tensor)
        except TypeError:
            return 0
        generation = next(_TORCH_CONTENT_GENERATIONS)
        if generation >= 1 << 64:
            return 0
        _TORCH_CONTENT_STATES[tensor_id] = (tensor_ref, version, generation)
        return generation


def make_chip_tensor_arg(tensor) -> ChipTensor:
    """Create a ``ChipTensor`` — the materialized chip POD — from a torch.Tensor.

    Distinct from ``Worker.make_tensor_arg``, which names a wire ``Tensor`` (identity, no
    address) for the ordinary dispatch path. This one carries a raw address and is only for
    the direct ``ChipWorker`` API.

    The result is always contiguous (row-major strides, ``start_offset == 0``) —
    the unified ``ChipTensor`` can express strided views, but this construction path
    is constrained to contiguous memory. The input torch tensor MUST therefore be
    contiguous; a non-contiguous tensor raises ``ValueError`` (call
    ``.contiguous()`` first). It must also be a CPU tensor: a device tensor's
    ``data_ptr()`` is a device pointer that requires ``child_memory=True``, which
    this helper does not set, so a non-CPU tensor raises ``ValueError``. Its
    ``data_ptr()``, shape, and dtype are read and stored in the returned
    ``ChipTensor``.
    """
    from simpler.task_interface import ChipTensor

    _ensure_torch_map()
    dt = _TORCH_DTYPE_MAP.get(tensor.dtype)  # pyright: ignore[reportOptionalMemberAccess]
    if dt is None:
        raise ValueError(f"Unsupported tensor dtype for ChipTensor: {tensor.dtype}")
    if tensor.device.type != "cpu":
        raise ValueError(
            f"make_chip_tensor_arg requires a CPU tensor, got device={tensor.device}. "
            "A device pointer must be wrapped explicitly via "
            "ChipTensor.make(..., child_memory=True)."
        )
    if not tensor.is_contiguous():
        raise ValueError(
            "make_chip_tensor_arg requires a contiguous tensor (TaskArgs ChipTensors are constructed "
            "contiguous); call tensor.contiguous() before passing it."
        )
    shapes = tuple(int(s) for s in tensor.shape)
    return ChipTensor.make(
        tensor.data_ptr(),
        shapes,
        dt,
        host_content_generation=torch_content_generation(tensor),
    )


def make_tensor_arg(worker, tensor):
    """A ``Tensor`` task arg over a **pre-fork** host torch tensor.

    Names ``tensor`` as a memoized ``FORK_SHM`` handle on ``worker`` (``worker.make_tensor_arg``), inferring
    shapes + dtype from the tensor. Use for standalone L3 examples whose host inputs/outputs are
    ``share_memory_()`` tensors allocated before ``worker.init()`` (fork-inherited). A contiguous CPU
    tensor is required (as for ``make_chip_tensor_arg``).
    """
    _ensure_torch_map()
    dt = _TORCH_DTYPE_MAP.get(tensor.dtype)  # pyright: ignore[reportOptionalMemberAccess]
    if dt is None:
        raise ValueError(f"Unsupported tensor dtype: {tensor.dtype}")
    if tensor.device.type != "cpu":
        raise ValueError(f"make_tensor_arg requires a CPU tensor, got device={tensor.device}.")
    if not tensor.is_contiguous():
        raise ValueError("make_tensor_arg requires a contiguous tensor; call tensor.contiguous() first.")
    shapes = tuple(int(s) for s in tensor.shape)
    return worker.make_tensor_arg(tensor, shapes=shapes, dtype=int(dt.value))
