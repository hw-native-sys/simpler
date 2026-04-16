# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLW0603, PLC0415
"""Public Python API for task_interface nanobind bindings.

Re-exports the canonical C++ types (DataType, ContinuousTensor, ChipStorageTaskArgs,
DynamicTaskArgs, TaggedTaskArgs, TensorArgType) and adds torch-aware convenience helpers.

Usage:
    from task_interface import DataType, ContinuousTensor, ChipStorageTaskArgs, make_tensor_arg
"""

from _task_interface import (  # pyright: ignore[reportMissingImports]
    CONTINUOUS_TENSOR_MAX_DIMS,
    DIST_CHIP_MAILBOX_SIZE,
    DIST_SUB_MAILBOX_SIZE,
    ArgDirection,
    ChipCallable,
    ChipCallConfig,
    ChipStorageTaskArgs,
    ContinuousTensor,
    CoreCallable,
    DataType,
    DistChipProcess,
    DistInputSpec,
    DistOutputSpec,
    DistSubmitOutput,
    DistSubmitResult,
    DistSubWorker,
    DistWorker,
    DynamicTaskArgs,
    TaggedTaskArgs,
    TaskState,
    TensorArgType,
    WorkerPayload,
    WorkerType,
    _ChipWorker,
    arg_direction_name,
    get_dtype_name,
    get_element_size,
)

__all__ = [
    "DataType",
    "get_element_size",
    "get_dtype_name",
    "CONTINUOUS_TENSOR_MAX_DIMS",
    "ContinuousTensor",
    "ChipStorageTaskArgs",
    "TensorArgType",
    "DynamicTaskArgs",
    "TaggedTaskArgs",
    "ArgDirection",
    "CoreCallable",
    "ChipCallable",
    "ChipCallConfig",
    "ChipWorker",
    "arg_direction_name",
    "torch_dtype_to_datatype",
    "make_tensor_arg",
    "scalar_to_uint64",
    "get_active_worker",
    "host_malloc",
    "host_free",
    "host_register_mapped",
    "host_unregister_mapped",
    "malloc_host_device_share_mem",
    "free_host_device_share_mem",
    "mallocHostDeviceShareMem",
    "freeHostDeviceShareMem",
    # Distributed runtime
    "WorkerType",
    "TaskState",
    "WorkerPayload",
    "DistInputSpec",
    "DistOutputSpec",
    "DistSubmitOutput",
    "DistSubmitResult",
    "DistSubWorker",
    "DistChipProcess",
    "DistWorker",
    "DIST_SUB_MAILBOX_SIZE",
    "DIST_CHIP_MAILBOX_SIZE",
]


# Lazy-loaded torch dtype → DataType map (avoids importing torch at module load)
_TORCH_DTYPE_MAP = None
_ACTIVE_WORKER = None


def _ensure_torch_map():
    global _TORCH_DTYPE_MAP
    if _TORCH_DTYPE_MAP is not None:
        return
    import torch  # pyright: ignore[reportMissingImports]

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


def make_tensor_arg(tensor) -> ContinuousTensor:
    """Create a ``ContinuousTensor`` from a torch.Tensor.

    The tensor must be CPU-contiguous. Its ``data_ptr()``, shape, and dtype
    are read and stored in the returned ``ContinuousTensor``.
    """
    _ensure_torch_map()
    dt = _TORCH_DTYPE_MAP.get(tensor.dtype)  # pyright: ignore[reportOptionalMemberAccess]
    if dt is None:
        raise ValueError(f"Unsupported tensor dtype for ContinuousTensor: {tensor.dtype}")
    shapes = tuple(int(s) for s in tensor.shape)
    return ContinuousTensor.make(tensor.data_ptr(), shapes, dt)


def scalar_to_uint64(value) -> int:
    """Convert a scalar value to ``uint64``.

    *value* can be a Python int, float, a ctypes scalar (``c_int64``,
    ``c_float``, etc.), or any object convertible to ``int``.

    Python float values are converted to IEEE 754 single precision (32-bit)
    and their bit pattern is zero-extended to uint64. This may cause a loss of
    precision. For double precision, use ``ctypes.c_double``.
    """
    import struct as _struct

    if isinstance(value, float):
        bits = _struct.unpack("<I", _struct.pack("<f", value))[0]
        return bits
    import ctypes as _ct

    if isinstance(value, _ct._SimpleCData):
        if isinstance(value, (_ct.c_float, _ct.c_double)):
            uint_type = _ct.c_uint32 if isinstance(value, _ct.c_float) else _ct.c_uint64
            return uint_type.from_buffer_copy(value).value
        return int(value.value) & 0xFFFFFFFFFFFFFFFF
    return int(value) & 0xFFFFFFFFFFFFFFFF


class ChipWorker:
    """Unified execution interface wrapping the host runtime C API.

    The runtime library is bound once via init() and cannot be changed.
    Devices can be set and reset independently.

    Usage::

        worker = ChipWorker()
        worker.init(host_path="build/lib/.../host.so",
                    aicpu_path="build/lib/.../aicpu.so",
                    aicore_path="build/lib/.../aicore.o")
        worker.set_device(device_id=0)
        worker.run(chip_callable, orch_args, block_dim=24)
        worker.reset_device()
        worker.finalize()
    """

    def __init__(self):
        global _ACTIVE_WORKER
        self._impl = _ChipWorker()
        _ACTIVE_WORKER = self

    def init(self, host_path, aicpu_path, aicore_path, sim_context_lib_path=""):
        """Load host runtime library and cache platform binaries.

        Can only be called once — the runtime cannot be changed.

        Args:
            host_path: Path to the host runtime shared library (.so).
            aicpu_path: Path to the AICPU binary (.so).
            aicore_path: Path to the AICore binary (.o).
            sim_context_lib_path: Path to libcpu_sim_context.so (sim only).
        """
        self._impl.init(str(host_path), str(aicpu_path), str(aicore_path), str(sim_context_lib_path))

    def set_device(self, device_id):
        """Set the target NPU device.

        Requires init() first. Can be called after reset_device() to switch devices.

        Args:
            device_id: NPU device ID.
        """
        self._impl.set_device(device_id)

    def reset_device(self):
        """Release device resources. The runtime binding remains intact."""
        self._impl.reset_device()

    def host_malloc(self, size):
        """Allocate pinned host memory through the bound host runtime."""
        return self._impl.host_malloc(int(size))

    def host_free(self, host_ptr):
        """Free pinned host memory previously allocated by host_malloc()."""
        self._impl.host_free(int(host_ptr))

    def host_register_mapped(self, host_ptr, size, device_id=None):
        """Register host memory and return the device-visible mapped pointer."""
        if device_id is None:
            device_id = self.device_id
        return self._impl.host_register_mapped(int(host_ptr), int(size), int(device_id))

    def host_unregister_mapped(self, host_ptr, device_id=None):
        """Unregister host memory previously registered via host_register_mapped()."""
        if device_id is None:
            device_id = self.device_id
        self._impl.host_unregister_mapped(int(host_ptr), int(device_id))

    def malloc_host_device_share_mem(self, size, device_id=None):
        """Allocate host memory and register it as a device-visible mapped buffer."""
        if device_id is None:
            device_id = self.device_id
        host_ptr, dev_ptr = self._impl.malloc_host_device_share_mem(int(size), int(device_id))
        return int(host_ptr), int(dev_ptr)

    def free_host_device_share_mem(self, host_ptr, device_id=None):
        """Unregister and free a mapped host buffer."""
        if device_id is None:
            device_id = self.device_id
        self._impl.free_host_device_share_mem(int(host_ptr), int(device_id))

    def finalize(self):
        """Tear down everything: device resources and runtime library.

        Terminal operation — the object cannot be reused after this.
        """
        global _ACTIVE_WORKER
        self._impl.finalize()
        if _ACTIVE_WORKER is self:
            _ACTIVE_WORKER = None

    def run(self, callable, args, config=None, **kwargs):
        """Execute a callable synchronously.

        Args:
            callable: ChipCallable built from orchestration + kernel binaries.
            args: ChipStorageTaskArgs for this invocation.
            config: Optional ChipCallConfig. If None, a default is created.
            **kwargs: Overrides applied to config (e.g. block_dim=24).
        """
        if config is None:
            config = ChipCallConfig()
        for k, v in kwargs.items():
            setattr(config, k, v)
        self._impl.run(callable, args, config)

    @property
    def device_id(self):
        return self._impl.device_id

    @property
    def initialized(self):
        return self._impl.initialized

    @property
    def device_set(self):
        return self._impl.device_set


def get_active_worker():
    """Return the most recently created ChipWorker in this process."""
    if _ACTIVE_WORKER is None:
        raise RuntimeError("No active ChipWorker is available")
    if not _ACTIVE_WORKER.initialized:
        raise RuntimeError("The active ChipWorker is not initialized")
    if not _ACTIVE_WORKER.device_set:
        raise RuntimeError("The active ChipWorker does not have a device set")
    return _ACTIVE_WORKER


def host_malloc(size):
    """Allocate pinned host memory using the active ChipWorker."""
    return get_active_worker().host_malloc(size)


def host_free(host_ptr):
    """Free pinned host memory using the active ChipWorker."""
    get_active_worker().host_free(host_ptr)


def host_register_mapped(host_ptr, size, device_id=None):
    """Register pinned host memory and return a device-visible mapped pointer."""
    return get_active_worker().host_register_mapped(host_ptr, size, device_id=device_id)


def host_unregister_mapped(host_ptr, device_id=None):
    """Unregister pinned host memory from the active ChipWorker."""
    get_active_worker().host_unregister_mapped(host_ptr, device_id=device_id)


def malloc_host_device_share_mem(size, device_id=None):
    """Allocate host memory and register it as a device-visible mapped buffer."""
    return get_active_worker().malloc_host_device_share_mem(size, device_id=device_id)


def free_host_device_share_mem(host_ptr, device_id=None):
    """Unregister and free a mapped host buffer."""
    get_active_worker().free_host_device_share_mem(host_ptr, device_id=device_id)


def mallocHostDeviceShareMem(device_id, size):
    """Compatibility helper matching the minimal C-style API naming."""
    return malloc_host_device_share_mem(size, device_id=device_id)


def freeHostDeviceShareMem(device_id, host_ptr):
    """Compatibility helper matching the minimal C-style API naming."""
    free_host_device_share_mem(host_ptr, device_id=device_id)
