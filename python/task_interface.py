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

from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

# CHIP_STORAGE_TASK_ARGS_SIZE is the authoritative C++ sizeof(ChipStorageTaskArgs),
# exported for mailbox memcpy paths in the L3 chip-process worker flow.
from _task_interface import (  # pyright: ignore[reportMissingImports]
    CHIP_STORAGE_TASK_ARGS_SIZE,
    CONTINUOUS_TENSOR_MAX_DIMS,
    DIST_CHIP_BOOTSTRAP_MAILBOX_SIZE,
    DIST_CHIP_MAILBOX_SIZE,
    DIST_SUB_MAILBOX_SIZE,
    ArgDirection,
    CallConfig,
    ChipCallable,
    ChipStorageTaskArgs,
    ChipBootstrapMailboxState,
    ContinuousTensor,
    CoreCallable,
    DataType,
    DistChipBootstrapChannel,
    DistChipProcess,
    DistInputSpec,
    DistOutputOwnership,
    DistOutputSpec,
    DistSubmitOutput,
    DistSubmitResult,
    DistTensorKey,
    DistSubWorker,
    DistWorker,
    DynamicTaskArgs,
    TaggedTaskArgs,
    TaskState,
    TensorStorageType,
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
    "CHIP_STORAGE_TASK_ARGS_SIZE",
    "CONTINUOUS_TENSOR_MAX_DIMS",
    "ContinuousTensor",
    "ChipStorageTaskArgs",
    "TensorArgType",
    "TensorStorageType",
    "DynamicTaskArgs",
    "TaggedTaskArgs",
    "ArgDirection",
    "CoreCallable",
    "ChipCallable",
    "CallConfig",
    "ChipWorker",
    "arg_direction_name",
    "torch_dtype_to_datatype",
    "make_tensor_arg",
    "make_device_tensor_arg",
    "scalar_to_uint64",
    # Distributed runtime
    "WorkerType",
    "TaskState",
    "WorkerPayload",
    "DistTensorKey",
    "DistInputSpec",
    "DistOutputOwnership",
    "DistOutputSpec",
    "DistSubmitOutput",
    "DistSubmitResult",
    "DistSubWorker",
    "DistChipBootstrapChannel",
    "DistChipProcess",
    "DistWorker",
    "ChipBootstrapMailboxState",
    "DIST_SUB_MAILBOX_SIZE",
    "DIST_CHIP_MAILBOX_SIZE",
    "DIST_CHIP_BOOTSTRAP_MAILBOX_SIZE",
    "ChipBootstrapResult",
]


@dataclass
class ChipBootstrapResult:
    """Parent-visible reply from per-chip bootstrap."""

    comm_handle: int
    device_ctx: int
    local_window_base: int
    actual_window_size: int
    buffer_ptrs: list[int]


# Lazy-loaded torch dtype → DataType map (avoids importing torch at module load)
_TORCH_DTYPE_MAP = None


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


def make_device_tensor_arg(ptr: int, shape, dtype) -> ContinuousTensor:
    """Create a device-resident ``ContinuousTensor`` from an external device pointer.

    Args:
        ptr: Device or window pointer already valid in the target chip process.
        shape: Iterable of tensor dimensions.
        dtype: Either ``DataType`` or a ``torch.dtype`` supported by ``make_tensor_arg``.
    """
    _ensure_torch_map()
    if not isinstance(dtype, DataType):
        dtype = _TORCH_DTYPE_MAP.get(dtype)  # pyright: ignore[reportOptionalMemberAccess]
    if dtype is None:
        raise ValueError(f"Unsupported dtype for ContinuousTensor: {dtype}")
    shapes = tuple(int(s) for s in shape)
    return ContinuousTensor.make(int(ptr), shapes, dtype, device_resident=True)


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
        self._impl = _ChipWorker()

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

    def finalize(self):
        """Tear down everything: device resources and runtime library.

        Terminal operation — the object cannot be reused after this.
        """
        self._impl.finalize()

    def run(self, callable, args, config=None, **kwargs):
        """Execute a callable synchronously.

        Args:
            callable: ChipCallable built from orchestration + kernel binaries.
            args: ChipStorageTaskArgs for this invocation.
            config: Optional CallConfig. If None, a default is created.
            **kwargs: Overrides applied to config (e.g. block_dim=24).
        """
        if config is None:
            config = CallConfig()
        for k, v in kwargs.items():
            setattr(config, k, v)
        self._impl.run(callable, args, config)

    def run_raw(self, callable, args, *, block_dim=1, aicpu_thread_num=3, enable_profiling=False):
        """Run a callable using raw pointer arguments."""
        self._impl.run_raw(int(callable), int(args), int(block_dim), int(aicpu_thread_num), bool(enable_profiling))

    def device_malloc(self, size):
        """Allocate device memory in the current device context."""
        return int(self._impl.device_malloc(int(size)))

    def device_free(self, dev_ptr):
        """Free device memory allocated by ``device_malloc()``."""
        self._impl.device_free(int(dev_ptr))

    def copy_to_device(self, dev_ptr, host_ptr, size):
        """Copy bytes from a host pointer into a device pointer."""
        self._impl.copy_to_device(int(dev_ptr), int(host_ptr), int(size))

    def copy_from_device(self, host_ptr, dev_ptr, size):
        """Copy bytes from a device pointer into a host pointer."""
        self._impl.copy_from_device(int(host_ptr), int(dev_ptr), int(size))

    def comm_init(self, rank, nranks, device_id, rootinfo_path):
        """Create a communicator in the current chip child."""
        return int(self._impl.comm_init(int(rank), int(nranks), int(device_id), str(rootinfo_path)))

    def comm_alloc_windows(self, comm_handle, win_size):
        """Allocate the communicator-owned window and return the device context."""
        return int(self._impl.comm_alloc_windows(int(comm_handle), int(win_size)))

    def comm_get_local_window_base(self, comm_handle):
        """Return the local base address of the communicator window."""
        return int(self._impl.comm_get_local_window_base(int(comm_handle)))

    def comm_get_window_size(self, comm_handle):
        """Return the actual communicator window size."""
        return int(self._impl.comm_get_window_size(int(comm_handle)))

    def comm_destroy(self, comm_handle):
        """Destroy a communicator previously created by ``comm_init()``."""
        self._impl.comm_destroy(int(comm_handle))

    def bootstrap(
        self,
        device_id,
        *,
        comm_rank=-1,
        comm_nranks=0,
        rootinfo_path="",
        window_size=0,
        win_sync_prefix=0,
        buffer_sizes,
        buffer_placements,
        input_blobs,
    ):
        """Bootstrap per-chip runtime state before the first task submission.

        This optional handshake extends plain ``init()`` with communicator setup,
        window/device buffer allocation, initial H2D staging, and a bootstrap
        reply that the parent process can use to build task arguments.
        """
        buffer_sizes = [int(size) for size in buffer_sizes]
        buffer_placements = [str(placement) for placement in buffer_placements]
        input_blobs = list(input_blobs)

        if len(buffer_sizes) != len(buffer_placements):
            raise ValueError("buffer_sizes and buffer_placements must have the same length")
        if len(buffer_sizes) != len(input_blobs):
            raise ValueError("input_blobs length must match buffer_sizes")

        enable_comm = int(comm_rank) >= 0
        comm_handle = 0
        device_ctx = 0
        local_window_base = 0
        actual_window_size = 0
        owned_device_ptrs: list[int] = []
        buffer_ptrs: list[int] = []

        try:
            if enable_comm:
                if int(comm_nranks) <= 0:
                    raise ValueError("comm_nranks must be positive when comm bootstrap is enabled")
                if not str(rootinfo_path):
                    raise ValueError("rootinfo_path is required when comm bootstrap is enabled")
                comm_handle = self.comm_init(comm_rank, comm_nranks, device_id, rootinfo_path)

            if not self.device_set:
                self.set_device(int(device_id))
            elif self.device_id != int(device_id):
                raise ValueError("ChipWorker already bound to a different device")

            if enable_comm:
                device_ctx = self.comm_alloc_windows(comm_handle, window_size)
                local_window_base = self.comm_get_local_window_base(comm_handle)
                actual_window_size = self.comm_get_window_size(comm_handle)

            win_offset = int(win_sync_prefix)
            for size, placement, blob in zip(buffer_sizes, buffer_placements, input_blobs, strict=True):
                ptr = 0
                if placement == "window":
                    if not enable_comm:
                        raise ValueError("window placement requires comm bootstrap")
                    ptr = local_window_base + win_offset
                    win_offset += size
                elif placement == "device":
                    ptr = self.device_malloc(size)
                    owned_device_ptrs.append(ptr)
                else:
                    raise ValueError(f"Unsupported buffer placement: {placement}")

                buffer_ptrs.append(ptr)

                if blob is not None:
                    if not isinstance(blob, bytes):
                        raise ValueError("input blobs must be bytes or None")
                    if len(blob) != size:
                        raise ValueError("input blob size must match buffer size")
                    if size > 0:
                        import ctypes as _ct

                        host_buf = _ct.create_string_buffer(blob, size)
                        self.copy_to_device(ptr, _ct.addressof(host_buf), size)

            if enable_comm:
                self.comm_barrier(comm_handle)
        except Exception:
            for ptr in owned_device_ptrs:
                try:
                    self.device_free(ptr)
                except Exception:
                    pass
            if comm_handle != 0:
                try:
                    self.comm_destroy(comm_handle)
                except Exception:
                    pass
            raise

        return {
            "comm_handle": comm_handle,
            "device_ctx": device_ctx,
            "local_window_base": local_window_base,
            "actual_window_size": actual_window_size,
            "buffer_ptrs": buffer_ptrs,
        }

    def shutdown_bootstrap(self, *, comm_handle=0, buffer_ptrs, buffer_placements):
        """Release per-chip runtime state previously created by ``bootstrap()``."""
        buffer_ptrs = [int(ptr) for ptr in buffer_ptrs]
        buffer_placements = [str(placement) for placement in buffer_placements]
        if len(buffer_ptrs) != len(buffer_placements):
            raise ValueError("buffer_ptrs and buffer_placements must have the same length")
        for ptr, placement in zip(buffer_ptrs, buffer_placements, strict=True):
            if placement == "device" and ptr != 0:
                self.device_free(ptr)
        if int(comm_handle) != 0:
            self.comm_destroy(int(comm_handle))

    @staticmethod
    def _read_bootstrap_input_bytes(shm_name: str, size: int) -> bytes:
        shm = SharedMemory(name=shm_name)
        try:
            if size == 0:
                return b""
            assert shm.buf is not None
            return bytes(shm.buf[:size])
        finally:
            shm.close()

    def bootstrap_context(self, device_id, chip_bootstrap_config) -> ChipBootstrapResult:
        """Bootstrap a chip child from a typed bootstrap config."""
        comm_cfg = getattr(chip_bootstrap_config, "comm", None)
        input_blobs = []
        for buf in chip_bootstrap_config.buffers:
            if buf.load_from_host:
                staged = chip_bootstrap_config.input_staging(buf.name)
                input_blobs.append(self._read_bootstrap_input_bytes(staged.shm_name, staged.size))
            else:
                input_blobs.append(None)
        reply = self.bootstrap(
            device_id,
            comm_rank=comm_cfg.rank if comm_cfg is not None else -1,
            comm_nranks=comm_cfg.nranks if comm_cfg is not None else 0,
            rootinfo_path=comm_cfg.rootinfo_path if comm_cfg is not None else "",
            window_size=comm_cfg.window_size if comm_cfg is not None else 0,
            win_sync_prefix=comm_cfg.win_sync_prefix if comm_cfg is not None else 0,
            buffer_sizes=[buf.nbytes for buf in chip_bootstrap_config.buffers],
            buffer_placements=[buf.placement for buf in chip_bootstrap_config.buffers],
            input_blobs=input_blobs,
        )
        return ChipBootstrapResult(
            comm_handle=int(reply["comm_handle"]),
            device_ctx=int(reply["device_ctx"]),
            local_window_base=int(reply["local_window_base"]),
            actual_window_size=int(reply["actual_window_size"]),
            buffer_ptrs=[int(ptr) for ptr in reply["buffer_ptrs"]],
        )

    def shutdown_bootstrap_context(self, chip_bootstrap_config, *, comm_handle=0, buffer_ptrs):
        """Release resources created by ``bootstrap_context``."""
        self.shutdown_bootstrap(
            comm_handle=comm_handle,
            buffer_ptrs=buffer_ptrs,
            buffer_placements=[buf.placement for buf in chip_bootstrap_config.buffers],
        )

    def copy_device_to_bytes(self, dev_ptr, size) -> bytes:
        """Copy a device buffer into a Python bytes object."""
        size = int(size)
        if size == 0:
            return b""
        import ctypes as _ct

        host_buf = _ct.create_string_buffer(size)
        self._impl.copy_from_device(_ct.addressof(host_buf), int(dev_ptr), size)
        return host_buf.raw[:size]

    def comm_barrier(self, comm_handle):
        """Synchronize all ranks in the current communicator."""
        self._impl.comm_barrier(int(comm_handle))

    @property
    def device_id(self):
        return self._impl.device_id

    @property
    def initialized(self):
        return self._impl.initialized

    @property
    def device_set(self):
        return self._impl.device_set
