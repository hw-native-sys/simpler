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
TaskArgs, TensorArgType) plus ``scalar_to_uint64``. Torch-aware helpers
(``make_tensor_arg``, ``torch_dtype_to_datatype``) live in
``simpler_setup.torch_interop`` — this module has no torch dependency.

Usage:
    from simpler.task_interface import DataType, ContinuousTensor, ChipStorageTaskArgs
    from simpler_setup.torch_interop import make_tensor_arg
"""

import ctypes
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

from _task_interface import (  # pyright: ignore[reportMissingImports]
    CHIP_BOOTSTRAP_MAILBOX_SIZE,
    CONTINUOUS_TENSOR_MAX_DIMS,
    MAILBOX_ERROR_MSG_SIZE,
    MAILBOX_OFF_ERROR_MSG,
    MAILBOX_SIZE,
    ArgDirection,
    CallConfig,
    ChipBootstrapChannel,
    ChipBootstrapMailboxState,
    ChipCallable,
    ChipStorageTaskArgs,
    ContinuousTensor,
    CoreCallable,
    DataType,
    SubmitResult,
    TaskArgs,
    TaskState,
    TensorArgType,
    WorkerType,
    _ChipWorker,
    _Orchestrator,
    _Worker,
    arg_direction_name,
    get_dtype_name,
    get_element_size,
    read_args_from_blob,
)

__all__ = [
    "DataType",
    "get_element_size",
    "get_dtype_name",
    "CONTINUOUS_TENSOR_MAX_DIMS",
    "ContinuousTensor",
    "ChipStorageTaskArgs",
    "TensorArgType",
    "TaskArgs",
    "ArgDirection",
    "CoreCallable",
    "ChipCallable",
    "CallConfig",
    "ChipWorker",
    "arg_direction_name",
    "scalar_to_uint64",
    # Distributed runtime
    "WorkerType",
    "TaskState",
    "_Orchestrator",
    "SubmitResult",
    "_Worker",
    "MAILBOX_SIZE",
    "MAILBOX_OFF_ERROR_MSG",
    "MAILBOX_ERROR_MSG_SIZE",
    "read_args_from_blob",
    # Chip bootstrap
    "CHIP_BOOTSTRAP_MAILBOX_SIZE",
    "ChipBootstrapChannel",
    "ChipBootstrapMailboxState",
    "ChipCommBootstrapConfig",
    "ChipBufferSpec",
    "HostBufferStaging",
    "ChipBootstrapConfig",
    "ChipBootstrapResult",
    # Worker-level chip bootstrap orchestration
    "ChipContext",
]


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


@dataclass
class ChipCommBootstrapConfig:
    """Per-chip communicator bring-up knobs consumed by `ChipWorker.bootstrap_context`.

    A ``ChipBootstrapConfig`` with ``comm=None`` skips the communicator step
    entirely; in that mode ``cfg.buffers`` must be empty because buffers are
    per-rank slices of the HCCL communicator window, and the window only
    exists once a communicator has been brought up.  Comm-less configs are
    used by error-path tests that need to trip ``bootstrap_context`` before
    it reaches any communicator call.
    """

    rank: int
    nranks: int
    rootinfo_path: str
    window_size: int
    """Requested per-rank window size in bytes.  HCCL may round this up — the
    actual allocation is reported back via
    ``ChipBootstrapResult.actual_window_size`` and must be what callers use
    when slicing the window."""


@dataclass
class ChipBufferSpec:
    """A named slice of the per-rank communicator window.

    Buffers are placed sequentially inside the window in declaration order —
    ``ChipBootstrapResult.buffer_ptrs`` is 1:1 aligned with the ``buffers``
    list so downstream code (the Worker's ``ChipContext``) can build a
    ``name → ptr`` dict by zipping the two.
    """

    name: str
    dtype: str
    count: int
    nbytes: int
    load_from_host: bool = False
    store_to_host: bool = False


@dataclass
class HostBufferStaging:
    """A POSIX shared-memory region staged by the parent for one named buffer.

    The parent creates the ``SharedMemory`` object and fills it with the input
    bytes *before* forking; the child attaches read-only via
    ``SharedMemory(name=shm_name)`` and does not unlink it.
    """

    name: str
    shm_name: str
    size: int


@dataclass
class ChipBootstrapConfig:
    """Inputs to `ChipWorker.bootstrap_context` for one chip child."""

    comm: Optional[ChipCommBootstrapConfig] = None
    buffers: list[ChipBufferSpec] = field(default_factory=list)
    host_inputs: list[HostBufferStaging] = field(default_factory=list)
    host_outputs: list[HostBufferStaging] = field(default_factory=list)

    def input_staging(self, buffer_name: str) -> HostBufferStaging:
        for s in self.host_inputs:
            if s.name == buffer_name:
                return s
        raise KeyError(buffer_name)

    def output_staging(self, buffer_name: str) -> HostBufferStaging:
        for s in self.host_outputs:
            if s.name == buffer_name:
                return s
        raise KeyError(buffer_name)


@dataclass
class ChipBootstrapResult:
    """Return value of `ChipWorker.bootstrap_context` — and the tuple the
    `ChipBootstrapChannel` publishes to the parent on success.
    """

    device_ctx: int
    local_window_base: int
    actual_window_size: int
    buffer_ptrs: list[int]


@dataclass
class ChipContext:
    """Per-chip view of a successful bootstrap, exposed to L3+ orch functions.

    Built by the parent `Worker` in `_start_hierarchical` from the
    `ChipBootstrapConfig` it forwarded to the chip child and the
    `ChipBootstrapResult` the child published via its `ChipBootstrapChannel`.
    `buffer_ptrs` is the `name → device pointer` map obtained by zipping the
    config's `ChipBufferSpec.name` with the result's `buffer_ptrs`, so orch
    code can address a named window slice without tracking list indices.
    """

    device_id: int
    rank: int
    nranks: int
    device_ctx: int
    local_window_base: int
    actual_window_size: int
    buffer_ptrs: dict[str, int]


class ChipWorker:
    """Unified execution interface wrapping the host runtime C API.

    The runtime library and target device are bound once via init() and
    cannot be changed.

    Usage::

        worker = ChipWorker()
        worker.init(device_id=0, bins=bins)
        worker.run(chip_callable, orch_args, block_dim=24)
        worker.finalize()
    """

    def __init__(self):
        self._impl = _ChipWorker()

    def init(self, device_id, bins, log_level=None, log_info_v=None):
        """Attach the calling thread to ``device_id``, load the host runtime
        library, and cache platform binaries.

        Can only be called once — the runtime and device cannot be changed
        after init.

        Args:
            device_id: NPU device ID to attach the calling thread to.
            bins: A `simpler_setup.runtime_builder.RuntimeBinaries` (or any
                object exposing host_path / aicpu_path / aicore_path /
                simpler_log_path / sim_context_path).
            log_level: Severity floor (0=DEBUG..4=NUL). Defaults to a snapshot
                of the simpler logger via `_log.get_current_config()`.
            log_info_v: INFO verbosity threshold (0..9). Same default.

        For tests that need to drive the binding directly with arbitrary path
        strings (e.g. to assert dlopen failure on `/nonexistent/foo.so`), call
        `_ChipWorker.init(...)` from `_task_interface` instead of going
        through this wrapper.
        """
        if log_level is None or log_info_v is None:
            from . import _log  # noqa: PLC0415

            sev, info_v = _log.get_current_config()
            if log_level is None:
                log_level = sev
            if log_info_v is None:
                log_info_v = info_v
        self._impl.init(
            str(bins.host_path),
            str(bins.aicpu_path),
            str(bins.aicore_path),
            str(bins.simpler_log_path),
            int(device_id),
            str(bins.sim_context_path) if bins.sim_context_path else "",
            log_level,
            log_info_v,
        )

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

    def run_from_blob(self, callable, blob_ptr, config):
        """Execute via a serialized args blob in shared memory.

        Used by `_chip_process_loop` after reading the mailbox: instead of
        deserializing the args into Python objects, the C++ side parses the
        POD blob directly at `blob_ptr`.
        """
        self._impl.run_from_blob(int(callable), int(blob_ptr), config)

    def malloc(self, size):
        """Allocate memory. Returns a pointer (uint64)."""
        return int(self._impl.malloc(int(size)))

    def free(self, ptr):
        """Free memory allocated by ``malloc()``."""
        self._impl.free(int(ptr))

    def copy_to(self, dst, src, size):
        """Copy *size* bytes from host *src* to worker *dst*."""
        self._impl.copy_to(int(dst), int(src), int(size))

    def copy_from(self, dst, src, size):
        """Copy *size* bytes from worker *src* to host *dst*."""
        self._impl.copy_from(int(dst), int(src), int(size))

    def comm_init(self, rank: int, nranks: int, rootinfo_path: str) -> int:
        """Initialize a distributed communicator for this rank.

        ChipWorker owns ACL bring-up and the aclrtStream internally, so
        callers never touch ``aclInit`` / ``aclrtSetDevice`` / stream
        lifetimes.  On sim, ACL / stream are not used.  Pair with
        ``comm_destroy`` for teardown.

        Args:
            rank: This process's rank (0-based).
            nranks: Total number of ranks.
            rootinfo_path: Filesystem path used for rank handshake.

        Returns:
            Opaque communicator handle (uint64) for the other ``comm_*`` calls.
        """
        return int(self._impl.comm_init(int(rank), int(nranks), str(rootinfo_path)))

    def comm_alloc_windows(self, comm_handle: int, win_size: int) -> int:
        """Allocate per-rank windows. Returns a device CommContext pointer (uint64)."""
        return int(self._impl.comm_alloc_windows(int(comm_handle), int(win_size)))

    def comm_get_local_window_base(self, comm_handle: int) -> int:
        """Return this rank's local window base address (uint64)."""
        return int(self._impl.comm_get_local_window_base(int(comm_handle)))

    def comm_get_window_size(self, comm_handle: int) -> int:
        """Return the actual per-rank window size in bytes."""
        return int(self._impl.comm_get_window_size(int(comm_handle)))

    def comm_barrier(self, comm_handle: int) -> None:
        """Synchronize all ranks."""
        self._impl.comm_barrier(int(comm_handle))

    def comm_destroy(self, comm_handle: int) -> None:
        """Destroy the communicator and release its resources."""
        self._impl.comm_destroy(int(comm_handle))

    def bootstrap_context(  # noqa: PLR0912 -- config validation + comm setup + window carving + H2D staging in one linear flow; splitting would obscure the ordered failure semantics
        self,
        device_id: int,
        cfg: ChipBootstrapConfig,
        channel: Optional[ChipBootstrapChannel] = None,
    ) -> ChipBootstrapResult:
        """One-shot per-chip bootstrap: build communicator, slice window,
        stage inputs from host shared memory, and (optionally) publish the result.

        The target device must already be attached via ``init(bins, device_id)``
        before invoking this method; ``device_id`` is supplied here only to
        catch a caller that wired up the wrong device on the wrong worker.

        Runs inside a forked chip child.  If ``channel`` is provided (the
        Worker-orchestrated integration path), the result is written as
        SUCCESS or — on any exception — as ERROR (code=1,
        ``"<ExceptionType>: <message>"``) before the exception is re-raised.
        Standalone callers can pass ``channel=None`` and consume the return
        value directly.

        The HCCL comm handle produced by ``comm_init`` is stashed on
        ``self._comm_handle`` so ``shutdown_bootstrap()`` can release it later;
        ``finalize()`` is intentionally *not* wired to this handle — teardown
        ordering is the caller's responsibility.
        """
        try:
            # Validate host-staging symmetry up-front — before any device or
            # communicator state is touched — so a missing staging entry
            # surfaces as a clean ValueError on the channel rather than a
            # KeyError from deep inside the flush/H2D loop (which would leave
            # the parent waiting on a silent chip child).
            for spec in cfg.buffers:
                if spec.load_from_host:
                    try:
                        cfg.input_staging(spec.name)
                    except KeyError:
                        raise ValueError(
                            f"ChipBufferSpec(name={spec.name!r}, load_from_host=True) requires a "
                            f"matching HostBufferStaging in host_inputs; none found"
                        ) from None
                if spec.store_to_host:
                    try:
                        cfg.output_staging(spec.name)
                    except KeyError:
                        raise ValueError(
                            f"ChipBufferSpec(name={spec.name!r}, store_to_host=True) requires a "
                            f"matching HostBufferStaging in host_outputs; none found"
                        ) from None

            if self.device_id != device_id:
                raise RuntimeError(
                    f"bootstrap_context(device_id={device_id}) called on a ChipWorker "
                    f"already initialized for device_id={self.device_id}"
                )

            device_ctx = 0
            local_base = 0
            actual_size = 0
            if cfg.comm is not None:
                handle = self.comm_init(cfg.comm.rank, cfg.comm.nranks, cfg.comm.rootinfo_path)
                if handle == 0:
                    raise RuntimeError(f"comm_init returned 0 handle (rank={cfg.comm.rank}, nranks={cfg.comm.nranks})")
                self._comm_handle = handle
                device_ctx = self.comm_alloc_windows(handle, cfg.comm.window_size)
                if device_ctx == 0:
                    raise RuntimeError("comm_alloc_windows returned null device_ctx")
                local_base = self.comm_get_local_window_base(handle)
                actual_size = self.comm_get_window_size(handle)

            offset = 0
            buffer_ptrs: list[int] = []
            for spec in cfg.buffers:
                if cfg.comm is None:
                    raise ValueError("ChipBufferSpec requires comm; cfg.comm is None")
                if offset + spec.nbytes > actual_size:
                    raise ValueError(
                        f"buffer '{spec.name}' (nbytes={spec.nbytes}) at offset={offset} "
                        f"overflows window size {actual_size}"
                    )
                buffer_ptrs.append(local_base + offset)
                offset += spec.nbytes

            for spec, ptr in zip(cfg.buffers, buffer_ptrs):
                if not spec.load_from_host:
                    continue
                staging = cfg.input_staging(spec.name)
                if staging.size != spec.nbytes:
                    raise ValueError(f"host_inputs[{spec.name!r}].size={staging.size} != buffer.nbytes={spec.nbytes}")
                if staging.size == 0:
                    continue
                shm = SharedMemory(name=staging.shm_name)
                try:
                    buf = shm.buf
                    assert buf is not None
                    host_ptr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
                    self.copy_to(ptr, host_ptr, staging.size)
                finally:
                    shm.close()

            result = ChipBootstrapResult(
                device_ctx=device_ctx,
                local_window_base=local_base,
                actual_window_size=actual_size,
                buffer_ptrs=buffer_ptrs,
            )
            if channel is not None:
                channel.write_success(
                    result.device_ctx,
                    result.local_window_base,
                    result.actual_window_size,
                    result.buffer_ptrs,
                )
            return result
        except Exception as e:
            if channel is not None:
                channel.write_error(1, f"{type(e).__name__}: {e}")
            raise

    def shutdown_bootstrap(self) -> None:
        """Release the communicator handle stashed by ``bootstrap_context``.

        Idempotent — safe to call multiple times, and safe to call if
        ``bootstrap_context`` was never invoked.  ``finalize()`` does *not*
        chain into this method, so callers (e.g. the Worker's chip child
        loop) must call ``shutdown_bootstrap()`` before ``finalize()`` (or
        after, if the comm handle was already destroyed — the zero-handle
        guard makes a second call a no-op).
        """
        handle = getattr(self, "_comm_handle", 0)
        if handle != 0:
            try:
                self.comm_destroy(handle)
            finally:
                self._comm_handle = 0

    @property
    def device_id(self):
        return self._impl.device_id

    @property
    def initialized(self):
        return self._impl.initialized
