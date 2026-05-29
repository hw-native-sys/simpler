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
import os
from dataclasses import dataclass

from _task_interface import (  # pyright: ignore[reportMissingImports]
    CONTINUOUS_TENSOR_MAX_DIMS,
    MAILBOX_ERROR_MSG_SIZE,
    MAILBOX_OFF_ERROR_MSG,
    MAILBOX_SIZE,
    ArgDirection,
    CallConfig,
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
    "SubmitResult",
    "_Worker",
    "MAILBOX_SIZE",
    "MAILBOX_OFF_ERROR_MSG",
    "MAILBOX_ERROR_MSG_SIZE",
    "read_args_from_blob",
    # Dynamic CommDomain allocation (orch-only API)
    "CommBufferSpec",
    "ChipDomainContext",
    "CommDomainHandle",
]

COMM_MAX_RANK_NUM = 64


class _CommContextStruct(ctypes.Structure):
    _fields_ = [
        ("workSpace", ctypes.c_uint64),
        ("workSpaceSize", ctypes.c_uint64),
        ("rankId", ctypes.c_uint32),
        ("rankNum", ctypes.c_uint32),
        ("winSize", ctypes.c_uint64),
        ("windowsIn", ctypes.c_uint64 * COMM_MAX_RANK_NUM),
        ("windowsOut", ctypes.c_uint64 * COMM_MAX_RANK_NUM),
    ]


assert ctypes.sizeof(_CommContextStruct) == 1056


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
class CommBufferSpec:
    """A named slice of the per-rank communicator window.

    Buffers are placed sequentially inside the window in declaration order —
    Buffers are placed sequentially inside the window in declaration order.
    The ``CommDomainHandle.contexts[chip_idx].buffer_ptrs`` dict returned by
    ``Orchestrator.allocate_domain`` is keyed by ``CommBufferSpec.name``.
    """

    name: str
    dtype: str
    count: int
    nbytes: int
    load_from_host: bool = False
    store_to_host: bool = False


@dataclass
class ChipDomainContext:
    name: str
    domain_rank: int
    domain_size: int
    device_ctx: int
    local_window_base: int
    actual_window_size: int
    buffer_ptrs: dict[str, int]


class CommDomainHandle:
    """User-facing handle for one dynamically-allocated CommDomain.

    Returned by ``Orchestrator.allocate_domain(...)``.  Acts as a context
    manager: ``with`` exit *marks* the handle for release and prevents
    further use; the actual backend free runs **after** ``Worker.run`` has
    drained any tasks the orch function submitted using this domain.  This
    is required because ``submit_*`` only enqueues to the DAG — freeing
    before drain would create a use-after-free on the chip side.

    Lifecycle states::

        live           — allocated, indexable, can be passed to submit_*
        released       — release() called; further indexing raises;
                          backend memory still alive until Worker.run drain
        freed          — backend release_domain has executed, memory gone

    Most users only see ``released``; the ``live → released`` transition
    happens at ``with`` exit (or explicit ``release()``), and the
    ``released → freed`` transition is the runtime's job at end-of-run.
    """

    __slots__ = ("name", "workers", "contexts", "allocation_id", "_release_fn", "_released", "_freed")

    def __init__(
        self,
        *,
        name: str,
        workers: tuple[int, ...],
        contexts: dict[int, "ChipDomainContext"],
        allocation_id: int,
        _release_fn,
    ) -> None:
        self.name = name
        self.workers = tuple(workers)
        # Frozen dict-ish — we don't expose mutation
        self.contexts: dict[int, ChipDomainContext] = dict(contexts)
        self.allocation_id = int(allocation_id)
        self._release_fn = _release_fn
        self._released = False
        self._freed = False

    def __getitem__(self, chip_idx: int) -> "ChipDomainContext":
        if self._released:
            raise RuntimeError(
                f"CommDomainHandle({self.name!r}) already released; do not pass it to submit_* "
                "after release(). Submitted tasks that captured device_ctx / buffer_ptrs before "
                "release will still see live memory until Worker.run drains."
            )
        return self.contexts[chip_idx]

    @property
    def released(self) -> bool:
        """True once ``release()`` (or ``with`` exit) has been called.

        Backend memory may still be alive — it is freed by the Worker after
        DAG drain at end-of-run.  Use this to gate further indexing /
        submission, not to assert physical teardown (use ``freed`` for that).
        """
        return self._released

    @property
    def freed(self) -> bool:
        """True once the backend ``comm_release_domain_windows`` has executed.

        Only flips after the owning ``Worker.run`` drains and processes the
        pending-release queue.  An ``orch_fn`` will never observe ``True``
        for a handle it released within the same ``run`` call.
        """
        return self._freed

    def release(self) -> None:
        """Mark this handle for collective release.  Idempotent.

        Inside an orch function, this is a non-blocking mark — the actual
        backend ``comm_release_domain_windows`` runs after
        ``Worker.run.drain()`` so that any tasks already submitted with
        this domain's ``device_ctx`` see live memory through execution.

        After this returns, the handle is treated as released for the
        user's purposes: ``__getitem__`` raises, repeated ``release()`` is
        a no-op, and the orch function must not pass it to further
        ``submit_*`` calls.
        """
        if self._released:
            return
        self._released = True
        # _release_fn is owned by Worker; it queues the actual backend
        # release and runs it after drain.  Worker also flips _freed.
        self._release_fn(self)

    def __enter__(self) -> "CommDomainHandle":
        return self

    def __exit__(self, *_):
        self.release()

    def __repr__(self) -> str:
        if self._freed:
            state = "freed"
        elif self._released:
            state = "released-pending-free"
        else:
            state = "live"
        return f"CommDomainHandle(name={self.name!r}, workers={self.workers}, {state})"


# Process-wide RTLD_GLOBAL preload registry. host_runtime.so resolves its
# undefined HostLogger / unified_log_* (and, on sim, sim_context_*) symbols
# against these globals, so they must be loaded — exactly once — before any
# host_runtime.so dlopen. Keyed by path; mirrors the C++ side's old
# std::once_flag semantics. Never closed.
_preloaded_globals: dict[str, ctypes.CDLL] = {}


def _preload_global(path: str) -> ctypes.CDLL:
    """dlopen `path` with RTLD_NOW | RTLD_GLOBAL, idempotently (one CDLL per path).

    Eager resolution (RTLD_NOW) mirrors the previous C++ dlopen flags and
    surfaces any missing-symbol problem at load time rather than first use.
    """
    handle = _preloaded_globals.get(path)
    if handle is None:
        handle = ctypes.CDLL(path, mode=os.RTLD_NOW | os.RTLD_GLOBAL)
        _preloaded_globals[path] = handle
    return handle


class ChipWorker:
    """Unified execution interface wrapping the host runtime C API.

    The runtime library and target device are bound once via init() and
    cannot be changed.
    This is the low-level direct chip execution API; its integer
    ``callable_id`` slots are private to the local ``ChipWorker`` instance and
    are separate from hierarchical ``Worker.register`` / ``CallableHandle``.

    Usage::

        worker = ChipWorker()
        worker.init(device_id=0, bins=bins)
        worker.prepare_callable(callable_id=0, callable=chip_callable)
        worker.run(callable_id=0, args=orch_args, config=CallConfig())  # block_dim defaults to 0 = auto
        worker.unregister_callable(callable_id=0)
        worker.finalize()
    """

    def __init__(self):
        self._impl = _ChipWorker()

    def init(self, device_id, bins, log_level=None, log_info_v=None):
        """Attach the calling thread to ``device_id``, load the host runtime
        library, and cache platform binaries.

        Can only be called once — the runtime and device cannot be changed
        after init.

        Performs the process-wide RTLD_GLOBAL bootstrap (libsimpler_log.so,
        plus libcpu_sim_context.so on sim platforms) and seeds the HostLogger
        via ``simpler_log_init`` *before* the C++ ``_ChipWorker.init`` dlopens
        host_runtime.so — host_runtime.so resolves its undefined HostLogger /
        unified_log_* (and, on sim, sim_context_*) symbols against those
        globals, and any LOG_* macro firing during its dlopen-time
        constructors must already see the right filter.

        Args:
            device_id: NPU device ID to attach the calling thread to.
            bins: A `simpler_setup.runtime_builder.RuntimeBinaries` (or any
                object exposing host_path / aicpu_path / aicore_path /
                simpler_log_path / sim_context_path / dispatcher_path).
                ``dispatcher_path`` is required for onboard platforms and
                ignored on sim (set to None).
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

        # 1. libsimpler_log.so — RTLD_GLOBAL singleton, before host_runtime.so.
        if not bins.simpler_log_path:
            raise ValueError("ChipWorker.init: bins.simpler_log_path is required")
        log_handle = _preload_global(str(bins.simpler_log_path))
        log_handle.simpler_log_init.argtypes = [ctypes.c_int, ctypes.c_int]
        log_handle.simpler_log_init.restype = ctypes.c_int
        rc = log_handle.simpler_log_init(int(log_level), int(log_info_v))
        if rc != 0:
            raise RuntimeError(f"simpler_log_init failed with code {rc}")

        # 2. libcpu_sim_context.so — sim platforms only (host_runtime.so's sim
        #    variant resolves sim_context_set_* / pto_sim_get_* against it).
        if bins.sim_context_path:
            _preload_global(str(bins.sim_context_path))

        # 3. host_runtime.so is dlopen'd RTLD_LOCAL inside _impl.init.
        #    dispatcher_path is passed as an empty string on sim (where bins
        #    has dispatcher_path=None); the onboard simpler_init reads it
        #    via LoadAicpuOp::BootstrapDispatcher, sim ignores it.
        dispatcher_path = getattr(bins, "dispatcher_path", None)
        self._impl.init(
            str(bins.host_path),
            str(bins.aicpu_path),
            str(bins.aicore_path),
            "" if dispatcher_path is None else str(dispatcher_path),
            int(device_id),
        )

    def finalize(self):
        """Tear down everything: device resources and runtime library.

        Terminal operation — the object cannot be reused after this.
        """
        self._impl.finalize()

    def prepare_callable(self, callable_id, callable):
        """Stage a ChipCallable under ``callable_id`` for repeated cheap launches.

        Uploads the kernel binaries + the orchestration SO once; subsequent
        ``run(callable_id, ...)`` skips that work. ``callable_id``
        must be in ``[0, 64)``. Requires ``init()``.
        """
        self._impl.prepare_callable(int(callable_id), callable)

    def run(self, callable_id, args, config=None, **kwargs):
        """Launch a ``callable_id`` previously staged via ``prepare_callable``.

        Args:
            callable_id: Stable id passed to a prior ``prepare_callable``.
            args: ChipStorageTaskArgs for this invocation.
            config: Optional CallConfig. If None, a default is created.
            **kwargs: Overrides applied to config (e.g. ``block_dim=8`` to
                pin a smaller value than the default). Omit ``block_dim`` (or
                set it to 0) to have DeviceRunner auto-resolve it to the max
                the AICore stream allows (``aclrtGetStreamResLimit`` on
                onboard, ``PLATFORM_MAX_BLOCKDIM`` on sim).

        Returns a :class:`RunTiming` with host + device wall.
        """
        if config is None:
            config = CallConfig()
        for k, v in kwargs.items():
            setattr(config, k, v)
        return self._impl.run(int(callable_id), args, config)

    def unregister_callable(self, callable_id):
        """Drop prepared state for ``callable_id`` and release its orch SO share."""
        self._impl.unregister_callable(int(callable_id))

    @property
    def aicpu_dlopen_count(self):
        """Number of distinct callable_ids the AICPU has dlopened for."""
        return self._impl.aicpu_dlopen_count

    @property
    def host_dlopen_count(self):
        """Number of host-side orch SO dlopens (host_build_graph variants)."""
        return self._impl.host_dlopen_count

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

    def comm_derive_context(
        self,
        comm_handle: int,
        rank_ids: list[int],
        domain_rank: int,
        window_offset: int,
        window_size: int,
    ) -> int:
        """Derive a domain-local device CommContext from an allocated base communicator."""
        return int(
            self._impl.comm_derive_context(
                int(comm_handle),
                [int(x) for x in rank_ids],
                int(domain_rank),
                int(window_offset),
                int(window_size),
            )
        )

    def comm_barrier(self, comm_handle: int) -> None:
        """Synchronize all ranks."""
        self._impl.comm_barrier(int(comm_handle))

    def comm_destroy(self, comm_handle: int) -> None:
        """Destroy the communicator and release its resources."""
        self._impl.comm_destroy(int(comm_handle))

    def comm_destroy_all(self) -> None:
        """Destroy all communicators owned by this worker."""
        self._impl.comm_destroy_all()

    @property
    def device_id(self):
        return self._impl.device_id

    @property
    def initialized(self):
        return self._impl.initialized
