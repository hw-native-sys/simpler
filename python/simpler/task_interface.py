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
import threading
import uuid
from dataclasses import dataclass
from typing import Any

from _task_interface import (  # pyright: ignore[reportMissingImports]
    CONTINUOUS_TENSOR_MAX_DIMS,
    MAILBOX_ERROR_MSG_SIZE,
    MAILBOX_OFF_ERROR_MSG,
    MAILBOX_SIZE,
    MAX_REGISTERED_CALLABLE_IDS,
    ArgDirection,
    CallConfig,
    ChipCallable,
    ChipStorageTaskArgs,
    ContinuousTensor,
    CoreCallable,
    DataType,
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
    Public dispatch uses opaque ``CallableHandle`` values. Integer execution
    slots are private to this wrapper and the runtime ABI.

    Usage::

        worker = ChipWorker()
        worker.init(device_id=0, bins=bins)
        handle = worker.prepare_callable(chip_callable)
        worker.run(handle, args=orch_args, config=CallConfig())  # block_dim defaults to 0 = auto
        worker.unregister_callable(handle)
        worker.finalize()
    """

    def __init__(self):
        self._impl = _ChipWorker()
        self._owner_id = uuid.uuid4().hex
        self._registry_lock = threading.Lock()
        self._callable_registry: dict[int, ChipCallable] = {}
        self._identity_registry: dict[bytes, Any] = {}
        self._live_handles: dict[int, bytes] = {}
        self._next_handle_id = 0

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
        for slot_id, callable_obj in list(self._callable_registry.items()):
            self._impl.prepare_callable(int(slot_id), callable_obj)

    def finalize(self):
        """Tear down everything: device resources and runtime library.

        Terminal operation — the object cannot be reused after this.
        """
        self._impl.finalize()
        with self._registry_lock:
            self._callable_registry.clear()
            self._identity_registry.clear()
            self._live_handles.clear()

    def _allocate_slot_locked(self) -> int:
        for slot_id in range(MAX_REGISTERED_CALLABLE_IDS):
            if slot_id not in self._callable_registry:
                return slot_id
        raise RuntimeError(
            "ChipWorker.prepare_callable: callable capacity exhausted "
            f"(MAX_REGISTERED_CALLABLE_IDS={MAX_REGISTERED_CALLABLE_IDS})"
        )

    def _make_handle_locked(self, state):
        from .callable_identity import CallableHandle  # noqa: PLC0415

        handle_id = self._next_handle_id
        self._next_handle_id += 1
        self._live_handles[handle_id] = state.digest
        return CallableHandle._from_registration(
            hashid=state.hashid,
            kind=state.kind,
            target_namespace=state.target_namespace,
            handle_id=handle_id,
            owner_id=self._owner_id,
        )

    def _rollback_handle_locked(self, handle) -> None:
        state = self._identity_registry.get(handle.digest)
        self._live_handles.pop(handle._handle_id, None)
        if state is None:
            return
        state.ref_count -= 1
        if state.ref_count > 0:
            return
        self._callable_registry.pop(state.slot_id, None)
        self._identity_registry.pop(state.digest, None)

    def _resolve_handle_locked(self, handle):
        from .callable_identity import CallableHandle  # noqa: PLC0415

        if not isinstance(handle, CallableHandle):
            raise TypeError("ChipWorker.run expects a CallableHandle returned by ChipWorker.prepare_callable")
        if handle._owner_id != self._owner_id:
            raise KeyError(f"CallableHandle {handle.hashid} does not belong to this ChipWorker")
        digest = self._live_handles.get(handle._handle_id)
        if digest is None or digest != handle.digest:
            raise KeyError(f"CallableHandle {handle.hashid} is not live on this ChipWorker")
        state = self._identity_registry.get(digest)
        if state is None:
            raise KeyError(f"CallableHandle {handle.hashid} is not registered")
        if (
            handle.hashid != state.hashid
            or handle.kind != state.kind
            or handle.target_namespace != state.target_namespace
        ):
            raise RuntimeError(f"CALLABLE_HANDLE_MUTATED: {handle.hashid}")
        return state

    def _resolve_handle(self, handle):
        with self._registry_lock:
            return self._resolve_handle_locked(handle)

    def prepare_callable(self, callable):
        """Prepare a ``ChipCallable`` and return an opaque handle.

        The runtime still uses an integer slot internally, but the caller never
        chooses or observes it.
        """
        if not isinstance(callable, ChipCallable):
            raise TypeError("ChipWorker.prepare_callable only supports ChipCallable targets")
        from .callable_identity import (  # noqa: PLC0415
            _CallableIdentityState,
            build_chip_callable_descriptor,
            compute_callable_hashid,
            hashid_to_digest,
        )

        descriptor = build_chip_callable_descriptor(target=callable)
        hashid = compute_callable_hashid(descriptor)
        digest = hashid_to_digest(hashid)
        with self._registry_lock:
            state = self._identity_registry.get(digest)
            if state is not None:
                if state.descriptor != descriptor or state.kind != "CHIP_CALLABLE":
                    raise RuntimeError(f"HASHID_DESCRIPTOR_MISMATCH: {hashid}")
                state.ref_count += 1
                return self._make_handle_locked(state)
            slot_id = self._allocate_slot_locked()
            state = _CallableIdentityState(
                hashid=hashid,
                digest=digest,
                kind="CHIP_CALLABLE",
                target_namespace="LOCAL_CHIP",
                descriptor=descriptor,
                payload_digest=descriptor,
                slot_id=slot_id,
                target=callable,
                ref_count=1,
            )
            self._identity_registry[digest] = state
            self._callable_registry[slot_id] = callable
            handle = self._make_handle_locked(state)

        if self.initialized:
            try:
                self._impl.prepare_callable(int(slot_id), callable)
            except Exception:
                with self._registry_lock:
                    self._rollback_handle_locked(handle)
                raise
        return handle

    def run(self, handle, args, config=None, **kwargs):
        """Launch a callable previously returned by ``prepare_callable``.

        Args:
            handle: ``CallableHandle`` returned by ``prepare_callable``.
            args: ChipStorageTaskArgs for this invocation.
            config: Optional CallConfig. If None, a default is created.
            **kwargs: Overrides applied to config (e.g. ``block_dim=8`` to
                pin a smaller value than the default). Omit ``block_dim`` (or
                set it to 0) to have DeviceRunner auto-resolve it to the max
                the AICore stream allows (``aclrtGetStreamResLimit`` on
                onboard, ``PLATFORM_MAX_BLOCKDIM`` on sim).

        Returns a :class:`RunTiming` with host + device wall.
        """
        state = self._resolve_handle(handle)
        return self._run_slot(state.slot_id, args, config, **kwargs)

    def unregister_callable(self, handle) -> None:
        """Drop one live callable handle and release its private resources when final."""
        with self._registry_lock:
            state = self._resolve_handle_locked(handle)
            self._live_handles.pop(handle._handle_id, None)
            state.ref_count -= 1
            if state.ref_count > 0:
                return
            slot_id = state.slot_id
            self._callable_registry.pop(slot_id, None)
            self._identity_registry.pop(state.digest, None)

        if self.initialized:
            self._impl.unregister_callable(int(slot_id))

    def _prepare_callable_at_slot(self, callable_id, callable):
        self._impl.prepare_callable(int(callable_id), callable)

    def _run_slot(self, callable_id, args, config=None, **kwargs):
        if config is None:
            config = CallConfig()
        for k, v in kwargs.items():
            setattr(config, k, v)
        return self._impl.run(int(callable_id), args, config)

    def _unregister_slot(self, callable_id):
        self._impl.unregister_callable(int(callable_id))

    @property
    def aicpu_dlopen_count(self):
        """Number of distinct callable identities the AICPU has dlopened for."""
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
