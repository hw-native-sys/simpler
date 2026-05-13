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
    "CommDomain",
    "CommDomainPlan",
    "ChipDomainBootstrapConfig",
    "ChipBufferSpec",
    "HostBufferStaging",
    "ChipBootstrapConfig",
    "ChipCommDomainContext",
    "ChipBootstrapResult",
    # Worker-level chip bootstrap orchestration
    "ChipContext",
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
    domain_name: Optional[str] = None


@dataclass
class CommDomain:
    """Parent-level communication domain specification.

    `worker_indices` are L3 child worker indices.  Their order defines dense
    domain ranks, so `worker_indices[0]` is domain rank 0.
    """

    name: str
    worker_indices: list[int]
    window_size: int
    buffers: list[ChipBufferSpec] = field(default_factory=list)


@dataclass
class ChipDomainBootstrapConfig:
    """Per-chip derived domain config consumed by `bootstrap_context`."""

    name: str
    sub_comm_id: int
    domain_rank: int
    domain_size: int
    rank_ids: list[int]
    window_size: int
    window_offset: int = 0
    base_window_size: int = 0
    buffers: list[ChipBufferSpec] = field(default_factory=list)
    host_inputs: list[HostBufferStaging] = field(default_factory=list)
    host_outputs: list[HostBufferStaging] = field(default_factory=list)

    def input_staging(self, buffer_name: str) -> HostBufferStaging:
        for s in self.host_inputs:
            if s.name == buffer_name and (s.domain_name is None or s.domain_name == self.name):
                return s
        raise KeyError(buffer_name)

    def output_staging(self, buffer_name: str) -> HostBufferStaging:
        for s in self.host_outputs:
            if s.name == buffer_name and (s.domain_name is None or s.domain_name == self.name):
                return s
        raise KeyError(buffer_name)


@dataclass
class CommDomainPlan:
    """L3-level source of truth for all communication domains."""

    domains: list[CommDomain] = field(default_factory=list)

    def validate(self, *, worker_count: int) -> None:
        seen_names: set[str] = set()
        for domain in self.domains:
            if not domain.name:
                raise ValueError("CommDomain.name must be non-empty")
            if domain.name in seen_names:
                raise ValueError(f"duplicate communication domain name {domain.name!r}")
            seen_names.add(domain.name)
            if not domain.worker_indices:
                raise ValueError(f"CommDomain({domain.name!r}) worker_indices must be non-empty")
            if len(set(domain.worker_indices)) != len(domain.worker_indices):
                raise ValueError(f"CommDomain({domain.name!r}) worker_indices contains duplicates")
            for idx in domain.worker_indices:
                if idx < 0 or idx >= worker_count:
                    raise ValueError(
                        f"CommDomain({domain.name!r}) worker index {idx} outside [0, {worker_count})"
                    )
            if len(domain.worker_indices) > COMM_MAX_RANK_NUM:
                raise ValueError(
                    f"CommDomain({domain.name!r}) size {len(domain.worker_indices)} exceeds "
                    f"COMM_MAX_RANK_NUM={COMM_MAX_RANK_NUM}"
                )
            if domain.window_size <= 0:
                raise ValueError(f"CommDomain({domain.name!r}) window_size must be positive")
            buffer_names = [b.name for b in domain.buffers]
            if len(set(buffer_names)) != len(buffer_names):
                raise ValueError(f"CommDomain({domain.name!r}) buffers contain duplicate names")

    def window_offsets(self) -> dict[str, int]:
        offsets: dict[str, int] = {}
        offset = 0
        for domain in sorted(self.domains, key=lambda d: d.name):
            offsets[domain.name] = offset
            offset += domain.window_size
        return offsets

    def base_window_size(self) -> int:
        return sum(domain.window_size for domain in self.domains)

    def bootstrap_for_worker(self, worker_idx: int) -> list[ChipDomainBootstrapConfig]:
        sub_comm_ids = {name: idx for idx, name in enumerate(sorted(d.name for d in self.domains))}
        window_offsets = self.window_offsets()
        base_window_size = self.base_window_size()
        configs: list[ChipDomainBootstrapConfig] = []
        for domain in self.domains:
            if worker_idx not in domain.worker_indices:
                continue
            rank = domain.worker_indices.index(worker_idx)
            configs.append(
                ChipDomainBootstrapConfig(
                    name=domain.name,
                    sub_comm_id=sub_comm_ids[domain.name],
                    domain_rank=rank,
                    domain_size=len(domain.worker_indices),
                    rank_ids=list(domain.worker_indices),
                    window_size=domain.window_size,
                    window_offset=window_offsets[domain.name],
                    base_window_size=base_window_size,
                    buffers=list(domain.buffers),
                )
            )
        configs.sort(key=lambda c: c.name)
        return configs


@dataclass
class ChipBootstrapConfig:
    """Inputs to `ChipWorker.bootstrap_context` for one chip child."""

    comm: Optional[list[ChipDomainBootstrapConfig]] = None
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

    def domain_bootstrap_configs(self) -> list[ChipDomainBootstrapConfig]:
        if self.comm is None:
            return []
        if not isinstance(self.comm, list):
            raise TypeError("ChipBootstrapConfig.comm must be a list of ChipDomainBootstrapConfig or None")

        for idx, domain in enumerate(self.comm):
            if not isinstance(domain, ChipDomainBootstrapConfig):
                raise TypeError(f"ChipBootstrapConfig.comm[{idx}] must be ChipDomainBootstrapConfig")
        domains = [self._attach_staging_to_domain(domain) for domain in self.comm]
        self._validate_staging_consumed(domains, self.host_inputs, label="host_inputs")
        self._validate_staging_consumed(domains, self.host_outputs, label="host_outputs")
        return domains

    def _attach_staging_to_domain(self, domain: ChipDomainBootstrapConfig) -> ChipDomainBootstrapConfig:
        host_inputs = list(domain.host_inputs) + self._domain_staging(self.host_inputs, domain.name)
        host_outputs = list(domain.host_outputs) + self._domain_staging(self.host_outputs, domain.name)
        return ChipDomainBootstrapConfig(
            name=domain.name,
            sub_comm_id=domain.sub_comm_id,
            domain_rank=domain.domain_rank,
            domain_size=domain.domain_size,
            rank_ids=list(domain.rank_ids),
            window_size=domain.window_size,
            window_offset=domain.window_offset,
            base_window_size=domain.base_window_size,
            buffers=list(domain.buffers),
            host_inputs=host_inputs,
            host_outputs=host_outputs,
        )

    @staticmethod
    def _domain_staging(items: list[HostBufferStaging], domain_name: str) -> list[HostBufferStaging]:
        return [s for s in items if s.domain_name == domain_name]

    @staticmethod
    def _validate_staging_consumed(
        domains: list[ChipDomainBootstrapConfig],
        items: list[HostBufferStaging],
        *,
        label: str,
    ) -> None:
        domain_names = {d.name for d in domains}
        seen: set[tuple[str, str]] = set()
        for s in items:
            if not s.domain_name:
                raise ValueError(f"{label} entry {s.name!r} requires domain_name in explicit ChipBootstrapConfig")
            if s.domain_name not in domain_names:
                raise ValueError(
                    f"{label} entry {s.name!r} has domain_name={s.domain_name!r}, "
                    "but this chip config has no such domain"
                )
            key = (s.domain_name, s.name)
            if key in seen:
                raise ValueError(f"duplicate {label} staging entry for {key}")
            seen.add(key)


@dataclass
class ChipCommDomainContext:
    name: str
    domain_rank: int
    domain_size: int
    device_ctx: int
    local_window_base: int
    actual_window_size: int
    buffer_ptrs: dict[str, int]


@dataclass
class ChipBootstrapResult:
    """Return value of `ChipWorker.bootstrap_context`."""

    domains: dict[str, ChipCommDomainContext] = field(default_factory=dict)


@dataclass
class ChipContext:
    """Per-chip view of a successful bootstrap, exposed to L3+ orch functions.

    Built by the parent `Worker` in `_start_hierarchical` from the
    `ChipBootstrapConfig` it forwarded to the chip child and the
    `ChipBootstrapResult` the child published via its `ChipBootstrapChannel`.
    Orchestration code addresses communication state through
    `domains[domain_name]`.
    """

    device_id: int
    worker_index: int
    domains: dict[str, ChipCommDomainContext] = field(default_factory=dict)


class ChipWorker:
    """Unified execution interface wrapping the host runtime C API.

    The runtime library and target device are bound once via init() and
    cannot be changed.

    Usage::

        worker = ChipWorker()
        worker.init(device_id=0, bins=bins)
        worker.prepare_callable(callable_id=0, callable=chip_callable)
        worker.run_prepared(callable_id=0, args=orch_args, config=CallConfig(block_dim=24))
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

    def prepare_callable(self, callable_id, callable):
        """Stage a ChipCallable under ``callable_id`` for repeated cheap launches.

        Uploads the kernel binaries + the orchestration SO once; subsequent
        ``run_prepared(callable_id, ...)`` skips that work. ``callable_id``
        must be in ``[0, 64)``. Requires ``init()``.
        """
        self._impl.prepare_callable(int(callable_id), callable)

    def run_prepared(self, callable_id, args, config=None, **kwargs):
        """Launch a ``callable_id`` previously staged via ``prepare_callable``.

        Args:
            callable_id: Stable id passed to a prior ``prepare_callable``.
            args: ChipStorageTaskArgs for this invocation.
            config: Optional CallConfig. If None, a default is created.
            **kwargs: Overrides applied to config (e.g. block_dim=24).
        """
        if config is None:
            config = CallConfig()
        for k, v in kwargs.items():
            setattr(config, k, v)
        self._impl.run_prepared(int(callable_id), args, config)

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

    def comm_create_subcomm(
        self,
        comm_handle: int,
        sub_comm_id: int,
        rank_ids: list[int],
        sub_comm_rank_id: int,
    ) -> int:
        """Create a domain communicator from a hidden base communicator."""
        return int(
            self._impl.comm_create_subcomm(
                int(comm_handle),
                int(sub_comm_id),
                [int(x) for x in rank_ids],
                int(sub_comm_rank_id),
            )
        )

    def comm_create_domain(
        self,
        sub_comm_id: int,
        rank_ids: list[int],
        sub_comm_rank_id: int,
    ) -> int:
        """Create a domain communicator from the ChipWorker-owned base communicator."""
        return int(
            self._impl.comm_create_domain(
                int(sub_comm_id),
                [int(x) for x in rank_ids],
                int(sub_comm_rank_id),
            )
        )

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

        Communication handles produced by bootstrap are stashed on
        ``self._comm_handles`` so ``shutdown_bootstrap()`` can release them
        later; ``finalize()`` is intentionally *not* wired to these handles — teardown
        ordering is the caller's responsibility.
        """
        try:
            domain_cfgs = self._domain_bootstrap_configs(cfg)
            # Validate host-staging symmetry up-front — before any device or
            # communicator state is touched — so a missing staging entry
            # surfaces as a clean ValueError on the channel rather than a
            # KeyError from deep inside the flush/H2D loop (which would leave
            # the parent waiting on a silent chip child).
            for domain in domain_cfgs:
                for spec in domain.buffers:
                    if spec.load_from_host:
                        try:
                            domain.input_staging(spec.name)
                        except KeyError:
                            raise ValueError(
                                f"ChipBufferSpec(domain={domain.name!r}, name={spec.name!r}, "
                                "load_from_host=True) requires matching HostBufferStaging in host_inputs; none found"
                            ) from None
                    if spec.store_to_host:
                        try:
                            domain.output_staging(spec.name)
                        except KeyError:
                            raise ValueError(
                                f"ChipBufferSpec(domain={domain.name!r}, name={spec.name!r}, "
                                "store_to_host=True) requires matching HostBufferStaging in host_outputs; none found"
                            ) from None

            if self.device_id != device_id:
                raise RuntimeError(
                    f"bootstrap_context(device_id={device_id}) called on a ChipWorker "
                    f"already initialized for device_id={self.device_id}"
                )
            handles: list[int] = []
            self._comm_handles = handles
            domains: dict[str, ChipCommDomainContext] = {}
            if cfg.comm == []:
                base_rank, base_size, rootinfo_path = self._base_comm_params(cfg)
                base = self.comm_init(base_rank, base_size, rootinfo_path)
                if base == 0:
                    raise RuntimeError("comm_init returned 0 handle for hidden base communicator")
                handles.append(base)
                base_window_size = getattr(cfg, "base_window_size", 0)
                if base_window_size:
                    base_device_ctx = self.comm_alloc_windows(base, int(base_window_size))
                    if base_device_ctx == 0:
                        raise RuntimeError("comm_alloc_windows returned null device_ctx for hidden base communicator")
                    base_local = self.comm_get_local_window_base(base)
                    actual_base_size = self.comm_get_window_size(base)
                    self._zero_device_memory(base_local, actual_base_size)
            elif domain_cfgs:
                base_rank, base_size, rootinfo_path = self._base_comm_params(cfg)
                base = self.comm_init(base_rank, base_size, rootinfo_path)
                if base == 0:
                    raise RuntimeError("comm_init returned 0 handle for hidden base communicator")
                handles.append(base)
                base_window_size = self._base_window_size(domain_cfgs)
                if base_window_size <= 0:
                    raise ValueError("multi-domain base window size must be positive")
                base_device_ctx = self.comm_alloc_windows(base, base_window_size)
                if base_device_ctx == 0:
                    raise RuntimeError("comm_alloc_windows returned null device_ctx for hidden base communicator")
                base_local = self.comm_get_local_window_base(base)
                actual_base_size = self.comm_get_window_size(base)
                self._zero_device_memory(base_local, actual_base_size)
                for domain in domain_cfgs:
                    self._validate_domain_window(domain, actual_base_size)
                    device_ctx = self.comm_derive_context(
                        base,
                        domain.rank_ids,
                        domain.domain_rank,
                        domain.window_offset,
                        domain.window_size,
                    )
                    local_base = base_local + domain.window_offset
                    actual_size = domain.window_size
                    buffer_ptrs = self._carve_domain_buffers(domain, local_base, actual_size)
                    self._stage_domain_inputs(domain, buffer_ptrs)
                    domains[domain.name] = ChipCommDomainContext(
                        name=domain.name,
                        domain_rank=domain.domain_rank,
                        domain_size=domain.domain_size,
                        device_ctx=device_ctx,
                        local_window_base=local_base,
                        actual_window_size=actual_size,
                        buffer_ptrs=buffer_ptrs,
                    )
            result = ChipBootstrapResult(domains=domains)
            if channel is not None:
                if hasattr(channel, "write_success_domains"):
                    from _task_interface import ChipBootstrapDomainResult  # pyright: ignore[reportMissingImports]

                    channel.write_success_domains(
                        [
                            ChipBootstrapDomainResult(
                                d.name,
                                d.domain_rank,
                                d.domain_size,
                                d.device_ctx,
                                d.local_window_base,
                                d.actual_window_size,
                                list(d.buffer_ptrs.values()),
                            )
                            for d in result.domains.values()
                        ]
                    )
                else:
                    raise RuntimeError("domain-aware ChipBootstrapChannel is required")
            return result
        except Exception as e:
            if channel is not None:
                channel.write_error(1, f"{type(e).__name__}: {e}")
            raise

    @staticmethod
    def _rootinfo_path(cfg: ChipBootstrapConfig) -> str:
        # Worker-derived multi-domain configs attach this private attribute.
        return getattr(cfg, "rootinfo_path", "")

    @staticmethod
    def _base_comm_params(cfg: ChipBootstrapConfig) -> tuple[int, int, str]:
        rank = getattr(cfg, "base_rank", None)
        size = getattr(cfg, "base_size", None)
        rootinfo_path = getattr(cfg, "rootinfo_path", None)
        if rank is None or size is None or rootinfo_path is None:
            raise ValueError("multi-domain ChipBootstrapConfig requires base_rank, base_size, and rootinfo_path")
        return int(rank), int(size), str(rootinfo_path)

    @staticmethod
    def _domain_bootstrap_configs(cfg: ChipBootstrapConfig) -> list[ChipDomainBootstrapConfig]:
        return cfg.domain_bootstrap_configs()

    @staticmethod
    def _base_window_size(domains: list[ChipDomainBootstrapConfig]) -> int:
        declared = {d.base_window_size for d in domains if d.base_window_size > 0}
        if len(declared) > 1:
            raise ValueError(f"inconsistent base_window_size values: {sorted(declared)}")
        if declared:
            return declared.pop()
        return max((d.window_offset + d.window_size for d in domains), default=0)

    @staticmethod
    def _validate_domain_window(domain: ChipDomainBootstrapConfig, actual_base_size: int) -> None:
        if domain.domain_size <= 0 or domain.domain_size > COMM_MAX_RANK_NUM:
            raise ValueError(f"domain {domain.name!r} has invalid size {domain.domain_size}")
        if len(domain.rank_ids) != domain.domain_size:
            raise ValueError(f"domain {domain.name!r} rank_ids length does not match domain_size")
        if domain.domain_rank < 0 or domain.domain_rank >= domain.domain_size:
            raise ValueError(f"domain {domain.name!r} has invalid domain_rank {domain.domain_rank}")
        if domain.window_offset < 0:
            raise ValueError(f"domain {domain.name!r} window_offset must be non-negative")
        if domain.window_size <= 0:
            raise ValueError(f"domain {domain.name!r} window_size must be positive")
        if domain.window_offset + domain.window_size > actual_base_size:
            raise ValueError(
                f"domain {domain.name!r} window range "
                f"[{domain.window_offset}, {domain.window_offset + domain.window_size}) "
                f"overflows base window size {actual_base_size}"
            )

    @staticmethod
    def _carve_domain_buffers(
        domain: ChipDomainBootstrapConfig,
        local_base: int,
        actual_size: int,
    ) -> dict[str, int]:
        offset = 0
        buffer_ptrs: dict[str, int] = {}
        for spec in domain.buffers:
            if offset + spec.nbytes > actual_size:
                raise ValueError(
                    f"domain {domain.name!r} buffer {spec.name!r} (nbytes={spec.nbytes}) at offset={offset} "
                    f"overflows window size {actual_size}"
                )
            buffer_ptrs[spec.name] = local_base + offset
            offset += spec.nbytes
        return buffer_ptrs

    def _stage_domain_inputs(
        self,
        domain: ChipDomainBootstrapConfig,
        buffer_ptrs: dict[str, int],
    ) -> None:
        for spec in domain.buffers:
            if not spec.load_from_host:
                continue
            staging = domain.input_staging(spec.name)
            if staging.size != spec.nbytes:
                raise ValueError(
                    f"host_inputs[{domain.name!r}, {spec.name!r}].size={staging.size} "
                    f"!= buffer.nbytes={spec.nbytes}"
                )
            if staging.size == 0:
                continue
            shm = SharedMemory(name=staging.shm_name)
            try:
                buf = shm.buf
                assert buf is not None
                host_ptr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
                self.copy_to(buffer_ptrs[spec.name], host_ptr, staging.size)
            finally:
                shm.close()

    def _zero_device_memory(self, dev_ptr: int, nbytes: int) -> None:
        if nbytes <= 0:
            return
        zeros = (ctypes.c_char * nbytes)()
        self.copy_to(dev_ptr, ctypes.addressof(zeros), nbytes)

    def shutdown_bootstrap(self) -> None:
        """Release the communicator handle stashed by ``bootstrap_context``.

        Idempotent — safe to call multiple times, and safe to call if
        ``bootstrap_context`` was never invoked.  ``finalize()`` does *not*
        chain into this method, so callers (e.g. the Worker's chip child
        loop) must call ``shutdown_bootstrap()`` before ``finalize()`` (or
        after, if the comm handle was already destroyed — the zero-handle
        guard makes a second call a no-op).
        """
        handles = list(getattr(self, "_comm_handles", []))
        first_error: BaseException | None = None
        for handle in reversed(handles):
            if handle == 0:
                continue
            try:
                self.comm_destroy(handle)
            except BaseException as e:  # noqa: BLE001
                if first_error is None:
                    first_error = e
        self._comm_handles = []
        if first_error is not None:
            raise first_error

    @property
    def device_id(self):
        return self._impl.device_id

    @property
    def initialized(self):
        return self._impl.initialized
