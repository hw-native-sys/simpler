# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Worker — unified factory for all hierarchy levels.

Callable identity is exposed as an opaque ``CallableHandle`` returned by
``Worker.register(callable)``. L2 ``Worker.run`` and hierarchical
``Orchestrator.submit_next_level`` / ``submit_sub`` consume handles, never raw
``ChipCallable`` objects. L3+ ``Worker.run`` keeps the existing raw Python
orchestration-function entry point; that function captures handles and submits
them through the Orchestrator. L≥3 targets resolve the handle's stable SHA-256
digest to a private L2-side slot; later Python registrations are serialized
and sent through the mailbox control plane.

Usage::

    # L2: one NPU chip
    w = Worker(level=2, device_id=8, platform="a2a3", runtime="tensormap_and_ringbuffer")
    w.init()
    chip_handle = w.register(chip_callable)                 # L2 may register pre or post init()
    w.run(chip_handle, chip_args, config)
    w.close()

    # L3: multiple chips + SubWorkers, auto-discovery in init()
    w = Worker(level=3, device_ids=[8, 9], num_sub_workers=2,
               platform="a2a3", runtime="tensormap_and_ringbuffer")
    chip_handle = w.register(chip_callable)                 # ChipCallable, before init()
    sub_handle  = w.register(lambda args: postprocess())    # Python sub, before init()
    w.init()

    def my_orch(orch, args, cfg):
        r = orch.submit_next_level(chip_handle, chip_args_ptr, cfg)
        orch.submit_sub(sub_handle, sub_args)

    w.run(my_orch, my_args, my_config)
    w.close()

    # L4: recursive composition — L3 Workers as children
    l3 = Worker(level=3, device_ids=[8, 9], num_sub_workers=1,
                platform="a2a3", runtime="tensormap_and_ringbuffer")
    w4 = Worker(level=4, num_sub_workers=1)
    l3_handle = w4.register(my_l3_orch)
    verify_handle = w4.register(lambda args: verify())
    l3_worker_id = w4.add_worker(l3)
    w4.init()

    def my_l4_orch(orch, args, config):
        orch.submit_next_level(l3_handle, chip_args, config, worker=l3_worker_id)
        orch.submit_sub(verify_handle)

    w4.run(my_l4_orch)
    w4.close()
"""

from __future__ import annotations

import bisect
import contextlib
import ctypes
import enum
import importlib
import json
import math
import os
import re
import signal
import socket
import struct
import sys
import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory
from typing import Any, cast

import cloudpickle
from _task_interface import (  # pyright: ignore[reportMissingImports]
    MAX_REGISTERED_CALLABLE_IDS,
    RUNTIME_ENV_RING_COUNT,
    TENSOR_CHILD_MEMORY_OFFSET,
    WorkerType,
    _l3_child_onboard_region_close,
    _l3_child_onboard_region_create,
    _l3_host_mapped_region_import_onboard,
    _l3_host_mapped_region_import_sim,
    _mailbox_load_i32,
    _mailbox_store_i32,
    read_args_from_blob,
)

from . import _log as _simpler_log
from .callable_identity import (
    CALLABLE_HASH_DIGEST_BYTES,
    CallableHandle,
    _CallableIdentityState,
    build_chip_callable_descriptor,
    build_python_import_descriptor,
    build_python_serialized_descriptor,
    compute_callable_hashid,
    hashid_to_digest,
    parse_python_callable_payload,
    parse_python_import_target,
)
from .l3_l2_orch_comm import (
    _CTRL_SHM_TOKEN_BYTES,
    _REGION_CREATE_REPLY,
    _REGION_CREATE_REPLY_BYTES,
    _REGION_CREATE_REQUEST,
    _REGION_CREATE_REQUEST_BYTES,
    _REGION_LAYOUT_ALIGNMENT,
    _REGION_MAGIC_VERSION,
    L3HostRegionMapping,
    L3L2OrchRegion,
    L3L2RegionAccessProfile,
    L3L2RegionCreateRequest,
    _align_up,
    _checked_add_u64,
    decode_region_create_reply,
    peek_region_create_reply_region_id,
    validate_region_create_reply,
)
from .orchestrator import Orchestrator
from .task_interface import (
    MAILBOX_ERROR_MSG_SIZE,
    MAILBOX_OFF_ERROR_MSG,
    MAILBOX_SIZE,
    CallConfig,
    ChipCallable,
    ChipDomainContext,
    ChipWorker,
    CommBufferSpec,
    CommDomainHandle,
    RemoteAddressSpace,
    RemoteBufferExport,
    RemoteBufferHandle,
    TaskArgs,
    Tensor,
    _Worker,
)

# Upper bound on how long the parent waits for every chip's bootstrap mailbox
# to leave IDLE.  Well above a realistic HCCL init (seconds) but short enough
# that a hung child fails the suite instead of the CI job timing out.
_BOOTSTRAP_WAIT_TIMEOUT_S = 120.0
_BOOTSTRAP_POLL_INTERVAL_S = 0.001
_PY_CONTROL_TIMEOUT_S = 30.0
# L2 endpoint metadata currently reaches the parent through the canonical fatal
# text emitted by the orchestration wrapper; keep this pattern in sync with the
# wrapper's ``L3-L2 endpoint error ... region=<id>`` format.
_L3_L2_ENDPOINT_ERROR_REGION_RE = re.compile(r"\bL3-L2 endpoint error\b[^\n]*\bregion=(\d+)\b")


# ---------------------------------------------------------------------------
# Unified mailbox layout (must match worker_manager.h MAILBOX_OFF_*)
# ---------------------------------------------------------------------------
#
# One layout for both NEXT_LEVEL (chip) and SUB workers. TASK_READY carries
# the stable callable digest prefix in the args region; children resolve it
# to their private integer slot before reading the TaskArgs blob.

_OFF_STATE = 0
_OFF_ERROR = 4
_OFF_CALLABLE = 8
_OFF_CONFIG = 16
# Packed CallConfig wire layout — must match call_config.h byte for byte:
# 8 int32 (block_dim, aicpu_thread_num, enable_l2_swimlane, enable_dump_args,
# enable_pmu, enable_dep_gen, enable_scope_stats, enable_graph_cache) + uint64 ring sizing
# overrides (3 per-ring arrays of RUNTIME_ENV_RING_COUNT: ring_task_window,
# ring_heap, ring_dep_pool) + 1024-byte NUL-terminated output_prefix. Log config
# travels separately via ChipWorker.init(log_level, log_info_v) — not on per-task wire.
_RUNTIME_ENV_UINT64_FIELD_COUNT = 3 * RUNTIME_ENV_RING_COUNT
_CFG_FMT = struct.Struct("=iiiiiiii" + ("Q" * _RUNTIME_ENV_UINT64_FIELD_COUNT) + "1024s")
# Args region starts after CONFIG, rounded up to 8 bytes so the first
# Tensor.data (uint64_t at OFF_ARGS+8) is 8-byte aligned, avoiding
# SIGBUS on strict-alignment platforms (aarch64 atomics, some ARM cores).
_OFF_ARGS = (_OFF_CONFIG + _CFG_FMT.size + 7) & ~7
assert _OFF_ARGS % 8 == 0, "_OFF_ARGS must be 8-aligned for Tensor.data"
_OFF_TASK_CALLABLE_HASH = _OFF_ARGS
_OFF_TASK_ARGS_BLOB = _OFF_TASK_CALLABLE_HASH + CALLABLE_HASH_DIGEST_BYTES
# MAILBOX_ARGS_CAPACITY mirrors the C++ constexpr in worker_manager.h so the
# Python reader can bounds-check incoming args blobs. Source-of-truth for the
# constants on the right is the nanobind binding (cannot drift).
_MAILBOX_ARGS_CAPACITY = MAILBOX_SIZE - _OFF_TASK_ARGS_BLOB - MAILBOX_ERROR_MSG_SIZE
_OFF_CONTROL_CALLABLE_HASH = _OFF_ARGS + 32
# MAILBOX_OFF_ERROR_MSG / MAILBOX_ERROR_MSG_SIZE come from the C++
# nanobind module so the two sides cannot drift.

_IDLE = 0
_TASK_READY = 1
_TASK_DONE = 2
_SHUTDOWN = 3
_CONTROL_REQUEST = 4
_CONTROL_DONE = 5
# Startup readiness handshake. A child writes INIT_READY after its own init
# (ChipWorker.init / inner Worker.init) succeeds, or INIT_FAILED after it fails,
# leaving the cause in the mailbox error region. The parent's readiness barrier
# (_await_children_ready) blocks on every child reaching INIT_READY before any
# dispatch, which also keeps cross-rank init skew out of the per-rank host-side
# stream sync budget (issue #897); INIT_FAILED, a dead child, or a blown
# deadline aborts startup with a bounded error instead of an unbounded spin.
_INIT_READY = 6
_INIT_FAILED = 7

# Startup readiness bound. A child that neither reports INIT_READY/INIT_FAILED
# nor exits within this window is treated as hung and startup is aborted.
# Generous by default so a legitimately slow device/runtime init (large
# PTO2_RING_HEAP, cold arena build) is never falsely reaped; override per Worker
# via the `startup_timeout_s` config kwarg. The point is to bound *hangs*, not
# to police slow-but-progressing init.
_STARTUP_TIMEOUT_S = 300.0
# Parent poll granularity while waiting for children to become ready. Cheap
# shared-memory reads dominate; the sleep only caps waitpid/deadline syscall
# frequency and is far below any real init-skew alignment concern.
_STARTUP_POLL_INTERVAL_S = 0.001
# On startup rollback, a next-level child that reached its serve loop is asked
# to close gracefully (so it unlinks the nested mailbox shms only it knows the
# names of) before being SIGKILLed. This bounds that graceful wait.
_ROLLBACK_GRACEFUL_TIMEOUT_S = 10.0
# Bounded re-check interval for a close() joiner waiting on an in-flight
# _CloseAttempt. A joiner normally wakes immediately on the completing thread's
# notify_all(); the timeout is a backstop so that if that notify is skipped (an
# async BaseException landing between publishing `done` and notifying), the
# joiner still re-observes `done` within this interval instead of blocking
# forever.
_CLOSE_JOIN_RECHECK_S = 1.0

# Control sub-commands (written at _OFF_CALLABLE as uint64)
_CTRL_MALLOC = 0
_CTRL_FREE = 1
_CTRL_COPY_TO = 2
_CTRL_COPY_FROM = 3
# Pre-warm a chip child by callable digest. The child resolves the digest to
# its own target-local slot and prepares that slot so the first run() does not
# pay the H2D upload cost. Sent from the parent right after startup.
_CTRL_PREPARE = 4
# Dynamic post-init register of a ChipCallable. Parent stages the bytes
# in a per-register POSIX shm and writes (digest, shm_name, blob_size) into
# the mailbox; the child mmaps the shm, allocates its own local slot, and
# prepares that slot. See docs/callable-identity-registration.md for the design.
_CTRL_REGISTER = 5
# Symmetric unregister by callable digest. The child drops one local reference
# and frees the target-local slot when the final digest reference is removed.
_CTRL_UNREGISTER = 6
# Dynamic CommDomain allocate / release (collective across the participating
# subset).  Parent stages the request in a POSIX shm whose name is at
# OFF_ARGS+0; for alloc, it also pre-allocates a reply shm whose name is at
# OFF_ARGS+32.  Both shms have a fixed header (see _DOMAIN_REQ_HEADER /
# _DOMAIN_REPLY_HEADER) followed by variable buffer/rank data.
_CTRL_ALLOC_DOMAIN = 7
_CTRL_RELEASE_DOMAIN = 8
# Lazy base-comm init driven from Orchestrator.allocate_domain on first use.
# Request shm carries `<II` header (rank, nranks) + NUL-terminated
# rootinfo_path bytes.  Chip child calls cw.comm_init(rank, nranks,
# rootinfo_path) and caches the handle on the ChipWorker so subsequent
# CTRL_ALLOC_DOMAIN calls can find it.
_CTRL_COMM_INIT = 9
_CTRL_PY_REGISTER = 10
_CTRL_PY_UNREGISTER = 11
_CTRL_PY_IMPORT_REGISTER = 12
# Host-buffer registration. MAP_HOST maps a named host-buffer shm
# into every local L3 child *post-fork* and keeps it mapped so later runs can copy
# through it; UNMAP_HOST drops one. The child also records the parent VA range
# the shm stands in for, so the per-task blob's host pointers (raw parent VAs)
# can be rewritten to the child's own mapping before the runtime dereferences
# them. Unlike _CTRL_REGISTER (one-shot H2D then close), these mappings persist
# for the buffer's registered lifetime — see docs/comm-domain.md.
_CTRL_MAP_HOST = 14
_CTRL_UNMAP_HOST = 15

# MAP_HOST payload: token (u64), parent_va (u64), nbytes (u64), then the
# NUL-free host-buffer shm name as the trailing bytes. UNMAP_HOST payload is the
# token alone.
_HOST_BUF_MAP_HEADER = struct.Struct("<QQQ")
_HOST_BUF_UNMAP = struct.Struct("<Q")

# Wire layout of a Tensor inside a task-args blob, pinned by static_assert in
# src/common/task_interface/tensor.h: each Tensor is 128 B and buffer.addr is its
# first field (offset 0). The blob is [int32 T][int32 S][Tensor[T]][scalars], so
# tensor i's host pointer lives at _OFF_TASK_ARGS_BLOB + 8 + i*128. The child
# rewrites that u64 in place to redirect a registered host pointer at its own
# mapping (the pure-Python blob-rewrite scheme, no runtime C++ change).
_BLOB_TENSOR_STRIDE = 128
_BLOB_HEADER_BYTES = 8
_CTRL_L3_L2_REGION_CREATE = 16
_CTRL_L3_L2_REGION_RELEASE = 17

# Layout of the CTRL_COMM_INIT request shm.
_COMM_INIT_HEADER = struct.Struct("<II")  # rank (u32), nranks (u32)
assert _COMM_INIT_HEADER.size == 8

_PY_CALLABLE_MAGIC = b"SPYC"
_PY_CALLABLE_VERSION = 1
_PY_CALLABLE_SERIALIZER_CLOUDPICKLE = 1
_PY_CALLABLE_HEADER = struct.Struct("<4sBBHQ")

# Reserved 32-byte region at the start of OFF_ARGS used by _CTRL_REGISTER to
# carry the NUL-terminated POSIX shm name. POSIX shm names on Linux are
# bounded well below this, but the on-wire field is fixed-width to keep
# the layout simple.
#
# _CTRL_ALLOC_DOMAIN uses two such slots back to back at OFF_ARGS (request
# shm at offset 0, reply shm at offset CTRL_SHM_NAME_BYTES).  _CTRL_RELEASE_DOMAIN
# uses only the first slot.
_CTRL_SHM_NAME_BYTES = 32

# Domain-allocation request shm layout: 32-byte header + buffer_nbytes (u64) +
# rank_ids (u32).  Buffer specs first so they remain 8-byte aligned regardless
# of rank_count parity; rank_ids come last (u32 has no alignment concern).
_DOMAIN_REQ_HEADER = struct.Struct("<QIIQI4x")
# fields: allocation_id (u64), rank_count (u32), domain_rank (u32),
#         window_size (u64), buffer_count (u32), padding (4 bytes)
assert _DOMAIN_REQ_HEADER.size == 32

# Domain-allocation reply shm layout: 24-byte header + buffer_ptrs (u64).
_DOMAIN_REPLY_HEADER = struct.Struct("<QQI4x")
# fields: device_ctx (u64), local_window_base (u64),
#         buffer_count (u32), padding (4 bytes)
assert _DOMAIN_REPLY_HEADER.size == 24

# Control args layout (reuses task mailbox fields when state == _CONTROL_*):
#   offset  8 (_OFF_CALLABLE):  uint64  sub-command
#   offset 16:                  uint64  arg0 (size for malloc/register; dev_ptr for free/copy)
#   offset 24:                  uint64  arg1 (host_ptr for copy)
#   offset 32:                  uint64  arg2 (nbytes for copy)
#   offset 40:                  uint64  result (returned ptr from malloc)
_CTRL_OFF_ARG0 = 16
_CTRL_OFF_ARG1 = 24
_CTRL_OFF_ARG2 = 32
_CTRL_OFF_RESULT = 40


@dataclass
class _CallableRegistration:
    target: Any
    kind: str
    target_namespace: str
    descriptor: bytes
    hashid: str
    digest: bytes
    payload_digest: bytes
    payload: bytes | None = None
    eligible_worker_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class RemoteCallable:
    """Import-path descriptor for a parent-facing remote L3 callable."""

    target: str

    def __post_init__(self) -> None:
        module, qualname = parse_python_import_target(self.target)
        object.__setattr__(self, "target", f"{module}:{qualname}")

    @property
    def module(self) -> str:
        return self.target.split(":", 1)[0]

    @property
    def qualname(self) -> str:
        return self.target.split(":", 1)[1]


@dataclass(frozen=True)
class RemoteWorkerSpec:
    endpoint: str
    platform: str
    runtime: str = "tensormap_and_ringbuffer"
    device_ids: tuple[int, ...] = ()
    num_sub_workers: int = 0
    transport: str = "sim"
    session_listen_host: str | None = None
    allow_wildcard_session_bind: bool = False

    def __post_init__(self) -> None:
        if not self.endpoint:
            raise ValueError("RemoteWorkerSpec.endpoint must be non-empty")
        if not self.platform:
            raise ValueError("RemoteWorkerSpec.platform must be non-empty")
        if self.session_listen_host is not None and not self.session_listen_host:
            raise ValueError("RemoteWorkerSpec.session_listen_host must be non-empty when set")
        object.__setattr__(self, "endpoint", str(self.endpoint))
        object.__setattr__(self, "platform", str(self.platform))
        object.__setattr__(self, "runtime", str(self.runtime))
        object.__setattr__(self, "transport", str(self.transport))
        object.__setattr__(
            self,
            "session_listen_host",
            None if self.session_listen_host is None else str(self.session_listen_host),
        )
        object.__setattr__(self, "allow_wildcard_session_bind", bool(self.allow_wildcard_session_bind))
        object.__setattr__(self, "device_ids", tuple(int(x) for x in self.device_ids))
        object.__setattr__(self, "num_sub_workers", int(self.num_sub_workers))
        if self.num_sub_workers < 0:
            raise ValueError("RemoteWorkerSpec.num_sub_workers must be non-negative")


@dataclass(frozen=True)
class _RemoteSession:
    worker_id: int
    session_id: int
    command_host: str
    command_port: int
    health_host: str
    health_port: int
    pid: int


_IdentitySnapshotEntry = tuple[bytes, Any, int, str, str]


class _ChildProvEntry:
    """Provenance record for one exact ``(worker_id, device_ptr)`` child pointer.

    Typed rather than a bare presence bit because the same ``(worker_id, ptr)``
    can carry more than one role at once: a ``malloc`` base and a CommDomain
    window / carved buffer pointer can legally alias the same device address.
    The key is live while ``malloc_owned or domain_allocation_ids``; only an
    exact ``malloc`` base is ``free``-able, while a domain pointer is revoked by
    its domain's release. Interior pointers are never recorded, so a pointer
    that merely lands inside a live allocation has no entry and is rejected.
    """

    __slots__ = ("malloc_owned", "domain_allocation_ids")

    def __init__(self) -> None:
        self.malloc_owned: bool = False
        self.domain_allocation_ids: set[int] = set()

    def is_live(self) -> bool:
        """True iff this entry still carries a role. A role-less entry is dead —
        live checks are fail-closed on this, never on key presence alone, so an
        entry momentarily left empty (e.g. an interrupted revoke) never
        re-authorizes a freed pointer."""
        return self.malloc_owned or bool(self.domain_allocation_ids)


@dataclass
class _HostBufEntry:
    """Parent-side record for a born-shared post-fork host buffer.

    The worker owns ``shm`` — a named buffer the local L3 children attach and
    read/write through. The user builds a tensor over it (via the buffer
    protocol on :class:`HostBuffer`), so the buffer *is* the shm: ``data_ptr ==
    shm_base`` and no per-run copy is needed (the child reads and writes the same
    physical pages the parent sees). ``shm_base`` caches the mapped address.
    """

    token: int
    data_ptr: int
    nbytes: int
    shm: SharedMemory
    shm_name: str
    shm_base: int


@dataclass(frozen=True, eq=False)
class HostBuffer:
    """Handle for a worker-allocated, born-shared host buffer (zero-copy).

    Returned by ``Worker.create_host_buffer``. ``buffer`` is a ``memoryview``
    over shared memory already attached into every local L3 child; wrap it with
    ``torch.frombuffer`` / ``np.frombuffer`` to get a real tensor whose writes
    land directly in the child-visible pages — no per-run copy. ``token`` /
    ``data_ptr`` / ``nbytes`` identify the mapping; pass this handle back to
    ``free_host_buffer`` to release it.

    ``eq=False`` keeps object-identity equality/hash so the (unhashable)
    ``memoryview`` field never blocks using the handle as a dict key or set
    member.
    """

    token: int
    data_ptr: int
    nbytes: int
    buffer: memoryview


def _rewrite_blob_host_addrs(buf: memoryview, blob_off: int, ranges: list[tuple[int, int, int]]) -> None:
    """Redirect registered host pointers in a task-args blob to child mappings.

    ``ranges`` is ``(parent_lo, parent_hi, child_base)`` for each host buffer the
    child has mapped via _CTRL_MAP_HOST. For every host tensor whose
    ``buffer.addr`` (a parent VA) lands in a registered range, rewrite it in
    place to ``child_base + (addr - parent_lo)`` so the runtime dereferences the
    child's own mapping. Tensors outside every range (fork-inherited or
    child-allocated) are left untouched. A ``child_memory`` tensor carries a
    child-owned device pointer, never a host VA, so it is skipped even when its
    address numerically falls inside a registered host range — rewriting it would
    corrupt the device pointer. See _BLOB_TENSOR_STRIDE for the wire layout.
    """
    tensor_count = struct.unpack_from("<i", buf, blob_off)[0]
    if tensor_count <= 0:
        return
    base = blob_off + _BLOB_HEADER_BYTES
    for i in range(tensor_count):
        addr_off = base + i * _BLOB_TENSOR_STRIDE
        if buf[addr_off + TENSOR_CHILD_MEMORY_OFFSET]:
            continue
        addr = struct.unpack_from("<Q", buf, addr_off)[0]
        for parent_lo, parent_hi, child_base in ranges:
            if parent_lo <= addr < parent_hi:
                struct.pack_into("<Q", buf, addr_off, child_base + (addr - parent_lo))
                break


def _read_ctrl_staged_shm_name(buf: memoryview) -> str:
    """Decode the staged-payload shm name a broadcast_control_all left at _OFF_ARGS."""
    raw = bytes(buf[_OFF_ARGS : _OFF_ARGS + _CTRL_SHM_NAME_BYTES])
    nul = raw.find(b"\x00")
    return raw[: nul if nul >= 0 else _CTRL_SHM_NAME_BYTES].decode("utf-8", "replace")


def _shm_base_addr(shm: SharedMemory) -> int:
    """Mapped base address of ``shm``. The mapping outlives the temporary buffer
    view, so the address stays valid until ``shm.close()``."""
    view = shm.buf
    assert view is not None
    exporter = ctypes.c_char.from_buffer(view)
    addr = ctypes.addressof(exporter)
    del exporter
    return addr


def _rebuild_host_buf_ranges(
    host_buf_table: dict[int, tuple[SharedMemory, int, int, int]], host_buf_ranges: list[tuple[int, int, int]]
) -> None:
    host_buf_ranges.clear()
    for _shm, lo, hi, base in host_buf_table.values():
        host_buf_ranges.append((lo, hi, base))


def _handle_ctrl_map_host(
    buf: memoryview,
    host_buf_table: dict[int, tuple[SharedMemory, int, int, int]],
    host_buf_ranges: list[tuple[int, int, int]],
) -> None:
    """Child handler for _CTRL_MAP_HOST: persist a host-buffer mapping.

    The staged payload is ``token, parent_va, nbytes`` followed by the host
    buffer's shm name. Map that shm and remember the parent VA range it stands
    in for so the per-task blob rewrite can redirect host pointers to this base.
    """
    payload_size = struct.unpack_from("Q", buf, _CTRL_OFF_ARG0)[0]
    staged = SharedMemory(name=_read_ctrl_staged_shm_name(buf))
    try:
        staged_buf = staged.buf
        assert staged_buf is not None
        payload = bytes(staged_buf[:payload_size])
    finally:
        staged.close()
    token, parent_va, nbytes = _HOST_BUF_MAP_HEADER.unpack_from(payload, 0)
    host_shm_name = payload[_HOST_BUF_MAP_HEADER.size :].decode("utf-8")
    prior = host_buf_table.pop(token, None)
    if prior is not None:
        prior[0].close()
    # Rebuild ranges in a finally so a raise from SharedMemory / _shm_base_addr
    # cannot leave the just-popped prior mapping's stale range in host_buf_ranges.
    try:
        host_shm = SharedMemory(name=host_shm_name)
        host_buf_table[token] = (host_shm, parent_va, parent_va + nbytes, _shm_base_addr(host_shm))
    finally:
        _rebuild_host_buf_ranges(host_buf_table, host_buf_ranges)


def _handle_ctrl_unmap_host(
    buf: memoryview,
    host_buf_table: dict[int, tuple[SharedMemory, int, int, int]],
    host_buf_ranges: list[tuple[int, int, int]],
) -> None:
    """Child handler for _CTRL_UNMAP_HOST: drop a host-buffer mapping by token."""
    payload_size = struct.unpack_from("Q", buf, _CTRL_OFF_ARG0)[0]
    staged = SharedMemory(name=_read_ctrl_staged_shm_name(buf))
    try:
        staged_buf = staged.buf
        assert staged_buf is not None
        token = _HOST_BUF_UNMAP.unpack_from(bytes(staged_buf[:payload_size]), 0)[0]
    finally:
        staged.close()
    entry = host_buf_table.pop(token, None)
    if entry is not None:
        entry[0].close()
        _rebuild_host_buf_ranges(host_buf_table, host_buf_ranges)


def _allocate_local_slot(registry: dict[int, Any]) -> int:
    for i in range(MAX_REGISTERED_CALLABLE_IDS):
        if i not in registry:
            return i
    raise RuntimeError(
        "LOCAL_SLOT_EXHAUSTED: no free target-local callable slots "
        f"(MAX_REGISTERED_CALLABLE_IDS={MAX_REGISTERED_CALLABLE_IDS})"
    )


def _install_local_identity(
    registry: dict[int, Any],
    identity_table: dict[bytes, int],
    identity_refs: dict[bytes, int],
    digest: bytes,
    target: Any,
) -> int:
    if len(digest) != CALLABLE_HASH_DIGEST_BYTES:
        raise RuntimeError(f"callable digest must be {CALLABLE_HASH_DIGEST_BYTES} bytes")
    slot = identity_table.get(digest)
    if slot is not None:
        identity_refs[digest] = identity_refs.get(digest, 1) + 1
        return int(slot)
    slot = _allocate_local_slot(registry)
    registry[slot] = target
    identity_table[digest] = slot
    identity_refs[digest] = 1
    return slot


def _remove_local_identity(
    registry: dict[int, Any],
    identity_table: dict[bytes, int],
    identity_refs: dict[bytes, int],
    digest: bytes,
) -> tuple[int | None, bool]:
    slot = identity_table.get(digest)
    if slot is None:
        return None, False
    refs = identity_refs.get(digest, 1) - 1
    if refs > 0:
        identity_refs[digest] = refs
        return int(slot), False
    identity_refs.pop(digest, None)
    identity_table.pop(digest, None)
    registry.pop(int(slot), None)
    return int(slot), True


def _make_local_identity_tables(
    snapshot: list[_IdentitySnapshotEntry],
    *,
    callable_kind: str | tuple[str, ...] | None = None,
    target_namespace: str | None = None,
) -> tuple[dict[int, Any], dict[bytes, int], dict[bytes, int]]:
    registry: dict[int, Any] = {}
    identity_table: dict[bytes, int] = {}
    identity_refs: dict[bytes, int] = {}
    callable_kinds = (callable_kind,) if isinstance(callable_kind, str) else callable_kind
    for digest, target, ref_count, kind, namespace in snapshot:
        if callable_kinds is not None and kind not in callable_kinds:
            continue
        if target_namespace is not None and namespace != target_namespace:
            continue
        if len(digest) != CALLABLE_HASH_DIGEST_BYTES:
            raise RuntimeError(f"callable digest must be {CALLABLE_HASH_DIGEST_BYTES} bytes")
        slot = _allocate_local_slot(registry)
        identity_table[digest] = slot
        identity_refs[digest] = max(int(ref_count), 1)
        registry[slot] = target
    return registry, identity_table, identity_refs


def _pack_py_callable_payload(target) -> bytes:
    payload = cloudpickle.dumps(target)
    return (
        _PY_CALLABLE_HEADER.pack(
            _PY_CALLABLE_MAGIC,
            _PY_CALLABLE_VERSION,
            _PY_CALLABLE_SERIALIZER_CLOUDPICKLE,
            0,
            len(payload),
        )
        + payload
    )


def _chip_descriptor_context(worker: Worker) -> tuple[str, str]:
    platform = str(worker._config.get("platform", ""))
    runtime = str(worker._config.get("runtime", ""))
    if platform or runtime:
        return platform, runtime

    contexts: list[tuple[str, str]] = []
    for child in getattr(worker, "_next_level_workers", []):
        child_context = _chip_descriptor_context(child)
        if child_context != ("", ""):
            contexts.append(child_context)
    if not contexts:
        return "", ""
    first = contexts[0]
    if any(ctx != first for ctx in contexts[1:]):
        raise RuntimeError("Worker.register: heterogeneous chip child contexts require separate callable namespaces")
    return first


def _build_callable_registration(worker: Worker, target, *, workers: list[int] | None = None) -> _CallableRegistration:
    if isinstance(target, RemoteCallable):
        if workers is None or len(workers) == 0:
            raise ValueError("Worker.register(RemoteCallable): workers must be an explicit non-empty list")
        worker_ids = tuple(int(w) for w in workers)
        if any(w < 0 for w in worker_ids):
            raise ValueError("Worker.register(RemoteCallable): worker ids must be non-negative")
        if len(set(worker_ids)) != len(worker_ids):
            raise ValueError("Worker.register(RemoteCallable): workers must not contain duplicates")
        descriptor = build_python_import_descriptor(target.module, target.qualname)
        hashid = compute_callable_hashid(descriptor)
        return _CallableRegistration(
            target=target,
            kind="PYTHON_IMPORT",
            target_namespace="REMOTE_TASK_DISPATCHER",
            descriptor=descriptor,
            hashid=hashid,
            digest=hashid_to_digest(hashid),
            payload_digest=descriptor,
            payload=target.target.encode("utf-8"),
            eligible_worker_ids=worker_ids,
        )
    if isinstance(target, ChipCallable):
        if workers is not None:
            raise TypeError("Worker.register: workers= is only supported for RemoteCallable")
        platform, runtime = _chip_descriptor_context(worker)
        descriptor = build_chip_callable_descriptor(
            target=target,
            platform=platform,
            runtime=runtime,
        )
        hashid = compute_callable_hashid(descriptor)
        return _CallableRegistration(
            target=target,
            kind="CHIP_CALLABLE",
            target_namespace="LOCAL_CHIP",
            descriptor=descriptor,
            hashid=hashid,
            digest=hashid_to_digest(hashid),
            payload_digest=descriptor,
            payload=None,
        )
    if workers is not None:
        raise TypeError("Worker.register: workers= is only supported for RemoteCallable")
    if not callable(target):
        raise TypeError("Worker.register: non-ChipCallable target must be callable")
    payload = _pack_py_callable_payload(target)
    descriptor = build_python_serialized_descriptor(payload)
    hashid = compute_callable_hashid(descriptor)
    return _CallableRegistration(
        target=target,
        kind="PYTHON_SERIALIZED",
        target_namespace="LOCAL_PYTHON",
        descriptor=descriptor,
        hashid=hashid,
        digest=hashid_to_digest(hashid),
        payload_digest=descriptor,
        payload=payload,
    )


def _descriptor_digest(descriptor: bytes) -> bytes:
    return hashid_to_digest(compute_callable_hashid(descriptor))


def _validate_descriptor_digest(*, expected: bytes, descriptor: bytes, context: str) -> None:
    actual = _descriptor_digest(descriptor)
    if actual != expected:
        raise RuntimeError(
            f"HASHID_DESCRIPTOR_MISMATCH: {context} requested {_format_digest(expected)} "
            f"but payload is {_format_digest(actual)}"
        )


def _validate_chip_payload_digest(
    callable_obj: ChipCallable,
    digest: bytes,
    *,
    platform: str = "",
    runtime: str = "",
    context: str,
) -> None:
    descriptor = build_chip_callable_descriptor(target=callable_obj, platform=platform, runtime=runtime)
    _validate_descriptor_digest(expected=digest, descriptor=descriptor, context=context)


def _read_py_callable_payload_from_shm(shm_name: str) -> bytes:
    shm = SharedMemory(name=shm_name)
    shm_buf = shm.buf
    assert shm_buf is not None
    try:
        if shm.size < _PY_CALLABLE_HEADER.size:
            raise RuntimeError(f"python callable payload too small: {shm.size} bytes")
        magic, version, serializer, flags, payload_size = _PY_CALLABLE_HEADER.unpack_from(shm_buf, 0)
        if magic != _PY_CALLABLE_MAGIC:
            raise RuntimeError(f"invalid python callable payload magic: {magic!r}")
        if version != _PY_CALLABLE_VERSION:
            raise RuntimeError(f"unsupported python callable payload version: {version}")
        if serializer != _PY_CALLABLE_SERIALIZER_CLOUDPICKLE:
            raise RuntimeError(f"unsupported python callable serializer: {serializer}")
        if flags != 0:
            raise RuntimeError(f"unsupported python callable payload flags: {flags}")
        expected_size = _PY_CALLABLE_HEADER.size + int(payload_size)
        if expected_size > shm.size:
            raise RuntimeError(f"python callable payload size mismatch: header={payload_size}, shm={shm.size}")
        payload = bytes(shm_buf[:expected_size])
        return payload
    finally:
        shm_buf.release()
        shm.close()


def _read_raw_payload_from_shm(shm_name: str, payload_size: int) -> bytes:
    shm = SharedMemory(name=shm_name)
    shm_buf = shm.buf
    assert shm_buf is not None
    try:
        if payload_size <= 0 or payload_size > shm.size:
            raise RuntimeError(f"raw control payload size mismatch: payload={payload_size}, shm={shm.size}")
        return bytes(shm_buf[:payload_size])
    finally:
        shm_buf.release()
        shm.close()


def _read_chip_callable_from_shm(shm_name: str, payload_size: int) -> ChipCallable:
    shm = SharedMemory(name=shm_name)
    shm_buf = shm.buf
    assert shm_buf is not None
    try:
        if payload_size <= 0 or payload_size > shm.size:
            raise RuntimeError(f"CTRL_REGISTER payload size mismatch: payload={payload_size}, shm={shm.size}")
        return ChipCallable.from_bytes(bytes(shm_buf[:payload_size]))
    finally:
        shm_buf.release()
        shm.close()


def _load_py_callable_from_payload(payload: bytes):
    _version, _serializer, serializer_payload = parse_python_callable_payload(payload)
    fn = cloudpickle.loads(serializer_payload)
    if not callable(fn):
        raise RuntimeError(f"python callable payload decoded to non-callable {type(fn).__name__}")
    return fn


def _load_py_callable_from_shm(shm_name: str):
    return _load_py_callable_from_payload(_read_py_callable_payload_from_shm(shm_name))


def _load_py_import_target(target: str):
    module_name, qualname = parse_python_import_target(target)
    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"python import target {target!r} is not callable")
    return obj


def _read_control_digest(buf) -> bytes:
    return bytes(buf[_OFF_CONTROL_CALLABLE_HASH : _OFF_CONTROL_CALLABLE_HASH + CALLABLE_HASH_DIGEST_BYTES])


def _read_task_digest(buf) -> bytes:
    return bytes(buf[_OFF_TASK_CALLABLE_HASH : _OFF_TASK_CALLABLE_HASH + CALLABLE_HASH_DIGEST_BYTES])


def _format_digest(digest: bytes) -> str:
    return "sha256:" + digest.hex()


def _handle_py_callable_control(
    buf,
    registry: dict[int, Any],
    identity_table: dict[bytes, int],
    identity_refs: dict[bytes, int],
    sub_cmd: int,
    *,
    context: str,
) -> None:
    digest = _read_control_digest(buf)
    if sub_cmd == _CTRL_PY_REGISTER:
        shm_name = _read_shm_name(buf, _OFF_ARGS)
        payload = _read_py_callable_payload_from_shm(shm_name)
        descriptor = build_python_serialized_descriptor(payload)
        _validate_descriptor_digest(expected=digest, descriptor=descriptor, context=f"{context} python callable")
        if digest in identity_table:
            identity_refs[digest] = identity_refs.get(digest, 1) + 1
            return
        _install_local_identity(
            registry,
            identity_table,
            identity_refs,
            digest,
            _load_py_callable_from_payload(payload),
        )
    elif sub_cmd == _CTRL_PY_IMPORT_REGISTER:
        shm_name = _read_shm_name(buf, _OFF_ARGS)
        payload_size = struct.unpack_from("Q", buf, _CTRL_OFF_ARG0)[0]
        payload = _read_raw_payload_from_shm(shm_name, int(payload_size))
        target = payload.decode("utf-8")
        module, qualname = parse_python_import_target(target)
        descriptor = build_python_import_descriptor(module, qualname)
        _validate_descriptor_digest(expected=digest, descriptor=descriptor, context=f"{context} python import")
        if digest in identity_table:
            identity_refs[digest] = identity_refs.get(digest, 1) + 1
            return
        _install_local_identity(
            registry,
            identity_table,
            identity_refs,
            digest,
            _load_py_import_target(target),
        )
    elif sub_cmd == _CTRL_PY_UNREGISTER:
        _remove_local_identity(registry, identity_table, identity_refs, digest)
    else:
        raise RuntimeError(f"{context}: unknown control sub-command {int(sub_cmd)}")


def _mailbox_addr(shm: SharedMemory) -> int:
    buf = shm.buf
    assert buf is not None
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


def _buffer_field_addr(buf, offset: int) -> int:
    """Absolute address of a field inside a shared-memory buffer.

    Used to feed `_mailbox_load_i32` / `_mailbox_store_i32`, which operate on
    raw pointers so the acquire/release semantics match the C++ side
    (worker_manager.cpp::read_mailbox_state / write_mailbox_state).
    """
    return ctypes.addressof(ctypes.c_char.from_buffer(buf)) + offset


def _write_error(buf, code: int, msg: str = "") -> None:
    """Write an (error code, message) tuple into the mailbox error region.

    The message is UTF-8-encoded and truncated to ``MAILBOX_ERROR_MSG_SIZE - 1``
    bytes so a NUL terminator always fits — the C++ reader assumes
    NUL-terminated content. On success (code=0) callers may pass an empty
    message; the region is zero-padded.
    """
    struct.pack_into("i", buf, _OFF_ERROR, code)
    encoded = msg.encode("utf-8", "replace")
    n = min(len(encoded), MAILBOX_ERROR_MSG_SIZE - 1)
    start = MAILBOX_OFF_ERROR_MSG
    buf[start : start + n] = encoded[:n]
    # Zero-pad the remaining bytes so stale content from a previous dispatch
    # never leaks into the current error report.
    buf[start + n : start + MAILBOX_ERROR_MSG_SIZE] = b"\x00" * (MAILBOX_ERROR_MSG_SIZE - n)


def _read_error_msg(buf) -> str:
    """Read the mailbox error message, trimming at the first NUL."""
    raw = bytes(buf[MAILBOX_OFF_ERROR_MSG : MAILBOX_OFF_ERROR_MSG + MAILBOX_ERROR_MSG_SIZE])
    nul = raw.find(b"\x00")
    if nul >= 0:
        raw = raw[:nul]
    return raw.decode("utf-8", "replace")


def _format_exc(prefix: str, exc: BaseException) -> str:
    return f"{prefix}: {type(exc).__name__}: {exc}"


def _read_args_from_mailbox(buf) -> TaskArgs:
    """Decode the TaskArgs blob written by C++ write_blob from the mailbox.

    Used by the Python-targeted child loops (sub_worker, nested L4+ child)
    where the destination of `args` is a Python callable that needs a
    typed TaskArgs object.  The chip-child loops that immediately forward
    to C++ run use the zero-copy `run_from_blob` path
    instead — see those loops for the matching comment.

    Delegates to the nanobind helper so the Tensor layout is
    parsed by C++ `read_blob` (single source of truth) instead of being
    reimplemented in Python.  The Python re-implementation that lived
    here previously dropped the `child_memory` byte (offset 33), which
    silently broke any tensor carrying a chip-owned device pointer
    (HCCL window slots etc.) — now structurally impossible.
    """
    mailbox_addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    return read_args_from_blob(mailbox_addr + _OFF_TASK_ARGS_BLOB)


def _sub_worker_loop(
    buf,
    registry: dict[int, Any],
    identity_table: dict[bytes, int],
    identity_refs: dict[bytes, int],
) -> None:
    """Runs in forked child process. Reads unified mailbox layout.

    On success writes ``error=0`` and an empty message. On failure writes
    ``error=1`` and ``f"sub_worker: <ExcType>: <msg>"`` into the mailbox
    error-message region; the parent's ``WorkerThread::dispatch_process``
    rethrows it as ``std::runtime_error``.
    """
    state_addr = _buffer_field_addr(buf, _OFF_STATE)
    host_buf_table: dict[int, tuple[SharedMemory, int, int, int]] = {}
    host_buf_ranges: list[tuple[int, int, int]] = []
    try:
        while True:
            state = _mailbox_load_i32(state_addr)
            if state == _TASK_READY:
                digest = _read_task_digest(buf)
                cid = identity_table.get(digest)
                fn = registry.get(int(cid)) if cid is not None else None
                code = 0
                msg = ""
                if fn is None:
                    code = 1
                    msg = f"sub_worker: callable hash {_format_digest(digest)} not registered"
                else:
                    try:
                        if host_buf_ranges:
                            _rewrite_blob_host_addrs(buf, _OFF_TASK_ARGS_BLOB, host_buf_ranges)
                        args = _read_args_from_mailbox(buf)
                        fn(args)
                    except Exception as e:  # noqa: BLE001
                        code = 1
                        msg = _format_exc("sub_worker", e)
                _write_error(buf, code, msg)
                _mailbox_store_i32(state_addr, _TASK_DONE)
            elif state == _CONTROL_REQUEST:
                sub_cmd = struct.unpack_from("Q", buf, _OFF_CALLABLE)[0]
                code = 0
                msg = ""
                try:
                    if sub_cmd == _CTRL_MAP_HOST:
                        _handle_ctrl_map_host(buf, host_buf_table, host_buf_ranges)
                    elif sub_cmd == _CTRL_UNMAP_HOST:
                        _handle_ctrl_unmap_host(buf, host_buf_table, host_buf_ranges)
                    else:
                        _handle_py_callable_control(
                            buf,
                            registry,
                            identity_table,
                            identity_refs,
                            int(sub_cmd),
                            context="sub_worker",
                        )
                except Exception as e:  # noqa: BLE001
                    code = 1
                    msg = _format_exc("sub_worker control", e)
                _write_error(buf, code, msg)
                _mailbox_store_i32(state_addr, _CONTROL_DONE)
            elif state == _SHUTDOWN:
                break
    finally:
        for host_shm, _lo, _hi, _base in host_buf_table.values():
            try:
                host_shm.close()
            except Exception:  # noqa: BLE001
                pass


def _read_shm_name(buf, offset: int) -> str:
    """Decode a NUL-terminated POSIX shm name out of a fixed-width slot.

    Shared by every control sub-command that stages payload via a separate
    shm — CTRL_REGISTER (one slot), CTRL_ALLOC_DOMAIN (two slots), and
    CTRL_RELEASE_DOMAIN (one slot).
    """
    raw = bytes(buf[offset : offset + _CTRL_SHM_NAME_BYTES])
    nul = raw.find(b"\x00")
    return raw[: nul if nul >= 0 else _CTRL_SHM_NAME_BYTES].decode("utf-8", "replace")


def _handle_ctrl_alloc_domain(cw: ChipWorker, buf: memoryview) -> None:
    """CTRL_ALLOC_DOMAIN handler — runs on the chip child.

    Reads the request shm (header + buffer_nbytes + rank_ids), calls
    ``ChipWorker.comm_alloc_domain_windows`` (which drives the collective
    handshake via file barriers), carves buffer pointers locally, and writes
    (device_ctx, local_window_base, buffer_ptrs) into the parent-owned reply
    shm.  Failures propagate as exceptions; the dispatch loop turns them into
    a CONTROL_DONE with non-zero error code.
    """
    request_shm_name = _read_shm_name(buf, _OFF_ARGS)
    reply_shm_name = _read_shm_name(buf, _OFF_ARGS + _CTRL_SHM_NAME_BYTES)

    req_shm = SharedMemory(name=request_shm_name)
    req_buf = req_shm.buf
    assert req_buf is not None
    try:
        (allocation_id, rank_count, domain_rank, window_size, buffer_count) = _DOMAIN_REQ_HEADER.unpack_from(req_buf, 0)
        # Layout: header | buffer_nbytes[buffer_count] (u64) | rank_ids[rank_count] (u32)
        nbytes_offset = _DOMAIN_REQ_HEADER.size
        nbytes_struct = struct.Struct(f"<{buffer_count}Q") if buffer_count else struct.Struct("")
        buffer_nbytes = nbytes_struct.unpack_from(req_buf, nbytes_offset) if buffer_count else ()
        rank_ids_offset = nbytes_offset + nbytes_struct.size
        rank_ids_struct = struct.Struct(f"<{rank_count}I")
        rank_ids = list(rank_ids_struct.unpack_from(req_buf, rank_ids_offset))
    finally:
        req_buf.release()
        req_shm.close()

    handle = _comm_base_handle(cw)  # base communicator handle (cached on the ChipWorker)
    device_ctx, local_window_base = cw._impl.comm_alloc_domain_windows(
        int(handle),
        int(allocation_id),
        rank_ids,
        int(domain_rank),
        int(window_size),
    )

    # Carve buffer pointers sequentially inside the local window.
    buffer_ptrs: list[int] = []
    offset = 0
    for nbytes in buffer_nbytes:
        if offset + nbytes > window_size:
            raise ValueError(
                f"alloc_domain: buffer #{len(buffer_ptrs)} (nbytes={nbytes}) at offset={offset} "
                f"overflows window_size {window_size}"
            )
        buffer_ptrs.append(int(local_window_base) + offset)
        offset += int(nbytes)

    reply_shm = SharedMemory(name=reply_shm_name)
    reply_buf = reply_shm.buf
    assert reply_buf is not None
    try:
        _DOMAIN_REPLY_HEADER.pack_into(reply_buf, 0, int(device_ctx), int(local_window_base), int(buffer_count))
        if buffer_ptrs:
            struct.pack_into(f"<{len(buffer_ptrs)}Q", reply_buf, _DOMAIN_REPLY_HEADER.size, *buffer_ptrs)
    finally:
        reply_buf.release()
        reply_shm.close()


def _handle_ctrl_comm_init(cw: ChipWorker, buf: memoryview) -> None:
    """CTRL_COMM_INIT handler — drives `cw.comm_init` on the chip child.

    Idempotent: ``ChipWorker.comm_init`` itself caches the handle and returns
    the existing one if already initialized, so a duplicate dispatch from the
    parent is a no-op.
    """
    request_shm_name = _read_shm_name(buf, _OFF_ARGS)
    req_shm = SharedMemory(name=request_shm_name)
    req_buf = req_shm.buf
    assert req_buf is not None
    try:
        (rank, nranks) = _COMM_INIT_HEADER.unpack_from(req_buf, 0)
        # rootinfo_path is the rest of the shm, NUL-terminated.
        raw = bytes(req_buf[_COMM_INIT_HEADER.size :])
        nul = raw.find(b"\x00")
        rootinfo_path = raw[: nul if nul >= 0 else len(raw)].decode("utf-8", "replace")
    finally:
        req_buf.release()
        req_shm.close()

    handle = cw.comm_init(int(rank), int(nranks), rootinfo_path)
    if handle == 0:
        raise RuntimeError("comm_init returned 0 handle for hidden base communicator")
    cw._comm_base_handle_cached = int(handle)


@dataclass
class _L2HostL3L2Region:
    region_id: int
    payload_bytes: int
    counter_offset: int
    counter_bytes: int
    total_bytes: int
    shm: SharedMemory | None = None
    dev_ptr: int = 0
    onboard_handle: int = 0


@dataclass
class _L2HostL3L2RegionStore:
    """Per-chip-child registry of live L3-L2 direct regions (loop-local state)."""

    regions: dict[int, _L2HostL3L2Region] = field(default_factory=dict)
    next_region_id: int = 1


@dataclass(frozen=True)
class _L2HostL3L2RegionReplyMeta:
    payload_base: int
    backing_name: bytes
    access_profile: L3L2RegionAccessProfile
    mapping_bytes: int
    shareable_handle: int


def _release_l2_host_l3_l2_region(region: _L2HostL3L2Region) -> None:
    if region.shm is not None:
        region.shm.close()
        region.shm.unlink()
        return
    if region.onboard_handle:
        _l3_child_onboard_region_close(region.onboard_handle)


def _create_sim_l3_l2_region(
    request: L3L2RegionCreateRequest, region_id: int, counter_offset: int, total_bytes: int
) -> tuple[_L2HostL3L2Region, _L2HostL3L2RegionReplyMeta]:
    shm = SharedMemory(create=True, size=total_bytes)
    region = _L2HostL3L2Region(
        region_id=region_id,
        payload_bytes=request.payload_bytes,
        counter_offset=counter_offset,
        counter_bytes=request.counter_bytes,
        total_bytes=total_bytes,
        shm=shm,
    )
    region_buf = cast(memoryview, shm.buf)
    region_buf[counter_offset : counter_offset + request.counter_bytes] = b"\x00" * request.counter_bytes
    exported = ctypes.c_char.from_buffer(region_buf)
    try:
        payload_base = ctypes.addressof(exported)
    finally:
        del exported
        del region_buf
    backing_name = shm.name.encode("utf-8")
    if len(backing_name) >= _CTRL_SHM_TOKEN_BYTES:
        raise RuntimeError("CTRL_L3_L2_REGION_CREATE backing shm token is too long")
    meta = _L2HostL3L2RegionReplyMeta(
        payload_base=payload_base,
        backing_name=backing_name,
        access_profile=L3L2RegionAccessProfile.SIM_POSIX_SHM,
        mapping_bytes=total_bytes,
        shareable_handle=0,
    )
    return region, meta


def _create_onboard_l3_l2_region(
    cw: ChipWorker, request: L3L2RegionCreateRequest, region_id: int, counter_offset: int, total_bytes: int
) -> tuple[_L2HostL3L2Region, _L2HostL3L2RegionReplyMeta]:
    export = _l3_child_onboard_region_create(total_bytes)
    dev_ptr = int(export.device_addr)
    region = _L2HostL3L2Region(
        region_id=region_id,
        payload_bytes=request.payload_bytes,
        counter_offset=counter_offset,
        counter_bytes=request.counter_bytes,
        total_bytes=total_bytes,
        dev_ptr=dev_ptr,
        onboard_handle=int(export.registry_handle),
    )
    zeros = ctypes.create_string_buffer(request.counter_bytes)
    cw.copy_to(dev_ptr + counter_offset, ctypes.addressof(zeros), request.counter_bytes)
    meta = _L2HostL3L2RegionReplyMeta(
        payload_base=dev_ptr,
        backing_name=b"",
        access_profile=L3L2RegionAccessProfile.ONBOARD_VMM,
        mapping_bytes=int(export.mapping_bytes),
        shareable_handle=int(export.shareable_handle),
    )
    return region, meta


def _handle_ctrl_l3_l2_region_create(
    cw: ChipWorker, buf: memoryview, chip_platform: str, store: _L2HostL3L2RegionStore
) -> None:
    request_shm_name = _read_shm_name(buf, _OFF_ARGS)
    reply_shm_name = _read_shm_name(buf, _OFF_ARGS + _CTRL_SHM_NAME_BYTES)
    req_shm = SharedMemory(name=request_shm_name)
    reply_shm = SharedMemory(name=reply_shm_name)
    req_buf = cast(memoryview, req_shm.buf)
    reply_buf = cast(memoryview, reply_shm.buf)
    region: _L2HostL3L2Region | None = None
    try:
        fields = _REGION_CREATE_REQUEST.unpack_from(req_buf, 0)
        request = L3L2RegionCreateRequest(
            magic_version=int(fields[0]),
            request_bytes=int(fields[1]),
            payload_bytes=int(fields[2]),
            counter_bytes=int(fields[3]),
        )
        # Reject ABI mismatches loudly. The reply carries this child's own
        # magic (not the request echo) so the L3 side can detect version skew.
        if request.magic_version != _REGION_MAGIC_VERSION:
            raise RuntimeError("CTRL_L3_L2_REGION_CREATE magic_version mismatch")
        if request.request_bytes != _REGION_CREATE_REQUEST_BYTES:
            raise RuntimeError("CTRL_L3_L2_REGION_CREATE request_bytes mismatch")
        if request.payload_bytes <= 0:
            raise RuntimeError("CTRL_L3_L2_REGION_CREATE payload_bytes must be positive")
        if request.counter_bytes <= 0 or request.counter_bytes % 4 != 0:
            raise RuntimeError("CTRL_L3_L2_REGION_CREATE counter_bytes must be positive and a multiple of 4")
        counter_offset = _align_up(request.payload_bytes, _REGION_LAYOUT_ALIGNMENT)
        total_bytes = _checked_add_u64(counter_offset, request.counter_bytes)

        region_id = store.next_region_id
        store.next_region_id += 1
        if str(chip_platform).endswith("sim"):
            region, meta = _create_sim_l3_l2_region(request, region_id, counter_offset, total_bytes)
        else:
            region, meta = _create_onboard_l3_l2_region(cw, request, region_id, counter_offset, total_bytes)
        _REGION_CREATE_REPLY.pack_into(
            reply_buf,
            0,
            _REGION_MAGIC_VERSION,
            region_id,
            meta.payload_base,
            request.payload_bytes,
            meta.payload_base + counter_offset,
            request.counter_bytes,
            int(meta.access_profile),
            0,
            int(getattr(cw, "device_id", -1)),
            meta.backing_name + b"\x00" * (_CTRL_SHM_TOKEN_BYTES - len(meta.backing_name)),
            meta.mapping_bytes,
            meta.shareable_handle,
        )
        store.regions[region_id] = region
        region = None
    finally:
        if region is not None:
            try:
                _release_l2_host_l3_l2_region(region)
            except (BufferError, FileNotFoundError, OSError, RuntimeError):
                pass
        del req_buf
        del reply_buf
        req_shm.close()
        reply_shm.close()


def _handle_ctrl_l3_l2_region_release(buf: memoryview, store: _L2HostL3L2RegionStore) -> None:
    region_id = struct.unpack_from("Q", buf, _CTRL_OFF_ARG0)[0]
    region = store.regions.pop(int(region_id), None)
    if region is None:
        return
    _release_l2_host_l3_l2_region(region)


def _sweep_l2_host_l3_l2_regions(store: _L2HostL3L2RegionStore) -> None:
    for region_id in list(store.regions):
        region = store.regions.pop(region_id)
        try:
            _release_l2_host_l3_l2_region(region)
        except (BufferError, FileNotFoundError, OSError, RuntimeError):
            pass


def _handle_ctrl_release_domain(cw: ChipWorker, buf: memoryview) -> None:
    """CTRL_RELEASE_DOMAIN handler — collective free for one allocation."""
    request_shm_name = _read_shm_name(buf, _OFF_ARGS)
    req_shm = SharedMemory(name=request_shm_name)
    req_buf = req_shm.buf
    assert req_buf is not None
    try:
        (allocation_id, rank_count, domain_rank, _ws, _bc) = _DOMAIN_REQ_HEADER.unpack_from(req_buf, 0)
    finally:
        req_buf.release()
        req_shm.close()

    handle = _comm_base_handle(cw)
    cw._impl.comm_release_domain_windows(int(handle), int(allocation_id), int(rank_count), int(domain_rank))


def _comm_base_handle(cw: ChipWorker) -> int:
    """Return the cached base-communicator handle the chip allocated during bootstrap.

    The dynamic-allocate path requires an established base communicator (HCCL
    RootInfo handshake already done).  ``bootstrap_context`` stashes the handle
    on the ChipWorker; this helper exposes it to the CTRL_* handlers.
    """
    handle = getattr(cw, "_comm_base_handle_cached", 0)
    if not handle:
        raise RuntimeError("CTRL_ALLOC_DOMAIN: chip has no base communicator — bootstrap_context must run first")
    return int(handle)


def _ensure_prepared(cw, registry, prepared, cid: int, *, device_id: int) -> None:
    if cid in prepared:
        return
    callable_obj = registry.get(cid)
    if callable_obj is None:
        raise RuntimeError(f"chip_process dev={device_id}: cid {cid} not in registry")
    cw._register_callable_at_slot(cid, callable_obj)
    prepared.add(cid)


def _run_chip_main_loop(  # noqa: PLR0912, PLR0913, PLR0915 -- unified TASK_READY / CONTROL_REQUEST state machine
    cw: ChipWorker,
    buf: memoryview,
    mailbox_addr: int,
    state_addr: int,
    device_id: int,
    registry: dict[int, Any],
    identity_table: dict[bytes, int],
    identity_refs: dict[bytes, int],
    *,
    chip_platform: str,
    chip_runtime: str = "",
    on_task_done_success=None,
    prepared: set[int] | None = None,
) -> None:
    """Unified TASK_READY / CONTROL_REQUEST / SHUTDOWN state machine.

    `on_task_done_success`, if provided, is invoked after a successful
    ``run_from_blob`` and before publishing TASK_DONE. It must
    return ``(code, msg)`` — typically ``(0, "")`` on success, or an
    error tuple if the hook itself failed (e.g. D2H staging error).
    Returning a non-zero code overrides the kernel's success.

    TASK_READY carries a callable digest. The child resolves it to a
    target-local slot and runs it. The slot must already be prepared: initial
    startup-snapshot ChipCallables are prepared before INIT_READY (carried in via
    ``prepared``), and callables registered dynamically after startup arrive via
    ``_CTRL_PREPARE``. A TASK_READY for an unprepared slot is a control-flow
    error and fails the task rather than lazily preparing it.
    """
    prepared = prepared if prepared is not None else set()
    l3_l2_region_store = _L2HostL3L2RegionStore()
    # Post-fork host buffers mapped into this child. `host_buf_table`
    # owns the mmap per token (for unmap + teardown); `host_buf_ranges` is the
    # parent-VA → child-VA translation table the per-task blob rewrite consults,
    # rebuilt from the table on every map/unmap.
    host_buf_table: dict[int, tuple[SharedMemory, int, int, int]] = {}  # token -> (shm, lo, hi, child_base)
    host_buf_ranges: list[tuple[int, int, int]] = []  # (parent_lo, parent_hi, child_base)
    try:
        while True:
            state = _mailbox_load_i32(state_addr)
            if state == _TASK_READY:
                digest = _read_task_digest(buf)
                cid = identity_table.get(digest)
                cfg = _read_config_from_mailbox(buf)

                code = 0
                msg = ""
                try:
                    if cid is None:
                        raise RuntimeError(f"callable hash {_format_digest(digest)} not registered")
                    # Run only consumes a prepared slot — it never lazily
                    # prepares. The callable must have been staged via
                    # _CTRL_PREPARE first; reaching TASK_READY without it is a
                    # control-flow bug, so fail loudly instead of masking the
                    # missing-prepare with a first-task latency spike.
                    if cid not in prepared:
                        raise RuntimeError(
                            f"chip_process dev={device_id}: cid {cid} not prepared before TASK_READY "
                            f"(register via _CTRL_PREPARE first)"
                        )
                    # Redirect any registered host pointer (a parent VA) in the
                    # blob to this child's own mapping before the runtime reads it.
                    # No-op when nothing is registered.
                    if host_buf_ranges:
                        _rewrite_blob_host_addrs(buf, _OFF_TASK_ARGS_BLOB, host_buf_ranges)
                    # Hand the mailbox bytes straight to C++ (zero-copy zero-decode):
                    # the blob layout is what `write_blob` already wrote, so re-parsing
                    # it in Python is N×40B of avoidable work and a permanent
                    # opportunity to drop a field.  C++ reinterpret_cast<ChipStorageTaskArgs*>
                    # is the source of truth.
                    cw._impl.run_from_blob(cid, mailbox_addr + _OFF_TASK_ARGS_BLOB, _MAILBOX_ARGS_CAPACITY, cfg)
                except Exception as e:  # noqa: BLE001
                    code = 1
                    msg = _format_exc(f"chip_process dev={device_id}", e)

                # On a successful kernel run, give the caller a chance to do
                # post-run work (e.g. store_to_host D2H staging) before the
                # parent sees TASK_DONE. The kernel's failure path skips the
                # hook because the device output region is undefined and
                # staging garbage would mask the real error in post-mortems.
                if code == 0 and on_task_done_success is not None:
                    code, msg = on_task_done_success()

                _write_error(buf, code, msg)
                _mailbox_store_i32(state_addr, _TASK_DONE)
            elif state == _CONTROL_REQUEST:
                sub_cmd = struct.unpack_from("Q", buf, _OFF_CALLABLE)[0]
                code = 0
                msg = ""
                try:
                    if sub_cmd == _CTRL_MALLOC:
                        size = struct.unpack_from("Q", buf, _CTRL_OFF_ARG0)[0]
                        ptr = cw.malloc(size)
                        struct.pack_into("Q", buf, _CTRL_OFF_RESULT, ptr)
                    elif sub_cmd == _CTRL_FREE:
                        ptr = struct.unpack_from("Q", buf, _CTRL_OFF_ARG0)[0]
                        cw.free(ptr)
                    elif sub_cmd == _CTRL_COPY_TO:
                        dst = struct.unpack_from("Q", buf, _CTRL_OFF_ARG0)[0]
                        src = struct.unpack_from("Q", buf, _CTRL_OFF_ARG1)[0]
                        n = struct.unpack_from("Q", buf, _CTRL_OFF_ARG2)[0]
                        cw.copy_to(dst, src, n)
                    elif sub_cmd == _CTRL_COPY_FROM:
                        dst = struct.unpack_from("Q", buf, _CTRL_OFF_ARG0)[0]
                        src = struct.unpack_from("Q", buf, _CTRL_OFF_ARG1)[0]
                        n = struct.unpack_from("Q", buf, _CTRL_OFF_ARG2)[0]
                        cw.copy_from(dst, src, n)
                    elif sub_cmd == _CTRL_PREPARE:
                        digest = _read_control_digest(buf)
                        cid = identity_table.get(digest)
                        if cid is None:
                            raise RuntimeError(
                                f"prepare chip={device_id}: callable hash {_format_digest(digest)} not registered"
                            )
                        _ensure_prepared(cw, registry, prepared, int(cid), device_id=device_id)
                    elif sub_cmd == _CTRL_REGISTER:
                        digest = _read_control_digest(buf)
                        payload_size = struct.unpack_from("Q", buf, _CTRL_OFF_ARG0)[0]
                        shm_name = _read_ctrl_staged_shm_name(buf)
                        shm = SharedMemory(name=shm_name)
                        shm_buf = shm.buf
                        assert shm_buf is not None
                        try:
                            if payload_size <= 0 or payload_size > shm.size:
                                raise RuntimeError(
                                    f"CTRL_REGISTER payload size mismatch: payload={payload_size}, shm={shm.size}"
                                )
                            callable_obj = ChipCallable.from_bytes(bytes(shm_buf[:payload_size]))
                            _validate_chip_payload_digest(
                                callable_obj,
                                digest,
                                platform=chip_platform,
                                runtime=chip_runtime,
                                context=f"chip_process dev={device_id}",
                            )
                            if digest in identity_table:
                                identity_refs[digest] = identity_refs.get(digest, 1) + 1
                            else:
                                cid = _install_local_identity(
                                    registry, identity_table, identity_refs, digest, callable_obj
                                )
                                # Self-heal when a prior unregister popped the local
                                # identity table but failed before clearing device
                                # prepared state for the reusable private slot.
                                if int(cid) in prepared:
                                    try:
                                        cw._unregister_slot(int(cid))
                                    except Exception:  # noqa: BLE001
                                        pass
                                    prepared.discard(int(cid))
                                exported = ctypes.c_char.from_buffer(shm_buf)
                                try:
                                    addr = ctypes.addressof(exported)
                                    cw._impl.register_callable_from_blob(int(cid), addr)
                                finally:
                                    del exported
                                prepared.add(int(cid))
                        finally:
                            shm_buf.release()
                            # Release the local mmap as soon as prepare returns;
                            # register_callable has already H2D-copied the bytes to
                            # device GM, so the child no longer needs the shm.
                            shm.close()
                    elif sub_cmd == _CTRL_UNREGISTER:
                        digest = _read_control_digest(buf)
                        cid, removed = _remove_local_identity(registry, identity_table, identity_refs, digest)
                        if removed and cid is not None:
                            cw._unregister_slot(int(cid))
                            prepared.discard(int(cid))
                    elif sub_cmd == _CTRL_ALLOC_DOMAIN:
                        _handle_ctrl_alloc_domain(cw, buf)
                    elif sub_cmd == _CTRL_RELEASE_DOMAIN:
                        _handle_ctrl_release_domain(cw, buf)
                    elif sub_cmd == _CTRL_COMM_INIT:
                        _handle_ctrl_comm_init(cw, buf)
                    elif sub_cmd == _CTRL_MAP_HOST:
                        _handle_ctrl_map_host(buf, host_buf_table, host_buf_ranges)
                    elif sub_cmd == _CTRL_UNMAP_HOST:
                        _handle_ctrl_unmap_host(buf, host_buf_table, host_buf_ranges)
                    elif sub_cmd == _CTRL_L3_L2_REGION_CREATE:
                        _handle_ctrl_l3_l2_region_create(cw, buf, chip_platform, l3_l2_region_store)
                    elif sub_cmd == _CTRL_L3_L2_REGION_RELEASE:
                        _handle_ctrl_l3_l2_region_release(buf, l3_l2_region_store)
                    else:
                        raise RuntimeError(f"unknown control sub-command {int(sub_cmd)}")
                except Exception as e:  # noqa: BLE001
                    code = 1
                    if sub_cmd in (_CTRL_REGISTER, _CTRL_UNREGISTER):
                        op = "register" if sub_cmd == _CTRL_REGISTER else "unregister"
                        msg = _format_exc(f"{op} hash={_format_digest(_read_control_digest(buf))} chip={device_id}", e)
                    else:
                        msg = _format_exc(f"chip_process dev={device_id} ctrl={int(sub_cmd)}", e)
                _write_error(buf, code, msg)
                _mailbox_store_i32(state_addr, _CONTROL_DONE)
            elif state == _SHUTDOWN:
                break
    finally:
        _sweep_l2_host_l3_l2_regions(l3_l2_region_store)
        for host_shm, _lo, _hi, _base in host_buf_table.values():
            try:
                host_shm.close()
            except Exception:  # noqa: BLE001
                pass


def _chip_process_loop(  # noqa: PLR0913 -- fork-child entry: all context (bins, identity tables, log config, prewarm sizing) must cross the fork as explicit COW args; the child cannot read parent state after os.fork
    buf: memoryview,
    bins,
    device_id: int,
    registry: dict[int, Any],
    identity_table: dict[bytes, int],
    identity_refs: dict[bytes, int],
    log_level: int = 1,
    log_info_v: int = 5,
    platform: str = "",
    runtime: str = "",
    prewarm_config=None,
) -> None:
    """Runs in forked child process. Loads host_runtime.so in own address space.

    `log_level` / `log_info_v` are the parent's snapshot of the simpler logger
    (computed via `_log.get_current_config()`); the child cannot read the
    parent's logger after fork, so the values are passed explicitly.

    The main loop is delegated to ``_run_chip_main_loop`` — see its docstring
    for the TASK_READY / CONTROL_REQUEST / SHUTDOWN state machine.
    """
    import traceback as _tb  # noqa: PLC0415

    try:
        cw = ChipWorker()
        cw.init(device_id, bins, log_level=log_level, log_info_v=log_info_v, prewarm_config=prewarm_config)
    except Exception as e:
        _tb.print_exc()
        # Publish the cause into the mailbox and flag INIT_FAILED so the
        # parent's readiness barrier returns a bounded error instead of
        # spinning forever on a child that will never reach INIT_READY.
        _write_error(buf, 1, _format_exc(f"chip_process dev={device_id} init", e))
        _mailbox_store_i32(_buffer_field_addr(buf, _OFF_STATE), _INIT_FAILED)
        return

    # Prepare every ChipCallable in the startup snapshot before publishing
    # INIT_READY, so the H2D upload + device-orch load is charged inside the
    # readiness barrier and the first task dispatch pays no upload. The set of
    # prepared cids carries into the main loop, which requires a cid be prepared
    # before it dispatches. The parent therefore issues no post-READY
    # control_prepare for the initial snapshot.
    prepared: set[int] = set()
    try:
        for cid, target in registry.items():
            if isinstance(target, ChipCallable):
                _ensure_prepared(cw, registry, prepared, int(cid), device_id=device_id)
    except Exception as e:
        _tb.print_exc()
        _write_error(buf, 1, _format_exc(f"chip_process dev={device_id} prepare", e))
        _mailbox_store_i32(_buffer_field_addr(buf, _OFF_STATE), _INIT_FAILED)
        cw.finalize()
        return

    mailbox_addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    state_addr = mailbox_addr + _OFF_STATE
    # Signal init success. The parent's readiness barrier waits for every chip
    # child to reach _INIT_READY before dispatching the first task, so the
    # per-rank host-side stream sync budget only covers actual op execution
    # rather than absorbing peer-rank init skew.
    _mailbox_store_i32(state_addr, _INIT_READY)
    sys.stderr.write(f"[chip_process pid={os.getpid()} dev={device_id}] ready\n")
    sys.stderr.flush()

    try:
        _run_chip_main_loop(
            cw,
            buf,
            mailbox_addr,
            state_addr,
            device_id,
            registry,
            identity_table,
            identity_refs,
            chip_platform=platform,
            chip_runtime=runtime,
            prepared=prepared,
        )
    finally:
        cw.finalize()


def _read_config_from_mailbox(buf: memoryview) -> CallConfig:
    """Reconstruct a CallConfig from the unified mailbox layout."""
    (
        block_dim,
        aicpu_tn,
        swl,
        dt,
        pmu,
        dep_gen,
        scope_stats,
        graph_cache,
        *ring_values,
        prefix_bytes,
    ) = _CFG_FMT.unpack_from(buf, _OFF_CONFIG)
    ring_task_window = list(ring_values[:RUNTIME_ENV_RING_COUNT])
    ring_heap = list(ring_values[RUNTIME_ENV_RING_COUNT : 2 * RUNTIME_ENV_RING_COUNT])
    ring_dep_pool = list(ring_values[2 * RUNTIME_ENV_RING_COUNT : 3 * RUNTIME_ENV_RING_COUNT])
    cfg = CallConfig()
    cfg.block_dim = block_dim
    cfg.aicpu_thread_num = aicpu_tn
    cfg.enable_l2_swimlane = swl
    cfg.enable_dump_args = int(dt)
    cfg.enable_pmu = pmu
    cfg.enable_dep_gen = bool(dep_gen)
    cfg.enable_scope_stats = bool(scope_stats)
    cfg.enable_graph_cache = bool(graph_cache)
    cfg.runtime_env.ring_task_window = ring_task_window
    cfg.runtime_env.ring_heap = ring_heap
    cfg.runtime_env.ring_dep_pool = ring_dep_pool
    # NUL-terminated C string in a 1024-byte field.
    cfg.output_prefix = prefix_bytes.split(b"\x00", 1)[0].decode("utf-8")
    return cfg


def _child_worker_loop(
    buf: memoryview,
    registry: dict[int, Any],
    identity_table: dict[bytes, int],
    identity_refs: dict[bytes, int],
    inner_worker: Worker,
) -> None:
    """Runs in forked child process. Any-level Worker as child of its parent.

    Polls the unified mailbox for (callable digest, config, args_blob). Looks
    up the orchestration function in the L2-side registry, then delegates to
    ``inner_worker.run(orch_fn, args, cfg)`` which opens its own scope,
    runs the orch function, and drains. Also services CONTROL_REQUEST
    so the L4 parent's dynamic register/unregister broadcasts cascade
    into the inner Worker (see docs section 7).
    """
    state_addr = _buffer_field_addr(buf, _OFF_STATE)
    while True:
        state = _mailbox_load_i32(state_addr)
        if state == _TASK_READY:
            digest = _read_task_digest(buf)
            cid = identity_table.get(digest)
            orch_fn = registry.get(int(cid)) if cid is not None else None
            code = 0
            msg = ""
            if orch_fn is None:
                code = 1
                msg = f"child_worker: callable hash {_format_digest(digest)} not registered"
            else:
                try:
                    args = _read_args_from_mailbox(buf)
                    cfg = _read_config_from_mailbox(buf)
                    inner_worker.run(orch_fn, args, cfg)
                except Exception as e:  # noqa: BLE001
                    code = 1
                    msg = _format_exc(f"child_worker level={inner_worker.level}", e)
            _write_error(buf, code, msg)
            _mailbox_store_i32(state_addr, _TASK_DONE)
        elif state == _CONTROL_REQUEST:
            sub_cmd = struct.unpack_from("Q", buf, _OFF_CALLABLE)[0]
            code = 0
            msg = ""
            try:
                if sub_cmd == _CTRL_REGISTER:
                    digest = _read_control_digest(buf)
                    payload_size = struct.unpack_from("Q", buf, _CTRL_OFF_ARG0)[0]
                    shm_name = _read_ctrl_staged_shm_name(buf)
                    callable_obj = _read_chip_callable_from_shm(shm_name, int(payload_size))
                    inner_registered = False
                    try:
                        inner_worker._register_child_chip(callable_obj, digest=digest)
                        inner_registered = True
                        _install_local_identity(
                            registry,
                            identity_table,
                            identity_refs,
                            digest,
                            callable_obj,
                        )
                    except Exception:
                        if inner_registered:
                            inner_worker._unregister_child_digest(digest=digest)
                        raise
                elif sub_cmd == _CTRL_UNREGISTER:
                    digest = _read_control_digest(buf)
                    inner_worker._unregister_child_digest(digest=digest)
                    _remove_local_identity(registry, identity_table, identity_refs, digest)
                elif sub_cmd in (_CTRL_PY_REGISTER, _CTRL_PY_IMPORT_REGISTER, _CTRL_PY_UNREGISTER):
                    _handle_py_callable_control(
                        buf,
                        registry,
                        identity_table,
                        identity_refs,
                        int(sub_cmd),
                        context=f"child_worker level={inner_worker.level}",
                    )
                else:
                    raise RuntimeError(f"unknown control sub-command {int(sub_cmd)}")
            except Exception as e:  # noqa: BLE001
                code = 1
                op = (
                    "register"
                    if sub_cmd == _CTRL_REGISTER
                    else (
                        "unregister"
                        if sub_cmd == _CTRL_UNREGISTER
                        else (
                            "py_register"
                            if sub_cmd in (_CTRL_PY_REGISTER, _CTRL_PY_IMPORT_REGISTER)
                            else ("py_unregister" if sub_cmd == _CTRL_PY_UNREGISTER else f"ctrl={int(sub_cmd)}")
                        )
                    )
                )
                msg = _format_exc(f"child_worker level={inner_worker.level} {op}", e)
            _write_error(buf, code, msg)
            _mailbox_store_i32(state_addr, _CONTROL_DONE)
        elif state == _SHUTDOWN:
            inner_worker.close()
            break


class _Lifecycle(enum.Enum):
    """The single authoritative *public-admission* lifecycle of a Worker (5
    states), guarded by ``_hierarchical_start_cv``.

    ``NEW → INITIALIZING → READY | FAILED → CLOSED``. Every level uses this
    machine: an L2 worker inits synchronously (no child barrier) but still claims
    INITIALIZING so two concurrent ``init()`` calls serialize on the same epoch.
    close() while INITIALIZING fails fast (this worker does not support
    cancelling an in-progress init); a caller must wait for READY or FAILED.

    Admission is decided solely by this state: CLOSED rejects every public
    live-tree API, permanently (close() is a commitment, not a reversible
    attempt — it never reverts to READY). "Closing in progress" is NOT a public
    state: it is a private per-attempt teardown phase (see ``_CloseAttempt``)
    that drives child teardown off *resource presence* (``_worker``/mailboxes),
    never off this lifecycle.
    """

    NEW = enum.auto()
    INITIALIZING = enum.auto()
    READY = enum.auto()
    FAILED = enum.auto()
    CLOSED = enum.auto()


class _CloseAttempt:
    """Private completion record for one close() teardown attempt.

    close() publishes CLOSED atomically and installs a fresh attempt; concurrent
    close()s pin to the attempt they observed (via ``_close_completion``) and
    wait on its ``done``, so every joiner of the same attempt sees the same
    outcome. ``incomplete=True`` means the tree was not fully reclaimed.

    Teardown is single-shot and terminal: once it *runs* (``_teardown_attempted``
    latches True), an un-reclaimed resource LEAKS — a later close() never re-drives
    a half-torn tree. The one retry path is a *drain-timeout*, which leaves
    teardown UN-attempted and the tree intact; a later close() may then drive
    drain+teardown once the in-flight operation finishes.
    """

    __slots__ = ("done", "error", "incomplete")

    def __init__(self) -> None:
        self.done: bool = False
        self.error: BaseException | None = None
        self.incomplete: bool = False


class _StartupCancelled(BaseException):
    """Raised inside a forked child when the parent cooperatively cancels its
    startup (SIGTERM). Unwinds the child's own ``setup`` — recursively rolling
    back any grandchildren it already forked — before it exits.

    Only the forked-child SIGTERM path raises this: the startup *root* is not
    cancellable (``close()`` fails fast while INITIALIZING)."""


def _forked_child_main(buf: memoryview, label: str, setup, serve, make_group_leader: bool = False) -> None:
    """Run a forked child to completion, always terminating via ``os._exit``.

    ``setup()`` runs the child's fallible init and returns an opaque context;
    any failure there publishes INIT_FAILED with the cause and exits. On
    success the child publishes INIT_READY (the parent's readiness barrier
    unblocks) and ``serve(ctx)`` runs the mailbox loop.

    ``make_group_leader`` puts the child in its own process group so the startup
    root can reap the whole subtree (this child plus every descendant it forks)
    with one ``killpg``; deeper descendants inherit the group and do not set
    their own. During ``setup`` a SIGTERM is a cooperative cancel: it raises
    ``_StartupCancelled``, which unwinds ``setup`` (recursively tearing down any
    grandchildren and their nested shms) before the child exits.

    Load-bearing invariant: a forked child must NEVER let an exception unwind
    back into the forked copy of the parent's ``_start_hierarchical`` frames.
    Those frames carry the parent's inherited child-PID lists, so an unwind
    into the startup rollback path would SIGKILL this child's *siblings* (real
    processes at those PIDs). Catch everything and exit instead.
    """
    import traceback as _tb  # noqa: PLC0415

    if make_group_leader:
        with contextlib.suppress(OSError):
            os.setpgid(0, 0)

    state_addr = _buffer_field_addr(buf, _OFF_STATE)

    def _on_cancel(_signum, _frame):
        raise _StartupCancelled()

    prev_term = signal.signal(signal.SIGTERM, _on_cancel)
    try:
        ctx = setup()
    except _StartupCancelled:
        # Parent cancelled us mid-init; setup() already unwound its own subtree.
        _tb.print_exc()
        os._exit(1)
    except BaseException as e:  # noqa: BLE001
        _tb.print_exc()
        _write_error(buf, 1, _format_exc(f"{label} init", e))
        _mailbox_store_i32(state_addr, _INIT_FAILED)
        os._exit(1)
    # Serving is torn down via the SHUTDOWN mailbox state, not the cancel signal.
    # signal.signal returns None when the prior handler was not installed from
    # Python (e.g. a C library / host default); restore SIG_DFL in that case so
    # the round-trip does not raise TypeError.
    signal.signal(signal.SIGTERM, prev_term if prev_term is not None else signal.SIG_DFL)
    _mailbox_store_i32(state_addr, _INIT_READY)
    try:
        serve(ctx)
    except BaseException:  # noqa: BLE001
        _tb.print_exc()
        os._exit(1)
    os._exit(0)


# ---------------------------------------------------------------------------
# Worker factory
# ---------------------------------------------------------------------------


class Worker:
    """Unified worker for all hierarchy levels.

    level=2: wraps the C++ ChipWorker (one NPU device).
    level=3: wraps the C++ Worker composite with ChipWorker×N + SubWorker×M,
             auto-created in init() from device_ids and num_sub_workers.
    level=4+: wraps the C++ Worker composite with Worker(level-1)×N as
              NEXT_LEVEL children + SubWorker×M. Children are added via
              add_worker() before init().
    """

    def __init__(
        self,
        level: int,
        **config,
    ) -> None:
        self.level = level
        self._config = config
        self._callable_registry: dict[int, Any] = {}
        self._identity_registry: dict[bytes, _CallableIdentityState] = {}
        self._live_handles: dict[int, bytes] = {}
        self._next_handle_id: int = 0
        self._owner_id = uuid.uuid4().hex
        self._uncertain_hashids: set[bytes] = set()
        # Single authoritative lifecycle state (see _Lifecycle). All reads and
        # writes hold _hierarchical_start_cv. `_initialized` / `_hierarchical_
        # started` are read-only views of this field, kept for call-site brevity.
        self._lifecycle = _Lifecycle.NEW
        # The first BaseException that unwound init(), captured before rollback
        # runs so every waiter observes the same original cause and a cleanup
        # error cannot overwrite it.
        self._startup_error: BaseException | None = None
        # The current/last close() teardown attempt (private teardown phase, not
        # a public lifecycle state). Concurrent close()s pin to the attempt they
        # observe and wait on its completion; None until the first close().
        self._close_completion: _CloseAttempt | None = None
        # One-way latch: True once close() has *entered* teardown. Teardown is
        # terminal — after it runs, a later close() never re-drives a half-torn
        # tree (un-reclaimed resources leak). Only a drain-timeout, which leaves
        # this False, permits a later close() to drive drain+teardown once.
        self._teardown_attempted: bool = False
        # Count of in-flight admitted operations (run / buffer / remote-memory)
        # that passed the READY gate and hold a lease. close() publishes CLOSED
        # (blocking new leases) and drains this to zero before teardown; if it
        # does NOT reach zero within the budget, teardown is deferred (the
        # attempt is marked INCOMPLETE, worker stays CLOSED) — a tree with a live
        # operation is never destroyed under it. Guarded by _hierarchical_start_cv.
        self._active_ops: int = 0
        # Per-thread lease depth. A thread inside a leased operation that calls
        # close() would drain its own never-releasing lease, so close() rejects
        # such a reentrant call (e.g. worker.close() from inside an orch fn).
        # Guarded by _hierarchical_start_cv.
        self._lease_depth: dict[int, int] = {}
        # Thread that claimed the current startup epoch (set at NEW->INITIALIZING).
        # Native objects (ChipWorker / _Worker) bind the device to the calling
        # thread (aclrtSetDevice) and are same-thread-only, so their teardown must
        # run on this thread: a non-owner close() of a READY tree is always
        # rejected — even after the owner thread has exited, because thread
        # affinity does not transfer (a foreign finalize would run against the
        # wrong / unbound device context). A close() while INITIALIZING fails
        # fast (this worker does not cancel an in-progress init); any thread may
        # join an in-flight close().
        self._init_owner_thread: threading.Thread | None = None

        # Narrow lock around `_callable_registry` mutation so concurrent
        # register / unregister calls don't trip CPython's non-atomic
        # len()+assign. The wire-level concurrency (Python control ↔ C++
        # dispatch) is now handled at the C++ boundary via mailbox_mu_, so
        # no quiescent-state guard is needed.
        self._registry_lock = threading.Lock()
        self._pending_unregister_cids: set[int] = set()
        self._pending_remote_unregister_hashids: set[bytes] = set()
        self._py_control_timeout_s = float(config.get("py_control_timeout_s", _PY_CONTROL_TIMEOUT_S))
        # Upper bound on how long the readiness barrier waits for a forked child
        # to report INIT_READY/INIT_FAILED before treating it as hung. Must be
        # finite, else the deadline can never trip and the "bounded startup"
        # guarantee is void (NaN compares false against every deadline).
        self._startup_timeout_s = float(config.get("startup_timeout_s", _STARTUP_TIMEOUT_S))
        if not (self._startup_timeout_s > 0 and math.isfinite(self._startup_timeout_s)):
            raise ValueError("Worker startup_timeout_s must be a positive finite number of seconds")
        # Per-startup bookkeeping consumed by the rollback path: PIDs the barrier
        # already reaped (must not be re-SIGKILLed — the PID may be reused) and
        # PIDs that reached their serve loop (READY → asked to close gracefully
        # so they unlink their own nested shms). Reset at each _start_hierarchical.
        self._startup_reaped_pids: set[int] = set()
        self._startup_ready_pids: set[int] = set()
        # Root-visible journal of this level's process-group-leader PIDs. On the
        # startup root each direct child is a group leader (pgid == pid), so its
        # whole inherited-group subtree — including grandchildren the leader
        # forked — is reachable by killpg(pid) even after the leader itself has
        # been reaped and dropped from the direct-pid lists. Reset per startup.
        self._startup_group_leader_pids: set[int] = set()
        # Disposition of the last rollback (graceful vs. killed PIDs); diagnostics
        # and tests read it to confirm READY children were closed, not killed.
        self._last_rollback: dict[str, list[int]] | None = None
        self._hierarchical_start_mu = threading.Lock()
        self._hierarchical_start_cv = threading.Condition(self._hierarchical_start_mu)
        # Absolute time.monotonic() deadline for the current startup epoch, set
        # once at init() and shared by every child group and recursive descendant
        # so the whole tree comes up within a single startup_timeout_s budget.
        self._startup_deadline: float = 0.0
        # True on the worker whose init() the user called (the startup root).
        # The root's direct children are process-group leaders, so the root can
        # killpg a whole subtree; nested (recursive) workers inherit their
        # parent's group and rely on the root's killpg as the hard backstop.
        self._is_startup_root: bool = True

        # Optional CallConfig whose ring sizing is pre-warmed at init() so the
        # first run() with the same sizing skips the (~800ms) cold prebuilt
        # runtime-arena build. Set by init(prewarm_config=...); None = disabled.
        self._prewarm_config: Any | None = None

        # Level-2 internals
        self._chip_worker: ChipWorker | None = None

        # Level-3+ internals
        self._worker: _Worker | None = None
        self._orch: Orchestrator | None = None
        self._chip_shms: list[SharedMemory] = []
        self._chip_pids: list[int] = []
        self._sub_shms: list[SharedMemory] = []
        self._sub_pids: list[int] = []

        # L4+ next-level Worker children (added via add_worker before init)
        self._next_level_workers: list[Worker] = []
        self._next_level_worker_ids: list[int] = []
        self._next_level_shms: list[SharedMemory] = []
        self._next_level_pids: list[int] = []
        self._remote_worker_specs: list[RemoteWorkerSpec] = []
        self._remote_worker_ids: list[int] = []
        self._remote_sessions: list[_RemoteSession] = []
        self._next_level_worker_id_count: int = 0
        self._active_remote_slot_refs: list[RemoteBufferHandle] = []
        self._pending_remote_buffer_frees: list[RemoteBufferHandle] = []
        self._pending_remote_import_releases: list[RemoteBufferHandle] = []

        # Dynamic CommDomain allocations.  Keyed by user-facing name (unique
        # among live handles).  ``orch.allocate_domain`` adds entries here;
        # ``release()`` removes them and queues a deferred backend free.
        self._live_domains: dict[str, CommDomainHandle] = {}
        # Handles whose `release()` has been called inside an orch function.
        # The backend free is deferred until after Worker.run.drain() so that
        # tasks already submitted with this domain's device_ctx / buffer_ptrs
        # see live memory through execution.
        self._pending_release_domains: list[CommDomainHandle] = []
        # Monotonic per-Worker counter; mixed into IPC barrier filenames so
        # two concurrent allocations don't share a marker file.  Wraps after
        # 2^64 allocations — far beyond any realistic Worker lifetime.
        self._next_alloc_id: int = 0
        self._alloc_id_lock = threading.Lock()
        # Base HCCL/sim communicator is built lazily on the first
        # ``orch.allocate_domain`` call (see ``_ensure_comm_base``).  We
        # keep ``Worker.init()`` cheap — it only forks chip children and
        # starts the C++ scheduler; no comm work happens there.
        self._comm_base_ready: bool = False

        self._live_l3_l2_regions: list[Any] = []
        self._l3_l2_orch_comm_host_buffers: dict[int, int] = {}

        # Live-provenance of child (kind4, device) pointers, keyed on the exact
        # ``(worker_id, device_ptr)`` composite: a raw device VA is not globally
        # unique (two chips can return the same numeric address), so a single
        # ptr->worker map would collide. Populated by malloc / allocate_domain,
        # consumed by free / copy_to / copy_from and by kind4 argument dispatch
        # so a device pointer is never freed, copied, or run on the wrong worker.
        # Guarded by ``_child_prov_lock``, which makes each op atomic. Ordering is
        # safety-first: malloc records only after the native alloc succeeds, while
        # free (and domain release) revokes BEFORE the native free, so an
        # interrupted op never leaves a freed address live. Cleared on close().
        self._child_alloc_prov: dict[tuple[int, int], _ChildProvEntry] = {}
        self._child_prov_lock = threading.Lock()

        # Post-fork zero-copy host buffers (``create_host_buffer``). Keyed by the
        # born-shared shm's mapped base (== the buffer's data_ptr); each entry maps
        # a named shm into every local L3 child so memory created after the children
        # were forked is still reachable by a later run — with no per-run copy.
        self._host_buf_registry: dict[int, _HostBufEntry] = {}
        # Immutable read snapshot for the lock-free per-submit lookup
        # (``_find_host_buf_entry``): a ``(sorted_ptrs_tuple, registry_copy)`` pair
        # rebuilt under ``_registry_lock`` on every create/free and rebound
        # atomically. The reader loads it once, so the sorted keys and the dict it
        # bisects into never mutate mid-lookup — no lock, no torn read, no
        # IndexError from a concurrent free shrinking the list. Host buffers are
        # distinct, non-overlapping allocations, so the unique candidate for an
        # address is the entry with the greatest base <= addr.
        self._host_buf_snapshot: tuple[tuple[int, ...], dict[int, _HostBufEntry]] = ((), {})
        self._host_buf_token_counter: int = 0

    @property
    def _initialized(self) -> bool:
        """True only in READY — the worker's tree is live and dispatchable.

        False once CLOSED (the moment close() claims the epoch), so a dispatch /
        register / create_host_buffer that races an in-progress close() is
        rejected rather than entering the teardown window.
        """
        return self._lifecycle is _Lifecycle.READY

    @property
    def _hierarchical_started(self) -> bool:
        """True while an L3+ hierarchy is READY (children forked, scheduler up).

        NOT true during teardown: teardown drives the children off resource
        presence (``_worker`` / child mailboxes), never off this property, so a
        CLOSED worker mid-teardown does not re-admit anything through it."""
        return self._lifecycle is _Lifecycle.READY and self.level >= 3

    def _comm_plan_rootinfo_path(self) -> str:
        """Per-Worker rootinfo path used by HCCL/sim base comm_init.

        Namespaced by parent pid + Python id(self) so two concurrent L3
        Workers in the same process do not collide on the handshake file.
        """
        tag = f"pto_multi_comm_{os.getpid()}_{id(self):x}.bin"
        return os.path.join("/tmp", tag)

    def _allocate_next_level_worker_id(self) -> int:
        worker_id = self._next_level_worker_id_count
        self._next_level_worker_id_count += 1
        return worker_id

    def add_remote_worker(self, spec: RemoteWorkerSpec) -> int:
        # Hold the lifecycle lock across the state check and the topology
        # mutation so a concurrent init() cannot freeze the topology snapshot
        # between them.
        with self._hierarchical_start_cv:
            if self._lifecycle is not _Lifecycle.NEW:
                raise RuntimeError("Worker.add_remote_worker after init")
            if self.level < 4:
                raise TypeError("Worker.add_remote_worker: remote L3 workers require a level >= 4 parent")
            if not isinstance(spec, RemoteWorkerSpec):
                raise TypeError("Worker.add_remote_worker expects a RemoteWorkerSpec")
            worker_id = self._allocate_next_level_worker_id()
            self._remote_worker_specs.append(spec)
            self._remote_worker_ids.append(worker_id)
            return worker_id

    @staticmethod
    def _parse_remote_endpoint(endpoint: str) -> tuple[str, int]:
        if endpoint.count(":") != 1:
            raise ValueError(f"RemoteWorkerSpec.endpoint must be host:port, got {endpoint!r}")
        host, port_s = endpoint.rsplit(":", 1)
        if not host:
            raise ValueError("RemoteWorkerSpec.endpoint host must be non-empty")
        port = int(port_s)
        if port <= 0 or port > 65535:
            raise ValueError(f"RemoteWorkerSpec.endpoint port out of range: {port}")
        return host, port

    @staticmethod
    def _is_wildcard_session_host(host: str) -> bool:
        return host in ("0.0.0.0", "::")

    def _remote_session_timeout_s(self) -> float:
        timeout_s = float(self._config.get("remote_session_timeout_s", 30.0))
        if not (timeout_s > 0 and math.isfinite(timeout_s)):
            raise ValueError("Worker remote_session_timeout_s must be a positive finite number of seconds")
        return timeout_s

    @staticmethod
    def _send_remote_daemon_json(sock: socket.socket, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        sock.sendall(struct.pack("<I", len(data)) + data)

    @staticmethod
    def _recv_remote_daemon_json(sock: socket.socket) -> dict[str, Any]:
        size_data = bytearray()
        while len(size_data) < 4:
            chunk = sock.recv(4 - len(size_data))
            if not chunk:
                raise EOFError("remote daemon closed before reply length")
            size_data.extend(chunk)
        size = struct.unpack("<I", bytes(size_data))[0]
        if size > 16 * 1024 * 1024:
            raise RuntimeError("remote daemon reply exceeds maximum")
        data = bytearray()
        while len(data) < size:
            chunk = sock.recv(size - len(data))
            if not chunk:
                raise EOFError("remote daemon closed before full reply")
            data.extend(chunk)
        return json.loads(bytes(data).decode("utf-8"))

    def _remote_dispatcher_entries_for_worker(self, worker_id: int) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        with self._registry_lock:
            states = list(self._identity_registry.values())
        for state in states:
            if state.target_namespace != "REMOTE_TASK_DISPATCHER":
                continue
            if worker_id not in state.eligible_worker_ids:
                continue
            if not isinstance(state.target, RemoteCallable):
                raise RuntimeError(f"remote dispatcher hashid {state.hashid} does not carry a RemoteCallable target")
            entries.append(
                {
                    "hashid": state.digest.hex(),
                    "kind": state.kind,
                    "target_registry": "REMOTE_TASK_DISPATCHER",
                    "target": state.target.target,
                }
            )
        return entries

    def _build_remote_manifest(
        self, *, spec: RemoteWorkerSpec, worker_id: int, session_id: int, startup_remaining_s: float
    ) -> dict[str, Any]:
        daemon_host, _daemon_port = self._parse_remote_endpoint(spec.endpoint)
        listen_host = spec.session_listen_host or ("127.0.0.1" if daemon_host == "localhost" else daemon_host)
        if self._is_wildcard_session_host(listen_host) and not spec.allow_wildcard_session_bind:
            raise ValueError("RemoteWorkerSpec wildcard session bind requires allow_wildcard_session_bind=True")
        return {
            "session_id": int(session_id),
            "parent_worker_level": int(self.level),
            "remote_worker_level": 3,
            "worker_id": int(worker_id),
            "platform": spec.platform,
            "runtime": spec.runtime,
            "device_ids": list(spec.device_ids),
            "num_sub_workers": int(spec.num_sub_workers),
            "heap_ring_size": self._config.get("remote_heap_ring_size", None),
            "transport": spec.transport,
            # session_timeout_s bounds the runtime command socket; startup_remaining_s
            # bounds this session's slice of the single root startup budget. They are
            # distinct: the remote must not spend runtime-command time as startup time.
            "session_timeout_s": self._remote_session_timeout_s(),
            "startup_remaining_s": float(startup_remaining_s),
            "listen_host": listen_host,
            "connect_host": daemon_host,
            "remote_task_dispatcher": self._remote_dispatcher_entries_for_worker(worker_id),
            "inner_l3_worker": [],
            "feature_flags": [],
        }

    def _open_remote_session(
        self, *, spec: RemoteWorkerSpec, worker_id: int, session_id: int, timeout_s: float, startup_remaining_s: float
    ) -> _RemoteSession:
        daemon_host, daemon_port = self._parse_remote_endpoint(spec.endpoint)
        manifest = self._build_remote_manifest(
            spec=spec, worker_id=worker_id, session_id=session_id, startup_remaining_s=startup_remaining_s
        )
        with socket.create_connection((daemon_host, daemon_port), timeout=timeout_s) as sock:
            sock.settimeout(timeout_s)
            self._send_remote_daemon_json(sock, manifest)
            reply = self._recv_remote_daemon_json(sock)
        if not reply.get("ok", False):
            raise RuntimeError(f"remote L3 session startup failed for worker {worker_id}: {reply.get('error')}")
        return _RemoteSession(
            worker_id=worker_id,
            session_id=session_id,
            command_host=str(reply["command_host"]),
            command_port=int(reply["command_port"]),
            health_host=str(reply["health_host"]),
            health_port=int(reply["health_port"]),
            pid=int(reply.get("pid", 0)),
        )

    def _close_remote_session(self, session: _RemoteSession, *, timeout_s: float = 1.0) -> None:
        """Best-effort protocol shutdown for a remote L3 session."""

        from .remote_l3_protocol import FrameHeader, FrameType, send_frame  # noqa: PLC0415

        try:
            with socket.create_connection((session.command_host, session.command_port), timeout=timeout_s) as sock:
                sock.settimeout(timeout_s)
                send_frame(sock, FrameHeader(FrameType.SHUTDOWN, session.session_id, session.worker_id, 0))
        except BaseException:  # noqa: BLE001
            pass

    def _close_remote_sessions(self, sessions: list[_RemoteSession]) -> None:
        for session in reversed(sessions):
            self._close_remote_session(session)

    def _require_remote_worker_started(self, worker_id: int) -> None:
        """Argument + resource gate for the public remote-memory APIs. Admission
        (READY) is decided by the ``_operation_lease`` these APIs already hold —
        this checks only worker id, level, and transport **presence** (not the
        public lifecycle), so an operation legitimately admitted before a
        concurrent ``close()`` published CLOSED still completes during the drain
        instead of spuriously failing."""
        if self.level < 4:
            raise TypeError("remote memory APIs require a level >= 4 parent Worker")
        if int(worker_id) not in set(self._remote_worker_ids):
            raise ValueError("remote memory APIs require a remote worker id returned by add_remote_worker")
        if self._worker is None:
            raise RuntimeError("remote memory APIs require a started hierarchical Worker")

    def _require_remote_transport(self, worker_id: int) -> None:
        """Internal transport gate for the low-level ``_send_*`` helpers, which
        also run from close()'s teardown (lifecycle is already CLOSED then).
        Gated purely on *resource presence* — the C++ ``_worker`` / remote
        sockets are still up until ``_worker.close()`` nulls it — never on the
        public lifecycle, so teardown keeps its capability without re-opening
        public admission. Public entrypoints validate READY separately via
        ``_require_remote_worker_started``."""
        if int(worker_id) not in set(self._remote_worker_ids):
            raise ValueError("remote memory APIs require a remote worker id returned by add_remote_worker")
        if self._worker is None:
            raise RuntimeError("remote memory APIs require a started hierarchical Worker")

    @staticmethod
    def _host_ptr_value(ptr: Any) -> int:
        if isinstance(ptr, int):
            return int(ptr)
        if isinstance(ptr, ctypes.c_void_p):
            if ptr.value is None:
                raise ValueError("host_ptr must not be NULL")
            return int(ptr.value)
        data_ptr = getattr(ptr, "data_ptr", None)
        if callable(data_ptr):
            data_ptr_value: Any = data_ptr()
            return int(data_ptr_value)
        try:
            return ctypes.addressof(ptr)
        except TypeError:
            pass
        try:
            return ctypes.addressof(ptr.contents)
        except AttributeError as exc:
            raise TypeError("host_ptr must be an integer address, ctypes object, or object with data_ptr()") from exc

    def _require_live_remote_buffer(self, handle: RemoteBufferHandle) -> None:
        if not isinstance(handle, RemoteBufferHandle):
            raise TypeError("expected a RemoteBufferHandle returned by Worker.remote_malloc/import")
        if handle.address_space == RemoteAddressSpace.HOST_INLINE:
            raise ValueError("HOST_INLINE RemoteBufferHandle is not a remote allocation")
        if handle.released:
            raise RuntimeError("RemoteBufferHandle has already been released")
        self._require_remote_worker_started(handle.worker_id)

    @staticmethod
    def _remote_access_flags(access: str | int) -> int:
        if isinstance(access, str):
            normalized = access.strip().lower().replace("_", "").replace("-", "")
            if normalized in ("read", "r"):
                return 1
            if normalized in ("write", "w"):
                return 2
            if normalized in ("readwrite", "rw", "writeread", "wr"):
                return 3
            raise ValueError("remote buffer access must be 'read', 'write', or 'readwrite'")
        flags = int(access)
        if flags <= 0 or flags & ~0x3:
            raise ValueError("remote buffer access flags must use read/write bits")
        return flags

    def _send_remote_free(self, handle: RemoteBufferHandle) -> None:
        if handle.is_imported:
            raise ValueError("remote_free is invalid for imported handles; use remote_release_import")
        self._require_remote_transport(handle.worker_id)
        assert self._worker is not None
        self._worker.remote_free(handle.worker_id, handle._buffer_id, handle._generation)

    def _send_remote_release_import(self, handle: RemoteBufferHandle) -> None:
        if not handle.is_imported:
            raise ValueError("remote_release_import expects an imported remote handle")
        self._require_remote_transport(handle.worker_id)
        assert self._worker is not None
        self._worker.remote_release_import(
            handle.worker_id,
            handle.owner_worker_id,
            handle._buffer_id,
            handle._generation,
            handle.import_id,
        )

    def _send_remote_release_import_fields(self, fields: Any) -> None:
        worker_id = int(fields[0])
        self._require_remote_transport(worker_id)
        assert self._worker is not None
        self._worker.remote_release_import(
            worker_id,
            int(fields[1]),
            int(fields[2]),
            int(fields[3]),
            int(fields[4]),
        )

    def remote_malloc(self, *, worker: int, nbytes: int) -> RemoteBufferHandle:
        worker_id = int(worker)
        size = int(nbytes)
        if size <= 0:
            raise ValueError("Worker.remote_malloc nbytes must be positive")
        with self._operation_lease("remote_malloc"):
            self._require_remote_worker_started(worker_id)
            assert self._worker is not None
            fields = self._worker.remote_malloc(worker_id, size)
            return RemoteBufferHandle._from_remote_allocation(
                worker_id=int(fields[0]),
                buffer_id=int(fields[1]),
                generation=int(fields[2]),
                address_space=RemoteAddressSpace(int(fields[3])),
                nbytes=int(fields[4]),
                remote_addr=int(fields[5]),
                rkey_or_token=int(fields[6]),
                ub_ldst_va=int(fields[7]),
            )

    def remote_free(self, handle: RemoteBufferHandle) -> None:
        if not isinstance(handle, RemoteBufferHandle):
            raise TypeError("expected a RemoteBufferHandle returned by Worker.remote_malloc/import")
        if handle.address_space == RemoteAddressSpace.HOST_INLINE:
            raise ValueError("HOST_INLINE RemoteBufferHandle is not a remote allocation")
        if handle.is_imported:
            raise ValueError("remote_free is invalid for imported handles; use remote_release_import")
        if handle.released:
            return
        # Public admission: READY-only + drained. (The private _send_* transport
        # helper accepts CLOSED so teardown can flush pending frees, so remote_free
        # must fence admission itself rather than lean on the transport gate.)
        with self._operation_lease("remote_free"):
            if handle._live_slot_refs > 0 or handle._live_import_refs > 0:
                handle._mark_released()
                if handle not in self._pending_remote_buffer_frees:
                    self._pending_remote_buffer_frees.append(handle)
                return
            self._send_remote_free(handle)
            handle._mark_released()

    def remote_copy_to(self, handle: RemoteBufferHandle, host_ptr: Any, nbytes: int, *, offset: int = 0) -> None:
        with self._operation_lease("remote_copy_to"):
            self._require_live_remote_buffer(handle)
            if handle.is_imported:
                raise ValueError("Worker.remote_copy_to expects an owner remote buffer handle")
            size = int(nbytes)
            start = int(offset)
            if size < 0 or start < 0:
                raise ValueError("Worker.remote_copy_to size and offset must be non-negative")
            if start + size > handle.nbytes:
                raise ValueError("Worker.remote_copy_to range exceeds RemoteBufferHandle.nbytes")
            assert self._worker is not None
            self._worker.remote_copy_to(
                handle.worker_id,
                handle._buffer_id,
                handle._generation,
                start,
                self._host_ptr_value(host_ptr),
                size,
                handle.nbytes,
            )

    def remote_copy_from(self, handle: RemoteBufferHandle, host_ptr: Any, nbytes: int, *, offset: int = 0) -> None:
        with self._operation_lease("remote_copy_from"):
            self._require_live_remote_buffer(handle)
            if handle.is_imported:
                raise ValueError("Worker.remote_copy_from expects an owner remote buffer handle")
            size = int(nbytes)
            start = int(offset)
            if size < 0 or start < 0:
                raise ValueError("Worker.remote_copy_from size and offset must be non-negative")
            if start + size > handle.nbytes:
                raise ValueError("Worker.remote_copy_from range exceeds RemoteBufferHandle.nbytes")
            assert self._worker is not None
            self._worker.remote_copy_from(
                self._host_ptr_value(host_ptr),
                handle.worker_id,
                handle._buffer_id,
                handle._generation,
                start,
                size,
                handle.nbytes,
            )

    def remote_export(
        self,
        handle: RemoteBufferHandle,
        *,
        offset: int = 0,
        nbytes: int | None = None,
        access: str | int = "readwrite",
        transport_profile: str = "sim",
    ) -> RemoteBufferExport:
        with self._operation_lease("remote_export"):
            return self._remote_export_locked(
                handle, offset=offset, nbytes=nbytes, access=access, transport_profile=transport_profile
            )

    def _remote_export_locked(
        self,
        handle: RemoteBufferHandle,
        *,
        offset: int = 0,
        nbytes: int | None = None,
        access: str | int = "readwrite",
        transport_profile: str = "sim",
    ) -> RemoteBufferExport:
        self._require_live_remote_buffer(handle)
        if handle.is_imported:
            raise ValueError("Worker.remote_export expects an owner remote buffer handle")
        start = int(offset)
        size = handle.nbytes - start if nbytes is None else int(nbytes)
        if start < 0 or size <= 0:
            raise ValueError("Worker.remote_export offset must be non-negative and nbytes must be positive")
        if start + size > handle.nbytes:
            raise ValueError("Worker.remote_export range exceeds RemoteBufferHandle.nbytes")
        flags = self._remote_access_flags(access)
        if flags & ~handle.access_flags:
            raise ValueError("Worker.remote_export requested access is not allowed by handle")
        assert self._worker is not None
        fields = self._worker.remote_export(
            handle.owner_worker_id,
            handle._buffer_id,
            handle._generation,
            handle._offset,
            start,
            size,
            flags,
            str(transport_profile),
            handle.nbytes,
        )
        return RemoteBufferExport._from_remote_export(
            owner_worker_id=int(fields[0]),
            buffer_id=int(fields[1]),
            generation=int(fields[2]),
            address_space=RemoteAddressSpace(int(fields[3])),
            offset=int(fields[4]),
            nbytes=int(fields[5]),
            export_id=int(fields[6]),
            remote_addr=int(fields[7]),
            rkey_or_token=int(fields[8]),
            ub_ldst_va=int(fields[9]),
            access_flags=int(fields[10]),
            transport_profile=str(fields[11]),
            transport_descriptor=bytes(fields[12]),
            _owner_handle=handle,
            worker_owner_id=self._owner_id,
        )

    def remote_import(
        self, exported: RemoteBufferExport, *, worker: int, access: str | int | None = None
    ) -> RemoteBufferHandle:
        # Argument validation (type / forged / stale) is independent of lifecycle
        # and runs before admission; the lease guards the actual transport.
        if not isinstance(exported, RemoteBufferExport):
            raise TypeError("Worker.remote_import expects a RemoteBufferExport returned by remote_export")
        if exported._worker_owner_id != self._owner_id:
            raise ValueError("Worker.remote_import rejects forged or different Worker RemoteBufferExport values")
        if exported._owner_handle is not None and exported._owner_handle.released:
            raise ValueError("Worker.remote_import rejects stale RemoteBufferExport values for released buffers")
        with self._operation_lease("remote_import"):
            return self._remote_import_locked(exported, worker=worker, access=access)

    def _remote_import_locked(
        self, exported: RemoteBufferExport, *, worker: int, access: str | int | None = None
    ) -> RemoteBufferHandle:
        importer_worker_id = int(worker)
        self._require_remote_worker_started(importer_worker_id)
        flags = exported._access_flags if access is None else self._remote_access_flags(access)
        if flags & ~exported._access_flags:
            raise ValueError("Worker.remote_import requested access is not a subset of export access")
        assert self._worker is not None
        owner_handle = exported._owner_handle
        if owner_handle is not None:
            owner_handle._acquire_import_ref()
        fields: Any | None = None
        try:
            fields = self._worker.remote_import(
                importer_worker_id,
                exported._owner_worker_id,
                exported._buffer_id,
                exported._generation,
                int(exported._address_space),
                exported._offset,
                exported._nbytes,
                exported._export_id,
                exported._remote_addr,
                exported._rkey_or_token,
                exported._ub_ldst_va,
                exported._access_flags,
                exported._transport_profile,
                exported._transport_descriptor,
                flags,
            )
            return RemoteBufferHandle._from_imported_mapping(
                worker_id=int(fields[0]),
                owner_worker_id=int(fields[1]),
                buffer_id=int(fields[2]),
                generation=int(fields[3]),
                import_id=int(fields[4]),
                address_space=RemoteAddressSpace(int(fields[5])),
                nbytes=int(fields[6]),
                offset=int(fields[7]),
                remote_addr=int(fields[8]),
                rkey_or_token=int(fields[9]),
                ub_ldst_va=int(fields[10]),
                access_flags=int(fields[11]),
                owner_handle_ref=owner_handle,
            )
        except BaseException:
            if fields is not None:
                try:
                    self._send_remote_release_import_fields(fields)
                except Exception:  # noqa: BLE001
                    pass
            if owner_handle is not None:
                owner_handle._release_import_ref()
            raise

    def remote_release_import(self, handle: RemoteBufferHandle) -> None:
        if not isinstance(handle, RemoteBufferHandle):
            raise TypeError("expected a RemoteBufferHandle returned by Worker.remote_import")
        if not handle.is_imported:
            raise ValueError("Worker.remote_release_import expects an imported remote handle")
        if handle.released:
            return
        # Public admission: READY-only + drained (the private _send_* transport
        # accepts CLOSED for teardown, so fence admission here).
        with self._operation_lease("remote_release_import"):
            if handle._live_slot_refs > 0:
                handle._mark_released()
                if handle not in self._pending_remote_import_releases:
                    self._pending_remote_import_releases.append(handle)
                return
            self._send_remote_release_import(handle)
            if handle._owner_handle_ref is not None:
                handle._owner_handle_ref._release_import_ref()
                handle._owner_handle_ref = None
            handle._mark_released()
            self._flush_pending_remote_frees()

    def _capture_remote_sidecar_refs(self, remote_sidecar: Any) -> list[RemoteBufferHandle]:
        captured: list[RemoteBufferHandle] = []
        if remote_sidecar is None:
            return captured
        try:
            for tensor_sidecar in getattr(remote_sidecar, "tensors", ()):
                if tensor_sidecar is None or not getattr(tensor_sidecar, "present", False):
                    continue
                handle = getattr(tensor_sidecar, "handle", None)
                if handle is None:
                    continue
                if not isinstance(handle, RemoteBufferHandle):
                    raise TypeError("remote sidecar handle must be a RemoteBufferHandle")
                handle._acquire_slot_ref()
                captured.append(handle)
        except BaseException:
            self._release_remote_slot_refs(captured)
            raise
        return captured

    def _adopt_remote_slot_refs(self, handles: list[RemoteBufferHandle]) -> None:
        self._active_remote_slot_refs.extend(handles)

    def _release_remote_slot_refs(self, handles: list[RemoteBufferHandle]) -> None:
        for handle in handles:
            handle._release_slot_ref()

    def _release_active_remote_slot_refs(self) -> None:
        refs = self._active_remote_slot_refs
        self._active_remote_slot_refs = []
        self._release_remote_slot_refs(refs)

    def _flush_pending_remote_frees(self) -> None:
        errors: list[str] = []
        pending_imports = self._pending_remote_import_releases
        self._pending_remote_import_releases = []
        remaining_imports: list[RemoteBufferHandle] = []
        for handle in pending_imports:
            if handle._live_slot_refs > 0:
                remaining_imports.append(handle)
                continue
            try:
                self._send_remote_release_import(handle)
            except Exception as exc:  # noqa: BLE001
                remaining_imports.append(handle)
                errors.append(f"release_import worker_id={handle.worker_id} import_id={handle.import_id}: {exc}")
                continue
            if handle._owner_handle_ref is not None:
                handle._owner_handle_ref._release_import_ref()
                handle._owner_handle_ref = None
        self._pending_remote_import_releases.extend(remaining_imports)

        pending = self._pending_remote_buffer_frees
        self._pending_remote_buffer_frees = []
        remaining: list[RemoteBufferHandle] = []
        for handle in pending:
            if handle._live_slot_refs > 0 or handle._live_import_refs > 0:
                remaining.append(handle)
                continue
            try:
                self._send_remote_free(handle)
            except Exception as exc:  # noqa: BLE001
                remaining.append(handle)
                errors.append(f"free worker_id={handle.worker_id} buffer_id={handle._buffer_id}: {exc}")
                continue
        self._pending_remote_buffer_frees.extend(remaining)
        if errors:
            sys.stderr.write(
                "Worker._flush_pending_remote_frees(): deferred remote buffer cleanup after control error. "
                f"First error: {errors[0]}\n"
            )
            sys.stderr.flush()

    # ------------------------------------------------------------------
    # Callable registration (before init)
    # ------------------------------------------------------------------

    def _make_handle_locked(self, state: _CallableIdentityState) -> CallableHandle:
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

    def _install_registration_locked(self, reg: _CallableRegistration) -> tuple[CallableHandle, bool]:
        if reg.digest in self._uncertain_hashids:
            raise RuntimeError(f"REGISTER_CLEANUP_UNCERTAIN: {reg.hashid}")
        state = self._identity_registry.get(reg.digest)
        if state is not None:
            if state.slot_id in self._pending_unregister_cids:
                raise RuntimeError(f"REGISTER_TOMBSTONE_ACTIVE: {reg.hashid}")
            if state.descriptor != reg.descriptor or state.kind != reg.kind:
                raise RuntimeError(f"HASHID_DESCRIPTOR_MISMATCH: {reg.hashid}")
            if state.eligible_worker_ids != reg.eligible_worker_ids:
                raise RuntimeError(f"REMOTE_CALLABLE_ENDPOINT_SCOPE_MISMATCH: {reg.hashid}")
            state.ref_count += 1
            return self._make_handle_locked(state), False

        is_remote = reg.target_namespace == "REMOTE_TASK_DISPATCHER"
        slot_id = -1 if is_remote else self._allocate_cid()
        state = _CallableIdentityState(
            hashid=reg.hashid,
            digest=reg.digest,
            kind=reg.kind,  # type: ignore[arg-type]
            target_namespace=reg.target_namespace,  # type: ignore[arg-type]
            descriptor=reg.descriptor,
            payload_digest=reg.payload_digest,
            slot_id=slot_id,
            target=reg.target,
            ref_count=1,
            eligible_worker_ids=reg.eligible_worker_ids,
        )
        self._identity_registry[reg.digest] = state
        if not is_remote:
            self._callable_registry[slot_id] = reg.target
        return self._make_handle_locked(state), True

    def _rollback_handle_locked(self, handle: CallableHandle) -> None:
        state = self._identity_registry.get(handle.digest)
        self._live_handles.pop(handle._handle_id, None)
        if state is None:
            return
        state.ref_count -= 1
        if state.ref_count > 0:
            return
        if state.slot_id in self._pending_unregister_cids:
            self._pending_unregister_cids.discard(state.slot_id)
        if state.slot_id >= 0:
            self._callable_registry.pop(state.slot_id, None)
        self._identity_registry.pop(state.digest, None)

    def _resolve_handle_locked(
        self,
        handle: CallableHandle,
        *,
        expected_namespace: str | None = None,
    ) -> _CallableIdentityState:
        if not isinstance(handle, CallableHandle):
            raise TypeError("expected a CallableHandle returned by Worker.register")
        if handle._owner_id != self._owner_id:
            raise KeyError(f"CallableHandle {handle.hashid} does not belong to this Worker")
        digest = self._live_handles.get(handle._handle_id)
        if digest is None or digest != handle.digest:
            raise KeyError(f"CallableHandle {handle.hashid} is not live on this Worker")
        if digest in self._uncertain_hashids:
            raise RuntimeError(f"REGISTER_CLEANUP_UNCERTAIN: {handle.hashid}")
        state = self._identity_registry.get(digest)
        if state is None:
            raise KeyError(f"CallableHandle {handle.hashid} is not registered")
        if (
            handle.hashid != state.hashid
            or handle.kind != state.kind
            or handle.target_namespace != state.target_namespace
        ):
            raise RuntimeError(f"CALLABLE_HANDLE_MUTATED: {handle.hashid}")
        if expected_namespace is not None and state.target_namespace != expected_namespace:
            raise TypeError(f"cannot run {state.target_namespace}; expected {expected_namespace} for {state.hashid}")
        return state

    def _resolve_handle(
        self,
        handle: CallableHandle,
        *,
        expected_namespace: str | None = None,
    ) -> _CallableIdentityState:
        with self._registry_lock:
            return self._resolve_handle_locked(handle, expected_namespace=expected_namespace)

    def _wait_out_init_locked(self, api: str) -> None:
        """Block while an epoch is INITIALIZING, then reject a terminal epoch.

        Must hold ``_hierarchical_start_cv``. Returns with the lifecycle in a
        non-INITIALIZING state; raises on FAILED (re-raising the original
        startup cause) or CLOSED so a mutation never lands on a dead epoch.
        """
        while self._lifecycle is _Lifecycle.INITIALIZING:
            self._hierarchical_start_cv.wait()
        if self._lifecycle is _Lifecycle.FAILED:
            raise RuntimeError(
                f"Worker.{api}: hierarchical startup failed; close this Worker and create a new one"
            ) from self._startup_error
        if self._lifecycle is _Lifecycle.CLOSED:
            # A register/unregister that lost the wake race to a concurrent
            # close() still sees the original startup cause if one was recorded
            # (a FAILED epoch that close() then reaped), not just "closed".
            raise RuntimeError(f"Worker.{api}: worker is closed") from self._startup_error

    @contextlib.contextmanager
    def _operation_lease(self, api: str):
        """Admit an operation onto a READY worker and hold a lease for its whole
        duration, so a concurrent close() drains it before teardown.

        Fail-fast: admits only a READY worker (a non-READY worker — NEW,
        INITIALIZING, CLOSED, FAILED — is rejected immediately, not waited on,
        per the state/API matrix for dispatch/buffer). The lease is
        released on exit and wakes a close() that is draining. Use around any API
        that touches the live tree and can run past its admission check (run /
        host-buffer / remote-memory)."""
        tid = threading.get_ident()
        with self._hierarchical_start_cv:
            if self._lifecycle is not _Lifecycle.READY:
                raise RuntimeError(f"Worker.{api}: requires an initialized (READY) worker") from self._startup_error
            self._active_ops += 1
            self._lease_depth[tid] = self._lease_depth.get(tid, 0) + 1
        try:
            yield
        finally:
            with self._hierarchical_start_cv:
                self._active_ops -= 1
                depth = self._lease_depth.get(tid, 0) - 1
                if depth <= 0:
                    self._lease_depth.pop(tid, None)
                else:
                    self._lease_depth[tid] = depth
                self._hierarchical_start_cv.notify_all()

    def _register_into_snapshot_or_wait(self, reg: _CallableRegistration) -> CallableHandle | None:
        """Linearize a level>=3 register against the startup epoch.

        Waits out an in-progress init() (INITIALIZING); a FAILED or CLOSED epoch
        raises. A pre-start (NEW) registration is installed into the startup
        snapshot and its handle returned; once the hierarchy is READY, returns
        None so the caller takes its post-start control-broadcast path.
        """
        with self._hierarchical_start_cv:
            self._wait_out_init_locked("register")
            if self._lifecycle is _Lifecycle.NEW:
                with self._registry_lock:
                    handle, _is_new = self._install_registration_locked(reg)
                return handle
        return None

    def register(self, target, *, workers: list[int] | None = None) -> CallableHandle:
        """Register a callable for dispatch and return an opaque handle.

        Integer execution slots remain private to the local target process.
        Submit APIs consume the returned handle and dispatch by its stable
        SHA-256 callable identity.

        Target eligibility (a callable's kind having a resolving child) is
        checked only at init(), over the pre-init registrations
        (``_validate_eligible_targets``). A post-init dynamic register does NOT
        re-validate against the frozen topology, so registering e.g. a
        ChipCallable on a chipless worker yields a handle that never dispatches.
        Unifying the two paths is a follow-up (needs a device-free chip-child
        test harness).
        """
        if isinstance(target, RemoteCallable) and self.level < 4:
            raise TypeError("Worker.register(RemoteCallable): remote L3 dispatch requires a level >= 4 parent")
        if self.level == 2 and not isinstance(target, ChipCallable):
            raise TypeError("Worker.register: level 2 only supports ChipCallable targets")
        reg = _build_callable_registration(self, target, workers=workers)
        if isinstance(target, RemoteCallable):
            if not self._remote_worker_specs:
                raise RuntimeError("Worker.register(RemoteCallable): add at least one remote worker first")
            remote_worker_ids = set(self._remote_worker_ids)
            for worker_id in reg.eligible_worker_ids:
                if worker_id not in remote_worker_ids:
                    raise ValueError(
                        "Worker.register(RemoteCallable): workers must name remote worker ids returned by "
                        "add_remote_worker"
                    )
            # Linearize against the startup epoch exactly like the local path: a
            # register that races an in-progress init() waits for it, then a
            # pre-start registration lands in the snapshot while a post-READY one
            # goes through the remote prepare/commit control path.
            handle = self._register_into_snapshot_or_wait(reg)
            if handle is not None:
                return handle
            # Post-start broadcast touches the live tree; hold a lease so close()
            # drains it before teardown (re-checks READY, closing the
            # gate-then-teardown race).
            with self._operation_lease("register"):
                return self._post_start_register_remote(reg)
        if self.level >= 3:
            handle = self._register_into_snapshot_or_wait(reg)
            if handle is not None:
                return handle
            if not isinstance(target, ChipCallable):
                with self._operation_lease("register"):
                    return self._post_start_register_python(reg)
        else:
            # L2 has no pre-start snapshot, but still linearizes against the
            # epoch: reject a terminal (CLOSED/FAILED) worker and wait out an
            # in-progress init so the callable is installed and its device slot
            # prepared after READY — never left registered-but-not-prepared, and
            # never accepted onto a closed worker as an inert handle.
            with self._hierarchical_start_cv:
                self._wait_out_init_locked("register")

        with self._registry_lock:
            handle, is_new = self._install_registration_locked(reg)

        # L3+ post-init ChipCallable: broadcast to chip / next-level children
        # via C++ after L3 Host-side slot allocation is complete. The slot is
        # target-private; task dispatches carry only handle.digest.
        if self.level >= 3 and self._initialized and isinstance(target, ChipCallable):
            try:
                with self._operation_lease("register"):
                    self._post_init_register(target, handle.digest, is_new=is_new)
            except Exception:
                with self._registry_lock:
                    self._rollback_handle_locked(handle)
                raise

        # L2 post-init: pre-warm immediately so the very first run(handle, …)
        # is a clean cache hit.
        if self.level == 2 and self._initialized and isinstance(target, ChipCallable) and is_new:
            assert self._chip_worker is not None
            with self._registry_lock:
                slot_id = self._identity_registry[handle.digest].slot_id
            with self._operation_lease("register"):
                self._chip_worker._register_callable_at_slot(slot_id, target)
        return handle

    def _python_worker_types(self) -> list[WorkerType]:
        worker_types: list[WorkerType] = []
        if self._config.get("num_sub_workers", 0) > 0:
            worker_types.append(WorkerType.SUB)
        if self._next_level_workers:
            worker_types.append(WorkerType.NEXT_LEVEL)
        return worker_types

    def _post_start_register_python(self, reg: _CallableRegistration) -> CallableHandle:
        worker_types = self._python_worker_types()
        if not worker_types:
            raise RuntimeError(
                "Worker.register: no Python-capable child workers are configured "
                "for dynamic Python callable registration"
            )
        with self._registry_lock:
            handle, _is_new = self._install_registration_locked(reg)
        try:
            results = self._broadcast_py_control_results(
                worker_types,
                _CTRL_PY_REGISTER,
                digest=handle.digest,
                payload=reg.payload,
            )
            errors = self._control_errors(results)
            if errors:
                cleanup_errors = self._cleanup_control_successes(results, _CTRL_PY_UNREGISTER, handle.digest)
                if cleanup_errors:
                    with self._registry_lock:
                        self._uncertain_hashids.add(handle.digest)
                raise RuntimeError(self._format_register_partial_failure(handle.digest, errors, cleanup_errors))
        except Exception:
            with self._registry_lock:
                self._rollback_handle_locked(handle)
            raise
        return handle

    @staticmethod
    def _format_remote_control_exception(worker_id: int, exc: BaseException) -> str:
        return f"NEXT_LEVEL[{int(worker_id)}]: {type(exc).__name__}: {exc}"

    def _post_start_register_remote(  # noqa: PLR0912 -- two-phase remote register/commit cleanup paths
        self, reg: _CallableRegistration
    ) -> CallableHandle:
        assert reg.target_namespace == "REMOTE_TASK_DISPATCHER"
        with self._registry_lock:
            state = self._identity_registry.get(reg.digest)
            if state is not None:
                handle, _is_new = self._install_registration_locked(reg)
                return handle

        if self._worker is None:
            raise RuntimeError("Worker.register(RemoteCallable): hierarchical worker is not started")

        prepared: list[int] = []
        errors: list[str] = []
        direct_error = False
        payload = reg.payload if reg.payload is not None else b""
        for worker_id in reg.eligible_worker_ids:
            try:
                result = self._worker.remote_prepare_register(
                    worker_id,
                    "REMOTE_TASK_DISPATCHER",
                    reg.kind,
                    payload,
                    reg.digest,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(self._format_remote_control_exception(worker_id, exc))
                direct_error = True
                break
            if result.ok:
                prepared.append(worker_id)
            else:
                errors.append(f"{result.worker_type}[{result.worker_id}]: {result.error_message}")
                break
        if errors:
            cleanup_errors = self._remote_abort_prepared(prepared, reg)
            if cleanup_errors or direct_error:
                with self._registry_lock:
                    self._uncertain_hashids.add(reg.digest)
            raise RuntimeError(self._format_register_partial_failure(reg.digest, errors, cleanup_errors))

        committed: list[int] = []
        direct_error = False
        for worker_id in reg.eligible_worker_ids:
            try:
                result = self._worker.remote_commit_register(
                    worker_id,
                    "REMOTE_TASK_DISPATCHER",
                    reg.kind,
                    reg.digest,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(self._format_remote_control_exception(worker_id, exc))
                direct_error = True
                break
            if result.ok:
                committed.append(worker_id)
            else:
                errors.append(f"{result.worker_type}[{result.worker_id}]: {result.error_message}")
                break
        if errors:
            cleanup_errors = self._remote_abort_prepared(
                [worker_id for worker_id in prepared if worker_id not in committed], reg
            )
            cleanup_errors.extend(self._remote_unregister_committed(committed, reg))
            if cleanup_errors or direct_error:
                with self._registry_lock:
                    self._uncertain_hashids.add(reg.digest)
            raise RuntimeError(self._format_register_partial_failure(reg.digest, errors, cleanup_errors))

        try:
            with self._registry_lock:
                handle, _is_new = self._install_registration_locked(reg)
            return handle
        except Exception:
            cleanup_errors = self._remote_unregister_committed(committed, reg)
            if cleanup_errors:
                with self._registry_lock:
                    self._uncertain_hashids.add(reg.digest)
            raise

    def _remote_abort_prepared(self, worker_ids: list[int], reg: _CallableRegistration) -> list[str]:
        if self._worker is None:
            return []
        errors: list[str] = []
        for worker_id in worker_ids:
            try:
                result = self._worker.remote_abort_register(
                    worker_id,
                    "REMOTE_TASK_DISPATCHER",
                    reg.kind,
                    reg.digest,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(self._format_remote_control_exception(worker_id, exc))
                continue
            if not result.ok:
                errors.append(f"{result.worker_type}[{result.worker_id}]: {result.error_message}")
        return errors

    def _remote_unregister_committed(self, worker_ids: list[int], reg: _CallableRegistration) -> list[str]:
        if self._worker is None:
            return []
        errors: list[str] = []
        for worker_id in worker_ids:
            try:
                result = self._worker.remote_unregister(
                    worker_id,
                    "REMOTE_TASK_DISPATCHER",
                    reg.kind,
                    reg.digest,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(self._format_remote_control_exception(worker_id, exc))
                continue
            if not result.ok:
                errors.append(f"{result.worker_type}[{result.worker_id}]: {result.error_message}")
        return errors

    def _broadcast_py_control_results(
        self,
        worker_types: list[WorkerType],
        sub_cmd: int,
        *,
        digest: bytes | None = None,
        payload: bytes | None = None,
    ) -> list[Any]:
        if not worker_types:
            return []
        assert self._worker is not None
        all_results: list[Any] = []
        for worker_type in worker_types:
            results = self._worker.broadcast_control_all(
                worker_type,
                int(sub_cmd),
                payload,
                digest,
                timeout_s=self._py_control_timeout_s,
            )
            all_results.extend(results)
        return all_results

    @staticmethod
    def _control_errors(results: list[Any]) -> list[str]:
        return [
            f"{result.worker_type}[{result.worker_id}]: {result.error_message}" for result in results if not result.ok
        ]

    def _broadcast_py_control(
        self,
        worker_types: list[WorkerType],
        sub_cmd: int,
        *,
        digest: bytes | None = None,
        payload: bytes | None = None,
        strict: bool,
    ) -> list[str]:
        errors = self._control_errors(
            self._broadcast_py_control_results(worker_types, sub_cmd, digest=digest, payload=payload)
        )
        if errors and strict:
            raise RuntimeError(
                f"Worker control broadcast hash={_format_digest(digest or b'')} sub_cmd={sub_cmd} failed on "
                f"{len(errors)} child workers; first error: {errors[0]}"
            )
        return errors

    def _allocate_cid(self) -> int:
        """Return the smallest unused cid in [0, MAX_REGISTERED_CALLABLE_IDS).

        Caller must hold ``_registry_lock``. Walks the integers in order so
        an ``unregister(handle)`` followed by a fresh ``register`` reuses K
        instead of colliding with an existing entry — ``len(registry)``
        would silently overwrite the next gap-after-the-hole.
        """
        for i in range(MAX_REGISTERED_CALLABLE_IDS):
            if i not in self._callable_registry and i not in self._pending_unregister_cids:
                return i
        # The AICPU side keeps a fixed-size orch_so_table_ keyed by cid;
        # raise here so the failure surfaces at register-time with a
        # protocol-aware message, not later from
        # DeviceRunner::register_callable with a generic
        # "out of range" log.
        raise RuntimeError(
            "Worker.register: callable capacity exhausted "
            f"(MAX_REGISTERED_CALLABLE_IDS={MAX_REGISTERED_CALLABLE_IDS}); "
            "unregister unused callables before registering more"
        )

    def _register_child_chip(  # noqa: PLR0912
        self, target: ChipCallable, *, digest: bytes, publish_handle: bool = False
    ) -> CallableHandle | None:
        """Install a cascaded ChipCallable on this child Worker by digest."""
        if not isinstance(target, ChipCallable):
            raise TypeError("_register_child_chip: target must be a ChipCallable")
        reg = _build_callable_registration(self, target)
        if digest != reg.digest:
            raise RuntimeError(
                f"HASHID_DESCRIPTOR_MISMATCH: requested {_format_digest(digest)} but rebuilt {reg.hashid}"
            )
        existing_slot: int | None = None
        with self._registry_lock:
            state = self._identity_registry.get(reg.digest)
            if state is not None:
                if state.slot_id in self._pending_unregister_cids:
                    raise RuntimeError(f"REGISTER_TOMBSTONE_ACTIVE: {reg.hashid}")
                state.ref_count += 1
                existing_slot = state.slot_id
                slot_id = state.slot_id
            else:
                slot_id = self._allocate_cid()
                state = _CallableIdentityState(
                    hashid=reg.hashid,
                    digest=reg.digest,
                    kind="CHIP_CALLABLE",
                    target_namespace="LOCAL_CHIP",
                    descriptor=reg.descriptor,
                    payload_digest=reg.payload_digest,
                    slot_id=slot_id,
                    target=target,
                    ref_count=1,
                )
                self._identity_registry[reg.digest] = state
                self._callable_registry[slot_id] = target

        if existing_slot is not None:
            if self.level >= 3 and self._initialized:
                try:
                    self._post_init_register(target, reg.digest, is_new=False)
                except Exception:
                    with self._registry_lock:
                        state = self._identity_registry.get(reg.digest)
                        if state is not None:
                            state.ref_count -= 1
                    raise
            if publish_handle:
                with self._registry_lock:
                    state = self._identity_registry.get(reg.digest)
                    if state is None:
                        raise RuntimeError(f"callable hash {_format_digest(reg.digest)} disappeared during register")
                    return self._make_handle_locked(state)
            return None

        if self.level >= 3 and self._initialized:
            try:
                self._post_init_register(target, reg.digest, is_new=True)
            except Exception:
                with self._registry_lock:
                    if self._callable_registry.get(slot_id) is target:
                        self._callable_registry.pop(slot_id, None)
                    self._identity_registry.pop(reg.digest, None)
                raise
        if publish_handle:
            with self._registry_lock:
                state = self._identity_registry.get(reg.digest)
                if state is None:
                    raise RuntimeError(f"callable hash {_format_digest(reg.digest)} disappeared during register")
                return self._make_handle_locked(state)
        return None

    def _register_child_python_import(self, target_path: str, *, digest: bytes) -> CallableHandle:
        module, qualname = parse_python_import_target(target_path)
        descriptor = build_python_import_descriptor(module, qualname)
        if digest != _descriptor_digest(descriptor):
            raise RuntimeError(
                f"HASHID_DESCRIPTOR_MISMATCH: requested {_format_digest(digest)} but rebuilt "
                f"{compute_callable_hashid(descriptor)}"
            )
        if self.level < 3:
            raise TypeError("_register_child_python_import requires level >= 3")
        worker_types = self._python_worker_types()
        if self._initialized and not worker_types:
            raise RuntimeError("_register_child_python_import: no Python-capable child workers are configured")
        target = _load_py_import_target(target_path)

        with self._registry_lock:
            state = self._identity_registry.get(digest)
            if state is not None:
                if state.slot_id in self._pending_unregister_cids:
                    raise RuntimeError(f"REGISTER_TOMBSTONE_ACTIVE: {_format_digest(digest)}")
                if (
                    state.descriptor != descriptor
                    or state.kind != "PYTHON_IMPORT"
                    or state.target_namespace != "LOCAL_PYTHON"
                ):
                    raise RuntimeError(f"HASHID_DESCRIPTOR_MISMATCH: {_format_digest(digest)}")
                state.ref_count += 1
                handle = self._make_handle_locked(state)
                is_new = False
            else:
                slot_id = self._allocate_cid()
                state = _CallableIdentityState(
                    hashid=_format_digest(digest),
                    digest=digest,
                    kind="PYTHON_IMPORT",
                    target_namespace="LOCAL_PYTHON",
                    descriptor=descriptor,
                    payload_digest=descriptor,
                    slot_id=slot_id,
                    target=target,
                    ref_count=1,
                )
                self._identity_registry[digest] = state
                self._callable_registry[slot_id] = target
                handle = self._make_handle_locked(state)
                is_new = True

        if self._initialized and getattr(self, "_hierarchical_started", False):
            try:
                results = self._broadcast_py_control_results(
                    worker_types,
                    _CTRL_PY_IMPORT_REGISTER,
                    digest=digest,
                    payload=target_path.encode("utf-8"),
                )
                errors = self._control_errors(results)
                if errors:
                    cleanup_errors = self._cleanup_control_successes(results, _CTRL_PY_UNREGISTER, digest)
                    if cleanup_errors:
                        with self._registry_lock:
                            self._uncertain_hashids.add(digest)
                    raise RuntimeError(self._format_register_partial_failure(digest, errors, cleanup_errors))
            except Exception:
                with self._registry_lock:
                    self._rollback_handle_locked(handle)
                raise
        elif self._initialized and is_new and not getattr(self, "_hierarchical_started", False):
            pass
        return handle

    def _post_init_register(self, target: ChipCallable, digest: bytes, *, is_new: bool) -> None:
        """Broadcast a new ChipCallable to every NEXT_LEVEL child via C++.

        Delegates the entire shm-staging + per-child mailbox handshake to
        ``_Worker.broadcast_register_all``, which holds per-WorkerThread
        ``mailbox_mu_`` so the broadcast serializes against any in-flight
        dispatch on each child mailbox. No Python lock required.
        """
        # Until init() has started the hierarchy the chip mailboxes have no
        # reader, so a CTRL_REGISTER broadcast would deadlock; a registration in
        # that window is instead carried by the startup snapshot and
        # COW-inherited by the children forked in init().
        if not getattr(self, "_hierarchical_started", False):
            return
        assert self._worker is not None
        try:
            results = self._worker.broadcast_register_all(int(target.buffer_ptr()), int(target.buffer_size()), digest)
        except Exception:
            cleanup_errors = self._cleanup_chip_registration(digest) if is_new else []
            if cleanup_errors:
                with self._registry_lock:
                    self._uncertain_hashids.add(digest)
            raise
        errors = self._control_errors(list(results))
        if errors:
            cleanup_errors = self._cleanup_control_successes(list(results), _CTRL_UNREGISTER, digest)
            if cleanup_errors:
                with self._registry_lock:
                    self._uncertain_hashids.add(digest)
            raise RuntimeError(self._format_register_partial_failure(digest, errors, cleanup_errors))

    @staticmethod
    def _format_register_partial_failure(digest: bytes, errors: list[str], cleanup_errors: list[str]) -> str:
        msg = (
            f"REGISTER_PARTIAL_FAILURE: Worker.register(hash={_format_digest(digest)}) failed on "
            f"{len(errors)} child workers; first error: {errors[0]}"
        )
        if cleanup_errors:
            msg += (
                f"; cleanup uncertain on {len(cleanup_errors)} child workers; first cleanup error: {cleanup_errors[0]}"
            )
        return msg

    def _cleanup_control_successes(self, results: list[Any], sub_cmd: int, digest: bytes) -> list[str]:
        if self._worker is None:
            return []
        errors: list[str] = []
        for result in results:
            if not result.ok:
                continue
            try:
                cleanup = self._worker.control_digest_only(
                    self._worker_type_from_result(result.worker_type),
                    int(result.worker_id),
                    int(sub_cmd),
                    digest,
                    timeout_s=self._py_control_timeout_s,
                )
                if not cleanup.ok:
                    errors.append(f"{cleanup.worker_type}[{cleanup.worker_id}]: {cleanup.error_message}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{result.worker_type}[{result.worker_id}]: {exc}")
        return errors

    @staticmethod
    def _worker_type_from_result(worker_type: str) -> WorkerType:
        if worker_type == "NEXT_LEVEL":
            return WorkerType.NEXT_LEVEL
        if worker_type == "SUB":
            return WorkerType.SUB
        raise RuntimeError(f"unknown worker type in control result: {worker_type}")

    def _coerce_handle_state(self, handle_or_slot) -> tuple[int, bytes, _CallableIdentityState]:
        if isinstance(handle_or_slot, CallableHandle):
            state = self._resolve_handle_locked(handle_or_slot)
            return handle_or_slot._handle_id, state.digest, state
        raise TypeError("Worker.unregister expects a CallableHandle returned by Worker.register")

    def _pre_start_unregister_if_needed(self, handle_or_slot) -> bool:
        if self.level < 3:
            # L2 has no pre-start snapshot, but still linearizes against the
            # epoch: reject a terminal worker and wait out an in-progress init.
            with self._hierarchical_start_cv:
                self._wait_out_init_locked("unregister")
            return False
        with self._hierarchical_start_cv:
            self._wait_out_init_locked("unregister")
            if self._lifecycle is not _Lifecycle.NEW:
                return False
            with self._registry_lock:
                handle_id, digest, state = self._coerce_handle_state(handle_or_slot)
                if state.target_namespace == "REMOTE_TASK_DISPATCHER":
                    return False
                cid = state.slot_id
                if cid in self._pending_unregister_cids:
                    raise KeyError("UNREGISTER_TOMBSTONE_ACTIVE: callable handle already pending unregister")
                self._live_handles.pop(handle_id, None)
                state.ref_count -= 1
                if state.ref_count > 0:
                    return True
                self._callable_registry.pop(cid)
                self._identity_registry.pop(digest, None)
            return True

    def unregister(self, handle_or_slot) -> None:
        """Drop a ``CallableHandle`` from the registry and propagate cleanup.

        Symmetric to ``Worker.register`` for the dynamic post-init path.
        The target-local resources become reusable for the next
        ``register`` call — the only practical way to keep a long-running worker under the
        ``MAX_REGISTERED_CALLABLE_IDS`` ceiling when JIT or plugin code
        churns through callables.

        Failure semantics (docs section 8): unregister is best-effort.
        If any chip child reports an error, the parent **warns and still
        pops the registry entry** — orch_so_table_ on the AICPU side will
        be overwritten on target-local resource reuse, and refusing to
        release a known-bad entry would just exhaust the resource space
        faster.

        Raises:
          KeyError: handle was never registered.
        """
        if isinstance(handle_or_slot, CallableHandle) and handle_or_slot.target_namespace == "REMOTE_TASK_DISPATCHER":
            # Linearize against the epoch before dispatching: wait out an
            # in-progress init so the remote cleanup is actually sent once READY
            # (an INITIALIZING worker has _initialized False and would drop local
            # state while skipping the remote send, leaving a dangling remote
            # dispatcher); reject a terminal worker.
            with self._hierarchical_start_cv:
                self._wait_out_init_locked("unregister")
            self._unregister_remote_handle(handle_or_slot)
            return
        if self._pre_start_unregister_if_needed(handle_or_slot):
            return
        target = None
        digest = b""
        cid = -1
        handle_id = -1
        remove_target = False
        with self._registry_lock:
            handle_id, digest, state = self._coerce_handle_state(handle_or_slot)
            cid = state.slot_id
            if cid in self._pending_unregister_cids:
                raise KeyError("UNREGISTER_TOMBSTONE_ACTIVE: callable handle already pending unregister")
            self._live_handles.pop(handle_id, None)
            state.ref_count -= 1
            should_broadcast_decrement = (
                self.level >= 3 and self._initialized and getattr(self, "_hierarchical_started", False)
            )
            if state.ref_count > 0 and not should_broadcast_decrement:
                return
            target = self._callable_registry[cid]
            remove_target = state.ref_count <= 0
            if should_broadcast_decrement:
                self._pending_unregister_cids.add(cid)
                if state.ref_count > 0:
                    remove_target = False
            elif self.level == 2 and self._initialized:
                assert self._chip_worker is not None
                self._chip_worker._unregister_slot(cid)
                self._callable_registry.pop(cid, None)
                self._identity_registry.pop(digest, None)
                return
            else:
                self._callable_registry.pop(cid, None)
                self._identity_registry.pop(digest, None)
                return

        try:
            if isinstance(target, ChipCallable):
                self._broadcast_unregister(digest)
            else:
                errors = self._broadcast_py_control(
                    self._python_worker_types(),
                    _CTRL_PY_UNREGISTER,
                    digest=digest,
                    strict=False,
                )
                if errors:
                    sys.stderr.write(
                        f"Worker.unregister(hash={_format_digest(digest)}): "
                        f"{len(errors)} Python children reported errors "
                        f"(continuing best-effort). First error: {errors[0]}\n"
                    )
                    sys.stderr.flush()
        finally:
            with self._registry_lock:
                current = self._identity_registry.get(digest)
                if remove_target and current is not None and current is state and current.ref_count <= 0:
                    self._callable_registry.pop(cid, None)
                    self._identity_registry.pop(digest, None)
                self._pending_unregister_cids.discard(cid)

    def _unregister_remote_handle(self, handle: CallableHandle) -> None:
        worker_ids: tuple[int, ...]
        kind: str
        digest: bytes
        remove_state = False
        with self._registry_lock:
            _handle_id, digest, state = self._coerce_handle_state(handle)
            if digest in self._pending_remote_unregister_hashids:
                raise KeyError("UNREGISTER_TOMBSTONE_ACTIVE: remote callable handle already pending unregister")
            self._live_handles.pop(handle._handle_id, None)
            state.ref_count -= 1
            if state.ref_count > 0:
                return
            self._pending_remote_unregister_hashids.add(digest)
            worker_ids = state.eligible_worker_ids
            kind = state.kind
            remove_state = True

        errors: list[str] = []
        try:
            if self._initialized:
                assert self._worker is not None
                for worker_id in worker_ids:
                    try:
                        result = self._worker.remote_unregister(
                            worker_id,
                            "REMOTE_TASK_DISPATCHER",
                            kind,
                            digest,
                        )
                    except Exception as exc:  # noqa: BLE001
                        errors.append(self._format_remote_control_exception(worker_id, exc))
                        continue
                    if not result.ok:
                        errors.append(f"{result.worker_type}[{result.worker_id}]: {result.error_message}")
                if errors:
                    with self._registry_lock:
                        self._uncertain_hashids.add(digest)
                    sys.stderr.write(
                        f"Worker.unregister(hash={_format_digest(digest)}): remote cleanup uncertain on "
                        f"{len(errors)} remote workers. First error: {errors[0]}\n"
                    )
                    sys.stderr.flush()
        finally:
            with self._registry_lock:
                if remove_state:
                    self._identity_registry.pop(digest, None)
                self._pending_remote_unregister_hashids.discard(digest)

    def _unregister_child_digest(self, *, digest: bytes) -> None:
        target = None
        cid = -1
        remove_target = False
        with self._registry_lock:
            state = self._identity_registry.get(digest)
            if state is None:
                return
            cid = state.slot_id
            if cid in self._pending_unregister_cids:
                raise KeyError("UNREGISTER_TOMBSTONE_ACTIVE: callable identity already pending unregister")
            target = self._callable_registry[cid]
            should_broadcast_decrement = (
                self.level >= 3 and self._initialized and getattr(self, "_hierarchical_started", False)
            )
            chip_worker = None
            if self.level == 2 and self._initialized:
                assert self._chip_worker is not None
                chip_worker = self._chip_worker

            new_ref_count = state.ref_count - 1
            if new_ref_count > 0 and not should_broadcast_decrement:
                state.ref_count = new_ref_count
                return
            state.ref_count = new_ref_count
            remove_target = state.ref_count <= 0
            if should_broadcast_decrement:
                self._pending_unregister_cids.add(cid)
                if new_ref_count > 0:
                    remove_target = False
            elif self.level == 2 and self._initialized:
                assert chip_worker is not None
                chip_worker._unregister_slot(cid)
                self._callable_registry.pop(cid, None)
                self._identity_registry.pop(digest, None)
                return
            else:
                self._callable_registry.pop(cid, None)
                self._identity_registry.pop(digest, None)
                return

        try:
            if isinstance(target, ChipCallable):
                self._broadcast_unregister(digest)
            else:
                errors = self._broadcast_py_control(
                    self._python_worker_types(),
                    _CTRL_PY_UNREGISTER,
                    digest=digest,
                    strict=False,
                )
                if errors:
                    sys.stderr.write(
                        f"Worker.unregister(hash={_format_digest(digest)}): "
                        f"{len(errors)} Python children reported errors "
                        f"(continuing best-effort). First error: {errors[0]}\n"
                    )
                    sys.stderr.flush()
        finally:
            with self._registry_lock:
                current = self._identity_registry.get(digest)
                if remove_target and current is not None and current is state and current.ref_count <= 0:
                    self._callable_registry.pop(cid, None)
                    self._identity_registry.pop(digest, None)
                self._pending_unregister_cids.discard(cid)

    def _cleanup_chip_registration(self, digest: bytes) -> list[str]:
        if self._worker is None:
            return []
        try:
            return list(self._worker.broadcast_unregister_all(digest))
        except Exception as exc:  # noqa: BLE001
            return [str(exc)]

    def _broadcast_unregister(self, digest: bytes) -> None:
        """Broadcast _CTRL_UNREGISTER via C++ to every NEXT_LEVEL child.

        Best-effort: any per-child errors are returned by C++ as a list of
        strings; we warn to stderr and let the caller still pop the registry.
        """
        assert self._worker is not None
        errors = self._worker.broadcast_unregister_all(digest)
        if errors:
            sys.stderr.write(
                f"Worker.unregister(hash={_format_digest(digest)}): {len(errors)} chips reported errors "
                f"(continuing best-effort). First error: {errors[0]}\n"
            )
            sys.stderr.flush()

    def add_worker(self, worker: Worker) -> int:
        """Add a lower-level Worker as a NEXT_LEVEL child. Must be called before init().

        The child Worker must NOT be init'd — init happens inside the forked
        child process (so the child's own children are forked in the right
        process tree). Returns this child's stable NEXT_LEVEL worker id.
        """
        if self.level < 4:
            raise RuntimeError("Worker.add_worker() requires level >= 4")
        if self._config.get("device_ids", []):
            raise RuntimeError("Worker.add_worker() cannot be combined with device_ids on the same Worker")
        if worker._lifecycle is not _Lifecycle.NEW:
            # init() happens inside the forked child process, so the child must
            # be pristine — not started, failed, or closed.
            raise RuntimeError("Child worker must be NEW (not started/failed/closed) before add_worker()")
        # Hold the lifecycle lock across the state check and the topology
        # mutation so a concurrent init() cannot freeze the topology snapshot
        # between them.
        with self._hierarchical_start_cv:
            if self._lifecycle is not _Lifecycle.NEW:
                raise RuntimeError("Worker.add_worker() must be called before init()")
            worker_id = self._allocate_next_level_worker_id()
            self._next_level_workers.append(worker)
            self._next_level_worker_ids.append(worker_id)
            return worker_id

    # ------------------------------------------------------------------
    # init — auto-discovery
    # ------------------------------------------------------------------

    def _eligible_target_need(self, namespace: str | None, eligible_worker_ids) -> str | None:
        """Return the missing dispatch target for a callable of this *kind*, or
        None if it is eligible in the current (frozen) topology.

        Keyed on the same ``target_namespace`` → child-loop mapping
        ``_make_local_identity_tables`` applies at fork:
          - ``LOCAL_PYTHON`` (Python callable) is installed only into SUB and
            next-level child loops — a chip child does NOT resolve it;
          - ``LOCAL_CHIP`` (ChipCallable) only into chip child loops;
          - ``REMOTE_TASK_DISPATCHER`` only onto its named remote worker(s).
        An L2 worker (or any non-dispatch namespace) is always eligible.

        Used only by ``_validate_eligible_targets`` at init (the *startup*
        eligibility gate). The post-init dynamic ``register`` path does NOT yet
        apply this rule — see that method for the deferred inconsistency.
        """
        if self.level < 3:
            return None
        if namespace == "LOCAL_PYTHON":
            has_python_child = self._config.get("num_sub_workers", 0) > 0 or bool(self._next_level_workers)
            return None if has_python_child else "a SUB or next-level child"
        if namespace == "LOCAL_CHIP":
            return None if bool(self._config.get("device_ids")) else "a chip device (device_ids)"
        if namespace == "REMOTE_TASK_DISPATCHER":
            has_remote_workers = set(self._remote_worker_ids)
            ok = bool(has_remote_workers) and set(eligible_worker_ids) <= has_remote_workers
            return None if ok else "its named remote worker(s) (add_remote_worker)"
        return None

    def _validate_eligible_targets(self) -> None:
        """Reject a pre-registered callable that no child of the frozen topology
        can resolve, before any startup resource is allocated.

        A registration whose namespace has no matching child is silently dropped
        by ``_make_local_identity_tables`` and would leave the worker
        READY-yet-inert, so raise here with its namespace + hashid. (An L3
        orchestrator passed to ``run()`` runs on this host and is never
        registered, so it is not subject to this check.) See
        ``_eligible_target_need`` for the per-kind rule.
        """
        if self.level < 3:
            return
        with self._registry_lock:
            states = list(self._identity_registry.items())
        for digest, state in states:
            need = self._eligible_target_need(state.target_namespace, state.eligible_worker_ids)
            if need is not None:
                raise RuntimeError(
                    f"Worker.init(): registered {state.target_namespace} callable {_format_digest(digest)} "
                    f"has no eligible dispatch target (needs {need})"
                )

    def init(self, prewarm_config=None, *, _startup_deadline: float | None = None) -> None:
        """Initialize the worker and bring its whole subtree to READY.

        For an L3+ worker ``init`` is the single startup submission point: it
        forks every local child (sub / chip / next-level), waits for the whole
        subtree — recursively, for L4+ — to publish INIT_READY, activates any
        remote L3 sessions, starts the C++ scheduler, and only then publishes
        READY in one atomic commit. It returns with the tree ready to run, or
        raises after a bounded rollback that reaps the children it forked
        best-effort (a child wedged in native code past the deadline may be left
        behind — see the deferred un-reaped-child / nested-shm items).
        ``run`` / ``create_host_buffer`` / the remote register/memory APIs never
        trigger startup.

        Args:
            prewarm_config: Optional CallConfig. When given, its ring sizing
                (``runtime_env.ring_task_window`` / ``ring_heap`` /
                ``ring_dep_pool``) is built + cached so the first ``run`` with the
                same sizing skips the (~800ms) cold prebuilt runtime-arena build.
                An L2 worker prewarms here; an L3+ worker prewarms each chip child
                during hierarchy startup, before it publishes INIT_READY. A no-op
                for runtimes without a prebuilt arena (host_build_graph). ``None``
                (default) disables prewarm.
            _startup_deadline: Internal. Absolute ``time.monotonic()`` deadline
                inherited from a parent's startup epoch so a recursive descendant
                consumes the parent's remaining budget instead of restarting the
                timeout. ``None`` starts a fresh epoch.
        """
        if prewarm_config is not None:
            prewarm_config.validate()
        # Claim the startup epoch atomically: NEW -> INITIALIZING under the
        # lifecycle lock so a concurrent init / register / close observes one
        # linear transition and never a half-built Worker. Every level claims the
        # epoch so two concurrent init() calls serialize on it; an L2 worker
        # still inits synchronously in-process, with no child barrier.
        with self._hierarchical_start_cv:
            if self._lifecycle is _Lifecycle.INITIALIZING:
                raise RuntimeError("Worker.init() is already in progress")
            if self._lifecycle is _Lifecycle.READY:
                raise RuntimeError("Worker already initialized")
            if self._lifecycle is _Lifecycle.FAILED:
                raise RuntimeError("Worker startup failed; close this Worker and create a new one")
            if self._lifecycle is _Lifecycle.CLOSED:
                # CLOSED is a permanent admission fence: a closed worker (even one
                # whose private teardown is still finishing) is never revived by a
                # concurrent init().
                raise RuntimeError("Worker is closed; create a new Worker")
            # Reject an initial callable that can never run before any startup
            # resource is spent: a childless worker that accepted a callable
            # would otherwise come up READY yet inert. Held under the lifecycle
            # lock so a concurrent register() cannot install a target between the
            # check and the epoch claim (register's snapshot install also holds
            # this lock).
            self._validate_eligible_targets()
            self._prewarm_config = prewarm_config
            self._startup_error = None
            self._init_owner_thread = threading.current_thread()
            self._lifecycle = _Lifecycle.INITIALIZING
            if self.level >= 3:
                self._is_startup_root = _startup_deadline is None
                own_deadline = time.monotonic() + self._startup_timeout_s
                # A recursive descendant caps its own timeout at the parent's
                # remaining budget so the whole tree fits one startup_timeout_s.
                self._startup_deadline = (
                    own_deadline if _startup_deadline is None else min(_startup_deadline, own_deadline)
                )
            self._hierarchical_start_cv.notify_all()

        try:
            if self.level == 2:
                self._init_level2()
            elif self.level >= 3:
                self._init_hierarchical()
                self._start_hierarchical()
            else:
                raise ValueError(f"Worker: level {self.level} not supported")
            # Atomic READY commit inside the exception boundary: publish the
            # single lifecycle state so no thread ever observes a started
            # hierarchy while the worker is not yet READY.
            with self._hierarchical_start_cv:
                self._lifecycle = _Lifecycle.READY
                self._hierarchical_start_cv.notify_all()
        except BaseException as exc:
            # Any unwind (init failure or KeyboardInterrupt) rolls back through
            # one path: capture the original cause first (so a cleanup error
            # cannot overwrite it and every waiter sees the same reason), roll
            # back, commit FAILED even if rollback raises, then surface.
            with self._hierarchical_start_cv:
                if self._startup_error is None:
                    self._startup_error = exc
            try:
                self._cleanup_partial_init()
            finally:
                with self._hierarchical_start_cv:
                    # Only an INITIALIZING epoch commits FAILED. This thread is
                    # the sole writer of the INITIALIZING -> FAILED edge (close()
                    # fails fast while INITIALIZING and never advances it).
                    if self._lifecycle is _Lifecycle.INITIALIZING:
                        self._lifecycle = _Lifecycle.FAILED
                    self._hierarchical_start_cv.notify_all()
            raise

    def _init_level2(self) -> None:
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        platform = self._config["platform"]
        runtime = self._config["runtime"]
        device_id = self._config.get("device_id", 0)

        builder = RuntimeBuilder(platform)
        binaries = builder.get_binaries(runtime)

        self._chip_worker = ChipWorker()
        # The prebuilt runtime-arena is prewarmed inside cw.init for the declared
        # config's ring sizing (built right after the device comes up), so the
        # first run() with matching sizing skips the cold arena build.
        self._chip_worker.init(device_id, binaries, prewarm_config=self._prewarm_config)

        # Pre-warm any registered ChipCallable so the first run(handle, …)
        # does not pay the H2D upload cost.
        assert self._chip_worker is not None
        for cid, target in self._callable_registry.items():
            if isinstance(target, ChipCallable):
                self._chip_worker._register_callable_at_slot(cid, target)

    def _init_hierarchical(self) -> None:
        device_ids = self._config.get("device_ids", [])
        n_sub = self._config.get("num_sub_workers", 0)
        heap_ring_size = self._config.get("heap_ring_size", None)
        if self.level >= 4 and device_ids:
            raise RuntimeError("Worker level >= 4 must use add_worker(); device_ids are only supported on L3 Workers")

        # Only a worker that carries remote workers has a remote session to
        # time out. Its remote_session_timeout_s is validated here, before any
        # startup resource (mailbox shm, pre-fork _Worker mmap, child fork,
        # daemon socket) exists, so an invalid value fails without a
        # partially-built subtree to roll back.
        if self._remote_worker_specs:
            self._remote_session_timeout_s()

        # 1. Allocate sub-worker mailboxes (unified layout, MAILBOX_SIZE each).
        for _ in range(n_sub):
            shm = SharedMemory(create=True, size=MAILBOX_SIZE)
            assert shm.buf is not None
            _mailbox_store_i32(_buffer_field_addr(shm.buf, _OFF_STATE), _IDLE)
            self._sub_shms.append(shm)

        # 2. Prepare chip-worker config (L3 only — L4+ has Worker children instead)
        if device_ids:
            from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

            platform = self._config["platform"]
            runtime = self._config["runtime"]
            builder = RuntimeBuilder(platform)
            binaries = builder.get_binaries(runtime)

            # Stash the full RuntimeBinaries so forked chip children can
            # construct a ChipWorker with one call (`cw.init(device_id, bins)`)
            # instead of taking ~10 path strings via positional args.  Forked-child
            # invocation is `os.fork()` + direct function call, so no pickle
            # barrier — the bins object is just a Python value passed through.
            self._l3_bins = binaries

            # Allocate chip mailboxes (unified layout, MAILBOX_SIZE each).
            for _ in device_ids:
                shm = SharedMemory(create=True, size=MAILBOX_SIZE)
                assert shm.buf is not None
                _mailbox_store_i32(_buffer_field_addr(shm.buf, _OFF_STATE), _IDLE)
                self._chip_shms.append(shm)

        # 3. Allocate next-level Worker child mailboxes (L4+ only).
        for _ in self._next_level_workers:
            shm = SharedMemory(create=True, size=MAILBOX_SIZE)
            assert shm.buf is not None
            _mailbox_store_i32(_buffer_field_addr(shm.buf, _OFF_STATE), _IDLE)
            self._next_level_shms.append(shm)

        # 4. Construct the _Worker *before* fork so the HeapRing mmap
        #    (taken in the C++ ctor) is inherited by every child process at
        #    the same virtual address. No C++ thread is spawned here; the
        #    scheduler + WorkerThreads start in init(), after forks.
        if heap_ring_size is None:
            self._worker = _Worker(self.level)
        else:
            self._worker = _Worker(self.level, int(heap_ring_size))

    def _activate_remote_sessions(self, deadline: float) -> None:
        """Open and register every remote L3 session within the shared startup budget.

        Called only from _start_hierarchical, after this process's last local
        fork, so opening a session (which starts the remote subtree) and
        registering its endpoint (which spawns the health thread) both stay
        behind every local fork. All remotes draw from the single root startup
        ``deadline``: each computes the remaining budget at the moment it opens,
        propagates it as the manifest's ``startup_remaining_s`` so the remote
        bounds its own subtree by this process's remaining time (measured on the
        remote's own monotonic clock) instead of a fresh full timeout. Any
        failure propagates to init()'s single rollback, which closes every
        session recorded in ``self._remote_sessions``.
        """
        if not self._remote_worker_specs:
            return
        session_timeout = self._remote_session_timeout_s()
        for worker_id, spec in zip(self._remote_worker_ids, self._remote_worker_specs, strict=True):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("remote L3 session activation: startup deadline exceeded")
            session_id = uuid.uuid4().int & ((1 << 63) - 1)
            if session_id == 0:
                session_id = 1
            # The handshake blocks until the remote subtree is READY, so the
            # socket timeout must cover the startup budget granted below — not
            # the (shorter) runtime command timeout.
            session = self._open_remote_session(
                spec=spec,
                worker_id=worker_id,
                session_id=session_id,
                timeout_s=remaining,
                startup_remaining_s=remaining,
            )
            self._remote_sessions.append(session)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("remote L3 endpoint attach: startup deadline exceeded")
            assert self._worker is not None
            self._worker.add_remote_l3_socket(
                session.worker_id,
                session.session_id,
                spec.transport,
                session.command_host,
                session.command_port,
                session.health_host,
                session.health_port,
                min(session_timeout, remaining),
            )

    def _start_hierarchical(self) -> None:  # noqa: PLR0912 -- three parallel fork loops (sub/chip/next) + bootstrap wait + scheduler register/init; branches track the fork order documented in the body
        """Fork every local child, await the subtree, register endpoints, start the scheduler.

        Called only by init(), which owns the lifecycle state. Any failure here
        propagates to init(), whose single rollback entry (_cleanup_partial_init)
        closes the C++ Worker (if the scheduler started) and tears the whole
        epoch down. The readiness barriers raise on any child failure/exit/hang.
        """
        device_ids = self._config.get("device_ids", [])
        n_sub = self._config.get("num_sub_workers", 0)
        deadline = self._startup_deadline

        # Freeze the startup registry snapshot. init() already holds the epoch in
        # the INITIALIZING state, so a concurrent register/unregister is blocked
        # on the lifecycle condition and cannot slip a mutation in after this point.
        with self._registry_lock:
            identity_snapshot = [
                (digest, state.target, state.ref_count, state.kind, state.target_namespace)
                for digest, state in self._identity_registry.items()
            ]

        self._startup_reaped_pids = set()
        self._startup_ready_pids = set()
        self._startup_group_leader_pids = set()

        # Fork SubWorker processes (MUST be before any C++ threads)
        for i in range(n_sub):
            pid = os.fork()
            if pid == 0:
                buf = self._sub_shms[i].buf
                assert buf is not None

                def _setup():
                    return _make_local_identity_tables(
                        identity_snapshot,
                        callable_kind=("PYTHON_SERIALIZED", "PYTHON_IMPORT"),
                        target_namespace="LOCAL_PYTHON",
                    )

                _forked_child_main(
                    buf,
                    f"sub worker {i}",
                    _setup,
                    lambda t, b=buf: _sub_worker_loop(b, *t),
                    make_group_leader=self._is_startup_root,
                )
            else:
                self._sub_pids.append(pid)
                if self._is_startup_root:
                    self._startup_group_leader_pids.add(pid)

        # SUB children have no fallible device/runtime init, but they join the
        # same readiness contract so a child that dies before entering its loop
        # aborts startup rather than surfacing later as a hung submit_sub.
        self._await_children_ready(self._sub_shms, self._sub_pids, "sub", deadline)

        # Fork ChipWorker processes (L3 with device_ids).  Always use the plain
        # task-loop variant; the base communicator is established lazily on first
        # ``orch.allocate_domain`` via CTRL_COMM_INIT.
        chip_log_level, chip_log_info_v = _simpler_log.get_current_config()
        if device_ids:
            for idx, dev_id in enumerate(device_ids):
                pid = os.fork()
                if pid == 0:
                    buf = self._chip_shms[idx].buf
                    assert buf is not None
                    if self._is_startup_root:
                        with contextlib.suppress(OSError):
                            os.setpgid(0, 0)
                    # _chip_process_loop publishes INIT_READY/INIT_FAILED itself
                    # (around cw.init + ChipCallable prepare). This guard only
                    # ensures the child exits rather than unwinding into the
                    # parent's startup frames (see _forked_child_main). A throw
                    # before cw.init (e.g. identity-table build) leaves the
                    # mailbox IDLE, so publish INIT_FAILED for a bounded parent
                    # error.
                    try:
                        _chip_process_loop(
                            buf,
                            self._l3_bins,
                            dev_id,
                            *_make_local_identity_tables(
                                identity_snapshot,
                                callable_kind="CHIP_CALLABLE",
                                target_namespace="LOCAL_CHIP",
                            ),
                            log_level=chip_log_level,
                            log_info_v=chip_log_info_v,
                            platform=str(self._config["platform"]),
                            runtime=str(self._config["runtime"]),
                            prewarm_config=self._prewarm_config,
                        )
                    except BaseException as e:  # noqa: BLE001
                        import traceback as _tb  # noqa: PLC0415

                        _tb.print_exc()
                        if _mailbox_load_i32(_buffer_field_addr(buf, _OFF_STATE)) == _IDLE:
                            _write_error(buf, 1, _format_exc(f"chip worker {idx} dev={dev_id} init", e))
                            _mailbox_store_i32(_buffer_field_addr(buf, _OFF_STATE), _INIT_FAILED)
                        os._exit(1)
                    os._exit(0)
                else:
                    self._chip_pids.append(pid)
                    if self._is_startup_root:
                        self._startup_group_leader_pids.add(pid)

            # Cross-chip init barrier.  ChipWorker.init can have a long right tail
            # (e.g. PTO2_RING_HEAP=4 GiB pushes per-rank device_malloc beyond the
            # host stream sync budget); without this barrier a fast-init chip
            # starts its aclrtSyncStream window N seconds before a slow peer
            # reaches the same point, and any cross-rank wait inside the op (HCCL
            # notify, etc.) charges the slow peer's remaining init time against
            # the fast peer's PLATFORM_STREAM_SYNC_TIMEOUT_MS budget — the cascade
            # documented in issue #897.  A chip that fails or dies during init
            # raises here rather than spinning forever.
            self._await_children_ready(self._chip_shms, self._chip_pids, "chip", deadline)

        # Fork next-level Worker children (L4+ with Worker children).
        # Each child process eagerly inits the inner Worker, which forks its own
        # chip/sub (and, for L5+, deeper next-level) children and blocks on their
        # readiness before returning — so the process tree nests correctly (L4 →
        # L3 child → L3's chip/sub grandchildren) and INIT_READY propagates up
        # only after the whole subtree is ready.
        for idx, inner_worker in enumerate(self._next_level_workers):
            pid = os.fork()
            if pid == 0:
                buf = self._next_level_shms[idx].buf
                assert buf is not None

                def _setup(inner=inner_worker):
                    # Propagate the fork-constant prewarm sizing and the shared
                    # startup deadline so the inner subtree comes up within the
                    # parent's remaining budget. INIT_READY is published only
                    # after BOTH the inner init (its whole subtree) and the
                    # identity-table build succeed, so the parent never observes
                    # READY for a child that then dies in fallible post-init
                    # setup. A failure after inner.init() succeeded tears the
                    # inner subtree back down before propagating, so a fallible
                    # post-init step leaves no orphaned grandchildren / shms.
                    inner.init(prewarm_config=self._prewarm_config, _startup_deadline=deadline)
                    try:
                        return _make_local_identity_tables(
                            identity_snapshot,
                            callable_kind=("PYTHON_SERIALIZED", "PYTHON_IMPORT"),
                            target_namespace="LOCAL_PYTHON",
                        )
                    except BaseException:
                        with contextlib.suppress(BaseException):
                            inner.close()
                        raise

                _forked_child_main(
                    buf,
                    f"next_level worker {idx}",
                    _setup,
                    lambda tables, b=buf, inner=inner_worker: _child_worker_loop(b, *tables, inner),
                    make_group_leader=self._is_startup_root,
                )
            else:
                self._next_level_pids.append(pid)
                if self._is_startup_root:
                    self._startup_group_leader_pids.add(pid)

        # The recursive readiness edge: a next-level child's own init blocks on
        # its descendants, so its INIT_READY means the whole subtree is ready. A
        # failure, exit, or hang aborts startup here.
        self._await_children_ready(self._next_level_shms, self._next_level_pids, "next_level", deadline)

        # Last local fork is done. Now — and only now — open and register remote
        # L3 sessions: opening starts the remote subtree and registering spawns
        # the RemoteL3Endpoint health thread, so both must follow every local
        # fork. Each remote consumes this process's remaining startup budget.
        self._activate_remote_sessions(deadline)

        # _Worker was constructed in _init_hierarchical (pre-fork) so children
        # inherit the HeapRing MAP_SHARED mmap. Register PROCESS-mode workers via
        # the unified mailbox.
        dw = self._worker
        assert dw is not None

        # Register chip workers as NEXT_LEVEL (L3)
        if device_ids:
            for shm in self._chip_shms:
                dw.add_next_level_worker(_mailbox_addr(shm))

        # Register Worker children as NEXT_LEVEL (L4+)
        if self._next_level_shms and not hasattr(dw, "add_next_level_worker_at"):
            raise RuntimeError("explicit NEXT_LEVEL worker ids require a rebuilt _task_interface module")
        for idx, shm in enumerate(self._next_level_shms):
            worker_id = self._next_level_worker_ids[idx]
            dw.add_next_level_worker_at(worker_id, _mailbox_addr(shm))

        for shm in self._sub_shms:
            dw.add_sub_worker(_mailbox_addr(shm))

        # Start Scheduler + WorkerThreads (C++ threads start here, after fork)
        dw.init()

        self._orch = Orchestrator(dw.get_orchestrator(), self)

        # Every ChipCallable in the startup snapshot was already uploaded by its
        # chip child before that child published INIT_READY (see
        # _chip_process_loop), and the runtime arena was prewarmed there too — so
        # there is no post-scheduler control_prepare on the startup path.

    def _await_children_ready(self, shms, pids, kind: str, deadline: float) -> None:
        """Block until every forked child reports INIT_READY, or abort.

        Polls each child's mailbox: INIT_READY resets the slot to _IDLE (so the
        C++ dispatch state machine resumes from the canonical "ready for work"
        state), records the pid as having reached its serve loop, and retires
        it; INIT_FAILED surfaces the child's own error; ``waitpid(WNOHANG)``
        catches a child that died before signalling (recording the reaped pid so
        rollback never re-SIGKILLs a possibly-reused PID). ``deadline`` is the
        single startup-epoch deadline shared by every child group and every
        recursive descendant, so a deep tree cannot multiply the timeout; a
        child that hangs past it aborts the epoch. A failure raises
        ``RuntimeError`` — the caller rolls back the whole startup epoch.
        """
        pending = list(range(len(shms)))
        while pending:
            still_pending = []
            for i in pending:
                buf = shms[i].buf
                assert buf is not None
                addr = _buffer_field_addr(buf, _OFF_STATE)
                state = _mailbox_load_i32(addr)
                if state == _INIT_READY:
                    _mailbox_store_i32(addr, _IDLE)
                    self._startup_ready_pids.add(pids[i])
                    continue
                if state == _INIT_FAILED:
                    raise RuntimeError(f"{kind} worker {i} (pid {pids[i]}) failed during init: {_read_error_msg(buf)}")
                try:
                    wpid, status = os.waitpid(pids[i], os.WNOHANG)
                except ChildProcessError:
                    self._startup_reaped_pids.add(pids[i])
                    raise RuntimeError(
                        f"{kind} worker {i} (pid {pids[i]}) exited during init before signalling ready"
                    ) from None
                if wpid != 0:
                    self._startup_reaped_pids.add(pids[i])
                    raise RuntimeError(
                        f"{kind} worker {i} (pid {pids[i]}) exited during init "
                        f"before signalling ready (wait status {status})"
                    )
                still_pending.append(i)
            pending = still_pending
            if pending:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"{kind} worker(s) {pending} did not become ready within "
                        f"{self._startup_timeout_s}s (startup deadline exceeded)"
                    )
                time.sleep(_STARTUP_POLL_INTERVAL_S)

    # ------------------------------------------------------------------
    # Hierarchical abort
    # ------------------------------------------------------------------

    def _abort_hierarchical(self, deadline: float | None = None) -> None:  # noqa: PLR0912 -- graceful/cooperative-cancel-then-killpg rollback across sub/chip/next-level, bounded-wait, reap, free shms
        """Tear down the whole forked subtree + shms after a bootstrap failure.

        Called from the init() failure path — the single rollback entry
        (_cleanup_partial_init) — so `dw.init()` may or may not have run.

        ``deadline`` is one absolute ``time.monotonic()`` budget shared by both
        phases (cooperative wait and the final reap), so the whole rollback is
        bounded end-to-end; a survivor still alive at the deadline is left to the
        OS/init rather than blocking this thread on a D-state child. Defaults to
        one ``_ROLLBACK_GRACEFUL_TIMEOUT_S`` window.

        Teardown proceeds in two bounded phases within one cleanup budget:

        1. Cooperative. A child that reached its serve loop (READY) is asked to
           close gracefully via SHUTDOWN so it finalizes its device / unlinks its
           own nested shms. A next-level child still inside ``inner.init()``
           (mid-init, and possibly already the parent of grandchildren) is sent
           SIGTERM, which unwinds its ``inner.init()`` and recursively reclaims
           its grandchildren and their nested shms.
        2. Hard backstop. Any child still alive past the cleanup deadline is
           reaped. As the startup root, ``killpg`` takes the whole subtree — the
           child and every descendant that inherited its process group — so a
           mid-init grandchild is reaped here rather than left to the
           multiprocessing resource_tracker; a nested (non-root) worker SIGKILLs
           the direct pid and relies on the root's killpg. PIDs the barrier
           already reaped are excluded so a reused PID is never signalled.
        """
        if deadline is None:
            deadline = time.monotonic() + _ROLLBACK_GRACEFUL_TIMEOUT_S
        reaped = set(self._startup_reaped_pids)
        graceful: list[int] = []
        cancelled: list[int] = []
        killed: list[int] = []

        # A next-level child may have published INIT_READY in the window between
        # the barrier last polling it and aborting on a failing sibling — its
        # mailbox still reads INIT_READY (only the barrier resets it to IDLE).
        # Promote it to READY so it is torn down gracefully (SHUTDOWN unlinks its
        # own nested shms) rather than cooperatively cancelled after it has
        # already restored the default SIGTERM disposition.
        for idx, pid in enumerate(self._next_level_pids):
            if pid in reaped or pid in self._startup_ready_pids:
                continue
            buf = self._next_level_shms[idx].buf if idx < len(self._next_level_shms) else None
            if buf is not None and _mailbox_load_i32(_buffer_field_addr(buf, _OFF_STATE)) == _INIT_READY:
                self._startup_ready_pids.add(pid)

        # Phase 1a: READY children (sub / chip / next-level) close gracefully.
        for pids_list, shms_list in (
            (self._next_level_pids, self._next_level_shms),
            (self._chip_pids, self._chip_shms),
            (self._sub_pids, self._sub_shms),
        ):
            for idx, pid in enumerate(pids_list):
                if pid in reaped or pid not in self._startup_ready_pids:
                    continue
                buf = shms_list[idx].buf if idx < len(shms_list) else None
                if buf is None:
                    continue
                _mailbox_store_i32(_buffer_field_addr(buf, _OFF_STATE), _SHUTDOWN)
                graceful.append(pid)

        # Phase 1b: mid-init next-level children get a cooperative cancel so they
        # unwind inner.init() and recursively reclaim their own subtree.
        for pid in self._next_level_pids:
            if pid in reaped or pid in self._startup_ready_pids:
                continue
            with contextlib.suppress(ProcessLookupError, OSError):
                os.kill(pid, signal.SIGTERM)
            cancelled.append(pid)

        waiting = set(graceful) | set(cancelled)
        if waiting:
            while waiting and time.monotonic() <= deadline:
                for pid in list(waiting):
                    try:
                        wpid, _status = os.waitpid(pid, os.WNOHANG)
                    except ChildProcessError:
                        waiting.discard(pid)
                        reaped.add(pid)
                        continue
                    if wpid != 0:
                        waiting.discard(pid)
                        reaped.add(pid)
                if waiting:
                    time.sleep(_STARTUP_POLL_INTERVAL_S)

        # Phase 2: hard backstop for any survivor. A not-yet-reaped pid still
        # holds its slot (no reuse), so killpg on the root reaps the survivor's
        # whole group (it + inherited-group descendants) safely.
        pids = list(self._chip_pids) + list(self._sub_pids) + list(self._next_level_pids)
        for pid in pids:
            if pid in reaped:
                continue
            if self._is_startup_root:
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.killpg(pid, signal.SIGKILL)
            with contextlib.suppress(ProcessLookupError, OSError):
                os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        # Bounded final reap within the shared deadline: a SIGKILL'd child exits
        # promptly, so poll with WNOHANG rather than a blocking waitpid — a
        # D-state (uninterruptible) survivor must not pin this thread past the
        # cleanup budget. One sweep always runs (a just-killed pid is usually
        # already reapable); a pid not reaped by the deadline is left to the
        # OS/init rather than extending the budget.
        to_reap = {p for p in pids if p in killed or p not in reaped}
        while to_reap:
            for pid in list(to_reap):
                try:
                    wpid, _status = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    to_reap.discard(pid)
                    continue
                if wpid != 0:
                    to_reap.discard(pid)
            if to_reap and time.monotonic() <= deadline:
                time.sleep(_STARTUP_POLL_INTERVAL_S)
            else:
                break

        # Leader-reaped-but-descendants-alive sweep: a group-leader child that
        # died on its own (barrier waitpid'd it, so it is in `reaped` and was
        # skipped above) may have left grandchildren it forked before dying.
        # Those inherited its process group and were reparented to init, so they
        # are unreachable by waitpid but still reachable by killpg on the leader's
        # pgid (== the leader pid) as long as the group has a live member.
        # Fire-and-forget: init reaps the orphans. Like every killpg-based
        # reclaim this assumes the reaped leader's pid has not yet been reused as
        # a new group leader (Linux allocates pids ~monotonically, so the reuse
        # window here is negligible).
        if self._is_startup_root:
            for leader_pid in self._startup_group_leader_pids:
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.killpg(leader_pid, signal.SIGKILL)

        self._last_rollback = {
            "graceful": [p for p in graceful if p not in killed],
            "killed": list(killed),
        }

        for shm in self._sub_shms + self._chip_shms + self._next_level_shms:
            try:
                shm.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:  # noqa: BLE001
                pass

        # Release the pre-fork _Worker so a retry / close() won't double-free
        # the HeapRing mmap the C++ ctor grabbed.
        self._worker = None
        self._orch = None

        self._chip_pids.clear()
        self._sub_pids.clear()
        self._next_level_pids.clear()
        self._startup_group_leader_pids.clear()
        self._sub_shms.clear()
        self._chip_shms.clear()
        self._next_level_shms.clear()

    def _cleanup_partial_init(self) -> None:
        """Best-effort cleanup for init() failures before the Worker is public-live.

        One absolute cleanup deadline is created here and shared by every phase
        (including ``_abort_hierarchical``) so the whole rollback is bounded
        end-to-end rather than each phase re-acquiring a full timeout.
        """
        deadline = time.monotonic() + _ROLLBACK_GRACEFUL_TIMEOUT_S

        try:
            self._release_active_remote_slot_refs()
        except BaseException:  # noqa: BLE001
            pass

        remote_sessions = list(self._remote_sessions)
        if self._worker is not None:
            try:
                self._worker.close()
            except BaseException:  # noqa: BLE001
                pass
        self._close_remote_sessions(remote_sessions)
        if self._chip_worker is not None:
            try:
                self._chip_worker.finalize()
            except BaseException:  # noqa: BLE001
                pass
            self._chip_worker = None

        self._remote_sessions.clear()
        self._abort_hierarchical(deadline=deadline)
        self._comm_base_ready = False

    @property
    def live_domains(self) -> dict[str, CommDomainHandle]:
        """Read-only snapshot of currently-live dynamic CommDomain handles.

        Useful for debugging.  Mutating the returned dict has no effect; use
        ``handle.release()`` or ``orch.release_domain(handle)`` to free.
        """
        return dict(self._live_domains)

    def _validate_l3_l2_worker_id(self, worker_id: int) -> None:
        if self.level < 3:
            raise RuntimeError("create_l3_l2_region requires a hierarchical Worker")
        if self._worker is None:
            raise RuntimeError("create_l3_l2_region requires Worker.init()")
        device_ids = self._config.get("device_ids", [])
        if worker_id < 0 or worker_id >= len(device_ids):
            raise ValueError(f"create_l3_l2_region: worker_id {worker_id} outside [0, {len(device_ids)})")

    def _poison_l3_l2_region_from_endpoint_error(self, exc: BaseException) -> bool:
        match = _L3_L2_ENDPOINT_ERROR_REGION_RE.search(str(exc))
        if match is None:
            return False
        region_id = int(match.group(1))
        if region_id == 0:
            return False
        poisoned = False
        for region in self._live_l3_l2_regions:
            if int(region.region_id) == region_id:
                region._poison()
                poisoned = True
        return poisoned

    def _register_l3_l2_orch_comm_host_buffer(self, tensor) -> None:
        if not isinstance(tensor, Tensor):
            raise TypeError("L3-L2 host buffer registration expects a Tensor")
        if tensor.child_memory:
            raise ValueError("L3-L2 payload buffer must be host storage, not child_memory device storage")
        if not tensor.is_contiguous:
            raise ValueError("L3-L2 payload buffer must be contiguous")
        base = int(tensor.data)
        nbytes = int(tensor.nbytes())
        if base <= 0 or nbytes <= 0:
            return
        self._l3_l2_orch_comm_host_buffers[base] = max(
            int(self._l3_l2_orch_comm_host_buffers.get(base, 0)),
            nbytes,
        )

    def _validate_l3_l2_orch_comm_host_buffer(self, tensor) -> None:
        if not isinstance(tensor, Tensor):
            raise ValueError("L3-L2 payload buffer must be a Tensor returned by orch.alloc(...)")
        if tensor.child_memory:
            raise ValueError("L3-L2 payload buffer must be host storage, not child_memory device storage")
        if not tensor.is_contiguous:
            raise ValueError("L3-L2 payload buffer must be contiguous")
        base = int(tensor.data)
        nbytes = int(tensor.nbytes())
        if base <= 0 or nbytes <= 0:
            raise ValueError("L3-L2 payload buffer must have a nonzero address and size")
        registered_nbytes = self._l3_l2_orch_comm_host_buffers.get(base)
        if registered_nbytes is None:
            raise ValueError("L3-L2 payload Tensor is not registered; use a tensor returned by orch.alloc(...)")
        if nbytes > int(registered_nbytes):
            raise ValueError(
                f"L3-L2 payload Tensor size {nbytes} exceeds registered shared storage {registered_nbytes}"
            )

    def _create_l3_l2_region(self, worker_id: int, payload_bytes: int, counter_bytes: int):  # noqa: PLR0912
        if payload_bytes <= 0:
            raise ValueError("create_l3_l2_region: payload_bytes must be positive")
        if counter_bytes <= 0 or counter_bytes % 4 != 0:
            raise ValueError("create_l3_l2_region: counter_bytes must be positive and a multiple of 4")
        self._validate_l3_l2_worker_id(int(worker_id))
        req_shm = SharedMemory(create=True, size=_REGION_CREATE_REQUEST_BYTES)
        reply_shm = SharedMemory(create=True, size=_REGION_CREATE_REPLY_BYTES)
        req_buf = cast(memoryview, req_shm.buf)
        reply_buf = cast(memoryview, reply_shm.buf)
        region_id = 0
        l3_host_mapping = None
        try:
            L3L2RegionCreateRequest(
                magic_version=_REGION_MAGIC_VERSION,
                request_bytes=_REGION_CREATE_REQUEST_BYTES,
                payload_bytes=int(payload_bytes),
                counter_bytes=int(counter_bytes),
            ).encode_into(req_buf)
            worker = self._worker
            assert worker is not None
            worker.control_l3_l2_region_create(int(worker_id), req_shm.name, reply_shm.name)
            # Peek before decode: decode rejects malformed replies, but the
            # child has already created the region and the rollback below
            # still needs the id.
            region_id = peek_region_create_reply_region_id(reply_buf)
            reply = decode_region_create_reply(reply_buf)
            platform = str(self._config.get("platform", ""))
            expected_access_profile = (
                L3L2RegionAccessProfile.SIM_POSIX_SHM
                if platform.endswith("sim")
                else L3L2RegionAccessProfile.ONBOARD_VMM
            )
            counter_offset, total_bytes = validate_region_create_reply(reply, expected_access_profile)
            if platform.endswith("sim"):
                handle = _l3_host_mapped_region_import_sim(reply.backing_shm, int(reply.mapping_bytes))
            else:
                handle = _l3_host_mapped_region_import_onboard(
                    int(reply.device_id),
                    int(reply.shareable_handle),
                    int(reply.mapping_bytes),
                )
            l3_host_mapping = L3HostRegionMapping(
                worker_id=int(worker_id),
                region_id=region_id,
                access_profile=reply.access_profile,
                total_bytes=total_bytes,
                payload_offset=0,
                payload_bytes=int(reply.desc.payload_bytes),
                counter_offset=counter_offset,
                counter_bytes=int(reply.desc.counter_bytes),
                handle=int(handle),
            )
            region = L3L2OrchRegion(self, int(worker_id), reply.desc, l3_host_mapping)
            self._live_l3_l2_regions.append(region)
            return region
        except Exception:
            if l3_host_mapping is not None:
                try:
                    l3_host_mapping.close()
                except RuntimeError:
                    pass
            if region_id:
                try:
                    assert self._worker is not None
                    self._worker.control_l3_l2_region_release(int(worker_id), int(region_id))
                except RuntimeError:
                    pass
            raise
        finally:
            del req_buf
            del reply_buf
            for shm in (req_shm, reply_shm):
                try:
                    shm.close()
                    shm.unlink()
                except (BufferError, FileNotFoundError, OSError):
                    pass

    def _cleanup_l3_l2_regions(self) -> None:
        # Per-region best-effort: every region is attempted (and _expire()d) even
        # if one raises, so a failing region never strands the rest; the first
        # error is raised after all are attempted so close() reports the leak.
        if not self._live_l3_l2_regions:
            return
        regions, self._live_l3_l2_regions = self._live_l3_l2_regions, []
        errors: list[BaseException] = []
        for region in regions:
            try:
                try:
                    region._close_l3_host_mapping()
                    if self._worker is not None:
                        self._worker.control_l3_l2_region_release(region._worker_id, region.region_id)
                finally:
                    region._expire()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            raise errors[0]

    def _close_l3_l2_orch_comm(self) -> None:
        for region in self._live_l3_l2_regions:
            try:
                region._close_l3_host_mapping()
            except RuntimeError:
                pass
        self._live_l3_l2_regions.clear()
        self._l3_l2_orch_comm_host_buffers.clear()

    # ------------------------------------------------------------------
    # Dynamic CommDomain allocation (driven by Orchestrator.allocate_domain;
    # do not call directly from user code — use the orch API.)
    # ------------------------------------------------------------------

    def _ensure_comm_base(self) -> None:
        """Lazily establish the base HCCL/sim communicator across all chips.

        Idempotent — sets ``self._comm_base_ready`` after the first
        successful collective so subsequent ``allocate_domain`` calls skip
        straight to the per-allocation IPC handshake.  Dispatched to every
        ``device_ids`` chip in parallel via CTRL_COMM_INIT control mailbox;
        the chip child runs ``ChipWorker.comm_init`` (which itself caches
        the handle, so a re-dispatch would be a no-op anyway).
        """
        if getattr(self, "_comm_base_ready", False):
            return
        assert self._worker is not None
        device_ids = self._config.get("device_ids", [])
        rootinfo_path = self._comm_plan_rootinfo_path()

        request_shms: dict[int, SharedMemory] = {}
        # Layout: header (rank, nranks) + NUL-terminated rootinfo_path bytes.
        path_bytes = rootinfo_path.encode("utf-8") + b"\x00"
        req_size = _COMM_INIT_HEADER.size + len(path_bytes)
        try:
            for chip_idx, _device_id in enumerate(device_ids):
                req = SharedMemory(create=True, size=req_size)
                req_buf = req.buf
                assert req_buf is not None
                _COMM_INIT_HEADER.pack_into(req_buf, 0, int(chip_idx), int(len(device_ids)))
                req_buf[_COMM_INIT_HEADER.size : _COMM_INIT_HEADER.size + len(path_bytes)] = path_bytes
                request_shms[chip_idx] = req

            dw = self._worker
            errors: dict[int, BaseException] = {}

            def dispatch(chip_idx: int) -> None:
                try:
                    dw.control_comm_init(chip_idx, request_shms[chip_idx].name)
                except BaseException as e:  # noqa: BLE001
                    errors[chip_idx] = e

            threads = [
                threading.Thread(target=dispatch, args=(i,), name=f"comm_init_chip_{i}") for i in range(len(device_ids))
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            if errors:
                first = next(iter(errors.items()))
                raise RuntimeError(
                    f"_ensure_comm_base failed on {len(errors)}/{len(device_ids)} chips; "
                    f"first error chip={first[0]}: {first[1]}"
                )
        finally:
            for shm in request_shms.values():
                try:
                    shm.close()
                    shm.unlink()
                except Exception:  # noqa: BLE001
                    pass
        self._comm_base_ready = True

    def _allocate_domain(  # noqa: PLR0912 -- linear input-validation + per-chip shm staging + dispatch + reply unpack; splitting obscures the fail-fast ordering
        self,
        *,
        name: str,
        workers: tuple[int, ...],
        window_size: int,
        buffers: list[CommBufferSpec],
    ) -> CommDomainHandle:
        # Admission is the run() lease that the driving orchestrator holds;
        # this checks resource presence (not the public lifecycle) so a domain
        # allocation admitted before a concurrent close() published CLOSED still
        # completes during the drain. The _worker check below is that gate.
        if self.level < 3:
            raise RuntimeError("allocate_domain requires level >= 3")
        if self._worker is None:
            raise RuntimeError("allocate_domain requires a hierarchical Worker (_start_hierarchical ran)")
        if not workers:
            raise ValueError("allocate_domain: workers must be non-empty")
        if len(set(workers)) != len(workers):
            raise ValueError(f"allocate_domain: workers contains duplicates: {workers}")
        device_ids = self._config.get("device_ids", [])
        for w in workers:
            if w < 0 or w >= len(device_ids):
                raise ValueError(f"allocate_domain: worker_id {w} outside [0, {len(device_ids)})")
        if window_size <= 0:
            raise ValueError("allocate_domain: window_size must be positive")
        buffer_names = [b.name for b in buffers]
        if len(set(buffer_names)) != len(buffer_names):
            raise ValueError(f"allocate_domain: duplicate buffer names: {buffer_names}")
        # Check buffer carving fits in window BEFORE dispatching: a chip-side
        # overflow would still register the backend allocation (aclrtMalloc
        # already succeeded) but never produce a Handle on the parent, so it
        # would silently leak.  Fail fast here instead.
        total_buffer_nbytes = sum(int(b.nbytes) for b in buffers)
        if total_buffer_nbytes > window_size:
            raise ValueError(
                f"allocate_domain: buffers sum to {total_buffer_nbytes} bytes, exceeds window_size={window_size}"
            )
        if name in self._live_domains:
            raise ValueError(f"allocate_domain: domain {name!r} already live")

        # Lazy base communicator: first orch.allocate_domain on this Worker
        # triggers HCCL RootInfo handshake + EnablePeerAccess on every chip.
        # Cheap enough to do once per Worker; defers cost from init() (which
        # used to pre-bootstrap) to the first DAG that actually needs comm.
        self._ensure_comm_base()

        with self._alloc_id_lock:
            allocation_id = self._next_alloc_id
            self._next_alloc_id += 1

        # Stage per-chip request shms (domain_rank differs per chip) and a
        # per-chip reply shm.  We let the chip child write back its own slot.
        buffer_count = len(buffers)
        req_size = _DOMAIN_REQ_HEADER.size + buffer_count * 8 + len(workers) * 4
        reply_size = _DOMAIN_REPLY_HEADER.size + buffer_count * 8
        # Precompute worker → dense rank for O(1) lookup in the staging /
        # context loops below (and again in _release_domain_handle).  Without
        # this, `workers.index(chip_idx)` makes the hot path quadratic.
        worker_to_rank = {w: r for r, w in enumerate(workers)}

        request_shms: dict[int, SharedMemory] = {}
        reply_shms: dict[int, SharedMemory] = {}
        try:
            for chip_idx in workers:
                req = SharedMemory(create=True, size=req_size)
                req_buf = req.buf
                assert req_buf is not None
                _DOMAIN_REQ_HEADER.pack_into(
                    req_buf,
                    0,
                    int(allocation_id),
                    int(len(workers)),
                    int(worker_to_rank[chip_idx]),  # domain_rank
                    int(window_size),
                    int(buffer_count),
                )
                nbytes_off = _DOMAIN_REQ_HEADER.size
                if buffer_count:
                    struct.pack_into(f"<{buffer_count}Q", req_buf, nbytes_off, *[int(b.nbytes) for b in buffers])
                rank_ids_off = nbytes_off + buffer_count * 8
                struct.pack_into(f"<{len(workers)}I", req_buf, rank_ids_off, *[int(w) for w in workers])
                request_shms[chip_idx] = req

                reply_shms[chip_idx] = SharedMemory(create=True, size=reply_size)

            self._dispatch_control_domain(
                workers=workers,
                request_shms=request_shms,
                reply_shms=reply_shms,
                op="alloc",
                allocation_id=allocation_id,
            )

            contexts: dict[int, ChipDomainContext] = {}
            for chip_idx in workers:
                reply_buf = reply_shms[chip_idx].buf
                assert reply_buf is not None
                (device_ctx, local_window_base, reply_buffer_count) = _DOMAIN_REPLY_HEADER.unpack_from(reply_buf, 0)
                if reply_buffer_count != buffer_count:
                    raise RuntimeError(
                        f"allocate_domain: chip {chip_idx} reply buffer_count={reply_buffer_count} "
                        f"!= requested {buffer_count}"
                    )
                ptrs: list[int] = []
                if buffer_count:
                    ptrs = list(struct.unpack_from(f"<{buffer_count}Q", reply_buf, _DOMAIN_REPLY_HEADER.size))
                contexts[chip_idx] = ChipDomainContext(
                    name=name,
                    domain_rank=worker_to_rank[chip_idx],
                    domain_size=len(workers),
                    device_ctx=int(device_ctx),
                    local_window_base=int(local_window_base),
                    actual_window_size=int(window_size),
                    buffer_ptrs={b.name: ptrs[i] for i, b in enumerate(buffers)},
                )
        finally:
            # Close + unlink local copies regardless of outcome.  Children
            # have already finished reading by the time CONTROL_DONE fires.
            for shm in request_shms.values():
                try:
                    shm.close()
                    shm.unlink()
                except Exception:  # noqa: BLE001
                    pass
            for shm in reply_shms.values():
                try:
                    shm.close()
                    shm.unlink()
                except Exception:  # noqa: BLE001
                    pass

        handle = CommDomainHandle(
            name=name,
            workers=workers,
            contexts=contexts,
            allocation_id=allocation_id,
            _release_fn=self._release_domain_handle,
        )
        self._live_domains[name] = handle
        # The backend windows are now live: record each chip's window base and
        # every carved buffer pointer so a later kind4 (child_memory) dispatch of
        # one of them is validated against its owning chip. Revoked by
        # _release_domain_now just before the backend free (a commit barrier),
        # not by the deferred marker — so the deferred window stays dispatchable.
        with self._child_prov_lock:
            for chip_idx, ctx in contexts.items():
                self._child_prov_record_domain(chip_idx, int(ctx.local_window_base), allocation_id)
                for buf_ptr in ctx.buffer_ptrs.values():
                    self._child_prov_record_domain(chip_idx, int(buf_ptr), allocation_id)
        return handle

    def _release_domain_handle(self, handle: CommDomainHandle) -> None:
        """Mark a handle for release.  Actual backend free is deferred.

        Called by ``CommDomainHandle.release()``.  We do NOT drive
        ``CTRL_RELEASE_DOMAIN`` here because the orch function is allowed
        to have already submitted DAG tasks that capture the handle's
        ``device_ctx`` / ``buffer_ptrs``.  Those tasks must see live
        memory through execution; ``Worker.run`` calls
        ``_execute_pending_domain_releases`` only after ``drain()``.
        """
        if self._worker is None:
            return
        # Pop from live_domains so a subsequent allocate_domain(name=...)
        # call within the same run can reuse the name.  The actual memory
        # is still live until _execute_pending_domain_releases runs.
        self._live_domains.pop(handle.name, None)
        self._pending_release_domains.append(handle)

    def _execute_pending_domain_releases(self) -> None:
        """Drive CTRL_RELEASE_DOMAIN for every queued handle.  Must run
        after ``self._orch._drain()`` so chip-side tasks have completed
        their use of the domain memory.
        """
        if not self._pending_release_domains:
            return
        pending, self._pending_release_domains = self._pending_release_domains, []
        for handle in pending:
            try:
                self._release_domain_now(handle)
                handle._freed = True  # noqa: SLF001 -- runtime owns this transition
            except Exception as e:  # noqa: BLE001
                # A failed release should not block other handles' frees or
                # the rest of Worker.run() shutdown.  Drop from any residual
                # tracking and log; the kernel-side memory may have already
                # been reclaimed by the device_ctx-owning chip's finalize.
                sys.stderr.write(
                    f"Worker._execute_pending_domain_releases: {handle.name!r} "
                    f"(allocation_id={handle.allocation_id}) failed: "
                    f"{type(e).__name__}: {e}\n"
                )
                sys.stderr.flush()

    def _release_domain_now(self, handle: CommDomainHandle) -> None:
        """Synchronous backend release for one handle.  Used by the
        deferred-release path and by the abort/close cleanup helpers."""
        if self._worker is None:
            return
        # Revoke provenance BEFORE the physical free: once release begins the
        # domain's pointers are no longer dispatchable. Revoking after the
        # backend free would leave a use-after-free window (a concurrent
        # copy/dispatch could still validate the being-freed pointer as live),
        # and a partial/failed release would strand a freed pointer as "live"
        # forever. Dropping first is the safe direction — a leak (if the backend
        # free later fails) is recoverable; a use-after-free is not.
        with self._child_prov_lock:
            self._child_prov_drop_domain(handle.allocation_id)
        workers = handle.workers
        # Release payload is just the fixed header — no rank_ids tail; the
        # backend looked them up from its own per-allocation record at
        # alloc time and doesn't need them again.
        req_size = _DOMAIN_REQ_HEADER.size
        worker_to_rank = {w: r for r, w in enumerate(workers)}

        request_shms: dict[int, SharedMemory] = {}
        try:
            for chip_idx in workers:
                req = SharedMemory(create=True, size=req_size)
                req_buf = req.buf
                assert req_buf is not None
                _DOMAIN_REQ_HEADER.pack_into(
                    req_buf,
                    0,
                    int(handle.allocation_id),
                    int(len(workers)),
                    int(worker_to_rank[chip_idx]),
                    0,  # window_size — ignored on release
                    0,  # buffer_count — ignored on release
                )
                request_shms[chip_idx] = req

            self._dispatch_control_domain(
                workers=workers,
                request_shms=request_shms,
                reply_shms=None,
                op="release",
                allocation_id=handle.allocation_id,
            )
        finally:
            for shm in request_shms.values():
                try:
                    shm.close()
                    shm.unlink()
                except Exception:  # noqa: BLE001
                    pass
        self._live_domains.pop(handle.name, None)

    def _dispatch_control_domain(
        self,
        *,
        workers: tuple[int, ...],
        request_shms: dict[int, SharedMemory],
        reply_shms: dict[int, SharedMemory] | None,
        op: str,
        allocation_id: int,
    ) -> None:
        """Fan out CTRL_ALLOC_DOMAIN / CTRL_RELEASE_DOMAIN to all participating chips.

        Each chip's `_Worker.control_*` is a blocking per-mailbox call; we issue
        them on separate threads so the L2-side file barrier can converge.
        Joins all threads; raises on first error after all join.
        """
        dw = self._worker
        assert dw is not None
        errors: dict[int, BaseException] = {}

        def dispatch(chip_idx: int) -> None:
            try:
                req_name = request_shms[chip_idx].name
                if op == "alloc":
                    assert reply_shms is not None
                    dw.control_alloc_domain(chip_idx, req_name, reply_shms[chip_idx].name)
                else:
                    dw.control_release_domain(chip_idx, req_name)
            except BaseException as e:  # noqa: BLE001
                errors[chip_idx] = e

        threads = [threading.Thread(target=dispatch, args=(w,), name=f"{op}_domain_chip_{w}") for w in workers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            first = next(iter(errors.items()))
            raise RuntimeError(
                f"{op}_domain(allocation_id={allocation_id}) failed on "
                f"{len(errors)}/{len(workers)} chips; first error chip={first[0]}: {first[1]}"
            )

    def _release_all_live_domains(self) -> None:
        """Best-effort release of every still-live domain handle (LIFO).

        Called from ``Worker.run`` end-of-run sweep (after pending releases)
        and ``Worker.close``.  Skips the deferred-release machinery
        (``_pending_release_domains``) because by the time this runs, drain
        has already happened — synchronous release of leftover handles is
        safe.  Falls back to immediate backend free + drop from
        ``_live_domains`` on each handle; logs and moves on if one fails.
        """
        for handle in list(self._live_domains.values())[::-1]:
            try:
                # Mark released first (flips handle._released so further
                # indexing raises), then synchronously free.  The handle is
                # not in _pending_release_domains, so we use the direct path.
                if not handle.released:
                    handle._released = True  # noqa: SLF001 -- runtime owns the transition
                self._release_domain_now(handle)
                handle._freed = True  # noqa: SLF001
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(
                    f"Worker._release_all_live_domains: {handle.name!r} release failed: {type(e).__name__}: {e}\n"
                )
                sys.stderr.flush()
                # Keep the un-freed handle in _live_domains so the leak stays
                # detectable: close() reports it as a terminal residual instead
                # of returning success (terminal — it is not retried).

    # ------------------------------------------------------------------
    # memory management — forward to C++ Orchestrator, which holds
    # per-WorkerThread mailbox_mu_ so these are safe to call concurrently
    # with in-flight dispatch on the same chip mailbox.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Child (kind4, device) pointer provenance (guard ②)
    #
    # Every mutator/reader below assumes the caller holds ``_child_prov_lock``,
    # so the enclosing op is atomic. Ordering is safety-first: record after a
    # successful native alloc; revoke before the native free.
    # ------------------------------------------------------------------

    def _child_prov_record_malloc(self, worker_id: int, ptr: int) -> None:
        """Mark ``(worker_id, ptr)`` as a live malloc base (after a successful malloc)."""
        entry = self._child_alloc_prov.get((worker_id, ptr))
        if entry is None:
            # Fully initialise the role BEFORE inserting, so the dict never holds
            # a role-less (dead) entry even if an async unwind lands here.
            entry = _ChildProvEntry()
            entry.malloc_owned = True
            self._child_alloc_prov[(worker_id, ptr)] = entry
        else:
            entry.malloc_owned = True

    def _child_prov_require_malloc_base(self, worker_id: int, ptr: int, *, api: str) -> None:
        """Require ``(worker_id, ptr)`` to be an exact live malloc base (freeable).

        Rejects a wrong-worker pointer, an interior/stale pointer, a double free,
        and a CommDomain pointer (which is revoked by its domain's release, never
        by ``free``).
        """
        entry = self._child_alloc_prov.get((worker_id, ptr))
        if entry is None or not entry.malloc_owned:
            raise ValueError(
                f"Worker.{api}: device pointer 0x{ptr:x} is not a live malloc base on worker "
                f"{worker_id} (wrong worker, already-freed/stale, an interior pointer, or a "
                f"CommDomain buffer that must be released via release_domain)"
            )

    def _child_prov_clear_malloc(self, worker_id: int, ptr: int) -> None:
        """Revoke the malloc role of ``(worker_id, ptr)`` — called BEFORE the native
        free (safety-first), so an interrupted free never leaves the address live."""
        key = (worker_id, ptr)
        entry = self._child_alloc_prov.get(key)
        if entry is None:
            return
        if entry.domain_allocation_ids:
            entry.malloc_owned = False  # still live via a domain — keep the entry
        else:
            del self._child_alloc_prov[key]  # last role — delete directly, no empty state

    def _child_prov_require_live(self, worker_id: int, ptr: int, *, api: str) -> None:
        """Require ``(worker_id, ptr)`` to be a live child pointer (malloc or domain)."""
        entry = self._child_alloc_prov.get((worker_id, ptr))
        if entry is None or not entry.is_live():
            raise ValueError(
                f"Worker.{api}: device pointer 0x{ptr:x} is not a live allocation on worker "
                f"{worker_id} (wrong worker, freed/stale, or an interior pointer)"
            )

    def _child_prov_record_domain(self, worker_id: int, ptr: int, allocation_id: int) -> None:
        """Record a CommDomain window / buffer pointer at exact ``(worker_id, ptr)``."""
        entry = self._child_alloc_prov.get((worker_id, ptr))
        if entry is None:
            entry = _ChildProvEntry()
            self._child_alloc_prov[(worker_id, ptr)] = entry
        entry.domain_allocation_ids.add(allocation_id)

    def _child_prov_drop_domain(self, allocation_id: int) -> None:
        """Drop every pointer recorded by a CommDomain allocation (at the start of
        its physical release, before the backend free — see _release_domain_now)."""
        for key in list(self._child_alloc_prov):
            entry = self._child_alloc_prov[key]
            if allocation_id not in entry.domain_allocation_ids:
                continue
            if entry.malloc_owned or len(entry.domain_allocation_ids) > 1:
                entry.domain_allocation_ids.discard(allocation_id)  # other roles remain
            else:
                del self._child_alloc_prov[key]  # last role — delete directly, no empty state

    @staticmethod
    def _child_ptrs_in_args(args: Any) -> list[tuple[int, int]]:
        """Extract ``(device_ptr, arg_index)`` for every child_memory tensor in ``args``."""
        out: list[tuple[int, int]] = []
        for i in range(args.tensor_count()):
            tensor = args.tensor(i)
            if tensor.child_memory:
                out.append((int(tensor.data), i))
        return out

    def _next_level_target_ids(self) -> Sequence[int]:
        """The full pool of dispatchable next-level worker ids.

        Chip ids ``0..N`` at L3; the stable ``_next_level_worker_ids`` at L4+ (an
        index range would not match the local/remote stable worker ids).
        """
        if self._chip_shms:
            return range(len(self._chip_shms))
        return self._next_level_worker_ids

    def _child_prov_check_dispatch(
        self, child_ptrs: list[tuple[int, int]], candidate_worker_ids: Any, *, api: str
    ) -> None:
        """Validate every child_memory pointer against its unique target worker.

        A child_memory argument must resolve to exactly one eligible target
        worker; ``0`` or ``>= 2`` candidates is ambiguous and rejected (judged on
        the resolved eligibility, not the raw ``worker=-1``). The pointer must be
        a live allocation on that target — else it is being routed to the wrong
        worker, or is stale.
        """
        if not child_ptrs:
            return
        candidates = set(candidate_worker_ids)
        if len(candidates) != 1:
            arg_index = child_ptrs[0][1]
            raise ValueError(
                f"orch.{api}: child_memory argument (arg {arg_index}) cannot resolve a unique "
                f"target worker (eligible={sorted(candidates)}); pin worker= explicitly"
            )
        target = next(iter(candidates))
        for ptr, arg_index in child_ptrs:
            entry = self._child_alloc_prov.get((target, ptr))
            if entry is None or not entry.is_live():
                raise ValueError(
                    f"orch.{api}: child_memory argument (arg {arg_index}, ptr 0x{ptr:x}) is not a "
                    f"live allocation on target worker {target} (wrong worker, stale, or interior pointer)"
                )

    def _clear_child_prov(self) -> None:
        """Drop the whole child-pointer provenance table (close-path hygiene)."""
        with self._child_prov_lock:
            self._child_alloc_prov.clear()

    def _check_chip_worker_id(self, worker_id: int) -> None:
        """Range-check ``worker_id`` against the L3-level chip mailbox set.

        Memory ops are only meaningful at L3 (one chip worker per id).
        At L4+ ``_chip_shms`` is empty and ``next_level_threads_`` holds
        L3 worker children that don't service CTRL_MALLOC / FREE / COPY_*
        — without this guard, ``_Orchestrator.malloc(0)`` would dispatch
        to an L3 child mailbox, get a silent CONTROL_DONE from its
        loop's default branch, and return a garbage pointer.
        """
        if worker_id < 0 or worker_id >= len(self._chip_shms):
            raise IndexError(f"worker_id {worker_id} out of range (have {len(self._chip_shms)} chips)")

    def malloc(self, size: int, worker_id: int = 0) -> int:
        """Allocate memory on next-level chip worker *worker_id*. Returns a pointer."""
        with self._operation_lease("malloc"):
            if self.level == 2:
                assert self._chip_worker is not None
                # L2 is a single chip; worker_id is meaningless there, so the
                # provenance is keyed on the canonical worker 0.
                with self._child_prov_lock:
                    ptr = self._chip_worker.malloc(size)
                    self._child_prov_record_malloc(0, int(ptr))
                    return ptr
            self._check_chip_worker_id(worker_id)
            assert self._orch is not None
            return self._orch.malloc(worker_id, size)

    def free(self, ptr: int, worker_id: int = 0) -> None:
        """Free memory allocated by ``malloc()``."""
        with self._operation_lease("free"):
            if self.level == 2:
                assert self._chip_worker is not None
                # Safety-first commit barrier (mirrors Orchestrator.free): revoke
                # provenance BEFORE the native free so an async unwind after a
                # successful free can never leave a freed address live.
                with self._child_prov_lock:
                    self._child_prov_require_malloc_base(0, int(ptr), api="free")
                    self._child_prov_clear_malloc(0, int(ptr))
                    self._chip_worker.free(ptr)
                return
            self._check_chip_worker_id(worker_id)
            assert self._orch is not None
            self._orch.free(worker_id, ptr)

    def copy_to(self, dst: int, src: int, size: int, worker_id: int = 0) -> None:
        """Copy *size* bytes from host *src* to chip worker *dst*."""
        with self._operation_lease("copy_to"):
            if self.level == 2:
                assert self._chip_worker is not None
                with self._child_prov_lock:
                    self._child_prov_require_live(0, int(dst), api="copy_to")
                    self._chip_worker.copy_to(dst, src, size)
                return
            self._check_chip_worker_id(worker_id)
            assert self._orch is not None
            self._orch.copy_to(worker_id, dst, src, size)

    def copy_from(self, dst: int, src: int, size: int, worker_id: int = 0) -> None:
        """Copy *size* bytes from chip worker *src* to host *dst*."""
        with self._operation_lease("copy_from"):
            if self.level == 2:
                assert self._chip_worker is not None
                with self._child_prov_lock:
                    self._child_prov_require_live(0, int(src), api="copy_from")
                    self._chip_worker.copy_from(dst, src, size)
                return
            self._check_chip_worker_id(worker_id)
            assert self._orch is not None
            self._orch.copy_from(worker_id, dst, src, size)

    # ------------------------------------------------------------------
    # Post-fork zero-copy host buffers
    # ------------------------------------------------------------------

    def create_host_buffer(self, nbytes: int) -> HostBuffer:
        """Allocate a born-shared host buffer, attached into every local L3 child,
        that a later ``run()`` reads/writes with **no per-run copy**.

        Local L3 children are forked during ``init()``; host memory allocated
        afterwards is not in their address space. This hands you memory that is
        *born* in a shm already attached into every child, so there is nothing to
        copy: the child reads and writes the same physical pages the parent sees.

        Returns a :class:`HostBuffer` whose ``buffer`` is a ``memoryview`` over
        that shm. Build a tensor over it with the buffer protocol, framework of
        your choice, and pass it to ``run()`` as usual::

            buf = worker.create_host_buffer(n * 4)
            t = torch.frombuffer(buf.buffer, dtype=torch.float32, count=n)
            t.uniform_(0, 1)                       # in place → lands in the shm
            worker.run(orch(chip, t, out), args=None, config=CallConfig())
            worker.free_host_buffer(buf)           # drop the tensor first

        simpler stays framework-free: torch/numpy appear only on the user's side
        (``frombuffer``). Blocks until every local L3 child has attached the buffer;
        not thread-safe against a concurrent ``run`` / ``create`` / ``free`` on
        the same Worker — drive them from one thread, as the L3 worker is
        otherwise.
        """
        if self.level < 3:
            raise TypeError("create_host_buffer requires a level >= 3 Worker")
        with self._operation_lease("create_host_buffer"):
            return self._create_host_buffer_locked(int(nbytes))

    def _create_host_buffer_locked(self, nbytes: int) -> HostBuffer:
        # A born-shared buffer is mapped into every direct process child (chip
        # and sub alike, via _broadcast_host_control). Only a truly childless L3
        # has nowhere to attach it.
        if not self._chip_shms and not self._sub_shms:
            raise RuntimeError(
                "create_host_buffer requires at least one forked chip or sub child (this Worker has none)"
            )
        assert self._worker is not None

        if nbytes <= 0:
            raise ValueError("create_host_buffer: nbytes must be positive")

        # Create the shm up front, then guard everything after it — mapping the
        # address (``_shm_base_addr``), reserving the registry slot, and the
        # broadcast — under one ``try`` so any failure closes and unlinks the shm
        # instead of leaking a /dev/shm segment. The registry mutation stays under
        # ``_registry_lock`` (mirrors Worker.register's discipline); the slow
        # broadcast runs *outside* the lock — wire-level concurrency is serialized
        # at the C++ mailbox, not here. The born-shared shm's own mapped base is
        # the buffer's data_ptr, so a tensor built over buffer.buffer resolves to
        # this registered range.
        shm = SharedMemory(create=True, size=nbytes)
        token: int | None = None
        data_ptr: int | None = None
        try:
            data_ptr = _shm_base_addr(shm)
            with self._registry_lock:
                token = self._host_buf_token_counter
                self._host_buf_token_counter += 1
                entry = _HostBufEntry(
                    token=token,
                    data_ptr=data_ptr,
                    nbytes=nbytes,
                    shm=shm,
                    shm_name=shm.name,
                    shm_base=data_ptr,
                )
                self._host_buf_registry[data_ptr] = entry
                self._rebuild_host_buf_snapshot()

            payload = _HOST_BUF_MAP_HEADER.pack(token, data_ptr, nbytes) + shm.name.encode("utf-8")
            errors = self._broadcast_host_control(_CTRL_MAP_HOST, payload)
            if errors:
                raise RuntimeError(
                    f"create_host_buffer: MAP_HOST failed on {len(errors)} local L3 children; first error: {errors[0]}"
                )
        except BaseException:
            # Roll back on any failure — a staging error before the map, a partial
            # map, or an exception from the broadcast itself (any of which would
            # otherwise leak the shm): unmap any child that took it, drop the
            # reservation, free the shm. No user view exists yet, so close() cannot
            # be blocked by an exported buffer.
            try:
                if token is not None:
                    self._broadcast_host_unmap(token)
            except Exception as unmap_exc:  # noqa: BLE001 -- must not mask the original failure below
                sys.stderr.write(
                    f"[worker pid={os.getpid()}] WARN: create_host_buffer rollback UNMAP_HOST "
                    f"failed (continuing best-effort): {unmap_exc}\n"
                )
                sys.stderr.flush()
            finally:
                with self._registry_lock:
                    if data_ptr is not None and self._host_buf_registry.pop(data_ptr, None) is not None:
                        self._rebuild_host_buf_snapshot()
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
            raise

        buf_view = shm.buf
        assert buf_view is not None
        return HostBuffer(token=token, data_ptr=data_ptr, nbytes=nbytes, buffer=buf_view)

    def free_host_buffer(self, handle: HostBuffer) -> None:
        """Release a born-shared buffer created by ``create_host_buffer``.

        Unmaps it from every local L3 child and frees the parent shm. Drop every
        tensor / ``memoryview`` you built over ``handle.buffer`` *first*: a live
        view keeps the shm's pages exported, so ``close()`` cannot release them
        and the buffer only warns (and is reclaimed once the last view is gone).

        Best-effort and idempotent: a stale handle whose token no longer matches
        (e.g. freed twice) is a silent no-op.
        """
        if not isinstance(handle, HostBuffer):
            raise TypeError("free_host_buffer expects a HostBuffer from create_host_buffer")
        with self._operation_lease("free_host_buffer"):
            self._free_host_buffer_locked(handle)

    def _free_host_buffer_locked(self, handle: HostBuffer) -> None:
        with self._registry_lock:
            entry = self._host_buf_registry.get(handle.data_ptr)
            if entry is None or entry.token != handle.token:
                return
            self._host_buf_registry.pop(handle.data_ptr, None)
            self._rebuild_host_buf_snapshot()
        errors: list[str] = []
        try:
            # Gate on resource presence, not lifecycle: the child mailboxes are
            # driveable whenever the C++ _worker is up — including during close()
            # teardown (CLOSED), when the children are still alive to unmap.
            if self._worker is not None:
                errors = self._broadcast_host_unmap(entry.token)
        except Exception as exc:  # noqa: BLE001
            errors = [str(exc)]
        finally:
            close_warn = self._close_host_shm(entry)
            if close_warn:
                errors.append(close_warn)
        if errors:
            sys.stderr.write(
                f"[worker pid={os.getpid()}] WARN: free_host_buffer token={entry.token} "
                f"failed on {len(errors)} local L3 children; first error: {errors[0]}\n"
            )
            sys.stderr.flush()

    @staticmethod
    def _close_host_shm(entry: _HostBufEntry) -> str | None:
        """Close + unlink a host-buffer's parent shm.

        Returns a warning string (else ``None``) when a still-live view over a
        zero-copy buffer blocks ``close()``: ``memoryview.release()`` raises
        ``BufferError`` while a tensor built via ``frombuffer`` still holds the
        pages exported. The name is unlinked regardless, so the OS reclaims the
        segment once the user drops that last view.
        """
        warn: str | None = None
        try:
            entry.shm.close()
        except BufferError:
            warn = (
                f"host buffer token={entry.token} still has a live view (a tensor/memoryview over "
                f"buffer.buffer); drop it before free_host_buffer/close() to release the shm promptly"
            )
        try:
            entry.shm.unlink()
        except FileNotFoundError:
            pass
        return warn

    def _release_all_host_buffers(self) -> None:
        """Unmap + free every still-registered host buffer (called from close()).

        Per-buffer best-effort: every buffer's shm is closed even if its unmap
        broadcast (or a prior buffer) fails, so one failure never strands the
        rest; the first error is raised after all are attempted so close()
        reports the leak rather than swallowing it to stderr."""
        with self._registry_lock:
            entries = list(self._host_buf_registry.values())
            self._host_buf_registry.clear()
            self._rebuild_host_buf_snapshot()
        errors: list[BaseException] = []
        for entry in entries:
            try:
                try:
                    if self._worker is not None:  # resource presence, not lifecycle (see _close_host_shm)
                        self._broadcast_host_unmap(entry.token)
                finally:
                    # Tolerates a still-live view over a zero-copy buffer at close():
                    # unlinks the name regardless so the OS reclaims it once dropped.
                    self._close_host_shm(entry)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            raise errors[0]

    def _broadcast_host_unmap(self, token: int) -> list[str]:
        """Broadcast _CTRL_UNMAP_HOST for ``token`` to every local L3 child."""
        return self._broadcast_host_control(_CTRL_UNMAP_HOST, _HOST_BUF_UNMAP.pack(token))

    def _broadcast_host_control(self, sub_cmd: int, payload: bytes) -> list[str]:
        if self._worker is None:
            return []
        results = []
        for worker_type in (WorkerType.NEXT_LEVEL, WorkerType.SUB):
            results.extend(
                self._worker.broadcast_control_all(
                    worker_type,
                    int(sub_cmd),
                    payload,
                    None,
                    timeout_s=self._py_control_timeout_s,
                )
            )
        return self._control_errors(results)

    def _stage_host_buffers_for_chip_submit(self, args: Any) -> None:
        """Validate the host tensors of one chip submit before dispatch.

        Called from ``Orchestrator.submit_next_level`` on the LOCAL_CHIP path —
        only there does the forked child dereference raw host pointers. Each host
        tensor is either:

        * inside a buffer from ``create_host_buffer`` → born-shared, so its bytes
          already live in the child-visible shm and the child writes results back
          into the same physical pages: nothing to copy. ``_find_host_buf_entry``
          still validates the view fits inside the buffer (else the child would
          read past its shm mapping);
        * unregistered → forwarded unvalidated. A fork-inherited ``share_memory_``
          tensor is the legitimate case; an unregistered post-fork tensor reads
          stale/unmapped memory in the child — allocate it with
          ``create_host_buffer`` instead.

        The child rewrites in-range host pointers to its own mapping; see
        _rewrite_blob_host_addrs.
        """
        for i in range(args.tensor_count()):
            tensor = args.tensor(i)
            if tensor.child_memory:
                continue
            addr = int(tensor.data)
            if addr == 0:
                continue
            # Raises if an in-range view overruns its buffer; otherwise there is
            # nothing to do — the born-shared bytes are already child-visible.
            self._find_host_buf_entry(addr, int(tensor.nbytes()))

    def _rebuild_host_buf_snapshot(self) -> None:
        """Rebuild the lock-free read snapshot from the registry.

        Caller must hold ``_registry_lock``. Rebinds ``_host_buf_snapshot`` to a
        fresh ``(sorted_ptrs_tuple, registry_copy)`` pair so an in-flight
        ``_find_host_buf_entry`` keeps reading the prior immutable snapshot until
        the single atomic rebind swaps it — see ``_find_host_buf_entry``.
        """
        registry = dict(self._host_buf_registry)
        self._host_buf_snapshot = (tuple(sorted(registry)), registry)

    def _find_host_buf_entry(self, addr: int, nbytes: int) -> _HostBufEntry | None:
        """Host buffer whose ``[data_ptr, data_ptr+nbytes)`` contains the whole
        ``[addr, addr+nbytes)`` view, or None. Raises if a view starts inside a
        buffer but runs past its end (would read past the shm in the child).

        Host buffers are distinct, non-overlapping allocations, so the only
        candidate for ``addr`` is the entry with the greatest base ``<= addr`` —
        found by bisecting the snapshot's sorted keys so this stays log-time on
        the per-submit hot path rather than scanning every buffer.

        Sub-view matching assumes the blob's ``Tensor.buffer.addr`` is the
        contiguous base of the host buffer (``make_tensor_arg`` builds tensors with
        ``start_offset == 0``); a non-zero ``start_offset`` would shift ``addr``
        and is not modelled here.
        """
        # Load the immutable snapshot once: sorted keys and the dict they index
        # into are captured together, so a concurrent create/free (which rebinds a
        # fresh snapshot) cannot mutate what we bisect and index here — no lock, no
        # IndexError from a shrinking list, no torn key/dict pairing.
        sorted_ptrs, registry = self._host_buf_snapshot
        idx = bisect.bisect_right(sorted_ptrs, addr) - 1
        if idx < 0:
            return None
        entry = registry.get(sorted_ptrs[idx])
        if entry is None or addr >= entry.data_ptr + entry.nbytes:
            return None
        if addr + nbytes > entry.data_ptr + entry.nbytes:
            raise RuntimeError(
                f"Host tensor 0x{addr:x} (+{nbytes} B) overruns its host buffer "
                f"0x{entry.data_ptr:x} (+{entry.nbytes} B); create a buffer at least as large."
            )
        return entry

    # ------------------------------------------------------------------
    # run — uniform entry point
    # ------------------------------------------------------------------

    def run(self, callable, args=None, config=None) -> None:
        """Execute one task (L2) or one DAG (L3+) synchronously.

        Dispatch:
          - L2: ``callable`` is a ``CallableHandle`` returned by
            ``Worker.register(chip_callable)``. Routes to the private slot
            carried by the handle.
          - L3+: ``callable`` is a Python orch fn invoked with the
            ``Orchestrator`` handle.

        ``args``  : TaskArgs (optional)
        ``config``: CallConfig (optional, default-constructed if None)

        Returns ``None``. Per-stage run timing (host wall, on-NPU device wall +
        AICPU phase breakdown) is no longer returned — the platform emits it as
        ``[STRACE]`` log markers from each L2 ``simpler_run``, so the L3
        dispatcher and its L2 children are observed uniformly. Parse the markers
        with ``simpler_setup.tools.strace_timing`` (see
        ``docs/dfx/host-trace.md``).
        """
        with self._operation_lease("run"):
            self._run_locked(callable, args, config)

    def _run_locked(self, callable, args, config) -> None:
        cfg = config if config is not None else CallConfig()

        if self.level == 2:
            assert self._chip_worker is not None
            state = self._resolve_handle(callable, expected_namespace="LOCAL_CHIP")
            self._chip_worker._run_slot(state.slot_id, args, cfg)
            return None

        assert self._orch is not None
        assert self._worker is not None
        # Drop any error stashed by a previous run() so this call starts
        # clean. drain() rethrows on the way out; every successful run()
        # leaves the error slot empty, but an unrelated caller may have
        # poked it.
        self._orch._clear_error()
        self._orch._scope_begin()
        try:
            callable(self._orch, args, cfg)
        finally:
            # Always release scope refs and drain so ring slots aren't
            # stranded when the orch fn raises mid-DAG. drain() also
            # rethrows the first dispatch failure for this run — that
            # is how child-task exceptions surface to the caller of
            # Worker.run(). scope_end deliberately does NOT throw: if
            # it did, released refs would be incomplete and drain
            # would hang on in-flight tasks.
            self._orch._scope_end()
            # ORDER MATTERS: drain() must complete first so any in-flight
            # task that captured a now-pending handle's device_ctx /
            # buffer_ptrs sees live memory.  THEN execute the pending
            # backend releases.  Last, sweep any handles that the orch
            # function neither released nor passed out (covers exception
            # unwind and "forgot to release" — auto-release in LIFO).
            # drain() rethrows the first chip-task/dispatch failure, so the
            # cleanup lives in a finally: a failed task must not strand
            # backend domain allocations into the next run.
            try:
                try:
                    self._orch._drain()
                except Exception as e:
                    self._poison_l3_l2_region_from_endpoint_error(e)
                    raise
            finally:
                self._release_active_remote_slot_refs()
                self._flush_pending_remote_frees()
                try:
                    self._cleanup_l3_l2_regions()
                finally:
                    self._l3_l2_orch_comm_host_buffers.clear()
                self._execute_pending_domain_releases()
                if self._live_domains:
                    self._release_all_live_domains()
        # L3+ returns None like every other worker level; per-L2-child timing
        # is emitted as `[STRACE]` markers from each simpler_run.
        return None

    @property
    def aicpu_dlopen_count(self) -> int:
        """L2 only: number of distinct callable identities the AICPU has dlopened for.

        Used by tests to assert that ``register`` + repeated ``run(handle)``
        calls do not retrigger the AICPU dlopen for an already-seen identity.
        Returns 0 on non-L2 workers.
        """
        if self.level != 2 or self._chip_worker is None:
            return 0
        return self._chip_worker.aicpu_dlopen_count

    @property
    def host_dlopen_count(self) -> int:
        """L2 only: number of host-side orch SO dlopens (hbg variants).

        Mirrors ``aicpu_dlopen_count`` for the host_build_graph path. Returns
        0 on non-L2 workers or device-orch variants (trb).
        """
        if self.level != 2 or self._chip_worker is None:
            return 0
        return self._chip_worker.host_dlopen_count

    # ------------------------------------------------------------------
    # close
    # ------------------------------------------------------------------

    def _has_native_tree(self) -> bool:
        """A device-bound native object (ChipWorker / _Worker) is live."""
        return self._worker is not None or self._chip_worker is not None

    def _has_live_resources(self) -> bool:
        """Any teardown-owned resource is still present. close() reads this once,
        to decide whether the first (and only) teardown needs to run — it is NOT
        a retry gate; teardown is terminal (see close()). Covers the native tree,
        child pids/shms, L3-L2 regions, live CommDomains, host buffers, and
        pending remote frees/import-releases."""
        return (
            self._has_native_tree()
            or bool(self._sub_pids or self._chip_pids or self._next_level_pids)
            or bool(self._sub_shms or self._chip_shms or self._next_level_shms)
            or bool(self._live_l3_l2_regions)
            or bool(self._live_domains)
            or bool(self._host_buf_registry)
            or bool(self._pending_remote_buffer_frees or self._pending_remote_import_releases)
        )

    def _describe_live_resources(self) -> str:
        """One-line inventory of the resource categories still present, for the
        terminal-close error synthesized when teardown leaves a residual."""
        parts: list[str] = []
        if self._has_native_tree():
            parts.append("native tree")
        n_pids = len(self._sub_pids) + len(self._chip_pids) + len(self._next_level_pids)
        if n_pids:
            parts.append(f"{n_pids} child pid(s)")
        n_shms = len(self._sub_shms) + len(self._chip_shms) + len(self._next_level_shms)
        if n_shms:
            parts.append(f"{n_shms} child shm(s)")
        if self._live_l3_l2_regions:
            parts.append(f"{len(self._live_l3_l2_regions)} L3-L2 region(s)")
        if self._live_domains:
            parts.append(f"{len(self._live_domains)} comm domain(s)")
        if self._host_buf_registry:
            parts.append(f"{len(self._host_buf_registry)} host buffer(s)")
        n_remote = len(self._pending_remote_buffer_frees) + len(self._pending_remote_import_releases)
        if n_remote:
            parts.append(f"{n_remote} pending remote free(s)")
        return ", ".join(parts) if parts else "(none)"

    def close(self) -> None:  # noqa: PLR0912, PLR0915 -- lifecycle linearization: reentrancy / init-guard / join / owner / claim / drain / teardown
        # close() is a permanent commitment against a resource, not a reversible
        # attempt: it publishes CLOSED atomically (the sole public admission
        # fence — the leased live-tree APIs are rejected once CLOSED) and NEVER
        # reverts to READY. Contract:
        #   - reentrant close() (from inside a leased op) is rejected;
        #   - close() while init() is INITIALIZING fails fast — this worker does
        #     not cancel an in-progress init; wait for READY or FAILED;
        #   - a concurrent close() joins the in-flight attempt and observes its
        #     result; the same worker's teardown never runs twice at once;
        #   - teardown is single-shot and TERMINAL: once it runs, an un-reclaimed
        #     resource leaks and a later close() re-raises the same result — it
        #     never re-drives a half-torn tree. Only a drain-timeout (teardown
        #     un-attempted, tree intact) lets a later close() retry once the
        #     in-flight op finishes; a tree with a live op is never torn down;
        #   - native teardown runs only on the init-owner thread (device-bound).
        # `attempt` is None until the claim installs it. The pre-claim checks
        # raise/return before that, so the finally skips completion for them. From
        # the claim on, the attempt is completed in an innermost resilient finally
        # whose only work is three plain attribute assigns (`error`, `incomplete`,
        # then `done`) followed by a locked `notify_all()`. Every fallible step —
        # drain, teardown, residual synthesis, registry detach — runs before it
        # and folds its error into `result`. `done` is set BEFORE the CV acquire,
        # so an async BaseException in the (interruptible) acquire or the notify
        # cannot strand a joiner — the joiner's bounded re-check recovers a
        # skipped notify. The only irreducible window is an async exception landing
        # between the `error`/`incomplete` and `done` plain assigns.
        attempt: _CloseAttempt | None = None
        result: BaseException | None = None
        teardown_tree = False
        try:
            with self._hierarchical_start_cv:
                if threading.get_ident() in self._lease_depth:
                    raise RuntimeError(
                        "Worker.close(): cannot be called from within a run() / create_host_buffer() "
                        "operation on this thread"
                    )
                if self._lifecycle is _Lifecycle.INITIALIZING:
                    raise RuntimeError(
                        "Worker.close(): cannot close while init() is in progress; "
                        "wait for the worker to reach READY or FAILED first"
                    )
                # A caller that WAITS on an in-flight attempt must always resolve
                # against THAT attempt — never re-read _close_completion (a
                # successor may already be installed) and never start a retry
                # (that would race the owner's own retry into a concurrent
                # teardown). Only a fresh entry (below) may retry a drain-timeout.
                joined = self._close_completion
                if joined is not None and not joined.done:
                    # Bounded re-check so a skipped notify (async exception
                    # between publishing `done` and notify_all()) cannot block a
                    # joiner forever — it re-observes `done` within the interval.
                    while not joined.done:
                        self._hierarchical_start_cv.wait(timeout=_CLOSE_JOIN_RECHECK_S)
                    if joined.error is not None:
                        raise joined.error
                    return
                # Fresh entry: the last attempt (if any) is already resolved. A
                # terminal result (teardown ran, or nothing to tear down)
                # replays; only a drain-timeout — teardown un-attempted, tree
                # intact — may be retried by this call.
                prior = self._close_completion
                if prior is not None and prior.done and (self._teardown_attempted or prior.error is None):
                    if prior.error is not None:
                        raise prior.error
                    return
                # A device-bound native object must be finalized on the init-owner
                # thread — always, even after that thread has exited (affinity
                # does not transfer). NEW/FAILED/reclaimed-CLOSED have none.
                owner = self._init_owner_thread
                if self._has_native_tree() and owner is not None and owner is not threading.current_thread():
                    raise RuntimeError(
                        "Worker.close(): a worker with a live native tree must be closed on the thread that "
                        "init()'d it (native teardown is thread-bound)"
                    )
                # Claim: publish CLOSED (permanent admission fence) and install a
                # fresh teardown attempt.
                self._lifecycle = _Lifecycle.CLOSED
                attempt = _CloseAttempt()
                self._close_completion = attempt
                self._hierarchical_start_cv.notify_all()
                # Drain in-flight leases before touching the tree. CLOSED already
                # rejects new leases; a tree with a live op is never torn down.
                # If an op outlives the budget, teardown stays UN-attempted and
                # the tree intact so a later close() can retry once it drains —
                # the one retryable close() path.
                if self._active_ops > 0:
                    drain_deadline = time.monotonic() + _ROLLBACK_GRACEFUL_TIMEOUT_S
                    while self._active_ops > 0:
                        remaining = drain_deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._hierarchical_start_cv.wait(timeout=remaining)
                    if self._active_ops > 0:
                        result = TimeoutError(
                            "Worker.close(): operation(s) still in flight after the cleanup budget "
                            f"({_ROLLBACK_GRACEFUL_TIMEOUT_S}s); teardown deferred (worker stays CLOSED)"
                        )
                if result is None:
                    teardown_tree = self._has_live_resources()
                    # Latch terminal: once we commit to teardown no later close()
                    # re-drives it, whatever the outcome.
                    if teardown_tree:
                        self._teardown_attempted = True
            if teardown_tree:
                self._teardown_ready_tree()
        except BaseException as exc:  # noqa: BLE001
            if result is None:
                result = exc
        finally:
            if attempt is not None:
                had_live = True  # conservative default if a read below is interrupted
                detached_registry: tuple[dict, dict, dict] | None = None
                try:
                    had_live = self._has_live_resources()
                    # Terminal teardown is single-shot and best-effort: a resource
                    # it could not reclaim LEAKS. Never return success with a
                    # residual — if teardown ran and left something behind without
                    # itself raising, synthesize a terminal error.
                    if teardown_tree and result is None and had_live:
                        result = RuntimeError(
                            "Worker.close(): teardown left resources un-reclaimed (leaked): "
                            f"{self._describe_live_resources()}"
                        )
                    # Detach the user-callable registries for every terminal close
                    # (including a NEW/FAILED worker with no native tree) — only a
                    # drain-timeout / mid-drain interrupt (teardown un-attempted)
                    # keeps them for the retry. Swap to a local under the lock;
                    # its refs are released after completion, outside the lock. A
                    # detach failure folds into `result` so every observer of this
                    # attempt sees the SAME outcome — never one success + one error.
                    if not (result is not None and not self._teardown_attempted):
                        with self._registry_lock:
                            detached_registry = (
                                self._callable_registry,
                                self._identity_registry,
                                self._live_handles,
                            )
                            self._callable_registry = {}
                            self._identity_registry = {}
                            self._live_handles = {}
                except BaseException as exc:  # noqa: BLE001
                    if result is None:
                        result = exc
                finally:
                    # Innermost, resilient publish: all plain attribute assigns
                    # (error/incomplete first, then `done`), so a joiner that
                    # observes `done` always observes the result. `done` is set
                    # BEFORE acquiring the CV — the only remaining work under the
                    # lock is notify_all(). A BaseException during the
                    # (interruptible, possibly-blocking) CV acquire therefore
                    # cannot strand the attempt at done=False; a joiner's bounded
                    # re-check then recovers the skipped notify on its own.
                    attempt.error = result
                    attempt.incomplete = result is not None or had_live
                    attempt.done = True
                    with self._hierarchical_start_cv:
                        self._hierarchical_start_cv.notify_all()
                # Post-completion, lock-free: dropping the last refs may run a
                # callable __del__ (which can reenter close()); the attempt is
                # already done, so the reentrant close() resolves against it
                # instead of self-deadlocking.
                del detached_registry
        if result is not None:
            raise result

    @staticmethod
    def _broadcast_child_shutdown(shms: list[SharedMemory]) -> None:
        """Store _SHUTDOWN into every child mailbox in one group (next-level
        children trigger ``inner_worker.close()``; chip/sub children exit their
        serve loop). The first store error is raised after all are attempted."""
        errors: list[BaseException] = []
        for shm in shms:
            try:
                buf = shm.buf
                if buf is not None:
                    _mailbox_store_i32(_buffer_field_addr(buf, _OFF_STATE), _SHUTDOWN)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            raise errors[0]

    @staticmethod
    def _reap_child_groups(  # noqa: PLR0912 -- interleaved reap across groups / bounded poll / conditional shm-free
        groups: list[tuple[list[SharedMemory], list[int]]], deadline: float
    ) -> None:
        """Reap + free every child across ALL groups within one shared deadline.

        SHUTDOWN must already have been broadcast to every group (see
        ``_broadcast_child_shutdown``): this polls every still-pending pid from
        every group each round, so a child wedged in one group never starves the
        reap of healthy children in another (the serial-per-group variant let the
        first stuck group burn the whole budget and left later groups as
        one-poll survivors). ``pids[i]`` pairs with ``shms[i]``; a shm is freed
        ONLY once its pid is reaped (freeing a live child's mailbox is a
        use-after-free), so a survivor keeps BOTH. Teardown is terminal — a
        survivor LEAKS and is reported as an error so close() never returns
        success while a child is alive; an abnormal exit (signal / non-zero code)
        is likewise reported. The first error is raised after every child is
        attempted.
        """
        errors: list[BaseException] = []
        bad_exits: list[str] = []
        # Flat (group, index) work-list over the reap-eligible pairs.
        pending: list[tuple[int, int]] = [
            (g, i) for g, (shms, pids) in enumerate(groups) for i in range(min(len(shms), len(pids)))
        ]
        reaped: set[tuple[int, int]] = set()
        while pending:
            still: list[tuple[int, int]] = []
            for g, i in pending:
                _shms, pids = groups[g]
                try:
                    wpid, status = os.waitpid(pids[i], os.WNOHANG)
                except ChildProcessError:
                    reaped.add((g, i))
                    continue
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)
                    continue  # leave un-reaped (kept below)
                if wpid != 0:
                    reaped.add((g, i))
                    if os.WIFSIGNALED(status):
                        bad_exits.append(f"pid {pids[i]} killed by signal {os.WTERMSIG(status)}")
                    elif os.WIFEXITED(status) and os.WEXITSTATUS(status) != 0:
                        bad_exits.append(f"pid {pids[i]} exited with code {os.WEXITSTATUS(status)}")
                else:
                    still.append((g, i))
            pending = still
            if pending and time.monotonic() <= deadline:
                time.sleep(_STARTUP_POLL_INTERVAL_S)
            else:
                break
        survivors: list[int] = []
        for g, (shms, pids) in enumerate(groups):
            n = min(len(shms), len(pids))
            keep_pids: list[int] = []
            keep_shms: list[SharedMemory] = []
            for i in range(n):
                if (g, i) not in reaped:
                    survivors.append(pids[i])
                    keep_pids.append(pids[i])
                    keep_shms.append(shms[i])
                    continue
                try:
                    shms[i].close()
                    try:
                        shms[i].unlink()
                    except FileNotFoundError:
                        pass
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)
                    keep_shms.append(shms[i])  # shm survives; its pid is already gone
            pids[:] = keep_pids + pids[n:]
            shms[:] = keep_shms + shms[n:]
        if survivors:
            errors.append(TimeoutError(f"child process(es) {survivors} did not exit within the close budget"))
        for msg in bad_exits:
            errors.append(RuntimeError(f"child teardown: {msg}"))
        if errors:
            raise errors[0]

    def _teardown_ready_tree(self) -> None:
        """Tear down the worker's live tree. Called only from close() after it
        has published CLOSED and drained the leased ops, so no leased operation
        is in flight. (register / unregister are not yet lease-fenced against
        close — see the deferred admission item — so a racing register broadcast
        is possible; that gap is out of this PR's scope.)

        Best-effort and error-accumulating: every step runs even if an earlier
        one raised, so one failing resource never strands the rest. Teardown is
        terminal — an un-reclaimed resource LEAKS (it is not retried) and the
        first collected error is re-raised after all steps complete so the leak
        surfaces to the caller. The child-reap grace starts only after SHUTDOWN
        has been broadcast to every group (below), not at teardown entry, so the
        (potentially blocking) pre-child cleanup cannot consume it and reduce the
        reap to a single poll.
        """
        errors: list[BaseException] = []

        def _step(fn) -> None:
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        # Release any orch-allocated CommDomain handles before tearing down the
        # C++ scheduler: once `dw.close()` runs the chip mailboxes are unusable
        # and we can no longer drive CTRL_RELEASE_DOMAIN.
        _step(self._cleanup_l3_l2_regions)
        if self._live_domains:
            _step(self._release_all_live_domains)
        _step(self._clear_child_prov)
        _step(self._release_active_remote_slot_refs)
        _step(self._flush_pending_remote_frees)
        # Host buffers must be released while the local L3 child mailboxes are
        # still usable (before _worker.close()).
        _step(self._release_all_host_buffers)

        if self.level == 2:

            def _finalize_chip() -> None:
                if self._chip_worker:
                    self._chip_worker.finalize()
                    self._chip_worker = None

            _step(_finalize_chip)
        else:

            def _close_worker() -> None:
                if self._worker:
                    self._worker.close()
                    self._worker = None
                    self._orch = None

            _step(_close_worker)
            # Two-phase child shutdown: broadcast SHUTDOWN to EVERY group first,
            # then reap all groups together within the shared deadline. Sending
            # SHUTDOWN per-group-then-reap (serial) let a stuck child in the first
            # group burn the whole budget, so later healthy children — SHUTDOWN
            # late — got a single WNOHANG poll and became permanent survivors.
            groups = [
                (self._sub_shms, self._sub_pids),
                (self._chip_shms, self._chip_pids),
                (self._next_level_shms, self._next_level_pids),
            ]
            for shms, _pids in groups:
                _step(lambda shms=shms: self._broadcast_child_shutdown(shms))
            # Grace starts NOW, once SHUTDOWN is delivered to every group — not at
            # teardown entry — so the (blocking) pre-child cleanup above cannot
            # eat it. Reap removes reclaimed pids/shms in place; a surviving child
            # is left in place and reported as an error (terminal, not retried).
            reap_deadline = time.monotonic() + _ROLLBACK_GRACEFUL_TIMEOUT_S
            _step(lambda: self._reap_child_groups(groups, reap_deadline))
            _step(self._close_l3_l2_orch_comm)
            # Drop next-level worker refs only once their pids/shms are reclaimed.
            if not self._next_level_pids and not self._next_level_shms:
                self._next_level_workers.clear()
                self._next_level_worker_ids.clear()

        if errors:
            raise errors[0]

    def __enter__(self) -> Worker:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
