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
        r = orch.submit_next_level(chip_handle, chip_args_ptr, cfg, worker=0)
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

import contextlib
import ctypes
import enum
import hashlib
import importlib
import json
import logging
import math
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from typing import Any, cast

import cloudpickle
from _task_interface import (  # pyright: ignore[reportMissingImports]
    HOST_STRACE_ENABLED,
    MAX_REGISTERED_CALLABLE_IDS,
    PTO_PIPELINE_MAX_DEPTH,
    RUNTIME_ENV_RING_COUNT,
    WorkerType,
    _emit_host_span,
    _l3_child_onboard_region_close,
    _l3_child_onboard_region_create,
    _mailbox_load_i32,
    _mailbox_store_i32,
    _read_control_copy_request,
    _set_host_span_level_prefix,
    _worker_host_mapped_region_ack_cleanup_error,
    _worker_host_mapped_region_close,
    _worker_host_mapped_region_import_onboard,
    _worker_host_mapped_region_import_sim,
    _worker_host_mapped_region_peek_cleanup_error,
    get_element_size,
    materialize_task_args,
    read_args_from_blob,
)
from _task_interface import (
    _host_spans_active as _native_host_spans_active,
)

from . import _log as _simpler_log
from .buffer import (
    OWNER_INSTANCE_ID_BYTES,
    AccessMode,
    AddressSpace,
    BackendKind,
    Buffer,
    BufferCapability,
    BufferDescriptor,
    CanonicalIdentity,
    ImportContext,
    ImportRegistry,
    capabilities_for_adapter,
    create_host_shared_buffer,
    host_ptr_nbytes,
    mint_owner_instance_id,
    re_export,
    select_adapter,
    wrap_device_malloc,
    wrap_fork_inherited,
    wrap_vmm_window,
)
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
from .comm_endpoints import (
    DEVICE_AICORE,
    DEVICE_AICPU,
    HOST_CPU,
    AdapterKind,
    AdapterProfile,
    AttachmentRole,
    BackendPlan,
    BackendResolver,
    DefaultRegionAccessService,
    EndpointRegistry,
    RegionAccessService,
    RegionLayoutSpec,
    SingleOwner,
    UnsupportedRegionPlan,
    _EndpointTopologyEntry,
    _EndpointTopologySnapshot,
    _format_worker_path,
    _normalize_node_identity,
    at,
    parse_endpoint_path,
)
from .comm_region import MaterializationContext, RegionInstance, materialize_region_instance
from .global_comm_domain import (
    CTRL_GLOBAL_DOMAIN_COPY_FROM,
    CTRL_GLOBAL_DOMAIN_COPY_TO,
    CTRL_GLOBAL_DOMAIN_IMPORT,
    CTRL_GLOBAL_DOMAIN_PREPARE,
    CTRL_GLOBAL_DOMAIN_RELEASE,
    GLOBAL_DOMAIN_DESCRIPTOR_BYTES,
    GLOBAL_DOMAIN_MAX_COPY_BYTES,
    GLOBAL_DOMAIN_MAX_RANKS,
    GLOBAL_DOMAIN_MAX_STRING_BYTES,
    GLOBAL_DOMAIN_PROFILE_A3_FABRIC,
    GLOBAL_DOMAIN_PROFILE_IDS,
    GLOBAL_DOMAIN_VERSION,
    LOCAL_COPY_REPLY,
    LOCAL_COPY_REQUEST,
    LOCAL_DOMAIN_MAGIC,
    LOCAL_IMPORT_REPLY,
    LOCAL_IMPORT_REQUEST,
    LOCAL_PREPARE_REPLY,
    LOCAL_PREPARE_REQUEST,
    LOCAL_RELEASE_REQUEST,
    GlobalCommInitCommand,
    GlobalDomainAttachment,
    GlobalDomainBuffer,
    GlobalDomainCommand,
    GlobalDomainCopyCommand,
    GlobalDomainDescriptor,
    GlobalDomainMember,
    GlobalDomainPhase,
    GlobalDomainReleaseCommand,
    decode_comm_init,
    decode_comm_init_result,
    decode_copy_command,
    decode_copy_result,
    decode_descriptor_table,
    decode_domain_command,
    decode_release_command,
    encode_comm_init,
    encode_comm_init_result,
    encode_copy_command,
    encode_copy_result,
    encode_descriptor_table,
    encode_domain_command,
    encode_release_command,
    resolve_global_comm_capability,
    validate_descriptor_table,
)
from .orchestrator import Orchestrator, _callback_frame_for, _callback_run, direct_control
from .remote_l3_protocol import HOST_TCP_TRANSPORT_PROFILE
from .task_interface import (
    MAILBOX_ERROR_MSG_SIZE,
    MAILBOX_FRAME_SIZE,
    MAILBOX_OFF_ERROR_MSG,
    MAILBOX_PREPARATION_DISPOSITION_VALUES,
    MAILBOX_SIZE,
    MAILBOX_STATE_VALUES,
    CallConfig,
    ChipCallable,
    ChipDomainContext,
    ChipWorker,
    CommBufferSpec,
    CommDomainHandle,
    GlobalCommDomainHandle,
    GlobalCommDomainView,
    RemoteAddressSpace,
    RemoteBufferExport,
    RemoteBufferHandle,
    TaskArgs,
    _initialize_host_log,
    _Worker,
)
from .worker_chip_orch_comm import (
    _CTRL_SHM_TOKEN_BYTES,
    _REGION_CREATE_REPLY,
    _REGION_CREATE_REPLY_BYTES,
    _REGION_CREATE_REQUEST,
    _REGION_CREATE_REQUEST_BYTES,
    _REGION_LAYOUT_ALIGNMENT,
    _REGION_MAGIC_VERSION,
    WorkerChipOrchRegion,
    WorkerChipRegionAccessProfile,
    WorkerChipRegionCreateRequest,
    WorkerHostRegionMapping,
    _align_up,
    _checked_add_u64,
    decode_region_create_reply,
    peek_region_create_reply_region_id,
    validate_region_create_reply,
)
from .worker_level import WorkerLevel
from .worker_level import span_prefix as _span_prefix

# Upper bound on how long the parent waits for every chip's bootstrap mailbox
# to leave IDLE.  Well above a realistic HCCL init (seconds) but short enough
# that a hung child fails the suite instead of the CI job timing out.
_BOOTSTRAP_WAIT_TIMEOUT_S = 120.0
_BOOTSTRAP_POLL_INTERVAL_S = 0.001
_PY_CONTROL_TIMEOUT_S = 30.0
# L2 endpoint metadata currently reaches the parent through the canonical fatal
# text emitted by the orchestration wrapper; keep this pattern in sync with the
# wrapper's ``L3-L2 endpoint error ... region=<id>`` format.
_WORKER_CHIP_ENDPOINT_ERROR_REGION_RE = re.compile(r"\bL3-L2 endpoint error\b[^\n]*\bregion=(\d+)\b")


def _host_spans_active() -> bool:
    """Whether host-span emission is compiled in and enabled at runtime."""
    return HOST_STRACE_ENABLED and _native_host_spans_active()


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
# 6 int32 (aicpu_thread_num, enable_chip_swimlane, enable_dump_args,
# enable_pmu, enable_dep_gen, enable_scope_stats) + uint64 ring sizing
# overrides (3 per-ring arrays of RUNTIME_ENV_RING_COUNT: ring_task_window,
# ring_heap, ring_dep_pool) + 1024-byte NUL-terminated output_prefix. Log config
# travels separately via ChipWorker.init(log_level) — not on per-task wire.
_RUNTIME_ENV_UINT64_FIELD_COUNT = 3 * RUNTIME_ENV_RING_COUNT
_CFG_FMT = struct.Struct("=iiiiii" + ("Q" * _RUNTIME_ENV_UINT64_FIELD_COUNT) + "1024s")
# The generation-safe pipeline lease follows CONFIG. Args start after the
# lease, rounded up to 8 bytes so the first
# Tensor.data (uint64_t at OFF_ARGS+8) is 8-byte aligned, avoiding
# SIGBUS on strict-alignment platforms (aarch64 atomics, some ARM cores).
_PIPELINE_LEASE_FMT = struct.Struct("=IIQ")
_OFF_PIPELINE_LEASE = (_OFF_CONFIG + _CFG_FMT.size + 7) & ~7
_OFF_ARGS = (_OFF_PIPELINE_LEASE + _PIPELINE_LEASE_FMT.size + 7) & ~7
assert _OFF_ARGS % 8 == 0, "_OFF_ARGS must be 8-aligned for Tensor.data"
_OFF_TASK_CALLABLE_HASH = _OFF_ARGS
_OFF_TASK_ARGS_BLOB = _OFF_TASK_CALLABLE_HASH + CALLABLE_HASH_DIGEST_BYTES
# MAILBOX_ARGS_CAPACITY mirrors the C++ constexpr in worker_manager.h so the
# Python reader can bounds-check incoming args blobs. Source-of-truth for the
# constants on the right is the nanobind binding (cannot drift).
# Mirrors MAILBOX_OFF_ACCEPTED / MAILBOX_TASK_ACCEPTED: launch acceptance is a
# sticky word rather than a MailboxState, because a state carrying it is lost
# whenever the child reaches TASK_DONE between two parent polls. The parent
# clears it when it publishes the next task frame.
_OFF_ACCEPTED = MAILBOX_FRAME_SIZE - MAILBOX_ERROR_MSG_SIZE - 8
_TASK_ACCEPTED = 1
_OFF_PREPARATION_DISPOSITION = _OFF_ACCEPTED + 4
# The parent resets the disposition word to NONE when it claims a frame, so a
# child never publishes it: staging reports VALIDATED_ONLY or NATIVE_PREPARED
# and nothing else. Declared so the wire check below covers the whole enum.
_DISPOSITION_NONE = 0
_VALIDATED_ONLY = 1
_NATIVE_PREPARED = 2
_OFF_FRAME_PROTOCOL = _OFF_ACCEPTED - 40
_OFF_FRAME_RUN_ID = _OFF_ACCEPTED - 32
_OFF_FRAME_SLOT_ID = _OFF_ACCEPTED - 24
_OFF_FRAME_GENERATION = _OFF_ACCEPTED - 16
_OFF_FRAME_DISPATCH_ID = _OFF_ACCEPTED - 8
_TASK_PROTOCOL_VERSION = 3
# Mirrors MAILBOX_OFF_SHUTDOWN / MAILBOX_SHUTDOWN_REQUESTED: termination is a
# sticky one-way word on the control frame, not a MailboxState. _OFF_STATE has
# three writers (parent CONTROL_REQUEST, child CONTROL_DONE, C++
# return-to-IDLE), any of which overwrites a _SHUTDOWN store; only a
# terminating parent writes this word, 0 -> 1, and nothing clears it. The word
# is reserved on every frame so a task-args blob can never reach it.
_OFF_SHUTDOWN = _OFF_FRAME_PROTOCOL - 8
_SHUTDOWN_REQUESTED = 1
_MAILBOX_ARGS_CAPACITY = _OFF_SHUTDOWN - _OFF_TASK_ARGS_BLOB
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
_FRAME_STAGED = 8
_TASK_LAUNCHED = 9
_TASK_FAILED = 10
_ACTIVATE = 11
_PREPARE_READY = 12
_TASK_FRAME_COUNT = 2


def _assert_mailbox_wire_constants() -> None:
    """Fail import if these constants disagree with the C++ enums.

    The state word and the disposition word are a cross-process contract: a
    parent writes them and its forked child reads them, so a value that differs
    from the C++ side is a protocol mismatch between two live processes. Nothing
    else catches it — the two declarations are in different languages, so there
    is no compile error, and a wrong value reads as a legal-but-different state
    rather than as corruption. The constants stay literals because they are read
    on the mailbox polling path; this checks them instead of replacing them.
    """
    declared = {
        "IDLE": _IDLE,
        "TASK_READY": _TASK_READY,
        "TASK_DONE": _TASK_DONE,
        "SHUTDOWN": _SHUTDOWN,
        "CONTROL_REQUEST": _CONTROL_REQUEST,
        "CONTROL_DONE": _CONTROL_DONE,
        "INIT_READY": _INIT_READY,
        "INIT_FAILED": _INIT_FAILED,
        "FRAME_STAGED": _FRAME_STAGED,
        "TASK_LAUNCHED": _TASK_LAUNCHED,
        "TASK_FAILED": _TASK_FAILED,
        "ACTIVATE": _ACTIVATE,
        "PREPARE_READY": _PREPARE_READY,
    }
    dispositions = {
        "NONE": _DISPOSITION_NONE,
        "VALIDATED_ONLY": _VALIDATED_ONLY,
        "NATIVE_PREPARED": _NATIVE_PREPARED,
    }
    for name, table, native in (
        ("MailboxState", declared, MAILBOX_STATE_VALUES),
        ("MailboxPreparationDisposition", dispositions, MAILBOX_PREPARATION_DISPOSITION_VALUES),
    ):
        # Report every disagreement at once: a renumbering usually moves several
        # values, and fixing them one import at a time is needless.
        mismatched = sorted(
            f"{key}: python={value} c++={native[key]}"
            for key, value in table.items()
            if key not in native or native[key] != value
        )
        if mismatched:
            raise RuntimeError(
                f"{name} constants in simpler.worker disagree with the C++ enum: "
                + "; ".join(mismatched)
                + ". These cross a process boundary, so the mismatch would surface as a hung or "
                "misrouted child rather than an error."
            )
        # A value the C++ side has and Python does not is a state a child can
        # legitimately publish and this module would not recognise.
        missing = sorted(set(native) - set(table))
        if missing:
            raise RuntimeError(
                f"{name} has enumerators the Python side does not declare: {', '.join(missing)}. "
                "A child can publish them and this module would not recognise the value."
            )


_assert_mailbox_wire_constants()


def _local_task_frame_count(platform: str, _runtime: str, pipeline_depth: int) -> int:
    if platform == "a2a3" and pipeline_depth >= 2:
        return _TASK_FRAME_COUNT
    return 1


def _shm_name(token: str, suffix: str):
    """Deterministic POSIX shm name from the root token and a per-child suffix.
    Returns None (random name) when token is empty. Truncates the token to
    8 hex characters so the full name fits within macOS's 30-character
    POSIX-shm-name limit (PSHMNAMLEN=31 including NUL).

    Each Worker generates its own token, so a name identifies one mailbox of
    one startup epoch. Being a fixed name rather than a random one, a segment
    left behind by a crashed prior run makes ``SharedMemory(create=True)`` fail
    with ``FileExistsError`` instead of being silently bypassed."""
    if not token:
        return None
    return f"sp-{token[:8]}-{suffix}"


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
# A chip/sub child exiting after SHUTDOWN still has to release everything it
# imported — on a large-scope run (dsv4 HBG: ~80 backings incl. a 2 GiB ring
# heap) the per-mapping teardown alone runs ~10 s — so the close-path reap
# gets its own, larger budget. Rollback stays at the tighter value above: its
# wait guards an unlink-only graceful path that is stuck when it exceeds it.
_CLOSE_CHILD_REAP_TIMEOUT_S = 60.0
# Bounded re-check interval for a close() joiner waiting on an in-flight
# _CloseAttempt. A joiner normally wakes immediately on the completing thread's
# notify_all(); the timeout is a backstop so that if that notify is skipped (an
# async BaseException landing between publishing `done` and notifying), the
# joiner still re-observes `done` within this interval instead of blocking
# forever.
_CLOSE_JOIN_RECHECK_S = 1.0
# Upper bound on how long close() waits for a cancelled init() to unwind.
# The cancel token is only observed at cooperative points (the child-readiness
# poll and the READY commit gate), so an init blocked inside a native segment
# — ChipWorker.init(), the fork phase — reaches none of them. Past this bound
# close() raises rather than blocking forever; the init epoch is left to
# complete on its own thread.
_CLOSE_CANCEL_UNWIND_TIMEOUT_S = 60.0
# Bounded re-check interval for a RunHandle waiter parked behind the elected
# waiter. Same backstop role as _CLOSE_JOIN_RECHECK_S: if the elected waiter's
# notify_all() is skipped (an async BaseException landing between publishing the
# terminal state and notifying), a parked waiter re-observes that state within
# this interval instead of blocking forever.
_RUN_HANDLE_WAIT_RECHECK_S = 1.0
# Native cancellation may finish fallible fan-in preparation only on a retry.
# Never turn that contract into an unbounded Python loop: one retry is enough
# to either settle the fence or declare the worker unusable.
_RUN_CANCELLATION_ATTEMPTS = 2


def _copy_span_base(base: int, offset: int) -> int:
    """The address a copy span starts at, ``offset`` bytes into the backing mapped at ``base``.

    The single definition of what a copy offset means. Both ends of every control-plane copy go
    through it: at L2 the Worker advances the base it resolved in this process, at L3+ the chip child
    advances the base its own ``ImportRegistry`` resolved. Neither ever receives an already-advanced
    address, so the two tiers cannot drift on whether an offset was applied once, twice, or by whom.
    """
    return base + offset


def _require_copy_span(extent: int, offset: int, nbytes: int, *, side: str, api: str) -> None:
    """Require ``[offset, offset + nbytes)`` to lie within a backing of ``extent`` bytes.

    Bounded by subtraction rather than by comparing ``offset + nbytes``: Python ints do not wrap, but
    the C++ receive-side gate bounds the same span the same way, and one shape across the boundary is
    what keeps the two from disagreeing about an edge case neither can reach alone.
    """
    if offset < 0:
        raise ValueError(f"Worker.{api}: {side}_offset must be non-negative, got {offset}")
    if nbytes < 0:
        raise ValueError(f"Worker.{api}: nbytes must be non-negative, got {nbytes}")
    if offset > extent or nbytes > extent - offset:
        raise ValueError(f"Worker.{api}: {side} range [{offset}, {offset}+{nbytes}) exceeds the {extent}-byte backing")


# Control sub-commands (written at _OFF_CALLABLE as uint64)
_CTRL_MALLOC = 0
_CTRL_FREE = 1
# Host<->device copy. Both ends are handles: the payload at _OFF_ARGS carries the two descriptors
# and the span (length + one offset per end), and the child resolves each through the ImportRegistry
# that also resolves task arguments — the owner's mapped address is not the child's, and never
# crosses the fork.
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
# Best-effort import-cache invalidation: the owner released a Buffer, this tells every consumer
# (chip or SUB child, or a nested NEXT_LEVEL Worker forwarding it further down its own tree) to
# drop its own materialized mapping for the identity if it has one. Reached via the generic
# broadcast_control_all path (Worker._broadcast_py_control), the same mechanism _CTRL_PY_REGISTER
# et al. already use — the digest slot carries a CanonicalIdentity's three meaningful fields
# packed by _pack_identity_wire, not a callable hash.
_CTRL_IMPORT_RELEASE = 13
# Operation names a child puts in its error message when a control command
# fails, so the parent's re-raised text names the operation and not just a
# numeric sub-command. Absent entries fall back to the raw number.
_CTRL_OP_NAMES = {
    _CTRL_REGISTER: "register",
    _CTRL_UNREGISTER: "unregister",
    _CTRL_PY_REGISTER: "py_register",
    _CTRL_PY_IMPORT_REGISTER: "py_register",
    _CTRL_PY_UNREGISTER: "py_unregister",
    _CTRL_IMPORT_RELEASE: "import_release",
}


_CTRL_WORKER_CHIP_REGION_CREATE = 16
_CTRL_WORKER_CHIP_REGION_RELEASE = 17
_CTRL_COMMITTED_DEVICE_MEMORY = 18
# L4-to-local-L3 envelope for the Global CommDomain control protocol. The
# enclosed command uses remote_l3_protocol.ControlName; values 18-23 belong to
# chip-child controls.
_CTRL_GLOBAL_DOMAIN_NODE = 24
_LOCAL_GLOBAL_CONTROL_HEADER = struct.Struct("<IIQ")
_CTRL_OP_NAMES[_CTRL_GLOBAL_DOMAIN_NODE] = "global_domain"

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

# Domain-allocation reply shm layout: 32-byte header + buffer_ptrs (u64).
_DOMAIN_REPLY_HEADER = struct.Struct("<QQQI4x")
# fields: committed (u64), device_ctx (u64), local_window_base (u64),
#         buffer_count (u32), padding (4 bytes)
assert _DOMAIN_REPLY_HEADER.size == 32
# `committed` leads the header and is written on the chip the instant its window
# exists, before anything else that can fail. It is the only honest answer to
# "does this chip hold an allocation": the RPC's outcome is not, because the
# window is committed well before the reply is written. The parent's shm is
# zero-filled, so an unwritten slot reads as not committed.
_OFF_DOMAIN_REPLY_COMMITTED = 0

# Control args layout (reuses task mailbox fields when state == _CONTROL_*):
#   offset  8 (_OFF_CALLABLE):  uint64  sub-command
#   offset 16:                  uint64  arg0 (size for malloc/register; ptr for free; region id)
#   offset 40:                  uint64  result (returned ptr from malloc)
_CTRL_OFF_ARG0 = 16
_CTRL_OFF_RESULT = 40


class _NoBufferConsumerError(RuntimeError):
    """A level >= 3 Worker has no forked child, so a Buffer it owns can reach no consumer.

    Typed so a caller whose buffers have an in-process consumer instead — the remote L3 runner, whose
    orchestration fn dereferences the backing itself — can tell this refusal apart from every other
    way ``create_buffer`` fails and supply its own backing.
    """


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
        """Module half of the ``module:qualname`` target."""
        return self.target.split(":", 1)[0]

    @property
    def qualname(self) -> str:
        """Qualified-name half of the ``module:qualname`` target."""
        return self.target.split(":", 1)[1]


def _validate_global_comm_capability(label: str, platform: str, comm_profile: str) -> None:
    """Reject a profile the platform's backend cannot implement.

    ``a3-fabric-v1`` maps peer windows through ``aclrtMemFabricHandle``, which
    exists only on real A3 silicon. Every spec that names a comm profile states
    the rule from here, so a new L3 launch path cannot admit a combination the
    others reject.
    """
    if comm_profile not in GLOBAL_DOMAIN_PROFILE_IDS:
        raise ValueError(f"{label}.comm_profile {comm_profile!r} is not supported")
    if comm_profile == GLOBAL_DOMAIN_PROFILE_A3_FABRIC:
        if not platform.startswith("a2a3"):
            raise ValueError(f"{label}.comm_profile {GLOBAL_DOMAIN_PROFILE_A3_FABRIC!r} requires an a2a3 platform")
        if platform.endswith("sim"):
            raise ValueError(f"{label}.comm_profile {GLOBAL_DOMAIN_PROFILE_A3_FABRIC!r} requires real A3 devices")


def _validate_global_device_ranks(label: str, global_device_ranks: tuple[int, ...], device_count: int) -> None:
    """One rank per local device, unique and non-negative."""
    if global_device_ranks and len(global_device_ranks) != device_count:
        raise ValueError(f"{label} must match device_ids length")
    if any(rank < 0 for rank in global_device_ranks) or len(set(global_device_ranks)) != len(global_device_ranks):
        raise ValueError(f"{label} must be unique and non-negative")


@dataclass(frozen=True)
class RemoteWorkerSpec:
    """Describes a remote L3 worker to attach via ``Worker.add_remote_worker``.

    ``transport`` selects the data plane. The shipped daemon accepts
    only the host_tcp profile today.
    """

    # endpoint is "host:port"; host must be a numeric IP (or "localhost").
    # Hostnames are rejected at add_remote_worker time — getaddrinfo resolution is
    # unbounded and uncancellable and would risk pinning startup on a hung DNS.
    endpoint: str
    platform: str
    runtime: str = "tensormap_and_ringbuffer"
    device_ids: tuple[int, ...] = ()
    num_sub_workers: int = 0
    transport: str = HOST_TCP_TRANSPORT_PROFILE
    comm_profile: str = "sim"
    global_device_ranks: tuple[int, ...] = ()
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
        object.__setattr__(self, "comm_profile", str(self.comm_profile))
        object.__setattr__(
            self,
            "session_listen_host",
            None if self.session_listen_host is None else str(self.session_listen_host),
        )
        object.__setattr__(self, "allow_wildcard_session_bind", bool(self.allow_wildcard_session_bind))
        object.__setattr__(self, "device_ids", tuple(int(x) for x in self.device_ids))
        object.__setattr__(self, "global_device_ranks", tuple(int(x) for x in self.global_device_ranks))
        object.__setattr__(self, "num_sub_workers", int(self.num_sub_workers))
        if self.num_sub_workers < 0:
            raise ValueError("RemoteWorkerSpec.num_sub_workers must be non-negative")
        if self.transport != HOST_TCP_TRANSPORT_PROFILE:
            raise ValueError(
                f"RemoteWorkerSpec.transport must be {HOST_TCP_TRANSPORT_PROFILE!r} for the TCP daemon control plane"
            )
        _validate_global_comm_capability("RemoteWorkerSpec", self.platform, self.comm_profile)
        _validate_global_device_ranks(
            "RemoteWorkerSpec.global_device_ranks", self.global_device_ranks, len(self.device_ids)
        )


@dataclass(frozen=True)
class MpiL3GroupSpec:
    """Describes L3 workers launched by one parent-owned ``mpirun``.

    ``command_port_base``, ``health_port_base``, ``session_listen_hosts``,
    ``connect_hosts``, ``allow_wildcard_session_bind``, ``ready_host``, and
    ``ready_port`` are accepted only for source compatibility with PR #1623.
    MPI groups ignore them and use the local named mailbox plus MPI collectives.
    Non-MPI ``RemoteWorkerSpec`` continues to use its TCP fields.
    """

    hosts: tuple[str, ...]
    platform: str
    device_ids_by_rank: tuple[tuple[int, ...], ...]
    runtime: str = "tensormap_and_ringbuffer"
    num_sub_workers_by_rank: tuple[int, ...] = ()
    transport: str = HOST_TCP_TRANSPORT_PROFILE
    comm_profile: str = "sim"
    global_device_ranks_by_rank: tuple[tuple[int, ...], ...] = ()
    command_port_base: int | None = None
    health_port_base: int | None = None
    session_listen_hosts: tuple[str, ...] = ()
    connect_hosts: tuple[str, ...] = ()
    allow_wildcard_session_bind: bool = False
    ready_host: str = ""
    ready_port: int = 0
    mpirun_path: str = "mpirun"
    mpirun_args: tuple[str, ...] = ()
    python_executable: str = field(default_factory=lambda: sys.executable)

    def __post_init__(self) -> None:  # noqa: PLR0912 -- one place validates the public mpirun rank contract
        hosts = tuple(str(host) for host in self.hosts)
        if not hosts:
            raise ValueError("MpiL3GroupSpec.hosts must be non-empty")
        if not self.platform:
            raise ValueError("MpiL3GroupSpec.platform must be non-empty")
        device_ids_by_rank = tuple(tuple(int(device_id) for device_id in rank) for rank in self.device_ids_by_rank)
        if len(device_ids_by_rank) != len(hosts):
            raise ValueError("MpiL3GroupSpec.device_ids_by_rank must match hosts length")
        if any(not rank for rank in device_ids_by_rank):
            raise ValueError("MpiL3GroupSpec.device_ids_by_rank entries must be non-empty")
        num_sub_workers_by_rank = (
            tuple(0 for _ in hosts)
            if not self.num_sub_workers_by_rank
            else tuple(int(count) for count in self.num_sub_workers_by_rank)
        )
        if len(num_sub_workers_by_rank) != len(hosts):
            raise ValueError("MpiL3GroupSpec.num_sub_workers_by_rank must match hosts length")
        if any(count < 0 for count in num_sub_workers_by_rank):
            raise ValueError("MpiL3GroupSpec.num_sub_workers_by_rank entries must be non-negative")
        global_device_ranks_by_rank = (
            tuple(() for _ in hosts)
            if not self.global_device_ranks_by_rank
            else tuple(tuple(int(rank) for rank in ranks) for ranks in self.global_device_ranks_by_rank)
        )
        if len(global_device_ranks_by_rank) != len(hosts):
            raise ValueError("MpiL3GroupSpec.global_device_ranks_by_rank must match hosts length")
        rank_pairs = zip(global_device_ranks_by_rank, device_ids_by_rank)
        for rank_index, rank_pair in enumerate(rank_pairs):
            global_ranks, device_ids = rank_pair
            _validate_global_device_ranks(
                f"MpiL3GroupSpec.global_device_ranks_by_rank[{rank_index}]", global_ranks, len(device_ids)
            )
        # A global device rank names one device in the whole cluster, so the
        # group's ranks must be disjoint across mpirun ranks as well as within
        # one. validate_member_table enforces the same rule on the members a
        # domain is built from; stating it here names the offending spec field
        # instead of surfacing as an opaque allocation failure.
        flat_global_ranks = [rank for ranks in global_device_ranks_by_rank for rank in ranks]
        if len(set(flat_global_ranks)) != len(flat_global_ranks):
            raise ValueError("MpiL3GroupSpec.global_device_ranks_by_rank must be unique across the whole group")
        session_listen_hosts = tuple(str(host) for host in self.session_listen_hosts)
        connect_hosts = tuple(str(host) for host in self.connect_hosts)
        if session_listen_hosts and len(session_listen_hosts) != len(hosts):
            raise ValueError("MpiL3GroupSpec.session_listen_hosts must match hosts length")
        if connect_hosts and len(connect_hosts) != len(hosts):
            raise ValueError("MpiL3GroupSpec.connect_hosts must match hosts length")
        if any(not host for host in session_listen_hosts) or any(not host for host in connect_hosts):
            raise ValueError("MpiL3GroupSpec host fields must be non-empty")
        command_port_base = None if self.command_port_base is None else int(self.command_port_base)
        health_port_base = None if self.health_port_base is None else int(self.health_port_base)
        for name, value in (("command_port_base", command_port_base), ("health_port_base", health_port_base)):
            if value is not None and (value <= 0 or value + len(hosts) - 1 > 65535):
                raise ValueError(f"MpiL3GroupSpec.{name} deprecated port range must be within 1..65535")
        ready_host = str(self.ready_host)
        ready_port = int(self.ready_port)
        if ready_port < 0 or ready_port > 65535:
            raise ValueError("MpiL3GroupSpec.ready_port must be within 0..65535")
        if self.transport != HOST_TCP_TRANSPORT_PROFILE:
            raise ValueError(f"MpiL3GroupSpec.transport must be {HOST_TCP_TRANSPORT_PROFILE!r}")
        _validate_global_comm_capability("MpiL3GroupSpec", self.platform, self.comm_profile)
        if not self.mpirun_path:
            raise ValueError("MpiL3GroupSpec.mpirun_path must be non-empty")
        if not self.python_executable:
            raise ValueError("MpiL3GroupSpec.python_executable must be non-empty")
        object.__setattr__(self, "hosts", hosts)
        object.__setattr__(self, "platform", str(self.platform))
        object.__setattr__(self, "runtime", str(self.runtime))
        object.__setattr__(self, "transport", str(self.transport))
        object.__setattr__(self, "comm_profile", str(self.comm_profile))
        object.__setattr__(self, "device_ids_by_rank", device_ids_by_rank)
        object.__setattr__(self, "num_sub_workers_by_rank", num_sub_workers_by_rank)
        object.__setattr__(self, "global_device_ranks_by_rank", global_device_ranks_by_rank)
        object.__setattr__(self, "session_listen_hosts", session_listen_hosts)
        object.__setattr__(self, "connect_hosts", connect_hosts)
        object.__setattr__(self, "command_port_base", command_port_base)
        object.__setattr__(self, "health_port_base", health_port_base)
        object.__setattr__(self, "allow_wildcard_session_bind", bool(self.allow_wildcard_session_bind))
        object.__setattr__(self, "ready_host", ready_host)
        object.__setattr__(self, "ready_port", ready_port)
        object.__setattr__(self, "mpirun_path", str(self.mpirun_path))
        object.__setattr__(self, "mpirun_args", tuple(str(arg) for arg in self.mpirun_args))
        object.__setattr__(self, "python_executable", str(self.python_executable))


@dataclass(frozen=True)
class _RemoteSession:
    worker_id: int
    session_id: int
    command_host: str
    command_port: int
    health_host: str
    health_port: int
    pid: int


@dataclass(frozen=True)
class _MpiL3RankSpec:
    platform: str
    runtime: str
    device_ids: tuple[int, ...]
    num_sub_workers: int
    transport: str
    comm_profile: str
    global_device_ranks: tuple[int, ...]


@dataclass(frozen=True)
class _MpiL3RankRuntime:
    group_id: str
    rank: int
    worker_id: int
    session_id: int
    spec: _MpiL3RankSpec


@dataclass
class _MpiL3GroupRuntime:
    group_id: str
    spec: MpiL3GroupSpec
    ranks: tuple[_MpiL3RankRuntime, ...]
    process: subprocess.Popen[Any] | None = None
    manifest_path: str | None = None
    ready_dir: str | None = None
    mailbox: Any | None = None
    monitor_thread: threading.Thread | None = None
    closing: bool = False


@dataclass(frozen=True)
class _RemoteSession:
    worker_id: int
    session_id: int
    command_host: str
    command_port: int
    health_host: str
    health_port: int
    pid: int


@dataclass
class _GlobalNodeDomainState:
    command: GlobalDomainCommand
    prepared_domain_ranks: set[int] = field(default_factory=set)
    descriptors: dict[int, GlobalDomainDescriptor] = field(default_factory=dict)
    local_window_bases: dict[int, int] = field(default_factory=dict)
    mapping_sizes: dict[int, int] = field(default_factory=dict)
    contexts: dict[int, ChipDomainContext] = field(default_factory=dict)
    view: GlobalCommDomainView | None = None
    phase: GlobalDomainPhase = GlobalDomainPhase.PREPARE_EXPORT


@dataclass(frozen=True)
class _GlobalNodeRuntime:
    worker_id: int
    device_ids: tuple[int, ...]
    platform: str
    comm_profile: str
    global_device_ranks: tuple[int, ...]
    node_rank: int
    node_count: int
    cluster_id: str
    is_remote: bool


_IdentitySnapshotEntry = tuple[bytes, Any, int, str, str]


class _SharedExclusiveLock:
    """Held by many in shared mode, or by one alone in exclusive mode.

    Run admission and device control both need ordering against each other, but
    not the same amount of it. A control command that belongs to no run needs
    "no run may be admitted while I run" — a property of the worker. Two such
    commands on different chips do not need to exclude *each other*, and making
    them do so serializes every mailbox round-trip in the tree behind one
    mutex. Admission takes this exclusively, control takes it shared.

    Writer-preferring: once a submit is waiting, new shared holders queue behind
    it, so a stream of control commands cannot starve admission.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._shared = 0
        self._exclusive = False
        self._waiting_exclusive = 0

    @contextlib.contextmanager
    def shared(self) -> Iterator[None]:
        """Admit alongside other shared holders; excludes exclusive holders."""
        with self._cv:
            while self._exclusive or self._waiting_exclusive:
                self._cv.wait()
            self._shared += 1
        try:
            yield
        finally:
            with self._cv:
                self._shared -= 1
                if self._shared == 0:
                    self._cv.notify_all()

    @contextlib.contextmanager
    def exclusive(self) -> Iterator[None]:
        """Admit alone: waits out every shared holder and excludes new ones."""
        with self._cv:
            self._waiting_exclusive += 1
            try:
                while self._exclusive or self._shared:
                    self._cv.wait()
            finally:
                self._waiting_exclusive -= 1
            self._exclusive = True
        try:
            yield
        finally:
            with self._cv:
                self._exclusive = False
                self._cv.notify_all()


# Which Workers this thread already holds a control reservation on. One control
# call can be built out of others (a queue out of a region) and the serializer
# it takes is not re-entrant — but the re-entrance is per Worker: a call that
# reaches a *different* Worker owes that Worker its own reservation.
_CONTROL_RESERVATION = threading.local()


def _held_control_reservations() -> set[int]:
    held = getattr(_CONTROL_RESERVATION, "workers", None)
    if held is None:
        held = set()
        _CONTROL_RESERVATION.workers = held
    return held


def _domain_reply_committed(reply_shm: SharedMemory | None) -> bool:
    """Whether the chip owning *reply_shm* published a committed window.

    The parent creates the shm zero-filled, so a chip that never ran, never
    allocated, or died before its window existed reads as not committed.
    """
    if reply_shm is None:
        return False
    buf = reply_shm.buf
    if buf is None:
        return False
    return struct.unpack_from("<Q", buf, _OFF_DOMAIN_REPLY_COMMITTED)[0] != 0


def _raise_first(step, items) -> None:
    """Apply ``step`` to every item, then raise the first error it produced.

    Teardown of a resource set is terminal: one handle that cannot be reclaimed
    must not strand the rest, and it must not be reported as success either —
    an unreclaimed device resource is what poisons the worker for its successor.
    """
    first_error: BaseException | None = None
    for item in items:
        try:
            step(item)
        except BaseException as exc:  # noqa: BLE001
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


@dataclass
class _ThreadFanoutState:
    item: Any
    claim_lock: Any = field(default_factory=threading.Lock)
    claimed: bool = False
    target_completed: threading.Event = field(default_factory=threading.Event)
    launch_confirmed: bool = False
    attempted_once: bool = False
    terminal_start_failure: bool = False


@dataclass
class _ThreadFanoutCandidate:
    completed: threading.Event = field(default_factory=threading.Event)
    decided: threading.Event = field(default_factory=threading.Event)
    admitted: bool | None = None
    thread: threading.Thread | None = None


@dataclass
class _ThreadFanoutDrainCursor:
    phases: tuple[Any, ...]
    after_phase: Any | None
    next_phase: int = 0

    @property
    def exhausted(self) -> bool:
        return self.next_phase == len(self.phases)

    def advance(self) -> None:
        phase = self.next_phase
        self.phases[phase]()
        self.next_phase = phase + 1
        if self.after_phase is not None:
            self.after_phase(phase)


class _ThreadFanout:
    """Own every possible launch until no target can retain caller state."""

    def __init__(self, items, target, name_prefix: str, after_start, after_phase) -> None:
        self._states = [_ThreadFanoutState(item) for item in items]
        self._target = target
        self._name_prefix = name_prefix
        self._after_start = after_start
        self._after_phase = after_phase
        self._candidates: list[_ThreadFanoutCandidate] = []
        self._first_error: BaseException | None = None

    def _record_error(self, exc: BaseException) -> None:
        if self._first_error is None:
            self._first_error = exc

    def _set_event(self, event: threading.Event) -> None:
        while not event.is_set():
            try:
                event.set()
            except BaseException as exc:  # noqa: BLE001, PERF203
                self._record_error(exc)

    def _wait_event(self, event: threading.Event) -> None:
        while not event.is_set():
            try:
                event.wait()
            except BaseException as exc:  # noqa: BLE001, PERF203
                self._record_error(exc)

    def _publish_decision(self, candidate: _ThreadFanoutCandidate, value: bool) -> None:
        # Admission is one-way.  An interrupt can land after the candidate was
        # admitted but before its rank is recorded as confirmed; cleanup must
        # never retract that publication under a target leaving the gate.
        if candidate.admitted is None:
            candidate.admitted = value
        self._set_event(candidate.decided)

    def _invoke(self, state: _ThreadFanoutState, candidate: _ThreadFanoutCandidate) -> None:
        try:
            self._wait_event(candidate.decided)
            if candidate.admitted is not True:
                return
            with state.claim_lock:
                if state.claimed:
                    return
                state.claimed = True
            try:
                self._target(state.item)
            finally:
                self._set_event(state.target_completed)
        finally:
            self._set_event(candidate.completed)

    def _attempt_start(self, state: _ThreadFanoutState) -> bool | None:
        candidate = _ThreadFanoutCandidate()
        try:
            thread = threading.Thread(
                target=self._invoke,
                args=(state, candidate),
                name=f"{self._name_prefix}{state.item}",
            )
        except Exception as exc:
            self._record_error(exc)
            state.terminal_start_failure = True
            return None
        except BaseException as exc:  # noqa: BLE001
            self._record_error(exc)
            return False
        try:
            candidate.thread = thread
            self._candidates.append(candidate)
            thread.start()
            self._publish_decision(candidate, True)
            state.launch_confirmed = True
            return True
        except Exception as exc:  # noqa: BLE001
            self._record_error(exc)
            self._publish_decision(candidate, False)
            state.terminal_start_failure = True
            return None
        except BaseException as exc:  # noqa: BLE001
            self._record_error(exc)
            self._publish_decision(candidate, False)
            return False

    def _finish_launches(self) -> None:
        # Give every participant one opportunity before retrying an ambiguous
        # BaseException boundary on any one participant.
        for state in self._states:
            if state.attempted_once:
                continue
            if state.launch_confirmed or state.terminal_start_failure:
                state.attempted_once = True
                continue
            result = self._attempt_start(state)
            if result is True and self._after_start is not None:
                self._after_start(state.item)
            state.attempted_once = True

        for state in self._states:
            while not state.launch_confirmed and not state.terminal_start_failure:
                result = self._attempt_start(state)
                if result is True and self._after_start is not None:
                    self._after_start(state.item)

    def _cancel_undecided_candidates(self) -> None:
        for candidate in self._candidates:
            if not candidate.decided.is_set():
                self._publish_decision(candidate, False)

    def _wait_for_targets(self) -> None:
        for state in self._states:
            if state.launch_confirmed:
                self._wait_event(state.target_completed)

    def _join_candidates(self) -> None:
        for candidate in self._candidates:
            thread = candidate.thread
            assert thread is not None
            if not thread._started.is_set():  # noqa: SLF001 -- a cancelled ambiguous start cannot access target state
                continue
            self._wait_event(candidate.completed)
            while True:
                try:
                    thread.join()
                    break
                except BaseException as exc:  # noqa: BLE001, PERF203
                    self._record_error(exc)

    def _drain(self, cursor: _ThreadFanoutDrainCursor) -> None:
        # Every launched candidate may still reference caller-owned state, so
        # this boundary cannot abandon the cursor after an interruption. Keep
        # retrying on a constant stack until ownership is fully drained.
        while True:
            try:
                if cursor.exhausted:
                    return
                cursor.advance()
            except BaseException as exc:  # noqa: BLE001
                self._record_error(exc)

    def run(self) -> None:
        cursor = _ThreadFanoutDrainCursor(
            phases=(
                self._finish_launches,
                self._cancel_undecided_candidates,
                self._wait_for_targets,
                self._join_candidates,
            ),
            after_phase=self._after_phase,
        )
        self._drain(cursor)
        if self._first_error is not None:
            raise self._first_error


def _start_and_join_threads(
    items,
    target,
    *,
    name_prefix: str,
    _after_start: Any | None = None,
    _after_phase: Any | None = None,
) -> None:
    """Start all targets and defer interruptions until every launch is drained.

    ``Thread.start`` may be interrupted after the OS thread exists but before
    it returns. Candidates are owned before start and gated until the launch is
    accepted, so an ambiguous candidate is harmless and can be retried.
    """
    _ThreadFanout(items, target, name_prefix, _after_start, _after_phase).run()


@dataclass
class _IsolatedCallResult:
    value: Any | None = None
    error: BaseException | None = None
    completed: bool = False


def _run_isolated_call(
    result: _IsolatedCallResult,
    operation,
    *,
    name_prefix: str,
    after_success: Any | None = None,
) -> None:
    """Run a control call outside the caller's async-exception boundary."""

    def invoke(_item) -> None:
        try:
            result.value = operation()
            if after_success is not None:
                after_success()
            result.completed = True
        except BaseException as exc:  # noqa: BLE001
            result.error = exc

    _start_and_join_threads((0,), invoke, name_prefix=name_prefix)


@dataclass
class _OwnedSharedMemorySlot:
    shm: SharedMemory | None = None
    owns_name: bool = False
    close_buffer: Any | None = None
    close_mmap: Any | None = None


@dataclass
class _SharedMemoryCreateResult:
    shm: SharedMemory | None = None
    error: BaseException | None = None


class _SharedMemoryOwner:
    """Own fixed named shm slots before any create operation can publish one."""

    def __init__(self, capacity: int) -> None:
        self._slots = [_OwnedSharedMemorySlot() for _ in range(capacity)]
        self._create_cursor = 0
        self._attempted = 0
        self._cleanup_step = 0
        self._cleanup_error: BaseException | None = None

    @staticmethod
    def _create_in_helper(slot: _OwnedSharedMemorySlot, size: int, result: _SharedMemoryCreateResult) -> None:
        while True:
            # Several mailbox control payloads reserve 32 bytes including the
            # trailing NUL for this token. Keep the explicit collision-safe
            # name below that ABI limit while retaining 96 bits of entropy.
            name = f"smp_{uuid.uuid4().hex[:24]}"
            shm = SharedMemory.__new__(SharedMemory)
            shm._name = f"/{name}" if SharedMemory._prepend_leading_slash else name  # noqa: SLF001
            shm._fd = -1  # noqa: SLF001
            shm._mmap = None  # noqa: SLF001
            shm._buf = None  # noqa: SLF001
            slot.shm = shm
            slot.owns_name = False
            try:
                SharedMemory.__init__(shm, name=name, create=True, size=size)
            except FileExistsError as exc:
                if _shm_slot_has_created_resource(slot):
                    slot.owns_name = True
                    result.error = exc
                    return
                slot.shm = None
                continue
            except BaseException as exc:  # noqa: BLE001
                slot.owns_name = _shm_slot_has_created_resource(slot)
                result.error = exc
                return
            slot.owns_name = True
            result.shm = shm
            return

    def create(self, size: int) -> SharedMemory:
        if self._create_cursor >= len(self._slots):
            raise RuntimeError("shared-memory owner capacity exhausted")
        slot = self._slots[self._create_cursor]
        result = _SharedMemoryCreateResult()
        self._attempted = self._create_cursor + 1

        def create_in_helper(_item) -> None:
            try:
                self._create_in_helper(slot, size, result)
            except BaseException as exc:  # noqa: BLE001
                result.error = exc

        _start_and_join_threads((0,), create_in_helper, name_prefix="shm-create-")
        if result.error is not None:
            raise result.error
        shm = result.shm
        if shm is None:
            raise RuntimeError("shared-memory helper completed without a result")
        self._create_cursor += 1
        return shm


def _remember_cleanup_error(first_error: BaseException | None, exc: BaseException) -> BaseException:
    return exc if first_error is None else first_error


def _shm_slot_has_created_resource(slot: _OwnedSharedMemorySlot) -> bool:
    shm = slot.shm
    if shm is None:
        return False
    if os.name == "posix" and getattr(shm, "_fd", -1) >= 0:
        return True
    return getattr(shm, "_mmap", None) is not None or getattr(shm, "_buf", None) is not None


def _release_owned_shm_buffer(slot: _OwnedSharedMemorySlot) -> None:
    shm = slot.shm
    if shm is None:
        return
    if slot.close_buffer is None:
        slot.close_buffer = shm._buf  # noqa: SLF001
    buffer = slot.close_buffer
    if shm._buf is buffer:  # noqa: SLF001
        shm._buf = None  # noqa: SLF001
    if buffer is not None:
        buffer.release()
    slot.close_buffer = None


def _close_owned_shm_mmap(slot: _OwnedSharedMemorySlot) -> None:
    shm = slot.shm
    if shm is None:
        return
    if slot.close_mmap is None:
        slot.close_mmap = shm._mmap  # noqa: SLF001
    mapped = slot.close_mmap
    if shm._mmap is mapped:  # noqa: SLF001
        shm._mmap = None  # noqa: SLF001
    if mapped is not None and not getattr(mapped, "closed", False):
        mapped.close()
    slot.close_mmap = None


def _close_owned_shm_fd(slot: _OwnedSharedMemorySlot) -> None:
    if os.name != "posix" or slot.shm is None:
        return
    shm = slot.shm
    fd = shm._fd  # noqa: SLF001
    if fd < 0:
        return
    shm._fd = -1  # noqa: SLF001
    os.close(fd)


_SHM_CLEANUP_PHASES = 7


class _SharedMemoryCleanupCursor:
    def __init__(self, owner: _SharedMemoryOwner, after_error: Any | None, after_step: Any | None) -> None:
        self._owner = owner
        self._after_error = after_error
        self._after_step = after_step
        self._pending_error: BaseException | None = None
        self._pending_advance = False

    @property
    def exhausted(self) -> bool:
        return self._owner._cleanup_step >= self._owner._attempted * _SHM_CLEANUP_PHASES

    def _finish_step(self) -> None:
        self._owner._cleanup_step += 1
        if self._after_step is not None:
            self._after_step()

    def _queue_error(self, exc: BaseException, *, advance: bool) -> None:
        if self._pending_error is None:
            self._pending_error = exc
            self._pending_advance = advance
        elif self._owner._cleanup_error is None:
            self._owner._cleanup_error = self._pending_error

    def _capture_interruption(self, exc: BaseException) -> None:
        seen: set[int] = set()
        first = exc
        while first.__context__ is not None and id(first) not in seen:
            seen.add(id(first))
            first = first.__context__
        if self._pending_error is None:
            self._pending_error = first
            self._pending_advance = False
        elif self._owner._cleanup_error is None:
            self._owner._cleanup_error = self._pending_error

    def _record_pending_error(self) -> None:
        pending = self._pending_error
        assert pending is not None
        self._owner._cleanup_error = _remember_cleanup_error(self._owner._cleanup_error, pending)
        if self._after_error is not None:
            self._after_error()
        advance = self._pending_advance
        self._pending_error = None
        self._pending_advance = False
        if advance:
            self._finish_step()

    def _run_step(self) -> None:
        slot_index, phase = divmod(self._owner._cleanup_step, _SHM_CLEANUP_PHASES)
        slot = self._owner._slots[slot_index]
        shm = slot.shm
        try:
            if phase == 0:
                slot.owns_name = slot.owns_name or _shm_slot_has_created_resource(slot)
            elif phase == 1:
                if slot.owns_name and os.name == "posix" and shm is not None and shm._name is not None:  # noqa: SLF001
                    resource_tracker.register(shm._name, "shared_memory")  # noqa: SLF001
            elif phase == 2:
                if slot.owns_name and shm is not None:
                    shm.unlink()
            elif phase == 3:
                if slot.owns_name and os.name == "posix" and shm is not None and shm._name is not None:  # noqa: SLF001
                    resource_tracker.register(shm._name, "shared_memory")  # noqa: SLF001
                    resource_tracker.unregister(shm._name, "shared_memory")  # noqa: SLF001
            elif phase == 4:
                _release_owned_shm_buffer(slot)
            elif phase == 5:
                _close_owned_shm_mmap(slot)
            else:
                _close_owned_shm_fd(slot)
        except FileNotFoundError:
            self._finish_step()
        except Exception as exc:  # noqa: BLE001
            self._queue_error(exc, advance=True)
        else:
            self._finish_step()

    def drain(self, pending_interruption: BaseException | None = None) -> None:
        if pending_interruption is not None:
            self._capture_interruption(pending_interruption)
        while not self.exhausted or self._pending_error is not None:
            try:
                if self._pending_error is not None:
                    self._record_pending_error()
                else:
                    self._run_step()
            except BaseException as exc:  # noqa: BLE001
                self._capture_interruption(exc)


def _close_unlink_shms(
    owner: _SharedMemoryOwner,
    *,
    _after_error: Any | None = None,
    _after_step: Any | None = None,
) -> BaseException | None:
    """Drain every owned shm and return the first non-benign cleanup error."""
    pending_interruption: BaseException | None = None
    while True:
        try:
            cursor = _SharedMemoryCleanupCursor(owner, _after_error, _after_step)
            if pending_interruption is None:
                _start_and_join_threads((0,), lambda _item: cursor.drain(), name_prefix="shm-cleanup-")
            else:
                # A setup/fanout interruption is itself a cleanup error, but
                # cannot be allowed to bypass the owner whose names are still
                # live. The cursor records it and drains from the durable step.
                cursor.drain(pending_interruption)
            if not cursor.exhausted:
                cursor.drain()
            return owner._cleanup_error
        except BaseException as exc:  # noqa: BLE001, PERF203
            if pending_interruption is None:
                pending_interruption = exc


def _run_with_owned_shared_memory(
    capacity: int,
    operation,
    *,
    name_prefix: str,
    after_success: Any | None = None,
) -> Any:
    """Run one complete shm lifecycle outside the caller's signal boundary.

    Python delivers ``KeyboardInterrupt`` to the main thread.  Constructing the
    owner inside the isolated helper means an interrupt can either prevent the
    lifecycle from starting, or be deferred until every acquired name has been
    unlinked and closed; it can never abort the caller immediately before its
    cleanup call.
    """
    result = _IsolatedCallResult()

    def run_lifecycle() -> Any:
        owner = _SharedMemoryOwner(capacity)
        value: Any | None = None
        operation_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        cleanup_done = False
        try:
            value = operation(owner)
        except BaseException as exc:  # noqa: BLE001
            operation_error = exc
        while not cleanup_done:
            try:
                cleanup_error = _close_unlink_shms(owner)
                cleanup_done = True
            except BaseException as exc:  # noqa: BLE001
                # This handles a synchronous failure at the cleanup call entry.
                # Caller-directed asynchronous exceptions cannot reach this
                # lifecycle thread.
                operation_error = _remember_cleanup_error(operation_error, exc)
        if operation_error is not None:
            raise operation_error
        if cleanup_error is not None:
            raise cleanup_error
        return value

    _run_isolated_call(result, run_lifecycle, name_prefix=name_prefix, after_success=after_success)
    if result.error is not None:
        raise result.error
    if not result.completed:
        raise RuntimeError("shared-memory lifecycle did not publish an outcome")
    return result.value


def _validate_domain_allocation(
    worker: Worker,
    name: str,
    workers: tuple[int, ...],
    window_size: int,
    buffers: list[CommBufferSpec],
) -> _RunResources:
    """Validate a domain request before any communicator or shm side effect."""
    if worker.level < 3:
        raise RuntimeError("allocate_domain requires level >= 3")
    if worker._worker is None:
        raise RuntimeError("allocate_domain requires a hierarchical Worker (_start_hierarchical ran)")
    resources = worker._building_run_resources
    if resources is None:
        raise RuntimeError("allocate_domain is only valid while a run's graph is being built")
    if not workers:
        raise ValueError("allocate_domain: workers must be non-empty")
    if len(set(workers)) != len(workers):
        raise ValueError(f"allocate_domain: workers contains duplicates: {workers}")
    device_ids = worker._config.get("device_ids", [])
    for worker_id in workers:
        if worker_id < 0 or worker_id >= len(device_ids):
            raise ValueError(f"allocate_domain: worker_id {worker_id} outside [0, {len(device_ids)})")
    if window_size <= 0:
        raise ValueError("allocate_domain: window_size must be positive")
    buffer_names = [buffer.name for buffer in buffers]
    if len(set(buffer_names)) != len(buffer_names):
        raise ValueError(f"allocate_domain: duplicate buffer names: {buffer_names}")
    total_buffer_nbytes = sum(int(buffer.nbytes) for buffer in buffers)
    if total_buffer_nbytes > window_size:
        raise ValueError(
            f"allocate_domain: buffers sum to {total_buffer_nbytes} bytes, exceeds window_size={window_size}"
        )
    if name in worker._live_domains:
        raise ValueError(f"allocate_domain: domain {name!r} already live")
    return resources


def _read_ctrl_staged_shm_name(buf: memoryview) -> str:
    """Decode the staged-payload shm name a broadcast_control_all left at _OFF_ARGS."""
    raw = bytes(buf[_OFF_ARGS : _OFF_ARGS + _CTRL_SHM_NAME_BYTES])
    nul = raw.find(b"\x00")
    return raw[: nul if nul >= 0 else _CTRL_SHM_NAME_BYTES].decode("utf-8", "replace")


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
    contexts: list[tuple[str, str]] = []
    if platform or runtime:
        contexts.append((platform, runtime))
    for child in getattr(worker, "_next_level_workers", []):
        child_context = _chip_descriptor_context(child)
        if child_context != ("", ""):
            contexts.append(child_context)
    for spec in getattr(worker, "_remote_worker_specs", []):
        contexts.append((str(spec.platform), str(spec.runtime)))
    for rank in getattr(worker, "_mpi_rank_by_worker_id", {}).values():
        contexts.append((str(rank.spec.platform), str(rank.spec.runtime)))
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


_IDENTITY_WIRE = struct.Struct(f"<{OWNER_INSTANCE_ID_BYTES}sQI")
assert _IDENTITY_WIRE.size <= CALLABLE_HASH_DIGEST_BYTES


def _pack_identity_wire(identity: CanonicalIdentity) -> bytes:
    """CanonicalIdentity's three meaningful fields, packed to fit the digest-sized control slot
    _CTRL_IMPORT_RELEASE reuses. Deliberately not the identity's own bytes — see
    ``CanonicalIdentity``'s binding docstring on why it exposes no ``pack()``: this reconstructs
    the wire form field-by-field, the same way ``remote_l3_protocol.py`` encodes one for the
    cross-machine wire, rather than dumping the object's raw memory.
    """
    body = _IDENTITY_WIRE.pack(identity.owner_instance_id, identity.buffer_id, identity.generation)
    return body.ljust(CALLABLE_HASH_DIGEST_BYTES, b"\x00")


def _unpack_identity_wire(buf: bytes) -> CanonicalIdentity:
    owner_instance_id, buffer_id, generation = _IDENTITY_WIRE.unpack_from(buf, 0)
    return CanonicalIdentity(owner_instance_id, buffer_id, generation)


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


def _require_matching_pids(shms: list[SharedMemory], pids: list[int], kind: str) -> None:
    """Guard the shm/pid pairing the C++ liveness check depends on.

    Registering a mailbox against the wrong child pid would make the endpoint
    watch an unrelated process, so a length mismatch is rejected instead of
    being silently truncated by ``zip``.
    """
    if len(shms) != len(pids):
        raise RuntimeError(f"{kind} worker shm/pid count mismatch: {len(shms)} mailboxes, {len(pids)} pids")


def _buffer_field_addr(buf, offset: int) -> int:
    """Absolute address of a field inside a shared-memory buffer.

    Used to feed `_mailbox_load_i32` / `_mailbox_store_i32`, which operate on
    raw pointers so the acquire/release semantics match the C++ side
    (worker_manager.cpp::read_mailbox_state / write_mailbox_state).
    """
    return ctypes.addressof(ctypes.c_char.from_buffer(buf)) + offset


def _request_child_shutdown(buf) -> None:
    """Ask the child owning this mailbox to leave its serve loop.

    Sole Python writer of the termination request; the C++ local endpoint is
    the other writer. The sticky ``_OFF_SHUTDOWN`` word goes first so a child
    sampling it between the two stores already leaves by the shutdown path;
    the ``_SHUTDOWN`` state word follows for a child parked on that word alone.
    """
    _mailbox_store_i32(_buffer_field_addr(buf, _OFF_SHUTDOWN), _SHUTDOWN_REQUESTED)
    _mailbox_store_i32(_buffer_field_addr(buf, _OFF_STATE), _SHUTDOWN)


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


def _reexport_args_from_mailbox(buf, worker: Worker) -> TaskArgs:
    """Re-export the mailbox tensor args for an orchestrator (nested L4→L3) child.

    Each received ref's backing is re-exported (per-backing, no map, canonical identity preserved), and
    a new ref carrying the original view (byte_offset / shapes / strides / dtype) is built over it. The
    inner orch fn forwards these to L2 with no map cost (no descriptor pass-through); dependency
    inference keys on the invariant identity. The compute leaf downstream maps lazily.

    Re-export is per-tensor; the container is the same ``TaskArgs`` the submitter built, so the
    scalars ride across unchanged and an orch fn reads its args the same way at every level.
    """
    args_ptr = _buffer_field_addr(buf, _OFF_TASK_ARGS_BLOB)
    args = read_args_from_blob(args_ptr, _MAILBOX_ARGS_CAPACITY)
    out = TaskArgs()
    for i in range(args.tensor_count()):
        ref = args.tensor(i)
        h_prime = worker._reexport(ref.buffer)
        out.add_tensor(
            h_prime.tensor(shapes=ref.shapes, dtype=ref.dtype, strides=ref.strides, byte_offset=ref.byte_offset),
            args.tag(i),
        )
    for i in range(args.scalar_count()):
        out.add_scalar(args.scalar(i))
    return out


# Idle mailbox polls between `getppid()` samples in a forked child. One poll
# costs ~0.1 us, so this samples roughly every 100 us — fast enough that an
# orphan is reaped before it is noticeable, cheap enough to be lost in the
# noise of the poll itself.
_PARENT_LIVENESS_POLL_INTERVAL = 1000


def _run_mailbox_loop(
    buf,
    state_addr: int,
    *,
    handle_task,
    handle_control,
    on_shutdown=None,
) -> None:
    """The mailbox state machine every forked child runs.

    Sole waiter on a child mailbox: sub workers, nested next-level workers and
    chip processes all reach the parent through this one loop, differing only
    in what they do with a task and which control sub-commands they accept.

    `handle_task(task_buf)` and `handle_control(sub_cmd)` each return an
    ``(error_code, message)`` pair and are responsible for catching their own
    exceptions, because the message wording is what the parent re-raises and
    only the caller knows its own context. The pair is published to the
    mailbox error region *before* the DONE state, so the parent never observes
    completion without the matching error report.

    `on_shutdown()` runs on SHUTDOWN before the loop exits, for children that
    own a nested Worker; per-child resource teardown that must survive an
    exception belongs in the caller's own ``finally``.

    A parent that dies without sending SHUTDOWN (SIGKILL from a timeout, an OOM
    kill, a cancelled CI job) would otherwise leave this loop polling a mailbox
    nobody writes to, for the lifetime of the machine. The loop therefore
    samples its own parent and leaves by the SHUTDOWN path once it changes.

    Termination is read from the sticky ``_OFF_SHUTDOWN`` word as well as the
    state word: the ``_CONTROL_DONE`` this loop publishes for an in-flight
    control command overwrites a concurrent ``_SHUTDOWN`` store, and only the
    sticky word survives that.
    """
    parent_pid = os.getppid()
    liveness_countdown = _PARENT_LIVENESS_POLL_INTERVAL
    shutdown_addr = _buffer_field_addr(buf, _OFF_SHUTDOWN)
    # `buf` is a whole mailbox: a base frame followed by the task frames, so
    # frame 1 always exists.
    task_buf = buf[MAILBOX_FRAME_SIZE : 2 * MAILBOX_FRAME_SIZE]
    task_state_addr = _buffer_field_addr(task_buf, _OFF_STATE)
    try:
        while True:
            state = _mailbox_load_i32(state_addr)
            if state == _SHUTDOWN or _mailbox_load_i32(shutdown_addr) == _SHUTDOWN_REQUESTED:
                if on_shutdown is not None:
                    on_shutdown()
                break
            if _mailbox_load_i32(task_state_addr) == _TASK_READY:
                code, msg = handle_task(task_buf)
                _write_error(task_buf, code, msg)
                _mailbox_store_i32(task_state_addr, _TASK_DONE)
            elif state == _CONTROL_REQUEST:
                sub_cmd = struct.unpack_from("Q", buf, _OFF_CALLABLE)[0]
                code, msg = handle_control(int(sub_cmd))
                _write_error(buf, code, msg)
                _mailbox_store_i32(state_addr, _CONTROL_DONE)
            else:
                liveness_countdown -= 1
                if liveness_countdown <= 0:
                    liveness_countdown = _PARENT_LIVENESS_POLL_INTERVAL
                    # Comparing against the pid captured at entry rather than
                    # testing for pid 1: a subreaper (container init, systemd
                    # user session) adopts orphans instead of init, so the pid
                    # changes but never becomes 1. A live parent's pid cannot
                    # change, so this cannot fire spuriously.
                    if os.getppid() != parent_pid:
                        if on_shutdown is not None:
                            on_shutdown()
                        break
    finally:
        if task_buf is not None:
            task_buf.release()


def _sub_worker_loop(
    buf,
    registry: dict[int, Any],
    identity_table: dict[bytes, int],
    identity_refs: dict[bytes, int],
) -> None:
    """Runs in forked child process. Reads unified mailbox layout.

    On success writes ``error=0`` and an empty message. On failure writes
    ``error=1`` and ``f"sub_worker: <ExcType>: <msg>"`` into the mailbox
    error-message region; the parent's endpoint reports it as a failed
    completion, which the run's waiter rethrows as ``std::runtime_error``.
    """
    state_addr = _buffer_field_addr(buf, _OFF_STATE)
    # SUB is a Python host process: no device VA is ever valid here.
    import_registry = ImportRegistry(ImportContext(is_host_endpoint=True))

    def handle_task(task_buf) -> tuple[int, str]:
        digest = _read_task_digest(task_buf)
        cid = identity_table.get(digest)
        fn = registry.get(int(cid)) if cid is not None else None
        if fn is None:
            return 1, f"sub_worker: callable hash {_format_digest(digest)} not registered"
        try:
            # Compute leaf: materialize each arg (map-once) into a MappedArg the Python
            # callable computes on via torch.frombuffer(arg.buffer, ...).
            args_ptr = _buffer_field_addr(task_buf, _OFF_TASK_ARGS_BLOB)
            args = import_registry.mapped_args_from_blob(args_ptr, _MAILBOX_ARGS_CAPACITY)
            fn(args)
        except Exception as e:  # noqa: BLE001
            return 1, _format_exc("sub_worker", e)
        return 0, ""

    def handle_control(sub_cmd: int) -> tuple[int, str]:
        try:
            if sub_cmd == _CTRL_IMPORT_RELEASE:
                import_registry.unregister(_unpack_identity_wire(_read_control_digest(buf)))
            else:
                _handle_py_callable_control(
                    buf,
                    registry,
                    identity_table,
                    identity_refs,
                    sub_cmd,
                    context="sub_worker",
                )
        except Exception as e:  # noqa: BLE001
            return 1, _format_exc("sub_worker control", e)
        return 0, ""

    try:
        _run_mailbox_loop(buf, state_addr, handle_task=handle_task, handle_control=handle_control)
    finally:
        import_registry.close()


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

    # Opened before the collective so the commit can be published the instant
    # the window exists. Everything after that point — the carving bounds check,
    # the pack — can fail with the allocation already made, and a parent that
    # inferred "not allocated" from the failure would leak it.
    reply_shm = SharedMemory(name=reply_shm_name)
    reply_buf = reply_shm.buf
    assert reply_buf is not None
    try:
        handle = _comm_base_handle(cw)  # base communicator handle (cached on the ChipWorker)
        device_ctx, local_window_base = cw._impl.comm_alloc_domain_windows(
            int(handle),
            int(allocation_id),
            rank_ids,
            int(domain_rank),
            int(window_size),
            _buffer_field_addr(reply_buf, _OFF_DOMAIN_REPLY_COMMITTED),
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

        _DOMAIN_REPLY_HEADER.pack_into(reply_buf, 0, 1, int(device_ctx), int(local_window_base), int(buffer_count))
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
class _HostWorkerChipRegion:
    region_id: int
    payload_bytes: int
    counter_offset: int
    counter_bytes: int
    total_bytes: int
    shm: SharedMemory | None = None
    dev_ptr: int = 0
    onboard_handle: int = 0


@dataclass
class _HostWorkerChipRegionStore:
    """Per-chip-child registry of live L3-L2 direct regions (loop-local state)."""

    regions: dict[int, _HostWorkerChipRegion] = field(default_factory=dict)
    next_region_id: int = 1


@dataclass
class _L2GlobalDomain:
    domain_id: int
    generation: int
    domain_rank: int
    rank_count: int
    descriptor: GlobalDomainDescriptor
    local_window_base: int
    mapping_size: int
    requested_window_size: int
    device_ctx: int = 0
    descriptor_table: bytes = b""


@dataclass
class _L2GlobalDomainStore:
    domains: dict[int, _L2GlobalDomain] = field(default_factory=dict)


@dataclass(frozen=True)
class _HostWorkerChipRegionReplyMeta:
    payload_base: int
    backing_name: bytes
    access_profile: WorkerChipRegionAccessProfile
    mapping_bytes: int
    shareable_handle: int


def _release_host_worker_chip_region(region: _HostWorkerChipRegion) -> None:
    if region.shm is not None:
        region.shm.close()
        region.shm.unlink()
        return
    if region.onboard_handle:
        _l3_child_onboard_region_close(region.onboard_handle)


def _create_sim_worker_chip_region(
    request: WorkerChipRegionCreateRequest, region_id: int, counter_offset: int, total_bytes: int
) -> tuple[_HostWorkerChipRegion, _HostWorkerChipRegionReplyMeta]:
    shm = SharedMemory(create=True, size=total_bytes)
    region = _HostWorkerChipRegion(
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
        raise RuntimeError("CTRL_WORKER_CHIP_REGION_CREATE backing shm token is too long")
    meta = _HostWorkerChipRegionReplyMeta(
        payload_base=payload_base,
        backing_name=backing_name,
        access_profile=WorkerChipRegionAccessProfile.SIM_POSIX_SHM,
        mapping_bytes=total_bytes,
        shareable_handle=0,
    )
    return region, meta


def _create_onboard_worker_chip_region(
    cw: ChipWorker, request: WorkerChipRegionCreateRequest, region_id: int, counter_offset: int, total_bytes: int
) -> tuple[_HostWorkerChipRegion, _HostWorkerChipRegionReplyMeta]:
    export = _l3_child_onboard_region_create(total_bytes)
    dev_ptr = int(export.device_addr)
    region = _HostWorkerChipRegion(
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
    meta = _HostWorkerChipRegionReplyMeta(
        payload_base=dev_ptr,
        backing_name=b"",
        access_profile=WorkerChipRegionAccessProfile.ONBOARD_VMM,
        mapping_bytes=int(export.mapping_bytes),
        shareable_handle=int(export.shareable_handle),
    )
    return region, meta


def _handle_ctrl_worker_chip_region_create(
    cw: ChipWorker, buf: memoryview, chip_platform: str, store: _HostWorkerChipRegionStore
) -> None:
    request_shm_name = _read_shm_name(buf, _OFF_ARGS)
    reply_shm_name = _read_shm_name(buf, _OFF_ARGS + _CTRL_SHM_NAME_BYTES)
    req_shm = SharedMemory(name=request_shm_name)
    reply_shm = SharedMemory(name=reply_shm_name)
    req_buf = cast(memoryview, req_shm.buf)
    reply_buf = cast(memoryview, reply_shm.buf)
    region: _HostWorkerChipRegion | None = None
    try:
        fields = _REGION_CREATE_REQUEST.unpack_from(req_buf, 0)
        request = WorkerChipRegionCreateRequest(
            magic_version=int(fields[0]),
            request_bytes=int(fields[1]),
            payload_bytes=int(fields[2]),
            counter_bytes=int(fields[3]),
        )
        # Reject ABI mismatches loudly. The reply carries this child's own
        # magic (not the request echo) so the L3 side can detect version skew.
        if request.magic_version != _REGION_MAGIC_VERSION:
            raise RuntimeError("CTRL_WORKER_CHIP_REGION_CREATE magic_version mismatch")
        if request.request_bytes != _REGION_CREATE_REQUEST_BYTES:
            raise RuntimeError("CTRL_WORKER_CHIP_REGION_CREATE request_bytes mismatch")
        if request.payload_bytes <= 0:
            raise RuntimeError("CTRL_WORKER_CHIP_REGION_CREATE payload_bytes must be positive")
        if request.counter_bytes <= 0 or request.counter_bytes % 4 != 0:
            raise RuntimeError("CTRL_WORKER_CHIP_REGION_CREATE counter_bytes must be positive and a multiple of 4")
        counter_offset = _align_up(request.payload_bytes, _REGION_LAYOUT_ALIGNMENT)
        total_bytes = _checked_add_u64(counter_offset, request.counter_bytes)

        region_id = store.next_region_id
        store.next_region_id += 1
        if str(chip_platform).endswith("sim"):
            region, meta = _create_sim_worker_chip_region(request, region_id, counter_offset, total_bytes)
        else:
            region, meta = _create_onboard_worker_chip_region(cw, request, region_id, counter_offset, total_bytes)
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
                _release_host_worker_chip_region(region)
            except (BufferError, FileNotFoundError, OSError, RuntimeError):
                pass
        del req_buf
        del reply_buf
        req_shm.close()
        reply_shm.close()


def _handle_ctrl_worker_chip_region_release(buf: memoryview, store: _HostWorkerChipRegionStore) -> None:
    region_id = struct.unpack_from("Q", buf, _CTRL_OFF_ARG0)[0]
    region = store.regions.pop(int(region_id), None)
    if region is None:
        return
    _release_host_worker_chip_region(region)


def _sweep_host_worker_chip_regions(store: _HostWorkerChipRegionStore) -> None:
    for region_id in list(store.regions):
        region = store.regions.pop(region_id)
        try:
            _release_host_worker_chip_region(region)
        except (BufferError, FileNotFoundError, OSError, RuntimeError):
            pass


def _open_global_domain_payload(buf: memoryview) -> tuple[SharedMemory, memoryview, int]:
    payload_size = int(struct.unpack_from("Q", buf, _CTRL_OFF_ARG0)[0])
    if payload_size <= 0:
        raise RuntimeError("Global CommDomain control payload must be non-empty")
    staged = SharedMemory(name=_read_ctrl_staged_shm_name(buf))
    staged_buf = cast(memoryview, staged.buf)
    if payload_size > staged.size:
        staged_buf.release()
        staged.close()
        raise RuntimeError("Global CommDomain control payload exceeds staged shm")
    return staged, staged_buf, payload_size


def _validate_local_global_header(
    magic: bytes, version: int, domain_id: int, generation: int, *, operation: str
) -> None:
    if magic != LOCAL_DOMAIN_MAGIC or version != GLOBAL_DOMAIN_VERSION:
        raise RuntimeError(f"{operation}: local protocol magic or version mismatch")
    if domain_id == 0 or generation == 0:
        raise RuntimeError(f"{operation}: domain identity must be positive")


def _handle_ctrl_global_domain_prepare(cw: ChipWorker, buf: memoryview, store: _L2GlobalDomainStore) -> None:
    staged, payload, payload_size = _open_global_domain_payload(buf)
    try:
        if payload_size < max(LOCAL_PREPARE_REQUEST.size, LOCAL_PREPARE_REPLY.size + GLOBAL_DOMAIN_DESCRIPTOR_BYTES):
            raise RuntimeError("Global CommDomain prepare payload is too small")
        fields = LOCAL_PREPARE_REQUEST.unpack_from(payload, 0)
        magic, version, domain_id, generation, domain_rank, rank_count, profile_id, window_size = fields
        _validate_local_global_header(magic, version, domain_id, generation, operation="prepare")
        if rank_count <= 0 or rank_count > GLOBAL_DOMAIN_MAX_RANKS or domain_rank >= rank_count:
            raise RuntimeError("Global CommDomain prepare rank identity is invalid")
        if profile_id not in GLOBAL_DOMAIN_PROFILE_IDS.values() or window_size <= 0:
            raise RuntimeError("Global CommDomain prepare profile or window size is invalid")
        prior = store.domains.get(int(domain_id))
        if prior is not None:
            if (
                prior.generation != generation
                or prior.domain_rank != domain_rank
                or prior.rank_count != rank_count
                or prior.descriptor.profile_id != profile_id
                or prior.requested_window_size != window_size
            ):
                raise RuntimeError("Global CommDomain prepare conflicts with a live domain")
            descriptor = prior.descriptor
            local_base = prior.local_window_base
            mapping_size = prior.mapping_size
        else:
            descriptor_bytes, local_base, mapping_size = cw._impl.comm_global_domain_prepare(
                int(domain_id),
                int(domain_rank),
                int(rank_count),
                int(window_size),
                int(profile_id),
            )
            descriptor = GlobalDomainDescriptor.decode(bytes(descriptor_bytes))
            if (
                descriptor.domain_rank != domain_rank
                or descriptor.rank_count != rank_count
                or descriptor.profile_id != profile_id
                or descriptor.mapping_size != mapping_size
                or descriptor.mapping_size < window_size
            ):
                cw._impl.comm_global_domain_release(int(domain_id))
                raise RuntimeError("Global CommDomain backend returned an inconsistent descriptor")
            store.domains[int(domain_id)] = _L2GlobalDomain(
                domain_id=int(domain_id),
                generation=int(generation),
                domain_rank=int(domain_rank),
                rank_count=int(rank_count),
                descriptor=descriptor,
                local_window_base=int(local_base),
                mapping_size=int(mapping_size),
                requested_window_size=int(window_size),
            )
        LOCAL_PREPARE_REPLY.pack_into(
            payload,
            0,
            LOCAL_DOMAIN_MAGIC,
            GLOBAL_DOMAIN_VERSION,
            int(domain_id),
            int(generation),
            int(local_base),
            int(mapping_size),
        )
        start = LOCAL_PREPARE_REPLY.size
        payload[start : start + GLOBAL_DOMAIN_DESCRIPTOR_BYTES] = descriptor.encode()
    finally:
        payload.release()
        staged.close()


def _handle_ctrl_global_domain_import(cw: ChipWorker, buf: memoryview, store: _L2GlobalDomainStore) -> None:
    staged, payload, payload_size = _open_global_domain_payload(buf)
    try:
        if payload_size < LOCAL_IMPORT_REQUEST.size:
            raise RuntimeError("Global CommDomain import payload is truncated")
        magic, version, domain_id, generation, descriptor_count = LOCAL_IMPORT_REQUEST.unpack_from(payload, 0)
        _validate_local_global_header(magic, version, domain_id, generation, operation="import")
        expected_size = LOCAL_IMPORT_REQUEST.size + int(descriptor_count) * GLOBAL_DOMAIN_DESCRIPTOR_BYTES
        if descriptor_count <= 0 or descriptor_count > GLOBAL_DOMAIN_MAX_RANKS or expected_size > payload_size:
            raise RuntimeError("Global CommDomain import descriptor table size is invalid")
        entry = store.domains.get(int(domain_id))
        if entry is None or entry.generation != generation:
            raise RuntimeError("Global CommDomain import requires a matching prepared domain")
        descriptor_bytes = bytes(payload[LOCAL_IMPORT_REQUEST.size : expected_size])
        descriptors = tuple(
            GlobalDomainDescriptor.decode(descriptor_bytes[offset : offset + GLOBAL_DOMAIN_DESCRIPTOR_BYTES])
            for offset in range(0, len(descriptor_bytes), GLOBAL_DOMAIN_DESCRIPTOR_BYTES)
        )
        profile = next(
            name for name, profile_id in GLOBAL_DOMAIN_PROFILE_IDS.items() if profile_id == entry.descriptor.profile_id
        )
        validate_descriptor_table(descriptors, rank_count=entry.rank_count, profile=profile)
        if descriptors[entry.domain_rank] != entry.descriptor:
            raise RuntimeError("Global CommDomain import table does not contain the local exported descriptor")
        if entry.descriptor_table and entry.descriptor_table != descriptor_bytes:
            raise RuntimeError("Global CommDomain repeated import carries a different descriptor table")
        if entry.device_ctx == 0:
            entry.device_ctx = int(cw._impl.comm_global_domain_import(int(domain_id), descriptor_bytes))
            if entry.device_ctx == 0:
                raise RuntimeError("Global CommDomain backend returned a zero device context")
            entry.descriptor_table = descriptor_bytes
        if payload_size < LOCAL_IMPORT_REPLY.size:
            raise RuntimeError("Global CommDomain import reply capacity is too small")
        LOCAL_IMPORT_REPLY.pack_into(
            payload,
            0,
            LOCAL_DOMAIN_MAGIC,
            GLOBAL_DOMAIN_VERSION,
            int(domain_id),
            int(generation),
            entry.device_ctx,
            entry.local_window_base,
            entry.mapping_size,
        )
    finally:
        payload.release()
        staged.close()


def _handle_ctrl_global_domain_release(cw: ChipWorker, buf: memoryview, store: _L2GlobalDomainStore) -> None:
    staged, payload, payload_size = _open_global_domain_payload(buf)
    try:
        if payload_size < LOCAL_RELEASE_REQUEST.size:
            raise RuntimeError("Global CommDomain release payload is truncated")
        magic, version, domain_id, generation = LOCAL_RELEASE_REQUEST.unpack_from(payload, 0)
        _validate_local_global_header(magic, version, domain_id, generation, operation="release")
        entry = store.domains.get(int(domain_id))
        if entry is not None and entry.generation != generation:
            raise RuntimeError("Global CommDomain release generation mismatch")
        if entry is not None:
            cw._impl.comm_global_domain_release(int(domain_id))
            store.domains.pop(int(domain_id), None)
    finally:
        payload.release()
        staged.close()


def _handle_ctrl_global_domain_copy(
    cw: ChipWorker, buf: memoryview, store: _L2GlobalDomainStore, *, copy_to_device: bool
) -> None:
    staged, payload, payload_size = _open_global_domain_payload(buf)
    try:
        if payload_size < LOCAL_COPY_REQUEST.size:
            raise RuntimeError("Global CommDomain copy payload is truncated")
        magic, version, domain_id, generation, offset, nbytes = LOCAL_COPY_REQUEST.unpack_from(payload, 0)
        operation = "copy-to" if copy_to_device else "copy-from"
        _validate_local_global_header(magic, version, domain_id, generation, operation=operation)
        entry = store.domains.get(int(domain_id))
        if entry is None or entry.generation != generation or entry.device_ctx == 0:
            raise RuntimeError(f"Global CommDomain {operation} requires an imported live domain")
        if nbytes <= 0 or nbytes > GLOBAL_DOMAIN_MAX_COPY_BYTES:
            raise RuntimeError(f"Global CommDomain {operation} size is invalid")
        if offset > entry.mapping_size or nbytes > entry.mapping_size - offset:
            raise RuntimeError(f"Global CommDomain {operation} range exceeds the local window")
        if copy_to_device:
            data_offset = LOCAL_COPY_REQUEST.size
            if data_offset + nbytes > payload_size:
                raise RuntimeError("Global CommDomain copy-to data is truncated")
            exported = ctypes.c_char.from_buffer(payload, data_offset)
            try:
                cw.copy_to(entry.local_window_base + int(offset), ctypes.addressof(exported), int(nbytes))
            finally:
                del exported
        else:
            data_offset = LOCAL_COPY_REPLY.size
            if data_offset + nbytes > payload_size:
                raise RuntimeError("Global CommDomain copy-from reply capacity is too small")
            exported = ctypes.c_char.from_buffer(payload, data_offset)
            try:
                cw.copy_from(ctypes.addressof(exported), entry.local_window_base + int(offset), int(nbytes))
            finally:
                del exported
        LOCAL_COPY_REPLY.pack_into(
            payload,
            0,
            LOCAL_DOMAIN_MAGIC,
            GLOBAL_DOMAIN_VERSION,
            int(domain_id),
            int(generation),
            int(nbytes),
        )
    finally:
        payload.release()
        staged.close()


def _sweep_l2_global_domains(cw: ChipWorker, store: _L2GlobalDomainStore) -> None:
    for domain_id in list(store.domains):
        store.domains.pop(domain_id, None)
        try:
            cw._impl.comm_global_domain_release(int(domain_id))
        except Exception:  # noqa: BLE001
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


def _run_chip_main_loop(  # noqa: PLR0913, PLR0915 -- fork-child entry: every dependency the handlers close over crosses the fork as an explicit arg, and the control handler carries one branch per sub-command
    cw: ChipWorker,
    buf: memoryview,
    mailbox_addr: int,
    state_addr: int,
    device_id: int,
    registry: dict[int, Any],
    identity_table: dict[bytes, int],
    identity_refs: dict[bytes, int],
    owner_instance_id: bytes,
    *,
    chip_platform: str,
    chip_runtime: str = "",
    on_task_done_success=None,
    prepared: set[int] | None = None,
    task_frame_count: int = 1,
) -> None:
    """Chip-process handlers for `_run_mailbox_loop`.

    `on_task_done_success`, if provided, is invoked after a successful chip task
    and before publishing TASK_DONE. It must return ``(code, msg)`` — typically
    ``(0, "")`` on success, or an error tuple if the hook itself failed (e.g.
    D2H staging error). Returning a non-zero code overrides the kernel's
    success.

    Published task frames carry a callable digest. The child resolves it to a
    target-local slot and runs it. The slot must already be prepared: initial
    startup-snapshot ChipCallables are prepared before INIT_READY (carried in
    via ``prepared``), and callables registered dynamically after startup
    arrive via ``_CTRL_PREPARE``. A task frame for an unprepared slot is a
    control-flow error and fails rather than lazily preparing it.

    ``owner_instance_id`` is the parent Worker's nonce — the only owner whose
    DEVICE_MALLOC/VMM_WINDOW backings this chip may materialize.
    """
    prepared = prepared if prepared is not None else set()
    worker_chip_region_store = _HostWorkerChipRegionStore()
    import_registry = ImportRegistry(ImportContext(is_host_endpoint=False, device_owner_instance_id=owner_instance_id))
    global_domain_store = _L2GlobalDomainStore()

    def handle_task(task_buf) -> tuple[int, str]:
        task_addr = ctypes.addressof(ctypes.c_char.from_buffer(task_buf))
        digest = _read_task_digest(task_buf)
        cid = identity_table.get(digest)
        cfg = _read_config_from_mailbox(task_buf)

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
            pipeline_slot, pipeline_reserved, pipeline_generation = _PIPELINE_LEASE_FMT.unpack_from(
                task_buf, _OFF_PIPELINE_LEASE
            )
            if pipeline_reserved != 0:
                raise RuntimeError(f"chip_process dev={device_id}: invalid pipeline lease reserved field")
            # The mailbox bytes decode once, into the wire TaskArgs; the mapping pass and the
            # chip-POD build both read that object. Each tensor's embedded handle resolves to a
            # local base by canonical identity (map-once, cached), and the POD is built at those
            # bases — an exact resolution, not the parent-VA numeric-range rewrite it replaced.
            args_ptr = task_addr + _OFF_TASK_ARGS_BLOB
            args = read_args_from_blob(args_ptr, _MAILBOX_ARGS_CAPACITY)
            resolved = import_registry.materialize_args(args)
            chip_args = materialize_task_args(args, resolved)
            # The acceptance flag lives in the mailbox, not in the materialized args, so
            # the fence still publishes through the address the parent polls.
            cw._impl.run_materialized(
                cid,
                chip_args,
                cfg,
                task_addr + _OFF_ACCEPTED,
                _TASK_ACCEPTED,
                pipeline_slot,
                pipeline_generation,
            )
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
        return code, msg

    def handle_control(  # noqa: PLR0912, PLR0915 -- one branch per control sub-command
        sub_cmd: int,
    ) -> tuple[int, str]:
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
            elif sub_cmd in (_CTRL_COPY_TO, _CTRL_COPY_FROM):
                # Both ends resolve through the same map-once cache the task-args path uses, so a
                # backing already imported for a task is the one this copy reaches. The offset is
                # applied here rather than sent as an address: the parent's mapping of either
                # backing means nothing in this process, so only a base resolved locally can be
                # advanced. `_read_control_copy_request` has already bounded both spans.
                dst_desc, src_desc, n, dst_off, src_off = _read_control_copy_request(mailbox_addr)
                dst = _copy_span_base(import_registry.materialize(dst_desc).base, dst_off)
                src = _copy_span_base(import_registry.materialize(src_desc).base, src_off)
                if sub_cmd == _CTRL_COPY_TO:
                    cw.copy_to(dst, src, n)
                else:
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
                        cid = _install_local_identity(registry, identity_table, identity_refs, digest, callable_obj)
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
                cid = identity_table.get(digest)
                if cid is not None:
                    refs = identity_refs.get(digest, 1)
                    if refs > 1:
                        identity_refs[digest] = refs - 1
                    else:
                        # Mutate the resolver only after native unregister
                        # succeeds. A device error must not leave the digest
                        # absent locally while its prepared slot remains live.
                        cw._unregister_slot(int(cid))
                        identity_refs.pop(digest, None)
                        identity_table.pop(digest, None)
                        registry.pop(int(cid), None)
                        prepared.discard(int(cid))
            elif sub_cmd == _CTRL_ALLOC_DOMAIN:
                _handle_ctrl_alloc_domain(cw, buf)
            elif sub_cmd == _CTRL_RELEASE_DOMAIN:
                _handle_ctrl_release_domain(cw, buf)
            elif sub_cmd == _CTRL_COMM_INIT:
                _handle_ctrl_comm_init(cw, buf)
            elif sub_cmd == _CTRL_WORKER_CHIP_REGION_CREATE:
                _handle_ctrl_worker_chip_region_create(cw, buf, chip_platform, worker_chip_region_store)
            elif sub_cmd == _CTRL_WORKER_CHIP_REGION_RELEASE:
                _handle_ctrl_worker_chip_region_release(buf, worker_chip_region_store)
            elif sub_cmd == _CTRL_COMMITTED_DEVICE_MEMORY:
                struct.pack_into("Q", buf, _CTRL_OFF_RESULT, cw.committed_device_memory)
            elif sub_cmd == _CTRL_IMPORT_RELEASE:
                import_registry.unregister(_unpack_identity_wire(_read_control_digest(buf)))
            elif sub_cmd == CTRL_GLOBAL_DOMAIN_PREPARE:
                _handle_ctrl_global_domain_prepare(cw, buf, global_domain_store)
            elif sub_cmd == CTRL_GLOBAL_DOMAIN_IMPORT:
                _handle_ctrl_global_domain_import(cw, buf, global_domain_store)
            elif sub_cmd == CTRL_GLOBAL_DOMAIN_RELEASE:
                _handle_ctrl_global_domain_release(cw, buf, global_domain_store)
            elif sub_cmd == CTRL_GLOBAL_DOMAIN_COPY_TO:
                _handle_ctrl_global_domain_copy(
                    cw,
                    buf,
                    global_domain_store,
                    copy_to_device=True,
                )
            elif sub_cmd == CTRL_GLOBAL_DOMAIN_COPY_FROM:
                _handle_ctrl_global_domain_copy(
                    cw,
                    buf,
                    global_domain_store,
                    copy_to_device=False,
                )
            else:
                raise RuntimeError(f"unknown control sub-command {int(sub_cmd)}")
        except Exception as e:  # noqa: BLE001
            code = 1
            if sub_cmd in (_CTRL_REGISTER, _CTRL_UNREGISTER):
                op = "register" if sub_cmd == _CTRL_REGISTER else "unregister"
                msg = _format_exc(f"{op} hash={_format_digest(_read_control_digest(buf))} chip={device_id}", e)
            else:
                msg = _format_exc(f"chip_process dev={device_id} ctrl={int(sub_cmd)}", e)
        return code, msg

    def run_two_frame_loop() -> None:  # noqa: PLR0912, PLR0915 -- one progress owner drives control and both task frames
        frame_bufs = [
            buf[(1 + index) * MAILBOX_FRAME_SIZE : (2 + index) * MAILBOX_FRAME_SIZE]
            for index in range(_TASK_FRAME_COUNT)
        ]
        frame_addrs = [mailbox_addr + (1 + index) * MAILBOX_FRAME_SIZE for index in range(_TASK_FRAME_COUNT)]

        @dataclass
        class _StagedFrame:
            index: int
            frame_buf: memoryview
            frame_addr: int
            identity: tuple[int, int, int, int, int]
            cid: int
            config: CallConfig
            activated: bool
            chip_run: Any = None
            launched_published: bool = False

        staged_frames: dict[int, _StagedFrame] = {}

        def read_identity(frame_buf: memoryview) -> tuple[int, int, int, int, int]:
            return (
                struct.unpack_from("=Q", frame_buf, _OFF_FRAME_PROTOCOL)[0],
                struct.unpack_from("=Q", frame_buf, _OFF_FRAME_RUN_ID)[0],
                struct.unpack_from("=Q", frame_buf, _OFF_FRAME_SLOT_ID)[0],
                struct.unpack_from("=Q", frame_buf, _OFF_FRAME_GENERATION)[0],
                struct.unpack_from("=Q", frame_buf, _OFF_FRAME_DISPATCH_ID)[0],
            )

        def task_frame_references_digest(digest: bytes) -> bool:
            live_states = (_TASK_READY, _PREPARE_READY, _ACTIVATE, _FRAME_STAGED, _TASK_LAUNCHED)
            for index, frame_buf in enumerate(frame_bufs):
                if _mailbox_load_i32(frame_addrs[index] + _OFF_STATE) not in live_states:
                    continue
                if _read_task_digest(frame_buf) == digest:
                    return True
            return False

        def fail_frame(frame: _StagedFrame, message: str) -> None:
            _write_error(frame.frame_buf, 1, message)
            _mailbox_store_i32(frame.frame_addr + _OFF_STATE, _TASK_FAILED)

        def stage_frame(index: int, initial_state: int) -> _StagedFrame | None:
            frame_buf = frame_bufs[index]
            frame_addr = frame_addrs[index]
            try:
                identity = read_identity(frame_buf)
                protocol, run_id, slot_id, generation, dispatch_id = identity
                pipeline_slot, pipeline_reserved, pipeline_generation = _PIPELINE_LEASE_FMT.unpack_from(
                    frame_buf, _OFF_PIPELINE_LEASE
                )
                if protocol != _TASK_PROTOCOL_VERSION:
                    raise RuntimeError(f"unsupported task frame protocol {protocol}")
                if run_id == 0 or generation == 0 or dispatch_id == 0:
                    raise RuntimeError(f"invalid task frame identity {identity}")
                if slot_id != index or pipeline_slot != index:
                    raise RuntimeError(
                        f"task frame {index} does not own pipeline slot (identity={slot_id}, lease={pipeline_slot})"
                    )
                if pipeline_reserved != 0 or pipeline_generation != generation:
                    raise RuntimeError(
                        "task frame pipeline lease does not match its identity "
                        f"(reserved={pipeline_reserved}, lease_generation={pipeline_generation}, "
                        f"identity_generation={generation})"
                    )

                digest = _read_task_digest(frame_buf)
                cid = identity_table.get(digest)
                if cid is None:
                    raise RuntimeError(f"callable hash {_format_digest(digest)} not registered")
                if cid not in prepared:
                    raise RuntimeError(
                        f"cid {cid} not prepared before task frame publication (register via _CTRL_PREPARE first)"
                    )
                return _StagedFrame(
                    index=index,
                    frame_buf=frame_buf,
                    frame_addr=frame_addr,
                    identity=identity,
                    cid=int(cid),
                    config=_read_config_from_mailbox(frame_buf),
                    activated=initial_state in (_TASK_READY, _ACTIVATE),
                )
            except Exception as e:  # noqa: BLE001
                _write_error(frame_buf, 1, _format_exc(f"chip_process dev={device_id} frame={index}", e))
                _mailbox_store_i32(frame_addr + _OFF_STATE, _TASK_FAILED)
                return None

        def submit_frame(frame: _StagedFrame) -> None:
            _protocol, run_id, slot_id, generation, dispatch_id = frame.identity
            # The frame carries the wire blob; the runtime reads the chip POD. The bytes decode
            # once into the wire TaskArgs, whose tensors resolve to local bases (map-once, cached
            # by canonical identity) and rebuild at those bases, as the non-pipelined task path
            # does. The lane copies the args into its own storage, so this POD need not outlive
            # the call.
            args_ptr = frame.frame_addr + _OFF_TASK_ARGS_BLOB
            args = read_args_from_blob(args_ptr, _MAILBOX_ARGS_CAPACITY)
            resolved = import_registry.materialize_args(args)
            chip_args = materialize_task_args(args, resolved)
            frame.chip_run = cw._impl._submit_chip_run_materialized(
                frame.cid,
                chip_args,
                frame.config,
                slot_id,
                generation,
                run_id,
                dispatch_id,
                frame.frame_addr + _OFF_ACCEPTED,
                _TASK_ACCEPTED,
                False,
            )
            raw_disposition = frame.chip_run.preparation_disposition
            disposition = int(getattr(raw_disposition, "value", raw_disposition))
            if disposition not in (_VALIDATED_ONLY, _NATIVE_PREPARED):
                raise RuntimeError(f"chip run lane returned invalid preparation disposition {disposition}")
            _mailbox_store_i32(frame.frame_addr + _OFF_PREPARATION_DISPOSITION, disposition)
            _write_error(frame.frame_buf, 0, "")
            _mailbox_store_i32(frame.frame_addr + _OFF_STATE, _FRAME_STAGED)
            if frame.activated:
                frame.chip_run.activate()

        parent_pid = os.getppid()
        liveness_countdown = _PARENT_LIVENESS_POLL_INTERVAL
        shutdown_message = f"chip_process dev={device_id}: task loop shut down"
        shutdown_addr = _buffer_field_addr(buf, _OFF_SHUTDOWN)
        try:
            while True:
                control_state = _mailbox_load_i32(state_addr)
                if control_state == _SHUTDOWN or _mailbox_load_i32(shutdown_addr) == _SHUTDOWN_REQUESTED:
                    break
                if control_state == _CONTROL_REQUEST:
                    sub_cmd = struct.unpack_from("Q", buf, _OFF_CALLABLE)[0]
                    registry_control = sub_cmd in (_CTRL_PREPARE, _CTRL_REGISTER, _CTRL_UNREGISTER)
                    defer_control = registry_control and bool(staged_frames)
                    if sub_cmd == _CTRL_UNREGISTER:
                        defer_control = defer_control or task_frame_references_digest(_read_control_digest(buf))
                    if not defer_control:
                        code, msg = handle_control(int(sub_cmd))
                        _write_error(buf, code, msg)
                        _mailbox_store_i32(state_addr, _CONTROL_DONE)

                new_frames: list[_StagedFrame] = []
                for index in range(_TASK_FRAME_COUNT):
                    frame_state = _mailbox_load_i32(frame_addrs[index] + _OFF_STATE)
                    staged = staged_frames.get(index)
                    if staged is None:
                        if frame_state in (_TASK_READY, _PREPARE_READY, _ACTIVATE):
                            staged = stage_frame(index, frame_state)
                            if staged is not None:
                                new_frames.append(staged)
                        continue
                    if frame_state == _ACTIVATE and not staged.activated:
                        if read_identity(staged.frame_buf) != staged.identity:
                            stale_message = f"chip_process dev={device_id}: stale activation identity"
                            try:
                                staged.chip_run.abandon()
                            except Exception as e:  # noqa: BLE001
                                stale_message += "; " + _format_exc("chip run abandonment", e)
                                shutdown_message = stale_message
                                break
                            fail_frame(staged, stale_message)
                            staged_frames.pop(index, None)
                            continue
                        try:
                            staged.chip_run.activate()
                            staged.activated = True
                        except Exception as e:  # noqa: BLE001
                            shutdown_message = _format_exc(f"chip_process dev={device_id}: native activation", e)
                            break
                else:
                    for staged in sorted(new_frames, key=lambda frame: frame.identity[4]):
                        try:
                            submit_frame(staged)
                        except Exception as e:  # noqa: BLE001
                            fail_frame(staged, _format_exc(f"chip_process dev={device_id}: chip run submit", e))
                        else:
                            staged_frames[staged.index] = staged

                    stop_progress = False
                    for staged in sorted(staged_frames.values(), key=lambda frame: frame.identity[4]):
                        try:
                            if staged.chip_run.launched and not staged.launched_published:
                                _mailbox_store_i32(staged.frame_addr + _OFF_STATE, _TASK_LAUNCHED)
                                staged.launched_published = True
                                continue
                            run_complete = bool(staged.chip_run.done())
                            if not run_complete and staged.chip_run.launched and not staged.launched_published:
                                _mailbox_store_i32(staged.frame_addr + _OFF_STATE, _TASK_LAUNCHED)
                                staged.launched_published = True
                            if not run_complete:
                                continue

                            code = 0
                            msg = ""
                            try:
                                staged.chip_run._raise_if_failed()
                            except Exception as e:  # noqa: BLE001
                                code = 1
                                msg = _format_exc(f"chip_process dev={device_id}: chip run", e)
                            if code == 0 and on_task_done_success is not None:
                                try:
                                    code, msg = on_task_done_success()
                                except Exception as e:  # noqa: BLE001
                                    code = 1
                                    msg = _format_exc(f"chip_process dev={device_id}: task completion hook", e)
                            _write_error(staged.frame_buf, code, msg)
                            _mailbox_store_i32(
                                staged.frame_addr + _OFF_STATE,
                                _TASK_DONE if code == 0 else _TASK_FAILED,
                            )
                            staged_frames.pop(staged.index, None)
                            if code != 0 and staged.chip_run.lane_poisoned:
                                shutdown_message = msg
                                stop_progress = True
                                break
                        except Exception as e:  # noqa: BLE001
                            shutdown_message = _format_exc(f"chip_process dev={device_id}: chip run progress", e)
                            stop_progress = True
                            break
                    if not stop_progress:
                        liveness_countdown -= 1
                        if liveness_countdown <= 0:
                            liveness_countdown = _PARENT_LIVENESS_POLL_INTERVAL
                            if os.getppid() != parent_pid:
                                shutdown_message = f"chip_process dev={device_id}: parent exited"
                                break
                        continue
                break
        finally:
            try:
                cw._impl._close_chip_run_lane()
            except Exception as e:  # noqa: BLE001
                shutdown_message += "; " + _format_exc("chip run lane close", e)
            for staged in staged_frames.values():
                fail_frame(staged, shutdown_message)
            for index, frame_buf in enumerate(frame_bufs):
                frame_state_addr = frame_addrs[index] + _OFF_STATE
                if _mailbox_load_i32(frame_state_addr) in (
                    _TASK_READY,
                    _PREPARE_READY,
                    _ACTIVATE,
                    _FRAME_STAGED,
                    _TASK_LAUNCHED,
                ):
                    _write_error(frame_buf, 1, shutdown_message)
                    _mailbox_store_i32(frame_state_addr, _TASK_FAILED)
            for frame_buf in frame_bufs:
                frame_buf.release()

    try:
        if task_frame_count >= 2:
            run_two_frame_loop()
        else:
            _run_mailbox_loop(buf, state_addr, handle_task=handle_task, handle_control=handle_control)
    finally:
        import_registry.close()
        _sweep_l2_global_domains(cw, global_domain_store)
        _sweep_host_worker_chip_regions(worker_chip_region_store)


def _chip_process_loop(  # noqa: PLR0913 -- fork-child entry: all context (bins, identity tables, log config, prewarm sizing) must cross the fork as explicit COW args; the child cannot read parent state after os.fork
    buf: memoryview,
    bins,
    device_id: int,
    registry: dict[int, Any],
    identity_table: dict[bytes, int],
    identity_refs: dict[bytes, int],
    owner_instance_id: bytes,
    log_level: int = 25,
    platform: str = "",
    runtime: str = "",
    prewarm_config=None,
    enable_sdma: bool = False,
) -> None:
    """Runs in forked child process. Loads host_runtime.so in own address space.

    `log_level` is the parent's snapshot of the simpler logger (computed via
    `_log.get_current_config()`); the child cannot read the parent's logger
    after fork, so the value is passed explicitly.

    The main loop is delegated to ``_run_chip_main_loop`` — see its docstring
    for the TASK_READY / CONTROL_REQUEST / SHUTDOWN state machine.
    """
    import traceback as _tb  # noqa: PLC0415

    try:
        cw = ChipWorker()
        cw.init(
            device_id,
            bins,
            log_level=log_level,
            prewarm_config=prewarm_config,
            enable_sdma=enable_sdma,
        )
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
    # Before the first task, the lease word is startup metadata: slot_id carries
    # the backend's supported admission depth. Dispatches later overwrite the
    # same fixed wire region with the run-owned slot/generation lease.
    _PIPELINE_LEASE_FMT.pack_into(buf, _OFF_PIPELINE_LEASE, int(cw.pipeline_depth), 0, 0)
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
            owner_instance_id,
            chip_platform=platform,
            chip_runtime=runtime,
            prepared=prepared,
            task_frame_count=_local_task_frame_count(platform, runtime, int(cw.pipeline_depth)),
        )
    finally:
        cw.finalize()


def _read_config_from_mailbox(buf: memoryview) -> CallConfig:
    """Reconstruct a CallConfig from the unified mailbox layout."""
    (
        aicpu_tn,
        swl,
        dt,
        pmu,
        dep_gen,
        scope_stats,
        *ring_values,
        prefix_bytes,
    ) = _CFG_FMT.unpack_from(buf, _OFF_CONFIG)
    ring_task_window = list(ring_values[:RUNTIME_ENV_RING_COUNT])
    ring_heap = list(ring_values[RUNTIME_ENV_RING_COUNT : 2 * RUNTIME_ENV_RING_COUNT])
    ring_dep_pool = list(ring_values[2 * RUNTIME_ENV_RING_COUNT : 3 * RUNTIME_ENV_RING_COUNT])
    cfg = CallConfig()
    cfg.aicpu_thread_num = aicpu_tn
    cfg.enable_chip_swimlane = swl
    cfg.enable_dump_args = int(dt)
    cfg.enable_pmu = pmu
    cfg.enable_dep_gen = bool(dep_gen)
    cfg.enable_scope_stats = bool(scope_stats)
    cfg.runtime_env.ring_task_window = ring_task_window
    cfg.runtime_env.ring_heap = ring_heap
    cfg.runtime_env.ring_dep_pool = ring_dep_pool
    # NUL-terminated C string in a 1024-byte field.
    cfg.output_prefix = prefix_bytes.split(b"\x00", 1)[0].decode("utf-8")
    return cfg


def _run_local_global_domain_control(  # noqa: PLR0912 -- one ordered dispatcher for the Global CommDomain protocol
    inner_worker: Worker,
    runtime: _GlobalNodeRuntime,
    comm_inits: dict[str, GlobalCommInitCommand],
    control_name: int,
    request: bytes,
) -> bytes:
    """Execute one Global CommDomain command inside an add_worker L3 child."""
    from .remote_l3_protocol import ControlName  # noqa: PLC0415

    control = ControlName(control_name)
    if control is ControlName.COMM_INIT:
        command = decode_comm_init(request)
        if command.cluster_id != runtime.cluster_id:
            raise ValueError("COMM_INIT cluster_id does not match the local L3 topology")
        if command.profile != runtime.comm_profile:
            raise ValueError("COMM_INIT profile does not match the local L3 topology")
        if command.node_rank != runtime.node_rank or command.node_count != runtime.node_count:
            raise ValueError("COMM_INIT node identity does not match the local L3 topology")
        local_members = tuple(member for member in command.members if member.node_worker_id == runtime.worker_id)
        if not local_members:
            raise ValueError("COMM_INIT topology has no local members")
        for member in local_members:
            if member.local_worker_id < 0 or member.local_worker_id >= len(runtime.global_device_ranks):
                raise ValueError("COMM_INIT local worker id exceeds the local L3 device list")
            if member.global_device_rank != runtime.global_device_ranks[member.local_worker_id]:
                raise ValueError("COMM_INIT global device rank does not match the local L3 topology")
        prior = comm_inits.get(command.topology_hash)
        if prior is not None and prior != command:
            raise ValueError("COMM_INIT topology hash conflicts with an earlier command")
        capability = resolve_global_comm_capability(
            platform=runtime.platform,
            profile=runtime.comm_profile,
            local_device_count=len(runtime.global_device_ranks),
        )
        comm_inits[command.topology_hash] = command
        return encode_comm_init_result(capability)

    if control is ControlName.ALLOC_DOMAIN:
        command = decode_domain_command(request)
        if not any(init.profile == command.profile and init.members == command.members for init in comm_inits.values()):
            raise RuntimeError("ALLOC_DOMAIN requires a matching COMM_INIT topology")
        if command.phase is GlobalDomainPhase.PREPARE_EXPORT:
            if command.descriptors:
                raise ValueError("PREPARE_EXPORT must not carry descriptors")
            return encode_descriptor_table(inner_worker._prepare_global_domain_node(command, runtime.worker_id))
        if command.phase is GlobalDomainPhase.IMPORT:
            inner_worker._import_global_domain_node(command, runtime.worker_id)
            return b""
        if command.phase is GlobalDomainPhase.COMMIT:
            inner_worker._commit_global_domain_node(command)
            return b""
        if command.phase is GlobalDomainPhase.ABORT:
            inner_worker._release_global_domain_node(
                GlobalDomainReleaseCommand(command.domain_id, command.generation),
                suppress_errors=True,
            )
            return b""
        raise ValueError("ALLOC_DOMAIN phase is not supported")

    if control is ControlName.RELEASE_DOMAIN:
        inner_worker._release_global_domain_node(decode_release_command(request))
        return b""
    if control is ControlName.COPY_TO_DOMAIN:
        inner_worker._copy_global_domain_node(
            decode_copy_command(request, include_data=True),
            copy_to_device=True,
        )
        return b""
    if control is ControlName.COPY_FROM_DOMAIN:
        result = inner_worker._copy_global_domain_node(
            decode_copy_command(request, include_data=False),
            copy_to_device=False,
        )
        return encode_copy_result(result)
    raise ValueError(f"unsupported local Global CommDomain control {int(control)}")


def _child_worker_loop(
    buf: memoryview,
    registry: dict[int, Any],
    identity_table: dict[bytes, int],
    identity_refs: dict[bytes, int],
    inner_worker: Worker,
    global_node: _GlobalNodeRuntime | None = None,
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
    global_comm_inits: dict[str, GlobalCommInitCommand] = {}

    def handle_task(task_buf) -> tuple[int, str]:
        digest = _read_task_digest(task_buf)
        cid = identity_table.get(digest)
        orch_fn = registry.get(int(cid)) if cid is not None else None
        if orch_fn is None:
            return 1, f"child_worker: callable hash {_format_digest(digest)} not registered"
        try:
            # Orchestrator (not a compute leaf): re-export each received backing to a local
            # handle H' (per-backing, no map) so the inner orch sees only its own handles;
            # pure forwarding to L2 carries no map cost.
            args = _reexport_args_from_mailbox(task_buf, inner_worker)
            cfg = _read_config_from_mailbox(task_buf)
            inner_worker.run(orch_fn, args, cfg)
        except Exception as e:  # noqa: BLE001
            return 1, _format_exc(f"child_worker level={inner_worker.level}", e)
        return 0, ""

    def handle_control(sub_cmd: int) -> tuple[int, str]:
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
            elif sub_cmd == _CTRL_IMPORT_RELEASE:
                inner_worker._release_import_recursive(_unpack_identity_wire(_read_control_digest(buf)))
            elif sub_cmd in (_CTRL_PY_REGISTER, _CTRL_PY_IMPORT_REGISTER, _CTRL_PY_UNREGISTER):
                _handle_py_callable_control(
                    buf,
                    registry,
                    identity_table,
                    identity_refs,
                    sub_cmd,
                    context=f"child_worker level={inner_worker.level}",
                )
            elif sub_cmd == _CTRL_GLOBAL_DOMAIN_NODE:
                if global_node is None:
                    raise RuntimeError("Global CommDomain control requires a local L3 child")
                staged, payload, payload_size = _open_global_domain_payload(buf)
                try:
                    if payload_size < _LOCAL_GLOBAL_CONTROL_HEADER.size:
                        raise RuntimeError("local Global CommDomain control payload is truncated")
                    control_name, request_size, response_size = _LOCAL_GLOBAL_CONTROL_HEADER.unpack_from(payload, 0)
                    capacity = payload_size - _LOCAL_GLOBAL_CONTROL_HEADER.size
                    if request_size > capacity:
                        raise RuntimeError("local Global CommDomain request exceeds staged payload")
                    if response_size != 0:
                        raise RuntimeError("local Global CommDomain request contains a response")
                    start = _LOCAL_GLOBAL_CONTROL_HEADER.size
                    request = bytes(payload[start : start + request_size])
                    response = _run_local_global_domain_control(
                        inner_worker,
                        global_node,
                        global_comm_inits,
                        int(control_name),
                        request,
                    )
                    if len(response) > capacity:
                        raise RuntimeError("local Global CommDomain response exceeds staged payload")
                    payload[start : start + len(response)] = response
                    _LOCAL_GLOBAL_CONTROL_HEADER.pack_into(
                        payload,
                        0,
                        int(control_name),
                        int(request_size),
                        len(response),
                    )
                finally:
                    payload.release()
                    staged.close()
            else:
                raise RuntimeError(f"unknown control sub-command {sub_cmd}")
        except Exception as e:  # noqa: BLE001
            op = _CTRL_OP_NAMES.get(sub_cmd, f"ctrl={sub_cmd}")
            return 1, _format_exc(f"child_worker level={inner_worker.level} {op}", e)
        return 0, ""

    _run_mailbox_loop(
        buf,
        state_addr,
        handle_task=handle_task,
        handle_control=handle_control,
        on_shutdown=inner_worker.close,
    )


def _journal_child_survivors(journal, sub_shms, sub_pids, chip_shms, chip_pids, next_shms, next_pids, reaped):
    """Register unreaped child processes and their paired shms in the cleanup
    journal so a subsequent close() can retry."""
    for shms, pids, kind in (
        (sub_shms, sub_pids, "sub"),
        (chip_shms, chip_pids, "chip"),
        (next_shms, next_pids, "next"),
    ):
        for i in range(min(len(shms), len(pids))):
            pid = pids[i]
            if pid in reaped:
                continue
            shm = shms[i]

            def _make_cleanup(_shm=shm, _pid=pid, _kind=kind):
                def _cleanup_child():
                    try:
                        wpid, _status = os.waitpid(_pid, os.WNOHANG)
                    except ChildProcessError:
                        wpid = _pid
                    except OSError:
                        wpid = 0
                    if wpid == 0:
                        raise RuntimeError(f"child {_kind} pid {_pid} still alive; shm not freed")
                    cleanup_error = None
                    try:
                        _shm.close()
                    except BaseException as exc:  # noqa: BLE001
                        cleanup_error = exc
                    try:
                        _shm.unlink()
                    except FileNotFoundError:
                        pass
                    except BaseException as exc:  # noqa: BLE001
                        cleanup_error = cleanup_error or exc
                    if cleanup_error is not None:
                        raise cleanup_error

                return _cleanup_child

            journal.add("child", f"{kind} pid {pid} (shm unreclaimed)", _make_cleanup())

        for orphan_index, shm in enumerate(shms[len(pids) :], start=len(pids)):

            def _cleanup_orphan_shm(_shm=shm):
                try:
                    _shm.close()
                finally:
                    with contextlib.suppress(FileNotFoundError):
                        _shm.unlink()

            journal.add("shm", f"{kind} mailbox {orphan_index} without live child", _cleanup_orphan_shm)


class _Lifecycle(enum.Enum):
    """The single authoritative *public-admission* lifecycle of a Worker (5
    states), guarded by ``_hierarchical_start_cv``.

    ``NEW → INITIALIZING → READY | FAILED → CLOSED``. Every level uses this
    machine: an L2 worker inits synchronously (no child barrier) but still claims
    INITIALIZING so two concurrent ``init()`` calls serialize on the same epoch.
    close() while INITIALIZING cooperatively cancels the in-progress init and
    ultimately reaches CLOSED.

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


@dataclass(frozen=True)
class _CloseOutcome:
    """One immutable result published by a close attempt."""

    error: BaseException | None
    incomplete: bool


class CleanupJournal:
    """Post-success resource cleanup journal shared by _teardown_ready_tree
    and _abort_hierarchical. Entries removed only after native free succeeds."""

    def __init__(self):
        self._entries = []
        self._errors = []

    def add(self, kind, identity, cleanup_fn):
        self._entries.append((kind, identity, cleanup_fn))

    def add_once(self, kind, identity, cleanup_fn):
        key = (kind, identity)
        if any((entry_kind, entry_identity) == key for entry_kind, entry_identity, _fn in self._entries):
            return
        self.add(kind, identity, cleanup_fn)

    def extend(self, entries):
        self._entries.extend(entries)

    @property
    def empty(self):
        return len(self._entries) == 0

    def __len__(self):
        return len(self._entries)

    @property
    def errors(self):
        return list(self._errors)

    def drive(self, only: set[tuple[str, str]] | None = None):
        errors = []
        remaining = []
        for kind, identity, cleanup_fn in self._entries:
            if only is not None and (kind, identity) not in only:
                remaining.append((kind, identity, cleanup_fn))
                continue
            try:
                cleanup_fn()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                remaining.append((kind, identity, cleanup_fn))
        self._entries = remaining
        if errors:
            self._errors.extend(errors)
        return errors[0] if errors else None

    def drive_kinds(self, kinds: set[str]):
        """Drive every retained entry whose resource kind is in ``kinds``."""
        return self.drive({(kind, identity) for kind, identity, _cleanup in self._entries if kind in kinds})


class _CloseAttempt:
    """Private completion record for one close() teardown attempt.

    close() publishes CLOSED atomically and installs a fresh attempt; concurrent
    close()s pin to the attempt they observed (via ``_close_completion``) and
    wait on its ``done``, so every joiner of the same attempt sees the same
    outcome. ``incomplete=True`` means the tree was not fully reclaimed.

    CLOSED is terminal for public admission, but teardown debt is retryable. A
    later close() re-drives entries left in ``CleanupJournal``; a drain timeout
    likewise leaves the native tree intact for a later attempt.
    """

    __slots__ = ("_outcome",)

    def __init__(self) -> None:
        self._outcome: _CloseOutcome | None = None

    @property
    def done(self) -> bool:
        return self._outcome is not None

    @property
    def error(self) -> BaseException | None:
        outcome = self._outcome
        return None if outcome is None else outcome.error

    @property
    def incomplete(self) -> bool:
        outcome = self._outcome
        return False if outcome is None else outcome.incomplete

    def publish(self, error: BaseException | None, incomplete: bool) -> _CloseOutcome:
        """Install the attempt's only outcome despite pre-commit interruptions."""
        effective_error = error
        effective_incomplete = incomplete
        try:
            while True:
                try:
                    committed = self._outcome
                    if committed is not None:
                        return committed
                    candidate = _CloseOutcome(effective_error, effective_incomplete)
                    self._outcome = candidate
                    return candidate
                except BaseException as exc:  # noqa: BLE001, PERF203
                    # An interruption after STORE_ATTR observes the immutable
                    # committed result and remains local to this publisher. An
                    # interruption before STORE_ATTR becomes the shared result.
                    if self._outcome is not None:
                        raise
                    if effective_error is None:
                        effective_error = exc
                    effective_incomplete = True
        except BaseException as exc:  # noqa: BLE001
            if self._outcome is not None:
                raise
            if effective_error is None:
                effective_error = exc
            return self.publish(effective_error, True)


class _StartupCancelled(BaseException):
    """Raised inside a forked child when the parent cooperatively cancels its
    startup (SIGTERM). Unwinds the child's own ``setup`` — recursively rolling
    back any grandchildren it already forked — before it exits.

    Derives from ``BaseException`` because it is delivered from a signal
    handler and must not be absorbed by the child's own ``except Exception``
    handlers. Root-thread cancellation uses ``InitCancelled`` instead."""


class InitCancelled(RuntimeError):
    """Raised from ``Worker.init()`` when a concurrent ``close()`` cancelled the
    startup epoch before it committed READY.

    An ordinary ``RuntimeError`` — the init-owner thread is a normal caller, so
    the cancellation must be catchable by ``except Exception``. Distinct from
    ``_StartupCancelled``, which is signal-delivered inside a forked child."""


@dataclass(eq=False)
class _RemoteSlotRefClaim:
    """One replayable remote-buffer reference owned by a run cleanup journal."""

    handle: RemoteBufferHandle
    token: object = field(default_factory=object)


@dataclass
class _RunResources:
    """Python resources whose lifetime ends at one native run fence."""

    remote_slot_refs: list[_RemoteSlotRefClaim | RemoteBufferHandle] = field(default_factory=list)
    live_domains: dict[str, CommDomainHandle] = field(default_factory=dict)
    pending_release_domains: list[CommDomainHandle] = field(default_factory=list)
    live_global_domains: dict[str, GlobalCommDomainHandle] = field(default_factory=dict)
    pending_release_global_domains: list[GlobalCommDomainHandle] = field(default_factory=list)
    worker_chip_regions: list[Any] = field(default_factory=list)
    worker_chip_orch_comm_host_buffers: dict[int, int] = field(default_factory=dict)
    # Every Buffer identity a NEXT_LEVEL dispatch (submit_next_level / _group) sent as a Tensor
    # arg during this run. release_buffer() checks this set across every not-yet-settled run before
    # releasing, so a Buffer never goes away while a dispatched task still names it.
    touched_identities: set[CanonicalIdentity] = field(default_factory=set)
    # True once the owning run's fence has claimed the domains above. A release
    # that arrives after this has no fence left to run behind and frees inline.
    # Read and written only under `domain_lock`.
    retired: bool = False
    # Serializes one domain's release transition against the fence that retires
    # this run. Without it a release can read `retired` as False, be preempted,
    # and append to a queue the fence has already drained for the last time —
    # unreachable from the fence and from close(), which the release itself
    # made blind by dropping the handle from `_live_domains`.
    domain_lock: threading.Lock = field(default_factory=threading.Lock)
    # Sticky: this run's cleanup itself touches the device, so a successor may
    # not be admitted until that cleanup has finished.
    #
    # The whole-run FIFO orders *tasks*. It cannot order cleanup, which runs
    # after the native fence and reaches a child through mailbox control rather
    # than a TaskSlot: N+1 allocating a domain while N is still releasing one
    # can leave two collectives each holding a different chip's mailbox and
    # waiting for the other. Runs that only dispatch tasks keep the full
    # admission depth; a run that acquires any of these degrades itself to
    # depth one. Set, never cleared.
    requires_ordered_cleanup: bool = False


@dataclass
class _PendingRemoteImportReleaseState:
    """Durable local phases after a deferred import-release RPC starts."""

    owner_ref: RemoteBufferHandle | None
    owner_ref_token: object | None = None
    rpc_complete: bool = False
    owner_release_complete: bool = False
    error: BaseException | None = None


@dataclass
class _RunFinalizationCursor:
    """Own the boundary between each one-shot run cleanup operation."""

    steps: tuple[tuple[str, Any], ...]
    next_step: int = 0
    cleanup_error: BaseException | None = None
    boundary_error: BaseException | None = None
    incomplete: bool = False

    @property
    def exhausted(self) -> bool:
        return self.next_step == len(self.steps)

    def _remember_cleanup_error(self, exc: BaseException) -> None:
        if self.cleanup_error is None:
            self.cleanup_error = exc

    def remember_boundary_error(self, exc: BaseException) -> None:
        if self.boundary_error is None:
            self.boundary_error = exc

    def _advance(self, after_step) -> None:
        index = self.next_step
        name, operation = self.steps[index]
        try:
            operation()
        except Exception as exc:  # noqa: BLE001
            self._remember_cleanup_error(exc)
        self.next_step = index + 1
        if after_step is not None:
            after_step(name)

    def drain(self, after_step) -> None:
        while not self.exhausted:
            starting_step = self.next_step
            try:
                self._advance(after_step)
            except BaseException as exc:  # noqa: BLE001
                self.remember_boundary_error(exc)
                if self.next_step == starting_step:
                    # An uncommitted operation has ambiguous ownership and is
                    # not replayable: its native side effect may have landed.
                    self.incomplete = True
                    return


class _AbandonedRunKeepaliveCursor:
    """Drain retained references only after the native tree is gone."""

    def __init__(self, handles: list[RunHandle]) -> None:
        self._handles = handles
        self.first_error: BaseException | None = None

    def drain(self, pending_error: BaseException | None = None) -> None:
        if pending_error is not None and self.first_error is None:
            self.first_error = pending_error
        # These references become safe to drop only after native teardown and
        # no later close attempt re-enters this terminal phase. Retry on a
        # constant stack so repeated interruptions neither leak stack frames
        # nor report cleanup complete early.
        while True:
            try:
                if not self._handles:
                    return
                handle = self._handles[-1]
                handle._keepalive = None
                self._handles.pop()
            except BaseException as exc:  # noqa: BLE001
                if self.first_error is None:
                    self.first_error = exc


class RunHandle:
    """Completion handle returned by :meth:`Worker.submit`.

    A handle owns the run's Python keepalives and keeps its Worker alive until
    the native completion fence has fired and run-owned resources are cleaned
    up. Waiting is idempotent; every waiter observes the same terminal result.
    """

    def __init__(
        self,
        worker: Worker,
        run_id: int,
        keepalive: tuple[Any, ...],
        resources: _RunResources | None = None,
    ) -> None:
        self._worker = worker
        self._run_id: int | None = run_id
        self._keepalive: tuple[Any, ...] | None = keepalive
        self._resources = resources if resources is not None else _RunResources()
        self._cv = threading.Condition()
        self._wait_in_progress = False
        self._accept_wait_in_progress = False
        self._launch_accepted = False
        self._terminal = False
        self._error: BaseException | None = None
        self._finalization_error: BaseException | None = None
        self._finalization_abandoned = False
        # Set only after this run's fence-owned cleanup has actually run. The
        # native fence firing is not the same event: it says the device is
        # drained, not that the CommDomain / L3-L2 / remote-slot teardown that
        # follows it has happened. A successor keyed on the native answer would
        # be admitted while that teardown is still outstanding.
        self._cleanup_published = False

    @classmethod
    def _completed(cls, worker: Worker) -> RunHandle:
        handle = cls.__new__(cls)
        handle._worker = worker
        handle._run_id = None
        handle._keepalive = None
        handle._resources = _RunResources()
        handle._cv = threading.Condition()
        handle._wait_in_progress = False
        handle._accept_wait_in_progress = False
        handle._launch_accepted = True
        handle._terminal = True
        handle._error = None
        handle._finalization_error = None
        handle._finalization_abandoned = False
        # Nothing was ever admitted, so there is no cleanup owing.
        handle._cleanup_published = True
        return handle

    @staticmethod
    def _deadline(timeout: float | None) -> float | None:
        if timeout is None:
            return None
        value = float(timeout)
        if value < 0 or not math.isfinite(value):
            raise ValueError("RunHandle timeout must be a non-negative finite number of seconds")
        return time.monotonic() + value

    @property
    def done(self) -> bool:
        """True once waiting on this run would not block.

        This answers the native fence — the device is drained and every task
        has reached its terminal state — plus this handle's own terminal flag.
        It does **not** say the run's fence-owned cleanup has happened: the
        CommDomain, L3-L2 and remote-slot teardown runs after the fence, in
        whichever thread waits. Anything that has to be ordered behind that
        teardown keys on ``_cleanup_published`` instead.

        Reads False while another waiter is crossing the fence, because the
        native run identity can disappear in that interval.
        """
        with self._cv:
            if self._terminal:
                return True
            # A waiter may have crossed the native fence and released the run
            # identity while it is still publishing cleanup. Avoid querying a
            # native id that can disappear in that interval.
            if self._wait_in_progress:
                return False
            run_id = self._run_id
            assert run_id is not None
            return self._worker._run_handle_done(run_id)

    def _clear_wait_owner(self) -> None:
        """Make an interrupted pre-finalization fence wait re-electable."""
        try:
            while self._wait_in_progress:
                try:
                    self._wait_in_progress = False
                except BaseException:  # noqa: BLE001, PERF203
                    pass
            try:
                with self._cv:
                    self._cv.notify_all()
            except BaseException:  # noqa: BLE001
                # Parked waiters use bounded re-checks and observe the plain flag.
                pass
        except BaseException:  # noqa: BLE001
            self._clear_wait_owner()

    def _clear_acceptance_owner(self) -> None:
        """Make an interrupted acceptance wait re-electable."""
        try:
            while self._accept_wait_in_progress:
                try:
                    self._accept_wait_in_progress = False
                except BaseException:  # noqa: BLE001, PERF203
                    pass
            try:
                with self._cv:
                    self._cv.notify_all()
            except BaseException:  # noqa: BLE001
                # Parked waiters use bounded re-checks and observe the plain flag.
                pass
        except BaseException:  # noqa: BLE001
            self._clear_acceptance_owner()

    def _publish_acceptance(self) -> None:
        """Publish launch acceptance before releasing its elected owner."""
        try:
            while not self._launch_accepted or self._accept_wait_in_progress:
                try:
                    self._launch_accepted = True
                    self._accept_wait_in_progress = False
                except BaseException:  # noqa: BLE001, PERF203
                    pass
            try:
                with self._cv:
                    self._cv.notify_all()
            except BaseException:  # noqa: BLE001
                # Launch acceptance is visible before notification.
                pass
        except BaseException:  # noqa: BLE001
            self._publish_acceptance()

    def _publish_terminal(self, error: BaseException | None) -> BaseException | None:
        """Publish one terminal result despite interruptions between assigns."""
        effective_error = error
        try:
            while not self._terminal or self._wait_in_progress or self._accept_wait_in_progress:
                try:
                    if not self._terminal:
                        self._error = effective_error
                        self._run_id = None
                        if not self._finalization_abandoned:
                            self._keepalive = None
                        self._launch_accepted = True
                        self._terminal = True
                    self._wait_in_progress = False
                    self._accept_wait_in_progress = False
                except BaseException as exc:  # noqa: BLE001, PERF203
                    if not self._terminal and effective_error is None:
                        effective_error = exc
            try:
                with self._cv:
                    self._cv.notify_all()
            except BaseException:  # noqa: BLE001
                # Terminal is visible before notification; bounded re-checks
                # cover a skipped wakeup.
                pass
            return self._error
        except BaseException as exc:  # noqa: BLE001
            if not self._terminal and effective_error is None:
                effective_error = exc
            return self._publish_terminal(effective_error)

    def _cache_finalization_error(self, error: BaseException | None) -> None:
        """Store the finalizer outcome before control returns to its caller."""
        try:
            while True:
                try:
                    self._finalization_error = error
                    return
                except BaseException:  # noqa: BLE001, PERF203
                    pass
        except BaseException:  # noqa: BLE001
            self._cache_finalization_error(error)

    def _recover_and_publish_terminal(self, error: BaseException) -> BaseException | None:
        """Finish interrupted finalization without reopening its cleanup."""
        try:
            while True:
                try:
                    if self._terminal:
                        return self._error
                    recover = getattr(self._worker, "_recover_interrupted_run_finalization", None)
                    if recover is not None:
                        recover(self, error)
                    return self._publish_terminal(error)
                except BaseException:  # noqa: BLE001, PERF203
                    # Recovery and terminal publication form one monotonic
                    # transition. Re-entering recovery only repeats ownership
                    # publication; it never replays a cleanup step.
                    pass
        except BaseException:  # noqa: BLE001
            return self._recover_and_publish_terminal(error)

    def wait(self, timeout: float | None = None) -> None:  # noqa: PLR0912 -- one owner spans fence through publication
        """Wait for completion, raising ``TimeoutError`` or the run's error."""
        # One owner spans the native fence through finalization. An interruption
        # before finalization clears that election; one after finalization starts
        # never re-enters cleanup and conservatively abandons any unpublished
        # ownership. Terminal fields are monotonic, and parked waiters use
        # bounded re-checks when notification itself is interrupted.
        deadline = self._deadline(timeout)
        elected = False
        finalization_started = False
        final_error: BaseException | None = None
        native_error: BaseException | None = None
        try:
            with self._cv:
                while not self._terminal and self._wait_in_progress:
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError("RunHandle.wait() timed out")
                    recheck = (
                        _RUN_HANDLE_WAIT_RECHECK_S if remaining is None else min(remaining, _RUN_HANDLE_WAIT_RECHECK_S)
                    )
                    self._cv.wait(timeout=recheck)
                if self._terminal:
                    error = self._error
                    if error is not None:
                        raise error
                    return
                run_id = self._run_id
                elected = True
                self._wait_in_progress = True

            boundary_hook = getattr(self, "_wait_boundary_hook", None)
            if boundary_hook is not None:
                boundary_hook("after_election")

            assert run_id is not None
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                completed = self._worker._wait_run_handle(run_id, remaining)
            except Exception as exc:  # native run failures are terminal
                completed = True
                native_error = exc

            if not completed:
                raise TimeoutError("RunHandle.wait() timed out")

            # An acceptance waiter that already captured this run id must leave
            # the native wait before finalize releases that id. It is terminal
            # now, so this ownership hand-off does not serialize device
            # execution with acceptance waiting.
            with self._cv:
                while self._accept_wait_in_progress:
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError("RunHandle.wait() timed out")
                    recheck = (
                        _RUN_HANDLE_WAIT_RECHECK_S if remaining is None else min(remaining, _RUN_HANDLE_WAIT_RECHECK_S)
                    )
                    self._cv.wait(timeout=recheck)

            # Finalization owns the accepted-set retirement from this point.
            # An escape after this flag takes conservative abandonment rather
            # than re-entering cleanup or native run release.
            finalization_started = True
            final_error = self._worker._finalize_run_handle(self, run_id, native_error)
            if boundary_hook is not None:
                boundary_hook("after_finalize")
            published_error = self._publish_terminal(final_error)
        except BaseException as exc:  # noqa: BLE001
            if not elected:
                raise
            if not finalization_started:
                self._clear_wait_owner()
                raise

            preferred_error = native_error
            if preferred_error is None:
                preferred_error = self._finalization_error
            if preferred_error is None:
                preferred_error = final_error if final_error is not None else exc
            published_error = self._recover_and_publish_terminal(preferred_error)

        if published_error is not None:
            raise published_error

    def result(self, timeout: float | None = None) -> None:
        """Alias for :meth:`wait`; successful runs have no return value."""
        self.wait(timeout)

    def _wait_for_serialization(self) -> None:
        """Drain this run before another submission, without poisoning it."""
        try:
            self.wait()
        except Exception:
            # The result remains cached for this handle's public wait/result.
            pass

    def _wait_for_handoff(self) -> None:
        """Drain this run *and* its device-touching cleanup before a successor.

        The run's own failure is not this caller's problem — a failed kernel
        says nothing about whether cleanup succeeded — so a task error is
        swallowed here exactly as pre-submission draining does. What must not be
        swallowed is a *cleanup* failure, and that is published by the cleanup
        itself rather than inferred from this exception: whoever ran the cleanup
        recorded it on the worker before dropping the handle, and any waiter
        could have been the one to run it. So this waits, then lets the
        admission check see the poison.
        """
        try:
            self.wait()
        except Exception:
            # Cached on the handle for its public wait()/result().
            pass

    def _wait_for_acceptance(self) -> None:
        """Wait until this run's dispatches cross their acceptance boundary."""
        elected = False
        acceptance_completed = False
        try:
            with self._cv:
                while not self._terminal and self._accept_wait_in_progress:
                    self._cv.wait(timeout=_RUN_HANDLE_WAIT_RECHECK_S)
                if self._terminal or self._launch_accepted:
                    return
                run_id = self._run_id
                elected = True
                self._accept_wait_in_progress = True

            boundary_hook = getattr(self, "_wait_boundary_hook", None)
            if boundary_hook is not None:
                boundary_hook("after_acceptance_election")

            assert run_id is not None
            self._worker._wait_run_handle_accepted(run_id)
            acceptance_completed = True
            if boundary_hook is not None:
                boundary_hook("after_acceptance_wait")
            self._publish_acceptance()
        except BaseException:
            if not elected:
                raise
            if acceptance_completed:
                self._publish_acceptance()
            else:
                self._clear_acceptance_owner()
            raise


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
        # Rebound from the level in `init()`; the default matches the C++ table's
        # so a span emitted before init names L3 rather than nothing.
        self._host_span_prefix = _span_prefix(WorkerLevel.node)
        self._config = config
        self._callable_registry: dict[int, Any] = {}
        self._identity_registry: dict[bytes, _CallableIdentityState] = {}
        self._live_handles: dict[int, bytes] = {}
        self._next_handle_id: int = 0
        self._owner_id = uuid.uuid4().hex
        self._shm_token: str = ""
        self._shm_tree_tokens: set[str] = set()
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
        # True once close() has entered teardown. A journal residual overrides
        # this replay latch and permits a later close() to re-drive the recorded
        # post-success cleanup actions.
        self._teardown_attempted: bool = False
        self._cancel_token: bool = False
        self._cleanup_journal = CleanupJournal()
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
        # wrong / unbound device context). A close() while INITIALIZING is served
        # by cooperative cancellation on a non-owner thread and rejected on the
        # init-owner thread itself; any thread may join an in-flight close().
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
        # Device *execution* is one run at a time — the whole-run FIFO admits a
        # prepared successor but dispatches only the head — while how many runs
        # may be admitted at once is the depth the child backends negotiated.
        # This gate serialises graph construction itself, which stays
        # synchronous on the submitting caller at any depth.
        # Exclusive for run admission, shared for control that belongs to no run:
        # such a command must exclude admission, but not other chips' commands.
        self._submit_mu = _SharedExclusiveLock()
        # Guarded by _hierarchical_start_cv. Handles are installed before their
        # orchestration callback can enqueue work and retired after fence-owned
        # cleanup, so close() can drain the exact accepted set.
        self._accepted_run_handles: set[RunHandle] = set()
        # A nonterminal cancellation or ambiguous finalization boundary has no
        # fence-owned cleanup Python can safely replay. Keep that run's callback
        # and resource references alive outside close()'s drain set; whole-tree
        # teardown is the only remaining safe reclamation boundary.
        self._abandoned_run_handles: list[RunHandle] = []
        # Set once a cleanup-bearing run's ordered cleanup failed. That cleanup
        # is collective control on the children, so a failure leaves device
        # state this process can neither describe nor reclaim; admitting more
        # work would build on it. Sticky — there is no recovery short of close.
        # Guarded by _hierarchical_start_cv.
        self._ordered_cleanup_error: BaseException | None = None
        # Sticky copy of this Worker's native return-boundary mapping cleanup
        # diagnostic. Native storage is acknowledged only after this field and
        # the admission poison above have been published.
        self._worker_host_mapped_cleanup_error: RuntimeError | None = None
        # (snapshot awaiting acknowledgement, retained distinct detail
        # fragments). This is an ordered diagnostic set, not a resource-count
        # ledger: an identical repeated failure adds no actionable information.
        # Keep it immutable so one attribute assignment publishes both pieces
        # atomically with respect to asynchronous Python exceptions.
        self._worker_host_mapped_cleanup_state: tuple[str | None, tuple[str, ...]] = (None, ())
        # submit graph construction is serialized by _submit_mu. Resource
        # creation helpers use this pointer to bind new objects to the handle
        # being built; the pointer is cleared before submit() returns.
        self._building_run_resources: _RunResources | None = None
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
        # Live direct-chip runs by the run id their RunHandle carries. The lane
        # is the admission authority and holds its own FIFO; this only maps a
        # handle back to its run so waiting and finalization can find it.
        self._chip_runs: dict[int, Any] = {}
        self._chip_run_seq: int = 0
        # Mirrors _chip_runs' own lifecycle (added/removed at the same two points) so
        # release_buffer() can see an L2 run in flight: L2 dispatch never touches
        # _accepted_run_handles/_submit_mu, so it is otherwise invisible to that check.
        self._chip_run_touched_identities: dict[int, set[CanonicalIdentity]] = {}

        # Level-3+ internals
        self._worker: _Worker | None = None
        self._orch: Orchestrator | None = None
        self._chip_shms: list[SharedMemory] = []
        self._chip_pids: list[int] = []
        self._sub_shms: list[SharedMemory] = []
        self._sub_pids: list[int] = []

        # L4+ next-level Worker children (added via add_worker before init)
        self._next_level_workers: list[Worker] = []
        self._topology_parent: Worker | None = None
        self._next_level_worker_ids: list[int] = []
        self._next_level_shms: list[SharedMemory] = []
        self._next_level_pids: list[int] = []
        self._remote_worker_specs: list[RemoteWorkerSpec] = []
        self._remote_worker_ids: list[int] = []
        self._remote_sessions: list[_RemoteSession] = []
        self._mpi_l3_groups: list[_MpiL3GroupRuntime] = []
        self._mpi_worker_ids: list[int] = []
        self._mpi_rank_by_worker_id: dict[int, _MpiL3RankRuntime] = {}
        self._next_level_worker_id_count: int = 0
        # Fallback ownership for private helpers used outside Worker.submit.
        # Normal orchestration-owned refs live in RunHandle._resources.
        self._active_remote_slot_refs: list[_RemoteSlotRefClaim | RemoteBufferHandle] = []
        self._pending_remote_buffer_frees: list[RemoteBufferHandle] = []
        # One publication/drain boundary for every remote release debt.  In
        # particular, an owner free must not append after a run fence has taken
        # the queue's only snapshot, and two callers must not both send it.
        self._remote_import_release_mu = threading.RLock()
        self._pending_remote_import_releases: list[RemoteBufferHandle] = []
        self._pending_remote_import_release_states: dict[RemoteBufferHandle, _PendingRemoteImportReleaseState] = {}

        # Dynamic CommDomain allocations.  Keyed by user-facing name (unique
        # among live handles).  ``orch.allocate_domain`` adds entries here;
        # ``release()`` removes them and queues a deferred backend free.
        self._live_domains: dict[str, CommDomainHandle] = {}
        self._live_global_domains: dict[str, GlobalCommDomainHandle] = {}
        self._failed_global_domain_releases: dict[int, GlobalCommDomainHandle] = {}
        self._global_node_domains: dict[int, _GlobalNodeDomainState] = {}
        self._global_cluster_id = uuid.uuid4().hex
        # Monotonic per-Worker counter; mixed into IPC barrier filenames so
        # two concurrent allocations don't share a marker file.  Wraps after
        # 2^64 allocations — far beyond any realistic Worker lifetime.
        self._next_alloc_id: int = 0
        # Exactly one caller drives CTRL_RELEASE_DOMAIN for a given allocation,
        # and every other caller waits for its outcome. Several paths can reach
        # one handle at once — an end-of-run or close() sweep working from its
        # snapshot, and a release on an already retired run — where a second
        # release of a live allocation is a backend double free, and returning
        # before the first one finishes would let close() tear the mailboxes
        # down under an in-flight release.
        self._domain_free_mu = threading.Lock()
        self._domain_free_results: dict[int, BaseException | None] = {}
        self._global_domain_free_mu = threading.Lock()
        self._global_domain_free_results: dict[int, BaseException | None] = {}
        self._alloc_id_lock = threading.Lock()
        # Base HCCL/sim communicator is built lazily on the first
        # ``orch.allocate_domain`` call (see ``_ensure_comm_base``).  We
        # keep ``Worker.init()`` cheap — it only forks chip children and
        # starts the C++ scheduler; no comm work happens there.
        self._comm_base_ready: bool = False

        self._endpoint_registry: EndpointRegistry | None = None
        self._endpoint_registry_epoch: int = 0
        self._region_access_service: RegionAccessService | None = None

        self._live_worker_chip_regions: list[Any] = []
        self._worker_chip_orch_comm_host_buffers: dict[int, int] = {}

        # Live device allocations, keyed by the identity that names one. Membership authorizes an
        # operand; it does not own the memory. Nothing in this table releases anything -- a malloc'd
        # allocation is freed by `free`, a domain's window by its collective release -- which is the
        # contract `self._buffers` does NOT have, where membership means "close() me".
        # Ordering is safety-first: an entry is recorded only after the native alloc succeeds, and
        # revoked BEFORE the native free (and before a domain's backend release), so an interrupted
        # op never leaves an identity resolving to memory that is already gone. Cleared on close().
        self._child_alloc: dict[CanonicalIdentity, Buffer] = {}
        # Which identities each CommDomain allocation minted, so its release revokes them together.
        self._domain_members: dict[int, set[CanonicalIdentity]] = {}
        # Guards both device-allocation tables. Entry points take it (`require`,
        # `_device_worker_for`, `_child_prov_check_dispatch`); the record/drop helpers assume the
        # caller holds it, so that a registration and the native call it follows commit together.
        # It is not reentrant, so an entry point must never be called with it already held.
        self._child_prov_lock = threading.Lock()
        # Per-worker locks for the *native* half of a provenance-guarded device op.
        # ``_child_prov_lock`` stays the bookkeeping lock (short, process-wide); the
        # long native call (malloc / free / copy) is serialized per worker instead, so
        # ops on different chips overlap while same-worker ordering is unchanged.
        self._child_prov_worker_locks: dict[int, threading.Lock] = {}

        # Owner-side Buffer state (P1-B): a per-incarnation opaque nonce, a monotonic buffer_id
        # (0 reserved), and the live handles this Worker owns. create_buffer allocates a handle whose
        # self-describing descriptor rides embedded in every Tensor built over it (no export
        # handshake); consumers materialize it lazily on receipt.
        #
        # The endpoint registry mints `EndpointIdentity.session_instance_id` from this same nonce,
        # and the sharing is load-bearing rather than incidental: it is the key of the registry's
        # `owner_instance_id -> owner endpoint` binding, which is the only way a consumer holding a
        # `BufferDescriptor` can name the endpoint that minted it — the descriptor itself cannot,
        # since the nonce is opaque, `owner_worker_path_id` is diagnostic by contract, and
        # `address_space` does not say which card.
        #
        # Both identities share this mint point. init() re-mints it (see the
        # `_lifecycle = _Lifecycle.INITIALIZING` assignment) rather than trusting the value from
        # here: for a next-level child, init() only ever runs inside the process that forked to
        # host it, so that later mint is the one that names the real incarnation and is never older
        # than its fork. The value assigned here exists only so a Worker that never reaches init()
        # (e.g. a test double that pokes `_lifecycle` directly) still has a well-formed nonce.
        self._owner_instance_id: bytes = mint_owner_instance_id()
        self._buffer_id_counter: int = 1
        self._buffers: dict[int, Buffer] = {}
        # Re-export table (points 1-4): an upper-level ref received by this worker's orch is re-exported
        # to a local handle H' under this worker's identity, per-backing (keyed by source identity),
        # so each level's orch sees only its own handles. No map here — H' relabels the backing;
        # a compute leaf maps lazily. Lifetime is worker-scoped for now.
        self._reexport_by_source: dict[CanonicalIdentity, Buffer] = {}
        # make_tensor_arg memo: a pre-fork host tensor's storage base -> its FORK_SHM handle, so every ref
        # over the same storage shares one canonical identity (dependencies key on it). Worker-scoped.
        self._fork_tensor_handles: dict[int, Buffer] = {}
        # L2 leaf only: the in-process consumer import cache. An L2 Worker materializes its own tensor
        # args itself (no forked child, no mailbox), resolving each ref's descriptor to a local base
        # map-once — the chip-child path minus the mailbox hop. Lazily created on first L2 run.
        self._chip_import_registry: ImportRegistry | None = None

    @property
    def _initialized(self) -> bool:
        """True only in READY — the worker's tree is live and dispatchable.

        False once CLOSED (the moment close() claims the epoch), so a dispatch /
        register / create_buffer that races an in-progress close() is
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
        """Register a remote L3 worker and return its NEXT_LEVEL worker id.

        Must be called before ``init()`` — the topology freezes there — and only
        on a ``level >= 4`` parent. ``spec.endpoint`` is validated here rather
        than at activation, so a bad address fails before any process is forked;
        its host must be a numeric IPv4 address or ``localhost``. IPv6 is not
        reachable here: the endpoint is parsed as a single ``host:port`` pair, so
        a literal carrying more than one colon is rejected before the numeric
        check runs (see ``RemoteWorkerSpec``).
        """
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
            # Validate the endpoint here, before any startup resource exists, so a
            # non-numeric host fails at registration rather than mid-activation
            # (which would roll back the whole already-forked tree).
            host, _port = self._parse_remote_endpoint(spec.endpoint)
            self._validate_numeric_endpoint_host(host)
            worker_id = self._allocate_next_level_worker_id()
            self._remote_worker_specs.append(spec)
            self._remote_worker_ids.append(worker_id)
            return worker_id

    def add_mpirun_worker_group(self, spec: MpiL3GroupSpec) -> tuple[int, ...]:
        """Register L3 workers that will be launched by one parent-owned ``mpirun``.

        A single named mailbox is created for the group during ``init()``.
        The returned ids remain exact NEXT_LEVEL targets, but all of them route
        through that group mailbox and the MPI collective dispatcher.
        """
        with self._hierarchical_start_cv:
            if self._lifecycle is not _Lifecycle.NEW:
                raise RuntimeError("Worker.add_mpirun_worker_group after init")
            if self.level < 4:
                raise TypeError("Worker.add_mpirun_worker_group: MPI L3 groups require a level >= 4 parent")
            if not isinstance(spec, MpiL3GroupSpec):
                raise TypeError("Worker.add_mpirun_worker_group expects a MpiL3GroupSpec")

            group_id = uuid.uuid4().hex
            ranks: list[_MpiL3RankRuntime] = []
            for rank in range(len(spec.hosts)):
                worker_id = self._allocate_next_level_worker_id()
                rank_spec = _MpiL3RankSpec(
                    platform=spec.platform,
                    runtime=spec.runtime,
                    device_ids=spec.device_ids_by_rank[rank],
                    num_sub_workers=spec.num_sub_workers_by_rank[rank],
                    transport=spec.transport,
                    comm_profile=spec.comm_profile,
                    global_device_ranks=spec.global_device_ranks_by_rank[rank],
                )
                runtime = _MpiL3RankRuntime(
                    group_id=group_id,
                    rank=rank,
                    worker_id=worker_id,
                    session_id=self._new_remote_session_id(),
                    spec=rank_spec,
                )
                ranks.append(runtime)
                self._mpi_worker_ids.append(worker_id)
                self._mpi_rank_by_worker_id[worker_id] = runtime
            group = _MpiL3GroupRuntime(group_id=group_id, spec=spec, ranks=tuple(ranks))
            self._mpi_l3_groups.append(group)
            return tuple(rank.worker_id for rank in ranks)

    def _remote_like_worker_ids(self) -> set[int]:
        return set(self._remote_worker_ids) | set(self._mpi_worker_ids)

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
    def _validate_numeric_endpoint_host(host: str) -> None:
        # Remote L3 endpoints are numeric-only (or localhost) by contract:
        # hostname resolution via getaddrinfo is unbounded and uncancellable, so
        # it is rejected rather than risk pinning startup on a hung resolver.
        if host == "localhost":
            return
        try:
            socket.getaddrinfo(host, None, flags=socket.AI_NUMERICHOST)
        except socket.gaierror as exc:
            raise ValueError(
                f"RemoteWorkerSpec.endpoint host must be a numeric IP address (hostname resolution is "
                f"unbounded and unsupported for remote L3); got {host!r}"
            ) from exc

    @staticmethod
    def _is_wildcard_session_host(host: str) -> bool:
        return host in ("0.0.0.0", "::")

    def _remote_session_timeout_s(self) -> float:
        timeout_s = float(self._config.get("remote_session_timeout_s", 30.0))
        if not (timeout_s > 0 and math.isfinite(timeout_s)):
            raise ValueError("Worker remote_session_timeout_s must be a positive finite number of seconds")
        return timeout_s

    @staticmethod
    def _remaining_until(deadline: float, what: str) -> float:
        # A blocking op's slice of the single root startup deadline. Raising here
        # keeps the timeout local and clear, and avoids settimeout(0.0) — which
        # would flip the socket to non-blocking and surface BlockingIOError.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"{what}: startup deadline exceeded")
        return remaining

    @staticmethod
    def _resolve_within_deadline(host: str, port: int, deadline: float) -> list[Any]:
        # Numeric-only, by contract: getaddrinfo is not cancellable, so a hung
        # NSS/DNS lookup could pin init() in INITIALIZING past the root deadline.
        # AI_NUMERICHOST performs NO name resolution (it parses a numeric literal
        # or fails immediately), so this never blocks; a hostname is rejected
        # outright rather than risk an unbounded stall. "localhost" is accepted as
        # the loopback literal. The deadline pre-check keeps a spent budget from
        # even attempting the parse.
        Worker._remaining_until(deadline, "remote L3 session resolve")
        lookup = "127.0.0.1" if host == "localhost" else host
        try:
            return socket.getaddrinfo(
                lookup, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP, flags=socket.AI_NUMERICHOST
            )
        except socket.gaierror as exc:
            raise ValueError(
                f"remote L3 endpoint host must be a numeric IP address (hostname resolution is "
                f"unbounded and unsupported); got {host!r}"
            ) from exc

    @staticmethod
    def _connect_within_deadline(host: str, port: int, deadline: float) -> socket.socket:
        # Bound name resolution AND every per-address connect attempt by the single
        # root deadline — mirroring the C++ connect_tcp_socket — so a slow resolver
        # or a black-holed first address cannot let this stage restart the clock or
        # outrun the startup budget (unlike socket.create_connection, which grants
        # a fresh full timeout to every address and never bounds getaddrinfo).
        infos = Worker._resolve_within_deadline(host, port, deadline)
        last_exc: BaseException | None = None
        for family, socktype, proto, _canonname, sockaddr in infos:
            remaining = Worker._remaining_until(deadline, "remote L3 session connect")
            sock = socket.socket(family, socktype, proto)
            try:
                sock.settimeout(remaining)
                sock.connect(sockaddr)
                return sock
            except OSError as exc:
                last_exc = exc
                sock.close()
        if last_exc is not None:
            raise last_exc
        raise OSError(f"remote L3 session connect: no address for {host}:{port}")

    @staticmethod
    def _send_remote_daemon_json(sock: socket.socket, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        sock.sendall(struct.pack("<I", len(data)) + data)

    @staticmethod
    def _recv_remote_daemon_json(sock: socket.socket, deadline: float) -> dict[str, Any]:
        size_data = bytearray()
        while len(size_data) < 4:
            sock.settimeout(Worker._remaining_until(deadline, "remote daemon reply"))
            chunk = sock.recv(4 - len(size_data))
            if not chunk:
                raise EOFError("remote daemon closed before reply length")
            size_data.extend(chunk)
        size = struct.unpack("<I", bytes(size_data))[0]
        if size > 16 * 1024 * 1024:
            raise RuntimeError("remote daemon reply exceeds maximum")
        data = bytearray()
        while len(data) < size:
            sock.settimeout(Worker._remaining_until(deadline, "remote daemon reply"))
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

    def _inner_registry_entries_for_spec(self, spec: RemoteWorkerSpec | _MpiL3RankSpec) -> list[dict[str, Any]]:
        from .remote_l3_protocol import (  # noqa: PLC0415
            ChipCallableBlobLocation,
            RemoteChipCallablePayload,
            encode_remote_chip_callable_payload,
        )

        entries: list[dict[str, Any]] = []
        with self._registry_lock:
            states = list(self._identity_registry.values())
        for state in states:
            if state.target_namespace != "LOCAL_CHIP":
                continue
            if not isinstance(state.target, ChipCallable):
                raise RuntimeError(f"inner chip hashid {state.hashid} does not carry a ChipCallable target")
            descriptor = build_chip_callable_descriptor(
                target=state.target,
                platform=spec.platform,
                runtime=spec.runtime,
            )
            if descriptor != state.descriptor:
                raise RuntimeError(f"inner chip hashid {state.hashid} was registered for a different platform/runtime")
            blob = ctypes.string_at(int(state.target.buffer_ptr()), int(state.target.buffer_size()))
            payload = encode_remote_chip_callable_payload(
                RemoteChipCallablePayload(
                    descriptor_bytes=descriptor,
                    blob_location=ChipCallableBlobLocation.INLINE_BLOB,
                    blob_size=len(blob),
                    blob_sha256=hashlib.sha256(blob).digest(),
                    inline_blob=blob,
                    staged_blob_token=b"",
                )
            )
            entries.append(
                {
                    "hashid": state.digest.hex(),
                    "kind": "CHIP_CALLABLE",
                    "target_registry": "INNER_L3_WORKER",
                    "payload_version": 1,
                    "payload_hex": payload.hex(),
                }
            )
        return entries

    @staticmethod
    def _validate_global_node_config(
        *,
        label: str,
        platform: str,
        device_ids: tuple[int, ...],
        comm_profile: str,
        global_device_ranks: tuple[int, ...],
    ) -> None:
        _validate_global_comm_capability(label, platform, comm_profile)
        _validate_global_device_ranks(f"{label}.global_device_ranks", global_device_ranks, len(device_ids))

    def _resolved_global_nodes(self) -> dict[int, _GlobalNodeRuntime]:
        configs: list[tuple[int, tuple[int, ...], str, str, tuple[int, ...], bool]] = []
        for worker_id, spec in zip(self._remote_worker_ids, self._remote_worker_specs):
            configs.append(
                (
                    int(worker_id),
                    tuple(spec.device_ids),
                    spec.platform,
                    spec.comm_profile,
                    tuple(spec.global_device_ranks),
                    True,
                )
            )
        for worker_id in self._mpi_worker_ids:
            rank = self._mpi_rank_by_worker_id[int(worker_id)]
            configs.append(
                (
                    int(worker_id),
                    tuple(rank.spec.device_ids),
                    rank.spec.platform,
                    rank.spec.comm_profile,
                    tuple(rank.spec.global_device_ranks),
                    True,
                )
            )
        for worker_id, child in zip(self._next_level_worker_ids, self._next_level_workers):
            if child.level != 3:
                continue
            device_ids = tuple(int(device_id) for device_id in child._config.get("device_ids", ()))
            platform = str(child._config.get("platform", ""))
            comm_profile = str(child._config.get("comm_profile", "sim"))
            global_device_ranks = tuple(int(rank) for rank in child._config.get("global_device_ranks", ()))
            self._validate_global_node_config(
                label=f"local L3 worker {worker_id}",
                platform=platform,
                device_ids=device_ids,
                comm_profile=comm_profile,
                global_device_ranks=global_device_ranks,
            )
            configs.append(
                (
                    int(worker_id),
                    device_ids,
                    platform,
                    comm_profile,
                    global_device_ranks,
                    False,
                )
            )
        configs.sort(key=lambda item: item[0])

        explicit_ranks: set[int] = set()
        for worker_id, _device_ids, _platform, _profile, ranks, _is_remote in configs:
            overlap = explicit_ranks.intersection(ranks)
            if overlap:
                raise ValueError(
                    f"Global CommDomain worker {worker_id} duplicates global_device_ranks {sorted(overlap)}"
                )
            explicit_ranks.update(ranks)

        used = set(explicit_ranks)
        next_rank = 0
        resolved: dict[int, _GlobalNodeRuntime] = {}
        node_count = len(configs)
        for node_rank, (worker_id, device_ids, platform, profile, ranks, is_remote) in enumerate(configs):
            self._validate_global_node_config(
                label=f"{'remote' if is_remote else 'local'} L3 worker {worker_id}",
                platform=platform,
                device_ids=device_ids,
                comm_profile=profile,
                global_device_ranks=ranks,
            )
            if not ranks:
                assigned: list[int] = []
                for _device_id in device_ids:
                    while next_rank in used:
                        next_rank += 1
                    assigned.append(next_rank)
                    used.add(next_rank)
                    next_rank += 1
                ranks = tuple(assigned)
            resolved[worker_id] = _GlobalNodeRuntime(
                worker_id=worker_id,
                device_ids=device_ids,
                platform=platform,
                comm_profile=profile,
                global_device_ranks=ranks,
                node_rank=node_rank,
                node_count=node_count,
                cluster_id=self._global_cluster_id,
                is_remote=is_remote,
            )
        return resolved

    def _resolved_global_device_ranks(self) -> dict[int, tuple[int, ...]]:
        return {worker_id: runtime.global_device_ranks for worker_id, runtime in self._resolved_global_nodes().items()}

    def _build_remote_manifest(
        self, *, spec: RemoteWorkerSpec, worker_id: int, session_id: int, startup_remaining_s: float
    ) -> dict[str, Any]:
        daemon_host, _daemon_port = self._parse_remote_endpoint(spec.endpoint)
        listen_host = spec.session_listen_host or ("127.0.0.1" if daemon_host == "localhost" else daemon_host)
        if self._is_wildcard_session_host(listen_host) and not spec.allow_wildcard_session_bind:
            raise ValueError("RemoteWorkerSpec wildcard session bind requires allow_wildcard_session_bind=True")
        if worker_id in self._remote_like_worker_ids():
            runtime = self._resolved_global_nodes()[int(worker_id)]
            node_rank = runtime.node_rank
            node_count = runtime.node_count
            global_device_ranks = runtime.global_device_ranks
        else:
            node_rank = 0
            node_count = 1
            global_device_ranks = spec.global_device_ranks or tuple(range(len(spec.device_ids)))
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
            "comm_profile": spec.comm_profile,
            "cluster_id": self._global_cluster_id,
            "node_rank": node_rank,
            "node_count": node_count,
            "global_device_ranks": list(global_device_ranks),
            # session_timeout_s bounds the runtime command socket; startup_remaining_s
            # bounds this session's slice of the single root startup budget. They are
            # distinct: the remote must not spend runtime-command time as startup time.
            "session_timeout_s": self._remote_session_timeout_s(),
            "startup_remaining_s": float(startup_remaining_s),
            "listen_host": listen_host,
            "connect_host": daemon_host,
            "remote_task_dispatcher": self._remote_dispatcher_entries_for_worker(worker_id),
            "inner_l3_worker": self._inner_registry_entries_for_spec(spec),
            "feature_flags": [],
        }

    def _build_mpi_rank_manifest(
        self,
        *,
        rank: _MpiL3RankRuntime,
        startup_remaining_s: float,
    ) -> dict[str, Any]:
        spec = rank.spec
        runtime = self._resolved_global_nodes()[int(rank.worker_id)]
        return {
            "session_id": int(rank.session_id),
            "parent_worker_level": int(self.level),
            "remote_worker_level": 3,
            "worker_id": int(rank.worker_id),
            "platform": spec.platform,
            "runtime": spec.runtime,
            "device_ids": list(spec.device_ids),
            "num_sub_workers": int(spec.num_sub_workers),
            "heap_ring_size": self._config.get("remote_heap_ring_size", None),
            "transport": spec.transport,
            "comm_profile": spec.comm_profile,
            "cluster_id": self._global_cluster_id,
            "node_rank": runtime.node_rank,
            "node_count": runtime.node_count,
            "global_device_ranks": list(runtime.global_device_ranks),
            "session_timeout_s": self._remote_session_timeout_s(),
            "startup_remaining_s": float(startup_remaining_s),
            "remote_task_dispatcher": self._remote_dispatcher_entries_for_worker(rank.worker_id),
            "inner_l3_worker": self._inner_registry_entries_for_spec(spec),
            "feature_flags": ["mpi-group-mailbox-v1"],
        }

    def _open_remote_session(
        self, *, spec: RemoteWorkerSpec, worker_id: int, session_id: int, deadline: float
    ) -> _RemoteSession:
        daemon_host, daemon_port = self._parse_remote_endpoint(spec.endpoint)
        # Every blocking op (resolve, connect, send, framed recv) derives its
        # remaining from the single root deadline, so their sum cannot exceed the
        # root startup budget.
        with self._connect_within_deadline(daemon_host, daemon_port, deadline) as sock:
            manifest = self._build_remote_manifest(
                spec=spec, worker_id=worker_id, session_id=session_id, startup_remaining_s=0.0
            )
            # Derive the send budget AFTER building the (registry-iterating)
            # manifest, right before send, so the socket timeout and the wire
            # duration reflect what is actually left — not a pre-build sample.
            startup_remaining_s = self._remaining_until(deadline, "remote L3 session handshake")
            manifest["startup_remaining_s"] = startup_remaining_s
            sock.settimeout(startup_remaining_s)
            self._send_remote_daemon_json(sock, manifest)
            reply = self._recv_remote_daemon_json(sock, deadline)
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

    def _release_remote_sessions(self) -> None:
        sessions = list(self._remote_sessions)
        self._close_remote_sessions(sessions)
        self._remote_sessions.clear()

    @staticmethod
    def _new_remote_session_id() -> int:
        session_id = uuid.uuid4().int & ((1 << 63) - 1)
        return session_id if session_id != 0 else 1

    @staticmethod
    def _mpirun_args_select_hosts(args: tuple[str, ...]) -> bool:
        host_args = {"--host", "-host", "-H", "--hostfile", "-hostfile", "--machinefile", "-machinefile"}
        return any(arg in host_args or arg.startswith("--host=") or arg.startswith("--hostfile=") for arg in args)

    @staticmethod
    def _close_mpirun_group(  # noqa: PLR0912 -- cleanup reports every independent process/resource failure
        group: _MpiL3GroupRuntime,
        *,
        timeout_s: float,
    ) -> list[str]:
        failures: list[str] = []
        group.closing = True
        proc = group.process
        if proc is not None:
            try:
                # The remote sessions received SHUTDOWN before this runs, so a
                # healthy group exits by itself once its ranks close their
                # inner workers. SIGTERM is the backstop, not the first move:
                # a rank interrupted mid native teardown dies uncleanly and
                # mpirun reports the whole job as a bad termination.
                if proc.poll() is None:
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        proc.wait(timeout=timeout_s)
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    except BaseException as exc:  # noqa: BLE001
                        failures.append(f"terminate: {exc}")
                try:
                    proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except BaseException as exc:  # noqa: BLE001
                        failures.append(f"kill: {exc}")
                    try:
                        proc.wait(timeout=timeout_s)
                    except BaseException as exc:  # noqa: BLE001
                        failures.append(f"wait after kill: {exc}")
                except BaseException as exc:  # noqa: BLE001
                    failures.append(f"wait after terminate: {exc}")
            finally:
                group.process = None
        if group.monitor_thread is not None:
            group.monitor_thread.join(timeout=timeout_s)
            group.monitor_thread = None
        if group.mailbox is not None:
            try:
                group.mailbox.close(unlink=True)
            except BaseException as exc:  # noqa: BLE001
                failures.append(f"close mailbox: {exc}")
            finally:
                group.mailbox = None
        try:
            if group.ready_dir is not None:
                shutil.rmtree(group.ready_dir)
        except FileNotFoundError:
            pass
        except BaseException as exc:  # noqa: BLE001
            failures.append(f"remove ready directory: {exc}")
        finally:
            group.ready_dir = None
            group.manifest_path = None
        return [f"MPI L3 group {group.group_id} cleanup {failure}" for failure in failures]

    def _close_mpirun_groups(
        self,
        *,
        timeout_s: float = _ROLLBACK_GRACEFUL_TIMEOUT_S,
        suppress_errors: bool = False,
    ) -> None:
        failures: list[str] = []
        for group in reversed(self._mpi_l3_groups):
            failures.extend(self._close_mpirun_group(group, timeout_s=timeout_s))
        if failures:
            sys.stderr.write("\n".join(f"[worker pid={os.getpid()}] WARN: {failure}" for failure in failures) + "\n")
            sys.stderr.flush()
            if not suppress_errors:
                raise RuntimeError(failures[0])

    @staticmethod
    def _monitor_mpirun_group(group: _MpiL3GroupRuntime, proc: subprocess.Popen[Any]) -> None:
        returncode = proc.wait()
        if group.closing or group.mailbox is None:
            return
        with contextlib.suppress(BaseException):
            group.mailbox.mark_terminal(f"mpirun exited unexpectedly with status {returncode}")

    @staticmethod
    def _mark_mpirun_groups_closing(groups: list[_MpiL3GroupRuntime]) -> None:
        for group in groups:
            group.closing = True

    def _activate_mpirun_worker_groups(self, deadline: float) -> None:
        if not self._mpi_l3_groups:
            return
        from .mpi_group_mailbox import MAILBOX_SIZE as MPI_MAILBOX_SIZE  # noqa: PLC0415
        from .mpi_group_mailbox import MailboxGroupState, MpiGroupMailbox  # noqa: PLC0415

        session_timeout = self._remote_session_timeout_s()
        assert self._worker is not None
        for group in self._mpi_l3_groups:
            ready_dir = tempfile.mkdtemp(prefix="simpler-mpirun-manifest-")
            manifest_path = os.path.join(ready_dir, "group.json")
            group.ready_dir = ready_dir
            group.manifest_path = manifest_path
            mailbox = MpiGroupMailbox.create(world_size=len(group.ranks))
            group.mailbox = mailbox
            rank_manifests: list[dict[str, Any]] = []
            startup_remaining_s = self._remaining_until(deadline, "MPI L3 group manifest")
            for rank in group.ranks:
                manifest = self._build_mpi_rank_manifest(
                    rank=rank,
                    startup_remaining_s=startup_remaining_s,
                )
                manifest.update(
                    {
                        "mpi_group_id": group.group_id,
                        "mpi_rank": rank.rank,
                        "mpi_world_size": len(group.ranks),
                        "mpi_group_worker_ids": [item.worker_id for item in group.ranks],
                        "mpi_global_domain_exchange": True,
                    }
                )
                rank_manifests.append(manifest)
            group_manifest = {
                "group_id": group.group_id,
                "world_size": len(group.ranks),
                "worker_ids": [rank.worker_id for rank in group.ranks],
                "mailbox": mailbox.manifest(),
                "rank_manifests": rank_manifests,
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(group_manifest, f, sort_keys=True)
                f.write("\n")

            cmd = [group.spec.mpirun_path, "-np", str(len(group.ranks))]
            if not self._mpirun_args_select_hosts(group.spec.mpirun_args):
                cmd.extend(["--host", ",".join(group.spec.hosts)])
            cmd.extend(group.spec.mpirun_args)
            cmd.extend(
                [
                    group.spec.python_executable,
                    "-m",
                    "simpler.mpi_l3_session",
                    "--group-manifest",
                    manifest_path,
                ]
            )
            group.process = subprocess.Popen(cmd, start_new_session=True)
            while mailbox.group_state is MailboxGroupState.INITIALIZING:
                if group.process.poll() is not None:
                    raise RuntimeError(
                        f"MPI L3 group {group.group_id} exited before READY (status {group.process.returncode})"
                    )
                self._remaining_until(deadline, "MPI L3 mailbox READY")
                time.sleep(_STARTUP_POLL_INTERVAL_S)
            if mailbox.group_state is not MailboxGroupState.READY:
                raise RuntimeError(f"MPI L3 group {group.group_id} failed before READY: {mailbox.terminal_reason()}")
            self._worker.add_mpi_group_mailbox(
                [rank.worker_id for rank in group.ranks],
                [rank.session_id for rank in group.ranks],
                mailbox.address,
                MPI_MAILBOX_SIZE,
                group.process.pid,
                session_timeout,
            )
            group.monitor_thread = threading.Thread(
                target=self._monitor_mpirun_group,
                args=(group, group.process),
                daemon=True,
                name=f"simpler-mpirun-monitor-{group.group_id[:8]}",
            )
            group.monitor_thread.start()
        if time.monotonic() >= deadline:
            raise RuntimeError("MPI L3 activation: startup deadline exceeded after attach")

    def _require_remote_worker_started(self, worker_id: int) -> None:
        """Argument + resource gate for the public remote-memory APIs. Admission
        (READY) is decided by the ``_operation_lease`` these APIs already hold —
        this checks only worker id, level, and transport **presence** (not the
        public lifecycle), so an operation legitimately admitted before a
        concurrent ``close()`` published CLOSED still completes during the drain
        instead of spuriously failing."""
        if self.level < 4:
            raise TypeError("remote memory APIs require a level >= 4 parent Worker")
        if int(worker_id) not in self._remote_like_worker_ids():
            raise ValueError(
                "remote memory APIs require a remote worker id returned by add_remote_worker or add_mpirun_worker_group"
            )
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
        if int(worker_id) not in self._remote_like_worker_ids():
            raise ValueError(
                "remote memory APIs require a remote worker id returned by add_remote_worker or add_mpirun_worker_group"
            )
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
        """Allocate ``nbytes`` on a started remote worker and return an owner handle.

        ``nbytes`` must be positive. The target remote worker must already be
        started, so this is callable only after ``init()``.
        """
        worker_id = int(worker)
        size = int(nbytes)
        if size <= 0:
            raise ValueError("Worker.remote_malloc nbytes must be positive")
        with self._operation_lease("remote_malloc"), self._control_admission("remote_malloc"):
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
        """Free an owner remote allocation.

        Idempotent: freeing an already-released handle is a no-op. Rejects
        imported handles (use ``remote_release_import``) and ``HOST_INLINE``
        handles, which are not remote allocations. If the buffer is still
        referenced by a live task slot or by an outstanding import, the free is
        recorded and deferred until those references drop rather than issued now.
        """
        if not isinstance(handle, RemoteBufferHandle):
            raise TypeError("expected a RemoteBufferHandle returned by Worker.remote_malloc/import")
        if handle.address_space == RemoteAddressSpace.HOST_INLINE:
            raise ValueError("HOST_INLINE RemoteBufferHandle is not a remote allocation")
        if handle.is_imported:
            raise ValueError("remote_free is invalid for imported handles; use remote_release_import")
        if handle.released:
            with self._remote_import_release_mu:
                if handle not in self._pending_remote_buffer_frees:
                    return
        # Public admission: READY-only + drained. (The private _send_* transport
        # helper accepts CLOSED so teardown can flush pending frees, so remote_free
        # must fence admission itself rather than lean on the transport gate.)
        with self._operation_lease("remote_free"):
            with self._remote_import_release_mu:
                if handle.released:
                    if handle not in self._pending_remote_buffer_frees:
                        return
                    if handle._live_slot_refs > 0 or handle._live_import_refs > 0:
                        return
                if handle._live_slot_refs > 0 or handle._live_import_refs > 0:
                    # Deferred: nothing reaches the owner here, so there is no
                    # command to order against a run. The fence flushes it, and
                    # the ordering that matters is the fence's own.
                    self._publish_pending_remote_buffer_free(handle)
                    return
            with self._control_admission("remote_free"):
                # Admission can block behind a graph callback that acquires a
                # slot/import ref, and another free can have completed while we
                # waited. Re-read both facts at the linearization boundary.
                with self._remote_import_release_mu:
                    if handle.released:
                        if handle not in self._pending_remote_buffer_frees:
                            return
                        if handle._live_slot_refs > 0 or handle._live_import_refs > 0:
                            return
                        self._flush_pending_remote_frees()
                        return
                    if handle._live_slot_refs > 0 or handle._live_import_refs > 0:
                        self._publish_pending_remote_buffer_free(handle)
                        return
                    # Publish logical release before the RPC. A Python async
                    # exception can otherwise land after the remote allocation
                    # is gone but before the handle is marked. FREE_REMOTE_BUFFER
                    # is idempotent by (buffer_id, generation), so a retained
                    # debt may safely be retried if queue retirement is
                    # interrupted after the reply.
                    self._publish_pending_remote_buffer_free(handle)
                    self._flush_pending_remote_frees()

    def remote_copy_to(self, handle: RemoteBufferHandle, host_ptr: Any, nbytes: int, *, offset: int = 0) -> None:
        """Copy ``nbytes`` from host memory into an owner remote buffer.

        Requires an owner handle, not an imported one. ``offset + nbytes`` must
        fall within ``handle.nbytes``.
        """
        with self._operation_lease("remote_copy_to"), self._control_admission("remote_copy_to"):
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
        """Copy ``nbytes`` out of an owner remote buffer into host memory.

        Requires an owner handle, not an imported one. ``offset + nbytes`` must
        fall within ``handle.nbytes``.
        """
        with self._operation_lease("remote_copy_from"), self._control_admission("remote_copy_from"):
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
        transport_profile: str = HOST_TCP_TRANSPORT_PROFILE,
    ) -> RemoteBufferExport:
        """Export a range of an owner buffer so another worker can import it.

        ``nbytes=None`` exports from ``offset`` to the end of the buffer. The
        requested ``access`` must be a subset of the handle's own access flags —
        an export can narrow permissions but never widen them.
        """
        with self._operation_lease("remote_export"), self._control_admission("remote_export"):
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
        transport_profile: str = HOST_TCP_TRANSPORT_PROFILE,
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
        """Import an exported buffer on ``worker`` and return an imported handle.

        ``access`` defaults to the export's own flags. Rejects an export minted
        by a different ``Worker`` and one whose owner buffer has been freed.
        """
        # Argument validation (type / forged / stale) is independent of lifecycle
        # and runs before admission; the lease guards the actual transport.
        if not isinstance(exported, RemoteBufferExport):
            raise TypeError("Worker.remote_import expects a RemoteBufferExport returned by remote_export")
        if exported._worker_owner_id != self._owner_id:
            raise ValueError("Worker.remote_import rejects forged or different Worker RemoteBufferExport values")
        if exported._owner_handle is not None and exported._owner_handle.released:
            raise ValueError("Worker.remote_import rejects stale RemoteBufferExport values for released buffers")
        with self._operation_lease("remote_import"), self._control_admission("remote_import"):
            return self._remote_import_locked(exported, worker=worker, access=access)

    def _remote_import_locked(  # noqa: PLR0912 -- import and rollback phases share one ownership journal
        self, exported: RemoteBufferExport, *, worker: int, access: str | int | None = None
    ) -> RemoteBufferHandle:
        importer_worker_id = int(worker)
        self._require_remote_worker_started(importer_worker_id)
        flags = exported._access_flags if access is None else self._remote_access_flags(access)
        if flags & ~exported._access_flags:
            raise ValueError("Worker.remote_import requested access is not a subset of export access")
        assert self._worker is not None
        native_worker = self._worker
        owner_handle = exported._owner_handle
        owner_ref_token = object() if owner_handle is not None else None
        owner_acquire = _IsolatedCallResult()
        call_result = _IsolatedCallResult()

        def retire_owner_reference(message: str) -> None:
            if owner_handle is None or owner_ref_token is None:
                return
            owner_release = _IsolatedCallResult()

            def release_owner_reference() -> None:
                with self._remote_import_release_mu:
                    owner_handle._release_import_ref(owner_ref_token)

            try:
                _run_isolated_call(
                    owner_release,
                    release_owner_reference,
                    name_prefix="simpler-remote-import-owner-release-",
                )
            except BaseException as exc:  # noqa: BLE001
                if owner_release.error is None:
                    owner_release.error = exc
            if not owner_release.completed:
                cleanup_error = owner_release.error or RuntimeError(
                    "Worker.remote_import: owner reference rollback did not settle"
                )
                self._record_unreclaimable(message, cleanup_error)

        try:
            if owner_handle is not None:

                def acquire_owner_reference() -> object:
                    with self._remote_import_release_mu:
                        return owner_handle._acquire_import_ref(owner_ref_token)

                _run_isolated_call(
                    owner_acquire,
                    acquire_owner_reference,
                    name_prefix="simpler-remote-import-owner-acquire-",
                )
                if owner_acquire.error is not None:
                    raise owner_acquire.error
                if not owner_acquire.completed:
                    raise RuntimeError("Worker.remote_import: owner reference acquisition did not settle")
            _run_isolated_call(
                call_result,
                lambda: native_worker.remote_import(
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
                ),
                name_prefix="simpler-remote-import-",
            )
            if call_result.error is not None:
                raise call_result.error
            fields = call_result.value
            if fields is None:
                raise RuntimeError("Worker.remote_import: remote transport returned no import descriptor")
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
                owner_import_ref_token=owner_ref_token,
            )
        except BaseException as original_error:
            fields = call_result.value
            if fields is not None:
                release_result = _IsolatedCallResult()
                try:
                    _run_isolated_call(
                        release_result,
                        lambda: self._send_remote_release_import_fields(fields),
                        name_prefix="simpler-remote-import-rollback-",
                    )
                except BaseException as exc:  # noqa: BLE001
                    if release_result.error is None:
                        release_result.error = exc
                if not release_result.completed:
                    cleanup_error = release_result.error or RuntimeError(
                        "Worker.remote_import: rollback did not settle"
                    )
                    self._record_unreclaimable(
                        "Worker.remote_import: a committed remote import could not be rolled back; "
                        "the mapping is retained until whole-tree teardown",
                        cleanup_error,
                    )
                else:
                    retire_owner_reference(
                        "Worker.remote_import: the remote mapping was rolled back but its owner reference "
                        "could not be retired"
                    )
            elif call_result.completed:
                self._record_unreclaimable(
                    "Worker.remote_import: the transport completed without publishing an import descriptor; "
                    "remote ownership is retained until whole-tree teardown",
                    original_error,
                )
            elif call_result.error is None:
                retire_owner_reference(
                    "Worker.remote_import: the transport failed before publishing an import descriptor, but "
                    "its owner reference could not be retired"
                )
            else:
                self._record_unreclaimable(
                    "Worker.remote_import: import ownership is ambiguous after a failed transport call; "
                    "the owner reference is retained until whole-tree teardown",
                    call_result.error,
                )
            raise original_error

    def _publish_pending_remote_import_release(
        self,
        handle: RemoteBufferHandle,
        state: _PendingRemoteImportReleaseState,
        pending_error: BaseException | None = None,
    ) -> None:
        """Publish release ownership before any import-release RPC can start."""
        with self._remote_import_release_mu:
            try:
                existing = self._pending_remote_import_release_states.get(handle)
                if existing is None:
                    self._pending_remote_import_release_states[handle] = state
                if handle not in self._pending_remote_import_releases:
                    self._pending_remote_import_releases.append(handle)
                handle._mark_released()
            except BaseException as exc:  # noqa: BLE001
                self._publish_pending_remote_import_release(handle, state, pending_error or exc)
                return
            if pending_error is not None:
                raise pending_error

    def _publish_pending_remote_buffer_free(
        self, handle: RemoteBufferHandle, pending_error: BaseException | None = None
    ) -> None:
        """Atomically publish an owner-free debt and logical release."""
        with self._remote_import_release_mu:
            try:
                if handle not in self._pending_remote_buffer_frees:
                    self._pending_remote_buffer_frees.append(handle)
                handle._mark_released()
            except BaseException as exc:  # noqa: BLE001
                self._publish_pending_remote_buffer_free(handle, pending_error or exc)
                return
            if pending_error is not None:
                raise pending_error

    def remote_release_import(self, handle: RemoteBufferHandle) -> None:
        """Release an imported remote handle.

        Idempotent, and rejects owner handles (use ``remote_free``). Deferred
        while a live task slot still references it. Releasing the last import of
        a buffer whose owner already called ``remote_free`` completes that free.
        """
        if not isinstance(handle, RemoteBufferHandle):
            raise TypeError("expected a RemoteBufferHandle returned by Worker.remote_import")
        if not handle.is_imported:
            raise ValueError("Worker.remote_release_import expects an imported remote handle")
        if handle.released:
            return
        # Public admission: READY-only + drained (the private _send_* transport
        # accepts CLOSED for teardown, so fence admission here).
        with self._operation_lease("remote_release_import"):
            with self._remote_import_release_mu:
                if handle.released:
                    return
                if handle._live_slot_refs > 0:
                    # Deferred: see remote_free — nothing is sent from here.
                    self._publish_pending_remote_import_release(
                        handle,
                        _PendingRemoteImportReleaseState(
                            owner_ref=handle._owner_handle_ref,
                            owner_ref_token=handle._owner_import_ref_token,
                        ),
                    )
                    return
            with self._control_admission("remote_release_import"):
                with self._remote_import_release_mu:
                    if handle.released:
                        return
                    if handle._live_slot_refs > 0:
                        self._publish_pending_remote_import_release(
                            handle,
                            _PendingRemoteImportReleaseState(
                                owner_ref=handle._owner_handle_ref,
                                owner_ref_token=handle._owner_import_ref_token,
                            ),
                        )
                        return
                    self._publish_pending_remote_import_release(
                        handle,
                        _PendingRemoteImportReleaseState(
                            owner_ref=handle._owner_handle_ref,
                            owner_ref_token=handle._owner_import_ref_token,
                        ),
                    )
                    self._flush_pending_remote_frees()

    @staticmethod
    def _remote_sidecar_handles(remote_sidecars: Any) -> list[RemoteBufferHandle]:
        handles: list[RemoteBufferHandle] = []
        for remote_sidecar in remote_sidecars:
            if remote_sidecar is None:
                continue
            for tensor_sidecar in getattr(remote_sidecar, "tensors", ()):
                if tensor_sidecar is None or not getattr(tensor_sidecar, "present", False):
                    continue
                handle = getattr(tensor_sidecar, "handle", None)
                if handle is None:
                    continue
                if not isinstance(handle, RemoteBufferHandle):
                    raise TypeError("remote sidecar handle must be a RemoteBufferHandle")
                handles.append(handle)
        return handles

    def _capture_remote_sidecar_refs(self, remote_sidecar: Any) -> list[_RemoteSlotRefClaim]:
        captured: list[_RemoteSlotRefClaim] = []
        try:
            for handle in self._remote_sidecar_handles((remote_sidecar,)):
                claim = _RemoteSlotRefClaim(handle)
                captured.append(claim)
                with self._remote_import_release_mu:
                    handle._acquire_slot_ref(claim.token)
        except BaseException:
            self._release_remote_slot_refs(captured)
            raise
        return captured

    def _adopt_remote_slot_refs(self, handles: list[Any]) -> None:
        resources = self._building_run_resources
        if resources is None:
            self._active_remote_slot_refs.extend(handles)
        else:
            resources.remote_slot_refs.extend(handles)
            # Releasing a remote slot ref is an RPC to its owner, so it is
            # cleanup that reaches off this process.
            if handles:
                resources.requires_ordered_cleanup = True

    def _adopt_remote_sidecar_refs(self, remote_sidecars: Any) -> None:
        handles = self._remote_sidecar_handles(remote_sidecars)
        if not handles:
            return
        resources = self._building_run_resources
        refs = self._active_remote_slot_refs if resources is None else resources.remote_slot_refs
        if resources is not None:
            resources.requires_ordered_cleanup = True
        for handle in handles:
            claim = _RemoteSlotRefClaim(handle)
            refs.append(claim)
            with self._remote_import_release_mu:
                handle._acquire_slot_ref(claim.token)

    def _release_remote_slot_refs(self, refs: list[Any]) -> None:
        while refs:
            # Preserve acquisition order. If a release is interrupted, later
            # claims must remain owned rather than being retired ahead of the
            # ambiguous claim. Token releases are idempotent across the
            # release/delete boundary.
            ref = refs[0]
            with self._remote_import_release_mu:
                if isinstance(ref, _RemoteSlotRefClaim):
                    ref.handle._release_slot_ref(ref.token)
                else:
                    ref._release_slot_ref()
            del refs[0]

    def _release_active_remote_slot_refs(self, resources: _RunResources | None = None) -> None:
        if resources is None:
            refs = self._active_remote_slot_refs
        else:
            refs = resources.remote_slot_refs
        self._release_remote_slot_refs(refs)

    def _flush_pending_remote_frees(self) -> None:  # noqa: PLR0912
        errors: list[str] = []
        first_async_error: BaseException | None = None

        def remember_error(exc: BaseException, context: str) -> None:
            nonlocal first_async_error
            if isinstance(exc, Exception):
                errors.append(f"{context}: {exc}")
            elif first_async_error is None:
                first_async_error = exc

        with self._remote_import_release_mu:
            release_states = self._pending_remote_import_release_states
            pending_imports = self._pending_remote_import_releases
            self._pending_remote_import_releases = []
            for handle in release_states:
                if handle not in pending_imports:
                    pending_imports.append(handle)
            remaining_imports: list[RemoteBufferHandle] = []
            for handle in pending_imports:
                context = f"release_import worker_id={handle.worker_id} import_id={handle.import_id}"
                state = release_states.get(handle)
                if state is None:
                    state = _PendingRemoteImportReleaseState(
                        owner_ref=handle._owner_handle_ref,
                        owner_ref_token=handle._owner_import_ref_token,
                    )
                    release_states[handle] = state
                if handle._live_slot_refs > 0:
                    remaining_imports.append(handle)
                    continue
                if state.error is not None:
                    remaining_imports.append(handle)
                    remember_error(state.error, context)
                    continue

                rpc_boundary_error: BaseException | None = None
                if not state.rpc_complete:
                    rpc_result = _IsolatedCallResult()
                    try:
                        _run_isolated_call(
                            rpc_result,
                            lambda: self._send_remote_release_import(handle),
                            name_prefix="simpler-remote-import-release-",
                            after_success=lambda: setattr(state, "rpc_complete", True),
                        )
                    except BaseException as exc:  # noqa: BLE001
                        rpc_boundary_error = exc
                    if rpc_result.error is not None:
                        rpc_boundary_error = rpc_result.error
                    if rpc_boundary_error is not None:
                        remember_error(rpc_boundary_error, context)
                        if not state.rpc_complete:
                            state.error = rpc_boundary_error
                            self._record_unreclaimable(
                                "Worker.remote_release_import: the release RPC did not publish completion; "
                                "the import is retained until whole-tree teardown",
                                rpc_boundary_error,
                            )
                            remaining_imports.append(handle)
                            continue

                if state.owner_ref is None:
                    state.owner_release_complete = True
                elif not state.owner_release_complete:
                    owner_boundary_error: BaseException | None = None
                    owner_result = _IsolatedCallResult()
                    owner_ref = state.owner_ref
                    owner_ref_token = state.owner_ref_token
                    owner_release = (
                        owner_ref._release_import_ref
                        if owner_ref_token is None
                        else lambda: owner_ref._release_import_ref(owner_ref_token)
                    )
                    try:
                        _run_isolated_call(
                            owner_result,
                            owner_release,
                            name_prefix="simpler-remote-import-owner-release-",
                            after_success=lambda: setattr(state, "owner_release_complete", True),
                        )
                    except BaseException as exc:  # noqa: BLE001
                        owner_boundary_error = exc
                    if owner_result.error is not None:
                        owner_boundary_error = owner_result.error
                    if owner_boundary_error is not None:
                        remember_error(owner_boundary_error, context)
                        if not state.owner_release_complete:
                            state.error = owner_boundary_error
                            self._record_unreclaimable(
                                "Worker.remote_release_import: the release RPC completed but its owner "
                                "reference could not be retired",
                                owner_boundary_error,
                            )
                            remaining_imports.append(handle)
                            continue

                while handle._owner_import_ref_token is not None:
                    try:
                        handle._owner_import_ref_token = None
                    except BaseException as exc:  # noqa: BLE001, PERF203
                        remember_error(exc, context)
                while handle._owner_handle_ref is not None:
                    try:
                        handle._owner_handle_ref = None
                    except BaseException as exc:  # noqa: BLE001, PERF203
                        remember_error(exc, context)
                release_states.pop(handle, None)
            self._pending_remote_import_releases.extend(remaining_imports)

        with self._remote_import_release_mu:
            # Keep the durable queue intact while RPCs are in flight. If an
            # async exception escapes between a successful idempotent free and
            # retirement below, the old debt is replayed rather than lost.
            pending = list(self._pending_remote_buffer_frees)
            remaining: list[RemoteBufferHandle] = []
            for handle in pending:
                if handle._live_slot_refs > 0 or handle._live_import_refs > 0:
                    remaining.append(handle)
                    continue
                try:
                    self._send_remote_free(handle)
                except BaseException as exc:  # noqa: BLE001
                    remaining.append(handle)
                    if isinstance(exc, Exception):
                        errors.append(f"free worker_id={handle.worker_id} buffer_id={handle._buffer_id}: {exc}")
                    elif first_async_error is None:
                        first_async_error = exc
                    continue
            self._pending_remote_buffer_frees[:] = remaining
        if first_async_error is not None:
            raise first_async_error
        if errors:
            # Every handle is attempted and the ones that failed stay pending,
            # but the failure is the caller's answer: this runs inside the run
            # fence's cleanup, and swallowing it would publish a run as cleanly
            # finished while a remote allocation it owns is still held.
            raise RuntimeError(
                f"Worker._flush_pending_remote_frees: {len(errors)} deferred remote buffer cleanup(s) "
                f"failed and remain owed. First error: {errors[0]}"
            )

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
        callable registration / host-buffer / remote-memory)."""
        tid = threading.get_ident()
        with self._hierarchical_start_cv:
            self._consume_worker_host_mapped_cleanup_error_locked(api)
            if self._lifecycle is not _Lifecycle.READY:
                raise RuntimeError(f"Worker.{api}: requires an initialized (READY) worker") from self._startup_error
            if self._ordered_cleanup_error is not None:
                raise RuntimeError(
                    f"Worker.{api}: a prior run's ordered cleanup failed, so this worker's device state is "
                    "unreclaimed and no further work is admitted; close() it"
                ) from self._ordered_cleanup_error
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

    def _invalidate_endpoint_registry(self) -> None:
        self._endpoint_registry = None
        self._region_access_service = None
        self._endpoint_registry_epoch += 1

    def _require_ready_for_region_planning(self, api: str = "region planning") -> None:
        """Admit only a Worker that owns a control subtree to declare a region.

        This is a control-tree visibility rule, not a capability decision: level says who may
        *declare* a region over its own subtree, the same way it says an L4 submits only to its L3
        children. It never decides whether two endpoints can share memory — that is deployment,
        interconnect, backing and live attachment, none of which appear here. A level < 3 Worker has
        no subtree to enumerate, so it has no members to name; it can still be a *member* of a
        region its parent declares.

        Lifecycle admission is deliberately absent: `_operation_lease` is the single fence for
        READY, cleanup errors and the close race.
        """
        if self.level < 3:
            raise RuntimeError(f"Worker.{api}: region planning requires a level >= 3 Worker")

    def _endpoint_topology_snapshot(self) -> _EndpointTopologySnapshot:
        self._require_ready_for_region_planning("_endpoint_topology_snapshot")
        root_level = int(self.level)
        root_path = _format_worker_path(root_level)
        entries: list[_EndpointTopologyEntry] = []
        self._append_endpoint_topology(entries, self, root_path, "local", include_self=True)
        entries.sort(key=lambda entry: self._endpoint_topology_sort_key(entry, root_level))
        return _EndpointTopologySnapshot(
            root_level=root_level,
            session_instance_id=self._owner_instance_id,
            entries=tuple(entries),
        )

    def _append_endpoint_topology(
        self,
        entries: list[_EndpointTopologyEntry],
        worker: Worker,
        path: str,
        node_identity: str,
        *,
        include_self: bool,
    ) -> None:
        if include_self:
            # The Worker's own buffer-owner nonce rides along, so the registry can resolve a
            # BufferDescriptor back to the endpoint that minted it. A remote child's nonce is
            # minted in its own process and is deliberately left absent below.
            entries.append(_EndpointTopologyEntry(path, HOST_CPU, node_identity, worker._owner_instance_id))
        if int(worker.level) == 3:
            self._append_device_endpoint_topology(entries, path, worker._config.get("device_ids", ()), node_identity)
        for child_index, child in zip(worker._next_level_worker_ids, worker._next_level_workers):
            child_path = _format_worker_path(int(child.level), parent_path=path, index=int(child_index))
            self._append_endpoint_topology(entries, child, child_path, node_identity, include_self=True)
        for child_index, spec in zip(worker._remote_worker_ids, worker._remote_worker_specs):
            remote_path = _format_worker_path(3, parent_path=path, index=int(child_index))
            remote_node_identity = self._node_identity_from_remote_endpoint(spec.endpoint)
            entries.append(_EndpointTopologyEntry(remote_path, HOST_CPU, remote_node_identity))
            self._append_device_endpoint_topology(entries, remote_path, spec.device_ids, remote_node_identity)
        mpi_groups = {group.group_id: group for group in worker._mpi_l3_groups}
        for child_index in worker._mpi_worker_ids:
            rank = worker._mpi_rank_by_worker_id[int(child_index)]
            group = mpi_groups[rank.group_id]
            mpi_node_identity = _normalize_node_identity(group.spec.hosts[rank.rank])
            mpi_path = _format_worker_path(3, parent_path=path, index=int(child_index))
            entries.append(_EndpointTopologyEntry(mpi_path, HOST_CPU, mpi_node_identity))
            self._append_device_endpoint_topology(entries, mpi_path, rank.spec.device_ids, mpi_node_identity)

    def _append_device_endpoint_topology(
        self,
        entries: list[_EndpointTopologyEntry],
        path_to_l3: str,
        device_ids,
        node_identity: str,
    ) -> None:
        for child_index, _device_id in enumerate(tuple(device_ids)):
            device_path = _format_worker_path(2, parent_path=path_to_l3, index=child_index)
            entries.append(_EndpointTopologyEntry(device_path, DEVICE_AICORE, node_identity))
            entries.append(_EndpointTopologyEntry(device_path, DEVICE_AICPU, node_identity))

    def _node_identity_from_remote_endpoint(self, endpoint: str) -> str:
        host, _port = self._parse_remote_endpoint(endpoint)
        return _normalize_node_identity(host)

    def _endpoint_topology_sort_key(self, entry: _EndpointTopologyEntry, root_level: int):
        deployment_order = {HOST_CPU: 0, DEVICE_AICORE: 1, DEVICE_AICPU: 2}
        return (parse_endpoint_path(entry.path, root_level=root_level).sort_key, deployment_order[entry.deployment])

    def _get_endpoint_registry(self) -> EndpointRegistry:
        self._require_ready_for_region_planning("_get_endpoint_registry")
        registry = self._endpoint_registry
        if registry is None:
            registry = EndpointRegistry.from_snapshot(
                self._endpoint_topology_snapshot(), registry_epoch=self._endpoint_registry_epoch
            )
            self._endpoint_registry = registry
        return registry

    def _get_region_access_service(self) -> RegionAccessService:
        service = self._region_access_service
        if service is None:
            service = DefaultRegionAccessService()
            self._region_access_service = service
        return service

    def _resolve_region_spec(self, members, topology: SingleOwner):
        self._require_ready_for_region_planning("_resolve_region_spec")
        with self._operation_lease("_resolve_region_spec"):
            return self._get_endpoint_registry().resolve_region_spec(members, topology)

    def _plan_region(
        self, members, topology: SingleOwner, layout_summary: RegionLayoutSpec
    ) -> BackendPlan | UnsupportedRegionPlan:
        self._require_ready_for_region_planning("_plan_region")
        with self._operation_lease("_plan_region"):
            registry = self._get_endpoint_registry()
            resolved = registry.resolve_region_spec(members, topology)
            resolver = BackendResolver(registry, self._get_region_access_service())
            return resolver.plan(resolved, layout_summary)

    def _materialize_region_instance(
        self, members, topology: SingleOwner, layout_summary: RegionLayoutSpec
    ) -> RegionInstance:
        self._require_ready_for_region_planning("_materialize_region_instance")
        with (
            self._operation_lease("_materialize_region_instance"),
            self._control_admission("materialize_region_instance"),
        ):
            registry = self._get_endpoint_registry()
            resolved = registry.resolve_region_spec(members, topology)
            resolver = BackendResolver(registry, self._get_region_access_service())
            plan = resolver.plan(resolved, layout_summary)
            return materialize_region_instance(
                MaterializationContext(worker=self, registry=registry, plan=plan, layout=layout_summary)
            )

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

        A post-init dynamic register re-validates eligibility against the
        frozen topology (``_eligible_target_need``), same as init().
        """
        if isinstance(target, RemoteCallable) and self.level < 4:
            raise TypeError("Worker.register(RemoteCallable): remote L3 dispatch requires a level >= 4 parent")
        if self.level == 2 and not isinstance(target, ChipCallable):
            raise TypeError("Worker.register: level 2 only supports ChipCallable targets")
        reg = _build_callable_registration(self, target, workers=workers)
        if isinstance(target, RemoteCallable):
            if not self._remote_worker_specs and not self._mpi_l3_groups:
                raise RuntimeError("Worker.register(RemoteCallable): add at least one remote worker first")
            remote_worker_ids = self._remote_like_worker_ids()
            for worker_id in reg.eligible_worker_ids:
                if worker_id not in remote_worker_ids:
                    raise ValueError(
                        "Worker.register(RemoteCallable): workers must name remote worker ids returned by "
                        "add_remote_worker or add_mpirun_worker_group"
                    )
            # Linearize against the startup epoch exactly like the local path: a
            # register that races an in-progress init() waits for it, then a
            # pre-start registration lands in the snapshot while a post-READY one
            # goes through the remote prepare/commit control path.
            handle = self._register_into_snapshot_or_wait(reg)
            if handle is not None:
                return handle
            need = self._eligible_target_need(reg.target_namespace, reg.eligible_worker_ids)
            if need is not None:
                raise ValueError(
                    f"Worker.register(): {reg.target_namespace} callable has no eligible dispatch target (needs {need})"
                )
            # Post-start broadcast touches the live tree; hold a lease so close()
            # drains it before teardown (re-checks READY, closing the
            # gate-then-teardown race).
            with self._operation_lease("register"):
                return self._post_start_register_remote(reg)
        if self.level >= 3:
            handle = self._register_into_snapshot_or_wait(reg)
            if handle is not None:
                return handle
            need = self._eligible_target_need(reg.target_namespace, reg.eligible_worker_ids)
            if need is not None:
                raise ValueError(
                    f"Worker.register(): {reg.target_namespace} callable has no eligible dispatch target (needs {need})"
                )
            # Post-start publication touches the live tree; hold a lease across
            # the whole transaction, publication included, so close() drains it
            # before teardown (re-checks READY, closing the gate-then-teardown
            # race).
            with self._operation_lease("register"):
                if not isinstance(target, ChipCallable):
                    return self._post_start_register_python(reg)
                return self._post_start_register_chip(reg, target)

        # L2 installs a NEW registration while holding the lifecycle lock, so
        # init cannot start and finish its pre-warm snapshot between the epoch
        # check and registry publication. A post-start registration always goes
        # through the READY-only lease; it never falls back to NEW semantics if
        # close claims the epoch after this check.
        with self._hierarchical_start_cv:
            self._wait_out_init_locked("register")
            if self._lifecycle is _Lifecycle.NEW:
                with self._registry_lock:
                    handle, _is_new = self._install_registration_locked(reg)
                return handle
        with self._operation_lease("register"):
            return self._post_start_register_l2(reg, target)

    def _post_start_register_chip(self, reg: _CallableRegistration, target: ChipCallable) -> CallableHandle:
        """Publish a post-READY L3+ ChipCallable and broadcast it to the chip /
        next-level children via C++ after Host-side slot allocation.

        Caller holds an ``_operation_lease``, so publication and broadcast are
        one transaction: a close() either drains the whole thing or is refused
        admission before anything is published. The slot is target-private; task
        dispatches carry only ``handle.digest``.
        """
        with self._registry_lock:
            handle, is_new = self._install_registration_locked(reg)
        try:
            self._post_init_register(target, handle.digest, is_new=is_new)
        except Exception:
            with self._registry_lock:
                self._rollback_handle_locked(handle)
            raise
        return handle

    def _post_start_register_l2(self, reg: _CallableRegistration, target: ChipCallable) -> CallableHandle:
        """Publish a post-READY L2 registration and pre-warm its device slot, so
        the very first ``run(handle, …)`` is a clean cache hit.

        Caller holds an ``_operation_lease``. L2 has no child subtree to
        broadcast to, so the lease covers only publication and the local
        pre-warm.
        """
        with self._registry_lock:
            handle, is_new = self._install_registration_locked(reg)
        if not is_new:
            return handle
        assert self._chip_worker is not None
        with self._registry_lock:
            slot_id = self._identity_registry[handle.digest].slot_id
        try:
            self._chip_worker._register_callable_at_slot(slot_id, target)
        except Exception:
            with self._registry_lock:
                self._rollback_handle_locked(handle)
            raise
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
        with self._hierarchical_start_cv:
            self._wait_out_init_locked("unregister")
            if self._lifecycle is not _Lifecycle.NEW:
                return False
            with self._registry_lock:
                handle_id, digest, state = self._coerce_handle_state(handle_or_slot)
                if state.target_namespace == "REMOTE_TASK_DISPATCHER":
                    if digest in self._pending_remote_unregister_hashids:
                        raise KeyError("UNREGISTER_TOMBSTONE_ACTIVE: remote callable handle already pending unregister")
                    self._rollback_handle_locked(handle_or_slot)
                    return True
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
        if self._pre_start_unregister_if_needed(handle_or_slot):
            return
        # Every post-start path takes the READY-only lease before touching the
        # registry. The lease rechecks lifecycle after the pre-start decision,
        # so a close claim in between rejects the operation rather than changing
        # it into a registry-only mutation on a CLOSED worker.
        with self._operation_lease("unregister"):
            if (
                isinstance(handle_or_slot, CallableHandle)
                and handle_or_slot.target_namespace == "REMOTE_TASK_DISPATCHER"
            ):
                self._unregister_remote_handle(handle_or_slot)
            else:
                self._unregister_handle(handle_or_slot)

    def _unregister_handle(self, handle_or_slot) -> None:
        """Pop a READY worker's handle and propagate cleanup to its target.

        Caller holds an operation lease that pins the worker in READY for the
        registry mutation and the complete child-facing cleanup.
        """
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
            should_broadcast_decrement = self.level >= 3
            if state.ref_count > 0 and not should_broadcast_decrement:
                return
            target = self._callable_registry[cid]
            remove_target = state.ref_count <= 0
            if should_broadcast_decrement:
                self._pending_unregister_cids.add(cid)
                if state.ref_count > 0:
                    remove_target = False
            elif self.level == 2:
                assert self._chip_worker is not None
                self._chip_worker._unregister_slot(cid)
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
        """Drop a READY worker's remote handle and release its dispatcher.

        Caller holds an operation lease, so the registry mutation and every
        remote cleanup RPC finish before close can tear down the transport.
        """
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

    def _broadcast_import_release(self, identity: CanonicalIdentity) -> None:
        """Broadcast _CTRL_IMPORT_RELEASE to every NEXT_LEVEL child (chip or nested Worker) and
        every SUB child, via the generic control channel _CTRL_PY_REGISTER et al. already use.

        A no-op before ``init()`` (``self._worker`` unset) — a Worker that never started has no
        forked children to reach regardless of what ``_python_worker_types()`` would report.
        Otherwise unconditional on both target lists: a Worker with no children of one kind sends
        to an empty list, which the underlying broadcast already no-ops on — no need to duplicate
        ``_python_worker_types()``'s gating here (that helper only counts nested-Worker NEXT_LEVEL
        children, not chip children, so it would wrongly skip the chip-only case).

        Best-effort, mirroring ``_broadcast_unregister``: a child that never materialized
        ``identity`` has nothing to drop, and a slow or dead child must not block or fail
        ``release_buffer()`` — the Buffer is already closed on the owner side by the time this
        runs, so this call is pure cleanup, not a correctness gate.
        """
        if self._worker is None:
            return
        errors = self._broadcast_py_control(
            [WorkerType.NEXT_LEVEL, WorkerType.SUB],
            _CTRL_IMPORT_RELEASE,
            digest=_pack_identity_wire(identity),
            strict=False,
        )
        if errors:
            sys.stderr.write(
                f"Worker.release_buffer(identity={identity}): {len(errors)} children reported errors "
                f"(continuing best-effort). First error: {errors[0]}\n"
            )
            sys.stderr.flush()

    def _release_import_recursive(self, identity: CanonicalIdentity) -> None:
        """Drop ``identity`` from this Worker's own same-process caches, then forward one
        more hop down this Worker's own children — same shape as ``_unregister_child_digest``'s
        recursive forward for callable cleanup, since a NEXT_LEVEL child may itself have
        materialized ``identity`` further down its own tree (chip/SUB leaves, or its own
        NEXT_LEVEL children in turn).

        Two caches name a released identity here, not one: the import cache holds a mapping of it,
        and ``_reexport_by_source`` holds the forwarding handle built from it. A retained re-export
        outlives its backing, and its ``to_descriptor()`` keeps answering — so a later forward of
        the same identity would hand a child a descriptor for a name the owner has unlinked.
        """
        if self._chip_import_registry is not None:
            self._chip_import_registry.unregister(identity)
        self._reexport_by_source.pop(identity, None)
        self._broadcast_import_release(identity)

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
        if worker is self:
            raise ValueError("Worker.add_worker() cannot add a Worker to itself")
        # Claim both lifecycle locks in a deterministic order. The old child
        # check happened before the parent lock, so child.init() could win in
        # between and publish an already-started child into a NEW parent.
        first, second = sorted((self._hierarchical_start_cv, worker._hierarchical_start_cv), key=id)
        with first, second:
            if self._lifecycle is not _Lifecycle.NEW:
                raise RuntimeError("Worker.add_worker() must be called before init()")
            if self._topology_parent is not None:
                raise RuntimeError("Worker.add_worker() cannot mutate a Worker already attached as a child")
            if worker._lifecycle is not _Lifecycle.NEW:
                raise RuntimeError("Child worker must be NEW (not started/failed/closed) before add_worker()")
            if worker._topology_parent is not None:
                raise RuntimeError("Child worker is already attached to another parent")
            worker_id = self._allocate_next_level_worker_id()
            worker._topology_parent = self
            self._next_level_workers.append(worker)
            self._next_level_worker_ids.append(worker_id)
            return worker_id

    def _assign_shm_namespace(self, used_prefixes: set[str] | None = None) -> set[str]:
        """Assign one root-visible mailbox token to every Worker in this tree."""
        if used_prefixes is None:
            used_prefixes = set()
            shm_dir = "/dev/shm"
            if os.path.isdir(shm_dir):
                for entry in os.scandir(shm_dir):
                    if entry.name.startswith("sp-") and len(entry.name) >= 12:
                        used_prefixes.add(entry.name[3:11])
        while True:
            token = uuid.uuid4().hex
            if token[:8] not in used_prefixes:
                break
        used_prefixes.add(token[:8])
        self._shm_token = token
        tokens = {self._shm_token}
        for child in self._next_level_workers:
            tokens.update(child._assign_shm_namespace(used_prefixes))
        self._shm_tree_tokens = tokens
        return tokens

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

        Applied at init by ``_validate_eligible_targets`` and on every
        post-init dynamic ``register`` path.
        """
        if self.level < 3:
            return None
        if namespace == "LOCAL_PYTHON":
            has_python_child = self._config.get("num_sub_workers", 0) > 0 or bool(self._next_level_workers)
            return None if has_python_child else "a SUB or next-level child"
        if namespace == "LOCAL_CHIP":
            # A chip target need not be this worker's own: an L4 parent carries
            # no device_ids and reaches its chips through a next-level child or
            # a remote spec, so the search walks the frozen topology.
            def has_chip_target(worker: Worker) -> bool:
                if worker._config.get("device_ids"):
                    return True
                if any(spec.device_ids for spec in worker._remote_worker_specs):
                    return True
                if any(rank.spec.device_ids for rank in worker._mpi_rank_by_worker_id.values()):
                    return True
                return any(has_chip_target(child) for child in worker._next_level_workers)

            return None if has_chip_target(self) else "a chip device (device_ids)"
        if namespace == "REMOTE_TASK_DISPATCHER":
            has_remote_workers = self._remote_like_worker_ids()
            ok = bool(has_remote_workers) and set(eligible_worker_ids) <= has_remote_workers
            return None if ok else "its named remote worker(s) (add_remote_worker/add_mpirun_worker_group)"
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

    def init(  # noqa: PLR0912, PLR0915
        self, prewarm_config: CallConfig | None = None, *, _startup_deadline: float | None = None
    ) -> None:
        """Initialize the worker and bring its whole subtree to READY.

        For an L3+ worker ``init`` is the single startup submission point: it
        forks every local child (sub / chip / next-level), waits for the whole
        subtree — recursively, for L4+ — to publish INIT_READY, activates any
        remote L3 sessions, starts the C++ scheduler, and only then publishes
        READY in one atomic commit. It returns with the tree ready to run, or
        raises after a bounded rollback that reaps the children it forked
        best-effort (a child wedged in native code past the deadline may be left
        behind — see the deferred un-reaped-child / nested-shm items).
        ``run`` / ``create_buffer`` / the remote register/memory APIs never
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
            if self._topology_parent is not None and _startup_deadline is None:
                raise RuntimeError("Worker.init(): a Worker attached as a child is initialized by its parent")
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
            self._cancel_token = False
            if _startup_deadline is None:
                self._assign_shm_namespace()
            self._lifecycle = _Lifecycle.INITIALIZING
            # Generated after this Worker's own fork: a next-level child's init() runs only inside
            # the process that forked to host it (see _start_hierarchical), so this nonce is never
            # older than the incarnation it names. Buffer and endpoint identity share this mint
            # point (see the Owner-side Buffer state comment in __init__).
            self._owner_instance_id: bytes = mint_owner_instance_id()
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
                # Final root-deadline gate, in the same critical section as the
                # commit: a thread descheduled between startup and here cannot
                # publish READY past the single root startup deadline. Applied to
                # every hierarchical worker, not just those with direct remote
                # sessions — a local child may have remote descendants whose
                # startup this deadline also bounds.
                if self.level >= 3 and time.monotonic() >= self._startup_deadline:
                    raise RuntimeError("hierarchical startup: startup deadline exceeded before READY")
                if self._cancel_token:
                    raise InitCancelled("init cancelled by close() before READY commit")
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
                    # Only an INITIALIZING epoch commits FAILED. FAILED is only
                    # written by the init thread. CLOSED is absorbing.
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
        # first run() with matching sizing skips the cold arena build. enable_sdma
        # opts this Worker into async-DMA (SDMA) workspace provisioning at init;
        # off by default so ordinary Workers create no SDMA streams.
        self._chip_worker.init(
            device_id,
            binaries,
            prewarm_config=self._prewarm_config,
            enable_sdma=bool(self._config.get("enable_sdma", False)),
        )

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
        if self._remote_worker_specs or self._mpi_l3_groups:
            self._remote_session_timeout_s()

        # 1. Allocate sub-worker mailboxes (unified layout, MAILBOX_SIZE each).
        for i in range(n_sub):
            shm = SharedMemory(create=True, size=MAILBOX_SIZE, name=_shm_name(self._shm_token, f"sub-{i}"))
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
            for i, _dev_id in enumerate(device_ids):
                shm = SharedMemory(create=True, size=MAILBOX_SIZE, name=_shm_name(self._shm_token, f"chip-{i}"))
                assert shm.buf is not None
                _mailbox_store_i32(_buffer_field_addr(shm.buf, _OFF_STATE), _IDLE)
                self._chip_shms.append(shm)

        # 3. Allocate next-level Worker child mailboxes (L4+ only).
        for i, _inner in enumerate(self._next_level_workers):
            shm = SharedMemory(create=True, size=MAILBOX_SIZE, name=_shm_name(self._shm_token, f"next-{i}"))
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
        if len(self._remote_worker_ids) != len(self._remote_worker_specs):
            raise ValueError("remote worker ids/specs length mismatch")
        session_timeout = self._remote_session_timeout_s()
        for worker_id, spec in zip(self._remote_worker_ids, self._remote_worker_specs):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("remote L3 session activation: startup deadline exceeded")
            session_id = uuid.uuid4().int & ((1 << 63) - 1)
            if session_id == 0:
                session_id = 1
            # The handshake blocks until the remote subtree is READY; the whole
            # open derives its per-op remaining from the shared root deadline.
            try:
                session = self._open_remote_session(
                    spec=spec,
                    worker_id=worker_id,
                    session_id=session_id,
                    deadline=deadline,
                )
            except TimeoutError as exc:
                raise TimeoutError(f"remote L3 session open failed for worker {worker_id}: {exc}") from exc
            except Exception as exc:
                raise RuntimeError(
                    f"remote L3 session open failed for worker {worker_id}: {type(exc).__name__}: {exc}"
                ) from exc
            self._remote_sessions.append(session)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("remote L3 endpoint attach: startup deadline exceeded")
            assert self._worker is not None
            # attach_timeout bounds the command/health connect + HELLO read by the
            # remaining startup budget; runtime_timeout is the full runtime command
            # budget, never clamped by leftover startup time.
            try:
                self._worker.add_remote_l3_socket(
                    session.worker_id,
                    session.session_id,
                    spec.comm_profile,
                    session.command_host,
                    session.command_port,
                    session.health_host,
                    session.health_port,
                    remaining,
                    session_timeout,
                )
            except TimeoutError as exc:
                raise TimeoutError(f"remote L3 endpoint attach failed for worker {worker_id}: {exc}") from exc
            except Exception as exc:
                raise RuntimeError(
                    f"remote L3 endpoint attach failed for worker {worker_id}: {type(exc).__name__}: {exc}"
                ) from exc
        # Attach may have consumed the last slice of the budget; a final root
        # deadline check keeps a just-over-budget attach from committing READY.
        if time.monotonic() >= deadline:
            raise RuntimeError("remote L3 activation: startup deadline exceeded after attach")

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
        direct_chip_pipeline_depth = PTO_PIPELINE_MAX_DEPTH
        chip_depths: list[int] = []
        global_nodes = self._resolved_global_nodes() if self.level >= 4 else {}

        # Freeze the startup registry snapshot. init() already holds the epoch in
        # the INITIALIZING state, so a concurrent register/unregister is blocked
        # on the lifecycle condition and cannot slip a mutation in after this point.
        with self._registry_lock:
            identity_snapshot = [
                (digest, state.target, state.ref_count, state.kind, state.target_namespace)
                for digest, state in self._identity_registry.items()
            ]

        # Seed this process's logger before the first fork: the spans its own
        # scheduler emits obey the Python logger level, and every child inherits
        # the state. Every level needs this — `init()` rejects device_ids above
        # L3, so a network1 process owns no chips yet still drives next-level Workers
        # and emits their spans. A chip child re-seeds its inherited state before
        # binding the logger copies embedded in the runtime modules it loads.
        chip_log_level = _simpler_log.get_current_config()
        _initialize_host_log(chip_log_level)

        # Bind the level word this process's host-scheduler spans lead with. The
        # C++ emit sites in Orchestrator / WorkerThread are level-agnostic — the
        # same code drives next-level children at every level above the chip — so
        # the word is a property of the process. Resolved once here and pushed, so
        # there is a single derivation rather than one on each side.
        #
        # The first binding in a process wins, because a SpanScope holds the name
        # pointer it was given. So a process that inits Workers at two levels keeps
        # the first level's word, and self._host_span_prefix is whatever is really
        # bound — never what this Worker asked for. Say so instead of letting the
        # mismatch show up as mislabelled spans much later.
        requested = _span_prefix(self.level)
        self._host_span_prefix = _set_host_span_level_prefix(requested)
        if self._host_span_prefix != requested:
            logging.getLogger("simpler").warning(
                "host-scheduler spans in this process are labelled %r, not %r: one process carries one "
                "span vocabulary and an earlier Worker bound it first",
                self._host_span_prefix,
                requested,
            )

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
                            self._owner_instance_id,
                            log_level=chip_log_level,
                            platform=str(self._config["platform"]),
                            runtime=str(self._config["runtime"]),
                            prewarm_config=self._prewarm_config,
                            enable_sdma=bool(self._config.get("enable_sdma", False)),
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
            for shm in self._chip_shms:
                buf = shm.buf
                assert buf is not None
                # INIT_READY repurposes the lease slot_id as the child's depth
                # advertisement; task dispatch restores normal lease semantics.
                chip_depths.append(_PIPELINE_LEASE_FMT.unpack_from(buf, _OFF_PIPELINE_LEASE)[0])
            if any(depth <= 0 or depth > PTO_PIPELINE_MAX_DEPTH for depth in chip_depths):
                raise RuntimeError(f"chip worker published invalid pipeline depths: {chip_depths}")
            direct_chip_pipeline_depth = min(chip_depths)

        # Fork next-level Worker children (L4+ with Worker children).
        # Each child process eagerly inits the inner Worker, which forks its own
        # chip/sub (and, for L5+, deeper next-level) children and blocks on their
        # readiness before returning — so the process tree nests correctly (L4 →
        # L3 child → L3's chip/sub grandchildren) and INIT_READY propagates up
        # only after the whole subtree is ready.
        for idx, inner_worker in enumerate(self._next_level_workers):
            worker_id = self._next_level_worker_ids[idx]
            global_node = global_nodes.get(worker_id)
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
                    lambda tables, b=buf, inner=inner_worker, node=global_node: _child_worker_loop(
                        b,
                        *tables,
                        inner,
                        node,
                    ),
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
        self._activate_mpirun_worker_groups(deadline)
        self._activate_remote_sessions(deadline)

        # _Worker was constructed in _init_hierarchical (pre-fork) so children
        # inherit the HeapRing MAP_SHARED mmap. Register PROCESS-mode workers via
        # the unified mailbox.
        dw = self._worker
        assert dw is not None
        dw.configure_pipeline_depth(direct_chip_pipeline_depth)

        # Register chip workers as NEXT_LEVEL (L3). The child pid lets the C++
        # endpoint fail a dispatch whose child died instead of spinning on a
        # mailbox that can no longer be completed.
        if device_ids:
            _require_matching_pids(self._chip_shms, self._chip_pids, "chip")
            for shm, pid, chip_depth in zip(self._chip_shms, self._chip_pids, chip_depths):
                task_frame_count = _local_task_frame_count(
                    str(self._config["platform"]), str(self._config["runtime"]), chip_depth
                )
                dw.add_next_level_worker(_mailbox_addr(shm), pid, task_frame_count)

        # Register Worker children as NEXT_LEVEL (L4+)
        if self._next_level_shms and not hasattr(dw, "add_next_level_worker_at"):
            raise RuntimeError("explicit NEXT_LEVEL worker ids require a rebuilt _task_interface module")
        _require_matching_pids(self._next_level_shms, self._next_level_pids, "next_level")
        for idx, shm in enumerate(self._next_level_shms):
            worker_id = self._next_level_worker_ids[idx]
            dw.add_next_level_worker_at(worker_id, _mailbox_addr(shm), self._next_level_pids[idx])

        _require_matching_pids(self._sub_shms, self._sub_pids, "sub")
        for shm, pid in zip(self._sub_shms, self._sub_pids):
            dw.add_sub_worker(_mailbox_addr(shm), pid)

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
                if self._cancel_token:
                    raise InitCancelled(f"{kind} worker readiness wait cancelled by close()")
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
                _request_child_shutdown(buf)
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
                    reaped.add(pid)
                    continue
                if wpid != 0:
                    to_reap.discard(pid)
                    reaped.add(pid)
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

        # Release the pre-fork _Worker so a retry / close() won't double-free
        # the HeapRing mmap the C++ ctor grabbed.
        self._worker = None
        self._orch = None

        # Signal delivery is not reclamation. Use the same pid/shm pairing and
        # journal handoff as normal close; a SIGKILL survivor remains retryable.
        try:
            self._reclaim_child_groups(deadline)
        except BaseException as exc:  # noqa: BLE001 -- rollback remains best-effort
            sys.stderr.write(
                f"[worker pid={os.getpid()}] WARN: startup child reclaim failed (continuing best-effort): {exc}\n"
            )
            sys.stderr.flush()
        self._startup_group_leader_pids.clear()

    def _cleanup_partial_init(self) -> None:
        """Best-effort cleanup for init() failures before the Worker is public-live.

        One absolute cleanup deadline is created here and shared by every phase
        (including ``_abort_hierarchical``) so the whole rollback is bounded
        end-to-end rather than each phase re-acquiring a full timeout.
        """
        deadline = time.monotonic() + _ROLLBACK_GRACEFUL_TIMEOUT_S

        with contextlib.suppress(BaseException):
            self._teardown_worker_tree(startup_abort=True, deadline=deadline)
        self._comm_base_ready = False

    @property
    def live_domains(self) -> dict[str, CommDomainHandle]:
        """Read-only snapshot of currently-live dynamic CommDomain handles.

        Useful for debugging.  Mutating the returned dict has no effect; use
        ``handle.release()`` or ``orch.release_domain(handle)`` to free.
        """
        return dict(self._live_domains)

    def _validate_worker_chip_id(self, worker_id: int) -> None:
        if self.level < 3:
            raise RuntimeError("create_worker_chip_region requires a hierarchical Worker")
        if self._worker is None:
            raise RuntimeError("create_worker_chip_region requires Worker.init()")
        device_ids = self._config.get("device_ids", [])
        if worker_id < 0 or worker_id >= len(device_ids):
            raise ValueError(f"create_worker_chip_region: worker_id {worker_id} outside [0, {len(device_ids)})")

    def _poison_worker_chip_region_from_endpoint_error(
        self, exc: BaseException, resources: _RunResources | None = None
    ) -> bool:
        match = _WORKER_CHIP_ENDPOINT_ERROR_REGION_RE.search(str(exc))
        if match is None:
            return False
        region_id = int(match.group(1))
        if region_id == 0:
            return False
        poisoned = False
        regions = self._live_worker_chip_regions if resources is None else resources.worker_chip_regions
        for region in regions:
            if int(region.region_id) == region_id:
                region._poison()
                poisoned = True
        return poisoned

    def _register_worker_chip_orch_comm_host_buffer(self, handle) -> None:
        if not isinstance(handle, Buffer):
            raise TypeError("L3-L2 host buffer registration expects a Buffer")
        if handle.address_space != AddressSpace.HOST:
            raise ValueError("L3-L2 payload buffer must be host storage, not device storage")
        base = int(handle.base)
        nbytes = int(handle.nbytes)
        if base <= 0 or nbytes <= 0:
            return
        resources = self._building_run_resources
        buffers = (
            self._worker_chip_orch_comm_host_buffers
            if resources is None
            else resources.worker_chip_orch_comm_host_buffers
        )
        buffers[base] = max(
            int(buffers.get(base, 0)),
            nbytes,
        )

    def _validate_worker_chip_orch_comm_host_buffer(self, handle) -> None:
        if not isinstance(handle, Buffer):
            raise ValueError("L3-L2 payload buffer must be a Buffer returned by orch.alloc(...)")
        if handle.address_space != AddressSpace.HOST:
            raise ValueError("L3-L2 payload buffer must be host storage, not device storage")
        base = int(handle.base)
        nbytes = int(handle.nbytes)
        if base <= 0 or nbytes <= 0:
            raise ValueError("L3-L2 payload buffer must have a nonzero address and size")
        resources = self._building_run_resources
        buffers = (
            self._worker_chip_orch_comm_host_buffers
            if resources is None
            else resources.worker_chip_orch_comm_host_buffers
        )
        registered_nbytes = buffers.get(base)
        if registered_nbytes is None:
            raise ValueError("L3-L2 payload buffer is not registered; use a handle returned by orch.alloc(...)")
        if nbytes > int(registered_nbytes):
            raise ValueError(
                f"L3-L2 payload buffer size {nbytes} exceeds registered shared storage {registered_nbytes}"
            )

    def _consume_worker_host_mapped_cleanup_error_locked(self, api: str) -> RuntimeError | None:
        """Publish and then acknowledge this Worker's native cleanup debt.

        Must hold ``_hierarchical_start_cv``. The native read is intentionally
        non-destructive: an asynchronous exception before both sticky Python
        fields are assigned leaves the diagnostic available for the next
        boundary. Acknowledgement may be interrupted only after admission is
        already poisoned.
        """
        cleanup_error = _worker_host_mapped_region_peek_cleanup_error(self._owner_id)
        pending_snapshot, details = self._worker_host_mapped_cleanup_state
        if not cleanup_error:
            if pending_snapshot is not None:
                self._worker_host_mapped_cleanup_state = (None, details)
            return self._worker_host_mapped_cleanup_error

        if cleanup_error != pending_snapshot:
            detail = cleanup_error
            if pending_snapshot is not None and cleanup_error.startswith(f"{pending_snapshot}; "):
                # Native acknowledgement removes an observed prefix. If a
                # finalizer appends another diagnostic before that ack, retain
                # only the newly appended suffix in Python's stable copy.
                detail = cleanup_error[len(pending_snapshot) + 2 :]
            if detail and detail not in details:
                details = (*details, detail)
            self._worker_host_mapped_cleanup_state = (cleanup_error, details)

        leaked = self._worker_host_mapped_cleanup_error
        if leaked is None:
            leaked = RuntimeError(
                f"Worker.{api}: a native L3 Host mapping owned by this Worker finalized without explicit close "
                "and could not be reclaimed; no further work is admitted"
            )
            self._worker_host_mapped_cleanup_error = leaked
        leaked.__cause__ = RuntimeError("; ".join(details))
        if self._ordered_cleanup_error is None:
            self._ordered_cleanup_error = leaked

        _worker_host_mapped_region_ack_cleanup_error(self._owner_id, cleanup_error)
        self._worker_host_mapped_cleanup_state = (None, details)
        return self._worker_host_mapped_cleanup_error

    def _consume_worker_host_mapped_cleanup_error(self, api: str) -> RuntimeError | None:
        with self._hierarchical_start_cv:
            return self._consume_worker_host_mapped_cleanup_error_locked(api)

    def _create_worker_chip_region(self, worker_id: int, payload_bytes: int, counter_bytes: int):  # noqa: PLR0912
        if payload_bytes <= 0:
            raise ValueError("create_worker_chip_region: payload_bytes must be positive")
        if counter_bytes <= 0 or counter_bytes % 4 != 0:
            raise ValueError("create_worker_chip_region: counter_bytes must be positive and a multiple of 4")
        self._validate_worker_chip_id(int(worker_id))
        prior_native_cleanup_error = self._consume_worker_host_mapped_cleanup_error("create_worker_chip_region")
        if prior_native_cleanup_error is not None:
            raise prior_native_cleanup_error
        resources = self._building_run_resources
        req_shm = SharedMemory(create=True, size=_REGION_CREATE_REQUEST_BYTES)
        reply_shm = SharedMemory(create=True, size=_REGION_CREATE_REPLY_BYTES)
        req_buf = cast(memoryview, req_shm.buf)
        reply_buf = cast(memoryview, reply_shm.buf)
        region_id = 0
        native_mapping_handle = None
        worker_host_mapping = None
        region = None
        dispatched = False
        required_ordered_cleanup_before = resources.requires_ordered_cleanup if resources is not None else False
        try:
            WorkerChipRegionCreateRequest(
                magic_version=_REGION_MAGIC_VERSION,
                request_bytes=_REGION_CREATE_REQUEST_BYTES,
                payload_bytes=int(payload_bytes),
                counter_bytes=int(counter_bytes),
            ).encode_into(req_buf)
            worker = self._worker
            assert worker is not None
            # From here the chip may have created a region. The create releases
            # the GIL, so an interrupt can land after the child committed and
            # before the id is read back — the rollback below cannot assume
            # "no id" means "nothing exists".
            dispatched = True
            worker.control_worker_chip_region_create(int(worker_id), req_shm.name, reply_shm.name)
            # Peek before decode: decode rejects malformed replies, but the
            # child has already created the region and the rollback below
            # still needs the id.
            region_id = peek_region_create_reply_region_id(reply_buf)
            reply = decode_region_create_reply(reply_buf)
            platform = str(self._config.get("platform", ""))
            expected_access_profile = (
                WorkerChipRegionAccessProfile.SIM_POSIX_SHM
                if platform.endswith("sim")
                else WorkerChipRegionAccessProfile.ONBOARD_VMM
            )
            counter_offset, total_bytes = validate_region_create_reply(reply, expected_access_profile)
            if platform.endswith("sim"):
                native_mapping_handle = _worker_host_mapped_region_import_sim(
                    reply.backing_shm, int(reply.mapping_bytes), self._owner_id
                )
            else:
                native_mapping_handle = _worker_host_mapped_region_import_onboard(
                    int(reply.device_id),
                    int(reply.shareable_handle),
                    int(reply.mapping_bytes),
                    self._owner_id,
                )
            worker_host_mapping = WorkerHostRegionMapping(
                worker_id=int(worker_id),
                region_id=region_id,
                access_profile=reply.access_profile,
                total_bytes=total_bytes,
                payload_offset=0,
                payload_bytes=int(reply.desc.payload_bytes),
                counter_offset=counter_offset,
                counter_bytes=int(reply.desc.counter_bytes),
                handle=native_mapping_handle,
            )
            native_mapping_handle = None
            region = WorkerChipOrchRegion(self, int(worker_id), reply.desc, worker_host_mapping)
            self._live_worker_chip_regions.append(region)
            if resources is not None:
                resources.worker_chip_regions.append(region)
                # Region teardown is mailbox control on its owning chip.
                resources.requires_ordered_cleanup = True
            return region
        except BaseException as exc:
            mapping_cleanup_error: BaseException | None = None
            if region is not None:
                try:
                    self._live_worker_chip_regions.remove(region)
                except ValueError:
                    pass
                if resources is not None:
                    try:
                        resources.worker_chip_regions.remove(region)
                    except ValueError:
                        pass
                    resources.requires_ordered_cleanup = required_ordered_cleanup_before
                region._expire()
            if worker_host_mapping is not None:
                try:
                    worker_host_mapping.close()
                except BaseException as mapping_exc:  # noqa: BLE001
                    mapping_cleanup_error = mapping_exc
            elif native_mapping_handle is not None:
                try:
                    _worker_host_mapped_region_close(int(native_mapping_handle))
                except BaseException as mapping_exc:  # noqa: BLE001
                    mapping_cleanup_error = mapping_exc
            if not region_id:
                # The failure may have landed after the child created its
                # region. The reply shm is parent-owned and zero-filled and the
                # child writes the id as part of committing, so a nonzero id
                # here is the child saying the region exists.
                with contextlib.suppress(BaseException):
                    region_id = peek_region_create_reply_region_id(reply_buf)
            if not region_id and dispatched and not isinstance(exc, Exception):
                # An asynchronous unwind through the create. The child may still
                # be finishing a region, and the id it would write lands in a
                # reply this frame is about to unlink — leaving something on the
                # chip that nothing here can name. An ordinary failure is not
                # this case: the child releases its own region before reporting
                # one, so a zero id there really does mean nothing exists.
                self._record_unreclaimable(
                    f"create_worker_chip_region: interrupted on worker {int(worker_id)} before the region id was "
                    "read back; a region may be live on the chip and no further work is admitted",
                    exc,
                )
            if region_id:
                try:
                    assert self._worker is not None
                    self._worker.control_worker_chip_region_release(int(worker_id), int(region_id))
                except BaseException as release_exc:  # noqa: BLE001
                    # The chip created the region and this call could not give
                    # it back. Nothing else knows the id — it was never tracked
                    # — so no later cleanup can reclaim it, and a rollback that
                    # swallows this reports an ordinary failure over a region
                    # that stays allocated for the worker's life. That is the
                    # unreclaimed-device-state condition, recorded here directly
                    # because there is no handle for a fence to fail on.
                    raise self._record_unreclaimable(
                        f"create_worker_chip_region: rollback could not release region {region_id} on worker "
                        f"{int(worker_id)}; it is leaked and no further work is admitted",
                        release_exc,
                    )
            deferred_native_cleanup_error = self._consume_worker_host_mapped_cleanup_error(
                "create_worker_chip_region rollback"
            )
            if deferred_native_cleanup_error is not None:
                deferred_exc = deferred_native_cleanup_error.__cause__ or deferred_native_cleanup_error
                if mapping_cleanup_error is not None:
                    combined_exc = RuntimeError(
                        f"{mapping_cleanup_error}; deferred native cleanup also failed: {deferred_exc}"
                    )
                    combined_exc.__cause__ = mapping_cleanup_error
                    mapping_cleanup_error = combined_exc
                else:
                    mapping_cleanup_error = deferred_exc
            if mapping_cleanup_error is not None:
                raise self._record_unreclaimable(
                    f"create_worker_chip_region: rollback could not close the L3 Host mapping for region "
                    f"{region_id} on worker {int(worker_id)}; it is leaked and no further work is admitted",
                    mapping_cleanup_error,
                )
            raise exc
        finally:
            del req_buf
            del reply_buf
            for shm in (req_shm, reply_shm):
                try:
                    shm.close()
                    shm.unlink()
                except (BufferError, FileNotFoundError, OSError):
                    pass

    def _close_worker_chip_region(
        self,
        region,
        resources: _RunResources | None = None,
        *,
        poison_on_error: bool = False,
    ) -> None:
        region_errors: list[BaseException] = []
        release_error: BaseException | None = None

        def primary_region_error() -> BaseException:
            primary = region_errors[0]
            tail = primary
            seen = {id(tail)}
            while tail.__cause__ is not None and id(tail.__cause__) not in seen:
                tail = tail.__cause__
                seen.add(id(tail))
            for extra in region_errors[1:]:
                tail.__cause__ = extra
                tail = extra
            return primary

        expired = bool(region.expired)
        if not expired:
            try:
                region._close_worker_host_mapping()
            except BaseException as exc:  # noqa: BLE001
                region_errors.append(exc)
            try:
                if self._worker is not None and not region._chip_release_committed:
                    self._worker.control_worker_chip_region_release(region._worker_id, region.region_id)
                    region._chip_release_committed = True
                    region.free()
            except BaseException as exc:  # noqa: BLE001
                release_error = exc
                region_errors.append(exc)
        # A region remains tracked while chip ownership may still be live, so
        # whole-tree close can replay it and instance cleanup can poison future
        # admission. Once the chip release is committed or the native handle is
        # already expired, cleanup may retire every tracking list that owns it.
        if region_errors and resources is None and not poison_on_error:
            raise primary_region_error()
        if poison_on_error and region_errors and release_error is not None:
            primary = primary_region_error()
            self._record_unreclaimable(
                f"close_worker_chip_region: region {region.region_id} on worker {region._worker_id} could not be "
                "fully reclaimed; no further work is admitted",
                primary,
            )
            raise primary
        try:
            if not expired:
                region._expire()
        except BaseException as exc:  # noqa: BLE001
            region_errors.append(exc)
        try:
            self._retire_worker_chip_region_tracking(region, resources)
        except BaseException as exc:  # noqa: BLE001
            region_errors.append(exc)
        if region_errors:
            primary = primary_region_error()
            if poison_on_error:
                self._record_unreclaimable(
                    f"close_worker_chip_region: region {region.region_id} on worker {region._worker_id} could not be "
                    "fully reclaimed; no further work is admitted",
                    primary,
                )
            raise primary

    def _retire_worker_chip_region_tracking(self, region, resources: _RunResources | None = None) -> None:
        tracking_lists = [self._live_worker_chip_regions]
        if resources is not None and resources.worker_chip_regions is not self._live_worker_chip_regions:
            tracking_lists.insert(0, resources.worker_chip_regions)
        errors: list[BaseException] = []

        for owned_regions in tracking_lists:
            try:
                owned_regions[:] = [owned for owned in owned_regions if owned is not region]
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            raise errors[0]

    def _cleanup_worker_chip_regions(self, resources: _RunResources | None = None) -> None:
        # A region stays tracked until both ownership debts commit. This makes a
        # failed close replayable by the cleanup journal instead of publishing a
        # false success after dropping the only reference to the region.
        tracked = self._live_worker_chip_regions if resources is None else resources.worker_chip_regions
        if not tracked:
            return
        regions = list(tracked)
        errors: list[BaseException] = []
        for region in regions:
            try:
                self._close_worker_chip_region(region, resources)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            raise errors[0]

    def _close_worker_chip_orch_comm(self) -> None:
        for region in self._live_worker_chip_regions:
            try:
                region._close_worker_host_mapping()
            except RuntimeError:
                pass
        self._live_worker_chip_regions.clear()
        self._worker_chip_orch_comm_host_buffers.clear()

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

        # Layout: header (rank, nranks) + NUL-terminated rootinfo_path bytes.
        path_bytes = rootinfo_path.encode("utf-8") + b"\x00"
        req_size = _COMM_INIT_HEADER.size + len(path_bytes)

        def initialize(request_owner: _SharedMemoryOwner) -> None:
            request_shms: dict[int, SharedMemory] = {}
            for chip_idx, _device_id in enumerate(device_ids):
                req = request_owner.create(req_size)
                request_shms[chip_idx] = req
                req_buf = req.buf
                assert req_buf is not None
                _COMM_INIT_HEADER.pack_into(req_buf, 0, int(chip_idx), int(len(device_ids)))
                req_buf[_COMM_INIT_HEADER.size : _COMM_INIT_HEADER.size + len(path_bytes)] = path_bytes

            dw = self._worker
            assert dw is not None
            errors: dict[int, BaseException] = {}

            def dispatch(chip_idx: int) -> None:
                try:
                    dw.control_comm_init(chip_idx, request_shms[chip_idx].name)
                except BaseException as e:  # noqa: BLE001
                    errors[chip_idx] = e

            _start_and_join_threads(range(len(device_ids)), dispatch, name_prefix="comm_init_chip_")
            if errors:
                first = next(iter(errors.items()))
                raise RuntimeError(
                    f"_ensure_comm_base failed on {len(errors)}/{len(device_ids)} chips; "
                    f"first error chip={first[0]}: {first[1]}"
                )

        _run_with_owned_shared_memory(
            len(device_ids),
            initialize,
            name_prefix="shm-comm-init-lifecycle-",
            after_success=lambda: setattr(self, "_comm_base_ready", True),
        )

    def _allocate_domain(  # noqa: PLR0912 -- linear input-validation + per-chip shm staging + dispatch + reply unpack; splitting obscures the fail-fast ordering
        self,
        *,
        name: str,
        workers: tuple[int, ...],
        window_size: int,
        buffers: list[CommBufferSpec],
    ) -> CommDomainHandle:
        # Admission is the run() lease that the driving orchestrator holds;
        # validation checks resource presence rather than the public lifecycle,
        # so an allocation admitted before concurrent close still drains.
        # Buffer carving is checked before communicator/device side effects.
        resources = _validate_domain_allocation(self, name, workers, window_size, buffers)

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

        handle: CommDomainHandle | None = None
        contexts: dict[int, ChipDomainContext] = {}
        previous_ordered_cleanup = resources.requires_ordered_cleanup

        def allocate(staged_shms: _SharedMemoryOwner) -> CommDomainHandle:
            nonlocal handle
            request_shms: dict[int, SharedMemory] = {}
            reply_shms: dict[int, SharedMemory] = {}
            dispatch_started = False
            try:
                for chip_idx in workers:
                    req = staged_shms.create(req_size)
                    request_shms[chip_idx] = req
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
                    reply = staged_shms.create(reply_size)
                    reply_shms[chip_idx] = reply

                # Ownership is complete before a chip can commit an allocation.
                # Until the replies are sampled, every requested rank is retained
                # conservatively; an interrupted sampler may cost a failed release,
                # but cannot make an allocated window unreachable.
                handle = CommDomainHandle(
                    name=name,
                    workers=workers,
                    contexts={},
                    allocation_id=allocation_id,
                    _release_fn=lambda released, owner=resources: self._release_domain_handle(released, owner),
                    _domain_size=len(workers),
                    _domain_ranks=worker_to_rank,
                )
                self._live_domains[name] = handle
                resources.live_domains[name] = handle
                resources.requires_ordered_cleanup = True

                # A chip that committed its window holds it whether or not this call
                # returns a handle, and whether or not its RPC reported success —
                # the window exists before the reply is written. Each chip says so
                # itself, and the chips that did are registered under a handle the
                # run owns, so the fence sweep releases exactly those and poisons
                # the worker if that release fails too. Inferring it from the RPC
                # outcome leaks an allocation whose reply could not be delivered.
                dispatch_started = True
                try:
                    self._dispatch_control_domain(
                        workers=workers,
                        request_shms=request_shms,
                        reply_shms=reply_shms,
                        op="alloc",
                        allocation_id=allocation_id,
                    )

                    for chip_idx in workers:
                        reply_buf = reply_shms[chip_idx].buf
                        assert reply_buf is not None
                        (committed, device_ctx, local_window_base, reply_buffer_count) = (
                            _DOMAIN_REPLY_HEADER.unpack_from(reply_buf, 0)
                        )
                        if not committed:
                            raise RuntimeError(f"allocate_domain: chip {chip_idx} reported success without committing")
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
                            buffers={
                                b.name: wrap_vmm_window(
                                    ptrs[i],
                                    int(b.nbytes),
                                    self._owner_instance_id,
                                    self._next_buffer_id(),
                                    f"L{self.level}",
                                    owner_worker_id=int(chip_idx),
                                )
                                for i, b in enumerate(buffers)
                            },
                        )
                    handle.contexts = contexts
                finally:
                    committed = tuple(w for w in workers if _domain_reply_committed(reply_shms.get(w)))
                    handle.workers = committed
                    if not committed:
                        if self._live_domains.get(name) is handle:
                            self._live_domains.pop(name)
                        if resources.live_domains.get(name) is handle:
                            resources.live_domains.pop(name)
                        resources.requires_ordered_cleanup = previous_ordered_cleanup
            except BaseException:  # noqa: BLE001
                if handle is not None and not dispatch_started:
                    if self._live_domains.get(name) is handle:
                        self._live_domains.pop(name)
                    if resources.live_domains.get(name) is handle:
                        resources.live_domains.pop(name)
                    resources.requires_ordered_cleanup = previous_ordered_cleanup
                raise
            assert handle is not None
            return handle

        def publish_provenance() -> None:
            assert handle is not None
            # The backend windows are now live: record each chip's window base
            # and every carved buffer pointer before the lifecycle publishes
            # success back to the interruptible caller.
            with self._child_prov_lock:
                # Only the carved buffers are registered: the window itself is named by no public
                # Buffer -- `ChipDomainContext` exposes `local_window_base` as a bare address -- so
                # no operand can reach it and nothing would resolve an entry for it.
                for ctx in contexts.values():
                    for buf in ctx.buffers.values():
                        self._record_device_alloc(buf, domain_allocation_id=allocation_id)

        published_handle = _run_with_owned_shared_memory(
            len(workers) * 2,
            allocate,
            name_prefix="shm-domain-alloc-lifecycle-",
            after_success=publish_provenance,
        )
        assert handle is not None and published_handle is handle
        return handle

    def _release_domain_handle(self, handle: CommDomainHandle, resources: _RunResources) -> None:
        """Mark a handle for release.  Actual backend free is deferred.

        Called by ``CommDomainHandle.release()``.  We do NOT drive
        ``CTRL_RELEASE_DOMAIN`` here because the orch function is allowed
        to have already submitted DAG tasks that capture the handle's
        ``device_ctx`` / ``buffers``.  Those tasks must see live
        memory through execution; the queue is drained by
        ``_execute_pending_domain_releases`` once the owning run's fence fires.

        ``resources`` is the run that allocated the handle, bound at
        allocation; a handle released after that run's fence has no drain left
        and is reported rather than silently queued.
        """
        if self._worker is None:
            return
        with resources.domain_lock:
            if not resources.retired:
                # Publish the fence-owned claim before retiring either live
                # registry. If append is interrupted, those registries still
                # make the allocation reachable by the end-of-run/close sweep.
                resources.pending_release_domains.append(handle)
                resources.live_domains.pop(handle.name, None)
                # Pop from _live_domains so a subsequent allocation in this
                # run can reuse the name. The pending claim keeps the old
                # allocation alive until the fence drains it.
                if self._live_domains.get(handle.name) is handle:
                    self._live_domains.pop(handle.name)
                return
        # Deferral exists so tasks that captured this domain still see live
        # memory; the owning run's fence has passed, so there is nothing left
        # to defer behind. Keep both live registries as the durable owner while
        # the backend call runs; exact-once free removes the global entry only
        # after it commits, and the run entry is retired below.
        self._free_domain_after_fence(handle)
        with resources.domain_lock:
            if resources.live_domains.get(handle.name) is handle:
                resources.live_domains.pop(handle.name)
            if self._live_domains.get(handle.name) is handle:
                self._live_domains.pop(handle.name)

    def _retire_run_domains(self, resources: _RunResources) -> None:
        """Close this run's deferred-release path and free anything left on it.

        A release that read `retired` as False contends for `domain_lock`, so
        it has either already appended (and is drained here) or observes
        retirement and frees itself.
        """
        with resources.domain_lock:
            resources.retired = True
            stragglers = list(resources.pending_release_domains)
            global_stragglers = list(resources.pending_release_global_domains)
            resources.live_global_domains.clear()
        self._drain_pending_domain_snapshot(resources, stragglers)
        self._drain_pending_global_domain_snapshot(resources, global_stragglers)

    def _drain_pending_domain_snapshot(self, resources: _RunResources, pending: list[CommDomainHandle]) -> None:
        """Free a snapshot while each source-queue claim stays durable."""

        def _release(handle: CommDomainHandle) -> None:
            self._free_domain_after_fence(handle)
            # Retire only after exact-once backend success. An interruption
            # before this deletion leaves a replayable claim; an interruption
            # after it is safe because the allocation is already gone.
            with resources.domain_lock:
                for index, candidate in enumerate(resources.pending_release_domains):
                    if candidate is handle:
                        del resources.pending_release_domains[index]
                        break

        _raise_first(_release, pending)

    def _drain_pending_global_domain_snapshot(
        self,
        resources: _RunResources,
        pending: list[GlobalCommDomainHandle],
    ) -> None:
        """Free global-domain claims without dropping unfinished cleanup debt."""

        def _release(handle: GlobalCommDomainHandle) -> None:
            self._free_global_domain_after_fence(handle)
            with resources.domain_lock:
                for index, candidate in enumerate(resources.pending_release_global_domains):
                    if candidate is handle:
                        del resources.pending_release_global_domains[index]
                        break

        _raise_first(_release, pending)

    def _free_domain_after_fence(self, handle: CommDomainHandle) -> None:
        """Back-end free for a handle whose owning run has retired.

        Tolerates a handle the end-of-run sweep already freed: the sweep works
        from a snapshot, so a release taken after that snapshot reaches here
        for a domain that is already gone.

        A failed release propagates. Its run carries
        ``requires_ordered_cleanup`` — that is set when the domain is created —
        so the error is what stops a successor from starting on top of device
        state nobody can describe.
        """
        if handle.freed:
            return
        self._release_domain_now(handle)
        handle._freed = True  # noqa: SLF001 -- runtime owns this transition

    def _execute_pending_domain_releases(self, resources: _RunResources) -> None:
        """Drive CTRL_RELEASE_DOMAIN for every queued handle.  Must run
        after ``self._orch._wait_run()`` so chip-side tasks have completed
        their use of the domain memory.

        Per-handle best-effort: one failing release never strands the rest, and
        the first error is raised once all are attempted.
        """
        # Snapshot under the same lock release() uses to append. Keep every
        # claim in the source queue while the backend call is in flight: a
        # release appended after this snapshot then remains for retirement's
        # final drain instead of being erased by an unlocked clear().
        with resources.domain_lock:
            pending = list(resources.pending_release_domains)
        self._drain_pending_domain_snapshot(resources, pending)

    def _release_domain_now(self, handle: CommDomainHandle) -> None:
        """Synchronous backend release for one handle, exactly once.

        Used by the deferred-release path and by the abort/close cleanup
        helpers. The first caller drives the release; a second caller for the
        same allocation blocks until that release finishes and then sees its
        outcome, so it never issues a duplicate CTRL_RELEASE_DOMAIN and never
        reports success early. Returning before the owner finished would let a
        caller mark the handle freed, drop it from ``_live_domains`` and — on
        the close() path — tear down the mailboxes the release is still using.

        A failure is replayed to every later caller rather than retried, so the
        sweeps keep the handle in ``_live_domains`` as a detectable residual.
        """
        if self._worker is None:
            return
        with self._domain_free_mu:
            if handle.allocation_id in self._domain_free_results:
                failure = self._domain_free_results[handle.allocation_id]
                if failure is not None:
                    raise failure
                return
            # Absence means no backend attempt has been claimed. Once the
            # isolated target is admitted it publishes a conservative outcome
            # before the backend call, so even failure to store the real
            # post-RPC result cannot turn a committed free back into an
            # apparently unclaimed allocation that a waiter replays. Publishing
            # inside the target matters: an interruption before admission must
            # leave the allocation retryable because no backend work began.
            unpublished_outcome = RuntimeError(
                "CommDomain backend release outcome publication was interrupted; refusing to replay the free"
            )
            result = _IsolatedCallResult()

            def release_claim() -> None:
                self._domain_free_results[handle.allocation_id] = unpublished_outcome
                try:
                    self._release_domain_claimed(handle)
                except BaseException as exc:
                    self._domain_free_results[handle.allocation_id] = exc
                    raise

            # The operation and its success publication run outside the
            # caller's async-exception boundary while contenders remain behind
            # _domain_free_mu. A KeyboardInterrupt after helper completion can
            # propagate, but a retry already observes the committed outcome.
            _run_isolated_call(
                result,
                release_claim,
                name_prefix="simpler-domain-free-",
                after_success=lambda: self._domain_free_results.__setitem__(handle.allocation_id, None),
            )
            if result.error is not None:
                raise result.error
            if not result.completed:
                raise RuntimeError("CommDomain backend release did not publish an outcome")

    def _release_domain_claimed(self, handle: CommDomainHandle) -> None:
        """Drive CTRL_RELEASE_DOMAIN. Caller holds this allocation's claim."""
        workers = handle.workers
        # Release payload is just the fixed header — no rank_ids tail; the
        # backend looked them up from its own per-allocation record at
        # alloc time and doesn't need them again.
        req_size = _DOMAIN_REQ_HEADER.size

        def release(request_owner: _SharedMemoryOwner) -> None:
            # Revoke provenance BEFORE the physical free: once release begins
            # the domain's pointers are no longer dispatchable. Dropping first
            # makes an interrupted/failed release a recoverable leak instead of
            # leaving a use-after-free validation window.
            with self._child_prov_lock:
                self._drop_domain_allocs(handle.allocation_id)
            request_shms: dict[int, SharedMemory] = {}
            for chip_idx in workers:
                req = request_owner.create(req_size)
                request_shms[chip_idx] = req
                req_buf = req.buf
                assert req_buf is not None
                _DOMAIN_REQ_HEADER.pack_into(
                    req_buf,
                    0,
                    int(handle.allocation_id),
                    int(handle._domain_size),  # noqa: SLF001 -- backend release identity belongs to the handle
                    int(handle._domain_ranks[chip_idx]),  # noqa: SLF001 -- preserve the allocation-time rank
                    0,  # window_size — ignored on release
                    0,  # buffer_count — ignored on release
                )
            self._dispatch_control_domain(
                workers=workers,
                request_shms=request_shms,
                reply_shms=None,
                op="release",
                allocation_id=handle.allocation_id,
            )

        def retire_live_handle() -> None:
            if self._live_domains.get(handle.name) is handle:
                self._live_domains.pop(handle.name)

        _run_with_owned_shared_memory(
            len(workers),
            release,
            name_prefix="shm-domain-release-lifecycle-",
            after_success=retire_live_handle,
        )

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

        _start_and_join_threads(workers, dispatch, name_prefix=f"{op}_domain_chip_")

        if errors:
            first = next(iter(errors.items()))
            raise RuntimeError(
                f"{op}_domain(allocation_id={allocation_id}) failed on "
                f"{len(errors)}/{len(workers)} chips; first error chip={first[0]}: {first[1]}"
            )

    @staticmethod
    def _global_domain_command_identity(command: GlobalDomainCommand) -> tuple[Any, ...]:
        return (
            command.domain_id,
            command.generation,
            command.name,
            command.profile,
            command.window_size,
            command.members,
            command.buffers,
            command.attachments,
        )

    def _global_domain_attachment_matrix(
        self,
        members: tuple[GlobalDomainMember, ...],
        receiver_nodes: tuple[int, ...],
        window_size: int,
    ) -> tuple[GlobalDomainAttachment, ...]:
        """Resolve one host-consumer row for every receiving L3 node.

        The endpoint planner remains the authority for adapter selection.  A
        relation that the current planner cannot serve is retained as a
        host-consumer attachment with no adapter; the wire therefore preserves
        the complete matrix without claiming that an unavailable remote path
        is usable.  A later access primitive can fill that capability without
        changing rank membership or row cardinality.
        """

        registry = self._get_endpoint_registry()
        resolver = BackendResolver(registry, self._get_region_access_service())
        root_path = _format_worker_path(int(self.level))
        attachments: list[GlobalDomainAttachment] = []
        for receiver_node_id in sorted({int(node) for node in receiver_nodes}):
            receiver_path = _format_worker_path(
                3,
                parent_path=root_path,
                index=receiver_node_id,
            )
            consumer_selector = at(receiver_path, HOST_CPU)
            for member in members:
                member_path = _format_worker_path(
                    2,
                    parent_path=receiver_path
                    if int(member.node_worker_id) == receiver_node_id
                    else _format_worker_path(
                        3,
                        parent_path=root_path,
                        index=int(member.node_worker_id),
                    ),
                    index=int(member.local_worker_id),
                )
                provider_selector = at(member_path, DEVICE_AICORE)
                resolved = registry.resolve_region_spec(
                    (provider_selector, consumer_selector),
                    SingleOwner(provider=provider_selector),
                )
                plan = resolver.plan(
                    resolved,
                    RegionLayoutSpec(payload_bytes=int(window_size), counter_bytes=0),
                )
                adapter_kind: AdapterKind | None = None
                adapter_profile: AdapterProfile | None = None
                if not isinstance(plan, UnsupportedRegionPlan):
                    consumer = next(
                        attachment
                        for attachment in plan.payload.attachments
                        if attachment.role is AttachmentRole.CONSUMER
                    )
                    adapter_kind = consumer.adapter_kind
                    adapter_profile = consumer.adapter_profile
                attachments.append(
                    GlobalDomainAttachment(
                        node_worker_id=receiver_node_id,
                        address_space=AddressSpace.HOST,
                        role=AttachmentRole.CONSUMER,
                        adapter_kind=adapter_kind,
                        adapter_profile=adapter_profile,
                    )
                )
        return tuple(attachments)

    @staticmethod
    def _global_domain_provenance_id(domain_id: int) -> int:
        # Local CommDomain allocation ids are positive. Keep remote Global
        # CommDomains in a disjoint namespace while reusing the same exact-pointer
        # provenance table that protects child-memory task submissions.
        return -int(domain_id)

    def _global_local_members(
        self, command: GlobalDomainCommand, node_worker_id: int
    ) -> tuple[GlobalDomainMember, ...]:
        members = tuple(member for member in command.members if member.node_worker_id == node_worker_id)
        if not members:
            raise ValueError(f"Global CommDomain has no members on node worker {node_worker_id}")
        local_count = len(self._config.get("device_ids", []))
        for member in members:
            if member.local_worker_id < 0 or member.local_worker_id >= local_count:
                raise ValueError(
                    f"Global CommDomain local worker {member.local_worker_id} is outside [0, {local_count})"
                )
        return members

    def _prepare_global_domain_node(
        self, command: GlobalDomainCommand, node_worker_id: int
    ) -> tuple[GlobalDomainDescriptor, ...]:
        if self.level != 3 or self._worker is None:
            raise RuntimeError("Global CommDomain node prepare requires a ready L3 Worker")
        # PREPARE_EXPORT is where the row is stored for IMPORT and COMMIT to reuse, so a table this
        # node has no row in is rejected here rather than at the first phase that reads one.
        _ = command.attachments_for_node(node_worker_id)
        prior = self._global_node_domains.get(command.domain_id)
        if prior is not None:
            if self._global_domain_command_identity(prior.command) != self._global_domain_command_identity(command):
                raise RuntimeError("Global CommDomain prepare conflicts with a live domain")
            return tuple(prior.descriptors[rank] for rank in sorted(prior.descriptors))

        state = _GlobalNodeDomainState(command=command)
        self._global_node_domains[command.domain_id] = state
        local_members = self._global_local_members(command, node_worker_id)
        capacity = max(LOCAL_PREPARE_REQUEST.size, LOCAL_PREPARE_REPLY.size + GLOBAL_DOMAIN_DESCRIPTOR_BYTES)
        try:
            for member in local_members:
                payload = bytearray(capacity)
                LOCAL_PREPARE_REQUEST.pack_into(
                    payload,
                    0,
                    LOCAL_DOMAIN_MAGIC,
                    GLOBAL_DOMAIN_VERSION,
                    command.domain_id,
                    command.generation,
                    member.domain_rank,
                    len(command.members),
                    GLOBAL_DOMAIN_PROFILE_IDS[command.profile],
                    command.window_size,
                )
                state.prepared_domain_ranks.add(member.domain_rank)
                reply = bytes(
                    self._worker.control_payload(
                        WorkerType.NEXT_LEVEL,
                        member.local_worker_id,
                        CTRL_GLOBAL_DOMAIN_PREPARE,
                        payload,
                        self._py_control_timeout_s,
                    )
                )
                fields = LOCAL_PREPARE_REPLY.unpack_from(reply, 0)
                magic, version, domain_id, generation, local_base, mapping_size = fields
                _validate_local_global_header(magic, version, domain_id, generation, operation="prepare reply")
                if domain_id != command.domain_id or generation != command.generation:
                    raise RuntimeError("Global CommDomain prepare reply identity mismatch")
                start = LOCAL_PREPARE_REPLY.size
                descriptor = GlobalDomainDescriptor.decode(reply[start : start + GLOBAL_DOMAIN_DESCRIPTOR_BYTES])
                if descriptor.domain_rank != member.domain_rank:
                    raise RuntimeError("Global CommDomain prepare reply rank mismatch")
                state.descriptors[member.domain_rank] = descriptor
                state.local_window_bases[member.local_worker_id] = int(local_base)
                state.mapping_sizes[member.local_worker_id] = int(mapping_size)
            return tuple(state.descriptors[rank] for rank in sorted(state.descriptors))
        except BaseException:
            self._release_global_domain_node(
                GlobalDomainReleaseCommand(command.domain_id, command.generation),
                suppress_errors=True,
            )
            raise

    def _import_global_domain_node(self, command: GlobalDomainCommand, node_worker_id: int) -> None:
        if self.level != 3 or self._worker is None:
            raise RuntimeError("Global CommDomain node import requires a ready L3 Worker")
        state = self._global_node_domains.get(command.domain_id)
        if state is None or state.command.generation != command.generation:
            raise RuntimeError("Global CommDomain import requires a matching prepared domain")
        if command.attachments:
            if command.attachments != state.command.attachments:
                raise RuntimeError("Global CommDomain import attachments conflict with prepare")
        elif state.command.attachments:
            # Attachments are immutable domain topology. IMPORT carries no
            # second copy; rehydrate the prepared row before comparing the
            # command identity and publishing the node view.
            command = replace(command, attachments=state.command.attachments)
        node_attachments = command.attachments_for_node(node_worker_id)
        if self._global_domain_command_identity(state.command) != self._global_domain_command_identity(command):
            raise RuntimeError("Global CommDomain import command conflicts with prepare")
        validate_descriptor_table(
            command.descriptors,
            rank_count=len(command.members),
            profile=command.profile,
        )
        local_members = self._global_local_members(command, node_worker_id)
        descriptor_bytes = b"".join(descriptor.encode() for descriptor in command.descriptors)
        request_size = LOCAL_IMPORT_REQUEST.size + len(descriptor_bytes)
        capacity = max(request_size, LOCAL_IMPORT_REPLY.size)
        try:
            for member in local_members:
                payload = bytearray(capacity)
                LOCAL_IMPORT_REQUEST.pack_into(
                    payload,
                    0,
                    LOCAL_DOMAIN_MAGIC,
                    GLOBAL_DOMAIN_VERSION,
                    command.domain_id,
                    command.generation,
                    len(command.descriptors),
                )
                payload[LOCAL_IMPORT_REQUEST.size : request_size] = descriptor_bytes
                reply = bytes(
                    self._worker.control_payload(
                        WorkerType.NEXT_LEVEL,
                        member.local_worker_id,
                        CTRL_GLOBAL_DOMAIN_IMPORT,
                        payload,
                        self._py_control_timeout_s,
                    )
                )
                fields = LOCAL_IMPORT_REPLY.unpack_from(reply, 0)
                magic, version, domain_id, generation, device_ctx, local_base, mapping_size = fields
                _validate_local_global_header(magic, version, domain_id, generation, operation="import reply")
                if domain_id != command.domain_id or generation != command.generation:
                    raise RuntimeError("Global CommDomain import reply identity mismatch")
                if mapping_size != command.descriptors[member.domain_rank].mapping_size:
                    raise RuntimeError("Global CommDomain import reply mapping size mismatch")
                offset = 0
                buffer_bases: dict[str, int] = {}
                domain_buffers: dict[str, Buffer] = {}
                for buffer in command.buffers:
                    base = int(local_base) + offset
                    buffer_bases[buffer.name] = base
                    domain_buffers[buffer.name] = wrap_vmm_window(
                        base,
                        int(buffer.nbytes),
                        self._owner_instance_id,
                        self._next_buffer_id(),
                        f"L{self.level}",
                        owner_worker_id=int(member.local_worker_id),
                    )
                    offset += buffer.nbytes
                state.contexts[member.local_worker_id] = ChipDomainContext(
                    name=command.name,
                    domain_rank=member.domain_rank,
                    domain_size=len(command.members),
                    device_ctx=int(device_ctx),
                    local_window_base=int(local_base),
                    actual_window_size=int(mapping_size),
                    buffers=domain_buffers,
                )
                provenance_id = self._global_domain_provenance_id(command.domain_id)
                with self._child_prov_lock:
                    for buffer in command.buffers:
                        self._record_device_alloc(domain_buffers[buffer.name], domain_allocation_id=provenance_id)
            state.command = command
            state.phase = GlobalDomainPhase.IMPORT
            state.view = GlobalCommDomainView(
                name=command.name,
                members=command.members,
                contexts=state.contexts,
                domain_id=command.domain_id,
                generation=command.generation,
                mapping_size=command.descriptors[0].mapping_size,
                attachments=node_attachments,
            )
        except BaseException:
            # A node may have imported one local rank before a later local rank
            # fails. Roll that partial node back here; the L4 ABORT fanout is an
            # idempotent second safety net, not the only cleanup owner.
            self._release_global_domain_node(
                GlobalDomainReleaseCommand(command.domain_id, command.generation),
                suppress_errors=True,
            )
            raise

    def _commit_global_domain_node(self, command: GlobalDomainCommand) -> None:
        state = self._global_node_domains.get(command.domain_id)
        if state is None or state.command.generation != command.generation:
            raise RuntimeError("Global CommDomain commit requires a matching imported domain")
        if command.attachments:
            if command.attachments != state.command.attachments:
                raise RuntimeError("Global CommDomain commit attachments conflict with prepare")
        elif state.command.attachments:
            command = replace(command, attachments=state.command.attachments)
        if state.phase is not GlobalDomainPhase.IMPORT or state.view is None:
            raise RuntimeError("Global CommDomain commit requires IMPORT completion")
        if (
            self._global_domain_command_identity(state.command) != self._global_domain_command_identity(command)
            or state.command.descriptors != command.descriptors
        ):
            raise RuntimeError("Global CommDomain commit command conflicts with IMPORT")
        state.phase = GlobalDomainPhase.COMMIT
        state.view._committed = True  # noqa: SLF001 -- session owns the transaction

    def _release_global_domain_node(
        self, command: GlobalDomainReleaseCommand, *, suppress_errors: bool = False
    ) -> None:
        state = self._global_node_domains.get(command.domain_id)
        if state is None:
            return
        if state.command.generation != command.generation:
            raise RuntimeError("Global CommDomain release generation mismatch")
        # Invalidate the public view before the first destructive child call.
        # If fanout later fails, callers can no longer retrieve or use pointers
        # into windows that may already have been released on some local ranks.
        state.phase = GlobalDomainPhase.ABORT
        if state.view is not None:
            state.view._committed = False  # noqa: SLF001 -- node session owns the transaction
        with self._child_prov_lock:
            self._drop_domain_allocs(self._global_domain_provenance_id(command.domain_id))
        if self._worker is None:
            return
        errors: list[BaseException] = []
        local_members = tuple(
            member for member in state.command.members if member.domain_rank in state.prepared_domain_ranks
        )
        for member in local_members:
            payload = bytearray(LOCAL_RELEASE_REQUEST.size)
            LOCAL_RELEASE_REQUEST.pack_into(
                payload,
                0,
                LOCAL_DOMAIN_MAGIC,
                GLOBAL_DOMAIN_VERSION,
                command.domain_id,
                command.generation,
            )
            try:
                self._worker.control_payload(
                    WorkerType.NEXT_LEVEL,
                    member.local_worker_id,
                    CTRL_GLOBAL_DOMAIN_RELEASE,
                    payload,
                    self._py_control_timeout_s,
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        if not errors:
            self._global_node_domains.pop(command.domain_id, None)
        if errors and not suppress_errors:
            raise RuntimeError(f"Global CommDomain node release failed: {errors[0]}") from errors[0]

    def _release_all_global_domain_nodes(self) -> None:
        for state in list(self._global_node_domains.values())[::-1]:
            try:
                self._release_global_domain_node(
                    GlobalDomainReleaseCommand(state.command.domain_id, state.command.generation)
                )
            except Exception as exc:  # noqa: BLE001
                # A node release that fails during teardown leaves device state
                # nothing else tracks; the same boundary as the ABORT fan-out
                # applies.
                self._record_unreclaimable(
                    f"Global CommDomain node release failed for domain_id={state.command.domain_id}; "
                    "backend windows may remain mapped",
                    exc,
                )

    def _get_global_domain(self, domain_id: int) -> GlobalCommDomainView:
        state = self._global_node_domains.get(int(domain_id))
        if state is None or state.phase is not GlobalDomainPhase.COMMIT or state.view is None:
            raise KeyError(f"Global CommDomain {domain_id} is not committed on this L3 node")
        return state.view

    def _copy_global_domain_node(self, command: GlobalDomainCopyCommand, *, copy_to_device: bool) -> bytes:
        state = self._global_node_domains.get(command.domain_id)
        if (
            state is None
            or state.command.generation != command.generation
            or state.phase is not GlobalDomainPhase.COMMIT
        ):
            raise RuntimeError("Global CommDomain copy requires a committed live domain")
        if command.domain_rank >= len(state.command.members):
            raise ValueError("Global CommDomain copy rank is out of range")
        member = state.command.members[command.domain_rank]
        if member.local_worker_id not in state.contexts:
            raise RuntimeError("Global CommDomain copy rank is not local to this L3 node")
        request_size = LOCAL_COPY_REQUEST.size + (command.nbytes if copy_to_device else 0)
        reply_size = LOCAL_COPY_REPLY.size + (command.nbytes if not copy_to_device else 0)
        payload = bytearray(max(request_size, reply_size))
        LOCAL_COPY_REQUEST.pack_into(
            payload,
            0,
            LOCAL_DOMAIN_MAGIC,
            GLOBAL_DOMAIN_VERSION,
            command.domain_id,
            command.generation,
            command.offset,
            command.nbytes,
        )
        if copy_to_device:
            payload[LOCAL_COPY_REQUEST.size : request_size] = command.data
        assert self._worker is not None
        reply = bytes(
            self._worker.control_payload(
                WorkerType.NEXT_LEVEL,
                member.local_worker_id,
                CTRL_GLOBAL_DOMAIN_COPY_TO if copy_to_device else CTRL_GLOBAL_DOMAIN_COPY_FROM,
                payload,
                self._py_control_timeout_s,
            )
        )
        magic, version, domain_id, generation, nbytes = LOCAL_COPY_REPLY.unpack_from(reply, 0)
        _validate_local_global_header(magic, version, domain_id, generation, operation="copy reply")
        if domain_id != command.domain_id or generation != command.generation or nbytes != command.nbytes:
            raise RuntimeError("Global CommDomain copy reply mismatch")
        if copy_to_device:
            return b""
        return reply[LOCAL_COPY_REPLY.size : LOCAL_COPY_REPLY.size + command.nbytes]

    @staticmethod
    def _local_global_domain_response_capacity(control_name: int, payload: bytes) -> int:
        from .remote_l3_protocol import ControlName  # noqa: PLC0415

        control = ControlName(control_name)
        if control is ControlName.COMM_INIT:
            return struct.calcsize("<III") + 4 + GLOBAL_DOMAIN_MAX_STRING_BYTES
        if control is ControlName.ALLOC_DOMAIN:
            command = decode_domain_command(payload)
            if command.phase is GlobalDomainPhase.PREPARE_EXPORT:
                return 4 + GLOBAL_DOMAIN_MAX_RANKS * GLOBAL_DOMAIN_DESCRIPTOR_BYTES
            return 0
        if control is ControlName.COPY_FROM_DOMAIN:
            command = decode_copy_command(payload, include_data=False)
            return 4 + command.nbytes
        return 0

    def _local_global_domain_control(self, worker_id: int, control_name: int, payload: bytes) -> bytes:
        if self._worker is None:
            raise RuntimeError("Global CommDomain control requires a ready hierarchical Worker")
        if worker_id not in self._next_level_worker_ids:
            raise ValueError(f"Global CommDomain worker {worker_id} is not a local L3 worker")
        response_capacity = self._local_global_domain_response_capacity(control_name, payload)
        capacity = max(len(payload), response_capacity)
        staged = bytearray(_LOCAL_GLOBAL_CONTROL_HEADER.size + capacity)
        _LOCAL_GLOBAL_CONTROL_HEADER.pack_into(staged, 0, int(control_name), len(payload), 0)
        start = _LOCAL_GLOBAL_CONTROL_HEADER.size
        staged[start : start + len(payload)] = payload
        reply = bytes(
            self._worker.control_payload(
                WorkerType.NEXT_LEVEL,
                int(worker_id),
                _CTRL_GLOBAL_DOMAIN_NODE,
                staged,
                self._py_control_timeout_s,
            )
        )
        reply_control, reply_request_size, response_size = _LOCAL_GLOBAL_CONTROL_HEADER.unpack_from(reply, 0)
        if reply_control != int(control_name) or reply_request_size != len(payload) or response_size > capacity:
            raise RuntimeError("local Global CommDomain control reply is invalid")
        return reply[start : start + response_size]

    def _global_domain_control(
        self, worker_id: int, control_name: int, payload: bytes, *, group: bool = False
    ) -> bytes:
        if self._worker is None:
            raise RuntimeError("Global CommDomain control requires a ready hierarchical Worker")
        if worker_id in self._remote_like_worker_ids():
            return bytes(
                self._worker.remote_domain_control(
                    int(worker_id), int(control_name), bytes(payload), group_target=bool(group)
                )
            )
        if worker_id in self._next_level_worker_ids:
            return self._local_global_domain_control(worker_id, control_name, payload)
        raise ValueError(f"Global CommDomain worker {worker_id} is not a registered L3 worker")

    def _mpi_group_for_involved_nodes(self, involved_nodes: tuple[int, ...]) -> _MpiL3GroupRuntime | None:
        involved = set(int(worker_id) for worker_id in involved_nodes)
        for group in self._mpi_l3_groups:
            group_workers = {rank.worker_id for rank in group.ranks}
            if involved == group_workers:
                return group
        return None

    def _mpi_group_control(
        self,
        group: _MpiL3GroupRuntime,
        control_name: int,
        payload: bytes,
    ) -> bytes:
        if not group.ranks:
            raise RuntimeError(f"MPI L3 group {group.group_id} has no ranks")
        # One L4 request enters the rank-0 mailbox with the group-target frame
        # flag set, so every MPI rank receives the same envelope.
        return self._global_domain_control(group.ranks[0].worker_id, control_name, payload, group=True)

    def _allocate_global_domain(  # noqa: PLR0912 -- transaction validation and prepare/import/commit rollback stay ordered
        self,
        *,
        name: str,
        members: tuple[tuple[int, int], ...],
        window_size: int,
        buffers: list[CommBufferSpec],
        retain_after_run: bool,
    ) -> GlobalCommDomainHandle:
        from .remote_l3_protocol import ControlName  # noqa: PLC0415

        if self.level < 4 or self._worker is None:
            raise RuntimeError("allocate_global_domain requires a ready L4+ Worker")
        resources = self._building_run_resources
        if resources is None:
            raise RuntimeError("allocate_global_domain is only valid while a run's graph is being built")
        if not name:
            raise ValueError("allocate_global_domain: name must be non-empty")
        if name in self._live_global_domains:
            raise ValueError(f"allocate_global_domain: domain {name!r} is already live")
        if not members or len(members) > GLOBAL_DOMAIN_MAX_RANKS:
            raise ValueError("allocate_global_domain: members must contain between 1 and 64 devices")
        if len(set(members)) != len(members):
            raise ValueError("allocate_global_domain: members contain duplicate node/local devices")
        if window_size <= 0:
            raise ValueError("allocate_global_domain: window_size must be positive")
        if len({buffer.name for buffer in buffers}) != len(buffers):
            raise ValueError("allocate_global_domain: buffer names must be unique")
        if any(not buffer.name or int(buffer.nbytes) <= 0 for buffer in buffers):
            raise ValueError("allocate_global_domain: buffers require a name and positive nbytes")
        if sum(int(buffer.nbytes) for buffer in buffers) > window_size:
            raise ValueError("allocate_global_domain: buffers exceed window_size")

        nodes = self._resolved_global_nodes()
        profiles: set[str] = set()
        domain_members: list[GlobalDomainMember] = []
        for domain_rank, (node_worker_id, local_worker_id) in enumerate(members):
            node = nodes.get(int(node_worker_id))
            if node is None:
                raise ValueError(f"allocate_global_domain: worker {node_worker_id} is not a registered L3")
            if local_worker_id < 0 or local_worker_id >= len(node.device_ids):
                raise ValueError(
                    f"allocate_global_domain: local worker {local_worker_id} is outside "
                    f"worker {node_worker_id}'s device list"
                )
            profiles.add(node.comm_profile)
            domain_members.append(
                GlobalDomainMember(
                    node_worker_id=int(node_worker_id),
                    local_worker_id=int(local_worker_id),
                    global_device_rank=node.global_device_ranks[int(local_worker_id)],
                    domain_rank=domain_rank,
                )
            )
        if len(profiles) != 1:
            raise ValueError("allocate_global_domain: all participating nodes must use the same comm_profile")
        profile = next(iter(profiles))
        global_buffers = tuple(GlobalDomainBuffer(buffer.name, int(buffer.nbytes)) for buffer in buffers)
        domain_members_tuple = tuple(domain_members)
        involved_nodes = tuple(dict.fromkeys(member.node_worker_id for member in domain_members_tuple))
        attachment_nodes = tuple(sorted(involved_nodes))
        attachment_matrix = self._global_domain_attachment_matrix(
            domain_members_tuple,
            attachment_nodes,
            int(window_size),
        )
        for node_worker_id in involved_nodes:
            node = nodes[node_worker_id]
            resolve_global_comm_capability(
                platform=node.platform,
                profile=node.comm_profile,
                local_device_count=len(node.device_ids),
            )
        topology_bytes = repr(
            (
                self._global_cluster_id,
                profile,
                tuple(
                    (
                        member.node_worker_id,
                        member.local_worker_id,
                        member.global_device_rank,
                        member.domain_rank,
                    )
                    for member in domain_members_tuple
                ),
            )
        ).encode()
        topology_hash = hashlib.sha256(topology_bytes).hexdigest()
        with self._alloc_id_lock:
            self._next_alloc_id += 1
            domain_id = self._next_alloc_id
        generation = 1
        base_command = GlobalDomainCommand(
            phase=GlobalDomainPhase.PREPARE_EXPORT,
            domain_id=domain_id,
            generation=generation,
            name=name,
            profile=profile,
            window_size=int(window_size),
            members=domain_members_tuple,
            buffers=global_buffers,
            attachments=attachment_matrix,
        )

        prepared_nodes: list[int] = []
        mpi_group: _MpiL3GroupRuntime | None = None
        try:
            for node_worker_id in involved_nodes:
                node = nodes[node_worker_id]
                init = GlobalCommInitCommand(
                    cluster_id=self._global_cluster_id,
                    topology_hash=topology_hash,
                    profile=profile,
                    node_rank=node.node_rank,
                    node_count=node.node_count,
                    members=domain_members_tuple,
                )
                result = decode_comm_init_result(
                    self._global_domain_control(node_worker_id, ControlName.COMM_INIT, encode_comm_init(init))
                )
                if (
                    result.profile != profile
                    or result.max_ranks < len(domain_members_tuple)
                    or result.descriptor_bytes != GLOBAL_DOMAIN_DESCRIPTOR_BYTES
                    or result.local_device_count != len(node.device_ids)
                ):
                    raise RuntimeError(f"Global CommDomain COMM_INIT capability mismatch on node {node_worker_id}")

            mpi_group = self._mpi_group_for_involved_nodes(involved_nodes)
            if mpi_group is not None:
                # A full MPI group exchanges descriptors rank-side (the session
                # runner's collective), so one ALLOC_DOMAIN fanout both prepares
                # and imports; the reply already carries the complete table.
                prepared_nodes.extend(involved_nodes)
                reply = self._mpi_group_control(
                    mpi_group,
                    ControlName.ALLOC_DOMAIN,
                    encode_domain_command(base_command),
                )
                descriptors = decode_descriptor_table(reply)
            else:
                descriptor_by_rank: dict[int, GlobalDomainDescriptor] = {}
                for node_worker_id in involved_nodes:
                    prepared_nodes.append(node_worker_id)
                    reply = self._global_domain_control(
                        node_worker_id,
                        ControlName.ALLOC_DOMAIN,
                        encode_domain_command(base_command),
                    )
                    for descriptor in decode_descriptor_table(reply):
                        if descriptor.domain_rank in descriptor_by_rank:
                            raise RuntimeError("Global CommDomain prepare returned a duplicate rank")
                        descriptor_by_rank[descriptor.domain_rank] = descriptor
                descriptors = tuple(descriptor_by_rank[rank] for rank in range(len(domain_members_tuple)))
            validate_descriptor_table(descriptors, rank_count=len(domain_members_tuple), profile=profile)
            if descriptors[0].mapping_size < window_size:
                raise RuntimeError("Global CommDomain backend mapped less than the requested window size")

            commit_command = GlobalDomainCommand(
                phase=GlobalDomainPhase.COMMIT,
                domain_id=domain_id,
                generation=generation,
                name=name,
                profile=profile,
                window_size=int(window_size),
                members=domain_members_tuple,
                buffers=global_buffers,
                descriptors=descriptors,
            )
            if mpi_group is not None:
                self._mpi_group_control(
                    mpi_group,
                    ControlName.ALLOC_DOMAIN,
                    encode_domain_command(commit_command),
                )
            else:
                import_command = GlobalDomainCommand(
                    phase=GlobalDomainPhase.IMPORT,
                    domain_id=domain_id,
                    generation=generation,
                    name=name,
                    profile=profile,
                    window_size=int(window_size),
                    members=domain_members_tuple,
                    buffers=global_buffers,
                    descriptors=descriptors,
                )
                for node_worker_id in involved_nodes:
                    self._global_domain_control(
                        node_worker_id,
                        ControlName.ALLOC_DOMAIN,
                        encode_domain_command(import_command),
                    )
                for node_worker_id in involved_nodes:
                    self._global_domain_control(
                        node_worker_id,
                        ControlName.ALLOC_DOMAIN,
                        encode_domain_command(commit_command),
                    )
        except BaseException:
            abort_command = GlobalDomainCommand(
                phase=GlobalDomainPhase.ABORT,
                domain_id=domain_id,
                generation=generation,
                name=name,
                profile=profile,
                window_size=int(window_size),
                members=domain_members_tuple,
                buffers=global_buffers,
            )
            if mpi_group is not None and prepared_nodes:
                try:
                    self._mpi_group_control(
                        mpi_group,
                        ControlName.ALLOC_DOMAIN,
                        encode_domain_command(abort_command),
                    )
                except BaseException as abort_error:  # noqa: BLE001
                    # The domain is not registered on this path, so no run fence
                    # and no close() sweep can reach whatever the failed ABORT
                    # leaves mapped. Refusing further work is the only boundary
                    # that keeps a later run from being admitted as if the
                    # teardown had succeeded.
                    self._record_unreclaimable(
                        f"Global CommDomain {name!r} ABORT cleanup failed for MPI group "
                        f"{mpi_group.group_id!r}; backend windows may remain mapped",
                        abort_error,
                    )
            else:
                for node_worker_id in prepared_nodes:
                    try:
                        self._global_domain_control(
                            node_worker_id,
                            ControlName.ALLOC_DOMAIN,
                            encode_domain_command(abort_command),
                        )
                    except BaseException as abort_error:  # noqa: BLE001
                        # The domain is not registered on this path, so no run fence
                        # and no close() sweep can reach whatever the failed ABORT
                        # leaves mapped. Refusing further work is the only boundary
                        # that keeps a later run from being admitted as if the
                        # teardown had succeeded.
                        self._record_unreclaimable(
                            f"Global CommDomain {name!r} ABORT cleanup failed for node worker "
                            f"{node_worker_id}; backend windows may remain mapped",
                            abort_error,
                        )
            raise

        handle = GlobalCommDomainHandle(
            name=name,
            members=domain_members_tuple,
            buffers=global_buffers,
            domain_id=domain_id,
            generation=generation,
            mapping_size=descriptors[0].mapping_size,
            retain_after_run=retain_after_run,
            _release_fn=lambda released, owner=resources: self._release_global_domain_handle(released, owner),
            attachments=attachment_matrix,
        )
        self._live_global_domains[name] = handle
        resources.live_global_domains[name] = handle
        resources.requires_ordered_cleanup = True
        return handle

    def _release_global_domain_handle(
        self,
        handle: GlobalCommDomainHandle,
        resources: _RunResources,
    ) -> None:
        if self._worker is None:
            return
        with resources.domain_lock:
            if resources.live_global_domains.get(handle.name) is handle:
                resources.live_global_domains.pop(handle.name)
            if self._live_global_domains.get(handle.name) is handle:
                self._live_global_domains.pop(handle.name)
            if not resources.retired:
                resources.pending_release_global_domains.append(handle)
                resources.requires_ordered_cleanup = True
                return
        # A retained handle may be released while a later run is being built.
        # Its allocation run is already retired then, so bind the release to the
        # current run's fence without losing the original-run capture that
        # protects release calls made after submit returns.
        current_resources = self._building_run_resources
        if current_resources is not None and current_resources is not resources:
            with current_resources.domain_lock:
                if not current_resources.retired:
                    current_resources.pending_release_global_domains.append(handle)
                    current_resources.requires_ordered_cleanup = True
                    return
        self._free_global_domain_after_fence(handle)

    def _free_global_domain_after_fence(self, handle: GlobalCommDomainHandle) -> None:
        if handle.freed:
            return
        try:
            self._release_global_domain_now(handle)
            handle._freed = True  # noqa: SLF001 -- runtime owns this transition
            self._failed_global_domain_releases.pop(handle.domain_id, None)
        except Exception:
            self._failed_global_domain_releases[handle.domain_id] = handle
            raise

    def _release_global_domain_now(self, handle: GlobalCommDomainHandle) -> None:
        with self._global_domain_free_mu:
            if handle.domain_id in self._global_domain_free_results:
                failure = self._global_domain_free_results[handle.domain_id]
                if failure is not None:
                    raise failure
                return
            try:
                self._release_global_domain_claimed(handle)
            except BaseException as exc:
                self._global_domain_free_results[handle.domain_id] = exc
                raise
            self._global_domain_free_results[handle.domain_id] = None

    def _release_global_domain_claimed(self, handle: GlobalCommDomainHandle) -> None:
        from .remote_l3_protocol import ControlName  # noqa: PLC0415

        command = encode_release_command(GlobalDomainReleaseCommand(handle.domain_id, handle.generation))
        errors: list[BaseException] = []
        involved_nodes = tuple(dict.fromkeys(member.node_worker_id for member in handle.members))
        mpi_group = self._mpi_group_for_involved_nodes(involved_nodes)
        if mpi_group is not None:
            try:
                self._mpi_group_control(mpi_group, ControlName.RELEASE_DOMAIN, command)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        else:
            for node_worker_id in involved_nodes:
                try:
                    self._global_domain_control(node_worker_id, ControlName.RELEASE_DOMAIN, command)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)
        if errors:
            raise RuntimeError(f"Global CommDomain release failed: {errors[0]}") from errors[0]
        if self._live_global_domains.get(handle.name) is handle:
            self._live_global_domains.pop(handle.name)

    def _execute_pending_global_domain_releases(self, resources: _RunResources) -> None:
        with resources.domain_lock:
            pending = list(resources.pending_release_global_domains)
        self._drain_pending_global_domain_snapshot(resources, pending)

    def _release_all_live_global_domains(
        self,
        resources: _RunResources | None = None,
        *,
        include_retained: bool = True,
    ) -> None:
        live_domains = self._live_global_domains if resources is None else resources.live_global_domains

        def _release(handle: GlobalCommDomainHandle) -> None:
            if handle.retain_after_run and not include_retained:
                return
            handle._released = True  # noqa: SLF001 -- runtime owns this transition
            self._free_global_domain_after_fence(handle)
            if live_domains.get(handle.name) is handle:
                live_domains.pop(handle.name)

        _raise_first(_release, list(live_domains.values())[::-1])

    def _copy_to_global_domain(
        self, handle: GlobalCommDomainHandle, domain_rank: int, data: bytes, offset: int
    ) -> None:
        from .remote_l3_protocol import ControlName  # noqa: PLC0415

        payload = bytes(data)
        member = handle.member(domain_rank)
        command = GlobalDomainCopyCommand(
            domain_id=handle.domain_id,
            generation=handle.generation,
            domain_rank=int(domain_rank),
            offset=int(offset),
            nbytes=len(payload),
            data=payload,
        )
        self._global_domain_control(
            member.node_worker_id,
            ControlName.COPY_TO_DOMAIN,
            encode_copy_command(command, include_data=True),
        )

    def _copy_from_global_domain(
        self, handle: GlobalCommDomainHandle, domain_rank: int, nbytes: int, offset: int
    ) -> bytes:
        from .remote_l3_protocol import ControlName  # noqa: PLC0415

        member = handle.member(domain_rank)
        command = GlobalDomainCopyCommand(
            domain_id=handle.domain_id,
            generation=handle.generation,
            domain_rank=int(domain_rank),
            offset=int(offset),
            nbytes=int(nbytes),
        )
        reply = self._global_domain_control(
            member.node_worker_id,
            ControlName.COPY_FROM_DOMAIN,
            encode_copy_command(command, include_data=False),
        )
        return decode_copy_result(reply)

    def _release_all_live_domains(self, resources: _RunResources | None = None) -> None:
        """Best-effort release of every still-live domain handle (LIFO).

        Called from the end-of-run sweep (after the owning run's pending
        releases) and from ``Worker.close``.  Skips the deferred-release
        queue because by the time this runs, drain has already happened —
        synchronous release of leftover handles is safe.  Falls back to
        immediate backend free + drop from ``_live_domains`` on each handle.

        Every handle is attempted and the first error is raised once they all
        have been. A failed handle stays in ``_live_domains`` so close()'s
        journal can retry it without losing the ownership record.
        """
        live_domains = self._live_domains if resources is None else resources.live_domains

        def _release(handle: CommDomainHandle) -> None:
            # Mark released first (flips handle._released so further indexing
            # raises), then synchronously free.  The handle is not in the
            # deferred-release queue, so we use the direct path.
            if not handle.released:
                handle._released = True  # noqa: SLF001 -- runtime owns the transition
            self._release_domain_now(handle)
            handle._freed = True  # noqa: SLF001
            if live_domains.get(handle.name) is handle:
                live_domains.pop(handle.name)

        _raise_first(_release, list(live_domains.values())[::-1])

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

    def _child_prov_worker_lock(self, worker_id: int) -> threading.Lock:
        """Return the lock serializing native device ops on *worker_id* alone.

        Acquired *before* ``_child_prov_lock`` wherever both are needed; never the
        other way round, so the two can never deadlock.
        """
        with self._child_prov_lock:
            lock = self._child_prov_worker_locks.get(int(worker_id))
            if lock is None:
                lock = threading.Lock()
                self._child_prov_worker_locks[int(worker_id)] = lock
            return lock

    def _record_device_alloc(self, handle: Buffer, *, domain_allocation_id: int | None = None) -> None:
        """Make ``handle`` a live device allocation operands may name. Caller holds ``_child_prov_lock``.

        ``domain_allocation_id`` ties the identity to the CommDomain allocation that minted it, so
        that domain's release revokes it. A ``malloc``-backed handle has none.
        """
        self._child_alloc[handle.identity] = handle
        if domain_allocation_id is not None:
            self._domain_members.setdefault(domain_allocation_id, set()).add(handle.identity)

    def _drop_device_alloc(self, identity: CanonicalIdentity) -> None:
        """Revoke one allocation. Caller holds ``_child_prov_lock``.

        Called BEFORE the native free (safety-first), so an interrupted free never leaves an
        identity resolvable to an address that is already gone.
        """
        self._child_alloc.pop(identity, None)

    def _drop_domain_allocs(self, allocation_id: int) -> None:
        """Revoke every identity a CommDomain allocation minted. Caller holds ``_child_prov_lock``.

        Runs at the start of the domain's physical release, before the backend free (see
        ``_release_domain_now``). The backend release is keyed by allocation id and takes no pointer,
        so revoking by allocation id is the same unit the release itself uses.
        """
        for identity in self._domain_members.pop(allocation_id, set()):
            self._child_alloc.pop(identity, None)

    def _require_freeable(self, handle: Buffer, *, api: str) -> Buffer:
        """The registered allocation ``handle`` names, provided ``free`` is what releases it.

        Caller holds ``_child_prov_lock``. A ``VMM_WINDOW`` identity is refused because that memory
        belongs to the comm backend, which reclaims the whole window collectively -- freeing it
        through the device allocator would release memory a different allocator owns.
        """
        registered = self._child_alloc.get(handle.identity)
        if registered is None:
            raise ValueError(
                f"Worker.{api}: {handle.identity} is not a live device allocation on this worker "
                f"(never allocated here, or already freed)"
            )
        if registered.backend_kind is not BackendKind.DEVICE_MALLOC:
            raise ValueError(
                f"Worker.{api}: {handle.identity} is a CommDomain buffer; it is released with its "
                f"domain, never through {api}"
            )
        return registered

    def _owner_endpoint_context(self) -> ImportContext:
        """This Worker as a consumer endpoint: a host process that owns this nonce's chip children.

        ``device_owner_instance_id`` is what separates it from a SUB child, the other host
        endpoint: neither can map a device backing, and only this one can still reach one, by
        driving the control mailbox of the chip that owns it.
        """
        return ImportContext(is_host_endpoint=True, device_owner_instance_id=self._owner_instance_id)

    def _device_worker_for(self, identity: CanonicalIdentity, *, api: str) -> int:
        """The next-level worker ``identity``'s device allocation lives on.

        Lock selection only -- ``require`` re-resolves the same identity under that lock and is the
        authorization. Reading the worker off the caller's handle instead would let a wrong id
        serialize this op against another chip's queue.
        """
        with self._child_prov_lock:
            bound = self._child_alloc.get(identity)
            if bound is None:
                raise ValueError(
                    f"Worker.{api}: {identity} is not a live device allocation on this worker "
                    f"(never allocated here, already freed, or its domain was released)"
                )
            return int(bound.owner_worker_id)

    def require(
        self,
        identity: CanonicalIdentity,
        capability: BufferCapability,
        nbytes: int,
        *,
        offset: int = 0,
        api: str,
    ) -> Buffer:
        """Authorize ``capability`` over ``[offset, offset + nbytes)`` of the device backing
        ``identity`` names, and return the registered ``Buffer`` it resolves to.

        The identity is the input and the pointer is the *result*, which is the whole point: a
        pointer says nothing about which allocation it currently belongs to, so a handle whose
        backing was freed and whose address the allocator then handed to a new allocation passes a
        pointer check and fails this one. ``buffer_id`` never repeats within an owner incarnation and
        ``generation`` separates incarnations, so a stale identity resolves to nothing.

        The mechanism is not assumed here: ``select_adapter`` decides it from the registered
        descriptor and this Worker's own endpoint, the same judgment a consumer endpoint runs. A
        Worker is a host endpoint that owns chip children, so a device backing minted by this Worker
        resolves to ``OWNER_DELEGATED_COPY`` -- it cannot hold a device VA, but it can ask the chip
        that owns the allocation to copy, which is exactly what this authorizes. The rights are that
        adapter's, narrowed by the allocation's own ``AccessMode``, so ``DIRECT_LOAD`` /
        ``DIRECT_STORE`` are absent: they would name a dereference no caller on this side performs.

        The extent is the registered handle's, never the caller's: a copy may cover any range inside
        the allocation, and ``offset`` is measured from the allocation's own base, so naming a
        sub-range never requires naming an address.
        """
        if nbytes < 0:
            raise ValueError(f"Worker.{api}: nbytes must be non-negative, got {nbytes}")
        if offset < 0:
            raise ValueError(f"Worker.{api}: offset must be non-negative, got {offset}")
        with self._child_prov_lock:
            handle = self._child_alloc.get(identity)
            if handle is None:
                raise ValueError(
                    f"Worker.{api}: {identity} is not a live device allocation on this worker "
                    f"(never allocated here, already freed, or its domain was released)"
                )
            kind, _profile = select_adapter(handle.to_descriptor(), self._owner_endpoint_context())
            granted = capabilities_for_adapter(kind, handle.access)
            if capability not in granted:
                raise ValueError(
                    f"Worker.{api}: {identity} was allocated {handle.access.name}, which grants "
                    f"{sorted(c.value for c in granted)} -- not {capability.value}"
                )
            extent = int(handle.nbytes)
            if offset > extent or nbytes > extent - offset:
                raise ValueError(
                    f"Worker.{api}: [{offset}, {offset}+{nbytes}) overruns {identity}, which spans {extent}"
                )
            return handle

    @staticmethod
    def _device_identities_in_args(args: Any) -> list[tuple[CanonicalIdentity, int]]:
        """``(identity, arg_index)`` for every device arg -- what the dispatch guard validates.

        A ``DEVICE_MALLOC`` (worker device malloc) or ``VMM_WINDOW`` (domain-carved) ref names an
        allocation behind a chip boundary, so its identity is what the owner can resolve. Host-backed
        refs (POSIX / fork shm) are not owner-side allocations and contribute nothing.
        """
        out: list[tuple[CanonicalIdentity, int]] = []
        for i in range(args.tensor_count()):
            desc = args.tensor(i).buffer
            if desc.backend_kind in (BackendKind.DEVICE_MALLOC, BackendKind.VMM_WINDOW):
                out.append((desc.identity, i))
        return out

    @staticmethod
    def _identities_in_args(args: Any) -> set[CanonicalIdentity]:
        """Every tensor arg's identity in ``args``."""
        return {args.tensor(i).buffer.identity for i in range(args.tensor_count())}

    def _record_touched_identities(self, args: Any) -> None:
        """Add every tensor arg's identity in ``args`` to the current run's touched set.

        A no-op when no run is being built (``_building_run_resources is None``) — tracking is
        opportunistic, only meaningful inside the run context of the four orchestrator dispatch
        entry points that carry Tensor args to another process (``submit_next_level``,
        ``submit_next_level_group``, ``submit_sub``, ``submit_sub_group``), per the
        ``current_resources = self._building_run_resources; if ... is not None`` idiom used
        elsewhere for the same "attach to the open run, if any" shape.

        The set over-approximates in one direction on purpose: every caller records before
        ``_admit_task_submission``, which can itself raise on a sticky ordered-cleanup failure, so an
        identity can be recorded for a task that never went out. That costs a spurious
        ``release_buffer`` refusal, recoverable through ``close()``, where recording after admission
        would leave a dispatched task's identity unrecorded and let a release unlink under it. The
        group entry points additionally validate every member before recording any, since a rejected
        member dispatches nothing at all.
        """
        resources = self._building_run_resources
        if resources is None:
            return
        resources.touched_identities.update(self._identities_in_args(args))

    def _child_prov_check_dispatch(
        self, device_args: list[tuple[CanonicalIdentity, int]], target_worker_id: int, *, api: str
    ) -> None:
        """Validate every device arg against the worker it is dispatched to.

        The identity carries which worker owns the allocation, so "wrong worker" is an equality on
        the registered handle rather than a lookup keyed by the pair.
        """
        if not device_args:
            return
        with self._child_prov_lock:
            for identity, arg_index in device_args:
                handle = self._child_alloc.get(identity)
                if handle is None or int(handle.owner_worker_id) != target_worker_id:
                    raise ValueError(
                        f"orch.{api}: device argument (arg {arg_index}, {identity}) is not a live "
                        f"allocation on target worker {target_worker_id} (wrong worker, or stale)"
                    )

    def _require_local_next_level_target(self, worker_id: int, *, api: str) -> None:
        """Reject a local callable pinned to a remote NEXT_LEVEL worker.

        A LOCAL_PYTHON / LOCAL_CHIP callable is installed only in the local
        children's registries; a remote-L3 worker's manifest carries only its
        dispatcher callables, so routing a local digest to a remote worker id
        fails asynchronously with an unknown-hashid on the remote endpoint. The
        C++ target check only rejects unregistered ids, so a registered remote
        worker slips through — this guards that hole.
        """
        if worker_id in self._remote_like_worker_ids():
            raise ValueError(
                f"orch.{api}: worker {worker_id} is a remote NEXT_LEVEL worker; a local callable "
                f"must target a local child (remote workers only run RemoteCallable dispatches)"
            )

    def _clear_child_prov(self) -> None:
        """Drop the whole device-allocation table (close-path hygiene).

        Revoking authorization only: the allocations themselves die with the chip children this
        teardown is already reaping.
        """
        with self._child_prov_lock:
            self._child_alloc.clear()
            self._domain_members.clear()

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

    def malloc(self, size: int) -> Buffer:
        """Allocate device memory on this L2 worker's own chip; returns a DEVICE_MALLOC ``Buffer``.

        Name a task arg with ``handle.tensor(shapes, dtype)`` and release with ``worker.free(handle)``. L3+
        allocates child device memory with ``alloc_child_tensor(worker_id, ...)`` instead — a Worker is
        the only allocator, the Orchestrator never allocates.
        """
        if self.level != 2:
            raise TypeError("worker.malloc is L2-only; at L3+ use worker.alloc_child_tensor(worker_id, ...)")
        with self._operation_lease("malloc"):
            assert self._chip_worker is not None
            # L2 is a single chip; worker_id is meaningless there, so the provenance is keyed
            # on the canonical worker 0.
            with self._child_prov_lock:
                ptr = int(self._chip_worker.malloc(int(size)))
                handle = wrap_device_malloc(
                    ptr,
                    int(size),
                    self._owner_instance_id,
                    self._next_buffer_id(),
                    f"L{self.level}",
                    owner_worker_id=0,
                )
                self._record_device_alloc(handle)
        return handle

    def alloc_child_tensor(self, worker_id: int, shapes: tuple[int, ...], dtype) -> Buffer:
        """Allocate device memory on next-level ``worker_id`` sized for ``shapes`` × ``dtype``; returns a
        DEVICE_MALLOC ``Buffer`` (successor of ``orch.malloc`` + ``child_memory``).

        Called from within an orchestration fn (capture the Worker in the closure). The pointer is
        private to ``worker_id``; name the arg with ``handle.tensor(shapes, dtype)``, dispatch it only to
        that worker, and load host data with ``copy_to``. Not auto-freed at end-of-task.
        """
        nbytes = get_element_size(dtype)
        for s in shapes:
            nbytes *= int(s)
        self._check_chip_worker_id(int(worker_id))
        assert self._worker is not None
        # The lease is re-entrant, so calling this inside the orch fn (the run already holds it) nests
        # safely, and calling it outside a run acquires it fresh.
        with (
            self._operation_lease("alloc_child_tensor"),
            self._device_control_admission("alloc_child_tensor"),
            self._child_prov_worker_lock(int(worker_id)),
        ):
            ptr = int(self._worker.malloc(int(worker_id), int(nbytes)))
            with self._child_prov_lock:
                handle = wrap_device_malloc(
                    ptr,
                    int(nbytes),
                    self._owner_instance_id,
                    self._next_buffer_id(),
                    f"L{self.level}",
                    owner_worker_id=int(worker_id),
                )
                self._record_device_alloc(handle)
        return handle

    def free(self, handle: Buffer) -> None:
        """Free a device ``Buffer`` allocated by ``malloc`` / ``alloc_child_tensor``.

        The operation lease is re-entrant, so an in-run ``orch.free`` that delegates here nests safely.
        """
        wid = int(handle.owner_worker_id)
        # Reject a non-chip target (L4+, or a bad id) before the lease and the fence: a device op is
        # only meaningful on a next-level chip, and an invalid id must fail now rather than after a
        # wait for the FIFO head.
        if self.level != 2:
            self._check_chip_worker_id(wid)
        with self._operation_lease("free"), self._device_control_admission("free"):
            with self._child_prov_worker_lock(wid):
                # Safety-first commit barrier: revoke provenance BEFORE the native free so an async unwind
                # after a successful free can never leave a freed address live. The revoke commits under
                # ``_child_prov_lock``; the native call runs under this worker's lock only, so a free on
                # one chip no longer blocks provenance or device ops on another.
                with self._child_prov_lock:
                    registered = self._require_freeable(handle, api="free")
                    ptr = int(registered.base)
                    self._drop_device_alloc(handle.identity)
                if self.level == 2:
                    assert self._chip_worker is not None
                    self._chip_worker.free(ptr)
                else:
                    assert self._worker is not None
                    self._worker.free(wid, ptr)

    def committed_device_memory(self, worker_id: int = 0) -> int:
        """Total device HBM (bytes) committed by chip worker *worker_id*'s
        ``MemoryAllocator`` (tensors + pooled arenas + runtime buffers; excludes
        HCCL/VMM comm windows). Useful for downstream
        runtimes to subtract simpler's own HBM from their cache budget.

        Level 2 returns the in-process chip worker's committed bytes directly;
        level 3 forwards a ``CTRL_COMMITTED_DEVICE_MEMORY`` query to the forked
        chip child *worker_id* (sum across worker_ids for a multi-chip total).
        """
        with self._operation_lease("committed_device_memory"):
            if self.level == 2:
                assert self._chip_worker is not None
                return int(self._chip_worker.committed_device_memory)
            if not self._chip_shms:
                raise NotImplementedError("committed_device_memory requires at least one forked chip worker")
            self._check_chip_worker_id(worker_id)
            assert self._orch is not None
            return int(self._orch.committed_device_memory(worker_id))

    @staticmethod
    def _check_copy_handle(handle: Buffer, offset: int, nbytes: int, *, writing: bool, side: str, api: str) -> None:
        """Require ``handle`` to be a device backing this copy may legally touch over
        ``[offset, offset + nbytes)``.

        Without this check a transfer longer than the device backing writes past it, and a READ-only
        backing accepts a write.
        """
        if handle.address_space != AddressSpace.DEVICE:
            raise ValueError(
                f"Worker.{api}: expected a DEVICE handle, got {handle.address_space.name} "
                f"({handle.backend_kind.name}); host-to-host copies do not go through this API"
            )
        if handle.backend_kind not in (BackendKind.DEVICE_MALLOC, BackendKind.VMM_WINDOW):
            raise ValueError(f"Worker.{api}: backend {handle.backend_kind.name} is not reachable from this process")
        needed = AccessMode.WRITE if writing else AccessMode.READ
        if handle.access not in (needed, AccessMode.READWRITE):
            raise ValueError(
                f"Worker.{api}: backing grants {handle.access.name} but this direction needs {needed.name}"
            )
        _require_copy_span(int(handle.nbytes), offset, nbytes, side=side, api=api)

    @staticmethod
    def _copy_extent(
        host_nbytes: int, device_offset: int, host_offset: int, nbytes: int | None, *, host_side: str, api: str
    ) -> tuple[int, int, int]:
        """Normalize a copy's two offsets and its length; returns ``(device_offset, host_offset, nbytes)``.

        ``nbytes`` defaults to the rest of the host backing after ``host_offset``: the length has
        always come from the host side, and an offset only moves where that side starts, so
        ``copy_to(dst, src)`` still means the whole host backing.

        The host end is bounded here because nothing downstream bounds it. The device end is not:
        ``require`` bounds it against the *registered* allocation, whose extent is not necessarily
        the caller's handle's.
        """
        device_offset, host_offset = int(device_offset), int(host_offset)
        if host_offset < 0:
            raise ValueError(f"Worker.{api}: {host_side}_offset must be non-negative, got {host_offset}")
        nbytes = max(0, host_nbytes - host_offset) if nbytes is None else int(nbytes)
        _require_copy_span(host_nbytes, host_offset, nbytes, side=host_side, api=api)
        return device_offset, host_offset, nbytes

    @staticmethod
    def _host_side_of_copy(obj, *, writing: bool, api: str) -> tuple[Buffer | None, int, int]:
        """``(buffer_or_None, address, nbytes)`` for the host end of a control-plane copy.

        A ``Buffer`` carries a self-describing descriptor a consumer resolves for itself, so the
        handle is what travels; a raw host object has only an address, which is meaningful in this
        process alone. The decision is by TYPE, never by matching an address against a range of known
        backings — an address range says nothing reliable about who owns memory.
        """
        if not isinstance(obj, Buffer):
            addr, nbytes = host_ptr_nbytes(obj)
            return None, addr, nbytes
        if obj.address_space != AddressSpace.HOST:
            raise ValueError(f"Worker.{api}: the host side must be a HOST handle, got {obj.address_space.name}")
        needed = AccessMode.WRITE if writing else AccessMode.READ
        if obj.access not in (needed, AccessMode.READWRITE):
            raise ValueError(
                f"Worker.{api}: host backing grants {obj.access.name} but this direction needs {needed.name}"
            )
        return obj, int(obj.base), int(obj.nbytes)

    def _require_buffer_host_side(self, host: Buffer | None, api: str) -> Buffer:
        """The host ``Buffer`` a child resolves for an L3+ control-plane copy.

        A forked child reaches host memory only through a backing its ``ImportRegistry`` can
        materialize, so at L3+ the host end of a copy is a handle, never bare memory.
        """
        if host is None:
            raise TypeError(
                f"Worker.{api}: an L{self.level} host side must be a Buffer from create_buffer "
                "(write the payload into buffer.shm.buf); raw host memory is L2-only"
            )
        return host

    def copy_to(self, dst: Buffer, src, *, dst_offset: int = 0, src_offset: int = 0, nbytes: int | None = None) -> None:
        """H2D: copy ``nbytes`` from ``src_offset`` in host ``src`` to ``dst_offset`` in device ``dst``.

        ``src`` is a host ``Buffer``; the chip child resolves both handles through its
        ``ImportRegistry`` and reads the host backing directly. At L2 the chip worker shares this
        process, so a torch tensor or any writable buffer works too.

        ``nbytes`` defaults to the rest of the host side after ``src_offset``, so a plain
        ``copy_to(dst, src)`` still transfers the whole host backing. An offset names a range of the
        allocation ``dst`` already names; there is no way to name a sub-range with a handle built at
        an interior address, because such a handle names no allocation at all.
        """
        host, src_addr, host_nbytes = self._host_side_of_copy(src, writing=False, api="copy_to")
        dst_offset, src_offset, nbytes = self._copy_extent(
            host_nbytes, dst_offset, src_offset, nbytes, host_side="src", api="copy_to"
        )
        self._check_copy_handle(dst, dst_offset, nbytes, writing=True, side="dst", api="copy_to")
        if self.level != 2:
            # Topology before identity: a Worker with no chip children can hold no device
            # allocation at all, so it owes that answer rather than "this identity is not live".
            self._check_chip_worker_id(int(dst.owner_worker_id))
        # Which per-worker lock to take, resolved from the registry rather than read off the handle:
        # holding the wrong worker's lock would serialize this device op against the wrong queue.
        wid = self._device_worker_for(dst.identity, api="copy_to")
        if self.level != 2:
            self._check_chip_worker_id(wid)
            host = self._require_buffer_host_side(host, "copy_to")
        with self._operation_lease("copy_to"), self._device_control_admission("copy_to"):
            with self._child_prov_worker_lock(wid):
                # The authorization is here, inside the lock: a free landing between resolution and
                # the copy would otherwise slip through. The device end is the only end authorized --
                # it names memory behind a chip boundary, so only owner-side state knows whether that
                # identity still resolves to a live allocation. A host end is either a Buffer the
                # child materializes for itself (L3+) or memory already mapped in this process (L2),
                # and the second is reachable by the caller with or without this API.
                backing = self.require(dst.identity, BufferCapability.COPY_TO, nbytes, offset=dst_offset, api="copy_to")
                if self.level == 2:
                    # No fork: the chip worker runs in this process, so the host address is valid and
                    # both offsets are applied here.
                    assert self._chip_worker is not None
                    self._chip_worker.copy_to(
                        _copy_span_base(int(backing.base), dst_offset),
                        _copy_span_base(src_addr, src_offset),
                        nbytes,
                    )
                else:
                    assert self._worker is not None
                    assert host is not None
                    self._worker.copy_to(wid, dst.to_descriptor(), host.to_descriptor(), nbytes, dst_offset, src_offset)

    def copy_from(
        self, dst, src: Buffer, *, dst_offset: int = 0, src_offset: int = 0, nbytes: int | None = None
    ) -> None:
        """D2H: copy ``nbytes`` from ``src_offset`` in device ``src`` to ``dst_offset`` in host ``dst``.

        ``dst`` is a host ``Buffer``; the chip child resolves both handles through its
        ``ImportRegistry`` and writes the host backing directly. At L2 the chip worker shares this
        process, so a torch tensor or any writable buffer works too.

        ``nbytes`` defaults to the rest of the host side after ``dst_offset``, so a plain
        ``copy_from(dst, src)`` still transfers a whole host backing's worth.
        """
        host, dst_addr, host_nbytes = self._host_side_of_copy(dst, writing=True, api="copy_from")
        src_offset, dst_offset, nbytes = self._copy_extent(
            host_nbytes, src_offset, dst_offset, nbytes, host_side="dst", api="copy_from"
        )
        self._check_copy_handle(src, src_offset, nbytes, writing=False, side="src", api="copy_from")
        if self.level != 2:
            # Topology before identity: a Worker with no chip children can hold no device
            # allocation at all, so it owes that answer rather than "this identity is not live".
            self._check_chip_worker_id(int(src.owner_worker_id))
        # Which per-worker lock to take, resolved from the registry rather than read off the handle:
        # holding the wrong worker's lock would serialize this device op against the wrong queue.
        wid = self._device_worker_for(src.identity, api="copy_from")
        if self.level != 2:
            self._check_chip_worker_id(wid)
            host = self._require_buffer_host_side(host, "copy_from")
        with self._operation_lease("copy_from"), self._device_control_admission("copy_from"):
            with self._child_prov_worker_lock(wid):
                # The authorization is here, inside the lock: a free landing between resolution and
                # the copy would otherwise slip through. The device end is the only end authorized --
                # it names memory behind a chip boundary, so only owner-side state knows whether that
                # identity still resolves to a live allocation. A host end is either a Buffer the
                # child materializes for itself (L3+) or memory already mapped in this process (L2),
                # and the second is reachable by the caller with or without this API.
                backing = self.require(
                    src.identity, BufferCapability.COPY_FROM, nbytes, offset=src_offset, api="copy_from"
                )
                if self.level == 2:
                    # No fork: the chip worker runs in this process, so the host address is valid and
                    # both offsets are applied here.
                    assert self._chip_worker is not None
                    self._chip_worker.copy_from(
                        _copy_span_base(dst_addr, dst_offset),
                        _copy_span_base(int(backing.base), src_offset),
                        nbytes,
                    )
                else:
                    assert self._worker is not None
                    assert host is not None
                    self._worker.copy_from(
                        wid, host.to_descriptor(), src.to_descriptor(), nbytes, dst_offset, src_offset
                    )

    # ------------------------------------------------------------------
    # Post-fork zero-copy host buffers
    # ------------------------------------------------------------------

    def create_buffer(self, nbytes: int) -> Buffer:
        """Allocate a shared ``Buffer`` owned by this Worker (P1-B).

        The backing is a POSIX shm; the Buffer carries a typed canonical identity and a
        self-describing descriptor, so a consumer can resolve it with no prior handshake: the
        descriptor travels embedded in every ``Tensor`` built over this Buffer and the consumer
        materializes it lazily on first receipt (map-once, keyed by canonical identity). At L3+ that
        consumer is a forked child; at L2 (a leaf, no children) the Worker itself materializes the
        tensor in-process on ``run``. Build a tensor over ``buffer.shm.buf`` with the buffer protocol.
        Not thread-safe against a concurrent run/create/free on the same Worker.
        """
        if self.level < 2:
            raise TypeError("create_buffer requires a level >= 2 Worker")
        with self._operation_lease("create_buffer"):
            return self._create_buffer_locked(int(nbytes))

    def alloc_shared_tensor(self, shapes: tuple[int, ...], dtype) -> Buffer:
        """Allocate a runtime-managed intermediate buffer (the ``Tensor`` form of ``orch.alloc``).

        Called inside an orchestration fn. The backing comes from the orchestrator's HeapRing
        (MAP_SHARED, visible to forked children) and is **auto-reclaimed** once every downstream consumer
        has completed and the scope ends — no manual free. Returns a ``FORK_SHM`` ``Buffer`` whose
        canonical identity is registered in the tensormap so a view of it (``handle.tensor(shapes, dtype)``)
        dependency-wires to this producer slot. Chip-A→chip-B intermediates: name it as an OUTPUT of the
        producing task and an INPUT of the consumer.
        """
        assert self._orch is not None, "alloc_shared_tensor requires an L3+ orchestration context"
        nbytes = get_element_size(dtype)
        for s in shapes:
            nbytes *= int(s)
        oid, buffer_id, path = self._owner_instance_id, self._next_buffer_id(), f"L{self.level}"
        identity = CanonicalIdentity(oid, buffer_id)
        va = int(self._orch._o.alloc(list(int(s) for s in shapes), dtype, identity))
        # Wrap the ring VA under the SAME identity: the child materializes to that VA (fork-inherited,
        # MAP_SHARED read-write) and infer_deps keys the ref to the slot registered above.
        return wrap_fork_inherited(
            va,
            int(nbytes),
            oid,
            buffer_id,
            path,
            access=AccessMode.READWRITE,
            backend_kind=BackendKind.FORK_SHM,
        )

    def make_tensor_arg(self, tensor, shapes: tuple[int, ...], dtype: int, *, strides: tuple[int, ...] | None = None):
        """Name a **pre-fork** host tensor as a ``Tensor`` over a memoized ``FORK_SHM`` handle.

        The torch (or buffer-protocol) tensor MUST be allocated before ``init()`` so its VA is
        fork-inherited by the children (the mainline "fork-inherited" contract). A ``share_memory_()``
        tensor is MAP_SHARED — read-write across the fork, so usable as an OUTPUT the parent reads back;
        a plain tensor is COW read-only (input only). The handle is memoized by the tensor's storage
        base, so every ref over the same storage shares one canonical identity and dependencies key on
        it; the ``byte_offset`` this computes is what then separates two views that do not intersect.
        At L2 (no fork) any host tensor works. ``dtype`` is the ``DataType`` int value.
        """
        untyped_storage = getattr(tensor, "untyped_storage", None)
        if callable(untyped_storage):
            st = untyped_storage()
            base, nbytes = int(st.data_ptr()), int(st.nbytes())
            byte_offset = int(tensor.data_ptr()) - base  # the view's start within its storage
        else:
            base, nbytes = host_ptr_nbytes(tensor)
            byte_offset = 0
        # Memoized per storage base so every view of one storage shares an identity and their
        # dependencies key together — a real dependency between two views cannot then hide behind a
        # differing offset, and the byte ranges tell the intersecting pairs from the disjoint ones
        # under that one key. The allocator reuses addresses, though, so a hit whose size no
        # longer matches is a *different* storage that happens to sit where the last one did: it must
        # get a fresh identity, or the two would fuse into one node in the dependency graph.
        handle = self._fork_tensor_handles.get(base)
        if handle is not None and handle.nbytes != nbytes:
            handle = None
        if handle is None:
            # Copy-on-write only bites when a fork stands between the writer and this process. An
            # L2 leaf consumes its own args in-process, so any host tensor is writable there; at L3+
            # the consumer is a forked child, and only a MAP_SHARED allocation carries its writes
            # back. Measuring this rather than assuming it is what stops a plain tensor from being
            # accepted as an OUTPUT and then silently losing every write in the child.
            shared = self.level == 2 or bool(getattr(tensor, "is_shared", lambda: False)())
            # The backend tag records whether a consumer's writes reach this process, which is what
            # the FORK_COW rejection protects. At L2 the consumer IS this process, so they reach it
            # trivially and FORK_COW's contract — a write splitting into a private copy the owner
            # never sees — is the one that would be false; the tag therefore follows `shared`.
            handle = wrap_fork_inherited(
                base,
                nbytes,
                self._owner_instance_id,
                self._next_buffer_id(),
                f"L{self.level}",
                access=AccessMode.READWRITE if shared else AccessMode.READ,
                backend_kind=BackendKind.FORK_SHM if shared else BackendKind.FORK_COW,
            )
            self._fork_tensor_handles[base] = handle
        return handle.tensor(shapes=tuple(shapes), dtype=dtype, strides=strides, byte_offset=byte_offset)

    def _next_buffer_id(self) -> int:
        with self._registry_lock:
            bid = self._buffer_id_counter
            self._buffer_id_counter += 1
        return bid

    def _reexport(self, source: BufferDescriptor) -> Buffer:
        """Re-export a received backing for forwarding (per-backing, memoized, no map).

        An upper-level ref reaching this worker's orch is forwarded as a handle H' that keeps the
        source's canonical identity unchanged (invariant across every edge, frozen model §5/§8) — H'
        is never mapped here (a downstream compute leaf maps it lazily), and is built once per source
        backing (keyed by identity). Worker-scoped lifetime for now.
        """
        key = source.identity
        handle = self._reexport_by_source.get(key)
        if handle is None:
            handle = re_export(source)
            self._reexport_by_source[key] = handle
        return handle

    def _create_buffer_locked(self, nbytes: int) -> Buffer:
        # An L3+ buffer is consumed by a forked child that lazily maps it, so a childless L3+ buffer
        # can reach no consumer. Every kind of forked child counts: a next-level Worker child maps a
        # POSIX_SHM backing by name exactly as a chip or sub child does, so an L4 whose only children
        # are local L3 Workers can consume one. An L2 leaf has no children and materializes
        # in-process, so it needs none.
        if self.level >= 3 and not self._chip_shms and not self._sub_shms and not self._next_level_shms:
            raise _NoBufferConsumerError(
                "create_buffer requires at least one forked chip, sub, or next-level child (this Worker has none)"
            )
        if nbytes <= 0:
            raise ValueError("create_buffer: nbytes must be positive")
        buffer_id = self._next_buffer_id()
        buffer = create_host_shared_buffer(
            nbytes,
            owner_instance_id=self._owner_instance_id,
            buffer_id=buffer_id,
            owner_worker_path=_format_worker_path(int(self.level)),
            generation=1,
        )
        with self._registry_lock:
            self._buffers[buffer_id] = buffer
        return buffer

    def _close_chip_import_registry(self) -> None:
        """Close the L2 in-process consumer import cache (drops its mapped shm imports)."""
        if self._chip_import_registry is not None:
            self._chip_import_registry.close()
            self._chip_import_registry = None

    def release_buffer(self, buffer: Buffer) -> None:
        """Close + unlink one owner Buffer, drop its registry entry, and tell every descendant to
        drop its own cached import for the identity.

        Rejects outright if any currently in-flight L3+ run (not yet past ``_cleanup_published``)
        sent this identity as a NEXT_LEVEL or SUB Tensor arg, or any in-flight L2 direct-chip run
        sent it — a Buffer never goes away while a dispatched task still names it. All three
        dispatch paths retain: a SUB task maps the identity into a sub-worker process just as a
        NEXT_LEVEL task maps it into a child, so unlinking the backing under either one faults the
        consumer on a segment that no longer has a name.

        The L3+ check takes ``_submit_mu`` first: a handle is visible in ``_accepted_run_handles``
        before its orchestration callback (where ``touched_identities`` gets populated) has run, and
        that callback is what ``_submit_mu`` already serializes graph construction against, so taking
        it here means the check only ever runs between callbacks, never mid-callback with a
        not-yet-complete touched set. ``_abandoned_run_handles`` is scanned in the same block and
        without the ``_cleanup_published`` test: ``_publish_abandoned_run`` sets that flag and drops
        the handle from the accepted set while the run itself stays retained until native teardown
        drains it, so an abandoned run is exactly the case where the flag stops describing whether
        the device is done with the backing. Such a buffer therefore stops being releasable through
        this API for the Worker's remaining life, which strands nothing: ``close()`` reclaims it via
        ``_release_all_buffers`` calling ``Buffer.close()`` directly.

        The L2 check is independent (a separate run-id namespace with no callback to serialize
        against — ``_chip_run_touched_identities`` is written atomically alongside ``_chip_runs``
        under ``_registry_lock`` instead, see ``_submit_l2_locked``), so the two checks run
        sequentially rather than under one shared lock. Neither is checked once ``buffer`` is already
        closed, matching ``Buffer.close()``'s own idempotency.

        The entry survives a failed close, so ``_release_all_buffers`` still reports the leak at
        close() rather than losing it here — the import-cache broadcast only fires once close() has
        actually succeeded, so a failed release never tells a descendant to drop a mapping the owner
        still considers live. The converse does not hold, and is the one asymmetry here: a
        ``Buffer.close()`` whose ``shm.close()`` raises has still unlinked the name (its ``finally``
        runs the unlink), so the backing can be nameless while descendants keep mappings this call
        never told them to drop. A descendant materializing that identity afterwards gets the named
        ``FileNotFoundError`` from ``ImportRegistry.materialize``, not a silent bad mapping.

        The slot is dropped only when it still holds *this* buffer: a buffer_id minted elsewhere can
        collide with a registry key, and evicting the live entry it names would strand that
        backing."""
        if not buffer.closed:
            # Exclusive, not shared: `shared()` would already exclude admission and so
            # satisfy the "never mid-callback" argument above, but this keeps the
            # ordering identical to the plain lock this used to be, and buffer release
            # is not on a hot path, so there is nothing to win by weakening it.
            with self._submit_mu.exclusive(), self._hierarchical_start_cv:
                for handle in self._accepted_run_handles:
                    if not handle._cleanup_published and buffer.identity in handle._resources.touched_identities:
                        raise RuntimeError(f"release_buffer: {buffer.identity} is still referenced by an in-flight run")
                for handle in self._abandoned_run_handles:
                    if buffer.identity in handle._resources.touched_identities:
                        raise RuntimeError(
                            f"release_buffer: {buffer.identity} is still referenced by an abandoned run "
                            f"whose native teardown has not completed"
                        )
            with self._registry_lock:
                for touched in self._chip_run_touched_identities.values():
                    if buffer.identity in touched:
                        raise RuntimeError(
                            f"release_buffer: {buffer.identity} is still referenced by an in-flight L2 run"
                        )
        buffer.close()
        self._release_import_recursive(buffer.identity)
        with self._registry_lock:
            buffer_id = int(buffer.identity.buffer_id)
            if self._buffers.get(buffer_id) is buffer:
                del self._buffers[buffer_id]

    def _release_all_buffers(self) -> None:
        """Close + unlink every owner Buffer (called from close()).

        Children drop their own lazily-mapped imports when their loops exit; the owner unlinks the
        backing shm here. Per-buffer best-effort: one buffer's failure never strands the rest, and the
        first error is raised after all are attempted so close() reports the leak rather than
        swallowing it. A buffer whose close fails keeps its registry entry so the cleanup journal can
        retry it."""
        with self._registry_lock:
            entries = list(self._buffers.items())
        errors: list[BaseException] = []
        for buffer_id, buffer in entries:
            try:
                buffer.close()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                continue
            with self._registry_lock:
                self._buffers.pop(buffer_id, None)
        if errors:
            raise errors[0]

    # ------------------------------------------------------------------
    # run — uniform entry point
    # ------------------------------------------------------------------

    def submit(self, callable, args=None, config=None) -> RunHandle:
        """Submit one task (L2) or one DAG (L3+) and return its completion handle.

        Dispatch:
          - L2: ``callable`` is a ``CallableHandle`` returned by
            ``Worker.register(chip_callable)``. Routes to the private slot
            carried by the handle and returns a live completion handle.
          - L3+: ``callable`` is a Python orch fn invoked with the
            ``Orchestrator`` handle. Graph construction completes synchronously;
            device completion is reported by the returned handle.

        ``args``  : TaskArgs (optional)
        ``config``: CallConfig (optional, default-constructed if None)

        Graph construction remains serialized. How many runs may be admitted is
        the depth this worker's backends negotiated, not a constant: at the
        negotiated depth two, one active plus one prepared run are permitted and
        a third submission blocks before invoking its graph callback; where a
        backend publishes depth one, the *second* submission already blocks
        there. A5 tensor-map-and-ring-buffer publishes depth two, while its
        local endpoint retains one mailbox frame and serial device execution;
        A5 host-build-graph publishes no contract and stays at depth one. A
        caller whose first run only completes because a later callback runs
        would deadlock on a depth-one backend. Completion and cleanup stay
        attached to each handle.
        """
        with self._operation_lease("submit"):
            return self._submit_locked(callable, args, config)

    def run(self, callable, args=None, config=None) -> None:
        """Execute one task or DAG synchronously as ``submit(...).wait()``.

        Per-stage run timing (host wall, on-NPU device wall +
        AICPU phase breakdown) is no longer returned — the platform emits it as
        ``[STRACE]`` log markers from each L2 ``simpler_run``, so the L3
        dispatcher and its L2 children are observed uniformly. Parse the markers
        with ``simpler_setup.tools.strace_timing`` (see
        ``docs/dfx/host-trace.md``).
        """
        self.submit(callable, args=args, config=config).wait()

    def _submit_locked(self, callable, args, config) -> RunHandle:
        cfg = config if config is not None else CallConfig()

        if self.level == 2:
            assert self._chip_worker is not None
            state = self._resolve_handle(callable, expected_namespace="LOCAL_CHIP")
            return self._submit_l2_locked(state.slot_id, args, cfg)

        with self._submit_mu.exclusive():
            # Graph callbacks stay serialized, so a predecessor's callback has
            # always returned by the time we get here — which is what makes the
            # decision below a fact about that run rather than a guess about
            # this one.
            #
            # Native begin_run owns depth-two backpressure and slot generations
            # for runs that only dispatch tasks. A predecessor whose cleanup
            # itself touches the device gets stricter treatment: its teardown is
            # mailbox control that the whole-run FIFO cannot order against this
            # run's control, so we wait for that teardown and this worker
            # degrades to depth one for exactly those runs.
            predecessor = self._cleanup_bearing_predecessor()
            if predecessor is not None:
                predecessor._wait_for_handoff()
            # Re-check under _submit_mu, after the wait. Two submissions can
            # both pass the lease's admission check before either reaches here,
            # and the poison may have been published by the cleanup this call
            # just waited on — the check at lease time cannot have seen it.
            self._require_no_ordered_cleanup_failure("submit")
            return self._submit_l3_locked(callable, args, cfg)

    def _record_unreclaimable(self, message: str, cause: BaseException | None = None) -> RuntimeError:
        """Refuse all further work on this worker, and return the reason.

        For device state that no cleanup can reach: nothing tracks it, so there
        is no handle for a run's fence to fail on and no sweep that could
        retry it. First-wins, like every other terminal error here.
        """
        leaked = RuntimeError(message)
        if cause is not None:
            leaked.__cause__ = cause
        with self._hierarchical_start_cv:
            if self._ordered_cleanup_error is None:
                self._ordered_cleanup_error = leaked
        return leaked

    def _recover_interrupted_run_finalization(self, handle: RunHandle, error: BaseException) -> BaseException:
        """Retire an accepted handle after its finalizer escaped unexpectedly."""
        poison = RuntimeError(
            "RunHandle finalization escaped before retirement; the run is retained until whole-tree teardown "
            "and no further work is admitted"
        )
        poison.__cause__ = error
        self._publish_abandoned_run(handle, poison)
        return error

    def _publish_abandoned_run(self, handle: RunHandle, poison: BaseException) -> None:
        """Publish poison and abandonment before accepted-set removal."""
        try:
            while True:
                try:
                    with self._hierarchical_start_cv:
                        if handle in self._accepted_run_handles:
                            if self._ordered_cleanup_error is None:
                                self._ordered_cleanup_error = poison
                            handle._finalization_abandoned = True
                            if handle not in self._abandoned_run_handles:
                                self._abandoned_run_handles.append(handle)
                            handle._cleanup_published = True
                            self._accepted_run_handles.discard(handle)
                            self._hierarchical_start_cv.notify_all()
                    return
                except BaseException:  # noqa: BLE001, PERF203
                    # Publication fields are monotonic; repetition never
                    # replays cleanup or a native release.
                    pass
        except BaseException:  # noqa: BLE001
            self._publish_abandoned_run(handle, poison)

    def _drain_abandoned_run_keepalives(self) -> BaseException | None:
        """Drop retained run references after native teardown has succeeded."""
        cursor = _AbandonedRunKeepaliveCursor(self._abandoned_run_handles)
        cursor.drain()
        return cursor.first_error

    def _abandon_unsettled_run(
        self, handle: RunHandle, message: str, cause: BaseException | None = None
    ) -> RuntimeError:
        """Retire a run whose native cancellation fence could not be settled.

        Waiting would be an unbounded operation because native never promised a
        terminal fence. Retain every run-owned Python reference until whole-tree
        teardown, publish the sticky poison before removing the handle from the
        accepted drain set, and make the otherwise-private handle terminal.
        """
        poison = handle._finalization_error
        try:
            while not handle._terminal:
                try:
                    if poison is None:
                        poison = RuntimeError(message)
                        if cause is not None:
                            poison.__cause__ = cause
                        handle._cache_finalization_error(poison)
                    self._publish_abandoned_run(handle, poison)
                    handle._publish_terminal(poison)
                except BaseException:  # noqa: BLE001, PERF203
                    # Every field above is monotonic. Retrying cannot restore
                    # acceptance or release retained ownership early.
                    pass
            assert isinstance(handle._error, RuntimeError)
            return handle._error
        except BaseException:  # noqa: BLE001
            return self._abandon_unsettled_run(handle, message, cause)

    def _require_no_ordered_cleanup_failure(self, api: str) -> None:
        """Refuse if a prior run's ordered cleanup failed."""
        with self._hierarchical_start_cv:
            if self._ordered_cleanup_error is not None:
                raise RuntimeError(
                    f"Worker.{api}: a prior run's ordered cleanup failed, so this worker's device state is "
                    "unreclaimed and no further work is admitted; close() it"
                ) from self._ordered_cleanup_error

    def _require_region_control_context(self, api: str) -> None:
        frame = _callback_frame_for(self)
        if frame is not None:
            if frame.has_submitted_task:
                raise RuntimeError(
                    f"Worker.{api}: RegionInstance access cannot follow a task submission in the same run"
                )
            self._require_no_ordered_cleanup_failure(api)
            return
        if id(self) in _held_control_reservations():
            self._require_no_ordered_cleanup_failure(api)
            return
        raise RuntimeError(f"Worker.{api}: RegionInstance access requires an active orchestration/control context")

    def _control_admission(self, api: str):
        """The direct-control ordering policy for a Worker-level command.

        Same policy the Orchestrator facade applies: a call inside a graph
        callback belongs to that run and waits for the FIFO head; one that
        belongs to no run reserves the worker for its duration.
        """
        native = self._orch._o if self._orch is not None else None
        return direct_control(self, native, f"Worker.{api}")

    def _device_control_admission(self, api: str):
        """Ordering for one device-memory command that reaches a child outside any TaskSlot.

        L2 owns its chip in-process and has no orchestrator to be ordered against,
        so only L3+ takes the whole-run fence.
        """
        if self.level == 2:
            return contextlib.nullcontext()
        return self._control_admission(api)

    @contextlib.contextmanager
    def _control_reservation(self, api: str):
        """Reserve this worker for one command that belongs to no run.

        A public `Worker.malloc/free/copy_*/remote_*` outside a graph callback
        never becomes a task, so the whole-run FIFO cannot order it against one:
        against a live run it can copy into a buffer that run is reading, or
        free one it still holds. It is ordered only by being alone.

        "Alone" has to cover the command, not just the moment before it. A
        sampled check leaves the caller free to send its mailbox command after a
        submit admitted a run behind its back, so this takes the serializer
        submission itself holds — no run can be admitted between the check and
        the command. It takes it *shared*: what the command needs is that no run
        is admitted while it runs, which two commands on different chips can
        guarantee at the same time. Re-entrant per thread: one control call may be built out of
        others (a queue out of a region), and the second must join the
        reservation rather than deadlock on it.
        """
        held = _held_control_reservations()
        if id(self) in held:
            yield
            return
        with self._submit_mu.shared():
            with self._hierarchical_start_cv:
                if self._ordered_cleanup_error is not None:
                    raise RuntimeError(
                        f"{api}: a prior run's ordered cleanup failed, so this worker's device state is "
                        "unreclaimed and no further work is admitted; close() it"
                    ) from self._ordered_cleanup_error
                live = [h for h in self._accepted_run_handles if not h._cleanup_published]
                if live:
                    raise RuntimeError(
                        f"{api}: {len(live)} run(s) still in flight. Device control that belongs to no "
                        "run is only ordered when nothing else is: wait on the outstanding handles first, "
                        "or issue it from inside a run's orchestration callback"
                    )
            held.add(id(self))
            try:
                yield
            finally:
                held.discard(id(self))

    def _cleanup_bearing_predecessor(self) -> RunHandle | None:
        """A live handle whose ordered cleanup must finish before admission.

        At most one can be outstanding: graph callbacks are serialized, so a
        run only acquires cleanup-bearing resources while it holds
        ``_submit_mu``, and the next submission waits for it here.
        """
        with self._hierarchical_start_cv:
            for handle in self._accepted_run_handles:
                # `done` answers the native fence, which fires before this
                # run's device-touching cleanup runs — keying on it would admit
                # a successor while that cleanup is still outstanding.
                if handle._resources.requires_ordered_cleanup and not handle._cleanup_published:
                    return handle
        return None

    def _submit_l2_locked(self, callable_id: int, args, cfg: CallConfig) -> RunHandle:
        """Admit one direct-chip run and return a handle to it while it runs.

        The lane is the sole admission authority. Its runtime contract permits
        one active plus one prepared compatible run; otherwise submission drains
        the predecessor and retains depth-one behavior. The caller owns the
        completion fence through :meth:`RunHandle.wait`.
        """
        assert self._chip_worker is not None
        touched = self._identities_in_args(args) if args is not None else set()
        # Publish touched_identities BEFORE materializing, not after: release_buffer() reads this
        # dict to decide whether a Buffer is still in flight, so if it were only written after
        # _materialize_l2_args() (which populates self._chip_import_registry, the very cache
        # release_buffer() pops), a release racing that window would see no entry for a run that
        # has already cached the mapping it is about to pop out from under it.
        with self._registry_lock:
            self._chip_run_seq += 1
            run_id = self._chip_run_seq
            self._chip_run_touched_identities[run_id] = touched
        try:
            chip_args = self._materialize_l2_args(args)
            chip_run = self._chip_worker._impl._submit_chip_run_direct(callable_id, chip_args, cfg)
        except BaseException:
            with self._registry_lock:
                self._chip_run_touched_identities.pop(run_id, None)
            raise
        with self._registry_lock:
            self._chip_runs[run_id] = chip_run
        # chip_args is kept alive by the handle: the lane copies the args into
        # its own storage, but the keepalive also pins the buffers the resolved
        # descriptors point at for as long as the run can still read them.
        return RunHandle(self, run_id, (callable_id, args, cfg, chip_args))

    def _chip_run_for(self, run_id: int) -> Any | None:
        return self._chip_runs.get(run_id)

    def _submit_l3_locked(self, callable, args, cfg: CallConfig) -> RunHandle:
        assert self._orch is not None
        assert self._worker is not None
        run_id = self._orch._begin_run()
        resources = _RunResources()
        handle = RunHandle(self, run_id, (callable, args, cfg), resources)
        with self._hierarchical_start_cv:
            self._accepted_run_handles.add(handle)
            self._hierarchical_start_cv.notify_all()

        scope_open = False
        self._building_run_resources = resources
        try:
            self._orch._scope_begin()
            scope_open = True
            with _callback_run(run_id, self):
                if _host_spans_active():
                    graph_start_ns = time.monotonic_ns()
                    try:
                        callable(self._orch, args, cfg)
                    finally:
                        graph_end_ns = time.monotonic_ns()
                        _emit_host_span(
                            f"{self._host_span_prefix}.graph_build",
                            run_id,
                            0,
                            0,
                            graph_start_ns,
                            graph_end_ns - graph_start_ns,
                            f"run_id={run_id} role=facade",
                        )
                else:
                    callable(self._orch, args, cfg)
            scope_open = False
            self._orch._scope_end()
            self._orch._close_run_submission(run_id)
        except BaseException as e:
            try:
                if scope_open:
                    scope_open = False
                    self._orch._scope_end()
            finally:
                cancellation_error: BaseException | None = None
                failure_text = _format_exc("orchestration", e)
                for _ in range(_RUN_CANCELLATION_ATTEMPTS):
                    try:
                        self._orch._fail_run_submission(run_id, failure_text)
                    except BaseException as exc:  # noqa: BLE001
                        cancellation_error = exc
                    else:
                        cancellation_error = None
                        break

                # Graph-construction failures remain synchronous, but any
                # tasks already submitted still own their resources until
                # either the run fence fires or whole-tree teardown reclaims an
                # unsettled cancellation.
                self._building_run_resources = None
                if cancellation_error is None:
                    handle._wait_for_serialization()
                else:
                    self._abandon_unsettled_run(
                        handle,
                        "Worker.submit(): native cancellation did not settle after its bounded retry; "
                        "the run fence is nonterminal and no further work is admitted",
                        cancellation_error,
                    )
            raise
        finally:
            self._building_run_resources = None
        return handle

    def _run_handle_done(self, run_id: int) -> bool:
        chip_run = self._chip_run_for(run_id)
        if chip_run is not None:
            return chip_run.done()
        assert self._orch is not None
        return self._orch._run_done(run_id)

    def _wait_run_handle(self, run_id: int, timeout: float | None) -> bool:
        chip_run = self._chip_run_for(run_id)
        if chip_run is not None:
            # Negative means "no deadline" to the binding, which then blocks on
            # the device instead of polling.
            return chip_run.wait(-1.0 if timeout is None else max(0.0, timeout))
        assert self._orch is not None
        if timeout is None:
            self._orch._wait_run(run_id)
            return True
        return self._orch._wait_run_for(run_id, timeout)

    def _wait_run_handle_accepted(self, run_id: int) -> None:
        assert self._orch is not None
        self._orch._wait_run_accepted(run_id)

    def _finalize_run_handle(  # noqa: PLR0912 -- one extra branch for the L2 pop-under-lock guard
        self,
        handle: RunHandle,
        run_id: int,
        native_error: BaseException | None,
        *,
        _after_step: Any | None = None,
    ) -> BaseException | None:
        """Run fence-owned cleanup exactly once and return the cached result."""
        # A direct-chip run owns no orchestration state: the lane finalized the
        # native run as part of reaching terminal, and this worker built no
        # domains, remote slots or chip regions for it. Retiring the lane entry
        # is the whole of its cleanup, so it never enters the cursor below —
        # driving those steps here would report a cleanup failure for state that
        # was never created. The membership check and both removals share one lock
        # acquisition (pop(..., None), not del) so a concurrent Worker.close() clearing
        # these same dicts can never make this raise KeyError or read a half-updated pair.
        with self._registry_lock:
            was_l2_run = self._chip_runs.pop(run_id, None) is not None
            if was_l2_run:
                self._chip_run_touched_identities.pop(run_id, None)
        if was_l2_run:
            return native_error

        # Two different failures, deliberately not merged. A task that failed is
        # this run's business and says nothing about the worker; a cleanup that
        # failed leaves collective device state nobody can describe, and poisons
        # the worker. Reporting a task failure as a cleanup failure would
        # permanently poison a worker whose cleanup was fine.
        cursor: _RunFinalizationCursor | None = None
        outer_error: BaseException | None = None
        try:
            resources = handle._resources
            orch = self._orch

            def _poison_endpoint() -> None:
                if native_error is not None:
                    self._poison_worker_chip_region_from_endpoint_error(native_error, resources)

            def _release_native_run() -> None:
                if orch is None:
                    raise RuntimeError("RunHandle cleanup lost its native Orchestrator")
                orch._release_run(run_id)

            cursor = _RunFinalizationCursor(
                steps=(
                    ("endpoint_poison", _poison_endpoint),
                    ("remote_slot_refs", lambda: self._release_active_remote_slot_refs(resources)),
                    ("remote_frees", self._flush_pending_remote_frees),
                    ("worker_chip_regions", lambda: self._cleanup_worker_chip_regions(resources)),
                    ("worker_chip_host_buffers", resources.worker_chip_orch_comm_host_buffers.clear),
                    (
                        "pending_global_domains",
                        lambda: self._execute_pending_global_domain_releases(resources),
                    ),
                    (
                        "live_global_domains",
                        lambda: self._release_all_live_global_domains(
                            resources,
                            include_retained=False,
                        ),
                    ),
                    ("pending_domains", lambda: self._execute_pending_domain_releases(resources)),
                    ("live_domains", lambda: self._release_all_live_domains(resources)),
                    ("retire_domains", lambda: self._retire_run_domains(resources)),
                    ("native_run", _release_native_run),
                )
            )
            cursor.drain(_after_step)
        except BaseException as exc:  # noqa: BLE001
            # Any escape outside a committed cursor edge has ambiguous cleanup
            # ownership. Native teardown is its only safe reclamation boundary.
            outer_error = exc
            if cursor is not None:
                cursor.remember_boundary_error(exc)
                cursor.incomplete = True

        cleanup_error = None if cursor is None else cursor.cleanup_error
        boundary_error = outer_error if cursor is None else cursor.boundary_error
        incomplete = cursor is None or cursor.incomplete or not cursor.exhausted
        abandonment_error: RuntimeError | None = None
        if incomplete:
            abandonment_error = RuntimeError(
                "RunHandle finalization stopped at an ambiguous cleanup boundary; the run is retained until "
                "whole-tree teardown and no further work is admitted"
            )
            abandonment_error.__cause__ = boundary_error if boundary_error is not None else cleanup_error

        # Poison/cleanup publication precedes accepted-set removal under the
        # same lifecycle lock. A successor therefore sees either the accepted
        # predecessor or the sticky poison, including the conservative path
        # whose keepalives stay owned until native teardown succeeds.
        while True:
            try:
                if native_error is not None:
                    result = native_error
                elif cleanup_error is not None:
                    result = cleanup_error
                elif boundary_error is not None:
                    result = boundary_error
                else:
                    result = abandonment_error
                handle._cache_finalization_error(result)
                with self._hierarchical_start_cv:
                    if incomplete:
                        assert abandonment_error is not None
                        if self._ordered_cleanup_error is None:
                            self._ordered_cleanup_error = abandonment_error
                        handle._finalization_abandoned = True
                        if handle not in self._abandoned_run_handles:
                            self._abandoned_run_handles.append(handle)
                    elif cleanup_error is not None and self._ordered_cleanup_error is None:
                        self._ordered_cleanup_error = cleanup_error
                    handle._cleanup_published = True
                    self._accepted_run_handles.discard(handle)
                    self._hierarchical_start_cv.notify_all()
                break
            except BaseException as exc:  # noqa: BLE001, PERF203
                if boundary_error is None:
                    boundary_error = exc
        # Precedence: the run's own failure is what its waiters came for. A
        # cleanup failure is reported when the run itself succeeded, since it is
        # the reason the worker is now shut.
        return handle._finalization_error

    def _materialize_l2_args(self, args) -> Any:
        """Resolve an L2 leaf's tensor args to a chip-POD blob in this process.

        The user builds args as ``TaskArgs`` (``Tensor``) at every level; an L2 leaf is the consumer of
        its own args, so it does exactly what a chip child does — resolve each ref's embedded descriptor
        to a local base (map-once, cached in ``_chip_import_registry``) and build the chip blob the
        runtime reads — only without a mailbox, since the args are already in this process.

        ``args`` is a ``TaskArgs`` or ``None`` (no args). The chip-only POD
        ``ChipStorageTaskArgs`` is not accepted here — submit it through ``ChipWorker._run_slot``.
        """
        registry = self._chip_import_registry
        if registry is None:
            # This worker runs its own chip in-process (no fork, no mailbox): it is its own device
            # endpoint, so DEVICE backings it materializes must be its own.
            context = ImportContext(is_host_endpoint=False, device_owner_instance_id=self._owner_instance_id)
            registry = ImportRegistry(context)
            self._chip_import_registry = registry
        if args is None:
            args = TaskArgs()
        resolved = registry.materialize_args(args)
        return materialize_task_args(args, resolved)

    def _run_l2_materialized(self, callable_id: int, args, cfg) -> None:
        """Materialize an L2 leaf's tensor args and run the kernel to completion."""
        chip_args = self._materialize_l2_args(args)
        assert self._chip_worker is not None
        self._chip_worker._impl.run_materialized(callable_id, chip_args, cfg)

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

    @property
    def run_stream_set_create_count(self) -> int:
        """L2 only: number of AICore run streams the runner has created.

        One AICPU + AICore pair serves every run for the runner's lifetime. The
        AICPU stream persists; the AICore stream is recreated when a new code
        upload makes it stale, and destroyed when an unproven completion retires
        it, so this advances per publication or unproven retirement rather than
        once per run or per pipeline slot. Returns 0 on non-L2 workers and on
        platforms whose runs use the persistent bootstrap stream pair
        (simulation, a5).
        """
        if self.level != 2 or self._chip_worker is None:
            return 0
        return self._chip_worker.run_stream_set_create_count

    # ------------------------------------------------------------------
    # close
    # ------------------------------------------------------------------

    def _has_native_tree(self) -> bool:
        """A device-bound native object (ChipWorker / _Worker) is live."""
        return self._worker is not None or self._chip_worker is not None

    def _has_live_resources(self) -> bool:
        """Any teardown-owned resource is still present. Covers the native tree,
        child pids/shms, Worker-Chip regions, live CommDomains, host buffers, and
        pending remote frees/import-releases."""
        return (
            self._has_native_tree()
            or bool(self._sub_pids or self._chip_pids or self._next_level_pids)
            or bool(self._sub_shms or self._chip_shms or self._next_level_shms)
            or any(group.process is not None or group.ready_dir is not None for group in self._mpi_l3_groups)
            or bool(self._live_worker_chip_regions)
            or bool(self._live_domains)
            or bool(self._live_global_domains or self._failed_global_domain_releases)
            or bool(self._global_node_domains)
            or bool(self._remote_sessions)
            or bool(self._pending_remote_buffer_frees or self._pending_remote_import_releases)
            or not self._cleanup_journal.empty
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
        n_mpi = sum(1 for group in self._mpi_l3_groups if group.process is not None or group.ready_dir is not None)
        if n_mpi:
            parts.append(f"{n_mpi} mpirun group(s)")
        if self._live_worker_chip_regions:
            parts.append(f"{len(self._live_worker_chip_regions)} L3-L2 region(s)")
        if self._live_domains:
            parts.append(f"{len(self._live_domains)} comm domain(s)")
        if self._live_global_domains or self._failed_global_domain_releases:
            live_global_ids = {handle.domain_id for handle in self._live_global_domains.values()}
            live_global_ids.update(self._failed_global_domain_releases)
            parts.append(f"{len(live_global_ids)} global comm domain(s)")
        if self._global_node_domains:
            parts.append(f"{len(self._global_node_domains)} imported global comm domain(s)")
        if self._remote_sessions:
            parts.append(f"{len(self._remote_sessions)} remote session(s)")
        n_remote = len(self._pending_remote_buffer_frees) + len(self._pending_remote_import_releases)
        if n_remote:
            parts.append(f"{n_remote} pending remote free(s)")
        if not self._cleanup_journal.empty:
            parts.append(f"{len(self._cleanup_journal)} cleanup journal item(s)")
        return ", ".join(parts) if parts else "(none)"

    def close(self) -> None:  # noqa: PLR0912, PLR0915 -- lifecycle linearization: reentrancy / init-guard / join / owner / claim / drain / teardown
        """Release this worker's resources. Publicly terminal and retryable.

        A permanent commitment, not a reversible attempt: CLOSED is published
        atomically and never reverts to READY, and the leased live-tree APIs are
        rejected from then on. Put the call in a ``finally`` — a worker that is
        never closed keeps its device held.

        - Reentrant ``close()`` from inside a leased operation is rejected.
        - ``close()`` during an in-progress ``init()`` on another thread
          cooperatively cancels it: the init epoch unwinds and this call
          proceeds to teardown. Cancellation is observed only at cooperative
          points, so an init blocked inside a native segment is not
          interrupted — this call raises after
          ``_CLOSE_CANCEL_UNWIND_TIMEOUT_S`` rather than blocking forever.
          Closing from the init-owner thread itself is rejected.
        - A concurrent ``close()`` joins the in-flight attempt and observes its
          result; teardown never runs twice at once.
        - A later ``close()`` retries journaled teardown debt. The journal keeps
          each resource until its native free succeeds and preserves the child
          pid/mailbox pair until ``waitpid`` proves the child is gone.
        - Native teardown runs on the ``init()``-owner thread, being device-bound.
        """
        # close() is a permanent commitment against a resource, not a reversible
        # attempt: it publishes CLOSED atomically (the sole public admission
        # fence — the leased live-tree APIs are rejected once CLOSED) and NEVER
        # reverts to READY. Contract:
        #   - reentrant close() (from inside a leased op) is rejected;
        #   - close() while init() is INITIALIZING latches a cancel token and
        #     waits (bounded) for the init thread to unwind the epoch;
        #     closing from the init-owner thread is rejected;
        #   - a concurrent close() joins the in-flight attempt and observes its
        #     result; the same worker's teardown never runs twice at once;
        #   - CLOSED remains absorbing, while journaled teardown debt may be
        #     re-driven by a later close(); a tree with a live op is never torn
        #     down;
        #   - native teardown runs only on the init-owner thread (device-bound).
        # `attempt` is None until the claim installs it. The pre-claim checks
        # raise/return before that, so the finally skips completion for them. From
        # the claim on, the attempt is completed in an innermost resilient finally
        # by one immutable outcome-reference publication followed by a locked
        # `notify_all()`. Every fallible step —
        # drain, teardown, residual synthesis, registry detach — runs before it
        # and folds its error into `result`. `done` is set BEFORE the CV acquire,
        # so an async BaseException in the (interruptible) acquire or the notify
        # cannot strand a joiner — the joiner's bounded re-check recovers a
        # skipped notify. A pre-publication interruption becomes that immutable
        # outcome's error; an interruption after publication cannot retract it.
        deferred_native_cleanup_error: RuntimeError | None = None
        attempt: _CloseAttempt | None = None
        result: BaseException | None = None
        teardown_tree = False
        teardown_completed = False
        drain_complete = False
        drain_deadline: float | None = None
        drain_deadline_expired = False
        handles_to_drain: tuple[RunHandle, ...] = ()
        try:
            with self._hierarchical_start_cv:
                if threading.get_ident() in self._lease_depth:
                    raise RuntimeError(
                        "Worker.close(): cannot be called from within a run() / submit() / create_buffer() / "
                        "register() / unregister() or other leased Worker operation on this thread"
                    )
                if self._lifecycle is _Lifecycle.INITIALIZING:
                    if self._init_owner_thread is threading.current_thread():
                        raise RuntimeError("Worker.close(): cannot cancel init() from the init-owner thread")
                    self._cancel_token = True
                    self._hierarchical_start_cv.notify_all()
                    _cancel_deadline = time.monotonic() + _CLOSE_CANCEL_UNWIND_TIMEOUT_S
                    while self._lifecycle is _Lifecycle.INITIALIZING:
                        _remaining = _cancel_deadline - time.monotonic()
                        if _remaining <= 0:
                            raise RuntimeError(
                                "Worker.close(): cancelled init() did not unwind within "
                                f"{_CLOSE_CANCEL_UNWIND_TIMEOUT_S}s; the init thread is blocked in a "
                                "native segment past every cooperative cancellation point"
                            )
                        self._hierarchical_start_cv.wait(timeout=min(_remaining, _CLOSE_JOIN_RECHECK_S))
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
                if (
                    prior is not None
                    and prior.done
                    and self._cleanup_journal.empty
                    and (self._teardown_attempted or prior.error is None)
                ):
                    deferred_native_cleanup_error = self._consume_worker_host_mapped_cleanup_error_locked("close")
                    if prior.error is not None:
                        raise prior.error
                    if deferred_native_cleanup_error is not None:
                        raise deferred_native_cleanup_error
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
                deferred_native_cleanup_error = self._consume_worker_host_mapped_cleanup_error_locked("close")
                # Claim: publish CLOSED (permanent admission fence) and install a
                # fresh teardown attempt.
                self._lifecycle = _Lifecycle.CLOSED
                self._invalidate_endpoint_registry()
                attempt = _CloseAttempt()
                self._close_completion = attempt
                self._hierarchical_start_cv.notify_all()
                # One absolute budget covers both kinds of admitted work. A
                # lease that consumes most of it must not be followed by a
                # fresh full-budget wait on an accepted run fence.
                drain_deadline = time.monotonic() + _ROLLBACK_GRACEFUL_TIMEOUT_S
                # Drain in-flight leases before touching the tree. CLOSED already
                # rejects new leases; a tree with a live op is never torn down.
                # If an op outlives the budget, teardown stays UN-attempted and
                # the tree intact so a later close() can retry once it drains —
                # one of the two paths that keep close() retryable (the other is
                # an async interruption leaving an accepted run fence undrained).
                if self._active_ops > 0:
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
                    handles_to_drain = tuple(self._accepted_run_handles)
                    drain_complete = True
            if drain_complete:
                # CLOSED prevents new admissions and active operations have
                # drained, so this is the complete accepted set. Wait outside
                # the lifecycle CV: handle retirement acquires the same lock.
                assert drain_deadline is not None
                for handle in handles_to_drain:
                    remaining = drain_deadline - time.monotonic()
                    if remaining <= 0:
                        drain_deadline_expired = True
                        break
                    try:
                        handle.wait(remaining)
                    except BaseException as exc:  # noqa: BLE001
                        if result is None:
                            result = exc
                        if isinstance(exc, TimeoutError) and time.monotonic() >= drain_deadline:
                            drain_deadline_expired = True
                            break
                with self._hierarchical_start_cv:
                    if self._accepted_run_handles:
                        # A timeout or asynchronous interruption left at least
                        # one fence undrained. Keep teardown retryable and the
                        # tree intact.
                        drain_complete = False
                        if drain_deadline_expired and (result is None or isinstance(result, TimeoutError)):
                            result = TimeoutError(
                                "Worker.close(): run fence(s) still in flight after the cleanup budget "
                                f"({_ROLLBACK_GRACEFUL_TIMEOUT_S}s); teardown deferred (worker stays CLOSED)"
                            )
                    else:
                        teardown_tree = self._has_live_resources()
                        # Fence drain makes this close terminal even when the
                        # run error is the only remaining outcome and no native
                        # resource needs teardown.
                        self._teardown_attempted = (
                            teardown_tree or result is not None or deferred_native_cleanup_error is not None
                        )
            if teardown_tree:
                self._teardown_ready_tree()
                teardown_completed = True
        except BaseException as exc:  # noqa: BLE001
            if result is None:
                result = exc
        finally:
            if attempt is not None:
                had_live = True  # conservative default if a read below is interrupted
                detached_registry: tuple[dict, dict, dict] | None = None
                try:
                    deferred_native_cleanup_error = self._consume_worker_host_mapped_cleanup_error("close")
                    if result is None and deferred_native_cleanup_error is not None:
                        result = deferred_native_cleanup_error
                    # A successful tree teardown is the reclamation boundary
                    # for abandoned run ownership. Drain it before the residual
                    # inventory: that diagnostic probe is interruptible and an
                    # interruption must not retain already-safe references.
                    if teardown_completed:
                        retained_error = self._drain_abandoned_run_keepalives()
                        if result is None and retained_error is not None:
                            result = retained_error
                    had_live = self._has_live_resources()
                    journal_pending = not self._cleanup_journal.empty
                    if teardown_tree and result is None and had_live and not journal_pending:
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
                    # The immutable outcome reference is the completion flag and
                    # result. A reader can therefore never observe completion
                    # without its error/incomplete payload. Publication precedes
                    # notification; bounded re-checks cover a skipped wakeup.
                    published: _CloseOutcome | None = None
                    incomplete = result is not None or had_live
                    while published is None:
                        try:
                            published = attempt.publish(result, incomplete)
                        except BaseException as exc:  # noqa: BLE001, PERF203
                            if attempt.done:
                                raise
                            if result is None:
                                result = exc
                            incomplete = True
                    result = published.error
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
        """Store the shutdown request into every child mailbox in one group
        (next-level children trigger ``inner_worker.close()``; chip/sub children
        exit their serve loop). The first store error is raised after all are
        attempted.

        The request is a sticky word plus the state word (see
        ``_request_child_shutdown``)."""
        errors: list[BaseException] = []
        for shm in shms:
            try:
                buf = shm.buf
                if buf is not None:
                    _request_child_shutdown(buf)
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
        use-after-free), so a survivor keeps BOTH for ``_reclaim_child_groups``
        to transfer into the retry journal. An abnormal exit (signal / non-zero
        code) is likewise reported. The first error is raised after every child
        is attempted.
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

    def _unlink_shm_namespace(self, protected_names: set[str]) -> None:
        """Unlink orphan mailboxes belonging to this exact startup tree."""
        shm_dir = "/dev/shm"
        if not self._shm_tree_tokens or not os.path.isdir(shm_dir):
            return
        prefixes = tuple(f"sp-{token[:8]}-" for token in self._shm_tree_tokens)
        errors: list[BaseException] = []
        for entry in os.scandir(shm_dir):
            if entry.name in protected_names or not entry.name.startswith(prefixes):
                continue
            try:
                orphan = SharedMemory(name=entry.name)
                try:
                    orphan.unlink()
                finally:
                    orphan.close()
            except FileNotFoundError:
                pass
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            raise errors[0]

    def _reclaim_child_groups(self, deadline: float) -> None:
        """One waitpid→mailbox-release sequence shared by abort and close."""
        groups = [
            (self._sub_shms, self._sub_pids),
            (self._chip_shms, self._chip_pids),
            (self._next_level_shms, self._next_level_pids),
        ]
        reap_error: BaseException | None = None
        try:
            self._reap_child_groups(groups, deadline)
        except BaseException as exc:  # noqa: BLE001
            reap_error = exc

        protected_names = {
            shm.name for shms, _pids in groups for shm in shms if isinstance(getattr(shm, "name", None), str)
        }
        _journal_child_survivors(
            self._cleanup_journal,
            self._sub_shms,
            self._sub_pids,
            self._chip_shms,
            self._chip_pids,
            self._next_level_shms,
            self._next_level_pids,
            set(),
        )
        self._sub_pids.clear()
        self._chip_pids.clear()
        self._next_level_pids.clear()
        self._sub_shms.clear()
        self._chip_shms.clear()
        self._next_level_shms.clear()

        try:
            self._unlink_shm_namespace(protected_names)
        except BaseException as exc:  # noqa: BLE001
            self._cleanup_journal.add_once(
                "shm",
                f"startup namespace {self._shm_token[:8]}",
                lambda names=protected_names: self._unlink_shm_namespace(names),
            )
            if reap_error is None:
                reap_error = exc
        if reap_error is not None:
            raise reap_error

    def _teardown_worker_tree(  # noqa: PLR0912 -- one ordered driver for rollback and close resource classes
        self, *, startup_abort: bool, deadline: float | None = None
    ) -> None:
        """Drive the one resource sequence used by startup rollback and close.

        Normal close enters after publishing CLOSED and draining leased ops.
        Startup rollback enters before FAILED publication, when the tree may be
        only partially constructed; the same journal actions tolerate absence.

        Every ownership-bearing step is first recorded in the post-success
        journal. Independent pre-transport steps are attempted together; if one
        remains owed, the native transport and children stay alive for a later
        close() retry. The child-reap grace starts only after SHUTDOWN
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

        # Register the whole pre-transport ownership set before driving it. The
        # journal attempts every independent action and removes only successful
        # entries. Any survivor fences the transport teardown below.
        pre_transport_keys: set[tuple[str, str]] = set()
        for kind, identity, cleanup in (
            ("region", "all Worker-Chip regions", self._cleanup_worker_chip_regions),
            ("domain", "all Global CommDomains", self._release_all_live_global_domains),
            ("domain", "Global CommDomain nodes", self._release_all_global_domain_nodes),
            ("domain", "all CommDomains", self._release_all_live_domains),
            ("provenance", "child allocation provenance", self._clear_child_prov),
            ("remote", "active remote slot references", self._release_active_remote_slot_refs),
            ("remote", "pending remote frees", self._flush_pending_remote_frees),
            ("buffer", "all owner Buffers", self._release_all_buffers),
            ("buffer", "fork-inherited tensor buffers", self._fork_tensor_handles.clear),
            ("buffer", "re-exported forwarding handles", self._reexport_by_source.clear),
            ("buffer", "chip import registry", self._close_chip_import_registry),
        ):
            self._cleanup_journal.add_once(kind, identity, cleanup)
            pre_transport_keys.add((kind, identity))
        journal_err = self._cleanup_journal.drive(pre_transport_keys)
        if journal_err is not None:
            errors.append(journal_err)
            if not startup_abort:
                raise journal_err

        # Session shutdown is the remote transport barrier: never run it in the
        # same all-attempt batch as pending remote frees.
        self._cleanup_journal.add_once("remote", "remote L3 sessions", self._release_remote_sessions)
        session_err = self._cleanup_journal.drive({("remote", "remote L3 sessions")})
        if session_err is not None:
            errors.append(session_err)
            if not startup_abort:
                raise session_err

        if self.level == 2:

            def _finalize_chip() -> None:
                if self._chip_worker:
                    # Close the lane before finalizing the worker: a handle the
                    # caller never waited on still owns device work, and the
                    # lane drains it here while the device is still up.
                    #
                    # The lane rethrows its poison on close. Whether that is
                    # news depends on who has already seen it: waiting on a
                    # handle delivers the run's error and retires its entry, so
                    # a remaining entry is a run whose failure nobody has been
                    # told about, and only then is close the first report. With
                    # every run waited, the poison is the error those waits
                    # already raised, and re-raising it here would turn a
                    # handled run failure into an unhandled close failure.
                    impl = getattr(self._chip_worker, "_impl", None)
                    if impl is not None:
                        undelivered = bool(self._chip_runs)
                        try:
                            impl._close_chip_run_lane()
                        except Exception:
                            if undelivered:
                                raise
                    with self._registry_lock:
                        self._chip_runs.clear()
                        self._chip_run_touched_identities.clear()
                    self._chip_worker.finalize()
                    self._chip_worker = None

            self._cleanup_journal.add_once("native", "ChipWorker", _finalize_chip)
            journal_err = self._cleanup_journal.drive({("native", "ChipWorker")})
            if journal_err is not None:
                errors.append(journal_err)
        else:

            def _close_worker() -> None:
                if self._worker:
                    self._mark_mpirun_groups_closing(self._mpi_l3_groups)
                    self._worker.close()
                    self._worker = None
                    self._orch = None

            self._cleanup_journal.add_once("native", "hierarchical Worker", _close_worker)
            journal_err = self._cleanup_journal.drive({("native", "hierarchical Worker")})
            if journal_err is not None:
                errors.append(journal_err)
            # A failed native close must not strand the mpirun children: their
            # reaping and mailbox unlink run before the error propagates.
            self._cleanup_journal.add_once("child", "MPI L3 groups", self._close_mpirun_groups)
            journal_err = self._cleanup_journal.drive({("child", "MPI L3 groups")})
            if journal_err is not None:
                errors.append(journal_err)
            if errors and not startup_abort:
                raise errors[0]
            self._cleanup_journal.add_once(
                "shm", "Worker-Chip orchestration mappings", self._close_worker_chip_orch_comm
            )
            journal_err = self._cleanup_journal.drive({("shm", "Worker-Chip orchestration mappings")})
            if journal_err is not None:
                errors.append(journal_err)
                if not startup_abort:
                    raise errors[0]
            if startup_abort:
                self._abort_hierarchical(deadline=deadline)
                if not self._next_level_pids and not self._next_level_shms:
                    self._next_level_workers.clear()
                    self._next_level_worker_ids.clear()
                if errors:
                    raise errors[0]
                return
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
            # eat it. Reap transfers a surviving pid/shm pair into the journal;
            # a later close() can retry without freeing a live mailbox.
            reap_deadline = time.monotonic() + _CLOSE_CHILD_REAP_TIMEOUT_S
            _step(lambda: self._reclaim_child_groups(reap_deadline))
            # A prior attempt may already have transferred a surviving child
            # and its mailbox into the journal. Retry those entries only after
            # this attempt has delivered SHUTDOWN and reaped the live groups;
            # otherwise an unreaped child can fence the very work that lets it
            # converge. Namespace/orphan-shm debt belongs to the same phase.
            post_child_err = self._cleanup_journal.drive_kinds({"child", "shm"})
            if post_child_err is not None:
                errors.append(post_child_err)
            # Drop next-level worker refs only once their pids/shms are reclaimed.
            if not self._next_level_pids and not self._next_level_shms:
                self._next_level_workers.clear()
                self._next_level_worker_ids.clear()

        if errors:
            raise errors[0]

    def _teardown_ready_tree(self) -> None:
        self._teardown_worker_tree(startup_abort=False)

    def __enter__(self) -> Worker:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
