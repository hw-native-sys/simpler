# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Worker — unified factory for all hierarchy levels.

Callable identity is a ``cid`` (int), allocated exclusively by
``Worker.register(callable)``. ``Worker.run`` and the orchestrator's
``submit_next_level`` / ``submit_sub`` all take this cid — never the raw
``ChipCallable`` / Python function. L≥3 Python callables registered before
child startup are inherited through the fork-time snapshot; later
registrations are serialized and sent through the mailbox control plane.

Usage::

    # L2: one NPU chip
    w = Worker(level=2, device_id=8, platform="a2a3", runtime="tensormap_and_ringbuffer")
    w.init()
    chip_cid = w.register(chip_callable)            # L2 may register pre or post init()
    w.run(chip_cid, chip_args, config)
    w.close()

    # L3: multiple chips + SubWorkers, auto-discovery in init()
    w = Worker(level=3, device_ids=[8, 9], num_sub_workers=2,
               platform="a2a3", runtime="tensormap_and_ringbuffer")
    chip_cid = w.register(chip_callable)            # ChipCallable, before init()
    sub_cid  = w.register(lambda args: postprocess())  # Python sub, before init()
    w.init()

    def my_orch(orch, args, cfg):
        r = orch.submit_next_level(chip_cid, chip_args_ptr, cfg)
        orch.submit_sub(sub_cid, sub_args)

    w.run(my_orch, my_args, my_config)
    w.close()

    # L4: recursive composition — L3 Workers as children
    l3 = Worker(level=3, device_ids=[8, 9], num_sub_workers=1,
                platform="a2a3", runtime="tensormap_and_ringbuffer")
    w4 = Worker(level=4, num_sub_workers=1)
    l3_cid = w4.register(my_l3_orch)
    verify_cid = w4.register(lambda: verify())
    w4.add_worker(l3)
    w4.init()

    def my_l4_orch(orch, args, config):
        orch.submit_next_level(l3_cid, chip_args, config)
        orch.submit_sub(verify_cid)

    w4.run(Task(orch=my_l4_orch))
    w4.close()
"""

import ctypes
import os
import signal
import struct
import sys
import threading
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Optional

import cloudpickle
from _task_interface import (  # pyright: ignore[reportMissingImports]
    MAX_REGISTERED_CALLABLE_IDS,
    RunTiming,
    WorkerType,
    _mailbox_load_i32,
    _mailbox_store_i32,
    read_args_from_blob,
)

from . import _log as _simpler_log
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
    TaskArgs,
    _Worker,
)

# Upper bound on how long the parent waits for every chip's bootstrap mailbox
# to leave IDLE.  Well above a realistic HCCL init (seconds) but short enough
# that a hung child fails the suite instead of the CI job timing out.
_BOOTSTRAP_WAIT_TIMEOUT_S = 120.0
_BOOTSTRAP_POLL_INTERVAL_S = 0.001
_PY_CONTROL_TIMEOUT_S = 30.0


# ---------------------------------------------------------------------------
# Unified mailbox layout (must match worker_manager.h MAILBOX_OFF_*)
# ---------------------------------------------------------------------------
#
# One layout for both NEXT_LEVEL (chip) and SUB workers. SUB children
# read `callable` as a uint64 encoding the callable_id and decode the
# args_blob region to pass TaskArgs to the registered callable.

_OFF_STATE = 0
_OFF_ERROR = 4
_OFF_CALLABLE = 8
_OFF_CONFIG = 16
# Packed CallConfig wire layout — must match call_config.h byte for byte:
# 6 int32 (block_dim, aicpu_thread_num, enable_l2_swimlane, enable_dump_tensor,
# enable_pmu, enable_dep_gen) + 1024-byte NUL-terminated output_prefix. Log
# config travels separately via ChipWorker.init(log_level, log_info_v) — not
# on per-task wire.
_CFG_FMT = struct.Struct("=iiiiii1024s")
# Args region starts after CONFIG, rounded up to 8 bytes so the first
# ContinuousTensor.data (uint64_t at OFF_ARGS+8) is 8-byte aligned, avoiding
# SIGBUS on strict-alignment platforms (aarch64 atomics, some ARM cores).
_OFF_ARGS = (_OFF_CONFIG + _CFG_FMT.size + 7) & ~7
assert _OFF_ARGS % 8 == 0, "_OFF_ARGS must be 8-aligned for ContinuousTensor.data"
# MAILBOX_ARGS_CAPACITY mirrors the C++ constexpr in worker_manager.h so the
# Python reader can bounds-check incoming args blobs. Source-of-truth for the
# constants on the right is the nanobind binding (cannot drift).
_MAILBOX_ARGS_CAPACITY = MAILBOX_SIZE - _OFF_ARGS - MAILBOX_ERROR_MSG_SIZE
# MAILBOX_OFF_ERROR_MSG / MAILBOX_ERROR_MSG_SIZE come from the C++
# nanobind module so the two sides cannot drift.

_IDLE = 0
_TASK_READY = 1
_TASK_DONE = 2
_SHUTDOWN = 3
_CONTROL_REQUEST = 4
_CONTROL_DONE = 5

# Control sub-commands (written at _OFF_CALLABLE as uint64)
_CTRL_MALLOC = 0
_CTRL_FREE = 1
_CTRL_COPY_TO = 2
_CTRL_COPY_FROM = 3
# Pre-warm a chip child for cid=arg0 by calling
# `prepare_callable(cid, registry[cid])` so the first run() does
# not pay the H2D upload cost.  Sent from the parent right after init()
# (or whenever a new ChipCallable cid is registered).
_CTRL_PREPARE = 4
# Dynamic post-init register of a ChipCallable. Parent stages the bytes
# in a per-register POSIX shm and writes (cid, shm_name) into the mailbox;
# the child mmaps the shm and calls prepare_callable_from_blob(cid, addr).
# See docs/callable-ipc-dynamic-register.md for the design.
_CTRL_REGISTER = 5
# Symmetric unregister: drop the cid from chip-child state so the AICPU
# orch_so_table_ slot can be reused. Payload is just the cid; no shm.
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
#   offset 16:                  uint64  arg0 (size for malloc; dev_ptr for free/copy)
#   offset 24:                  uint64  arg1 (host_ptr for copy)
#   offset 32:                  uint64  arg2 (nbytes for copy)
#   offset 40:                  uint64  result (returned ptr from malloc)
_CTRL_OFF_ARG0 = 16
_CTRL_OFF_ARG1 = 24
_CTRL_OFF_ARG2 = 32
_CTRL_OFF_RESULT = 40


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


def _load_py_callable_from_shm(shm_name: str):
    shm = SharedMemory(name=shm_name)
    try:
        shm_buf = shm.buf
        assert shm_buf is not None
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
        payload = bytes(shm_buf[_PY_CALLABLE_HEADER.size : expected_size])
    finally:
        shm.close()

    fn = cloudpickle.loads(payload)
    if not callable(fn):
        raise RuntimeError(f"python callable payload decoded to non-callable {type(fn).__name__}")
    return fn


def _handle_py_callable_control(buf, registry: dict, sub_cmd: int, *, context: str) -> None:
    cid = int(struct.unpack_from("Q", buf, _CTRL_OFF_ARG0)[0]) & 0xFFFFFFFF
    if cid >= MAX_REGISTERED_CALLABLE_IDS:
        raise RuntimeError(f"{context}: cid {cid} out of range")
    if sub_cmd == _CTRL_PY_REGISTER:
        shm_name = _read_shm_name(buf, _OFF_ARGS)
        registry[cid] = _load_py_callable_from_shm(shm_name)
    elif sub_cmd == _CTRL_PY_UNREGISTER:
        registry.pop(cid, None)
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
    to C++ run use the zero-copy `run_prepared_from_blob` path
    instead — see those loops for the matching comment.

    Delegates to the nanobind helper so the ContinuousTensor layout is
    parsed by C++ `read_blob` (single source of truth) instead of being
    reimplemented in Python.  The Python re-implementation that lived
    here previously dropped the `child_memory` byte (offset 33), which
    silently broke any tensor carrying a chip-owned device pointer
    (HCCL window slots etc.) — now structurally impossible.
    """
    mailbox_addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    return read_args_from_blob(mailbox_addr + _OFF_ARGS)


def _sub_worker_loop(buf, registry: dict) -> None:
    """Runs in forked child process. Reads unified mailbox layout.

    On success writes ``error=0`` and an empty message. On failure writes
    ``error=1`` and ``f"sub_worker: <ExcType>: <msg>"`` into the mailbox
    error-message region; the parent's ``WorkerThread::dispatch_process``
    rethrows it as ``std::runtime_error``.
    """
    state_addr = _buffer_field_addr(buf, _OFF_STATE)
    while True:
        state = _mailbox_load_i32(state_addr)
        if state == _TASK_READY:
            cid = struct.unpack_from("Q", buf, _OFF_CALLABLE)[0]
            fn = registry.get(int(cid))
            code = 0
            msg = ""
            if fn is None:
                code = 1
                msg = f"sub_worker: callable id {int(cid)} not registered"
            else:
                try:
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
                _handle_py_callable_control(buf, registry, int(sub_cmd), context="sub_worker")
            except Exception as e:  # noqa: BLE001
                code = 1
                msg = _format_exc("sub_worker control", e)
            _write_error(buf, code, msg)
            _mailbox_store_i32(state_addr, _CONTROL_DONE)
        elif state == _SHUTDOWN:
            break


def _read_shm_name(buf, offset: int) -> str:
    """Decode a NUL-terminated POSIX shm name out of a fixed-width slot.

    Shared by every control sub-command that stages payload via a separate
    shm — CTRL_REGISTER (one slot), CTRL_ALLOC_DOMAIN (two slots), and
    CTRL_RELEASE_DOMAIN (one slot).
    """
    raw = bytes(buf[offset : offset + _CTRL_SHM_NAME_BYTES])
    nul = raw.find(b"\x00")
    return raw[: nul if nul >= 0 else _CTRL_SHM_NAME_BYTES].decode("utf-8", "replace")


def _handle_ctrl_alloc_domain(cw: "ChipWorker", buf: memoryview) -> None:
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
    try:
        req_buf = req_shm.buf
        assert req_buf is not None
        (allocation_id, rank_count, domain_rank, window_size, buffer_count) = _DOMAIN_REQ_HEADER.unpack_from(req_buf, 0)
        # Layout: header | buffer_nbytes[buffer_count] (u64) | rank_ids[rank_count] (u32)
        nbytes_offset = _DOMAIN_REQ_HEADER.size
        nbytes_struct = struct.Struct(f"<{buffer_count}Q") if buffer_count else struct.Struct("")
        buffer_nbytes = nbytes_struct.unpack_from(req_buf, nbytes_offset) if buffer_count else ()
        rank_ids_offset = nbytes_offset + nbytes_struct.size
        rank_ids_struct = struct.Struct(f"<{rank_count}I")
        rank_ids = list(rank_ids_struct.unpack_from(req_buf, rank_ids_offset))
    finally:
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
    try:
        reply_buf = reply_shm.buf
        assert reply_buf is not None
        _DOMAIN_REPLY_HEADER.pack_into(reply_buf, 0, int(device_ctx), int(local_window_base), int(buffer_count))
        if buffer_ptrs:
            struct.pack_into(f"<{len(buffer_ptrs)}Q", reply_buf, _DOMAIN_REPLY_HEADER.size, *buffer_ptrs)
    finally:
        reply_shm.close()


def _handle_ctrl_comm_init(cw: "ChipWorker", buf: memoryview) -> None:
    """CTRL_COMM_INIT handler — drives `cw.comm_init` on the chip child.

    Idempotent: ``ChipWorker.comm_init`` itself caches the handle and returns
    the existing one if already initialized, so a duplicate dispatch from the
    parent is a no-op.
    """
    request_shm_name = _read_shm_name(buf, _OFF_ARGS)
    req_shm = SharedMemory(name=request_shm_name)
    try:
        req_buf = req_shm.buf
        assert req_buf is not None
        (rank, nranks) = _COMM_INIT_HEADER.unpack_from(req_buf, 0)
        # rootinfo_path is the rest of the shm, NUL-terminated.
        raw = bytes(req_buf[_COMM_INIT_HEADER.size :])
        nul = raw.find(b"\x00")
        rootinfo_path = raw[: nul if nul >= 0 else len(raw)].decode("utf-8", "replace")
    finally:
        req_shm.close()

    handle = cw.comm_init(int(rank), int(nranks), rootinfo_path)
    if handle == 0:
        raise RuntimeError("comm_init returned 0 handle for hidden base communicator")
    cw._comm_base_handle_cached = int(handle)


def _handle_ctrl_release_domain(cw: "ChipWorker", buf: memoryview) -> None:
    """CTRL_RELEASE_DOMAIN handler — collective free for one allocation."""
    request_shm_name = _read_shm_name(buf, _OFF_ARGS)
    req_shm = SharedMemory(name=request_shm_name)
    try:
        req_buf = req_shm.buf
        assert req_buf is not None
        (allocation_id, rank_count, domain_rank, _ws, _bc) = _DOMAIN_REQ_HEADER.unpack_from(req_buf, 0)
    finally:
        req_shm.close()

    handle = _comm_base_handle(cw)
    cw._impl.comm_release_domain_windows(int(handle), int(allocation_id), int(rank_count), int(domain_rank))


def _comm_base_handle(cw: "ChipWorker") -> int:
    """Return the cached base-communicator handle the chip allocated during bootstrap.

    The dynamic-allocate path requires an established base communicator (HCCL
    RootInfo handshake already done).  ``bootstrap_context`` stashes the handle
    on the ChipWorker; this helper exposes it to the CTRL_* handlers.
    """
    handle = getattr(cw, "_comm_base_handle_cached", 0)
    if not handle:
        raise RuntimeError("CTRL_ALLOC_DOMAIN: chip has no base communicator — bootstrap_context must run first")
    return int(handle)


def _ensure_prepared(cw, registry, prepared, cid: int, *, lazy: bool, device_id: int) -> None:
    if cid in prepared:
        return
    callable_obj = registry.get(cid)
    if callable_obj is None:
        raise RuntimeError(f"chip_process dev={device_id}: cid {cid} not in registry")
    if lazy:
        # Reaching the lazy branch means _CTRL_PREPARE prewarm did not run
        # for this cid before the first TASK_READY; the child still does
        # the work, but the resulting H2D + dlopen cost lands on the
        # first task's latency.  Log so the gap is visible in stderr.
        sys.stderr.write(
            f"[chip_process pid={os.getpid()} dev={device_id}] WARN: lazy-prepare cid={cid}; prewarm path missed it\n"
        )
        sys.stderr.flush()
    cw.prepare_callable(cid, callable_obj)
    prepared.add(cid)


def _run_chip_main_loop(  # noqa: PLR0912 -- TASK_READY + 6 control sub-commands + SHUTDOWN form the unified state machine; cannot collapse without obscuring dispatch
    cw: ChipWorker,
    buf: memoryview,
    mailbox_addr: int,
    state_addr: int,
    device_id: int,
    registry: dict,
    *,
    on_task_done_success=None,
) -> None:
    """Unified TASK_READY / CONTROL_REQUEST / SHUTDOWN state machine.

    `on_task_done_success`, if provided, is invoked after a successful
    ``run_prepared_from_blob`` and before publishing TASK_DONE. It must
    return ``(code, msg)`` — typically ``(0, "")`` on success, or an
    error tuple if the hook itself failed (e.g. D2H staging error).
    Returning a non-zero code overrides the kernel's success.

    Per-callable_id dispatch: TASK_READY carries a cid in OFF_CALLABLE;
    the child looks the cid up in the COW-inherited Python ``registry``
    to get the ChipCallable, calls ``cw.prepare_callable(cid, callable)``
    once, then ``cw.run(cid, args, cfg)``. ``_CTRL_PREPARE`` is
    the explicit pre-warm path (parent pushes after init() to amortise
    the first H2D upload); TASK_READY also lazy-prepares as a safety net.
    """
    prepared: set[int] = set()
    while True:
        state = _mailbox_load_i32(state_addr)
        if state == _TASK_READY:
            cid = int(struct.unpack_from("Q", buf, _OFF_CALLABLE)[0]) & 0xFFFFFFFF
            cfg = _read_config_from_mailbox(buf)

            code = 0
            msg = ""
            try:
                _ensure_prepared(cw, registry, prepared, cid, lazy=True, device_id=device_id)
                # Hand the mailbox bytes straight to C++ (zero-copy zero-decode):
                # the blob layout is what `write_blob` already wrote, so re-parsing
                # it in Python is N×40B of avoidable work and a permanent
                # opportunity to drop a field.  C++ reinterpret_cast<ChipStorageTaskArgs*>
                # is the source of truth.
                cw._impl.run_prepared_from_blob(cid, mailbox_addr + _OFF_ARGS, _MAILBOX_ARGS_CAPACITY, cfg)
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
                    cid = int(struct.unpack_from("Q", buf, _CTRL_OFF_ARG0)[0]) & 0xFFFFFFFF
                    _ensure_prepared(cw, registry, prepared, cid, lazy=False, device_id=device_id)
                elif sub_cmd == _CTRL_REGISTER:
                    cid = int(struct.unpack_from("Q", buf, _CTRL_OFF_ARG0)[0]) & 0xFFFFFFFF
                    if cid >= MAX_REGISTERED_CALLABLE_IDS:
                        raise RuntimeError(f"register cid={cid} chip={device_id}: cid out of range")
                    raw = bytes(buf[_OFF_ARGS : _OFF_ARGS + _CTRL_SHM_NAME_BYTES])
                    nul = raw.find(b"\x00")
                    shm_name = raw[: nul if nul >= 0 else _CTRL_SHM_NAME_BYTES].decode("utf-8", "replace")
                    # Self-heal when the parent thinks the cid slot is free but
                    # this child's local view still has it prepared — happens
                    # when a prior _CTRL_UNREGISTER failed before reaching
                    # prepared.discard, while the parent still popped its
                    # registry under best-effort semantics. Without this,
                    # register_prepared_callable would fail-fast on a slot the
                    # user was told is reusable. The `cid in prepared` gate
                    # keeps the happy path at zero added cost.
                    if int(cid) in prepared:
                        try:
                            cw.unregister_callable(int(cid))
                        except Exception:  # noqa: BLE001
                            pass
                        prepared.discard(int(cid))
                    shm = SharedMemory(name=shm_name)
                    try:
                        shm_buf = shm.buf
                        assert shm_buf is not None
                        addr = ctypes.addressof(ctypes.c_char.from_buffer(shm_buf))
                        cw._impl.prepare_callable_from_blob(int(cid), addr)
                    finally:
                        # Release the local mmap as soon as prepare returns;
                        # prepare_callable has already H2D-copied the bytes to
                        # device GM, so the child no longer needs the shm.
                        shm.close()
                    prepared.add(int(cid))
                elif sub_cmd == _CTRL_UNREGISTER:
                    cid = int(struct.unpack_from("Q", buf, _CTRL_OFF_ARG0)[0]) & 0xFFFFFFFF
                    cw.unregister_callable(int(cid))
                    # Drop from the prepared set so a future CTRL_REGISTER /
                    # CTRL_PREPARE for the same cid is treated as a fresh
                    # registration (re-runs the H2D upload / AICPU dlopen).
                    prepared.discard(int(cid))
                elif sub_cmd == _CTRL_ALLOC_DOMAIN:
                    _handle_ctrl_alloc_domain(cw, buf)
                elif sub_cmd == _CTRL_RELEASE_DOMAIN:
                    _handle_ctrl_release_domain(cw, buf)
                elif sub_cmd == _CTRL_COMM_INIT:
                    _handle_ctrl_comm_init(cw, buf)
                else:
                    raise RuntimeError(f"unknown control sub-command {int(sub_cmd)}")
            except Exception as e:  # noqa: BLE001
                code = 1
                if sub_cmd in (_CTRL_REGISTER, _CTRL_UNREGISTER):
                    # Docs §6 mandates `register cid=<N> chip=<id>: <reason>`
                    # so the parent can pinpoint failures across many chips.
                    op = "register" if sub_cmd == _CTRL_REGISTER else "unregister"
                    try:
                        cid_v = int(struct.unpack_from("Q", buf, _CTRL_OFF_ARG0)[0]) & 0xFFFFFFFF
                    except Exception:  # noqa: BLE001
                        cid_v = -1
                    msg = _format_exc(f"{op} cid={cid_v} chip={device_id}", e)
                else:
                    msg = _format_exc(f"chip_process dev={device_id} ctrl={int(sub_cmd)}", e)
            _write_error(buf, code, msg)
            _mailbox_store_i32(state_addr, _CONTROL_DONE)
        elif state == _SHUTDOWN:
            break


def _chip_process_loop(
    buf: memoryview,
    bins,
    device_id: int,
    registry: dict,
    log_level: int = 1,
    log_info_v: int = 5,
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
        cw.init(device_id, bins, log_level=log_level, log_info_v=log_info_v)
    except Exception as e:
        _tb.print_exc()
        # Write the message so any parent reader that *does* inspect this
        # path sees the real cause. State handshake for this init-time
        # failure is broken — see KNOWN_ISSUES.md — and that is not part
        # of the L4 scope.
        _write_error(buf, 1, _format_exc(f"chip_process dev={device_id} init", e))
        return

    mailbox_addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    state_addr = mailbox_addr + _OFF_STATE
    sys.stderr.write(f"[chip_process pid={os.getpid()} dev={device_id}] ready\n")
    sys.stderr.flush()

    try:
        _run_chip_main_loop(cw, buf, mailbox_addr, state_addr, device_id, registry)
    finally:
        cw.finalize()


def _read_config_from_mailbox(buf: memoryview) -> "CallConfig":
    """Reconstruct a CallConfig from the unified mailbox layout."""
    block_dim, aicpu_tn, swl, dt, pmu, dep_gen, prefix_bytes = _CFG_FMT.unpack_from(buf, _OFF_CONFIG)
    cfg = CallConfig()
    cfg.block_dim = block_dim
    cfg.aicpu_thread_num = aicpu_tn
    cfg.enable_l2_swimlane = swl
    cfg.enable_dump_tensor = bool(dt)
    cfg.enable_pmu = pmu
    cfg.enable_dep_gen = bool(dep_gen)
    # NUL-terminated C string in a 1024-byte field.
    cfg.output_prefix = prefix_bytes.split(b"\x00", 1)[0].decode("utf-8")
    return cfg


def _child_worker_loop(
    buf: memoryview,
    registry: dict,
    inner_worker: "Worker",
) -> None:
    """Runs in forked child process. Any-level Worker as child of its parent.

    Polls the unified mailbox for (cid, config, args_blob). Looks up the
    orch function in the COW-inherited registry, then delegates to
    ``inner_worker.run(orch_fn, args, cfg)`` which opens its own scope,
    runs the orch function, and drains. Also services CONTROL_REQUEST
    so the L4 parent's dynamic register/unregister broadcasts cascade
    into the inner Worker (see docs section 7).
    """
    state_addr = _buffer_field_addr(buf, _OFF_STATE)
    while True:
        state = _mailbox_load_i32(state_addr)
        if state == _TASK_READY:
            cid = struct.unpack_from("Q", buf, _OFF_CALLABLE)[0]
            orch_fn = registry.get(int(cid))
            code = 0
            msg = ""
            if orch_fn is None:
                code = 1
                msg = f"child_worker: callable id {int(cid)} not registered"
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
                    cid_val = int(struct.unpack_from("Q", buf, _CTRL_OFF_ARG0)[0]) & 0xFFFFFFFF
                    raw = bytes(buf[_OFF_ARGS : _OFF_ARGS + _CTRL_SHM_NAME_BYTES])
                    nul = raw.find(b"\x00")
                    shm_name = raw[: nul if nul >= 0 else _CTRL_SHM_NAME_BYTES].decode("utf-8", "replace")
                    shm = SharedMemory(name=shm_name)
                    try:
                        shm_buf = shm.buf
                        assert shm_buf is not None
                        callable_obj = ChipCallable.from_bytes(bytes(shm_buf[: shm.size]))
                    finally:
                        shm.close()
                    # Delegate to the inner Worker's register so its own
                    # _post_init_register handles broadcasting to its chip
                    # / next-level children (recursive cascade). Forcing
                    # cid_val onto the registry slot keeps the inner-side
                    # cid identical to the outer-side cid — both the L4
                    # scheduler and the L3 children index by the same int.
                    registry.pop(int(cid_val), None)
                    inner_worker._register_at(int(cid_val), callable_obj)
                elif sub_cmd == _CTRL_UNREGISTER:
                    cid_val = int(struct.unpack_from("Q", buf, _CTRL_OFF_ARG0)[0]) & 0xFFFFFFFF
                    inner_worker.unregister(int(cid_val))
                elif sub_cmd in (_CTRL_PY_REGISTER, _CTRL_PY_UNREGISTER):
                    _handle_py_callable_control(
                        buf,
                        registry,
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
                            if sub_cmd == _CTRL_PY_REGISTER
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
        self._initialized = False

        # Narrow lock around `_callable_registry` mutation so concurrent
        # register / unregister calls don't trip CPython's non-atomic
        # len()+assign. The wire-level concurrency (Python control ↔ C++
        # dispatch) is now handled at the C++ boundary via mailbox_mu_, so
        # no quiescent-state guard is needed.
        self._registry_lock = threading.Lock()
        self._pending_unregister_cids: set[int] = set()
        self._py_callable_cids_seen: set[int] = set()
        self._py_control_timeout_s = float(config.get("py_control_timeout_s", _PY_CONTROL_TIMEOUT_S))
        self._hierarchical_start_state = "not_started"
        self._hierarchical_start_mu = threading.Lock()
        self._hierarchical_start_cv = threading.Condition(self._hierarchical_start_mu)

        # Level-2 internals
        self._chip_worker: Optional[ChipWorker] = None

        # Level-3+ internals
        self._worker: Optional[_Worker] = None
        self._orch: Optional[Orchestrator] = None
        self._chip_shms: list[SharedMemory] = []
        self._chip_pids: list[int] = []
        self._sub_shms: list[SharedMemory] = []
        self._sub_pids: list[int] = []

        # L4+ next-level Worker children (added via add_worker before init)
        self._next_level_workers: list[Worker] = []
        self._next_level_shms: list[SharedMemory] = []
        self._next_level_pids: list[int] = []

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

    def _comm_plan_rootinfo_path(self) -> str:
        """Per-Worker rootinfo path used by HCCL/sim base comm_init.

        Namespaced by parent pid + Python id(self) so two concurrent L3
        Workers in the same process do not collide on the handshake file.
        """
        tag = f"pto_multi_comm_{os.getpid()}_{id(self):x}.bin"
        return os.path.join("/tmp", tag)

    # ------------------------------------------------------------------
    # Callable registration (before init)
    # ------------------------------------------------------------------

    def register(self, target) -> int:
        """Register a callable. Returns the cid passed to ``run`` / ``submit_*``.

        A unified id space serves Python functions (sub fn / orch fn) and
        ``ChipCallable`` instances at every level. L2 returns a cid the
        user passes to ``Worker.run(cid, args, cfg)``; L3+ returns a cid
        the orch function passes to ``orch.submit_next_level(cid, …)`` /
        ``orch.submit_sub(cid, …)``.

        Timing constraints:
          - L3+: registrations before child processes start are inherited
            by forked children through the startup registry snapshot.
            Registrations after child processes start use the mailbox
            control plane: ChipCallables keep the binary path, while Python
            callables are serialized with cloudpickle and broadcast to
            Python-capable child groups.
          - L2: may be called either before or after ``init()`` (no fork,
            no COW constraint).  When called post-init, ChipCallables are
            prepared on the device immediately; pre-init registrations are
            batched and prepared at the end of ``init()``.
        """
        if self.level == 2 and not isinstance(target, ChipCallable):
            raise TypeError("Worker.register: level 2 only supports ChipCallable targets")
        if self.level >= 3:
            self._wait_hierarchical_start_if_needed()
            if not isinstance(target, ChipCallable):
                if not callable(target):
                    raise TypeError("Worker.register: non-ChipCallable target must be callable")
                if self._initialized and getattr(self, "_hierarchical_started", False):
                    return self._post_start_register_python(target)

        with self._registry_lock:
            cid = self._allocate_cid()
            self._callable_registry[cid] = target
            if self.level >= 3 and not isinstance(target, ChipCallable):
                self._py_callable_cids_seen.add(cid)

        # L3+ post-init ChipCallable: broadcast to chip / next-level children
        # via C++ after parent-side cid allocation is complete. The registry
        # entry keeps the cid reserved while mailbox_mu_ serializes the wire
        # round trip against dispatch.
        if self.level >= 3 and self._initialized and isinstance(target, ChipCallable):
            try:
                self._post_init_register(cid, target)
            except Exception:
                with self._registry_lock:
                    if self._callable_registry.get(cid) is target:
                        self._callable_registry.pop(cid, None)
                raise
            return cid

        # L2 post-init: pre-warm immediately so the very first
        # `Worker.run(cid, …)` is a clean cache hit.
        if self.level == 2 and self._initialized and isinstance(target, ChipCallable):
            assert self._chip_worker is not None
            self._chip_worker.prepare_callable(cid, target)
        return cid

    def _wait_hierarchical_start_if_needed(self) -> None:
        if self.level < 3:
            return
        with self._hierarchical_start_cv:
            while self._hierarchical_start_state == "starting":
                self._hierarchical_start_cv.wait()
            if self._hierarchical_start_state == "failed":
                raise RuntimeError("Worker hierarchical startup failed; close this Worker and create a new one")

    def _python_worker_types(self) -> list[WorkerType]:
        worker_types: list[WorkerType] = []
        if self._config.get("num_sub_workers", 0) > 0:
            worker_types.append(WorkerType.SUB)
        if self._next_level_workers:
            worker_types.append(WorkerType.NEXT_LEVEL)
        return worker_types

    def _post_start_register_python(self, target) -> int:
        worker_types = self._python_worker_types()
        if not worker_types:
            raise RuntimeError(
                "Worker.register: no Python-capable child workers are configured "
                "for dynamic Python callable registration"
            )
        payload = _pack_py_callable_payload(target)
        with self._registry_lock:
            cid = self._allocate_cid()
            self._callable_registry[cid] = target
            self._py_callable_cids_seen.add(cid)
        try:
            self._broadcast_py_control(worker_types, _CTRL_PY_REGISTER, cid, payload=payload, strict=True)
        except Exception:
            with self._registry_lock:
                if self._callable_registry.get(cid) is target:
                    self._callable_registry.pop(cid, None)
            raise
        return cid

    def _broadcast_py_control(
        self,
        worker_types: list[WorkerType],
        sub_cmd: int,
        cid: int,
        *,
        payload: Optional[bytes] = None,
        strict: bool,
    ) -> list[str]:
        if not worker_types:
            return []
        assert self._worker is not None
        errors: list[str] = []
        payload_bytes = payload if payload is not None else None
        for worker_type in worker_types:
            results = self._worker.broadcast_control_all(
                worker_type,
                int(sub_cmd),
                int(cid),
                payload_bytes,
                timeout_s=self._py_control_timeout_s,
            )
            for result in results:
                if not result.ok:
                    errors.append(f"{result.worker_type}[{result.worker_index}]: {result.error_message}")
        if errors and strict:
            raise RuntimeError(
                f"Worker control broadcast cid={cid} sub_cmd={sub_cmd} failed on "
                f"{len(errors)} child workers; first error: {errors[0]}"
            )
        return errors

    def _allocate_cid(self) -> int:
        """Return the smallest unused cid in [0, MAX_REGISTERED_CALLABLE_IDS).

        Caller must hold ``_registry_lock``. Walks the integers in order so
        an ``unregister(K)`` followed by a fresh ``register`` reuses K
        instead of colliding with an existing entry — ``len(registry)``
        would silently overwrite the next gap-after-the-hole.
        """
        for i in range(MAX_REGISTERED_CALLABLE_IDS):
            if i not in self._callable_registry and i not in self._pending_unregister_cids:
                return i
        # The AICPU side keeps a fixed-size orch_so_table_ keyed by cid;
        # raise here so the failure surfaces at register-time with a
        # protocol-aware message, not later from
        # DeviceRunner::register_prepared_callable with a generic
        # "out of range" log.
        raise RuntimeError(
            "Worker.register: cid space exhausted "
            f"(MAX_REGISTERED_CALLABLE_IDS={MAX_REGISTERED_CALLABLE_IDS}); "
            "unregister unused callables before registering more"
        )

    def _register_at(self, cid: int, target: ChipCallable) -> None:
        """Register *target* under a caller-specified *cid* (L4 cascade only).

        Used by ``_child_worker_loop`` when forwarding a CTRL_REGISTER from
        an L4 parent: the outer cid must match the inner cid so the L4
        scheduler's dispatch table and the inner worker's registry agree
        on a single integer key. Plain ``register`` allocates the next
        free slot and is therefore unsuitable here.
        """
        if not isinstance(target, ChipCallable):
            raise TypeError("_register_at: target must be a ChipCallable")
        with self._registry_lock:
            if cid in self._callable_registry:
                raise RuntimeError(f"_register_at: cid={cid} already occupied")
            self._callable_registry[cid] = target

        if self.level >= 3 and self._initialized:
            try:
                self._post_init_register(cid, target)
            except Exception:
                with self._registry_lock:
                    if self._callable_registry.get(cid) is target:
                        self._callable_registry.pop(cid, None)
                raise

    def _post_init_register(self, cid: int, target: ChipCallable) -> None:
        """Broadcast a new ChipCallable to every NEXT_LEVEL child via C++.

        Delegates the entire shm-staging + per-child mailbox handshake to
        ``_Worker.broadcast_register_all``, which holds per-WorkerThread
        ``mailbox_mu_`` so the broadcast serializes against any in-flight
        dispatch on each child mailbox. No Python lock required.
        """
        # Chip children are forked lazily on the first Worker.run() via
        # _start_hierarchical; before that point the chip mailboxes have
        # no reader and a CTRL_REGISTER broadcast would deadlock. In that
        # pre-fork window, just leave the cid in the parent's registry —
        # _start_hierarchical's prewarm loop will _CTRL_PREPARE it for
        # every chip child once they come up (the entry is CoW-inherited).
        if not getattr(self, "_hierarchical_started", False):
            return
        assert self._worker is not None
        if cid in self._py_callable_cids_seen:
            self._broadcast_py_control(self._python_worker_types(), _CTRL_PY_UNREGISTER, cid, strict=True)
        self._worker.broadcast_register_all(int(cid), int(target.buffer_ptr()), int(target.buffer_size()))

    def unregister(self, cid: int) -> None:
        """Drop *cid* from the registry and propagate to chip children.

        Symmetric to ``Worker.register`` for the dynamic post-init path.
        The cid slot becomes reusable for the next ``register`` call — the
        only practical way to keep a long-running worker under the
        ``MAX_REGISTERED_CALLABLE_IDS`` ceiling when JIT or plugin code
        churns through callables.

        Failure semantics (docs section 8): unregister is best-effort.
        If any chip child reports an error, the parent **warns and still
        pops the registry entry** — orch_so_table_ on the AICPU side will
        be overwritten on cid reuse, and refusing to release a known-bad
        cid would just exhaust the slot space faster.

        Raises:
          KeyError: cid was never registered.
        """
        self._wait_hierarchical_start_if_needed()
        target = None
        with self._registry_lock:
            if cid not in self._callable_registry:
                raise KeyError(f"Worker.unregister: cid={cid} not registered")
            if cid in self._pending_unregister_cids:
                raise KeyError(f"Worker.unregister: cid={cid} already pending unregister")
            target = self._callable_registry[cid]
            if self.level >= 3 and self._initialized and getattr(self, "_hierarchical_started", False):
                self._pending_unregister_cids.add(cid)
            elif self.level == 2 and self._initialized:
                assert self._chip_worker is not None
                self._chip_worker.unregister_callable(cid)
                self._callable_registry.pop(cid, None)
                return
            else:
                self._callable_registry.pop(cid, None)
                return

        try:
            if isinstance(target, ChipCallable):
                self._broadcast_unregister(cid)
            else:
                errors = self._broadcast_py_control(
                    self._python_worker_types(),
                    _CTRL_PY_UNREGISTER,
                    cid,
                    strict=False,
                )
                if errors:
                    sys.stderr.write(
                        f"Worker.unregister(cid={cid}): {len(errors)} Python children reported errors "
                        f"(continuing best-effort). First error: {errors[0]}\n"
                    )
                    sys.stderr.flush()
        finally:
            with self._registry_lock:
                self._callable_registry.pop(cid, None)
                self._pending_unregister_cids.discard(cid)

    def _broadcast_unregister(self, cid: int) -> None:
        """Broadcast _CTRL_UNREGISTER via C++ to every NEXT_LEVEL child.

        Best-effort: any per-child errors are returned by C++ as a list of
        strings; we warn to stderr and let the caller still pop the registry.
        """
        assert self._worker is not None
        errors = self._worker.broadcast_unregister_all(int(cid))
        if errors:
            sys.stderr.write(
                f"Worker.unregister(cid={cid}): {len(errors)} chips reported errors "
                f"(continuing best-effort). First error: {errors[0]}\n"
            )
            sys.stderr.flush()

    def add_worker(self, worker: "Worker") -> None:
        """Add a lower-level Worker as a NEXT_LEVEL child. Must be called before init().

        The child Worker must NOT be init'd — init happens inside the forked
        child process (so the child's own children are forked in the right
        process tree).
        """
        if self.level < 4:
            raise RuntimeError("Worker.add_worker() requires level >= 4")
        if self._config.get("device_ids", []):
            raise RuntimeError("Worker.add_worker() cannot be combined with device_ids on the same Worker")
        if self._initialized:
            raise RuntimeError("Worker.add_worker() must be called before init()")
        if worker._initialized:
            raise RuntimeError("Child worker must not be initialized before add_worker()")
        self._next_level_workers.append(worker)

    # ------------------------------------------------------------------
    # init — auto-discovery
    # ------------------------------------------------------------------

    def init(self) -> None:
        if self._initialized:
            raise RuntimeError("Worker already initialized")

        if self.level == 2:
            self._init_level2()
        elif self.level >= 3:
            self._init_hierarchical()
        else:
            raise ValueError(f"Worker: level {self.level} not supported")

        self._initialized = True

    def _init_level2(self) -> None:
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        platform = self._config["platform"]
        runtime = self._config["runtime"]
        device_id = self._config.get("device_id", 0)

        builder = RuntimeBuilder(platform)
        binaries = builder.get_binaries(runtime, build=self._config.get("build", False))

        self._chip_worker = ChipWorker()
        self._chip_worker.init(device_id, binaries)

        # Pre-warm any registered ChipCallable so the first run(cid, …)
        # does not pay the H2D upload cost.
        assert self._chip_worker is not None
        for cid, target in self._callable_registry.items():
            if isinstance(target, ChipCallable):
                self._chip_worker.prepare_callable(cid, target)

    def _init_hierarchical(self) -> None:
        device_ids = self._config.get("device_ids", [])
        n_sub = self._config.get("num_sub_workers", 0)
        heap_ring_size = self._config.get("heap_ring_size", None)
        if self.level >= 4 and device_ids:
            raise RuntimeError("Worker level >= 4 must use add_worker(); device_ids are only supported on L3 Workers")

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
            binaries = builder.get_binaries(runtime, build=self._config.get("build", False))

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

        self._hierarchical_started = False

    def _start_hierarchical(self) -> None:  # noqa: PLR0912 -- three parallel fork loops (sub/chip/next) + bootstrap wait + scheduler register/init; branches track the fork order documented in the body
        """Fork child processes and start C++ scheduler. Called on first run()."""
        with self._hierarchical_start_cv:
            while self._hierarchical_start_state == "starting":
                self._hierarchical_start_cv.wait()
            if self._hierarchical_start_state == "started":
                return
            if self._hierarchical_start_state == "failed":
                raise RuntimeError("Worker hierarchical startup failed; close this Worker and create a new one")
            self._hierarchical_start_state = "starting"

        device_ids = self._config.get("device_ids", [])
        n_sub = self._config.get("num_sub_workers", 0)

        try:
            # Fork children from an immutable snapshot. Dynamic register callers
            # that race this startup wait and then use the post-start path.
            with self._registry_lock:
                registry = dict(self._callable_registry)

            # Fork SubWorker processes (MUST be before any C++ threads)
            for i in range(n_sub):
                pid = os.fork()
                if pid == 0:
                    buf = self._sub_shms[i].buf
                    assert buf is not None
                    _sub_worker_loop(buf, registry)
                    os._exit(0)
                else:
                    self._sub_pids.append(pid)

            # Fork ChipWorker processes (L3 with device_ids).  Always use the
            # plain task-loop variant; the base communicator is established
            # lazily on first ``orch.allocate_domain`` via CTRL_COMM_INIT.
            chip_log_level, chip_log_info_v = _simpler_log.get_current_config()
            if device_ids:
                for idx, dev_id in enumerate(device_ids):
                    pid = os.fork()
                    if pid == 0:
                        buf = self._chip_shms[idx].buf
                        assert buf is not None
                        _chip_process_loop(
                            buf,
                            self._l3_bins,
                            dev_id,
                            registry,
                            chip_log_level,
                            chip_log_info_v,
                        )
                        os._exit(0)
                    else:
                        self._chip_pids.append(pid)

            # Fork next-level Worker children (L4+ with Worker children).
            # Each child process: init the inner Worker (which mmaps its own
            # HeapRing and allocates its own child mailboxes), then enter
            # _child_worker_loop. The inner Worker's own children are forked
            # lazily on first run() inside _child_worker_loop, so the process
            # tree nests correctly: L4 → L3 child → L3's chip/sub children.
            for idx, inner_worker in enumerate(self._next_level_workers):
                pid = os.fork()
                if pid == 0:
                    buf = self._next_level_shms[idx].buf
                    assert buf is not None
                    inner_worker.init()
                    _child_worker_loop(buf, registry, inner_worker)
                    os._exit(0)
                else:
                    self._next_level_pids.append(pid)

            # _Worker was constructed in _init_hierarchical (pre-fork) so
            # children inherit the HeapRing MAP_SHARED mmap. Register PROCESS-mode
            # workers via the unified mailbox.
            dw = self._worker
            assert dw is not None

            # Register chip workers as NEXT_LEVEL (L3)
            if device_ids:
                for shm in self._chip_shms:
                    dw.add_next_level_worker(_mailbox_addr(shm))

            # Register Worker children as NEXT_LEVEL (L4+)
            for shm in self._next_level_shms:
                dw.add_next_level_worker(_mailbox_addr(shm))

            for shm in self._sub_shms:
                dw.add_sub_worker(_mailbox_addr(shm))

            # Start Scheduler + WorkerThreads (C++ threads start here, after fork)
            dw.init()

            self._orch = Orchestrator(dw.get_orchestrator(), self)

            # Pre-warm every chip child: for each registered ChipCallable cid,
            # send `_CTRL_PREPARE` to all chip children so the first
            # `submit_next_level` does not pay the H2D upload cost.  Sub fns /
            # orch fns do not need pre-warming — the registry is already
            # COW-inherited.
            if device_ids:
                for cid, target in registry.items():
                    if isinstance(target, ChipCallable):
                        for worker_id in range(len(self._chip_shms)):
                            dw.control_prepare(worker_id, int(cid))

            self._hierarchical_started = True
            with self._hierarchical_start_cv:
                self._hierarchical_start_state = "started"
                self._hierarchical_start_cv.notify_all()
        except Exception:
            with self._hierarchical_start_cv:
                self._hierarchical_start_state = "failed"
                self._hierarchical_start_cv.notify_all()
            raise

    # ------------------------------------------------------------------
    # Hierarchical abort
    # ------------------------------------------------------------------

    def _abort_hierarchical(self) -> None:
        """Tear down all forked children + shms after a bootstrap failure.

        Best-effort: SIGKILL every child we spawned, reap them, then close
        and unlink every mailbox.  Called only from the init() failure path,
        so `dw.init()` has not run and the C++ scheduler is not holding any
        mailbox references.
        """
        pids = list(self._chip_pids) + list(self._sub_pids) + list(self._next_level_pids)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                pass
        for pid in pids:
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass

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
        self._sub_shms.clear()
        self._chip_shms.clear()
        self._next_level_shms.clear()

    @property
    def live_domains(self) -> dict[str, "CommDomainHandle"]:
        """Read-only snapshot of currently-live dynamic CommDomain handles.

        Useful for debugging.  Mutating the returned dict has no effect; use
        ``handle.release()`` or ``orch.release_domain(handle)`` to free.
        """
        return dict(self._live_domains)

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
        if not self._initialized:
            raise RuntimeError("allocate_domain requires Worker.init() (HCCL membership) to have run")
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
                raise ValueError(f"allocate_domain: worker index {w} outside [0, {len(device_ids)})")
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
        reply_shms: Optional[dict[int, SharedMemory]],
        op: str,
        allocation_id: int,
    ) -> None:
        """Fan out CTRL_ALLOC_DOMAIN / CTRL_RELEASE_DOMAIN to all participating chips.

        Each chip's `_Worker.control_*` is a blocking per-mailbox call; we issue
        them on separate threads so the child-side file barrier can converge.
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
                # Drop from live_domains anyway — leaving a known-bad handle
                # would just block close().
                self._live_domains.pop(handle.name, None)

    # ------------------------------------------------------------------
    # memory management — forward to C++ Orchestrator, which holds
    # per-WorkerThread mailbox_mu_ so these are safe to call concurrently
    # with in-flight dispatch on the same chip mailbox.
    # ------------------------------------------------------------------

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
        if self.level == 2:
            assert self._chip_worker is not None
            return self._chip_worker.malloc(size)
        self._check_chip_worker_id(worker_id)
        assert self._orch is not None
        return self._orch.malloc(worker_id, size)

    def free(self, ptr: int, worker_id: int = 0) -> None:
        """Free memory allocated by ``malloc()``."""
        if self.level == 2:
            assert self._chip_worker is not None
            self._chip_worker.free(ptr)
            return
        self._check_chip_worker_id(worker_id)
        assert self._orch is not None
        self._orch.free(worker_id, ptr)

    def copy_to(self, dst: int, src: int, size: int, worker_id: int = 0) -> None:
        """Copy *size* bytes from host *src* to chip worker *dst*."""
        if self.level == 2:
            assert self._chip_worker is not None
            self._chip_worker.copy_to(dst, src, size)
            return
        self._check_chip_worker_id(worker_id)
        assert self._orch is not None
        self._orch.copy_to(worker_id, dst, src, size)

    def copy_from(self, dst: int, src: int, size: int, worker_id: int = 0) -> None:
        """Copy *size* bytes from chip worker *src* to host *dst*."""
        if self.level == 2:
            assert self._chip_worker is not None
            self._chip_worker.copy_from(dst, src, size)
            return
        self._check_chip_worker_id(worker_id)
        assert self._orch is not None
        self._orch.copy_from(worker_id, dst, src, size)

    # ------------------------------------------------------------------
    # run — uniform entry point
    # ------------------------------------------------------------------

    def run(self, callable, args=None, config=None) -> RunTiming:
        """Execute one task (L2) or one DAG (L3+) synchronously.

        Dispatch:
          - L2: ``callable`` is a cid returned by ``Worker.register(chip_callable)``.
            Routes to ``_chip_worker.run(cid, args, cfg)``.
          - L3+: ``callable`` is a Python orch fn invoked with the
            ``Orchestrator`` handle.

        ``args``  : TaskArgs (optional)
        ``config``: CallConfig (optional, default-constructed if None)

        Returns a :class:`RunTiming` with ``host_wall_us`` (Python wall-clock
        around the dispatch) and ``device_wall_us`` (on-NPU orchestrator wall,
        populated whenever the runtime was built with ``PTO2_PROFILING`` —
        the default build has it on). For L3+ DAGs, ``host_wall_us`` covers
        the whole orch fn and ``device_wall_us`` is unset (0) — per-task
        device timings are not aggregated here.
        """
        assert self._initialized, "Worker not initialized; call init() first"
        cfg = config if config is not None else CallConfig()

        if self.level == 2:
            assert self._chip_worker is not None
            return self._chip_worker.run(int(callable), args, cfg)

        self._start_hierarchical()
        assert self._orch is not None
        assert self._worker is not None
        # Drop any error stashed by a previous run() so this call starts
        # clean. drain() rethrows on the way out; every successful run()
        # leaves the error slot empty, but an unrelated caller may have
        # poked it.
        self._orch._clear_error()
        self._orch._scope_begin()
        t_start = time.perf_counter_ns()
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
                self._orch._drain()
            finally:
                self._execute_pending_domain_releases()
                if self._live_domains:
                    self._release_all_live_domains()
        # device_wall stays 0 for L3+: aggregating per-task device cycles
        # across a DAG isn't implemented here (would need accumulation in the
        # ring scheduler). Callers wanting per-task device wall should issue
        # individual run calls.
        return RunTiming(time.perf_counter_ns() - t_start, 0)

    def prepare_callable(self, callable_id: int, callable) -> None:
        """L2 only: pre-stage a callable under ``callable_id`` (see
        ``ChipWorker.prepare_callable``). Subsequent ``run`` skips
        per-run kernel/orch SO upload.
        """
        assert self._initialized, "Worker not initialized; call init() first"
        if self.level != 2:
            raise NotImplementedError("prepare_callable is L2-only")
        assert self._chip_worker is not None
        self._chip_worker.prepare_callable(callable_id, callable)

    def unregister_callable(self, callable_id: int) -> None:
        """L2 only: drop the prepared state for ``callable_id``.

        Releases the host-side share of the orch SO buffer (refcounted across
        cids that share identical SO bytes) and the host dlopen handle on
        host_build_graph variants. Kernel binaries stay resident until
        ``finalize`` — they are shared across callables by ``func_id``.

        AICPU-side dlopen state in ``orch_so_table_[callable_id]`` is **not**
        released by this call. It is reclaimed lazily when the cid is reused
        (the next register triggers ``dlclose`` + reload), or at process exit.
        Long-running processes that register / unregister cids without ever
        reusing them will hold the AICPU SO handle until shutdown.
        """
        assert self._initialized, "Worker not initialized; call init() first"
        if self.level != 2:
            raise NotImplementedError("unregister_callable is L2-only")
        assert self._chip_worker is not None
        self._chip_worker.unregister_callable(callable_id)

    @property
    def aicpu_dlopen_count(self) -> int:
        """L2 only: number of distinct callable_ids the AICPU has dlopened for.

        Used by tests to assert that ``register`` + repeated ``run(cid)`` calls
        do not retrigger the AICPU dlopen for an already-seen cid. Returns 0
        on non-L2 workers (no per-cid registration there).
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

    def close(self) -> None:  # noqa: PLR0912 -- parallel teardown for _worker + sub/chip/next/bootstrap shms with ordering constraints documented inline
        if not self._initialized:
            return

        # Release any orch-allocated CommDomain handles before tearing down
        # the C++ scheduler.  Once `dw.close()` runs, the chip mailboxes
        # become unusable and we can no longer drive CTRL_RELEASE_DOMAIN.
        if self._live_domains:
            self._release_all_live_domains()

        if self.level == 2:
            if self._chip_worker:
                self._chip_worker.finalize()
        else:
            if self._worker:
                self._worker.close()
                self._worker = None
                self._orch = None

            # Shutdown SubWorker processes: write SHUTDOWN to each mailbox,
            # then waitpid + free shm.
            for shm in self._sub_shms:
                buf = shm.buf
                assert buf is not None
                _mailbox_store_i32(_buffer_field_addr(buf, _OFF_STATE), _SHUTDOWN)
            for pid in self._sub_pids:
                os.waitpid(pid, 0)
            for shm in self._sub_shms:
                shm.close()
                shm.unlink()

            # Shutdown ChipWorker processes: same pattern.
            for shm in self._chip_shms:
                buf = shm.buf
                assert buf is not None
                _mailbox_store_i32(_buffer_field_addr(buf, _OFF_STATE), _SHUTDOWN)
            for pid in self._chip_pids:
                os.waitpid(pid, 0)
            for shm in self._chip_shms:
                shm.close()
                shm.unlink()

            # Shutdown next-level Worker children (L4+): SHUTDOWN triggers
            # _child_worker_loop to call inner_worker.close() before exiting.
            for shm in self._next_level_shms:
                buf = shm.buf
                assert buf is not None
                _mailbox_store_i32(_buffer_field_addr(buf, _OFF_STATE), _SHUTDOWN)
            for pid in self._next_level_pids:
                os.waitpid(pid, 0)
            for shm in self._next_level_shms:
                shm.close()
                shm.unlink()

            self._sub_shms.clear()
            self._sub_pids.clear()
            self._chip_shms.clear()
            self._chip_pids.clear()
            self._next_level_shms.clear()
            self._next_level_pids.clear()
            self._next_level_workers.clear()

        self._initialized = False

    def __enter__(self) -> "Worker":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
