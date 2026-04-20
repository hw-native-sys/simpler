# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Worker — unified factory for all hierarchy levels.

Usage::

    # L2: one NPU chip
    w = Worker(level=2, device_id=8, platform="a2a3", runtime="tensormap_and_ringbuffer")
    w.init()
    w.run(chip_callable, chip_args, config)
    w.close()

    # L3: multiple chips + SubWorkers, auto-discovery in init()
    w = Worker(level=3, device_ids=[8, 9], num_sub_workers=2,
               platform="a2a3", runtime="tensormap_and_ringbuffer")
    cid = w.register(lambda args: postprocess())
    w.init()

    def my_orch(orch, args, cfg):
        r = orch.submit_next_level(chip_callable, chip_args_ptr, cfg)
        orch.submit_sub(cid, sub_args)

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
import struct
import sys
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable, Optional

from .orchestrator import Orchestrator
from .task_interface import (
    MAILBOX_ERROR_MSG_SIZE,
    MAILBOX_OFF_ERROR_MSG,
    MAILBOX_SIZE,
    ChipCallConfig,
    ChipWorker,
    ContinuousTensor,
    DataType,
    TaskArgs,
    _ChipWorker,
    _Worker,
)

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
_OFF_BLOCK_DIM = 16
_OFF_AICPU_THREAD_NUM = 20
_OFF_ENABLE_PROFILING = 24
_OFF_ENABLE_DUMP_TENSOR = 28
_OFF_ARGS = 64
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


def _mailbox_addr(shm: SharedMemory) -> int:
    buf = shm.buf
    assert buf is not None
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


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

    Blob layout at _OFF_ARGS:
      int32 tensor_count (T), int32 scalar_count (S),
      ContinuousTensor[T] (40 B each), uint64_t[S] (8 B each).
    """
    base = _OFF_ARGS
    t_count = struct.unpack_from("i", buf, base)[0]
    s_count = struct.unpack_from("i", buf, base + 4)[0]

    args = TaskArgs()
    ct_off = base + 8
    for i in range(t_count):
        off = ct_off + i * 40
        data = struct.unpack_from("Q", buf, off)[0]
        shapes = struct.unpack_from("5I", buf, off + 8)
        ndims = struct.unpack_from("I", buf, off + 28)[0]
        dtype_val = struct.unpack_from("B", buf, off + 32)[0]
        ct = ContinuousTensor.make(data, tuple(shapes[:ndims]), DataType(dtype_val))
        args.add_tensor(ct)

    sc_off = ct_off + t_count * 40
    for i in range(s_count):
        args.add_scalar(struct.unpack_from("Q", buf, sc_off + i * 8)[0])

    return args


def _sub_worker_loop(buf, registry: dict) -> None:
    """Runs in forked child process. Reads unified mailbox layout.

    On success writes ``error=0`` and an empty message. On failure writes
    ``error=1`` and ``f"sub_worker: <ExcType>: <msg>"`` into the mailbox
    error-message region; the parent's ``WorkerThread::dispatch_process``
    rethrows it as ``std::runtime_error``.
    """
    while True:
        state = struct.unpack_from("i", buf, _OFF_STATE)[0]
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
            struct.pack_into("i", buf, _OFF_STATE, _TASK_DONE)
        elif state == _SHUTDOWN:
            break


def _chip_process_loop(
    buf: memoryview,
    host_lib_path: str,
    device_id: int,
    aicpu_path: str,
    aicore_path: str,
    sim_context_lib_path: str = "",
) -> None:
    """Runs in forked child process. Loads host_runtime.so in own address space.

    Reads the unified mailbox layout (same offsets as _sub_worker_loop, but
    this loop also consumes config fields + args_blob).
    """
    import traceback as _tb  # noqa: PLC0415

    try:
        cw = _ChipWorker()
        cw.init(host_lib_path, aicpu_path, aicore_path, sim_context_lib_path)
        cw.set_device(device_id)
    except Exception as e:
        _tb.print_exc()
        # Write the message so any parent reader that *does* inspect this
        # path sees the real cause. State handshake for this init-time
        # failure is broken — see KNOWN_ISSUES.md — and that is not part
        # of the L4 scope.
        _write_error(buf, 1, _format_exc(f"chip_process dev={device_id} init", e))
        return

    mailbox_addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    args_ptr = mailbox_addr + _OFF_ARGS
    sys.stderr.write(f"[chip_process pid={os.getpid()} dev={device_id}] ready\n")
    sys.stderr.flush()

    while True:
        state = struct.unpack_from("i", buf, _OFF_STATE)[0]
        if state == _TASK_READY:
            callable_ptr = struct.unpack_from("Q", buf, _OFF_CALLABLE)[0]
            block_dim = struct.unpack_from("i", buf, _OFF_BLOCK_DIM)[0]
            aicpu_tn = struct.unpack_from("i", buf, _OFF_AICPU_THREAD_NUM)[0]
            profiling = struct.unpack_from("i", buf, _OFF_ENABLE_PROFILING)[0]

            code = 0
            msg = ""
            try:
                cw.run_from_blob(callable_ptr, args_ptr, block_dim, aicpu_tn, bool(profiling))
            except Exception as e:  # noqa: BLE001
                code = 1
                msg = _format_exc(f"chip_process dev={device_id}", e)
            _write_error(buf, code, msg)
            struct.pack_into("i", buf, _OFF_STATE, _TASK_DONE)
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
            except Exception as e:  # noqa: BLE001
                code = 1
                msg = _format_exc(f"chip_process dev={device_id} ctrl={int(sub_cmd)}", e)
            _write_error(buf, code, msg)
            struct.pack_into("i", buf, _OFF_STATE, _CONTROL_DONE)
        elif state == _SHUTDOWN:
            cw.finalize()
            break


def _read_config_from_mailbox(buf: memoryview) -> "ChipCallConfig":
    """Reconstruct a ChipCallConfig from the unified mailbox layout."""
    cfg = ChipCallConfig()
    cfg.block_dim = struct.unpack_from("i", buf, _OFF_BLOCK_DIM)[0]
    cfg.aicpu_thread_num = struct.unpack_from("i", buf, _OFF_AICPU_THREAD_NUM)[0]
    cfg.enable_profiling = bool(struct.unpack_from("i", buf, _OFF_ENABLE_PROFILING)[0])
    cfg.enable_dump_tensor = bool(struct.unpack_from("i", buf, _OFF_ENABLE_DUMP_TENSOR)[0])
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
    runs the orch function, and drains.
    """
    while True:
        state = struct.unpack_from("i", buf, _OFF_STATE)[0]
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
            struct.pack_into("i", buf, _OFF_STATE, _TASK_DONE)
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

    def __init__(self, level: int, **config) -> None:
        self.level = level
        self._config = config
        self._callable_registry: dict[int, Callable] = {}
        self._initialized = False

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

    # ------------------------------------------------------------------
    # Callable registration (before init)
    # ------------------------------------------------------------------

    def register(self, fn: Callable) -> int:
        """Register a callable (sub or orch fn). Must be called before init()."""
        if self.level < 3:
            raise RuntimeError("Worker.register() is only available at level 3+")
        if self._initialized:
            raise RuntimeError("Worker.register() must be called before init()")
        cid = len(self._callable_registry)
        self._callable_registry[cid] = fn
        return cid

    def add_worker(self, worker: "Worker") -> None:
        """Add a lower-level Worker as a NEXT_LEVEL child. Must be called before init().

        The child Worker must NOT be init'd — init happens inside the forked
        child process (so the child's own children are forked in the right
        process tree).
        """
        if self.level < 4:
            raise RuntimeError("Worker.add_worker() requires level >= 4")
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
        self._chip_worker.init(
            str(binaries.host_path),
            str(binaries.aicpu_path),
            str(binaries.aicore_path),
            str(binaries.sim_context_path) if binaries.sim_context_path else "",
        )
        self._chip_worker.set_device(device_id)

    def _init_hierarchical(self) -> None:
        device_ids = self._config.get("device_ids", [])
        n_sub = self._config.get("num_sub_workers", 0)
        heap_ring_size = self._config.get("heap_ring_size", None)

        # 1. Allocate sub-worker mailboxes (unified layout, MAILBOX_SIZE each).
        for _ in range(n_sub):
            shm = SharedMemory(create=True, size=MAILBOX_SIZE)
            assert shm.buf is not None
            struct.pack_into("i", shm.buf, _OFF_STATE, _IDLE)
            self._sub_shms.append(shm)

        # 2. Prepare chip-worker config (L3 only — L4+ has Worker children instead)
        if device_ids:
            from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

            platform = self._config["platform"]
            runtime = self._config["runtime"]
            builder = RuntimeBuilder(platform)
            binaries = builder.get_binaries(runtime, build=self._config.get("build", False))

            self._l3_host_lib_path = str(binaries.host_path)
            self._l3_aicpu_path = str(binaries.aicpu_path)
            self._l3_aicore_path = str(binaries.aicore_path)
            self._l3_sim_ctx_path = (
                str(binaries.sim_context_path) if getattr(binaries, "sim_context_path", None) else ""
            )

            # Allocate chip mailboxes (unified layout, MAILBOX_SIZE each).
            for _ in device_ids:
                shm = SharedMemory(create=True, size=MAILBOX_SIZE)
                assert shm.buf is not None
                struct.pack_into("i", shm.buf, _OFF_STATE, _IDLE)
                self._chip_shms.append(shm)

        # 3. Allocate next-level Worker child mailboxes (L4+ only).
        for _ in self._next_level_workers:
            shm = SharedMemory(create=True, size=MAILBOX_SIZE)
            assert shm.buf is not None
            struct.pack_into("i", shm.buf, _OFF_STATE, _IDLE)
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

    def _start_hierarchical(self) -> None:
        """Fork child processes and start C++ scheduler. Called on first run()."""
        if self._hierarchical_started:
            return
        self._hierarchical_started = True

        device_ids = self._config.get("device_ids", [])
        n_sub = self._config.get("num_sub_workers", 0)

        # Fork SubWorker processes (MUST be before any C++ threads)
        registry = self._callable_registry
        for i in range(n_sub):
            pid = os.fork()
            if pid == 0:
                buf = self._sub_shms[i].buf
                assert buf is not None
                _sub_worker_loop(buf, registry)
                os._exit(0)
            else:
                self._sub_pids.append(pid)

        # Fork ChipWorker processes (L3 with device_ids)
        if device_ids:
            for idx, dev_id in enumerate(device_ids):
                pid = os.fork()
                if pid == 0:
                    buf = self._chip_shms[idx].buf
                    assert buf is not None
                    _chip_process_loop(
                        buf,
                        self._l3_host_lib_path,
                        dev_id,
                        self._l3_aicpu_path,
                        self._l3_aicore_path,
                        self._l3_sim_ctx_path,
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
                dw.add_next_level_process(_mailbox_addr(shm))

        # Register Worker children as NEXT_LEVEL (L4+)
        for shm in self._next_level_shms:
            dw.add_next_level_process(_mailbox_addr(shm))

        for shm in self._sub_shms:
            dw.add_sub_process(_mailbox_addr(shm))

        # Start Scheduler + WorkerThreads (C++ threads start here, after fork)
        dw.init()

        self._orch = Orchestrator(dw.get_orchestrator())

    # ------------------------------------------------------------------
    # memory management
    # ------------------------------------------------------------------

    def _chip_control(self, worker_id: int, sub_cmd: int, arg0: int = 0, arg1: int = 0, arg2: int = 0) -> int:
        """Send a control command to chip child *worker_id* via mailbox IPC."""
        if worker_id < 0 or worker_id >= len(self._chip_shms):
            raise IndexError(f"worker_id {worker_id} out of range (have {len(self._chip_shms)} chips)")
        shm = self._chip_shms[worker_id]
        buf = shm.buf
        assert buf is not None
        _write_error(buf, 0, "")
        struct.pack_into("Q", buf, _OFF_CALLABLE, sub_cmd)
        struct.pack_into("Q", buf, _CTRL_OFF_ARG0, arg0)
        struct.pack_into("Q", buf, _CTRL_OFF_ARG1, arg1)
        struct.pack_into("Q", buf, _CTRL_OFF_ARG2, arg2)
        struct.pack_into("i", buf, _OFF_STATE, _CONTROL_REQUEST)
        while struct.unpack_from("i", buf, _OFF_STATE)[0] != _CONTROL_DONE:
            pass
        error = struct.unpack_from("i", buf, _OFF_ERROR)[0]
        if error != 0:
            err_msg = _read_error_msg(buf)
            struct.pack_into("i", buf, _OFF_STATE, _IDLE)
            raise RuntimeError(f"chip control command {sub_cmd} failed on worker {worker_id}: {err_msg}")
        result = struct.unpack_from("Q", buf, _CTRL_OFF_RESULT)[0]
        struct.pack_into("i", buf, _OFF_STATE, _IDLE)
        return result

    def malloc(self, size: int, worker_id: int = 0) -> int:
        """Allocate memory on next-level worker *worker_id*. Returns a pointer."""
        if self.level == 2:
            assert self._chip_worker is not None
            return self._chip_worker.malloc(size)
        return self._chip_control(worker_id, _CTRL_MALLOC, arg0=size)

    def free(self, ptr: int, worker_id: int = 0) -> None:
        """Free memory allocated by ``malloc()``."""
        if self.level == 2:
            assert self._chip_worker is not None
            self._chip_worker.free(ptr)
            return
        self._chip_control(worker_id, _CTRL_FREE, arg0=ptr)

    def copy_to(self, dst: int, src: int, size: int, worker_id: int = 0) -> None:
        """Copy *size* bytes from host *src* to worker *dst*."""
        if self.level == 2:
            assert self._chip_worker is not None
            self._chip_worker.copy_to(dst, src, size)
            return
        self._chip_control(worker_id, _CTRL_COPY_TO, arg0=dst, arg1=src, arg2=size)

    def copy_from(self, dst: int, src: int, size: int, worker_id: int = 0) -> None:
        """Copy *size* bytes from worker *src* to host *dst*."""
        if self.level == 2:
            assert self._chip_worker is not None
            self._chip_worker.copy_from(dst, src, size)
            return
        self._chip_control(worker_id, _CTRL_COPY_FROM, arg0=dst, arg1=src, arg2=size)

    # ------------------------------------------------------------------
    # run — uniform entry point
    # ------------------------------------------------------------------

    def run(self, callable, args=None, config=None) -> None:
        """Execute one task (L2) or one DAG (L3+) synchronously.

        callable: ChipCallable (L2) or Python orch fn (L3+)
        args:     TaskArgs (optional)
        config:   ChipCallConfig (optional, default-constructed if None)
        """
        assert self._initialized, "Worker not initialized; call init() first"
        cfg = config if config is not None else ChipCallConfig()

        if self.level == 2:
            assert self._chip_worker is not None
            self._chip_worker.run(callable, args, cfg)
        else:
            self._start_hierarchical()
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
                self._orch._drain()

    def _run_as_child(self, cid: int, args, config) -> None:
        """Called from C++ _Worker::run when this Worker is a THREAD-mode child.

        Looks up the orch function from the callable registry and delegates
        to ``self.run(orch_fn, args, config)``.
        """
        orch_fn = self._callable_registry.get(cid)
        if orch_fn is None:
            raise KeyError(f"callable id {cid} not found in registry")
        self.run(orch_fn, args, config)

    # ------------------------------------------------------------------
    # close
    # ------------------------------------------------------------------

    def close(self) -> None:
        if not self._initialized:
            return

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
                struct.pack_into("i", buf, _OFF_STATE, _SHUTDOWN)
            for pid in self._sub_pids:
                os.waitpid(pid, 0)
            for shm in self._sub_shms:
                shm.close()
                shm.unlink()

            # Shutdown ChipWorker processes: same pattern.
            for shm in self._chip_shms:
                buf = shm.buf
                assert buf is not None
                struct.pack_into("i", buf, _OFF_STATE, _SHUTDOWN)
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
                struct.pack_into("i", buf, _OFF_STATE, _SHUTDOWN)
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
