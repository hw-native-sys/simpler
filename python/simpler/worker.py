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
    w.run(chip_callable, chip_args, block_dim=24)
    w.close()

    # L3: multiple chips + SubWorkers, auto-discovery in init()
    w = Worker(level=3, device_ids=[8, 9], num_sub_workers=2,
               platform="a2a3", runtime="tensormap_and_ringbuffer")
    cid = w.register(lambda: postprocess())
    w.init()

    def my_orch(orch, args):
        r = orch.submit_next_level(chip_callable, chip_args_ptr, config, outputs=[64])
        orch.submit_sub(cid, inputs=[r.outputs[0].ptr])

    w.run(Task(orch=my_orch, args=my_args))
    w.close()
"""

import ctypes
import os
import struct
import sys
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable, Optional

from .orchestrator import Orchestrator
from .task_interface import (
    DIST_MAILBOX_SIZE,
    ChipWorker,
    DistWorker,
    _ChipWorker,
)

# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """Execution unit for Worker.run() at any level.

    For L2: call ``Worker.run(chip_callable, chip_args, config)`` directly.
    For L3+: provide an orch function ``fn(orchestrator, args)`` that builds
    the DAG via ``orchestrator.submit_*``.
    """

    orch: Callable
    args: Any = field(default=None)


# ---------------------------------------------------------------------------
# Unified mailbox layout (must match dist_worker_manager.h MAILBOX_OFF_*)
# ---------------------------------------------------------------------------
#
# One layout for both NEXT_LEVEL (chip) and SUB workers. SUB children
# read `callable` as a uint64 encoding the callable_id and ignore the
# config + args_blob region.

_OFF_STATE = 0
_OFF_ERROR = 4
_OFF_CALLABLE = 8
_OFF_BLOCK_DIM = 16
_OFF_AICPU_THREAD_NUM = 20
_OFF_ENABLE_PROFILING = 24
_OFF_ENABLE_DUMP_TENSOR = 28
_OFF_ARGS = 64

_IDLE = 0
_TASK_READY = 1
_TASK_DONE = 2
_SHUTDOWN = 3


def _mailbox_addr(shm: SharedMemory) -> int:
    buf = shm.buf
    assert buf is not None
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


def _sub_worker_loop(buf, registry: dict) -> None:
    """Runs in forked child process. Reads unified mailbox layout."""
    while True:
        state = struct.unpack_from("i", buf, _OFF_STATE)[0]
        if state == _TASK_READY:
            cid = struct.unpack_from("Q", buf, _OFF_CALLABLE)[0]
            fn = registry.get(int(cid))
            error = 0
            if fn is None:
                error = 1
            else:
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    error = 2
            struct.pack_into("i", buf, _OFF_ERROR, error)
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
    except Exception:
        _tb.print_exc()
        struct.pack_into("i", buf, _OFF_ERROR, 99)
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

            error = 0
            try:
                cw.run_from_blob(callable_ptr, args_ptr, block_dim, aicpu_tn, bool(profiling))
            except Exception:  # noqa: BLE001
                error = 1
            struct.pack_into("i", buf, _OFF_ERROR, error)
            struct.pack_into("i", buf, _OFF_STATE, _TASK_DONE)
        elif state == _SHUTDOWN:
            cw.finalize()
            break


# ---------------------------------------------------------------------------
# Worker factory
# ---------------------------------------------------------------------------


class Worker:
    """Unified worker for all hierarchy levels.

    level=2: wraps ChipWorker (one NPU device).
    level=3: wraps DistWorker(3) with ChipWorker×N + SubWorker×M,
             auto-created in init() from device_ids and num_sub_workers.
    """

    def __init__(self, level: int, **config) -> None:
        self.level = level
        self._config = config
        self._callable_registry: dict[int, Callable] = {}
        self._initialized = False

        # Level-2 internals
        self._chip_worker: Optional[ChipWorker] = None

        # Level-3 internals
        self._dist_worker: Optional[DistWorker] = None
        self._orch: Optional[Orchestrator] = None
        self._chip_shms: list[SharedMemory] = []
        self._chip_pids: list[int] = []
        self._sub_shms: list[SharedMemory] = []
        self._sub_pids: list[int] = []

    # ------------------------------------------------------------------
    # Callable registration (before init)
    # ------------------------------------------------------------------

    def register(self, fn: Callable) -> int:
        """Register a callable for SubWorker use. Must be called before init()."""
        if self.level < 3:
            raise RuntimeError("Worker.register() is only available at level 3+")
        if self._initialized:
            raise RuntimeError("Worker.register() must be called before init()")
        cid = len(self._callable_registry)
        self._callable_registry[cid] = fn
        return cid

    # ------------------------------------------------------------------
    # init — auto-discovery
    # ------------------------------------------------------------------

    def init(self) -> None:
        if self._initialized:
            raise RuntimeError("Worker already initialized")

        if self.level == 2:
            self._init_level2()
        elif self.level == 3:
            self._init_level3()
        else:
            raise ValueError(f"Worker: level {self.level} not yet supported")

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

    def _init_level3(self) -> None:
        device_ids = self._config.get("device_ids", [])
        n_sub = self._config.get("num_sub_workers", 0)
        heap_ring_size = self._config.get("heap_ring_size", None)

        # 1. Allocate sub-worker mailboxes (unified layout, DIST_MAILBOX_SIZE each).
        for _ in range(n_sub):
            shm = SharedMemory(create=True, size=DIST_MAILBOX_SIZE)
            assert shm.buf is not None
            struct.pack_into("i", shm.buf, _OFF_STATE, _IDLE)
            self._sub_shms.append(shm)

        # 2. Prepare chip-worker config (but do NOT fork yet — deferred to _start_level3)
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

            # Allocate chip mailboxes (unified layout, DIST_MAILBOX_SIZE each).
            for _ in device_ids:
                shm = SharedMemory(create=True, size=DIST_MAILBOX_SIZE)
                assert shm.buf is not None
                struct.pack_into("i", shm.buf, _OFF_STATE, _IDLE)
                self._chip_shms.append(shm)

        # 3. Construct the DistWorker *before* fork so the HeapRing mmap
        #    (taken in the C++ ctor) is inherited by every child process at
        #    the same virtual address. No C++ thread is spawned here; the
        #    scheduler + WorkerThreads start in init(), after forks.
        if heap_ring_size is None:
            self._dist_worker = DistWorker(3)
        else:
            self._dist_worker = DistWorker(3, int(heap_ring_size))

        self._l3_started = False

    def _start_level3(self) -> None:
        """Fork child processes and start C++ scheduler. Called on first run()."""
        if self._l3_started:
            return
        self._l3_started = True

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

        # Fork ChipWorker processes
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

        # DistWorker was constructed in _init_level3 (pre-fork) so children
        # inherit the HeapRing MAP_SHARED mmap. Register PROCESS-mode workers
        # via the unified mailbox — no DistChipProcess/DistSubWorker wrappers.
        dw = self._dist_worker
        assert dw is not None

        if device_ids:
            for shm in self._chip_shms:
                dw.add_next_level_process(_mailbox_addr(shm))

        for shm in self._sub_shms:
            dw.add_sub_process(_mailbox_addr(shm))

        # Start Scheduler + WorkerThreads (C++ threads start here, after fork)
        dw.init()

        self._orch = Orchestrator(dw.get_orchestrator())

    # ------------------------------------------------------------------
    # run — uniform entry point
    # ------------------------------------------------------------------

    def run(self, task_or_callable, args=None, **kwargs) -> None:
        """Execute one task synchronously.

        L2: run(chip_callable, chip_args, block_dim=N)
        L3: run(Task(orch=fn, args=...))
        """
        assert self._initialized, "Worker not initialized; call init() first"

        if self.level == 2:
            assert self._chip_worker is not None
            self._chip_worker.run(task_or_callable, args, **kwargs)
        else:
            self._start_level3()
            assert self._orch is not None
            assert self._dist_worker is not None
            task = task_or_callable
            self._orch._scope_begin()
            try:
                task.orch(self._orch, task.args)
            finally:
                # Always release scope refs and drain so ring slots aren't
                # stranded when the orch fn raises mid-DAG.
                self._orch._scope_end()
                self._orch._drain()

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
            if self._dist_worker:
                self._dist_worker.close()
                self._dist_worker = None
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

            self._sub_shms.clear()
            self._sub_pids.clear()
            self._chip_shms.clear()
            self._chip_pids.clear()

        self._initialized = False

    def __enter__(self) -> "Worker":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
