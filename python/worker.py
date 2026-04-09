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

    def my_orch(w, args):
        r = w.submit(WorkerType.CHIP, chip_payload, inputs=[...], outputs=[64])
        w.submit(WorkerType.SUB, sub_payload(cid), inputs=[r.outputs[0].ptr])

    w.run(Task(orch=my_orch, args=my_args))
    w.close()
"""

import ctypes
import os
import struct
import sys
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Callable, Optional

# Make sure examples/scripts is importable for runtime_builder
_SCRIPTS = str(Path(__file__).parent.parent / "examples" / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from task_interface import (  # noqa: E402
    DIST_CHIP_MAILBOX_SIZE,
    DIST_SUB_MAILBOX_SIZE,
    ChipWorker,
    DistChipProcess,
    DistInputSpec,
    DistOutputSpec,
    DistSubWorker,
    DistWorker,
    WorkerPayload,
    WorkerType,
    _ChipWorker,
)

# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """Execution unit for Worker.run() at any level.

    For L2: set callable/args directly on a WorkerPayload and pass to run().
    For L3+: provide an orch function that calls worker.submit().
    """

    orch: Callable
    args: Any = field(default=None)


# ---------------------------------------------------------------------------
# Mailbox helpers (shared with host_worker)
# ---------------------------------------------------------------------------

_OFF_STATE = 0
_OFF_CALLABLE_ID = 4
_IDLE = 0
_TASK_READY = 1
_TASK_DONE = 2
_SHUTDOWN = 3


def _mailbox_addr(shm: SharedMemory) -> int:
    buf = shm.buf
    assert buf is not None
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


def _sub_worker_loop(buf, registry: dict) -> None:
    """Runs in forked child process."""
    while True:
        state = struct.unpack_from("i", buf, _OFF_STATE)[0]
        if state == _TASK_READY:
            cid = struct.unpack_from("i", buf, _OFF_CALLABLE_ID)[0]
            fn = registry.get(cid)
            error = 0
            if fn is None:
                error = 1
            else:
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    error = 2
            struct.pack_into("i", buf, 24, error)
            struct.pack_into("i", buf, _OFF_STATE, _TASK_DONE)
        elif state == _SHUTDOWN:
            break


# Chip process mailbox offsets (must match dist_chip_process.h)
_CHIP_OFF_STATE = 0
_CHIP_OFF_ERROR = 4
_CHIP_OFF_CALLABLE = 8
_CHIP_OFF_BLOCK_DIM = 16
_CHIP_OFF_AICPU_THREAD_NUM = 20
_CHIP_OFF_ENABLE_PROFILING = 24
_CHIP_OFF_ARGS = 64


def _chip_process_loop(
    buf: memoryview,
    host_lib_path: str,
    device_id: int,
    aicpu_path: str,
    aicore_path: str,
    sim_context_lib_path: str = "",
    args_size: int = 1712,
) -> None:
    """Runs in forked child process. Loads host_runtime.so in own address space."""
    import traceback as _tb  # noqa: PLC0415

    try:
        cw = _ChipWorker()
        cw.init(host_lib_path, aicpu_path, aicore_path, sim_context_lib_path)
        cw.set_device(device_id)
    except Exception:
        _tb.print_exc()
        struct.pack_into("i", buf, _CHIP_OFF_ERROR, 99)
        return

    mailbox_addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    sys.stderr.write(f"[chip_process pid={os.getpid()} dev={device_id}] ready\n")
    sys.stderr.flush()

    while True:
        state = struct.unpack_from("i", buf, _CHIP_OFF_STATE)[0]
        if state == _TASK_READY:
            callable_ptr = struct.unpack_from("Q", buf, _CHIP_OFF_CALLABLE)[0]
            block_dim = struct.unpack_from("i", buf, _CHIP_OFF_BLOCK_DIM)[0]
            aicpu_tn = struct.unpack_from("i", buf, _CHIP_OFF_AICPU_THREAD_NUM)[0]
            profiling = struct.unpack_from("i", buf, _CHIP_OFF_ENABLE_PROFILING)[0]
            args_ptr = mailbox_addr + _CHIP_OFF_ARGS

            # Copy args from shm to heap — run_runtime requires heap-backed args
            args_buf = ctypes.create_string_buffer(args_size)
            ctypes.memmove(args_buf, args_ptr, args_size)
            heap_args_ptr = ctypes.addressof(args_buf)

            error = 0
            try:
                cw.run_raw(callable_ptr, heap_args_ptr, block_dim, aicpu_tn, bool(profiling))
            except Exception:  # noqa: BLE001
                error = 1
            struct.pack_into("i", buf, _CHIP_OFF_ERROR, error)
            struct.pack_into("i", buf, _CHIP_OFF_STATE, _TASK_DONE)
        elif state == _SHUTDOWN:
            cw.finalize()
            break


# ---------------------------------------------------------------------------
# Worker factory
# ---------------------------------------------------------------------------


class _ScopeGuard:
    """RAII scope guard for DistWorker.scope_begin/scope_end."""

    def __init__(self, dw: DistWorker) -> None:
        self._dw = dw

    def __enter__(self):
        self._dw.scope_begin()
        return self

    def __exit__(self, *_):
        self._dw.scope_end()


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
        self._dist_chip_procs: list[DistChipProcess] = []
        self._chip_shms: list[SharedMemory] = []
        self._chip_pids: list[int] = []
        self._dist_sub_workers: list[DistSubWorker] = []
        self._shms: list[SharedMemory] = []
        self._pids: list[int] = []

    # ------------------------------------------------------------------
    # Callable registration (before init)
    # ------------------------------------------------------------------

    def register(self, fn: Callable) -> int:
        """Register a callable for SubWorker use. Must be called before init()."""
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
        from runtime_builder import RuntimeBuilder  # noqa: PLC0415

        platform = self._config["platform"]
        runtime = self._config["runtime"]
        device_id = self._config.get("device_id", 0)

        builder = RuntimeBuilder(platform)
        binaries = builder.get_binaries(runtime, build=False)

        self._chip_worker = ChipWorker()
        self._chip_worker.init(
            str(binaries.host_path),
            str(binaries.aicpu_path),
            str(binaries.aicore_path),
            str(binaries.sim_context_path) if hasattr(binaries, "sim_context_path") else "",
        )
        self._chip_worker.set_device(device_id)

    def _init_level3(self) -> None:
        device_ids = self._config.get("device_ids", [])
        n_sub = self._config.get("num_sub_workers", 0)

        # 1. Allocate mailboxes
        for _ in range(n_sub):
            shm = SharedMemory(create=True, size=DIST_SUB_MAILBOX_SIZE)
            assert shm.buf is not None
            struct.pack_into("i", shm.buf, _OFF_STATE, _IDLE)
            self._shms.append(shm)

        # 2. Fork SubWorker processes (MUST be before any C++ threads)
        registry = self._callable_registry
        for i in range(n_sub):
            pid = os.fork()
            if pid == 0:
                buf = self._shms[i].buf
                assert buf is not None
                _sub_worker_loop(buf, registry)
                os._exit(0)
            else:
                self._pids.append(pid)

        # 3. Fork ChipWorker processes (only if device_ids provided)
        if device_ids:
            from runtime_builder import RuntimeBuilder  # noqa: PLC0415
            from task_interface import ChipStorageTaskArgs as _CSA  # noqa: PLC0415

            platform = self._config["platform"]
            runtime = self._config["runtime"]
            builder = RuntimeBuilder(platform)
            binaries = builder.get_binaries(runtime, build=False)

            # Determine args_size (sizeof ChipStorageTaskArgs)
            _objs = [_CSA() for _ in range(5)]
            _ptrs = [o.__ptr__() for o in _objs]
            args_size = min(abs(_ptrs[i + 1] - _ptrs[i]) for i in range(len(_ptrs) - 1))
            del _objs, _ptrs

            host_lib_path = str(binaries.host_path)
            aicpu_path = str(binaries.aicpu_path)
            aicore_path = str(binaries.aicore_path)
            sim_ctx_path = str(binaries.sim_context_path) if getattr(binaries, "sim_context_path", None) else ""

            for dev_id in device_ids:
                shm = SharedMemory(create=True, size=DIST_CHIP_MAILBOX_SIZE)
                assert shm.buf is not None
                struct.pack_into("i", shm.buf, _CHIP_OFF_STATE, _IDLE)
                self._chip_shms.append(shm)

                pid = os.fork()
                if pid == 0:
                    buf = shm.buf
                    assert buf is not None
                    _chip_process_loop(buf, host_lib_path, dev_id, aicpu_path, aicore_path, sim_ctx_path, args_size)
                    os._exit(0)
                else:
                    self._chip_pids.append(pid)

        # 4. Create DistWorker and wire chip processes + sub workers
        dw = DistWorker(3)
        self._dist_worker = dw

        if device_ids:
            for shm in self._chip_shms:
                cp = DistChipProcess(_mailbox_addr(shm), args_size)
                self._dist_chip_procs.append(cp)
                dw.add_chip_process(cp)

        for shm in self._shms:
            sw = DistSubWorker(_mailbox_addr(shm))
            self._dist_sub_workers.append(sw)
            dw.add_sub_worker(sw)

        # 6. Start Scheduler + WorkerThreads (C++ threads start here, after fork)
        dw.init()

    # ------------------------------------------------------------------
    # run — uniform entry point
    # ------------------------------------------------------------------

    def run(self, task_or_payload, args=None, **kwargs) -> None:
        """Execute one task synchronously.

        L2: run(chip_callable, chip_args, block_dim=N)
            or run(WorkerPayload(...))
        L3: run(Task(orch=fn, args=...))
        """
        assert self._initialized, "Worker not initialized; call init() first"

        if self.level == 2:
            assert self._chip_worker is not None
            if isinstance(task_or_payload, WorkerPayload):
                from task_interface import CallConfig  # noqa: PLC0415

                config = CallConfig()
                config.block_dim = task_or_payload.block_dim
                config.aicpu_thread_num = task_or_payload.aicpu_thread_num
                config.enable_profiling = task_or_payload.enable_profiling
                self._chip_worker.run(
                    task_or_payload.callable,  # type: ignore[arg-type]
                    task_or_payload.args,
                    config,
                )
            else:
                # run(callable, args, **kwargs)
                self._chip_worker.run(task_or_payload, args, **kwargs)
        else:
            assert self._dist_worker is not None
            task = task_or_payload
            task.orch(self, task.args)
            self._dist_worker.drain()

    # ------------------------------------------------------------------
    # Orchestration API (called from inside orch functions at L3+)
    # ------------------------------------------------------------------

    def submit(
        self,
        worker_type: WorkerType,
        payload: WorkerPayload,
        inputs: Optional[list[int]] = None,
        outputs: Optional[list[int]] = None,
        args_list: Optional[list[int]] = None,
    ):
        """Submit a task. If args_list has >1 entries, submits a group task."""
        assert self._dist_worker is not None
        in_specs = [DistInputSpec(p) for p in (inputs or [])]
        out_specs = [DistOutputSpec(s) for s in (outputs or [])]
        if args_list and len(args_list) > 1:
            return self._dist_worker.submit_group(worker_type, payload, args_list, in_specs, out_specs)
        return self._dist_worker.submit(worker_type, payload, in_specs, out_specs)

    def scope(self):
        """Context manager for scope lifetime. Usage: ``with w.scope(): ...``"""
        assert self._dist_worker is not None
        return _ScopeGuard(self._dist_worker)

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

            # Shutdown SubWorker processes
            for sw in self._dist_sub_workers:
                sw.shutdown()
            for shm in self._shms:
                buf = shm.buf
                assert buf is not None
                struct.pack_into("i", buf, _OFF_STATE, _SHUTDOWN)
            for pid in self._pids:
                os.waitpid(pid, 0)
            for shm in self._shms:
                shm.close()
                shm.unlink()

            # Shutdown ChipWorker processes
            for cp in self._dist_chip_procs:
                cp.shutdown()
            for pid in self._chip_pids:
                os.waitpid(pid, 0)
            for shm in self._chip_shms:
                shm.close()
                shm.unlink()

            self._shms.clear()
            self._pids.clear()
            self._chip_shms.clear()
            self._chip_pids.clear()
            self._dist_sub_workers.clear()
            self._dist_chip_procs.clear()

        self._initialized = False

    def __enter__(self) -> "Worker":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
