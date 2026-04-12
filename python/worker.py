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

    # L3 chip bootstrap extension: keep run/submit standard, pass optional
    # per-chip bootstrap metadata through Worker.init().
    w = Worker(level=3, device_ids=[8, 9],
               chip_bootstrap_configs=[...])
"""

import ctypes
import os
import signal
import struct
import sys
import time
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Callable, Optional

# Make sure examples/scripts is importable for runtime_builder
_SCRIPTS = str(Path(__file__).parent.parent / "examples" / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from task_interface import (  # noqa: E402
    DIST_CHIP_BOOTSTRAP_MAILBOX_SIZE,
    DIST_CHIP_MAILBOX_SIZE,
    DIST_SUB_MAILBOX_SIZE,
    ChipBootstrapMailboxState,
    ChipWorker,
    DataType,
    DistChipBootstrapChannel,
    DistChipProcess,
    DistInputSpec,
    DistOutputSpec,
    DistSubWorker,
    DistWorker,
    WorkerPayload,
    WorkerType,
    make_device_tensor_arg,
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


@dataclass
class ChipBufferSpec:
    """Per-chip buffer contract used by the optional L3 chip bootstrap path."""

    name: str
    dtype: str
    count: int
    placement: str
    nbytes: int
    load_from_host: bool = False
    store_to_host: bool = False

    def make_tensor_arg(self, ptr: int) -> Any:
        return make_device_tensor_arg(ptr, (self.count,), _buffer_dtype_to_task_dtype(self.dtype))


@dataclass
class HostBufferStaging:
    """Named shared-memory staging region prepared by the parent process."""

    name: str
    shm_name: str
    size: int


@dataclass
class ChipCommBootstrapConfig:
    """Optional communicator bootstrap for a chip child."""

    rank: int
    nranks: int
    rootinfo_path: str
    window_size: int
    win_sync_prefix: int = 0


@dataclass
class ChipBootstrapConfig:
    """Worker-side chip child bootstrap input."""

    comm: Optional[ChipCommBootstrapConfig] = None
    buffers: list[ChipBufferSpec] = field(default_factory=list)
    host_inputs: list[HostBufferStaging] = field(default_factory=list)
    host_outputs: list[HostBufferStaging] = field(default_factory=list)

    def input_staging(self, name: str) -> HostBufferStaging:
        for staging in self.host_inputs:
            if staging.name == name:
                return staging
        raise KeyError(f"Missing staged host input for chip buffer '{name}'")

    def output_staging(self, name: str) -> HostBufferStaging:
        for staging in self.host_outputs:
            if staging.name == name:
                return staging
        raise KeyError(f"Missing staged host output for chip buffer '{name}'")


@dataclass
class ChipBootstrapReply:
    """Child -> parent bootstrap reply carried over the bootstrap mailbox."""

    device_ctx: int
    local_window_base: int
    actual_window_size: int
    buffer_ptrs: list[int]


@dataclass
class ChipBootstrapState:
    """Child-local chip bootstrap state kept alive for the chip process lifetime."""

    bootstrap_config: ChipBootstrapConfig
    comm_handle: Optional[int]
    bootstrap_reply: ChipBootstrapReply

    @property
    def buffers(self) -> list[ChipBufferSpec]:
        return self.bootstrap_config.buffers

    @property
    def buffer_ptrs(self) -> dict[str, int]:
        return {
            buf.name: int(ptr)
            for buf, ptr in zip(self.bootstrap_config.buffers, self.bootstrap_reply.buffer_ptrs, strict=True)
        }

    @property
    def device_ctx(self) -> int:
        return self.bootstrap_reply.device_ctx

    @property
    def local_window_base(self) -> int:
        return self.bootstrap_reply.local_window_base

    @property
    def actual_window_size(self) -> int:
        return self.bootstrap_reply.actual_window_size

    def input_staging(self, name: str) -> HostBufferStaging:
        return self.bootstrap_config.input_staging(name)

    def output_staging(self, name: str) -> HostBufferStaging:
        return self.bootstrap_config.output_staging(name)


@dataclass
class ChipContext:
    """Parent-visible chip bootstrap result used to build task args."""

    bootstrap_config: ChipBootstrapConfig
    device_id: int
    bootstrap_reply: ChipBootstrapReply
    buffer_tensors: dict[str, Any]

    @property
    def rank(self) -> int:
        if self.bootstrap_config.comm is None:
            raise AttributeError("ChipContext.rank is only available when comm bootstrap is configured")
        return self.bootstrap_config.comm.rank

    @property
    def nranks(self) -> int:
        if self.bootstrap_config.comm is None:
            raise AttributeError("ChipContext.nranks is only available when comm bootstrap is configured")
        return self.bootstrap_config.comm.nranks

    @property
    def device_ctx(self) -> int:
        return self.bootstrap_reply.device_ctx

    @property
    def local_window_base(self) -> int:
        return self.bootstrap_reply.local_window_base

    @property
    def actual_window_size(self) -> int:
        return self.bootstrap_reply.actual_window_size

    @property
    def buffer_ptrs(self) -> dict[str, int]:
        return {
            buf.name: int(ptr)
            for buf, ptr in zip(self.bootstrap_config.buffers, self.bootstrap_reply.buffer_ptrs, strict=True)
        }


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


def _write_shared_memory_bytes(shm_name: str, data: bytes, expected_size: int) -> None:
    if len(data) != expected_size:
        raise ValueError(f"shared-memory staging size mismatch: got {len(data)}, expected {expected_size}")
    shm = SharedMemory(name=shm_name)
    try:
        assert shm.buf is not None
        if expected_size:
            shm.buf[:expected_size] = data
    finally:
        shm.close()


_DIST_DTYPE_MAP = {
    "float32": DataType.FLOAT32,
    "float16": DataType.FLOAT16,
    "bfloat16": DataType.BFLOAT16,
    "int64": DataType.INT64,
    "int32": DataType.INT32,
    "int16": DataType.INT16,
    "int8": DataType.INT8,
    "uint8": DataType.UINT8,
}


def _buffer_dtype_to_task_dtype(dtype: str) -> DataType:
    key = str(dtype).lower()
    if key not in _DIST_DTYPE_MAP:
        raise ValueError(f"Unsupported chip buffer dtype: {dtype}")
    return _DIST_DTYPE_MAP[key]


def _materialize_buffer_tensors(
    chip_bootstrap_config: ChipBootstrapConfig, buffer_ptrs: list[int]
) -> dict[str, Any]:
    buffer_tensors: dict[str, Any] = {}
    for buf_cfg, ptr in zip(chip_bootstrap_config.buffers, buffer_ptrs, strict=True):
        name = buf_cfg.name
        buffer_tensors[name] = buf_cfg.make_tensor_arg(ptr)
    return buffer_tensors


def _enrich_chip_context(
    chip_bootstrap_config: ChipBootstrapConfig, device_id: int, reply: ChipBootstrapReply
) -> ChipContext:
    return ChipContext(
        bootstrap_config=chip_bootstrap_config,
        device_id=device_id,
        bootstrap_reply=reply,
        buffer_tensors=_materialize_buffer_tensors(chip_bootstrap_config, reply.buffer_ptrs),
    )


def _write_chip_bootstrap_reply(bootstrap_channel: DistChipBootstrapChannel, reply: ChipBootstrapReply) -> None:
    bootstrap_channel.write_success(
        reply.device_ctx,
        reply.local_window_base,
        reply.actual_window_size,
        reply.buffer_ptrs,
    )


def _run_chip_bootstrap(
    cw: ChipWorker, device_id: int, chip_bootstrap_config: ChipBootstrapConfig
) -> ChipBootstrapState:
    bootstrap = cw.bootstrap_context(device_id, chip_bootstrap_config)
    return ChipBootstrapState(
        bootstrap_config=chip_bootstrap_config,
        comm_handle=bootstrap.comm_handle,
        bootstrap_reply=ChipBootstrapReply(
            device_ctx=bootstrap.device_ctx,
            local_window_base=bootstrap.local_window_base,
            actual_window_size=bootstrap.actual_window_size,
            buffer_ptrs=list(bootstrap.buffer_ptrs),
        ),
    )


def _chip_process_loop(
    buf: memoryview,
    host_lib_path: str,
    device_id: int,
    aicpu_path: str,
    aicore_path: str,
    sim_context_lib_path: str = "",
    args_size: int = 1712,
    bootstrap_mailbox_ptr: Optional[int] = None,
    bootstrap_buffer_count: int = 0,
    chip_bootstrap_config: Optional[ChipBootstrapConfig] = None,
) -> None:
    """Runs in forked child process. Loads host_runtime.so in own address space."""
    import traceback as _tb  # noqa: PLC0415

    cw: Optional[ChipWorker] = None
    chip_context: Optional[ChipBootstrapState] = None
    bootstrap_channel = (
        DistChipBootstrapChannel(bootstrap_mailbox_ptr, bootstrap_buffer_count)
        if bootstrap_mailbox_ptr is not None
        else None
    )
    try:
        cw = ChipWorker()
        cw.init(host_lib_path, aicpu_path, aicore_path, sim_context_lib_path)
        if chip_bootstrap_config is not None:
            chip_context = _run_chip_bootstrap(cw, device_id, chip_bootstrap_config)
            if bootstrap_channel is not None:
                _write_chip_bootstrap_reply(bootstrap_channel, chip_context.bootstrap_reply)
        elif bootstrap_channel is not None:
            cw.set_device(device_id)
            bootstrap_channel.write_success(0, 0, 0, [])
        else:
            cw.set_device(device_id)
    except Exception:
        if bootstrap_channel is not None:
            try:
                bootstrap_channel.write_error(1, _tb.format_exc())
            except Exception:  # noqa: BLE001
                pass
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
                cw.run_raw(
                    callable_ptr,
                    heap_args_ptr,
                    block_dim=block_dim,
                    aicpu_thread_num=aicpu_tn,
                    enable_profiling=bool(profiling),
                )
                if chip_context is not None:
                    if chip_context.comm_handle is not None:
                        cw.comm_barrier(chip_context.comm_handle)
                    for buf_cfg in chip_context.buffers:
                        if not buf_cfg.store_to_host:
                            continue
                        ptr = chip_context.buffer_ptrs[buf_cfg.name]
                        staged = chip_context.output_staging(buf_cfg.name)
                        _write_shared_memory_bytes(
                            staged.shm_name,
                            cw.copy_device_to_bytes(ptr, buf_cfg.nbytes),
                            staged.size,
                        )
            except Exception:  # noqa: BLE001
                error = 1
            struct.pack_into("i", buf, _CHIP_OFF_ERROR, error)
            struct.pack_into("i", buf, _CHIP_OFF_STATE, _TASK_DONE)
        elif state == _SHUTDOWN:
            if chip_context is not None:
                cw.shutdown_bootstrap_context(
                    chip_context.bootstrap_config,
                    comm_handle=chip_context.comm_handle or 0,
                    buffer_ptrs=[chip_context.buffer_ptrs[buf.name] for buf in chip_context.buffers],
                )
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
        self._chip_contexts: list[ChipContext] = []
        self._chip_bootstrap_shms: list[SharedMemory] = []
        self._chip_shms: list[SharedMemory] = []
        self._chip_pids: list[int] = []
        self._dist_sub_workers: list[DistSubWorker] = []
        self._subworker_shms: list[SharedMemory] = []
        self._subworker_pids: list[int] = []

    def _resolve_chip_bootstrap_configs(self, device_ids: list[int]) -> Optional[list[ChipBootstrapConfig]]:
        configs = self._config.get("chip_bootstrap_configs")
        if configs is None:
            return None
        if not isinstance(configs, list):
            raise TypeError("chip_bootstrap_configs must be a list of ChipBootstrapConfig")
        if any(not isinstance(cfg, ChipBootstrapConfig) for cfg in configs):
            raise TypeError("chip_bootstrap_configs items must be ChipBootstrapConfig")
        if len(configs) != len(device_ids):
            raise ValueError("chip bootstrap config length must match device_ids")
        return configs

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

        try:
            if self.level == 2:
                self._init_level2()
            elif self.level == 3:
                self._init_level3()
            else:
                raise ValueError(f"Worker: level {self.level} not yet supported")
        except Exception:
            if self.level == 3:
                self._cleanup_level3_resources()
            raise

        self._initialized = True

    def _wait_for_pid_exit(self, pid: int, timeout_s: float = 2.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            waited_pid, _ = os.waitpid(pid, os.WNOHANG)
            if waited_pid == pid:
                return
            time.sleep(0.05)

        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        os.waitpid(pid, 0)

    def _cleanup_level3_resources(self) -> None:
        if self._dist_worker:
            self._dist_worker.close()
            self._dist_worker = None

        for sw in self._dist_sub_workers:
            sw.shutdown()
        for shm in self._subworker_shms:
            buf = shm.buf
            if buf is not None:
                struct.pack_into("i", buf, _OFF_STATE, _SHUTDOWN)
        for pid in self._subworker_pids:
            self._wait_for_pid_exit(pid)
        for shm in self._subworker_shms:
            shm.close()
            shm.unlink()

        for cp in self._dist_chip_procs:
            cp.shutdown()
        for shm in self._chip_shms:
            buf = shm.buf
            if buf is not None:
                struct.pack_into("i", buf, _CHIP_OFF_STATE, _SHUTDOWN)
        for pid in self._chip_pids:
            self._wait_for_pid_exit(pid)
        for shm in self._chip_shms:
            shm.close()
            shm.unlink()
        for shm in self._chip_bootstrap_shms:
            shm.close()
            shm.unlink()

        self._subworker_shms.clear()
        self._subworker_pids.clear()
        self._chip_bootstrap_shms.clear()
        self._chip_shms.clear()
        self._chip_pids.clear()
        self._dist_sub_workers.clear()
        self._dist_chip_procs.clear()
        self._chip_contexts.clear()

    def _init_level2(self) -> None:
        device_id = self._config.get("device_id", 0)
        host_lib_path, aicpu_path, aicore_path, sim_ctx_path = self._resolve_runtime_binaries()

        self._chip_worker = ChipWorker()
        self._chip_worker.init(
            host_lib_path,
            aicpu_path,
            aicore_path,
            sim_ctx_path,
        )
        self._chip_worker.set_device(device_id)

    def _resolve_runtime_binaries(self) -> tuple[str, str, str, str]:
        explicit = (
            self._config.get("host_path"),
            self._config.get("aicpu_path"),
            self._config.get("aicore_path"),
        )
        if all(explicit):
            return (
                str(explicit[0]),
                str(explicit[1]),
                str(explicit[2]),
                str(self._config.get("sim_context_path", "")),
            )

        from runtime_builder import RuntimeBuilder  # noqa: PLC0415

        platform = self._config["platform"]
        runtime = self._config["runtime"]
        builder = RuntimeBuilder(platform)
        binaries = builder.get_binaries(runtime, build=False)
        return (
            str(binaries.host_path),
            str(binaries.aicpu_path),
            str(binaries.aicore_path),
            str(binaries.sim_context_path) if hasattr(binaries, "sim_context_path") else "",
        )

    def _init_level3(self) -> None:
        device_ids = self._config.get("device_ids", [])
        n_sub = self._config.get("num_sub_workers", 0)
        chip_bootstrap_configs = self._resolve_chip_bootstrap_configs(device_ids)

        # 1. Allocate mailboxes
        for _ in range(n_sub):
            shm = SharedMemory(create=True, size=DIST_SUB_MAILBOX_SIZE)
            assert shm.buf is not None
            struct.pack_into("i", shm.buf, _OFF_STATE, _IDLE)
            self._subworker_shms.append(shm)

        # 2. Fork SubWorker processes (MUST be before any C++ threads)
        registry = self._callable_registry
        for i in range(n_sub):
            pid = os.fork()
            if pid == 0:
                buf = self._subworker_shms[i].buf
                assert buf is not None
                _sub_worker_loop(buf, registry)
                os._exit(0)
            else:
                self._subworker_pids.append(pid)

        # 3. Fork ChipWorker processes (only if device_ids provided)
        if device_ids:
            from task_interface import CHIP_STORAGE_TASK_ARGS_SIZE  # noqa: PLC0415

            # Mailbox transport memcpy's a fixed-size ChipStorageTaskArgs blob.
            # Use the binding-exported C++ sizeof(...) instead of inferring from
            # Python object addresses, which is not layout-safe.
            args_size = int(CHIP_STORAGE_TASK_ARGS_SIZE)

            host_lib_path, aicpu_path, aicore_path, sim_ctx_path = self._resolve_runtime_binaries()
            pending_bootstrap_channels: list[tuple[DistChipBootstrapChannel, int, int]] = []

            for chip_index, dev_id in enumerate(device_ids):
                shm = SharedMemory(create=True, size=DIST_CHIP_MAILBOX_SIZE)
                assert shm.buf is not None
                struct.pack_into("i", shm.buf, _CHIP_OFF_STATE, _IDLE)
                self._chip_shms.append(shm)

                chip_bootstrap_config = None
                bootstrap_mailbox_ptr = None
                bootstrap_buffer_count = 0
                bootstrap_channel = None
                if chip_bootstrap_configs is not None:
                    chip_bootstrap_config = chip_bootstrap_configs[chip_index]
                    bootstrap_shm = SharedMemory(create=True, size=DIST_CHIP_BOOTSTRAP_MAILBOX_SIZE)
                    self._chip_bootstrap_shms.append(bootstrap_shm)
                    bootstrap_channel = DistChipBootstrapChannel(
                        _mailbox_addr(bootstrap_shm), len(chip_bootstrap_config.buffers)
                    )
                    bootstrap_channel.reset()
                    bootstrap_mailbox_ptr = _mailbox_addr(bootstrap_shm)
                    bootstrap_buffer_count = len(chip_bootstrap_config.buffers)

                pid = os.fork()
                if pid == 0:
                    buf = shm.buf
                    assert buf is not None
                    _chip_process_loop(
                        buf,
                        host_lib_path,
                        dev_id,
                        aicpu_path,
                        aicore_path,
                        sim_ctx_path,
                        args_size,
                        bootstrap_mailbox_ptr,
                        bootstrap_buffer_count,
                        chip_bootstrap_config,
                    )
                    os._exit(0)
                else:
                    self._chip_pids.append(pid)
                    if bootstrap_channel is not None:
                        pending_bootstrap_channels.append((bootstrap_channel, chip_index, dev_id))

            for bootstrap_channel, chip_index, dev_id in pending_bootstrap_channels:
                deadline = time.monotonic() + 30.0
                while bootstrap_channel.state == ChipBootstrapMailboxState.IDLE and time.monotonic() < deadline:
                    time.sleep(0.01)
                if bootstrap_channel.state == ChipBootstrapMailboxState.ERROR:
                    raise RuntimeError(
                        f"chip bootstrap failed on device {dev_id}: {bootstrap_channel.error_message}"
                    )
                if bootstrap_channel.state != ChipBootstrapMailboxState.SUCCESS:
                    raise RuntimeError(f"chip bootstrap timed out on device {dev_id}")
                reply = ChipBootstrapReply(
                    device_ctx=int(bootstrap_channel.device_ctx),
                    local_window_base=int(bootstrap_channel.local_window_base),
                    actual_window_size=int(bootstrap_channel.actual_window_size),
                    buffer_ptrs=[int(ptr) for ptr in bootstrap_channel.buffer_ptrs],
                )
                if chip_bootstrap_configs is not None:
                    self._chip_contexts.append(_enrich_chip_context(chip_bootstrap_configs[chip_index], dev_id, reply))

        # 4. Create DistWorker and wire chip processes + sub workers
        dw = DistWorker(3)
        self._dist_worker = dw

        if device_ids:
            for shm in self._chip_shms:
                cp = DistChipProcess(_mailbox_addr(shm), args_size)
                self._dist_chip_procs.append(cp)
                dw.add_chip_process(cp)

        for shm in self._subworker_shms:
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
        in_specs = []
        for inp in inputs or []:
            if isinstance(inp, tuple):
                in_specs.append(DistInputSpec(inp[1], inp[0]))
            else:
                in_specs.append(DistInputSpec(inp))

        out_specs = []
        for out in outputs or []:
            if isinstance(out, dict):
                ptr = int(out["ptr"])
                size = int(out.get("size", 0))
                worker_index = int(out.get("worker_index", -1))
                out_specs.append(DistOutputSpec.external(ptr, size, worker_index))
            else:
                out_specs.append(DistOutputSpec(out))
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
        if (
            not self._initialized
            and not self._chip_pids
            and not self._subworker_pids
            and not self._chip_shms
            and not self._subworker_shms
        ):
            return

        if self.level == 2:
            if self._chip_worker:
                self._chip_worker.finalize()
        else:
            self._cleanup_level3_resources()

        self._initialized = False

    @property
    def chip_contexts(self) -> list[ChipContext]:
        return list(self._chip_contexts)

    def __enter__(self) -> "Worker":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
