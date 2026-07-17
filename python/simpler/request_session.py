# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Non-blocking request submission for a hierarchical Worker."""

from __future__ import annotations

import queue
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

HOST_GRAPH_TOKEN_STREAM_MAGIC_VERSION = 0x4847535400010000
HOST_GRAPH_TOKEN_STREAM_TRAILER_SCALARS = 14
HOST_GRAPH_TOKEN_FINAL = 1 << 0
HOST_GRAPH_TOKEN_SYNTHETIC = 1 << 1
HOST_GRAPH_PREPARED_REQUEST_MAGIC = 0x4850525100010000
_HOST_GRAPH_TOKEN_PACKET = struct.Struct("<QQQqIi")
_STOP = object()

HOST_REQUEST_CONTROL_OFFSET = 4096
HOST_REQUEST_CONTROL_BYTES = 60 * 1024
_HOST_REQUEST_STATE_IDLE = 0
_HOST_REQUEST_STATE_READY = 1
_HOST_REQUEST_STATE_RUNNING = 2
_HOST_REQUEST_STATE_DONE = 3
_HOST_REQUEST_CMD_PREPARE = 1
_HOST_REQUEST_HEADER = struct.Struct("<IIQQiIIII")
_HOST_REQUEST_ERROR_OFFSET = 64
_HOST_REQUEST_ERROR_BYTES = 256
_HOST_REQUEST_DIGEST_OFFSET = _HOST_REQUEST_ERROR_OFFSET + _HOST_REQUEST_ERROR_BYTES
_HOST_REQUEST_DIGEST_BYTES = 32
_HOST_REQUEST_PAYLOAD_OFFSET = _HOST_REQUEST_DIGEST_OFFSET + _HOST_REQUEST_DIGEST_BYTES


class HostRequestAdmissionClient:
    """Independent L3-to-L2 Host control lane inside the comm bootstrap shm."""

    def __init__(self, shm: Any) -> None:
        from _task_interface import _mailbox_load_i32, _mailbox_store_i32  # noqa: PLC0415

        self._load_i32 = _mailbox_load_i32
        self._store_i32 = _mailbox_store_i32
        self._shm = shm
        self._buf = shm.buf
        if self._buf is None or len(self._buf) < HOST_REQUEST_CONTROL_OFFSET + HOST_REQUEST_CONTROL_BYTES:
            raise RuntimeError("Host request admission control shm is too small")
        import ctypes  # noqa: PLC0415

        self._base_addr = ctypes.addressof(ctypes.c_char.from_buffer(self._buf)) + HOST_REQUEST_CONTROL_OFFSET
        self._lock = threading.Lock()
        self._store_i32(self._base_addr, _HOST_REQUEST_STATE_IDLE)

    def prepare(
        self,
        request_id: int,
        digest: bytes,
        args: Any,
        config: Any,
        *,
        arena_bank: int,
        timeout: float,
    ) -> None:
        request_id = int(request_id)
        digest = bytes(digest)
        arena_bank = int(arena_bank)
        if len(digest) != _HOST_REQUEST_DIGEST_BYTES:
            raise ValueError("Host request callable digest must be 32 bytes")
        if arena_bank not in (0, 1):
            raise ValueError("Host request arena bank must be 0 or 1")

        config_size = int(config._blob_size)
        config_offset = _HOST_REQUEST_PAYLOAD_OFFSET
        args_offset = (config_offset + config_size + 7) & ~7
        args_capacity = HOST_REQUEST_CONTROL_BYTES - args_offset
        if args_capacity <= 0:
            raise RuntimeError("Host request control layout has no TaskArgs capacity")

        deadline = time.monotonic() + float(timeout)
        with self._lock:
            while self._load_i32(self._base_addr) != _HOST_REQUEST_STATE_IDLE:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Host request admission lane did not become idle")
                time.sleep(0.00005)

            base = self._base_addr
            config._write_blob(base + config_offset, config_size)
            blob_size = int(args._write_blob(base + args_offset, args_capacity))
            digest_start = HOST_REQUEST_CONTROL_OFFSET + _HOST_REQUEST_DIGEST_OFFSET
            self._buf[digest_start : digest_start + len(digest)] = digest
            error_start = HOST_REQUEST_CONTROL_OFFSET + _HOST_REQUEST_ERROR_OFFSET
            self._buf[error_start : error_start + _HOST_REQUEST_ERROR_BYTES] = b"\x00" * _HOST_REQUEST_ERROR_BYTES
            _HOST_REQUEST_HEADER.pack_into(
                self._buf,
                HOST_REQUEST_CONTROL_OFFSET,
                _HOST_REQUEST_STATE_IDLE,
                _HOST_REQUEST_CMD_PREPARE,
                request_id,
                0,
                0,
                arena_bank,
                config_size,
                blob_size,
                0,
            )
            self._store_i32(self._base_addr, _HOST_REQUEST_STATE_READY)
            while self._load_i32(self._base_addr) != _HOST_REQUEST_STATE_DONE:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Host request prepare timed out")
                time.sleep(0.00005)
            fields = _HOST_REQUEST_HEADER.unpack_from(self._buf, HOST_REQUEST_CONTROL_OFFSET)
            _state, _cmd, _request_id, response_id, status = fields[:5]
            raw_error = bytes(self._buf[error_start : error_start + _HOST_REQUEST_ERROR_BYTES])
            message = raw_error.split(b"\x00", 1)[0].decode("utf-8", "replace")
            self._store_i32(self._base_addr, _HOST_REQUEST_STATE_IDLE)
            if status != 0 or int(response_id) != request_id:
                raise RuntimeError(message or f"Host request prepare failed with status {status}")


@dataclass(frozen=True)
class HostGraphToken:
    request_id: int
    token_seq: int
    token_id: int
    is_final: bool
    status: int
    synthetic: bool


def append_host_graph_token_stream_args(task_args: Any, queue_handle: Any, request_id: int) -> None:
    """Append the optional HostGraph token-stream trailer to L2 task args."""
    request_id = int(request_id)
    if request_id < 0 or request_id > (1 << 64) - 1:
        raise ValueError("HostGraph request_id must fit uint64")
    queue_scalars = queue_handle.l2_task_arg_scalars()
    if len(queue_scalars) + 2 != HOST_GRAPH_TOKEN_STREAM_TRAILER_SCALARS:
        raise RuntimeError("HostGraph token-stream trailer layout has changed")
    task_args.add_scalar(HOST_GRAPH_TOKEN_STREAM_MAGIC_VERSION)
    task_args.add_scalar(request_id)
    for scalar in queue_scalars:
        task_args.add_scalar(int(scalar))


def append_host_graph_prepared_request_args(task_args: Any, request_id: int) -> None:
    """Mark args so the chip mailbox executes already-prepared request state."""
    task_args.add_scalar(HOST_GRAPH_PREPARED_REQUEST_MAGIC)
    task_args.add_scalar(int(request_id))


def decode_host_graph_token(payload: bytes | bytearray | memoryview) -> HostGraphToken:
    """Decode one HostGraph token packet read from an L3-L2 queue."""
    view = memoryview(payload)
    if view.nbytes != _HOST_GRAPH_TOKEN_PACKET.size:
        raise ValueError(f"HostGraph token packet has {view.nbytes} bytes, expected {_HOST_GRAPH_TOKEN_PACKET.size}")
    magic, request_id, token_seq, token_id, flags, status = _HOST_GRAPH_TOKEN_PACKET.unpack(view)
    if magic != HOST_GRAPH_TOKEN_STREAM_MAGIC_VERSION:
        raise ValueError(f"HostGraph token packet has unsupported magic/version 0x{magic:016x}")
    return HostGraphToken(
        request_id=request_id,
        token_seq=token_seq,
        token_id=token_id,
        is_final=bool(flags & HOST_GRAPH_TOKEN_FINAL),
        status=status,
        synthetic=bool(flags & HOST_GRAPH_TOKEN_SYNTHETIC),
    )


@dataclass(frozen=True)
class _Terminal:
    error: BaseException | None


@dataclass(frozen=True)
class _StreamItem:
    value: Any
    consumed: threading.Event


class RequestBackpressureError(RuntimeError):
    """Raised when the session's bounded pending queue is full."""


class RequestCancelledError(RuntimeError):
    """Raised by a stream whose request was cancelled."""


class RequestStream:
    """Thread-safe stream returned immediately by ``RequestSession.submit``."""

    def __init__(self, request_id: int, session: RequestSession) -> None:
        self.request_id = int(request_id)
        self._session = session
        self._items: queue.Queue[Any] = queue.Queue()
        self._done = threading.Event()
        self._terminal_seen = False
        self._terminal_error: BaseException | None = None
        self._finish_lock = threading.Lock()
        self._pending_consumed: threading.Event | None = None

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def next(self, timeout: float | None = None) -> Any:
        if self._terminal_seen:
            if self._terminal_error is not None:
                raise self._terminal_error
            raise StopIteration
        try:
            item = self._items.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError(f"request {self.request_id} stream timed out") from exc
        if isinstance(item, _Terminal):
            self._terminal_seen = True
            self._terminal_error = item.error
            if item.error is not None:
                raise item.error
            raise StopIteration
        assert isinstance(item, _StreamItem)
        item.consumed.set()
        return item.value

    def wait(self, timeout: float | None = None) -> None:
        if not self._done.wait(timeout):
            raise TimeoutError(f"request {self.request_id} did not finish")
        if self._terminal_error is not None:
            raise self._terminal_error

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def cancel(self) -> bool:
        """Request cancellation; active device work is drained but no longer emitted."""
        return self._session.cancel(self.request_id)

    def _emit(self, item: Any, *, final: bool = False) -> None:
        consumed = threading.Event()
        with self._finish_lock:
            if self._done.is_set():
                if isinstance(self._terminal_error, RequestCancelledError):
                    return
                raise RuntimeError(f"request {self.request_id} stream is already complete")
            if self._pending_consumed is not None:
                raise RuntimeError(f"request {self.request_id} already has an outstanding item")
            self._pending_consumed = consumed
            self._items.put(_StreamItem(item, consumed))
        consumed.wait()
        with self._finish_lock:
            if self._pending_consumed is consumed:
                self._pending_consumed = None
            if self._done.is_set():
                if isinstance(self._terminal_error, RequestCancelledError):
                    return
                if self._terminal_error is not None:
                    raise self._terminal_error
        if final:
            self._finish(None)

    def _finish(self, error: BaseException | None) -> None:
        with self._finish_lock:
            if self._done.is_set():
                return
            self._terminal_error = error
            self._items.put(_Terminal(error))
            self._done.set()
            if self._pending_consumed is not None:
                self._pending_consumed.set()


class RequestEmitter:
    """Producer handle passed to the session orchestration callback."""

    def __init__(self, stream: RequestStream, session: RequestSession | None = None) -> None:
        self._stream = stream
        self._session = session

    def emit(self, item: Any, *, final: bool = False) -> None:
        self._stream._emit(item, final=final)

    def prepare_host_request(
        self,
        worker_id: int,
        request_id: int,
        callable_handle: Any,
        args: Any,
        config: Any,
        *,
        arena_bank: int = 1,
        timeout: float = 30.0,
    ) -> None:
        if self._session is None:
            raise RuntimeError("RequestEmitter is not attached to a RequestSession")
        if int(request_id) != self._stream.request_id:
            raise ValueError("prepared request_id must match the emitter's stream")
        self._session._prepare_host_request(
            worker_id, request_id, callable_handle, args, config, arena_bank=arena_bank, timeout=timeout
        )


@dataclass
class _RequestWork:
    request: Any
    request_id: int
    config: Any
    stream: RequestStream
    cancelled: threading.Event


RequestOrchestration = Callable[[Any, Any, int, RequestEmitter, Any], None]


class RequestSession:
    """Dispatch bounded concurrent ``Worker.run`` calls with per-request streams."""

    def __init__(
        self,
        worker: Any,
        orchestration: RequestOrchestration,
        *,
        max_pending: int = 8,
        max_active_runs: int = 1,
    ) -> None:
        if not callable(orchestration):
            raise TypeError("RequestSession orchestration must be callable")
        if int(max_pending) <= 0:
            raise ValueError("RequestSession max_pending must be positive")
        if int(max_active_runs) <= 0:
            raise ValueError("RequestSession max_active_runs must be positive")
        self._worker = worker
        self._orchestration = orchestration
        self._requests: queue.Queue[_RequestWork | object] = queue.Queue(maxsize=int(max_pending))
        self._state_lock = threading.Lock()
        self._accepting = True
        self._next_request_id = 1
        self._live_work: dict[int, _RequestWork] = {}
        self._dispatcher_idents: set[int] = set()
        self._threads = [
            threading.Thread(
                target=self._dispatch_loop,
                name=f"simpler-request-session-{index}",
                daemon=True,
            )
            for index in range(int(max_active_runs))
        ]

    def _start(self) -> None:
        for thread in self._threads:
            thread.start()

    def submit(self, request: Any, *, config: Any = None, request_id: int | None = None) -> RequestStream:
        with self._state_lock:
            if not self._accepting:
                raise RuntimeError("RequestSession is closed")
            if request_id is None:
                while self._next_request_id in self._live_work:
                    self._next_request_id += 1
                request_id = self._next_request_id
                self._next_request_id += 1
            request_id = int(request_id)
            if request_id < 0 or request_id > (1 << 64) - 1:
                raise ValueError("request_id must fit uint64")
            if request_id in self._live_work:
                raise ValueError(f"request_id {request_id} is already live in this session")
            stream = RequestStream(request_id, self)
            work = _RequestWork(request, request_id, config, stream, threading.Event())
            try:
                self._requests.put_nowait(work)
            except queue.Full as exc:
                raise RequestBackpressureError(
                    f"RequestSession pending queue is full (request_id={request_id})"
                ) from exc
            self._live_work[request_id] = work
            return stream

    def cancel(self, request_id: int) -> bool:
        request_id = int(request_id)
        with self._state_lock:
            work = self._live_work.get(request_id)
            if work is None or work.cancelled.is_set() or work.stream.done:
                return False
            work.cancelled.set()
        work.stream._finish(RequestCancelledError(f"request {request_id} was cancelled"))
        return True

    def close(self) -> None:
        should_stop = False
        with self._state_lock:
            if self._accepting:
                self._accepting = False
                should_stop = True
        if should_stop:
            for _thread in self._threads:
                self._requests.put(_STOP)
        current_ident = threading.get_ident()
        for thread in self._threads:
            if thread.ident != current_ident:
                thread.join()
        self._worker._release_request_session(self)

    def _owns_current_thread(self) -> bool:
        with self._state_lock:
            return threading.get_ident() in self._dispatcher_idents

    def _retire(self, work: _RequestWork) -> None:
        with self._state_lock:
            if self._live_work.get(work.request_id) is work:
                self._live_work.pop(work.request_id, None)

    def _prepare_host_request(
        self,
        worker_id: int,
        request_id: int,
        callable_handle: Any,
        args: Any,
        config: Any,
        *,
        arena_bank: int,
        timeout: float,
    ) -> None:
        self._worker._prepare_host_request(
            int(worker_id), int(request_id), callable_handle, args, config, int(arena_bank), float(timeout)
        )

    def _dispatch_loop(self) -> None:
        ident = threading.get_ident()
        with self._state_lock:
            self._dispatcher_idents.add(ident)
        try:
            while True:
                work = self._requests.get()
                if work is _STOP:
                    return
                assert isinstance(work, _RequestWork)
                if work.cancelled.is_set():
                    self._retire(work)
                    continue
                emitter = RequestEmitter(work.stream, self)

                def task_orch(orch, _args, cfg, _work=work, _emitter=emitter):
                    self._orchestration(orch, _work.request, _work.request_id, _emitter, cfg)

                try:
                    self._worker.run(task_orch, args=None, config=work.config)
                    work.stream._finish(None)
                except BaseException as exc:  # noqa: BLE001
                    work.stream._finish(exc)
                finally:
                    self._retire(work)
        finally:
            with self._state_lock:
                self._dispatcher_idents.discard(ident)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
