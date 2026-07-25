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
from dataclasses import dataclass
from typing import Any, Callable

HOST_GRAPH_TOKEN_STREAM_MAGIC_VERSION = 0x4847535400010000
HOST_GRAPH_TOKEN_STREAM_TRAILER_SCALARS = 14
HOST_GRAPH_TOKEN_FINAL = 1 << 0
HOST_GRAPH_TOKEN_SYNTHETIC = 1 << 1
_HOST_GRAPH_TOKEN_PACKET = struct.Struct("<QQQqIi")
_STOP = object()


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
        self._close_lock = threading.Lock()
        self._closed = threading.Event()
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
        with self._close_lock:
            if self._closed.is_set():
                return
            with self._state_lock:
                self._accepting = False
                live_work = list(self._live_work.values())
                for work in live_work:
                    work.cancelled.set()

            for work in live_work:
                work.stream._finish(RequestCancelledError(f"request {work.request_id} was cancelled by session close"))
            for _thread in self._threads:
                self._requests.put(_STOP)
            current_ident = threading.get_ident()
            for thread in self._threads:
                if thread.ident != current_ident:
                    thread.join()
            self._worker._release_request_session(self)
            self._closed.set()

    def _owns_current_thread(self) -> bool:
        with self._state_lock:
            return threading.get_ident() in self._dispatcher_idents

    def _retire(self, work: _RequestWork) -> None:
        with self._state_lock:
            if self._live_work.get(work.request_id) is work:
                self._live_work.pop(work.request_id, None)

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
