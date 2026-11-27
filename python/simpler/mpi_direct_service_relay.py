# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Loopback TCP streams carried by MPI Direct service ranks."""

from __future__ import annotations

import contextlib
import itertools
import os
import queue
import socket
import threading
from collections.abc import Callable

LOOPBACK = "127.0.0.1"
_CHUNK_BYTES = 64 * 1024
_MAX_STREAMS = 64
_PENDING_CHUNKS = 128


def listen(port: int = 0) -> socket.socket:
    listener = socket.socket()
    try:
        # Allow a new job to bind the same loopback endpoint after the old
        # connections close. Windows gives SO_REUSEADDR different semantics.
        if os.name != "nt":
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((LOOPBACK, port))
        listener.listen()
    except BaseException:
        listener.close()
        raise
    return listener


def close_socket(sock: socket.socket) -> None:
    with contextlib.suppress(OSError):
        sock.shutdown(socket.SHUT_RDWR)
    sock.close()


class _Stream:
    def __init__(self, sock, key, send, events) -> None:
        self.sock = sock
        self.peer, self.stream_id = key
        self.send = send
        self.events = events
        self.inbound = queue.Queue(maxsize=_PENDING_CHUNKS)
        self.closed = threading.Event()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.writer = threading.Thread(target=self._write, daemon=True)
        self.ends = 0

    def start(self) -> None:
        self.reader.start()
        self.writer.start()

    def stop(self) -> None:
        self.closed.set()
        close_socket(self.sock)
        with contextlib.suppress(queue.Full):
            self.inbound.put_nowait(None)

    def offer(self, data: bytes) -> None:
        if not self.closed.is_set():
            try:
                self.inbound.put_nowait(data)
            except queue.Full:
                self.closed.set()
                self.events.put(("relay_reset", self))

    def _read(self) -> None:
        try:
            while not self.closed.is_set():
                data = self.sock.recv(_CHUNK_BYTES)
                if self.closed.is_set():
                    return
                self.send(self.peer, "stream_data", self.stream_id, data)
                if not data:
                    self.events.put(("relay_eof", self))
                    return
        except OSError:
            self.events.put(("relay_reset", self))
        except Exception as exc:
            self.events.put(("error", exc))

    def _write(self) -> None:
        try:
            while not self.closed.is_set():
                data = self.inbound.get()
                if data is None or self.closed.is_set():
                    return
                if not data:
                    self.sock.shutdown(socket.SHUT_WR)
                    self.events.put(("relay_eof", self))
                    return
                self.sock.sendall(data)
        except OSError:
            self.events.put(("relay_reset", self))


class Relay:
    def __init__(self, role, bind_port, peers, events, timeout) -> None:
        self.role = role
        self.bind_port = bind_port
        self.events = events
        self.timeout = timeout
        self.listeners = {}
        self.streams = {}
        self.lock = threading.Lock()
        self.ids = itertools.count(1)
        self.send: Callable[[int, str, int, bytes | None], None]
        try:
            for rank, instance_id in peers.items():
                self.listeners[rank] = (instance_id, listen())
        except BaseException:
            self.close()
            raise

    def ports(self) -> dict[str, int]:
        return {instance_id: listener.getsockname()[1] for instance_id, listener in self.listeners.values()}

    def start(self, send) -> None:
        self.send = send
        for rank, (_instance_id, listener) in self.listeners.items():
            threading.Thread(target=self._accept, args=(rank, listener), daemon=True).start()

    def _accept(self, peer, listener) -> None:
        while True:
            try:
                sock, _ = listener.accept()
            except OSError:
                return
            self.events.put(("relay_accept", peer, sock))

    def _add(self, sock, key) -> _Stream:
        stream = _Stream(sock, key, self.send, self.events)
        with self.lock:
            self.streams[key] = stream
        return stream

    def accept(self, peer: int, sock: socket.socket) -> None:
        if len(self.streams) >= _MAX_STREAMS:
            sock.close()
            return
        stream_id = next(self.ids)
        self._add(sock, (peer, stream_id))
        self.send(peer, "stream_open", stream_id, None)

    def on_open(self, peer: int, stream_id: int) -> None:
        if self.role == "decode" and len(self.streams) < _MAX_STREAMS:
            try:
                sock = socket.create_connection((LOOPBACK, self.bind_port), timeout=self.timeout)
            except OSError:
                pass
            else:
                sock.settimeout(None)
                stream = self._add(sock, (peer, stream_id))
                self.send(peer, "stream_opened", stream_id, None)
                stream.start()
                return
        self.send(peer, "stream_reset", stream_id, None)

    def on_opened(self, peer: int, stream_id: int) -> None:
        stream = self.streams.get((peer, stream_id))
        if stream is not None:
            stream.start()

    def deliver(self, peer: int, stream_id: int, data: bytes) -> None:
        with self.lock:
            stream = self.streams.get((peer, stream_id))
        if stream is not None:
            stream.offer(data)

    def on_reset(self, peer: int, stream_id: int) -> None:
        with self.lock:
            stream = self.streams.pop((peer, stream_id), None)
        if stream is not None:
            stream.stop()
            if stream.reader.ident is not None:
                stream.reader.join(timeout=self.timeout)
                if stream.reader.is_alive():
                    raise TimeoutError("MPI Direct relay reader did not stop")

    def finish(self, stream: _Stream, reset: bool) -> None:
        key = stream.peer, stream.stream_id
        if self.streams.get(key) is not stream:
            return
        stream.ends += 1
        if reset or stream.ends == 2:
            self.on_reset(*key)
            if reset:
                self.send(stream.peer, "stream_reset", stream.stream_id, None)

    def close(self) -> None:
        for _instance_id, listener in self.listeners.values():
            close_socket(listener)
        for key in tuple(self.streams):
            self.on_reset(*key)
