# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cooperative, runtime-managed access to chip buffers by trusted transports."""

from __future__ import annotations

import base64
import ctypes
import json
import math
import multiprocessing
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any

from _task_interface import _mailbox_load_i32, _mailbox_store_i32

from .buffer import AccessMode, Buffer

_RESULT_BYTES = 4096
_HEADER = struct.Struct("<IIII")  # cancellation, outcome, payload length, error length
_SUCCEEDED, _FAILED, _UNCONFIRMED = 1, 2, 3
_providers: dict[str, ExternalTransferProvider] = {}
_providers_lock = threading.Lock()


@dataclass(frozen=True)
class ExternalBufferRange:
    """An allocation identity plus a byte range and required device access."""

    buffer: Buffer
    offset: int
    nbytes: int
    access: AccessMode

    def __post_init__(self) -> None:
        if not isinstance(self.buffer, Buffer):
            raise TypeError("external transfer range requires a Buffer, not a device pointer")
        if type(self.offset) is not int or self.offset < 0:
            raise ValueError("external transfer offset must be a non-negative integer")
        if type(self.nbytes) is not int or self.nbytes <= 0:
            raise ValueError("external transfer nbytes must be a positive integer")
        if self.access not in (AccessMode.READ, AccessMode.WRITE, AccessMode.READWRITE):
            raise ValueError("external transfer access must be READ, WRITE, or READWRITE")


@dataclass(frozen=True)
class ExternalTransferSpan:
    """A checked chip-local address; valid only until the provider returns."""

    address: int
    nbytes: int
    access: AccessMode


@dataclass(frozen=True)
class ExternalTransferRequest:
    """Provider input, without allocation, stream, or ChipWorker capabilities."""

    device_id: int
    buffers: tuple[ExternalTransferSpan, ...]
    payload: bytes


@dataclass(frozen=True)
class ExternalTransferResult:
    """Returning this result certifies that all access to the supplied spans has stopped.

    Failure is also a quiescence certificate. An exception is NOT one: it leaves
    the operation retained and prevents normal Worker teardown.
    """

    succeeded: bool = True
    payload: bytes = b""
    error: str = ""


class ExternalTransferCancellation:
    """Cooperative cancellation request; observing it does not complete a transfer."""

    def __init__(self, address: int):
        self._address = address

    @property
    def requested(self) -> bool:
        return bool(_mailbox_load_i32(self._address))


ExternalTransferProvider = Callable[[ExternalTransferRequest, ExternalTransferCancellation], ExternalTransferResult]


def register_external_transfer_provider(name: str, provider: ExternalTransferProvider) -> None:
    """Register a trusted provider before Worker.init() snapshots the registry.

    Runtime binds the chip-child background thread to its device before calling
    the provider. Providers must not reset the device or change runtime-owned
    contexts, streams, allocations, or allocator state.
    """
    if not isinstance(name, str) or not name or len(name.encode("utf-8")) > 255:
        raise ValueError("external transfer provider name must contain 1..255 UTF-8 bytes")
    if not callable(provider):
        raise TypeError("external transfer provider must be callable")
    with _providers_lock:
        if name in _providers and _providers[name] is not provider:
            raise ValueError(f"external transfer provider {name!r} is already registered")
        _providers[name] = provider


def _snapshot_providers() -> dict[str, ExternalTransferProvider]:
    with _providers_lock:
        return dict(_providers)


class ExternalTransferSubmissionError(RuntimeError):
    """Submission has an uncertain outcome; handle retains the destination resources."""

    def __init__(self, handle: ExternalTransferHandle):
        super().__init__("external transfer submission was not acknowledged; resources remain retained")
        self.handle = handle


class ExternalTransferUnconfirmedError(RuntimeError):
    """The provider did not certify quiescence; resources must not be reclaimed."""


class ExternalTransferHandle:
    """A retained external operation. Timeouts never release its buffers.

    Observe completion with done()/wait()/result(); Worker.close() also drains
    handles. request_cancel() only sets a cooperative request flag.
    """

    def __init__(self, pool: _ExternalTransferPool, slot: int, buffers: tuple, worker_id: int):
        self._pool = pool
        self._slot = slot
        self._buffers = buffers
        self._worker_id = worker_id
        self._wait_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._outcome: tuple[int, bytes, str] | None = None
        self._retired = False
        self._shm = SharedMemory(create=True, size=_HEADER.size + _RESULT_BYTES)
        self._shm.buf[:] = bytes(self._shm.size)

    @property
    def nbytes(self) -> int:
        return sum(span.nbytes for span in self._buffers)

    def request_cancel(self) -> bool:
        """Request cancellation, returning False if an outcome has already been observed."""
        with self._state_lock:
            if self._outcome is not None:
                return False
            address = ctypes.addressof(ctypes.c_char.from_buffer(self._shm.buf))
            _mailbox_store_i32(address, 1)
            return True

    def done(self) -> bool:
        """Return whether quiescence is confirmed; also retire a completed operation."""
        try:
            self.wait(0)
        except TimeoutError:
            return False
        except ExternalTransferUnconfirmedError:
            return False
        except RuntimeError:
            return self._retired
        return True

    def wait(self, timeout: float | None = None) -> bytes:
        """Wait for quiescence and return the payload, or raise on failure/timeout."""
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("external transfer timeout must be non-negative and finite, or None")
        deadline = None if timeout is None else time.monotonic() + timeout
        acquired = self._wait_lock.acquire() if deadline is None else self._wait_lock.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError("external transfer is still in flight")
        try:
            if self._outcome is None:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if not self._pool.signals[self._slot].wait(timeout=remaining):
                    raise TimeoutError("external transfer is still in flight; buffers remain retained")
                self._observe_outcome()
            assert self._outcome is not None
            code, payload, error = self._outcome
            if code == _UNCONFIRMED:
                raise ExternalTransferUnconfirmedError(error)
            self._retire_resources()
            if code == _FAILED:
                raise RuntimeError(error or "external transfer failed")
            return payload
        finally:
            self._wait_lock.release()

    def result(self, timeout: float | None = None) -> bytes:
        """Alias for wait()."""
        return self.wait(timeout)

    def _observe_outcome(self) -> None:
        with self._state_lock:
            code, payload_size, error_size = struct.unpack_from("<III", self._shm.buf, 4)
            if code not in (_SUCCEEDED, _FAILED) or payload_size + error_size > _RESULT_BYTES:
                code = _UNCONFIRMED
            raw = bytes(self._shm.buf[_HEADER.size : _HEADER.size + min(payload_size + error_size, _RESULT_BYTES)])
            self._outcome = (code, raw[:payload_size], raw[payload_size:].decode("utf-8", "replace"))

    def _retire_resources(self) -> None:
        if self._retired:
            return
        with self._state_lock:
            self._shm.close()
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
        self._pool.retire(self)
        self._retired = True


class _ExternalTransferPool:
    def __init__(self, count: int, max_bytes: int, retire: Callable[[ExternalTransferHandle], None]):
        self.signals = tuple(multiprocessing.get_context("fork").Event() for _ in range(count))
        self.max_bytes = max_bytes
        self.pending: dict[int, ExternalTransferHandle] = {}
        self.retire = retire

    def reserve(self, spans: tuple[ExternalTransferSpan, ...], worker_id: int) -> ExternalTransferHandle:
        if len(self.pending) >= len(self.signals):
            raise RuntimeError("external transfer pending-count limit exceeded")
        if sum(h.nbytes for h in self.pending.values()) + sum(s.nbytes for s in spans) > self.max_bytes:
            raise RuntimeError("external transfer pending-byte limit exceeded")
        existing = [span for h in self.pending.values() if h._worker_id == worker_id for span in h._buffers]
        last_end = last_write_end = 0
        for span in sorted((*existing, *spans), key=lambda span: span.address):
            if span.address < last_write_end or (span.access != AccessMode.READ and span.address < last_end):
                raise RuntimeError("external transfer conflicts with another buffer access")
            last_end = max(last_end, span.address + span.nbytes)
            if span.access != AccessMode.READ:
                last_write_end = max(last_write_end, span.address + span.nbytes)
        slot = next(slot for slot in range(len(self.signals)) if slot not in self.pending)
        self.signals[slot].clear()
        handle = ExternalTransferHandle(self, slot, spans, worker_id)
        self.pending[slot] = handle
        return handle


def _start_chip_transfer(
    payload: bytes,
    device_id: int,
    providers: dict,
    signals: tuple,
    bind_thread: Callable[[], None],
) -> None:
    command = json.loads(payload)
    provider = providers[command["provider"]]
    signal = signals[command["slot"]]
    request = ExternalTransferRequest(
        device_id,
        tuple(
            ExternalTransferSpan(address, nbytes, AccessMode(access)) for address, nbytes, access in command["spans"]
        ),
        base64.b64decode(command["payload"], validate=True),
    )
    shm = SharedMemory(name=command["completion"])
    if shm.size != _HEADER.size + _RESULT_BYTES:
        shm.close()
        raise ValueError("external transfer completion backing has an invalid size")
    try:
        threading.Thread(
            target=_run_chip_transfer,
            args=(provider, request, shm, signal, bind_thread),
            daemon=True,
            name=f"external-transfer-{command['slot']}",
        ).start()
    except BaseException:
        shm.close()
        raise


def _run_chip_transfer(
    provider: ExternalTransferProvider,
    request: ExternalTransferRequest,
    shm: SharedMemory,
    signal: Any,
    bind_thread: Callable[[], None],
) -> None:
    code, payload, error = _UNCONFIRMED, b"", b""
    try:
        bind_thread()
        token = ExternalTransferCancellation(ctypes.addressof(ctypes.c_char.from_buffer(shm.buf)))
        result = provider(request, token)
        if not isinstance(result, ExternalTransferResult) or type(result.succeeded) is not bool:
            raise TypeError("external transfer provider must return ExternalTransferResult")
        if not isinstance(result.payload, bytes) or not isinstance(result.error, str):
            raise TypeError("external transfer result requires a bytes payload and a str error")
        payload, error = result.payload, result.error.encode("utf-8")
        if len(payload) + len(error) > _RESULT_BYTES:
            payload, error = b"", b"external transfer result exceeds the 4096-byte limit"
            code = _FAILED
        else:
            code = _SUCCEEDED if result.succeeded else _FAILED
    except BaseException as exc:
        payload, error = b"", f"provider quiescence unconfirmed: {type(exc).__name__}: {exc}".encode()[:_RESULT_BYTES]
    shm.buf[_HEADER.size : _HEADER.size + len(payload) + len(error)] = payload + error
    # The cancellation word is independently accessed through native atomics.
    struct.pack_into("<III", shm.buf, 4, code, len(payload), len(error))
    shm.close()
    signal.set()
