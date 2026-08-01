# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Named shared-memory protocol used between L4 and an MPI L3 group."""

from __future__ import annotations

import ctypes
import enum
import json
import struct
from dataclasses import asdict, dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any

MAILBOX_MAGIC = b"SMPIBOX\0"
MAILBOX_PROTOCOL_VERSION = 1
MAILBOX_HEADER_BYTES = 256
MAILBOX_PAYLOAD_BYTES = 16 * 1024 * 1024
MAILBOX_ERROR_BYTES = 64 * 1024
MAILBOX_REQUEST_OFFSET = MAILBOX_HEADER_BYTES
MAILBOX_RESPONSE_OFFSET = MAILBOX_REQUEST_OFFSET + MAILBOX_PAYLOAD_BYTES
MAILBOX_ERROR_OFFSET = MAILBOX_RESPONSE_OFFSET + MAILBOX_PAYLOAD_BYTES
MAILBOX_SIZE = MAILBOX_ERROR_OFFSET + MAILBOX_ERROR_BYTES

_OFF_MAGIC = 0
_OFF_PROTOCOL_VERSION = 8
_OFF_HEADER_BYTES = 12
_OFF_MAILBOX_BYTES = 16
_OFF_WORLD_SIZE = 24
_OFF_GROUP_STATE = 28
_OFF_REQUEST_STATE = 32
_OFF_SEQUENCE_ID = 40
_OFF_OPCODE = 48
_OFF_TARGET = 52
_OFF_TARGET_RANK = 56
_OFF_REQUEST_COUNT = 60
_OFF_REQUEST_BYTES = 64
_OFF_RESPONSE_COUNT = 68
_OFF_RESPONSE_BYTES = 72
_OFF_ERROR_BYTES = 76


class MailboxGroupState(enum.IntEnum):
    INITIALIZING = 0
    READY = 1
    TERMINAL = 2
    CLOSED = 3


class MailboxRequestState(enum.IntEnum):
    IDLE = 0
    REQUEST_READY = 1
    TASK_ACCEPTED = 2
    TASK_DONE = 3
    TASK_FAILED = 4
    SHUTDOWN_READY = 5
    SHUTDOWN_DONE = 6


class MailboxOpcode(enum.IntEnum):
    TASK = 1
    CONTROL = 2
    PING = 3
    SHUTDOWN = 4


class MailboxTarget(enum.IntEnum):
    GROUP = 1
    RANK = 2
    PER_RANK = 3


@dataclass(frozen=True)
class MpiRankError:
    rank: int
    error_type: str
    message: str


@dataclass(frozen=True)
class MailboxRequest:
    sequence_id: int
    opcode: MailboxOpcode
    target: MailboxTarget
    target_rank: int
    payloads: tuple[bytes, ...]


@dataclass(frozen=True)
class MailboxResult:
    sequence_id: int
    payloads: tuple[bytes, ...]


class MpiGroupError(RuntimeError):
    """A mailbox or MPI group operation failed."""


def _encode_payloads(payloads: tuple[bytes, ...]) -> bytes:
    values = tuple(bytes(payload) for payload in payloads)
    prefix_bytes = 4 + 4 * len(values)
    payload_bytes = sum(len(payload) for payload in values)
    if prefix_bytes + payload_bytes > MAILBOX_PAYLOAD_BYTES:
        raise ValueError("MPI group mailbox payload vector exceeds capacity")
    out = bytearray(struct.pack("<I", len(values)))
    for payload in values:
        out.extend(struct.pack("<I", len(payload)))
    for payload in values:
        out.extend(payload)
    return bytes(out)


def _decode_payloads(data: bytes, expected_count: int) -> tuple[bytes, ...]:
    if len(data) < 4:
        raise MpiGroupError("MPI group mailbox payload vector is truncated")
    (count,) = struct.unpack_from("<I", data)
    if count != expected_count:
        raise MpiGroupError(f"MPI group mailbox payload count mismatch: header={expected_count}, vector={count}")
    prefix_bytes = 4 + 4 * count
    if prefix_bytes > len(data):
        raise MpiGroupError("MPI group mailbox payload lengths are truncated")
    lengths = struct.unpack_from(f"<{count}I", data, 4) if count else ()
    offset = prefix_bytes
    payloads: list[bytes] = []
    for length in lengths:
        if offset > len(data) or length > len(data) - offset:
            raise MpiGroupError("MPI group mailbox payload entry is truncated")
        payloads.append(bytes(data[offset : offset + length]))
        offset += length
    if offset != len(data):
        raise MpiGroupError("MPI group mailbox payload vector has trailing bytes")
    return tuple(payloads)


def _as_byte_view(buffer: memoryview) -> memoryview:
    """Normalize platform-specific shared-memory formats to a byte view."""
    return buffer.cast("B")


def _buffer_address(buffer: memoryview) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(buffer))


class MpiGroupMailbox:
    """Owner or reopened view of one MPI group mailbox."""

    def __init__(self, shm: SharedMemory, *, owner: bool) -> None:
        self._shm = shm
        self._owner = bool(owner)
        self._closed = False
        self._validate_header()

    @classmethod
    def create(cls, *, world_size: int) -> MpiGroupMailbox:
        if int(world_size) <= 0:
            raise ValueError("MPI group mailbox world_size must be positive")
        shm = SharedMemory(create=True, size=MAILBOX_SIZE)
        buffer = shm.buf
        if buffer is None:
            shm.close()
            shm.unlink()
            raise MpiGroupError("MPI group mailbox mapping returned no buffer")
        buffer = _as_byte_view(buffer)
        ctypes.memset(_buffer_address(buffer), 0, MAILBOX_SIZE)
        struct.pack_into(
            "<8sIIQ", buffer, _OFF_MAGIC, MAILBOX_MAGIC, MAILBOX_PROTOCOL_VERSION, MAILBOX_HEADER_BYTES, MAILBOX_SIZE
        )
        struct.pack_into("<I", buffer, _OFF_WORLD_SIZE, int(world_size))
        struct.pack_into("<i", buffer, _OFF_GROUP_STATE, int(MailboxGroupState.INITIALIZING))
        struct.pack_into("<i", buffer, _OFF_REQUEST_STATE, int(MailboxRequestState.IDLE))
        return cls(shm, owner=True)

    @classmethod
    def open(cls, *, name: str) -> MpiGroupMailbox:
        try:
            shm = SharedMemory(name=str(name), create=False, track=False)
        except TypeError:  # Python < 3.13 has no per-instance resource-tracker switch.
            shm = SharedMemory(name=str(name), create=False)
            # A reopened rank-0 view is not the owner. Older Python versions
            # register every SharedMemory view and otherwise try to unlink it
            # when rank 0 exits, racing the L4 owner and emitting leak warnings.
            from multiprocessing import resource_tracker  # noqa: PLC0415

            resource_tracker.unregister(shm._name, "shared_memory")  # noqa: SLF001
        return cls(shm, owner=False)

    @property
    def name(self) -> str:
        return self._shm.name

    @property
    def address(self) -> int:
        self._require_open()
        return ctypes.addressof(ctypes.c_char.from_buffer(self._buffer))

    @property
    def world_size(self) -> int:
        return self._read_u32(_OFF_WORLD_SIZE)

    @property
    def group_state(self) -> MailboxGroupState:
        return MailboxGroupState(self._load_i32(_OFF_GROUP_STATE))

    @property
    def request_state(self) -> MailboxRequestState:
        return MailboxRequestState(self._load_i32(_OFF_REQUEST_STATE))

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "protocol_version": MAILBOX_PROTOCOL_VERSION,
            "mailbox_bytes": MAILBOX_SIZE,
            "world_size": self.world_size,
        }

    def publish_ready(self) -> None:
        if self.group_state is not MailboxGroupState.INITIALIZING:
            raise MpiGroupError("MPI group mailbox READY can only be published from INITIALIZING")
        self._store_i32(_OFF_GROUP_STATE, int(MailboxGroupState.READY))

    def publish_closed(self) -> None:
        if self.group_state is MailboxGroupState.TERMINAL:
            return
        self._store_i32(_OFF_GROUP_STATE, int(MailboxGroupState.CLOSED))

    def write_request(
        self,
        *,
        sequence_id: int,
        opcode: MailboxOpcode,
        target: MailboxTarget,
        target_rank: int,
        payloads: tuple[bytes, ...],
    ) -> None:
        self._require_ready()
        if self.request_state is not MailboxRequestState.IDLE:
            raise MpiGroupError(f"MPI group mailbox is busy in state {self.request_state.name}")
        sequence_id = int(sequence_id)
        if sequence_id <= 0:
            raise ValueError("MPI group mailbox sequence_id must be positive")
        opcode = MailboxOpcode(opcode)
        target = MailboxTarget(target)
        target_rank = int(target_rank)
        payloads = tuple(bytes(payload) for payload in payloads)
        if target is MailboxTarget.RANK:
            if target_rank < 0 or target_rank >= self.world_size:
                raise ValueError("MPI group mailbox target rank is outside the group")
            if len(payloads) != 1:
                raise ValueError("rank-targeted MPI group request requires one payload")
        elif target is MailboxTarget.GROUP:
            if target_rank != -1 or len(payloads) != 1:
                raise ValueError("group-targeted MPI request requires target_rank=-1 and one payload")
        elif target is MailboxTarget.PER_RANK:
            if target_rank != -1 or len(payloads) != self.world_size:
                raise ValueError("per-rank MPI request requires one payload for every rank")
        data = _encode_payloads(payloads)
        self._write_bytes(MAILBOX_REQUEST_OFFSET, data)
        self._write_u64(_OFF_SEQUENCE_ID, sequence_id)
        self._write_u32(_OFF_OPCODE, int(opcode))
        self._write_u32(_OFF_TARGET, int(target))
        self._write_i32(_OFF_TARGET_RANK, target_rank)
        self._write_u32(_OFF_REQUEST_COUNT, len(payloads))
        self._write_u32(_OFF_REQUEST_BYTES, len(data))
        self._write_u32(_OFF_RESPONSE_COUNT, 0)
        self._write_u32(_OFF_RESPONSE_BYTES, 0)
        self._write_u32(_OFF_ERROR_BYTES, 0)
        ready_state = (
            MailboxRequestState.SHUTDOWN_READY
            if opcode is MailboxOpcode.SHUTDOWN
            else MailboxRequestState.REQUEST_READY
        )
        self._store_i32(_OFF_REQUEST_STATE, int(ready_state))

    def accept_request(self, *, last_sequence_id: int) -> MailboxRequest:
        state = self.request_state
        if state not in (MailboxRequestState.REQUEST_READY, MailboxRequestState.SHUTDOWN_READY):
            raise MpiGroupError(f"MPI group mailbox has no request to accept (state={state.name})")
        sequence_id = self._read_u64(_OFF_SEQUENCE_ID)
        request_bytes = self._read_u32(_OFF_REQUEST_BYTES)
        request_count = self._read_u32(_OFF_REQUEST_COUNT)
        if request_bytes > MAILBOX_PAYLOAD_BYTES:
            self.mark_terminal("request payload exceeds mailbox capacity")
            raise MpiGroupError("MPI group mailbox request payload exceeds capacity")
        data = self._read_bytes(MAILBOX_REQUEST_OFFSET, request_bytes)
        try:
            if sequence_id <= int(last_sequence_id):
                raise MpiGroupError(
                    f"MPI group mailbox sequence_id {sequence_id} is not newer than {int(last_sequence_id)}"
                )
            request = MailboxRequest(
                sequence_id=sequence_id,
                opcode=MailboxOpcode(self._read_u32(_OFF_OPCODE)),
                target=MailboxTarget(self._read_u32(_OFF_TARGET)),
                target_rank=self._read_i32(_OFF_TARGET_RANK),
                payloads=_decode_payloads(data, request_count),
            )
        except BaseException as exc:
            self.mark_terminal(str(exc))
            raise
        self._store_i32(_OFF_REQUEST_STATE, int(MailboxRequestState.TASK_ACCEPTED))
        return request

    def complete_request(self, *, sequence_id: int, payloads: tuple[bytes, ...]) -> None:
        self._validate_active_sequence(sequence_id)
        values = tuple(bytes(payload) for payload in payloads)
        data = _encode_payloads(values)
        self._write_bytes(MAILBOX_RESPONSE_OFFSET, data)
        self._write_u32(_OFF_RESPONSE_COUNT, len(values))
        self._write_u32(_OFF_RESPONSE_BYTES, len(data))
        self._store_i32(_OFF_REQUEST_STATE, int(MailboxRequestState.TASK_DONE))

    def complete_shutdown(self, *, sequence_id: int) -> None:
        self._validate_active_sequence(sequence_id)
        self._store_i32(_OFF_REQUEST_STATE, int(MailboxRequestState.SHUTDOWN_DONE))

    def fail_request(
        self,
        *,
        sequence_id: int,
        errors: tuple[MpiRankError, ...],
        terminal: bool,
    ) -> None:
        self._validate_active_sequence(sequence_id)
        if not errors:
            raise ValueError("MPI group mailbox failure requires at least one rank error")
        data = json.dumps([asdict(error) for error in errors], sort_keys=True).encode("utf-8")
        if len(data) > MAILBOX_ERROR_BYTES:
            data = data[: MAILBOX_ERROR_BYTES - 1]
        self._write_bytes(MAILBOX_ERROR_OFFSET, data)
        self._write_u32(_OFF_ERROR_BYTES, len(data))
        if terminal:
            self._store_i32(_OFF_GROUP_STATE, int(MailboxGroupState.TERMINAL))
        self._store_i32(_OFF_REQUEST_STATE, int(MailboxRequestState.TASK_FAILED))

    def read_result(self, *, sequence_id: int) -> MailboxResult:
        if self._read_u64(_OFF_SEQUENCE_ID) != int(sequence_id):
            raise MpiGroupError("MPI group mailbox result sequence does not match the request")
        state = self.request_state
        if state is MailboxRequestState.TASK_DONE:
            response_bytes = self._read_u32(_OFF_RESPONSE_BYTES)
            response_count = self._read_u32(_OFF_RESPONSE_COUNT)
            if response_bytes > MAILBOX_PAYLOAD_BYTES:
                self.mark_terminal("response payload exceeds mailbox capacity")
                raise MpiGroupError("MPI group mailbox response payload exceeds capacity")
            data = self._read_bytes(MAILBOX_RESPONSE_OFFSET, response_bytes)
            result = MailboxResult(int(sequence_id), _decode_payloads(data, response_count))
            self._store_i32(_OFF_REQUEST_STATE, int(MailboxRequestState.IDLE))
            return result
        if state is MailboxRequestState.TASK_FAILED:
            error_bytes = min(self._read_u32(_OFF_ERROR_BYTES), MAILBOX_ERROR_BYTES)
            raw = self._read_bytes(MAILBOX_ERROR_OFFSET, error_bytes)
            try:
                entries = json.loads(raw.decode("utf-8"))
                message = "; ".join(
                    f"rank {int(entry['rank'])}: {entry['error_type']}: {entry['message']}" for entry in entries
                )
            except BaseException:
                message = raw.decode("utf-8", errors="replace") or "MPI group request failed"
            if self.group_state is MailboxGroupState.READY:
                self._store_i32(_OFF_REQUEST_STATE, int(MailboxRequestState.IDLE))
            raise MpiGroupError(message)
        raise MpiGroupError(f"MPI group mailbox result is not ready (state={state.name})")

    def mark_terminal(self, reason: str) -> None:
        data = str(reason).encode("utf-8")[:MAILBOX_ERROR_BYTES]
        self._write_bytes(MAILBOX_ERROR_OFFSET, data)
        self._write_u32(_OFF_ERROR_BYTES, len(data))
        self._store_i32(_OFF_GROUP_STATE, int(MailboxGroupState.TERMINAL))
        self._store_i32(_OFF_REQUEST_STATE, int(MailboxRequestState.TASK_FAILED))

    def terminal_reason(self) -> str:
        error_bytes = min(self._read_u32(_OFF_ERROR_BYTES), MAILBOX_ERROR_BYTES)
        return self._read_bytes(MAILBOX_ERROR_OFFSET, error_bytes).decode("utf-8", errors="replace")

    def overwrite_request_payload_for_test(self, data: bytes) -> None:
        value = bytes(data)
        self._write_bytes(MAILBOX_REQUEST_OFFSET, value)

    def close(self, *, unlink: bool = False) -> None:
        if self._closed:
            return
        if unlink and not self._owner:
            raise RuntimeError("only the MPI group mailbox owner may unlink it")
        self._shm.close()
        if unlink:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                # Python < 3.13 may let a non-owner process resource tracker
                # unlink the name first; the owner mapping still closes here.
                pass
        self._closed = True

    def _validate_header(self) -> None:
        if len(self._buffer) < MAILBOX_SIZE:
            raise MpiGroupError("MPI group mailbox is smaller than the protocol layout")
        magic, version, header_bytes, mailbox_bytes = struct.unpack_from("<8sIIQ", self._buffer, _OFF_MAGIC)
        if magic != MAILBOX_MAGIC:
            raise MpiGroupError("MPI group mailbox magic does not match")
        if version != MAILBOX_PROTOCOL_VERSION:
            raise MpiGroupError(f"MPI group mailbox protocol version {version} is not supported")
        if header_bytes != MAILBOX_HEADER_BYTES or mailbox_bytes != MAILBOX_SIZE:
            raise MpiGroupError("MPI group mailbox layout does not match the protocol")
        if self._read_u32(_OFF_WORLD_SIZE) == 0:
            raise MpiGroupError("MPI group mailbox world_size must be positive")

    def _validate_active_sequence(self, sequence_id: int) -> None:
        if self.request_state is not MailboxRequestState.TASK_ACCEPTED:
            raise MpiGroupError("MPI group mailbox request has not been accepted")
        if self._read_u64(_OFF_SEQUENCE_ID) != int(sequence_id):
            raise MpiGroupError("MPI group mailbox active sequence does not match")

    def _require_ready(self) -> None:
        state = self.group_state
        if state is MailboxGroupState.TERMINAL:
            reason = self.terminal_reason() or "terminal failure"
            raise MpiGroupError(f"MPI group mailbox is terminal: {reason}")
        if state is not MailboxGroupState.READY:
            raise MpiGroupError(f"MPI group mailbox is not ready (state={state.name})")

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("MPI group mailbox is closed")

    @property
    def _buffer(self) -> memoryview:
        buffer = self._shm.buf
        if buffer is None:
            raise RuntimeError("MPI group mailbox mapping has no buffer")
        return _as_byte_view(buffer)

    def _write_bytes(self, offset: int, data: bytes) -> None:
        self._require_open()
        offset = int(offset)
        buffer = self._buffer
        if offset < 0 or offset > len(buffer) or len(data) > len(buffer) - offset:
            raise ValueError("MPI group mailbox write is outside the mapping")
        if data:
            ctypes.memmove(_buffer_address(buffer) + offset, data, len(data))

    def _read_bytes(self, offset: int, length: int) -> bytes:
        self._require_open()
        offset = int(offset)
        length = int(length)
        buffer = self._buffer
        if offset < 0 or length < 0 or offset > len(buffer) or length > len(buffer) - offset:
            raise ValueError("MPI group mailbox read is outside the mapping")
        if length == 0:
            return b""
        return ctypes.string_at(_buffer_address(buffer) + offset, length)

    def _load_i32(self, offset: int) -> int:
        self._require_open()
        try:
            from .task_interface import _mailbox_load_i32  # noqa: PLC0415
        except (ImportError, AttributeError):
            return self._read_i32(offset)
        return int(_mailbox_load_i32(self.address + offset))

    def _store_i32(self, offset: int, value: int) -> None:
        self._require_open()
        try:
            from .task_interface import _mailbox_store_i32  # noqa: PLC0415
        except (ImportError, AttributeError):
            self._write_i32(offset, value)
            return
        _mailbox_store_i32(self.address + offset, int(value))

    def _read_i32(self, offset: int) -> int:
        return int(struct.unpack_from("<i", self._buffer, offset)[0])

    def _write_i32(self, offset: int, value: int) -> None:
        struct.pack_into("<i", self._buffer, offset, int(value))

    def _read_u32(self, offset: int) -> int:
        return int(struct.unpack_from("<I", self._buffer, offset)[0])

    def _write_u32(self, offset: int, value: int) -> None:
        struct.pack_into("<I", self._buffer, offset, int(value))

    def _read_u64(self, offset: int) -> int:
        return int(struct.unpack_from("<Q", self._buffer, offset)[0])

    def _write_u64(self, offset: int, value: int) -> None:
        struct.pack_into("<Q", self._buffer, offset, int(value))


def open_rank_mailbox(manifest: dict[str, Any], *, rank: int) -> MpiGroupMailbox | None:
    """Open the L4 mailbox on rank 0; all other ranks stay MPI-only."""

    if int(rank) != 0:
        return None
    if int(manifest.get("protocol_version", -1)) != MAILBOX_PROTOCOL_VERSION:
        raise MpiGroupError("MPI group manifest mailbox protocol version does not match")
    if int(manifest.get("mailbox_bytes", -1)) != MAILBOX_SIZE:
        raise MpiGroupError("MPI group manifest mailbox size does not match")
    mailbox = MpiGroupMailbox.open(name=str(manifest["name"]))
    if mailbox.world_size != int(manifest.get("world_size", -1)):
        mailbox.close()
        raise MpiGroupError("MPI group manifest world_size does not match the mailbox")
    return mailbox
