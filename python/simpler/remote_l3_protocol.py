# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Python codec for the Remote L3 wire protocol used by session runners."""

from __future__ import annotations

import enum
import socket
import struct
from dataclasses import dataclass

from .task_interface import CONTINUOUS_TENSOR_MAX_DIMS, CallConfig, ContinuousTensor, DataType

PROTOCOL_VERSION = 1
MAX_FRAME_PAYLOAD_BYTES = 16 * 1024 * 1024
MAX_STRING_BYTES = 1024
MAX_ERROR_BYTES = 4096
MAX_TENSORS = 4096
MAX_SCALARS = 4096
MAX_INLINE_PAYLOAD_BYTES = 1024 * 1024
MAX_TRANSPORT_PROFILE_BYTES = 128
MAX_TRANSPORT_DESCRIPTOR_BYTES = 4096
MAX_CHIP_CALLABLE_DESCRIPTOR_BYTES = 4096
MAX_STAGED_BLOB_TOKEN_BYTES = 1024
REMOTE_BUFFER_ACCESS_READ = 1 << 0
REMOTE_BUFFER_ACCESS_WRITE = 1 << 1
REMOTE_BUFFER_ACCESS_READ_WRITE = REMOTE_BUFFER_ACCESS_READ | REMOTE_BUFFER_ACCESS_WRITE
CALLABLE_HASH_DIGEST_BYTES = 32
FRAME_HEADER_BYTES = 40
MAGIC = b"SLR3"


class FrameType(enum.IntEnum):
    HELLO = 1
    TASK = 2
    CONTROL = 3
    CONTROL_REPLY = 4
    COMPLETION = 5
    HEALTH = 6
    SHUTDOWN = 7


class ControlName(enum.IntEnum):
    UNREGISTER_CALLABLE = 1
    PREPARE_REGISTER_CALLABLE = 2
    COMMIT_REGISTER_CALLABLE = 3
    ABORT_REGISTER_CALLABLE = 4
    PREPARE_CALLABLE = 5
    ALLOC_REMOTE_BUFFER = 6
    FREE_REMOTE_BUFFER = 7
    COPY_TO_REMOTE = 8
    COPY_FROM_REMOTE = 9
    EXPORT_BUFFER = 10
    IMPORT_BUFFER = 11
    RELEASE_IMPORT = 12
    COMM_INIT = 13
    ALLOC_DOMAIN = 14
    RELEASE_DOMAIN = 15


class RemoteRegistryTarget(enum.IntEnum):
    REMOTE_TASK_DISPATCHER = 1
    INNER_L3_WORKER = 2


class CallableKind(enum.IntEnum):
    CHIP_CALLABLE = 1
    PYTHON_SERIALIZED = 2
    PYTHON_IMPORT = 3


class ChipCallableBlobLocation(enum.IntEnum):
    INLINE_BLOB = 1
    STAGED_BLOB = 2


class ReadyState(enum.IntEnum):
    NOT_READY = 0
    READY = 1


class RemoteAddressSpace(enum.IntEnum):
    HOST_INLINE = 1
    REMOTE_DEVICE = 2
    REMOTE_WINDOW = 3
    UB_LDST = 4


@dataclass(frozen=True)
class FrameHeader:
    frame_type: FrameType
    session_id: int
    endpoint_id: int
    sequence: int
    payload_bytes: int = 0
    flags: int = 0


@dataclass(frozen=True)
class Frame:
    header: FrameHeader
    payload: bytes


@dataclass(frozen=True)
class HelloPayload:
    session_id: int
    endpoint_id: int
    protocol_version: int
    comm_profile: str
    feature_flags: int
    ready_state: ReadyState


@dataclass(frozen=True)
class RemoteTensorDesc:
    address_space: RemoteAddressSpace
    owner_endpoint_id: int
    buffer_id: int
    offset: int
    nbytes: int
    remote_addr: int
    rkey_or_token: int
    generation: int
    inline_payload_offset: int
    inline_payload_len: int
    flags: int


@dataclass(frozen=True)
class RemoteTensorSidecar:
    present: bool
    desc: RemoteTensorDesc | None = None


@dataclass(frozen=True)
class RemoteTaskArgsWire:
    tensor_metadata: tuple[ContinuousTensor, ...]
    remote_desc: tuple[RemoteTensorSidecar, ...]
    scalars: tuple[int, ...]
    inline_payload: bytes


@dataclass(frozen=True)
class TaskPayloadWire:
    callable_digest: bytes
    config: CallConfig
    args: RemoteTaskArgsWire


@dataclass(frozen=True)
class ControlPayload:
    control_name: ControlName
    control_version: int
    command_bytes: bytes


@dataclass(frozen=True)
class RegisterCallableCommand:
    target_registry: RemoteRegistryTarget
    callable_kind: CallableKind
    digest: bytes
    payload_version: int
    payload: bytes


@dataclass(frozen=True)
class DigestCallableCommand:
    target_registry: RemoteRegistryTarget
    callable_kind: CallableKind
    digest: bytes


@dataclass(frozen=True)
class ExportBufferRequest:
    owner_endpoint_id: int
    buffer_id: int
    generation: int
    offset: int
    nbytes: int
    access_flags: int
    transport_profile: str


@dataclass(frozen=True)
class ExportBufferResult:
    owner_endpoint_id: int
    buffer_id: int
    generation: int
    address_space: RemoteAddressSpace
    offset: int
    nbytes: int
    export_id: int
    remote_addr: int
    rkey_or_token: int
    ub_ldst_va: int
    access_flags: int
    transport_profile: str
    transport_descriptor: bytes


@dataclass(frozen=True)
class ImportBufferRequest:
    importer_endpoint_id: int
    requested_access_flags: int
    export_desc: ExportBufferResult


@dataclass(frozen=True)
class ImportBufferResult:
    importer_endpoint_id: int
    owner_endpoint_id: int
    buffer_id: int
    generation: int
    import_id: int
    address_space: RemoteAddressSpace
    offset: int
    nbytes: int
    remote_addr: int
    rkey_or_token: int
    ub_ldst_va: int
    access_flags: int
    transport_profile: str
    import_descriptor: bytes


@dataclass(frozen=True)
class ReleaseImportRequest:
    importer_endpoint_id: int
    owner_endpoint_id: int
    buffer_id: int
    generation: int
    import_id: int


@dataclass(frozen=True)
class RemoteChipCallablePayload:
    descriptor_bytes: bytes
    blob_location: ChipCallableBlobLocation
    blob_size: int
    blob_sha256: bytes
    inline_blob: bytes
    staged_blob_token: bytes


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def require(self, n: int, what: str) -> None:
        if self.offset > len(self.data) or n > len(self.data) - self.offset:
            raise ValueError(f"remote_wire: truncated {what}")

    def u8(self) -> int:
        self.require(1, "uint8")
        v = self.data[self.offset]
        self.offset += 1
        return v

    def u32(self) -> int:
        self.require(4, "uint32")
        v = struct.unpack_from("<I", self.data, self.offset)[0]
        self.offset += 4
        return int(v)

    def i32(self) -> int:
        self.require(4, "int32")
        v = struct.unpack_from("<i", self.data, self.offset)[0]
        self.offset += 4
        return int(v)

    def u64(self) -> int:
        self.require(8, "uint64")
        v = struct.unpack_from("<Q", self.data, self.offset)[0]
        self.offset += 8
        return int(v)

    def raw(self, n: int, what: str) -> bytes:
        self.require(n, what)
        out = self.data[self.offset : self.offset + n]
        self.offset += n
        return out

    def string(self, max_bytes: int, field_name: str) -> str:
        n = self.u32()
        if n > max_bytes:
            raise ValueError(f"remote_wire: {field_name} exceeds max length")
        return self.raw(n, field_name).decode("utf-8")

    def blob(self, max_bytes: int, field_name: str) -> bytes:
        n = self.u32()
        if n > max_bytes:
            raise ValueError(f"remote_wire: {field_name} exceeds max length")
        return self.raw(n, field_name)

    def done(self, what: str) -> None:
        if self.offset != len(self.data):
            raise ValueError(f"remote_wire: trailing bytes after {what}")


def _put_string(out: bytearray, value: str, max_bytes: int, field_name: str) -> None:
    data = value.encode("utf-8")
    if len(data) > max_bytes:
        raise ValueError(f"remote_wire: {field_name} exceeds max length")
    out.extend(struct.pack("<I", len(data)))
    out.extend(data)


def _put_blob(out: bytearray, value: bytes, max_bytes: int, field_name: str) -> None:
    data = bytes(value)
    if len(data) > max_bytes:
        raise ValueError(f"remote_wire: {field_name} exceeds max length")
    out.extend(struct.pack("<I", len(data)))
    out.extend(data)


def _validate_access_flags(flags: int, field_name: str) -> None:
    if int(flags) == 0:
        raise ValueError(f"remote_wire: {field_name} must be non-zero")
    if int(flags) & ~REMOTE_BUFFER_ACCESS_READ_WRITE:
        raise ValueError(f"remote_wire: {field_name} contains unknown bits")


def encode_frame(header: FrameHeader, payload: bytes) -> bytes:
    if len(payload) > MAX_FRAME_PAYLOAD_BYTES:
        raise ValueError("remote_wire: frame payload exceeds maximum")
    if header.flags != 0:
        raise ValueError("remote_wire: frame flags are reserved in v1")
    return (
        MAGIC
        + struct.pack(
            "<IIQiQII",
            PROTOCOL_VERSION,
            int(header.frame_type),
            int(header.session_id),
            int(header.endpoint_id),
            int(header.sequence),
            len(payload),
            int(header.flags),
        )
        + payload
    )


def decode_frame(data: bytes) -> Frame:
    if len(data) < FRAME_HEADER_BYTES:
        raise ValueError("remote_wire: truncated frame header")
    if data[:4] != MAGIC:
        raise ValueError("remote_wire: bad frame magic")
    version, raw_type, session_id, endpoint_id, sequence, payload_bytes, flags = struct.unpack_from("<IIQiQII", data, 4)
    if version != PROTOCOL_VERSION:
        raise ValueError("remote_wire: unsupported frame version")
    try:
        frame_type = FrameType(raw_type)
    except ValueError as exc:
        raise ValueError("remote_wire: unknown frame type") from exc
    if flags != 0:
        raise ValueError("remote_wire: frame flags are reserved in v1")
    if payload_bytes > MAX_FRAME_PAYLOAD_BYTES:
        raise ValueError("remote_wire: frame payload exceeds maximum")
    if len(data) - FRAME_HEADER_BYTES != payload_bytes:
        raise ValueError("remote_wire: frame payload length mismatch")
    return Frame(
        FrameHeader(frame_type, int(session_id), int(endpoint_id), int(sequence), int(payload_bytes), int(flags)),
        data[FRAME_HEADER_BYTES:],
    )


def read_exact(sock: socket.socket, n: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < n:
        data = sock.recv(n - len(chunks))
        if not data:
            raise EOFError("remote socket closed")
        chunks.extend(data)
    return bytes(chunks)


def read_frame(sock: socket.socket) -> Frame:
    header = read_exact(sock, FRAME_HEADER_BYTES)
    payload_bytes = struct.unpack_from("<I", header, 32)[0]
    if payload_bytes > MAX_FRAME_PAYLOAD_BYTES:
        raise ValueError("remote_wire: frame payload exceeds maximum")
    return decode_frame(header + read_exact(sock, payload_bytes))


def send_frame(sock: socket.socket, header: FrameHeader, payload: bytes = b"") -> None:
    sock.sendall(encode_frame(header, payload))


def encode_hello(payload: HelloPayload) -> bytes:
    out = bytearray()
    out.extend(struct.pack("<QiI", int(payload.session_id), int(payload.endpoint_id), int(payload.protocol_version)))
    _put_string(out, payload.comm_profile, MAX_STRING_BYTES, "HELLO.comm_profile")
    out.extend(struct.pack("<QI", int(payload.feature_flags), int(payload.ready_state)))
    return bytes(out)


def decode_call_config(reader: _Reader) -> CallConfig:
    cfg = CallConfig()
    cfg.block_dim = reader.i32()
    cfg.aicpu_thread_num = reader.i32()
    cfg.enable_l2_swimlane = reader.i32()
    cfg.enable_dump_tensor = reader.i32()
    cfg.enable_pmu = reader.i32()
    cfg.enable_dep_gen = bool(reader.i32())
    cfg.enable_scope_stats = False
    prefix = reader.string(MAX_STRING_BYTES, "CallConfig.output_prefix")
    cfg.output_prefix = prefix
    return cfg


def decode_continuous_tensor(reader: _Reader) -> ContinuousTensor:
    data = reader.u64()
    if data != 0:
        raise ValueError("remote_wire: remote TASK tensor data must be zero")
    shapes = [reader.u32() for _ in range(CONTINUOUS_TENSOR_MAX_DIMS)]
    ndims = reader.u32()
    if ndims > CONTINUOUS_TENSOR_MAX_DIMS:
        raise ValueError("remote_wire: tensor ndims exceeds maximum")
    dtype = DataType(reader.u32())
    child_memory = reader.u8()
    if child_memory not in (0, 1):
        raise ValueError("remote_wire: tensor child_memory must be 0 or 1")
    for _ in range(7):
        if reader.u8() != 0:
            raise ValueError("remote_wire: ContinuousTensor reserved bytes must be zero")
    return ContinuousTensor.make(0, tuple(shapes[:ndims]), dtype, bool(child_memory))


def decode_remote_tensor_desc(reader: _Reader) -> RemoteTensorDesc:
    return RemoteTensorDesc(
        address_space=RemoteAddressSpace(reader.u32()),
        owner_endpoint_id=reader.i32(),
        buffer_id=reader.u64(),
        offset=reader.u64(),
        nbytes=reader.u64(),
        remote_addr=reader.u64(),
        rkey_or_token=reader.u64(),
        generation=reader.u64(),
        inline_payload_offset=reader.u64(),
        inline_payload_len=reader.u64(),
        flags=reader.u64(),
    )


def _validate_desc(desc: RemoteTensorDesc, inline_payload_len: int) -> None:
    if desc.flags != 0:
        raise ValueError("remote_wire: RemoteTensorDesc flags are reserved in v1")
    if desc.offset + desc.nbytes > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("remote_wire: RemoteTensorDesc offset+nbytes overflows")
    if desc.address_space == RemoteAddressSpace.HOST_INLINE:
        if (
            desc.owner_endpoint_id != 0
            or desc.buffer_id != 0
            or desc.remote_addr != 0
            or desc.rkey_or_token != 0
            or desc.generation != 0
        ):
            raise ValueError("remote_wire: HOST_INLINE remote handle fields must be zero")
        if desc.inline_payload_len != desc.nbytes:
            raise ValueError("remote_wire: HOST_INLINE inline length must equal nbytes")
        if desc.inline_payload_offset + desc.inline_payload_len > inline_payload_len:
            raise ValueError("remote_wire: HOST_INLINE payload range exceeds inline arena")
    else:
        if desc.inline_payload_offset != 0 or desc.inline_payload_len != 0:
            raise ValueError("remote_wire: non-HOST_INLINE inline fields must be zero")
        if desc.owner_endpoint_id < 0:
            raise ValueError("remote_wire: remote descriptor owner endpoint must be non-negative")
        if desc.buffer_id == 0 or desc.generation == 0:
            raise ValueError("remote_wire: remote descriptor buffer_id and generation must be non-zero")


def decode_remote_task_args(data: bytes) -> RemoteTaskArgsWire:
    reader = _Reader(data)
    tensor_count = reader.u32()
    scalar_count = reader.u32()
    if tensor_count > MAX_TENSORS:
        raise ValueError("remote_wire: tensor count exceeds maximum")
    if scalar_count > MAX_SCALARS:
        raise ValueError("remote_wire: scalar count exceeds maximum")
    tensors = tuple(decode_continuous_tensor(reader) for _ in range(tensor_count))
    sidecars: list[RemoteTensorSidecar] = []
    for _ in range(tensor_count):
        present = reader.u8()
        if present not in (0, 1):
            raise ValueError("remote_wire: remote descriptor presence must be 0 or 1")
        sidecars.append(
            RemoteTensorSidecar(False, None)
            if present == 0
            else RemoteTensorSidecar(True, decode_remote_tensor_desc(reader))
        )
    scalars = tuple(reader.u64() for _ in range(scalar_count))
    inline_len = reader.u32()
    if inline_len > MAX_INLINE_PAYLOAD_BYTES:
        raise ValueError("remote_wire: inline payload exceeds maximum")
    inline_payload = reader.raw(inline_len, "inline payload")
    reader.done("RemoteTaskArgs")
    for sidecar in sidecars:
        if sidecar.present:
            assert sidecar.desc is not None
            _validate_desc(sidecar.desc, len(inline_payload))
    return RemoteTaskArgsWire(tensors, tuple(sidecars), scalars, inline_payload)


def decode_task_payload(data: bytes) -> TaskPayloadWire:
    reader = _Reader(data)
    digest = reader.raw(CALLABLE_HASH_DIGEST_BYTES, "callable digest")
    cfg = decode_call_config(reader)
    args = decode_remote_task_args(data[reader.offset :])
    return TaskPayloadWire(digest, cfg, args)


def encode_completion(sequence: int, error_code: int, error_message: str) -> bytes:
    data = error_message.encode("utf-8")
    if len(data) > MAX_ERROR_BYTES:
        data = data[:MAX_ERROR_BYTES]
    return struct.pack("<QiI", int(sequence), int(error_code), len(data)) + data


def decode_control(data: bytes) -> ControlPayload:
    reader = _Reader(data)
    control_name = ControlName(reader.u32())
    control_version = reader.u32()
    if control_version == 0:
        raise ValueError("remote_wire: control version must be non-zero")
    command_len = reader.u32()
    if command_len > MAX_FRAME_PAYLOAD_BYTES:
        raise ValueError("remote_wire: control payload too large")
    command_bytes = reader.raw(command_len, "control payload")
    reader.done("control")
    return ControlPayload(control_name, control_version, command_bytes)


def decode_register_callable_command(data: bytes) -> RegisterCallableCommand:
    reader = _Reader(data)
    target_registry = RemoteRegistryTarget(reader.u32())
    callable_kind = CallableKind(reader.u32())
    digest = reader.raw(CALLABLE_HASH_DIGEST_BYTES, "callable digest")
    payload_version = reader.u32()
    if payload_version == 0:
        raise ValueError("remote_wire: callable payload version must be non-zero")
    payload_len = reader.u32()
    if payload_len > MAX_FRAME_PAYLOAD_BYTES:
        raise ValueError("remote_wire: callable payload too large")
    payload = reader.raw(payload_len, "callable payload")
    reader.done("register callable command")
    return RegisterCallableCommand(target_registry, callable_kind, digest, payload_version, payload)


def encode_register_callable_command(
    target_registry: RemoteRegistryTarget,
    callable_kind: CallableKind,
    digest: bytes,
    payload_version: int,
    payload: bytes,
) -> bytes:
    if len(digest) != CALLABLE_HASH_DIGEST_BYTES:
        raise ValueError("remote_wire: callable digest must be 32 bytes")
    if int(payload_version) == 0:
        raise ValueError("remote_wire: callable payload version must be non-zero")
    payload_bytes = bytes(payload)
    if len(payload_bytes) > MAX_FRAME_PAYLOAD_BYTES:
        raise ValueError("remote_wire: callable payload too large")
    out = bytearray()
    out.extend(struct.pack("<II", int(target_registry), int(callable_kind)))
    out.extend(bytes(digest))
    out.extend(struct.pack("<II", int(payload_version), len(payload_bytes)))
    out.extend(payload_bytes)
    return bytes(out)


def decode_digest_callable_command(data: bytes) -> DigestCallableCommand:
    reader = _Reader(data)
    target_registry = RemoteRegistryTarget(reader.u32())
    callable_kind = CallableKind(reader.u32())
    digest = reader.raw(CALLABLE_HASH_DIGEST_BYTES, "callable digest")
    reader.done("digest callable command")
    return DigestCallableCommand(target_registry, callable_kind, digest)


def encode_digest_callable_command(
    target_registry: RemoteRegistryTarget,
    callable_kind: CallableKind,
    digest: bytes,
) -> bytes:
    if len(digest) != CALLABLE_HASH_DIGEST_BYTES:
        raise ValueError("remote_wire: callable digest must be 32 bytes")
    return struct.pack("<II", int(target_registry), int(callable_kind)) + bytes(digest)


def encode_remote_chip_callable_payload(payload: RemoteChipCallablePayload) -> bytes:
    if len(payload.blob_sha256) != CALLABLE_HASH_DIGEST_BYTES:
        raise ValueError("remote_wire: CHIP_CALLABLE blob_sha256 must be 32 bytes")
    blob_size = int(payload.blob_size)
    if blob_size <= 0:
        raise ValueError("remote_wire: CHIP_CALLABLE blob_size must be non-zero")
    location = ChipCallableBlobLocation(payload.blob_location)
    if location == ChipCallableBlobLocation.INLINE_BLOB:
        if len(payload.inline_blob) != blob_size or payload.staged_blob_token:
            raise ValueError("remote_wire: CHIP_CALLABLE inline payload fields are inconsistent")
    elif location == ChipCallableBlobLocation.STAGED_BLOB:
        if payload.inline_blob or not payload.staged_blob_token:
            raise ValueError("remote_wire: CHIP_CALLABLE staged payload fields are inconsistent")
    out = bytearray()
    _put_blob(out, payload.descriptor_bytes, MAX_CHIP_CALLABLE_DESCRIPTOR_BYTES, "CHIP_CALLABLE.descriptor_bytes")
    out.extend(struct.pack("<IQ", int(location), blob_size))
    out.extend(payload.blob_sha256)
    _put_blob(out, payload.inline_blob, MAX_FRAME_PAYLOAD_BYTES, "CHIP_CALLABLE.inline_blob")
    _put_blob(out, payload.staged_blob_token, MAX_STAGED_BLOB_TOKEN_BYTES, "CHIP_CALLABLE.staged_blob_token")
    out.extend(struct.pack("<I", 0))
    return bytes(out)


def decode_remote_chip_callable_payload(data: bytes) -> RemoteChipCallablePayload:
    reader = _Reader(data)
    descriptor_bytes = reader.blob(MAX_CHIP_CALLABLE_DESCRIPTOR_BYTES, "CHIP_CALLABLE.descriptor_bytes")
    blob_location = ChipCallableBlobLocation(reader.u32())
    blob_size = reader.u64()
    blob_sha256 = reader.raw(CALLABLE_HASH_DIGEST_BYTES, "CHIP_CALLABLE.blob_sha256")
    inline_blob = reader.blob(MAX_FRAME_PAYLOAD_BYTES, "CHIP_CALLABLE.inline_blob")
    staged_blob_token = reader.blob(MAX_STAGED_BLOB_TOKEN_BYTES, "CHIP_CALLABLE.staged_blob_token")
    if reader.u32() != 0:
        raise ValueError("remote_wire: CHIP_CALLABLE reserved field must be zero")
    reader.done("CHIP_CALLABLE payload")
    payload = RemoteChipCallablePayload(
        descriptor_bytes=descriptor_bytes,
        blob_location=blob_location,
        blob_size=blob_size,
        blob_sha256=blob_sha256,
        inline_blob=inline_blob,
        staged_blob_token=staged_blob_token,
    )
    encode_remote_chip_callable_payload(payload)
    return payload


def encode_export_buffer_result(result: ExportBufferResult) -> bytes:
    _validate_access_flags(result.access_flags, "export result access_flags")
    if result.address_space not in (RemoteAddressSpace.REMOTE_WINDOW, RemoteAddressSpace.UB_LDST):
        raise ValueError("remote_wire: export result address_space is invalid")
    if result.nbytes <= 0 or result.export_id == 0:
        raise ValueError("remote_wire: export result requires non-zero nbytes and export_id")
    out = bytearray()
    out.extend(
        struct.pack(
            "<iQQIQQQQQQI",
            int(result.owner_endpoint_id),
            int(result.buffer_id),
            int(result.generation),
            int(result.address_space),
            int(result.offset),
            int(result.nbytes),
            int(result.export_id),
            int(result.remote_addr),
            int(result.rkey_or_token),
            int(result.ub_ldst_va),
            int(result.access_flags),
        )
    )
    _put_string(out, result.transport_profile, MAX_TRANSPORT_PROFILE_BYTES, "export result transport_profile")
    _put_blob(
        out,
        result.transport_descriptor,
        MAX_TRANSPORT_DESCRIPTOR_BYTES,
        "export result transport_descriptor",
    )
    out.extend(struct.pack("<I", 0))
    return bytes(out)


def decode_export_buffer_request(data: bytes) -> ExportBufferRequest:
    reader = _Reader(data)
    request = ExportBufferRequest(
        owner_endpoint_id=reader.i32(),
        buffer_id=reader.u64(),
        generation=reader.u64(),
        offset=reader.u64(),
        nbytes=reader.u64(),
        access_flags=reader.u32(),
        transport_profile=reader.string(MAX_TRANSPORT_PROFILE_BYTES, "EXPORT_BUFFER.transport_profile"),
    )
    if reader.u32() != 0:
        raise ValueError("remote_wire: EXPORT_BUFFER reserved field must be zero")
    reader.done("EXPORT_BUFFER")
    if request.owner_endpoint_id < 0 or request.buffer_id == 0 or request.generation == 0:
        raise ValueError("remote_wire: EXPORT_BUFFER requires live owner buffer identity")
    if request.nbytes <= 0:
        raise ValueError("remote_wire: EXPORT_BUFFER nbytes must be non-zero")
    _validate_access_flags(request.access_flags, "EXPORT_BUFFER access_flags")
    return request


def decode_export_buffer_result(data: bytes) -> ExportBufferResult:
    reader = _Reader(data)
    result = ExportBufferResult(
        owner_endpoint_id=reader.i32(),
        buffer_id=reader.u64(),
        generation=reader.u64(),
        address_space=RemoteAddressSpace(reader.u32()),
        offset=reader.u64(),
        nbytes=reader.u64(),
        export_id=reader.u64(),
        remote_addr=reader.u64(),
        rkey_or_token=reader.u64(),
        ub_ldst_va=reader.u64(),
        access_flags=reader.u32(),
        transport_profile=reader.string(MAX_TRANSPORT_PROFILE_BYTES, "export result transport_profile"),
        transport_descriptor=reader.blob(MAX_TRANSPORT_DESCRIPTOR_BYTES, "export result transport_descriptor"),
    )
    if reader.u32() != 0:
        raise ValueError("remote_wire: export result reserved field must be zero")
    reader.done("export result")
    _validate_access_flags(result.access_flags, "export result access_flags")
    if result.address_space not in (RemoteAddressSpace.REMOTE_WINDOW, RemoteAddressSpace.UB_LDST):
        raise ValueError("remote_wire: export result address_space is invalid")
    if result.nbytes <= 0 or result.export_id == 0:
        raise ValueError("remote_wire: export result requires non-zero nbytes and export_id")
    return result


def decode_import_buffer_request(data: bytes) -> ImportBufferRequest:
    reader = _Reader(data)
    importer_endpoint_id = reader.i32()
    requested_access_flags = reader.u32()
    export_start = reader.offset
    if len(data) < export_start + 4:
        raise ValueError("remote_wire: IMPORT_BUFFER payload is truncated")
    export_desc = decode_export_buffer_result(data[export_start:-4])
    reader.offset = len(data) - 4
    if reader.u32() != 0:
        raise ValueError("remote_wire: IMPORT_BUFFER reserved field must be zero")
    reader.done("IMPORT_BUFFER")
    if importer_endpoint_id < 0:
        raise ValueError("remote_wire: IMPORT_BUFFER importer endpoint must be non-negative")
    _validate_access_flags(requested_access_flags, "IMPORT_BUFFER requested_access_flags")
    if requested_access_flags & ~export_desc.access_flags:
        raise ValueError("remote_wire: IMPORT_BUFFER requested access is not a subset of export access")
    return ImportBufferRequest(importer_endpoint_id, requested_access_flags, export_desc)


def encode_import_buffer_result(result: ImportBufferResult) -> bytes:
    _validate_access_flags(result.access_flags, "import result access_flags")
    if result.address_space not in (RemoteAddressSpace.REMOTE_WINDOW, RemoteAddressSpace.UB_LDST):
        raise ValueError("remote_wire: import result address_space is invalid")
    if result.import_id == 0 or result.nbytes <= 0:
        raise ValueError("remote_wire: import result requires non-zero import_id and nbytes")
    out = bytearray()
    out.extend(
        struct.pack(
            "<iiQQQIQQQQQI",
            int(result.importer_endpoint_id),
            int(result.owner_endpoint_id),
            int(result.buffer_id),
            int(result.generation),
            int(result.import_id),
            int(result.address_space),
            int(result.offset),
            int(result.nbytes),
            int(result.remote_addr),
            int(result.rkey_or_token),
            int(result.ub_ldst_va),
            int(result.access_flags),
        )
    )
    _put_string(out, result.transport_profile, MAX_TRANSPORT_PROFILE_BYTES, "import result transport_profile")
    _put_blob(out, result.import_descriptor, MAX_TRANSPORT_DESCRIPTOR_BYTES, "import result import_descriptor")
    out.extend(struct.pack("<I", 0))
    return bytes(out)


def decode_release_import_request(data: bytes) -> ReleaseImportRequest:
    reader = _Reader(data)
    request = ReleaseImportRequest(
        importer_endpoint_id=reader.i32(),
        owner_endpoint_id=reader.i32(),
        buffer_id=reader.u64(),
        generation=reader.u64(),
        import_id=reader.u64(),
    )
    if reader.u32() != 0:
        raise ValueError("remote_wire: RELEASE_IMPORT reserved field must be zero")
    reader.done("RELEASE_IMPORT")
    if (
        request.importer_endpoint_id < 0
        or request.owner_endpoint_id < 0
        or request.buffer_id == 0
        or request.generation == 0
        or request.import_id == 0
    ):
        raise ValueError("remote_wire: RELEASE_IMPORT requires live importer and owner identity")
    return request


def encode_control_reply(
    sequence: int,
    control_name: ControlName,
    control_version: int,
    error_code: int,
    error_message: str,
    result_bytes: bytes = b"",
) -> bytes:
    msg = error_message.encode("utf-8")
    if len(msg) > MAX_ERROR_BYTES:
        msg = msg[:MAX_ERROR_BYTES]
    if len(result_bytes) > MAX_FRAME_PAYLOAD_BYTES:
        raise ValueError("remote_wire: control reply result too large")
    return (
        struct.pack("<QIIiI", int(sequence), int(control_name), int(control_version), int(error_code), len(msg))
        + msg
        + struct.pack("<I", len(result_bytes))
        + result_bytes
    )
