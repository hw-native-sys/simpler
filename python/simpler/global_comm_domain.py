# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Versioned codecs and data models for L4-brokered Global CommDomains."""

from __future__ import annotations

import enum
import struct
from dataclasses import dataclass

GLOBAL_DOMAIN_VERSION = 1
GLOBAL_DOMAIN_MAX_RANKS = 64
GLOBAL_DOMAIN_HANDLE_BYTES = 256
GLOBAL_DOMAIN_DESCRIPTOR = struct.Struct("<IIIIQII256s")
GLOBAL_DOMAIN_DESCRIPTOR_BYTES = GLOBAL_DOMAIN_DESCRIPTOR.size
GLOBAL_DOMAIN_PROFILE_SIM = "sim"
GLOBAL_DOMAIN_PROFILE_A3_FABRIC = "a3-fabric-v1"
GLOBAL_DOMAIN_PROFILE_IDS = {
    GLOBAL_DOMAIN_PROFILE_SIM: 1,
    GLOBAL_DOMAIN_PROFILE_A3_FABRIC: 2,
}
GLOBAL_DOMAIN_MAX_STRING_BYTES = 1024
GLOBAL_DOMAIN_MAX_COPY_BYTES = 8 * 1024 * 1024

CTRL_GLOBAL_DOMAIN_PREPARE = 18
CTRL_GLOBAL_DOMAIN_IMPORT = 19
CTRL_GLOBAL_DOMAIN_RELEASE = 20
CTRL_GLOBAL_DOMAIN_COPY_TO = 21
CTRL_GLOBAL_DOMAIN_COPY_FROM = 22

LOCAL_DOMAIN_MAGIC = b"SGD1"
LOCAL_PREPARE_REQUEST = struct.Struct("<4sIQQIIIQ")
LOCAL_PREPARE_REPLY = struct.Struct("<4sIQQQQ")
LOCAL_IMPORT_REQUEST = struct.Struct("<4sIQQI")
LOCAL_IMPORT_REPLY = struct.Struct("<4sIQQQQQ")
LOCAL_RELEASE_REQUEST = struct.Struct("<4sIQQ")
LOCAL_COPY_REQUEST = struct.Struct("<4sIQQQQ")
LOCAL_COPY_REPLY = struct.Struct("<4sIQQQ")


class GlobalDomainPhase(enum.IntEnum):
    PREPARE_EXPORT = 1
    IMPORT = 2
    COMMIT = 3
    ABORT = 4


@dataclass(frozen=True)
class GlobalDomainMember:
    node_worker_id: int
    local_worker_id: int
    global_device_rank: int
    domain_rank: int


@dataclass(frozen=True)
class GlobalDomainBuffer:
    name: str
    nbytes: int


@dataclass(frozen=True)
class GlobalDomainDescriptor:
    version: int
    profile_id: int
    domain_rank: int
    rank_count: int
    mapping_size: int
    handle: bytes

    def encode(self) -> bytes:
        _validate_descriptor(self)
        handle = bytes(self.handle)
        return GLOBAL_DOMAIN_DESCRIPTOR.pack(
            int(self.version),
            int(self.profile_id),
            int(self.domain_rank),
            int(self.rank_count),
            int(self.mapping_size),
            len(handle),
            0,
            handle + b"\x00" * (GLOBAL_DOMAIN_HANDLE_BYTES - len(handle)),
        )

    @classmethod
    def decode(cls, data: bytes) -> GlobalDomainDescriptor:
        if len(data) != GLOBAL_DOMAIN_DESCRIPTOR_BYTES:
            raise ValueError("global domain descriptor size mismatch")
        version, profile_id, domain_rank, rank_count, mapping_size, handle_size, reserved, handle = (
            GLOBAL_DOMAIN_DESCRIPTOR.unpack(data)
        )
        if reserved != 0:
            raise ValueError("global domain descriptor reserved field must be zero")
        if handle_size > GLOBAL_DOMAIN_HANDLE_BYTES:
            raise ValueError("global domain descriptor handle size exceeds maximum")
        descriptor = cls(
            version=int(version),
            profile_id=int(profile_id),
            domain_rank=int(domain_rank),
            rank_count=int(rank_count),
            mapping_size=int(mapping_size),
            handle=bytes(handle[:handle_size]),
        )
        _validate_descriptor(descriptor)
        return descriptor


@dataclass(frozen=True)
class GlobalCommInitCommand:
    cluster_id: str
    topology_hash: str
    profile: str
    node_rank: int
    node_count: int
    members: tuple[GlobalDomainMember, ...]


@dataclass(frozen=True)
class GlobalCommInitResult:
    profile: str
    max_ranks: int
    descriptor_bytes: int
    local_device_count: int


@dataclass(frozen=True)
class GlobalDomainCommand:
    phase: GlobalDomainPhase
    domain_id: int
    generation: int
    name: str
    profile: str
    window_size: int
    members: tuple[GlobalDomainMember, ...]
    buffers: tuple[GlobalDomainBuffer, ...]
    descriptors: tuple[GlobalDomainDescriptor, ...] = ()


@dataclass(frozen=True)
class GlobalDomainReleaseCommand:
    domain_id: int
    generation: int


@dataclass(frozen=True)
class GlobalDomainCopyCommand:
    domain_id: int
    generation: int
    domain_rank: int
    offset: int
    nbytes: int
    data: bytes = b""


class _Reader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    def _take(self, size: int, field: str) -> bytes:
        if size < 0 or self._offset > len(self._data) or size > len(self._data) - self._offset:
            raise ValueError(f"global domain wire truncated {field}")
        result = self._data[self._offset : self._offset + size]
        self._offset += size
        return result

    def u32(self) -> int:
        return int(struct.unpack("<I", self._take(4, "uint32"))[0])

    def i32(self) -> int:
        return int(struct.unpack("<i", self._take(4, "int32"))[0])

    def u64(self) -> int:
        return int(struct.unpack("<Q", self._take(8, "uint64"))[0])

    def string(self, field: str) -> str:
        size = self.u32()
        if size > GLOBAL_DOMAIN_MAX_STRING_BYTES:
            raise ValueError(f"global domain wire {field} exceeds maximum")
        return self._take(size, field).decode("utf-8")

    def blob(self, maximum: int, field: str) -> bytes:
        size = self.u32()
        if size > maximum:
            raise ValueError(f"global domain wire {field} exceeds maximum")
        return self._take(size, field)

    def fixed(self, size: int, field: str) -> bytes:
        return self._take(size, field)

    def done(self, field: str) -> None:
        if self._offset != len(self._data):
            raise ValueError(f"global domain wire trailing bytes after {field}")


def _put_string(out: bytearray, value: str, field: str) -> None:
    encoded = str(value).encode("utf-8")
    if len(encoded) > GLOBAL_DOMAIN_MAX_STRING_BYTES:
        raise ValueError(f"global domain wire {field} exceeds maximum")
    out.extend(struct.pack("<I", len(encoded)))
    out.extend(encoded)


def _put_blob(out: bytearray, value: bytes, maximum: int, field: str) -> None:
    encoded = bytes(value)
    if len(encoded) > maximum:
        raise ValueError(f"global domain wire {field} exceeds maximum")
    out.extend(struct.pack("<I", len(encoded)))
    out.extend(encoded)


def _validate_descriptor(descriptor: GlobalDomainDescriptor) -> None:
    if descriptor.version != GLOBAL_DOMAIN_VERSION:
        raise ValueError("global domain descriptor version mismatch")
    if descriptor.profile_id not in GLOBAL_DOMAIN_PROFILE_IDS.values():
        raise ValueError("global domain descriptor profile is unknown")
    if descriptor.rank_count <= 0 or descriptor.rank_count > GLOBAL_DOMAIN_MAX_RANKS:
        raise ValueError("global domain descriptor rank_count is invalid")
    if descriptor.domain_rank < 0 or descriptor.domain_rank >= descriptor.rank_count:
        raise ValueError("global domain descriptor domain_rank is invalid")
    if descriptor.mapping_size <= 0:
        raise ValueError("global domain descriptor mapping_size must be positive")
    if not descriptor.handle or len(descriptor.handle) > GLOBAL_DOMAIN_HANDLE_BYTES:
        raise ValueError("global domain descriptor handle size is invalid")


def validate_member_table(members: tuple[GlobalDomainMember, ...]) -> None:
    if not members or len(members) > GLOBAL_DOMAIN_MAX_RANKS:
        raise ValueError("global domain members must contain between 1 and 64 devices")
    ranks = [member.domain_rank for member in members]
    if ranks != list(range(len(members))):
        raise ValueError("global domain members must be in dense domain-rank order")
    devices = [(member.node_worker_id, member.local_worker_id) for member in members]
    if len(set(devices)) != len(devices):
        raise ValueError("global domain members contain duplicate node/local devices")
    if any(node < 0 or local < 0 for node, local in devices):
        raise ValueError("global domain member node/local ids must be non-negative")
    global_ranks = [member.global_device_rank for member in members]
    if len(set(global_ranks)) != len(global_ranks) or any(rank < 0 for rank in global_ranks):
        raise ValueError("global domain members require unique non-negative global device ranks")


def validate_descriptor_table(
    descriptors: tuple[GlobalDomainDescriptor, ...], *, rank_count: int, profile: str
) -> None:
    if profile not in GLOBAL_DOMAIN_PROFILE_IDS:
        raise ValueError(f"unsupported global domain profile {profile!r}")
    if len(descriptors) != rank_count:
        raise ValueError("global domain descriptor table is incomplete")
    expected_profile = GLOBAL_DOMAIN_PROFILE_IDS[profile]
    ranks: set[int] = set()
    mapping_size: int | None = None
    for descriptor in descriptors:
        _validate_descriptor(descriptor)
        if descriptor.profile_id != expected_profile or descriptor.rank_count != rank_count:
            raise ValueError("global domain descriptor profile or rank_count mismatch")
        if descriptor.domain_rank in ranks:
            raise ValueError("global domain descriptor table contains a duplicate rank")
        ranks.add(descriptor.domain_rank)
        if mapping_size is None:
            mapping_size = descriptor.mapping_size
        elif mapping_size != descriptor.mapping_size:
            raise ValueError("global domain descriptor mapping sizes differ")
    if ranks != set(range(rank_count)):
        raise ValueError("global domain descriptor table has missing ranks")


def _put_member(out: bytearray, member: GlobalDomainMember) -> None:
    out.extend(
        struct.pack(
            "<iIII",
            int(member.node_worker_id),
            int(member.local_worker_id),
            int(member.global_device_rank),
            int(member.domain_rank),
        )
    )


def _read_member(reader: _Reader) -> GlobalDomainMember:
    return GlobalDomainMember(
        node_worker_id=reader.i32(),
        local_worker_id=reader.u32(),
        global_device_rank=reader.u32(),
        domain_rank=reader.u32(),
    )


def encode_comm_init(command: GlobalCommInitCommand) -> bytes:
    validate_member_table(command.members)
    if not command.cluster_id or not command.topology_hash:
        raise ValueError("global comm init cluster_id and topology_hash must be non-empty")
    if command.profile not in GLOBAL_DOMAIN_PROFILE_IDS:
        raise ValueError(f"unsupported global domain profile {command.profile!r}")
    if command.node_rank < 0 or command.node_count <= 0 or command.node_rank >= command.node_count:
        raise ValueError("global comm init node identity is invalid")
    out = bytearray(struct.pack("<III", GLOBAL_DOMAIN_VERSION, command.node_rank, command.node_count))
    _put_string(out, command.cluster_id, "cluster_id")
    _put_string(out, command.topology_hash, "topology_hash")
    _put_string(out, command.profile, "profile")
    out.extend(struct.pack("<I", len(command.members)))
    for member in command.members:
        _put_member(out, member)
    return bytes(out)


def decode_comm_init(data: bytes) -> GlobalCommInitCommand:
    reader = _Reader(data)
    version = reader.u32()
    if version != GLOBAL_DOMAIN_VERSION:
        raise ValueError("global comm init version mismatch")
    node_rank = reader.u32()
    node_count = reader.u32()
    cluster_id = reader.string("cluster_id")
    topology_hash = reader.string("topology_hash")
    profile = reader.string("profile")
    member_count = reader.u32()
    if member_count > GLOBAL_DOMAIN_MAX_RANKS:
        raise ValueError("global comm init member count exceeds maximum")
    members = tuple(_read_member(reader) for _ in range(member_count))
    reader.done("COMM_INIT")
    command = GlobalCommInitCommand(cluster_id, topology_hash, profile, node_rank, node_count, members)
    validate_member_table(command.members)
    if not command.cluster_id or not command.topology_hash:
        raise ValueError("global comm init cluster_id and topology_hash must be non-empty")
    if command.profile not in GLOBAL_DOMAIN_PROFILE_IDS:
        raise ValueError(f"unsupported global domain profile {command.profile!r}")
    if node_count <= 0 or node_rank >= node_count:
        raise ValueError("global comm init node identity is invalid")
    return command


def encode_comm_init_result(result: GlobalCommInitResult) -> bytes:
    out = bytearray(
        struct.pack(
            "<III",
            int(result.max_ranks),
            int(result.descriptor_bytes),
            int(result.local_device_count),
        )
    )
    _put_string(out, result.profile, "profile")
    return bytes(out)


def decode_comm_init_result(data: bytes) -> GlobalCommInitResult:
    reader = _Reader(data)
    result = GlobalCommInitResult(
        profile="",
        max_ranks=reader.u32(),
        descriptor_bytes=reader.u32(),
        local_device_count=reader.u32(),
    )
    result = GlobalCommInitResult(
        profile=reader.string("profile"),
        max_ranks=result.max_ranks,
        descriptor_bytes=result.descriptor_bytes,
        local_device_count=result.local_device_count,
    )
    reader.done("COMM_INIT result")
    return result


def encode_domain_command(command: GlobalDomainCommand) -> bytes:
    validate_member_table(command.members)
    if command.domain_id == 0 or command.generation == 0 or command.window_size <= 0:
        raise ValueError("global domain command identity and window_size must be positive")
    if not command.name:
        raise ValueError("global domain command name must be non-empty")
    if command.profile not in GLOBAL_DOMAIN_PROFILE_IDS:
        raise ValueError(f"unsupported global domain profile {command.profile!r}")
    if len({buffer.name for buffer in command.buffers}) != len(command.buffers):
        raise ValueError("global domain command contains duplicate buffer names")
    if any(not buffer.name or buffer.nbytes <= 0 for buffer in command.buffers):
        raise ValueError("global domain buffers require a name and positive size")
    if sum(buffer.nbytes for buffer in command.buffers) > command.window_size:
        raise ValueError("global domain buffers exceed the requested window")
    if command.descriptors:
        validate_descriptor_table(command.descriptors, rank_count=len(command.members), profile=command.profile)
    if command.phase in (GlobalDomainPhase.IMPORT, GlobalDomainPhase.COMMIT):
        if len(command.descriptors) != len(command.members):
            raise ValueError("global domain IMPORT/COMMIT requires a complete descriptor table")
    elif command.descriptors:
        raise ValueError("global domain PREPARE/ABORT must not carry descriptors")

    out = bytearray(
        struct.pack(
            "<IIQQQ",
            GLOBAL_DOMAIN_VERSION,
            int(command.phase),
            int(command.domain_id),
            int(command.generation),
            int(command.window_size),
        )
    )
    _put_string(out, command.name, "name")
    _put_string(out, command.profile, "profile")
    out.extend(struct.pack("<I", len(command.members)))
    for member in command.members:
        _put_member(out, member)
    out.extend(struct.pack("<I", len(command.buffers)))
    for buffer in command.buffers:
        _put_string(out, buffer.name, "buffer.name")
        out.extend(struct.pack("<Q", int(buffer.nbytes)))
    out.extend(struct.pack("<I", len(command.descriptors)))
    for descriptor in command.descriptors:
        out.extend(descriptor.encode())
    return bytes(out)


def decode_domain_command(data: bytes) -> GlobalDomainCommand:
    reader = _Reader(data)
    version = reader.u32()
    if version != GLOBAL_DOMAIN_VERSION:
        raise ValueError("global domain command version mismatch")
    try:
        phase = GlobalDomainPhase(reader.u32())
    except ValueError as exc:
        raise ValueError("global domain command phase is unknown") from exc
    domain_id = reader.u64()
    generation = reader.u64()
    window_size = reader.u64()
    name = reader.string("name")
    profile = reader.string("profile")
    member_count = reader.u32()
    if member_count > GLOBAL_DOMAIN_MAX_RANKS:
        raise ValueError("global domain command member count exceeds maximum")
    members = tuple(_read_member(reader) for _ in range(member_count))
    buffer_count = reader.u32()
    if buffer_count > GLOBAL_DOMAIN_MAX_RANKS:
        raise ValueError("global domain command buffer count exceeds maximum")
    buffers = tuple(GlobalDomainBuffer(reader.string("buffer.name"), reader.u64()) for _ in range(buffer_count))
    descriptor_count = reader.u32()
    if descriptor_count > GLOBAL_DOMAIN_MAX_RANKS:
        raise ValueError("global domain command descriptor count exceeds maximum")
    descriptors = tuple(
        GlobalDomainDescriptor.decode(reader.fixed(GLOBAL_DOMAIN_DESCRIPTOR_BYTES, "descriptor"))
        for _ in range(descriptor_count)
    )
    reader.done("ALLOC_DOMAIN")
    command = GlobalDomainCommand(
        phase=phase,
        domain_id=domain_id,
        generation=generation,
        name=name,
        profile=profile,
        window_size=window_size,
        members=members,
        buffers=buffers,
        descriptors=descriptors,
    )
    encode_domain_command(command)
    return command


def encode_descriptor_table(descriptors: tuple[GlobalDomainDescriptor, ...]) -> bytes:
    if len(descriptors) > GLOBAL_DOMAIN_MAX_RANKS:
        raise ValueError("global domain descriptor table exceeds maximum")
    return struct.pack("<I", len(descriptors)) + b"".join(descriptor.encode() for descriptor in descriptors)


def decode_descriptor_table(data: bytes) -> tuple[GlobalDomainDescriptor, ...]:
    reader = _Reader(data)
    count = reader.u32()
    if count > GLOBAL_DOMAIN_MAX_RANKS:
        raise ValueError("global domain descriptor table exceeds maximum")
    descriptors = tuple(
        GlobalDomainDescriptor.decode(reader.fixed(GLOBAL_DOMAIN_DESCRIPTOR_BYTES, "descriptor")) for _ in range(count)
    )
    reader.done("descriptor table")
    return descriptors


def encode_release_command(command: GlobalDomainReleaseCommand) -> bytes:
    if command.domain_id == 0 or command.generation == 0:
        raise ValueError("global domain release identity must be positive")
    return struct.pack("<IQQ", GLOBAL_DOMAIN_VERSION, int(command.domain_id), int(command.generation))


def decode_release_command(data: bytes) -> GlobalDomainReleaseCommand:
    if len(data) != struct.calcsize("<IQQ"):
        raise ValueError("global domain release size mismatch")
    version, domain_id, generation = struct.unpack("<IQQ", data)
    if version != GLOBAL_DOMAIN_VERSION or domain_id == 0 or generation == 0:
        raise ValueError("global domain release identity or version is invalid")
    return GlobalDomainReleaseCommand(int(domain_id), int(generation))


def encode_copy_command(command: GlobalDomainCopyCommand, *, include_data: bool) -> bytes:
    if (
        command.domain_id == 0
        or command.generation == 0
        or command.domain_rank < 0
        or command.offset < 0
        or command.nbytes <= 0
        or command.nbytes > GLOBAL_DOMAIN_MAX_COPY_BYTES
    ):
        raise ValueError("global domain copy fields are invalid")
    if include_data and len(command.data) != command.nbytes:
        raise ValueError("global domain copy payload size mismatch")
    if not include_data and command.data:
        raise ValueError("global domain copy-from request must not contain data")
    out = bytearray(
        struct.pack(
            "<IQQIQQ",
            GLOBAL_DOMAIN_VERSION,
            int(command.domain_id),
            int(command.generation),
            int(command.domain_rank),
            int(command.offset),
            int(command.nbytes),
        )
    )
    if include_data:
        out.extend(command.data)
    return bytes(out)


def decode_copy_command(data: bytes, *, include_data: bool) -> GlobalDomainCopyCommand:
    header_size = struct.calcsize("<IQQIQQ")
    if len(data) < header_size:
        raise ValueError("global domain copy request is truncated")
    version, domain_id, generation, domain_rank, offset, nbytes = struct.unpack_from("<IQQIQQ", data)
    payload = data[header_size:]
    command = GlobalDomainCopyCommand(
        domain_id=int(domain_id),
        generation=int(generation),
        domain_rank=int(domain_rank),
        offset=int(offset),
        nbytes=int(nbytes),
        data=bytes(payload),
    )
    if version != GLOBAL_DOMAIN_VERSION:
        raise ValueError("global domain copy version mismatch")
    encode_copy_command(command, include_data=include_data)
    return command


def encode_copy_result(data: bytes) -> bytes:
    out = bytearray()
    _put_blob(out, data, GLOBAL_DOMAIN_MAX_COPY_BYTES, "copy result")
    return bytes(out)


def decode_copy_result(data: bytes) -> bytes:
    reader = _Reader(data)
    result = reader.blob(GLOBAL_DOMAIN_MAX_COPY_BYTES, "copy result")
    reader.done("copy result")
    return result
