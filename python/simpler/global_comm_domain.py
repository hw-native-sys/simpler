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
import operator
import struct
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from _task_interface import AddressSpace, CanonicalIdentity  # pyright: ignore[reportMissingImports]

from .comm_endpoints import (
    _ADAPTER_KIND_IDS,  # noqa: F401 — re-export of the comm_endpoints numeric table
    _ADAPTER_PROFILE_IDS,  # noqa: F401 — re-export of the comm_endpoints numeric table
    AdapterKind,
    AdapterProfile,
    AttachmentRole,
    BufferAccessQuery,
    EndpointDeployment,
    EndpointIdentity,
    EndpointRecord,
    _backend_kind_for_provider,
    buffer_adapter_candidates,
)
from .comm_endpoints import (
    _adapter_kind_from_id as _neutral_adapter_kind_from_id,
)
from .comm_endpoints import (
    _adapter_kind_id as _neutral_adapter_kind_id,
)
from .comm_endpoints import (
    _adapter_profile_from_id as _neutral_adapter_profile_from_id,
)
from .comm_endpoints import (
    _adapter_profile_id as _neutral_adapter_profile_id,
)

if TYPE_CHECKING:
    from .buffer import Buffer

#: Descriptor layout. Paired with ``COMM_GLOBAL_DOMAIN_VERSION`` in
#: ``src/common/platform_comm/comm.h``: the platform backend stamps this number into every
#: descriptor it exports and ``_validate_descriptor`` checks it on decode, so advancing one
#: without the other fails every allocation at PREPARE. The ``LOCAL_*`` L3->L2 mailbox structs
#: in ``worker.py`` also carry it; producer and consumer there are the same build.
GLOBAL_DOMAIN_VERSION = 2
GLOBAL_DOMAIN_MAX_RANKS = 64
GLOBAL_DOMAIN_MAX_BUFFERS = 64
GLOBAL_DOMAIN_MAX_ATTACHMENT_ROWS = GLOBAL_DOMAIN_MAX_RANKS
GLOBAL_DOMAIN_MAX_ATTACHMENTS = GLOBAL_DOMAIN_MAX_RANKS * GLOBAL_DOMAIN_MAX_ATTACHMENT_ROWS
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

CTRL_GLOBAL_DOMAIN_PREPARE = 19
CTRL_GLOBAL_DOMAIN_IMPORT = 20
CTRL_GLOBAL_DOMAIN_RELEASE = 21
CTRL_GLOBAL_DOMAIN_COPY_TO = 22
CTRL_GLOBAL_DOMAIN_COPY_FROM = 23

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
    buffers: dict[str, Buffer] = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True)
class GlobalDomainAttachment:
    """One host/device attachment to a rank window on a receiving node.

    The target rank is the attachment's position within its command row.  The
    row is grouped by ``node_worker_id`` and has one entry for every member;
    keeping the target out of this record lets one rank's window be attached
    by multiple receiving nodes without adding a second rank identity.
    """

    node_worker_id: int
    address_space: AddressSpace
    role: AttachmentRole
    adapter_kind: AdapterKind | None
    adapter_profile: AdapterProfile | None


@dataclass(frozen=True)
class GlobalDomainBuffer:
    name: str
    nbytes: int
    dtype: str = ""
    count: int = 0
    load_from_host: bool = False
    store_to_host: bool = False


@dataclass(frozen=True)
class GlobalDomainBufferRef:
    rank: int
    slice_index: int
    identity: CanonicalIdentity
    reference_id: int = 0
    reference_generation: int = 0


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


def resolve_global_comm_capability(*, platform: str, profile: str, local_device_count: int) -> GlobalCommInitResult:
    """Return the capability implemented by the selected platform backend."""
    platform = str(platform)
    profile = str(profile)
    supported = (platform.endswith("sim") and profile == GLOBAL_DOMAIN_PROFILE_SIM) or (
        platform.startswith("a2a3") and not platform.endswith("sim") and profile == GLOBAL_DOMAIN_PROFILE_A3_FABRIC
    )
    if not supported:
        raise ValueError(f"Global CommDomain is not supported by platform {platform!r} with comm_profile {profile!r}")
    if local_device_count <= 0:
        raise ValueError("Global CommDomain capability requires at least one local device")
    return GlobalCommInitResult(
        profile=profile,
        max_ranks=GLOBAL_DOMAIN_MAX_RANKS,
        descriptor_bytes=GLOBAL_DOMAIN_DESCRIPTOR_BYTES,
        local_device_count=int(local_device_count),
    )


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
    attachments: tuple[GlobalDomainAttachment, ...] = ()
    owner_instance_id: bytes = b""
    retain_after_run: bool = False

    def attachments_for_node(self, node_worker_id: int) -> tuple[GlobalDomainAttachment, ...]:
        """Return the row addressed to one receiving L3 node.

        A command sent through MPI is intentionally identical on every rank;
        selecting the row at the receiver is therefore part of the command
        contract rather than a transport-specific fanout rule.
        """

        validate_attachment_table(self.attachments, self.members)
        node_worker_id = int(node_worker_id)
        rank_count = len(self.members)
        for start in range(0, len(self.attachments), rank_count):
            row = self.attachments[start : start + rank_count]
            if row and row[0].node_worker_id == node_worker_id:
                return row
        if self.attachments:
            raise ValueError(f"Global CommDomain has no attachment row for node worker {node_worker_id}")
        return ()


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


# AttachmentRole remains a Global CommDomain-only wire id. Adapter kind/profile numeric
# authority lives in comm_endpoints; this module only wraps those helpers with domain-local
# error text. 0 stays reserved for "no adapter".
_ATTACHMENT_ROLE_IDS = {
    AttachmentRole.PROVIDER: 1,
    AttachmentRole.CONSUMER: 2,
}
_ATTACHMENT_ROLE_BY_ID = {value: key for key, value in _ATTACHMENT_ROLE_IDS.items()}


def _assert_every_plannable_profile_is_numbered() -> None:
    """Fail import if region planning can name an ``AdapterProfile`` this wire cannot encode.

    ``AdapterProfile`` is deliberately wider than this table: the Buffer access judgment is shared
    with the per-Tensor path, which reaches mechanisms (``FORK_INHERITED_VA``, ``DEVICE_LOCAL``,
    ``OWNER_DEVICE_COPY``) that no region plan can produce and that therefore never travel in a
    ``GlobalDomainCommand``. That is a fact about ``buffer_adapter_candidates`` and
    ``_backend_kind_for_provider``, not a property either declaration states, so nothing but this
    check ties the two together — and the failure it prevents surfaces as a ``ValueError`` at
    allocation time, from a planner that had already accepted the plan.
    """

    # Drives the real `_backend_kind_for_provider` rather than restating which backends a provider
    # can name; only the deployment is read, so the rest of the record is filler.
    def _provider(deployment: EndpointDeployment) -> EndpointRecord:
        return EndpointRecord(
            identity=EndpointIdentity(session_instance_id=b"", registry_epoch=0, endpoint_id=0),
            path="L0",
            deployment=deployment,
            node_scope_id=0,
        )

    plannable = {
        candidate.profile
        for backend_kind in {_backend_kind_for_provider(_provider(deployment)) for deployment in EndpointDeployment}
        for deployment in EndpointDeployment
        for same_node in (True, False)
        # Region planning attaches consumers, never the provider itself, so the Buffer is never
        # already local to the endpoint being planned for.
        for candidate in buffer_adapter_candidates(
            BufferAccessQuery(backend_kind, deployment, same_node, same_endpoint=False)
        )
    }
    missing = sorted(profile.value for profile in plannable if profile not in _ADAPTER_PROFILE_IDS)
    if missing:
        raise AssertionError(
            f"region planning can produce adapter profiles the global-domain wire cannot number: "
            f"{missing}; add them to _ADAPTER_PROFILE_IDS (and to COMM_GLOBAL_DOMAIN_VERSION's "
            f"contract) or keep them out of buffer_adapter_candidates for a planner backend"
        )


_assert_every_plannable_profile_is_numbered()


def _attachment_role_id(role: AttachmentRole) -> int:
    try:
        return _ATTACHMENT_ROLE_IDS[AttachmentRole(role)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"global domain attachment role is unknown: {role!r}") from exc


def _attachment_role_from_id(value: int) -> AttachmentRole:
    try:
        return _ATTACHMENT_ROLE_BY_ID[int(value)]
    except KeyError as exc:
        raise ValueError(f"global domain attachment role id is unknown: {value}") from exc


def _adapter_kind_id(kind: AdapterKind | None) -> int:
    try:
        return _neutral_adapter_kind_id(kind)
    except ValueError as exc:
        raise ValueError(f"global domain attachment adapter_kind is unknown: {kind!r}") from exc


def _adapter_kind_from_id(value: int) -> AdapterKind | None:
    try:
        return _neutral_adapter_kind_from_id(value)
    except ValueError as exc:
        raise ValueError(f"global domain attachment adapter_kind id is unknown: {value}") from exc


def _adapter_profile_id(profile: AdapterProfile | None) -> int:
    try:
        return _neutral_adapter_profile_id(profile)
    except ValueError as exc:
        raise ValueError(f"global domain attachment adapter_profile is unknown: {profile!r}") from exc


def _adapter_profile_from_id(value: int) -> AdapterProfile | None:
    try:
        return _neutral_adapter_profile_from_id(value)
    except ValueError as exc:
        raise ValueError(f"global domain attachment adapter_profile id is unknown: {value}") from exc


def _normalize_attachment(attachment: GlobalDomainAttachment) -> GlobalDomainAttachment:
    try:
        address_space = AddressSpace(int(attachment.address_space))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"global domain attachment address_space is unknown: {attachment.address_space!r}") from exc
    try:
        role = AttachmentRole(attachment.role)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"global domain attachment role is unknown: {attachment.role!r}") from exc
    try:
        adapter_kind = None if attachment.adapter_kind is None else AdapterKind(attachment.adapter_kind)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"global domain attachment adapter_kind is unknown: {attachment.adapter_kind!r}") from exc
    try:
        adapter_profile = None if attachment.adapter_profile is None else AdapterProfile(attachment.adapter_profile)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"global domain attachment adapter_profile is unknown: {attachment.adapter_profile!r}"
        ) from exc
    if (adapter_kind is None) != (adapter_profile is None):
        raise ValueError("global domain attachment adapter_kind and adapter_profile must be paired")
    if int(attachment.node_worker_id) < 0:
        raise ValueError("global domain attachment node_worker_id must be non-negative")
    return GlobalDomainAttachment(
        node_worker_id=int(attachment.node_worker_id),
        address_space=address_space,
        role=role,
        adapter_kind=adapter_kind,
        adapter_profile=adapter_profile,
    )


def validate_attachment_table(
    attachments: tuple[GlobalDomainAttachment, ...], members: tuple[GlobalDomainMember, ...]
) -> None:
    """Validate the flattened receiver-node by rank attachment matrix."""

    if not attachments:
        return
    rank_count = len(members)
    if rank_count <= 0:
        raise ValueError("global domain attachment table requires members")
    if len(attachments) > GLOBAL_DOMAIN_MAX_ATTACHMENTS:
        raise ValueError("global domain attachment table exceeds maximum")
    if len(attachments) % rank_count:
        raise ValueError("global domain attachment table must contain complete rank rows")
    row_count = len(attachments) // rank_count
    if row_count > GLOBAL_DOMAIN_MAX_ATTACHMENT_ROWS:
        raise ValueError("global domain attachment row count exceeds maximum")
    seen_nodes: set[int] = set()
    for start in range(0, len(attachments), rank_count):
        row = tuple(_normalize_attachment(item) for item in attachments[start : start + rank_count])
        row_nodes = {item.node_worker_id for item in row}
        if len(row_nodes) != 1:
            raise ValueError("global domain attachment row must target one node worker")
        node_worker_id = next(iter(row_nodes))
        if node_worker_id in seen_nodes:
            raise ValueError("global domain attachment table contains a duplicate node row")
        seen_nodes.add(node_worker_id)


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


def _put_attachment(out: bytearray, attachment: GlobalDomainAttachment) -> None:
    attachment = _normalize_attachment(attachment)
    out.extend(
        struct.pack(
            "<iIIII",
            attachment.node_worker_id,
            int(attachment.address_space),
            _attachment_role_id(attachment.role),
            _adapter_kind_id(attachment.adapter_kind),
            _adapter_profile_id(attachment.adapter_profile),
        )
    )


def _read_attachment(reader: _Reader) -> GlobalDomainAttachment:
    node_worker_id, address_space, role, adapter_kind, adapter_profile = struct.unpack(
        "<iIIII", reader._take(struct.calcsize("<iIIII"), "attachment")
    )
    attachment = GlobalDomainAttachment(
        node_worker_id=int(node_worker_id),
        address_space=AddressSpace(int(address_space)),
        role=_attachment_role_from_id(int(role)),
        adapter_kind=_adapter_kind_from_id(int(adapter_kind)),
        adapter_profile=_adapter_profile_from_id(int(adapter_profile)),
    )
    return _normalize_attachment(attachment)


def encode_comm_init(command: GlobalCommInitCommand) -> bytes:
    validate_member_table(command.members)
    if not command.cluster_id or not command.topology_hash:
        raise ValueError("global comm init cluster_id and topology_hash must be non-empty")
    if command.profile not in GLOBAL_DOMAIN_PROFILE_IDS:
        raise ValueError(f"unsupported global domain profile {command.profile!r}")
    if command.node_rank < 0 or command.node_count <= 0 or command.node_rank >= command.node_count:
        raise ValueError("global comm init node identity is invalid")
    out = bytearray(struct.pack("<II", command.node_rank, command.node_count))
    _put_string(out, command.cluster_id, "cluster_id")
    _put_string(out, command.topology_hash, "topology_hash")
    _put_string(out, command.profile, "profile")
    out.extend(struct.pack("<I", len(command.members)))
    for member in command.members:
        _put_member(out, member)
    return bytes(out)


def decode_comm_init(data: bytes) -> GlobalCommInitCommand:
    reader = _Reader(data)
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


def encode_domain_command(command: GlobalDomainCommand) -> bytes:  # noqa: PLR0912 -- wire validation keeps phase rules adjacent to encoding
    validate_member_table(command.members)
    validate_attachment_table(command.attachments, command.members)
    if command.phase is not GlobalDomainPhase.PREPARE_EXPORT and command.attachments:
        raise ValueError("global domain attachments are only carried by PREPARE_EXPORT")
    if command.domain_id == 0 or command.generation == 0 or command.window_size <= 0:
        raise ValueError("global domain command identity and window_size must be positive")
    if not command.name:
        raise ValueError("global domain command name must be non-empty")
    if command.profile not in GLOBAL_DOMAIN_PROFILE_IDS:
        raise ValueError(f"unsupported global domain profile {command.profile!r}")
    if len(command.buffers) > GLOBAL_DOMAIN_MAX_BUFFERS:
        raise ValueError("global domain command buffer count exceeds maximum")
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
            "<IQQQ",
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
    out.extend(struct.pack("<I", len(command.attachments)))
    for attachment in command.attachments:
        _put_attachment(out, attachment)
    out.extend(struct.pack("<I", len(command.buffers)))
    for buffer in command.buffers:
        _put_string(out, buffer.name, "buffer.name")
        out.extend(struct.pack("<Q", int(buffer.nbytes)))
    out.extend(struct.pack("<I", len(command.descriptors)))
    for descriptor in command.descriptors:
        out.extend(descriptor.encode())
    if command.owner_instance_id:
        if len(command.owner_instance_id) != 8 or not any(command.owner_instance_id):
            raise ValueError("global domain creator must be a nonzero 8-byte instance identity")
        out.extend(b"SDB2" + command.owner_instance_id)
        out.extend(struct.pack("<?", command.retain_after_run))
        for buffer in command.buffers:
            _put_string(out, buffer.dtype, "buffer.dtype")
            out.extend(struct.pack("<q??", buffer.count, buffer.load_from_host, buffer.store_to_host))
    elif command.retain_after_run or any(
        b.dtype or b.count or b.load_from_host or b.store_to_host for b in command.buffers
    ):
        raise ValueError("global domain extended layout requires a creator identity")
    return bytes(out)


def decode_domain_command(data: bytes) -> GlobalDomainCommand:
    reader = _Reader(data)
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
    attachment_count = reader.u32()
    if attachment_count > GLOBAL_DOMAIN_MAX_ATTACHMENTS:
        raise ValueError("global domain command attachment count exceeds maximum")
    attachments = tuple(_read_attachment(reader) for _ in range(attachment_count))
    buffer_count = reader.u32()
    if buffer_count > GLOBAL_DOMAIN_MAX_BUFFERS:
        raise ValueError("global domain command buffer count exceeds maximum")
    buffers = tuple(GlobalDomainBuffer(reader.string("buffer.name"), reader.u64()) for _ in range(buffer_count))
    descriptor_count = reader.u32()
    if descriptor_count > GLOBAL_DOMAIN_MAX_RANKS:
        raise ValueError("global domain command descriptor count exceeds maximum")
    descriptors = tuple(
        GlobalDomainDescriptor.decode(reader.fixed(GLOBAL_DOMAIN_DESCRIPTOR_BYTES, "descriptor"))
        for _ in range(descriptor_count)
    )
    owner_instance_id = b""
    retain_after_run = False
    if reader._offset != len(data):
        if reader.fixed(4, "Buffer extension magic") != b"SDB2":
            raise ValueError("global domain Buffer extension is unknown")
        owner_instance_id = reader.fixed(8, "creator identity")
        retention = reader.fixed(1, "retention policy")
        if retention not in (b"\x00", b"\x01"):
            raise ValueError("global domain retention policy is invalid")
        retain_after_run = retention == b"\x01"
        definitions = []
        for buffer in buffers:
            dtype = reader.string("buffer.dtype")
            count, load, store = struct.unpack("<q??", reader.fixed(10, "buffer definition"))
            definitions.append(replace(buffer, dtype=dtype, count=count, load_from_host=load, store_to_host=store))
        buffers = tuple(definitions)
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
        attachments=attachments,
        owner_instance_id=owner_instance_id,
        retain_after_run=retain_after_run,
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


_DOMAIN_BUFFER_REF = struct.Struct("<II8sQIQQ")


def encode_domain_buffer_refs(references: tuple[GlobalDomainBufferRef, ...]) -> bytes:
    """Encode a node's committed slice identities and optional session locators, without addresses."""
    if len(references) > GLOBAL_DOMAIN_MAX_RANKS * GLOBAL_DOMAIN_MAX_BUFFERS:
        raise ValueError("global domain Buffer reference count exceeds maximum")
    out = bytearray(b"SBR1" + struct.pack("<I", len(references)))
    slices, identities, locators = set(), set(), set()
    for reference in references:
        rank = operator.index(reference.rank)
        index = operator.index(reference.slice_index)
        identity = reference.identity
        owner = bytes(identity.owner_instance_id)
        buffer_id, generation = int(identity.buffer_id), int(identity.generation)
        reference_id = operator.index(reference.reference_id)
        reference_generation = operator.index(reference.reference_generation)
        if not 0 <= rank < GLOBAL_DOMAIN_MAX_RANKS or not 0 <= index < GLOBAL_DOMAIN_MAX_BUFFERS:
            raise ValueError("global domain Buffer reference rank or slice is out of range")
        if len(owner) != 8 or not any(owner) or not 0 < buffer_id < 2**64 or not 0 < generation < 2**32:
            raise ValueError("global domain Buffer reference has an invalid canonical identity")
        if not 0 <= reference_id < 2**64 or not 0 <= reference_generation < 2**64:
            raise ValueError("global domain Buffer reference locator is out of range")
        if bool(reference_id) != bool(reference_generation):
            raise ValueError("global domain Buffer reference locator must be zero or a positive pair")
        locator = reference_id, reference_generation
        if (rank, index) in slices or identity in identities or (reference_id and locator in locators):
            raise ValueError("global domain Buffer references contain a duplicate slice, identity or locator")
        slices.add((rank, index))
        identities.add(identity)
        locators.add(locator)
        out.extend(_DOMAIN_BUFFER_REF.pack(rank, index, owner, buffer_id, generation, *locator))
    return bytes(out)


def decode_domain_buffer_refs(data: bytes) -> tuple[GlobalDomainBufferRef, ...]:
    reader = _Reader(data)
    if reader.fixed(4, "Buffer reference magic") != b"SBR1":
        raise ValueError("global domain Buffer reference format is unknown")
    count = reader.u32()
    if count > GLOBAL_DOMAIN_MAX_RANKS * GLOBAL_DOMAIN_MAX_BUFFERS:
        raise ValueError("global domain Buffer reference count exceeds maximum")
    references = []
    for _ in range(count):
        rank, index, owner, buffer_id, generation, reference_id, reference_generation = _DOMAIN_BUFFER_REF.unpack(
            reader.fixed(_DOMAIN_BUFFER_REF.size, "Buffer reference")
        )
        references.append(
            GlobalDomainBufferRef(
                rank,
                index,
                CanonicalIdentity(owner, buffer_id, generation),
                reference_id,
                reference_generation,
            )
        )
    reader.done("Buffer references")
    result = tuple(references)
    encode_domain_buffer_refs(result)
    return result


def encode_release_command(command: GlobalDomainReleaseCommand) -> bytes:
    if command.domain_id == 0 or command.generation == 0:
        raise ValueError("global domain release identity must be positive")
    return struct.pack("<QQ", int(command.domain_id), int(command.generation))


def decode_release_command(data: bytes) -> GlobalDomainReleaseCommand:
    if len(data) != struct.calcsize("<QQ"):
        raise ValueError("global domain release size mismatch")
    domain_id, generation = struct.unpack("<QQ", data)
    if domain_id == 0 or generation == 0:
        raise ValueError("global domain release identity is invalid")
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
            "<QQIQQ",
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
    header_size = struct.calcsize("<QQIQQ")
    if len(data) < header_size:
        raise ValueError("global domain copy request is truncated")
    domain_id, generation, domain_rank, offset, nbytes = struct.unpack_from("<QQIQQ", data)
    payload = data[header_size:]
    command = GlobalDomainCopyCommand(
        domain_id=int(domain_id),
        generation=int(generation),
        domain_rank=int(domain_rank),
        offset=int(offset),
        nbytes=int(nbytes),
        data=bytes(payload),
    )
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
