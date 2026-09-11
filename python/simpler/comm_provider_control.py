# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Delegated-region control wire (DRCT v2) and terminal transaction table.

This module owns the delegated-region envelope codec. Physical allocation, release, cleanup
ledger, and sweep authority stay in ProviderRegionStore.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Protocol, TypeVar

from _task_interface import (  # pyright: ignore[reportMissingImports]
    AccessMode,
    BackendKind,
    BufferDescriptor,
    _decode_buffer_descriptor_wire,
    _encode_buffer_descriptor_wire,
)

from .comm_endpoints import (
    AdapterKind,
    AdapterProfile,
    EndpointDeploymentKind,
    EndpointPathSegment,
    RegionTopologyKind,
    _adapter_kind_from_id,
    _adapter_kind_id,
    _adapter_profile_from_id,
    _adapter_profile_id,
    parse_endpoint_path,
)
from .comm_provider import (
    ProviderCleanupFailure,
    ProviderReleaseResult,
    ProviderReleaseStatus,
    RegionAllocationError,
    RegionAllocationResult,
    RegionAllocationSpec,
    RegionCleanupCause,
    RegionControlError,
    RegionControlErrorKind,
    RegionExportDescriptor,
    RegionOperationKind,
    RegionPartAllocationSpec,
    RegionPartKind,
    RegionPartLocalView,
    validate_independent_local_views,
)

DELEGATED_REGION_CTRL_MAGIC = 0x44524354
DELEGATED_REGION_CTRL_ABI_MAJOR = 2
DELEGATED_REGION_CTRL_ABI_MINOR = 0
DELEGATED_REGION_CTRL_MAGIC_VERSION = (
    (DELEGATED_REGION_CTRL_MAGIC << 32) | (DELEGATED_REGION_CTRL_ABI_MAJOR << 16) | DELEGATED_REGION_CTRL_ABI_MINOR
)

REQUEST_HEADER_BYTES = 40
ALLOCATE_PROJECTION_BYTES = 64
ALLOCATE_REQUEST_HARD_CEILING = 616
RELEASE_REQUEST_HARD_CEILING = 296
REPLY_HEADER_BYTES = 40
REPLY_TAG_OFFSET = 12
ALLOCATE_REPLY_BYTES = 288
RELEASE_REPLY_BYTES = 72
PATH_CEILING_BYTES = 256
BUFFER_DESCRIPTOR_WIRE_BYTES = 88
LOCAL_VIEW_BYTES = 24
ALLOCATE_OUTCOME_BYTES = 248
RELEASE_OUTCOME_BYTES = 32
ALLOCATE_OUTCOME_OFFSET = REPLY_HEADER_BYTES
ALLOCATE_PAYLOAD_DESCRIPTOR_OFFSET = ALLOCATE_OUTCOME_OFFSET + 24
ALLOCATE_COUNTER_DESCRIPTOR_OFFSET = ALLOCATE_PAYLOAD_DESCRIPTOR_OFFSET + BUFFER_DESCRIPTOR_WIRE_BYTES
ALLOCATE_PAYLOAD_VIEW_OFFSET = ALLOCATE_COUNTER_DESCRIPTOR_OFFSET + BUFFER_DESCRIPTOR_WIRE_BYTES
ALLOCATE_COUNTER_VIEW_OFFSET = ALLOCATE_PAYLOAD_VIEW_OFFSET + LOCAL_VIEW_BYTES
RELEASE_OUTCOME_OFFSET = REPLY_HEADER_BYTES
FAILURE_MASK_PAYLOAD = 1
FAILURE_MASK_COUNTER = 2

_REQUEST_HEADER = struct.Struct("<QII8sQII")
_ALLOCATE_PROJECTION = struct.Struct("<IIIIQIIIIQIIII")
_REPLY_HEADER = struct.Struct("<QIIII8sQ")
_REPLY_TAG = struct.Struct("<I")
_ALLOCATE_OUTCOME_PREFIX = struct.Struct("<QIIII")
_LOCAL_VIEW = struct.Struct("<IIQQ")
_RELEASE_OUTCOME = struct.Struct("<QIIIIII")
_PATH_ROOT_RE = re.compile(r"^L([0-9]+)$")
_UINT64_MAX = (1 << 64) - 1
_BACKEND_KINDS = {int(kind): kind for kind in BackendKind}

assert DELEGATED_REGION_CTRL_MAGIC_VERSION == 0x4452435400020000
assert _REQUEST_HEADER.size == REQUEST_HEADER_BYTES
assert _ALLOCATE_PROJECTION.size == ALLOCATE_PROJECTION_BYTES
assert _REPLY_HEADER.size == REPLY_HEADER_BYTES
assert _ALLOCATE_OUTCOME_PREFIX.size == 24
assert _LOCAL_VIEW.size == LOCAL_VIEW_BYTES
assert _RELEASE_OUTCOME.size == RELEASE_OUTCOME_BYTES
assert ALLOCATE_PAYLOAD_DESCRIPTOR_OFFSET == 64
assert ALLOCATE_COUNTER_DESCRIPTOR_OFFSET == 152
assert ALLOCATE_PAYLOAD_VIEW_OFFSET == 240
assert ALLOCATE_COUNTER_VIEW_OFFSET == 264
assert ALLOCATE_COUNTER_VIEW_OFFSET + LOCAL_VIEW_BYTES == ALLOCATE_REPLY_BYTES
assert REPLY_HEADER_BYTES + ALLOCATE_OUTCOME_BYTES == ALLOCATE_REPLY_BYTES
assert REPLY_HEADER_BYTES + RELEASE_OUTCOME_BYTES == RELEASE_REPLY_BYTES
assert REQUEST_HEADER_BYTES + ALLOCATE_PROJECTION_BYTES + 2 * PATH_CEILING_BYTES == ALLOCATE_REQUEST_HARD_CEILING
assert REQUEST_HEADER_BYTES + PATH_CEILING_BYTES == RELEASE_REQUEST_HARD_CEILING
assert struct.calcsize("<QI") == REPLY_TAG_OFFSET

_REQUEST_ERROR_KINDS = frozenset(
    {
        RegionControlErrorKind.BAD_MAGIC_VERSION,
        RegionControlErrorKind.BAD_MESSAGE_SIZE,
        RegionControlErrorKind.INVALID_ENUM_VALUE,
        RegionControlErrorKind.RESERVED_NONZERO,
        RegionControlErrorKind.INVALID_FIELD_VALUE,
    }
)
_ALLOCATION_ERROR_KINDS = frozenset(
    {
        RegionControlErrorKind.BACKEND_FAILURE,
        RegionControlErrorKind.INTERNAL_INVARIANT,
    }
)
_RELEASE_ERROR_KINDS = frozenset(
    {
        RegionControlErrorKind.INVALID_FIELD_VALUE,
        RegionControlErrorKind.STORE_LIFECYCLE,
        RegionControlErrorKind.INTERNAL_INVARIANT,
    }
)
_ALLOCATE_PROTOCOL_ERROR_KINDS = _REQUEST_ERROR_KINDS | {
    RegionControlErrorKind.STORE_LIFECYCLE,
    RegionControlErrorKind.INTERNAL_INVARIANT,
}

_IntEnumT = TypeVar("_IntEnumT", bound=IntEnum)


class DelegatedRegionOperation(IntEnum):
    INVALID = 0
    DELEGATED_ALLOCATE = 1
    DELEGATED_RELEASE = 2


class DelegatedAllocateReplyTag(IntEnum):
    EMPTY = 0
    ALLOCATED = 1
    ERROR = 2


class DelegatedReleaseReplyTag(IntEnum):
    EMPTY = 0
    RELEASED = 1
    ALREADY_GONE = 2
    UNKNOWN_RESOURCE = 3
    CLEANUP_INCOMPLETE = 4
    ERROR = 5
    UNKNOWN_TRANSACTION = 6


@dataclass(frozen=True)
class DelegatedAllocateRequest:
    session_instance_id: bytes
    transaction_id: int
    initiator_path: bytes
    provider_path: bytes
    payload_logical_bytes: int
    counter_logical_bytes: int
    topology: RegionTopologyKind = RegionTopologyKind.SINGLE_OWNER
    initiator_deployment: EndpointDeploymentKind = EndpointDeploymentKind.HOST_CPU
    provider_deployment: EndpointDeploymentKind = EndpointDeploymentKind.DEVICE_AICPU
    payload_backend_kind: BackendKind = BackendKind.VMM_SHAREABLE
    counter_backend_kind: BackendKind = BackendKind.VMM_SHAREABLE
    payload_consumer_adapter_kind: AdapterKind = AdapterKind.OWNER_DELEGATED_COPY
    payload_consumer_adapter_profile: AdapterProfile = AdapterProfile.HOST_VMM_COPY
    counter_consumer_adapter_kind: AdapterKind = AdapterKind.OWNER_DELEGATED_COPY
    counter_consumer_adapter_profile: AdapterProfile = AdapterProfile.HOST_VMM_COPY

    @property
    def spec(self) -> RegionAllocationSpec:
        return RegionAllocationSpec(
            payload=RegionPartAllocationSpec(
                planned_backing_kind=self.payload_backend_kind,
                logical_bytes=self.payload_logical_bytes,
            ),
            counter=RegionPartAllocationSpec(
                planned_backing_kind=self.counter_backend_kind,
                logical_bytes=self.counter_logical_bytes,
            ),
        )

    @property
    def projection_bytes(self) -> bytes:
        return _pack_projection(self)


@dataclass(frozen=True)
class DelegatedReleaseRequest:
    session_instance_id: bytes
    transaction_id: int
    provider_path: bytes


@dataclass(frozen=True)
class DelegatedAllocateReply:
    tag: DelegatedAllocateReplyTag
    session_instance_id: bytes
    transaction_id: int
    result: RegionAllocationResult | None = None
    payload_view: RegionPartLocalView | None = None
    counter_view: RegionPartLocalView | None = None
    error_kind: RegionControlErrorKind = RegionControlErrorKind.NONE
    failed_part: RegionPartKind = RegionPartKind.INVALID
    failed_operation: RegionOperationKind = RegionOperationKind.NONE
    cleanup_debt_remaining: bool = False
    provisional_resource_id: int = 0
    error: RegionControlError | RegionAllocationError | None = None


@dataclass(frozen=True)
class DelegatedReleaseReply:
    tag: DelegatedReleaseReplyTag
    session_instance_id: bytes
    transaction_id: int
    result: ProviderReleaseResult | None = None
    error_kind: RegionControlErrorKind = RegionControlErrorKind.NONE
    error: RegionControlError | None = None


@dataclass(frozen=True)
class DelegatedRegionRequestEnvelope:
    operation: DelegatedRegionOperation
    session_instance_id: bytes
    transaction_id: int
    request_bytes: int
    provider_path_bytes: int
    initiator_path_bytes: int
    frame: bytes

    @property
    def provider_path(self) -> bytes:
        return self.frame[self.request_bytes - self.provider_path_bytes : self.request_bytes]

    def decode_terminal(self) -> DelegatedAllocateRequest | DelegatedReleaseRequest:
        if self.operation is DelegatedRegionOperation.DELEGATED_ALLOCATE:
            return _decode_allocate_request(self)
        return _decode_release_request(self)


@dataclass(frozen=True)
class DelegatedRegionReplyEnvelope:
    operation: DelegatedRegionOperation
    reply_tag: int
    session_instance_id: bytes
    transaction_id: int
    reply_bytes: int
    frame: bytes

    def decode_outcome(self) -> DelegatedAllocateReply | DelegatedReleaseReply:
        if self.operation is DelegatedRegionOperation.DELEGATED_ALLOCATE:
            return _decode_allocate_reply(self)
        return _decode_release_reply(self)


def encode_request(request: DelegatedAllocateRequest | DelegatedReleaseRequest, *, staged_capacity: int) -> bytearray:
    if isinstance(request, DelegatedAllocateRequest):
        active = _encode_allocate_request(request)
        ceiling = ALLOCATE_REQUEST_HARD_CEILING
        reply_bytes = ALLOCATE_REPLY_BYTES
    elif isinstance(request, DelegatedReleaseRequest):
        active = _encode_release_request(request)
        ceiling = RELEASE_REQUEST_HARD_CEILING
        reply_bytes = RELEASE_REPLY_BYTES
    else:
        raise TypeError("request must be DelegatedAllocateRequest or DelegatedReleaseRequest")
    if len(active) > ceiling:
        raise RegionControlError(
            RegionControlErrorKind.BAD_MESSAGE_SIZE,
            f"request_bytes {len(active)} exceeds hard ceiling {ceiling}",
        )
    minimum = max(len(active), reply_bytes)
    if type(staged_capacity) is not int or staged_capacity < minimum:
        raise RegionControlError(
            RegionControlErrorKind.BAD_MESSAGE_SIZE,
            f"staged_capacity must be at least {minimum}",
        )
    staged = bytearray(staged_capacity)
    staged[: len(active)] = active
    return staged


def _operation_fixed_reply_bytes(operation: DelegatedRegionOperation) -> int:
    if operation is DelegatedRegionOperation.DELEGATED_ALLOCATE:
        return ALLOCATE_REPLY_BYTES
    return RELEASE_REPLY_BYTES


def parse_request(payload: bytes | bytearray | memoryview) -> DelegatedRegionRequestEnvelope:
    raw = _owned_bytes(payload)
    if len(raw) < REQUEST_HEADER_BYTES:
        raise RegionControlError(RegionControlErrorKind.BAD_MESSAGE_SIZE, "request shorter than 40-byte header")
    magic, request_bytes, operation, session, transaction_id, provider_path_bytes, initiator_path_bytes = (
        _REQUEST_HEADER.unpack_from(raw, 0)
    )
    _require_magic(int(magic))
    op = _require_operation(int(operation))
    request_size = _require_u32("request_bytes", int(request_bytes))
    provider_size = _require_u32("provider_path_bytes", int(provider_path_bytes))
    initiator_size = _require_u32("initiator_path_bytes", int(initiator_path_bytes))
    ceiling = (
        ALLOCATE_REQUEST_HARD_CEILING
        if op is DelegatedRegionOperation.DELEGATED_ALLOCATE
        else RELEASE_REQUEST_HARD_CEILING
    )
    if request_size > ceiling:
        raise RegionControlError(
            RegionControlErrorKind.BAD_MESSAGE_SIZE,
            f"request_bytes {request_size} exceeds hard ceiling {ceiling}",
        )
    fixed_reply_bytes = _operation_fixed_reply_bytes(op)
    if len(raw) < fixed_reply_bytes:
        raise RegionControlError(
            RegionControlErrorKind.BAD_MESSAGE_SIZE,
            "staging is smaller than the operation fixed reply",
        )
    if request_size > len(raw) or request_size < REQUEST_HEADER_BYTES:
        raise RegionControlError(RegionControlErrorKind.BAD_MESSAGE_SIZE, "request_bytes does not fit the payload")
    if any(raw[request_size:]):
        raise RegionControlError(RegionControlErrorKind.RESERVED_NONZERO, "request staging tail must be zero")
    if provider_size > PATH_CEILING_BYTES or initiator_size > PATH_CEILING_BYTES:
        raise RegionControlError(RegionControlErrorKind.BAD_MESSAGE_SIZE, "path exceeds 256-byte ceiling")
    if op is DelegatedRegionOperation.DELEGATED_ALLOCATE:
        expected = REQUEST_HEADER_BYTES + ALLOCATE_PROJECTION_BYTES + initiator_size + provider_size
    else:
        if initiator_size != 0:
            raise RegionControlError(
                RegionControlErrorKind.INVALID_FIELD_VALUE,
                "release initiator_path_bytes must be zero",
            )
        expected = REQUEST_HEADER_BYTES + provider_size
    if request_size != expected:
        raise RegionControlError(
            RegionControlErrorKind.BAD_MESSAGE_SIZE,
            "request_bytes does not match the operation length equation",
        )
    if provider_size == 0:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, "provider path must be non-empty")
    frame = raw[:request_size]
    provider_path = frame[request_size - provider_size : request_size]
    _require_canonical_path(provider_path, field="provider path")
    return DelegatedRegionRequestEnvelope(
        operation=op,
        session_instance_id=_require_session(session),
        transaction_id=_require_transaction_id(int(transaction_id)),
        request_bytes=request_size,
        provider_path_bytes=provider_size,
        initiator_path_bytes=initiator_size,
        frame=frame,
    )


def encode_reply(reply: DelegatedAllocateReply | DelegatedReleaseReply) -> bytes:
    if isinstance(reply, DelegatedAllocateReply):
        return _encode_allocate_reply(reply)
    if isinstance(reply, DelegatedReleaseReply):
        return _encode_release_reply(reply)
    raise TypeError("reply must be DelegatedAllocateReply or DelegatedReleaseReply")


def parse_reply(payload: bytes | bytearray | memoryview) -> DelegatedRegionReplyEnvelope:
    raw = _owned_bytes(payload)
    if len(raw) < REPLY_HEADER_BYTES:
        raise RegionControlError(RegionControlErrorKind.BAD_MESSAGE_SIZE, "reply shorter than 40-byte header")
    magic, reply_bytes, reply_tag, operation, reserved, session, transaction_id = _REPLY_HEADER.unpack_from(raw, 0)
    _require_magic(int(magic))
    if int(reserved) != 0:
        raise RegionControlError(RegionControlErrorKind.RESERVED_NONZERO, "reply header reserved must be zero")
    op = _require_operation(int(operation))
    reply_size = _require_u32("reply_bytes", int(reply_bytes))
    expected = ALLOCATE_REPLY_BYTES if op is DelegatedRegionOperation.DELEGATED_ALLOCATE else RELEASE_REPLY_BYTES
    if reply_size != expected:
        raise RegionControlError(
            RegionControlErrorKind.BAD_MESSAGE_SIZE,
            f"reply_bytes must be {expected}",
        )
    if reply_size > len(raw):
        raise RegionControlError(RegionControlErrorKind.BAD_MESSAGE_SIZE, "reply_bytes does not fit the payload")
    if any(raw[reply_size:]):
        raise RegionControlError(RegionControlErrorKind.RESERVED_NONZERO, "reply staging tail must be zero")
    _require_known_reply_tag(op, int(reply_tag))
    return DelegatedRegionReplyEnvelope(
        operation=op,
        reply_tag=int(reply_tag),
        session_instance_id=_require_session(session),
        transaction_id=_require_transaction_id(int(transaction_id)),
        reply_bytes=reply_size,
        frame=raw[:reply_size],
    )


def publish_reply(staged: memoryview, committed_reply: bytes) -> None:
    view = _as_writable_bytes(staged)
    committed = bytes(committed_reply)
    if len(committed) not in (ALLOCATE_REPLY_BYTES, RELEASE_REPLY_BYTES):
        raise RegionControlError(
            RegionControlErrorKind.BAD_MESSAGE_SIZE,
            "committed reply must be the operation fixed size",
        )
    if view.nbytes < len(committed):
        raise RegionControlError(
            RegionControlErrorKind.BAD_MESSAGE_SIZE,
            "staging is smaller than the committed reply",
        )
    envelope = parse_reply(committed)
    if envelope.reply_tag == 0:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, "cannot publish an EMPTY reply tag")
    view[:] = b"\x00" * view.nbytes
    prefix = bytearray(committed)
    _REPLY_TAG.pack_into(prefix, REPLY_TAG_OFFSET, 0)
    view[: len(committed)] = prefix
    _REPLY_TAG.pack_into(view, REPLY_TAG_OFFSET, envelope.reply_tag)


def _owned_bytes(payload: bytes | bytearray | memoryview) -> bytes:
    if isinstance(payload, memoryview):
        return payload.tobytes()
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    raise TypeError("payload must be bytes, bytearray, or memoryview")


def _as_writable_bytes(staged: memoryview) -> memoryview:
    if not isinstance(staged, memoryview):
        raise TypeError("staged reply buffer must be a memoryview")
    view = staged.cast("B") if staged.format != "B" else staged
    if view.readonly:
        raise RegionControlError(RegionControlErrorKind.INTERNAL_INVARIANT, "reply buffer must be writable")
    return view


def _require_magic(magic: int) -> None:
    if magic != DELEGATED_REGION_CTRL_MAGIC_VERSION:
        raise RegionControlError(RegionControlErrorKind.BAD_MAGIC_VERSION, "unsupported delegated-region version")


def _require_operation(value: int) -> DelegatedRegionOperation:
    typed = _require_enum(DelegatedRegionOperation, value)
    if typed is DelegatedRegionOperation.INVALID:
        raise RegionControlError(RegionControlErrorKind.INVALID_ENUM_VALUE, "operation must be nonzero")
    return typed


def _require_enum(enum_cls: type[_IntEnumT], value: int, *, allow_zero: bool = False) -> _IntEnumT:
    try:
        typed = enum_cls(value)
    except ValueError as exc:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_ENUM_VALUE,
            f"unknown {enum_cls.__name__} {value}",
        ) from exc
    if not allow_zero and int(typed) == 0:
        raise RegionControlError(RegionControlErrorKind.INVALID_ENUM_VALUE, f"{enum_cls.__name__} must be nonzero")
    return typed


def _require_known_reply_tag(operation: DelegatedRegionOperation, tag: int) -> None:
    enum_cls = (
        DelegatedAllocateReplyTag
        if operation is DelegatedRegionOperation.DELEGATED_ALLOCATE
        else DelegatedReleaseReplyTag
    )
    _require_enum(enum_cls, tag, allow_zero=True)


def _require_session(value: object) -> bytes:
    if isinstance(value, memoryview):
        raw = value.tobytes()
    elif isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    else:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            "session_instance_id must be 8 opaque bytes",
        )
    if len(raw) != 8:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            "session_instance_id must be 8 opaque bytes",
        )
    return raw


def _require_transaction_id(value: int) -> int:
    if type(value) is not int or value < 1 or value > _UINT64_MAX:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, "transaction_id 0 is illegal")
    return value


def _require_u32(name: str, value: int) -> int:
    if value < 0 or value > 0xFFFFFFFF:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, f"{name} is out of range")
    return value


def _require_backend_kind(value: int) -> BackendKind:
    kind = _BACKEND_KINDS.get(int(value))
    if kind is None:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_ENUM_VALUE,
            f"unknown planned backing kind {value}",
        )
    return kind


def _format_parsed_path(segments: tuple[EndpointPathSegment, ...]) -> str:
    parts: list[str] = []
    for segment in segments:
        if segment.index is None:
            parts.append(f"L{segment.level}")
        else:
            parts.append(f"L{segment.level}[{segment.index}]")
    return "/".join(parts)


@dataclass(frozen=True)
class _DelegatedRouteHop:
    child_id: int
    child_level: int


def _inspect_delegated_route(current_path: str, provider_path: bytes) -> _DelegatedRouteHop:
    if not isinstance(current_path, str) or not current_path:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, "current path must be a non-empty path")
    first = current_path.split("/", 1)[0]
    root = _PATH_ROOT_RE.fullmatch(first)
    if root is None:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            "current path must start with a root L<n> segment",
        )
    try:
        current = parse_endpoint_path(current_path, root_level=int(root.group(1)))
    except ValueError as exc:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            str(exc) or "current path is invalid",
        ) from exc
    if _format_parsed_path(current.segments) != current_path:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            "current path must use canonical ASCII path text",
        )
    provider = _require_canonical_path(bytes(provider_path), field="provider path")
    try:
        provider_parsed = parse_endpoint_path(provider.decode("ascii"), root_level=int(root.group(1)))
    except ValueError as exc:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            str(exc) or "provider path is not under the current path root",
        ) from exc
    if provider_parsed.segments[: len(current.segments)] != current.segments:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            "current path is not a prefix of the provider path",
        )
    if len(provider_parsed.segments) <= len(current.segments):
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            "current path must be a strict prefix of the provider path",
        )
    hop = provider_parsed.segments[len(current.segments)]
    remainder = provider_parsed.segments[len(current.segments) + 1 :]
    if hop.index is None:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, "next hop must include a child index")
    if hop.level == 2 and remainder:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            "provider path must terminate at the selected L2",
        )
    return _DelegatedRouteHop(child_id=int(hop.index), child_level=int(hop.level))


def _hop_staging_copy(envelope: DelegatedRegionRequestEnvelope) -> bytearray:
    reply_bytes = (
        ALLOCATE_REPLY_BYTES
        if envelope.operation is DelegatedRegionOperation.DELEGATED_ALLOCATE
        else RELEASE_REPLY_BYTES
    )
    capacity = max(int(envelope.request_bytes), reply_bytes)
    staged = bytearray(capacity)
    staged[: envelope.request_bytes] = envelope.frame
    return staged


def _require_canonical_path(path: bytes, *, field: str) -> bytes:
    if not path:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, f"{field} must be a non-empty path")
    if len(path) > PATH_CEILING_BYTES:
        raise RegionControlError(RegionControlErrorKind.BAD_MESSAGE_SIZE, f"{field} exceeds 256-byte ceiling")
    try:
        text = path.decode("ascii")
    except UnicodeDecodeError as exc:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, f"{field} must be ASCII") from exc
    first = text.split("/", 1)[0]
    root = _PATH_ROOT_RE.fullmatch(first)
    if root is None:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            f"{field} must start with a root L<n> segment",
        )
    try:
        parsed = parse_endpoint_path(text, root_level=int(root.group(1)))
    except ValueError as exc:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, str(exc) or f"{field} is invalid") from exc
    canonical = _format_parsed_path(parsed.segments).encode("ascii")
    if canonical != path:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            f"{field} must use canonical ASCII path bytes",
        )
    return path


def _pack_projection(request: DelegatedAllocateRequest) -> bytes:
    return _ALLOCATE_PROJECTION.pack(
        int(request.topology),
        int(request.initiator_deployment),
        int(request.provider_deployment),
        0,
        int(request.payload_logical_bytes),
        int(request.payload_backend_kind),
        _adapter_kind_id(request.payload_consumer_adapter_kind),
        _adapter_profile_id(request.payload_consumer_adapter_profile),
        0,
        int(request.counter_logical_bytes),
        int(request.counter_backend_kind),
        _adapter_kind_id(request.counter_consumer_adapter_kind),
        _adapter_profile_id(request.counter_consumer_adapter_profile),
        0,
    )


def _encode_allocate_request(request: DelegatedAllocateRequest) -> bytes:
    session = _require_session(request.session_instance_id)
    transaction_id = _require_transaction_id(int(request.transaction_id))
    initiator_path = _require_canonical_path(bytes(request.initiator_path), field="initiator path")
    provider_path = _require_canonical_path(bytes(request.provider_path), field="provider path")
    _require_first_shape_fields(request)
    try:
        _ = request.spec
    except (TypeError, ValueError) as exc:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            str(exc) or "allocate spec is invalid",
        ) from exc
    projection = _pack_projection(request)
    request_bytes = REQUEST_HEADER_BYTES + ALLOCATE_PROJECTION_BYTES + len(initiator_path) + len(provider_path)
    header = _REQUEST_HEADER.pack(
        DELEGATED_REGION_CTRL_MAGIC_VERSION,
        request_bytes,
        int(DelegatedRegionOperation.DELEGATED_ALLOCATE),
        session,
        transaction_id,
        len(provider_path),
        len(initiator_path),
    )
    return header + projection + initiator_path + provider_path


def _encode_release_request(request: DelegatedReleaseRequest) -> bytes:
    session = _require_session(request.session_instance_id)
    transaction_id = _require_transaction_id(int(request.transaction_id))
    provider_path = _require_canonical_path(bytes(request.provider_path), field="provider path")
    request_bytes = REQUEST_HEADER_BYTES + len(provider_path)
    header = _REQUEST_HEADER.pack(
        DELEGATED_REGION_CTRL_MAGIC_VERSION,
        request_bytes,
        int(DelegatedRegionOperation.DELEGATED_RELEASE),
        session,
        transaction_id,
        len(provider_path),
        0,
    )
    return header + provider_path


def _decode_allocate_request(envelope: DelegatedRegionRequestEnvelope) -> DelegatedAllocateRequest:
    projection = envelope.frame[REQUEST_HEADER_BYTES : REQUEST_HEADER_BYTES + ALLOCATE_PROJECTION_BYTES]
    initiator_path = envelope.frame[
        REQUEST_HEADER_BYTES + ALLOCATE_PROJECTION_BYTES : envelope.request_bytes - envelope.provider_path_bytes
    ]
    if envelope.initiator_path_bytes == 0:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, "initiator path must be non-empty")
    _require_canonical_path(initiator_path, field="initiator path")
    (
        topology,
        initiator_deployment,
        provider_deployment,
        reserved,
        payload_logical,
        payload_backend,
        payload_kind,
        payload_profile,
        payload_reserved,
        counter_logical,
        counter_backend,
        counter_kind,
        counter_profile,
        counter_reserved,
    ) = _ALLOCATE_PROJECTION.unpack(projection)
    if int(reserved) != 0 or int(payload_reserved) != 0 or int(counter_reserved) != 0:
        raise RegionControlError(RegionControlErrorKind.RESERVED_NONZERO, "projection reserved fields must be zero")
    typed_topology = _require_wire_enum(RegionTopologyKind, int(topology), "topology")
    typed_initiator = _require_wire_enum(EndpointDeploymentKind, int(initiator_deployment), "initiator_deployment")
    typed_provider = _require_wire_enum(EndpointDeploymentKind, int(provider_deployment), "provider_deployment")
    payload_backend_kind = _require_backend_kind(int(payload_backend))
    counter_backend_kind = _require_backend_kind(int(counter_backend))
    payload_adapter = _require_adapter_kind(int(payload_kind), "PAYLOAD consumer_adapter_kind")
    payload_adapter_profile = _require_adapter_profile(int(payload_profile), "PAYLOAD consumer_adapter_profile")
    counter_adapter = _require_adapter_kind(int(counter_kind), "COUNTER consumer_adapter_kind")
    counter_adapter_profile = _require_adapter_profile(int(counter_profile), "COUNTER consumer_adapter_profile")
    request = DelegatedAllocateRequest(
        session_instance_id=envelope.session_instance_id,
        transaction_id=envelope.transaction_id,
        initiator_path=initiator_path,
        provider_path=envelope.provider_path,
        payload_logical_bytes=int(payload_logical),
        counter_logical_bytes=int(counter_logical),
        topology=typed_topology,
        initiator_deployment=typed_initiator,
        provider_deployment=typed_provider,
        payload_backend_kind=payload_backend_kind,
        counter_backend_kind=counter_backend_kind,
        payload_consumer_adapter_kind=payload_adapter,
        payload_consumer_adapter_profile=payload_adapter_profile,
        counter_consumer_adapter_kind=counter_adapter,
        counter_consumer_adapter_profile=counter_adapter_profile,
    )
    _require_first_shape_fields(request)
    try:
        _ = request.spec
    except (TypeError, ValueError) as exc:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            str(exc) or "allocate spec is invalid",
        ) from exc
    return request


def _decode_release_request(envelope: DelegatedRegionRequestEnvelope) -> DelegatedReleaseRequest:
    return DelegatedReleaseRequest(
        session_instance_id=envelope.session_instance_id,
        transaction_id=envelope.transaction_id,
        provider_path=envelope.provider_path,
    )


def _require_first_shape_fields(request: DelegatedAllocateRequest) -> None:
    if request.topology is not RegionTopologyKind.SINGLE_OWNER:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, "topology must be SINGLE_OWNER")
    if request.initiator_deployment is not EndpointDeploymentKind.HOST_CPU:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, "initiator_deployment must be HOST_CPU")
    if request.provider_deployment is not EndpointDeploymentKind.DEVICE_AICPU:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, "provider_deployment must be DEVICE_AICPU")
    if request.payload_backend_kind is not BackendKind.VMM_SHAREABLE:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE, "PAYLOAD backend_kind must be VMM_SHAREABLE"
        )
    if request.counter_backend_kind is not BackendKind.VMM_SHAREABLE:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE, "COUNTER backend_kind must be VMM_SHAREABLE"
        )
    if request.payload_consumer_adapter_kind is not AdapterKind.OWNER_DELEGATED_COPY:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            "PAYLOAD consumer_adapter_kind must be OWNER_DELEGATED_COPY",
        )
    if request.counter_consumer_adapter_kind is not AdapterKind.OWNER_DELEGATED_COPY:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            "COUNTER consumer_adapter_kind must be OWNER_DELEGATED_COPY",
        )
    if request.payload_consumer_adapter_profile is not AdapterProfile.HOST_VMM_COPY:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            "PAYLOAD consumer_adapter_profile must be HOST_VMM_COPY",
        )
    if request.counter_consumer_adapter_profile is not AdapterProfile.HOST_VMM_COPY:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            "COUNTER consumer_adapter_profile must be HOST_VMM_COPY",
        )


def _require_wire_enum(enum_cls: type[_IntEnumT], value: int, name: str) -> _IntEnumT:
    typed = _require_enum(enum_cls, value, allow_zero=True)
    if int(typed) == 0:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, f"{name} must satisfy first shape")
    return typed


def _require_adapter_kind(value: int, name: str) -> AdapterKind:
    try:
        kind = _adapter_kind_from_id(value)
    except ValueError as exc:
        raise RegionControlError(RegionControlErrorKind.INVALID_ENUM_VALUE, f"{name} is unknown") from exc
    if kind is None:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, f"{name} must satisfy first shape")
    return kind


def _require_adapter_profile(value: int, name: str) -> AdapterProfile:
    try:
        profile = _adapter_profile_from_id(value)
    except ValueError as exc:
        raise RegionControlError(RegionControlErrorKind.INVALID_ENUM_VALUE, f"{name} is unknown") from exc
    if profile is None:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, f"{name} must satisfy first shape")
    return profile


def _encode_allocate_reply(reply: DelegatedAllocateReply) -> bytes:
    session = _require_session(reply.session_instance_id)
    transaction_id = _require_transaction_id(int(reply.transaction_id))
    tag = _require_enum(DelegatedAllocateReplyTag, int(reply.tag))
    if tag is DelegatedAllocateReplyTag.EMPTY:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, "cannot encode an EMPTY allocate reply")
    frame = bytearray(ALLOCATE_REPLY_BYTES)
    _REPLY_HEADER.pack_into(
        frame,
        0,
        DELEGATED_REGION_CTRL_MAGIC_VERSION,
        ALLOCATE_REPLY_BYTES,
        0,
        int(DelegatedRegionOperation.DELEGATED_ALLOCATE),
        0,
        session,
        transaction_id,
    )
    if tag is DelegatedAllocateReplyTag.ALLOCATED:
        if reply.result is None or reply.payload_view is None or reply.counter_view is None:
            raise RegionControlError(
                RegionControlErrorKind.INVALID_FIELD_VALUE,
                "ALLOCATED requires a result and both local views",
            )
        if (
            reply.error_kind is not RegionControlErrorKind.NONE
            or reply.failed_part is not RegionPartKind.INVALID
            or reply.failed_operation is not RegionOperationKind.NONE
            or reply.cleanup_debt_remaining
            or reply.provisional_resource_id != 0
        ):
            raise RegionControlError(
                RegionControlErrorKind.INVALID_FIELD_VALUE,
                "ALLOCATED requires inactive error fields",
            )
        validate_independent_local_views(reply.payload_view, reply.counter_view)
        _ALLOCATE_OUTCOME_PREFIX.pack_into(
            frame,
            ALLOCATE_OUTCOME_OFFSET,
            int(reply.result.provider_resource_id),
            0,
            0,
            0,
            0,
        )
        _encode_buffer_descriptor(frame, ALLOCATE_PAYLOAD_DESCRIPTOR_OFFSET, reply.result.export_descriptor.payload)
        _encode_buffer_descriptor(frame, ALLOCATE_COUNTER_DESCRIPTOR_OFFSET, reply.result.export_descriptor.counter)
        _encode_local_view(frame, ALLOCATE_PAYLOAD_VIEW_OFFSET, reply.payload_view)
        _encode_local_view(frame, ALLOCATE_COUNTER_VIEW_OFFSET, reply.counter_view)
    elif tag is DelegatedAllocateReplyTag.ERROR:
        resource_id, error_kind, failed_part, failed_operation, debt = _allocate_error_fields(reply)
        _ALLOCATE_OUTCOME_PREFIX.pack_into(
            frame,
            ALLOCATE_OUTCOME_OFFSET,
            resource_id,
            int(error_kind),
            failed_part,
            failed_operation,
            debt,
        )
    else:
        raise RegionControlError(RegionControlErrorKind.INVALID_ENUM_VALUE, f"unknown allocate reply tag {int(tag)}")
    _REPLY_TAG.pack_into(frame, REPLY_TAG_OFFSET, int(tag))
    return bytes(frame)


def _allocate_error_fields(reply: DelegatedAllocateReply) -> tuple[int, RegionControlErrorKind, int, int, int]:
    error = reply.error
    if isinstance(error, RegionAllocationError):
        return (
            int(error.provisional_resource_id),
            error.control_kind,
            int(error.failed_part),
            int(error.failed_operation),
            1 if error.cleanup_debt_remaining else 0,
        )
    if isinstance(error, RegionControlError):
        if error.kind not in _ALLOCATE_PROTOCOL_ERROR_KINDS:
            raise RegionControlError(
                RegionControlErrorKind.INTERNAL_INVARIANT,
                "allocate protocol ERROR kind must be a unique no-resource code",
            )
        return 0, error.kind, 0, 0, 0
    kind = _require_enum(RegionControlErrorKind, int(reply.error_kind))
    if kind in _ALLOCATE_PROTOCOL_ERROR_KINDS and int(reply.provisional_resource_id) == 0:
        if (
            reply.failed_part is not RegionPartKind.INVALID
            or reply.failed_operation is not RegionOperationKind.NONE
            or reply.cleanup_debt_remaining
            or reply.result is not None
        ):
            raise RegionControlError(
                RegionControlErrorKind.INVALID_FIELD_VALUE,
                "protocol ERROR requires inactive resource and error slots",
            )
        return 0, kind, 0, 0, 0
    if kind in _ALLOCATION_ERROR_KINDS:
        resource_id = int(reply.provisional_resource_id)
        if resource_id == 0:
            raise RegionControlError(
                RegionControlErrorKind.INVALID_FIELD_VALUE,
                "allocation ERROR requires a provisional resource id",
            )
        return (
            resource_id,
            kind,
            int(reply.failed_part),
            int(reply.failed_operation),
            1 if reply.cleanup_debt_remaining else 0,
        )
    raise RegionControlError(RegionControlErrorKind.INVALID_ENUM_VALUE, "allocate ERROR kind is not a unique authority")


def _decode_allocate_reply(envelope: DelegatedRegionReplyEnvelope) -> DelegatedAllocateReply:
    tag = _require_enum(DelegatedAllocateReplyTag, envelope.reply_tag, allow_zero=True)
    if tag is DelegatedAllocateReplyTag.EMPTY:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, "allocate reply is missing commit tag")
    resource_id, error_kind, failed_part, failed_operation, debt = _ALLOCATE_OUTCOME_PREFIX.unpack_from(
        envelope.frame, ALLOCATE_OUTCOME_OFFSET
    )
    if tag is DelegatedAllocateReplyTag.ALLOCATED:
        if (
            int(resource_id) == 0
            or int(error_kind) != 0
            or int(failed_part) != 0
            or int(failed_operation) != 0
            or int(debt) != 0
        ):
            raise RegionControlError(
                RegionControlErrorKind.INVALID_FIELD_VALUE,
                "ALLOCATED requires a resource id and zero error fields",
            )
        descriptor = RegionExportDescriptor(
            payload=_decode_buffer_descriptor(envelope.frame, ALLOCATE_PAYLOAD_DESCRIPTOR_OFFSET),
            counter=_decode_buffer_descriptor(envelope.frame, ALLOCATE_COUNTER_DESCRIPTOR_OFFSET),
        )
        _validate_decoded_descriptor_pair(descriptor)
        payload_view = _decode_local_view(envelope.frame, ALLOCATE_PAYLOAD_VIEW_OFFSET, RegionPartKind.PAYLOAD)
        counter_view = _decode_local_view(envelope.frame, ALLOCATE_COUNTER_VIEW_OFFSET, RegionPartKind.COUNTER)
        validate_independent_local_views(payload_view, counter_view)
        result = RegionAllocationResult(provider_resource_id=int(resource_id), export_descriptor=descriptor)
        return DelegatedAllocateReply(
            tag=tag,
            session_instance_id=envelope.session_instance_id,
            transaction_id=envelope.transaction_id,
            result=result,
            payload_view=payload_view,
            counter_view=counter_view,
        )
    _require_zero_span(
        envelope.frame, ALLOCATE_PAYLOAD_DESCRIPTOR_OFFSET, 2 * BUFFER_DESCRIPTOR_WIRE_BYTES + 2 * LOCAL_VIEW_BYTES
    )
    kind = _require_enum(RegionControlErrorKind, int(error_kind), allow_zero=True)
    if kind is RegionControlErrorKind.NONE:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, "ERROR requires a nonzero error_kind")
    if int(resource_id) == 0 and kind in _ALLOCATE_PROTOCOL_ERROR_KINDS:
        if int(failed_part) != 0 or int(failed_operation) != 0 or int(debt) != 0:
            raise RegionControlError(
                RegionControlErrorKind.INVALID_FIELD_VALUE,
                "protocol ERROR requires resource id zero and inactive error slots",
            )
        error = RegionControlError(kind)
        return DelegatedAllocateReply(
            tag=DelegatedAllocateReplyTag.ERROR,
            session_instance_id=envelope.session_instance_id,
            transaction_id=envelope.transaction_id,
            error_kind=kind,
            error=error,
        )
    if kind in _ALLOCATION_ERROR_KINDS:
        if int(resource_id) == 0:
            raise RegionControlError(
                RegionControlErrorKind.INVALID_FIELD_VALUE,
                "allocation ERROR requires a provisional resource id",
            )
        if int(debt) not in (0, 1):
            raise RegionControlError(
                RegionControlErrorKind.INVALID_FIELD_VALUE, "cleanup debt remaining must be 0 or 1"
            )
        error = RegionAllocationError(
            provisional_resource_id=int(resource_id),
            control_kind=kind,
            failed_part=int(failed_part),
            failed_operation=int(failed_operation),
            cleanup_debt_remaining=bool(debt),
        )
        return DelegatedAllocateReply(
            tag=DelegatedAllocateReplyTag.ERROR,
            session_instance_id=envelope.session_instance_id,
            transaction_id=envelope.transaction_id,
            error_kind=kind,
            failed_part=error.failed_part,
            failed_operation=error.failed_operation,
            cleanup_debt_remaining=bool(debt),
            provisional_resource_id=int(resource_id),
            error=error,
        )
    raise RegionControlError(RegionControlErrorKind.INVALID_ENUM_VALUE, "allocate ERROR kind is not a unique authority")


def _encode_release_reply(reply: DelegatedReleaseReply) -> bytes:
    session = _require_session(reply.session_instance_id)
    transaction_id = _require_transaction_id(int(reply.transaction_id))
    tag = _require_enum(DelegatedReleaseReplyTag, int(reply.tag))
    if tag is DelegatedReleaseReplyTag.EMPTY:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, "cannot encode an EMPTY release reply")
    frame = bytearray(RELEASE_REPLY_BYTES)
    _REPLY_HEADER.pack_into(
        frame,
        0,
        DELEGATED_REGION_CTRL_MAGIC_VERSION,
        RELEASE_REPLY_BYTES,
        0,
        int(DelegatedRegionOperation.DELEGATED_RELEASE),
        0,
        session,
        transaction_id,
    )
    if tag is DelegatedReleaseReplyTag.UNKNOWN_TRANSACTION:
        if reply.result is not None or reply.error_kind is not RegionControlErrorKind.NONE:
            raise RegionControlError(
                RegionControlErrorKind.INVALID_FIELD_VALUE,
                "UNKNOWN_TRANSACTION requires a zero outcome",
            )
    elif tag is DelegatedReleaseReplyTag.ERROR:
        kind = reply.error.kind if isinstance(reply.error, RegionControlError) else reply.error_kind
        typed = _require_enum(RegionControlErrorKind, int(kind))
        if typed not in _RELEASE_ERROR_KINDS:
            raise RegionControlError(
                RegionControlErrorKind.INTERNAL_INVARIANT,
                "release ERROR kind must be INVALID_FIELD_VALUE, STORE_LIFECYCLE, or INTERNAL_INVARIANT",
            )
        resource_id = 0 if reply.result is None else int(reply.result.provider_resource_id)
        _RELEASE_OUTCOME.pack_into(frame, RELEASE_OUTCOME_OFFSET, resource_id, 0, int(typed), 0, 0, 0, 0)
    else:
        if reply.result is None:
            raise RegionControlError(
                RegionControlErrorKind.INVALID_FIELD_VALUE, f"{tag.name} requires a release result"
            )
        status_tag = {
            ProviderReleaseStatus.RELEASED: DelegatedReleaseReplyTag.RELEASED,
            ProviderReleaseStatus.ALREADY_GONE: DelegatedReleaseReplyTag.ALREADY_GONE,
            ProviderReleaseStatus.UNKNOWN_RESOURCE: DelegatedReleaseReplyTag.UNKNOWN_RESOURCE,
            ProviderReleaseStatus.CLEANUP_INCOMPLETE: DelegatedReleaseReplyTag.CLEANUP_INCOMPLETE,
        }[reply.result.status]
        if status_tag is not tag:
            raise RegionControlError(
                RegionControlErrorKind.INVALID_FIELD_VALUE,
                "release tag does not match ProviderReleaseResult.status",
            )
        mask = payload_op = payload_cause = counter_op = counter_cause = 0
        if reply.result.status is ProviderReleaseStatus.CLEANUP_INCOMPLETE:
            for failure in reply.result.failures:
                if failure.part is RegionPartKind.PAYLOAD:
                    mask |= FAILURE_MASK_PAYLOAD
                    payload_op = int(failure.backend_operation)
                    payload_cause = int(failure.typed_cause)
                else:
                    mask |= FAILURE_MASK_COUNTER
                    counter_op = int(failure.backend_operation)
                    counter_cause = int(failure.typed_cause)
            if mask == 0:
                raise RegionControlError(
                    RegionControlErrorKind.INTERNAL_INVARIANT,
                    "CLEANUP_INCOMPLETE requires a nonzero failure mask",
                )
        _RELEASE_OUTCOME.pack_into(
            frame,
            RELEASE_OUTCOME_OFFSET,
            int(reply.result.provider_resource_id),
            mask,
            0,
            payload_op,
            payload_cause,
            counter_op,
            counter_cause,
        )
    _REPLY_TAG.pack_into(frame, REPLY_TAG_OFFSET, int(tag))
    return bytes(frame)


def _decode_release_reply(envelope: DelegatedRegionReplyEnvelope) -> DelegatedReleaseReply:
    tag = _require_enum(DelegatedReleaseReplyTag, envelope.reply_tag, allow_zero=True)
    if tag is DelegatedReleaseReplyTag.EMPTY:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, "release reply is missing commit tag")
    resource_id, mask, error_kind, payload_op, payload_cause, counter_op, counter_cause = _RELEASE_OUTCOME.unpack_from(
        envelope.frame, RELEASE_OUTCOME_OFFSET
    )
    if tag is DelegatedReleaseReplyTag.UNKNOWN_TRANSACTION:
        if (
            int(resource_id) != 0
            or int(mask) != 0
            or int(error_kind) != 0
            or int(payload_op) != 0
            or int(payload_cause) != 0
            or int(counter_op) != 0
            or int(counter_cause) != 0
        ):
            raise RegionControlError(
                RegionControlErrorKind.INVALID_FIELD_VALUE,
                "UNKNOWN_TRANSACTION requires a zero outcome",
            )
        return DelegatedReleaseReply(
            tag=tag,
            session_instance_id=envelope.session_instance_id,
            transaction_id=envelope.transaction_id,
        )
    if tag is DelegatedReleaseReplyTag.ERROR:
        typed = _require_enum(RegionControlErrorKind, int(error_kind))
        if typed not in _RELEASE_ERROR_KINDS:
            raise RegionControlError(
                RegionControlErrorKind.INVALID_ENUM_VALUE,
                "release ERROR kind must be INVALID_FIELD_VALUE, STORE_LIFECYCLE, or INTERNAL_INVARIANT",
            )
        if (
            int(mask) != 0
            or int(payload_op) != 0
            or int(payload_cause) != 0
            or int(counter_op) != 0
            or int(counter_cause) != 0
        ):
            raise RegionControlError(
                RegionControlErrorKind.INVALID_FIELD_VALUE,
                "release ERROR requires inactive failure entries",
            )
        return DelegatedReleaseReply(
            tag=tag,
            session_instance_id=envelope.session_instance_id,
            transaction_id=envelope.transaction_id,
            error_kind=typed,
            error=RegionControlError(typed),
        )
    if int(error_kind) != 0:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE, "clean release tags require error_kind NONE"
        )
    if tag is DelegatedReleaseReplyTag.CLEANUP_INCOMPLETE:
        failures = _decode_cleanup_incomplete_failures(
            int(mask), int(payload_op), int(payload_cause), int(counter_op), int(counter_cause)
        )
        result = ProviderReleaseResult(
            provider_resource_id=int(resource_id),
            status=ProviderReleaseStatus.CLEANUP_INCOMPLETE,
            failures=tuple(failures),
        )
        return DelegatedReleaseReply(
            tag=tag,
            session_instance_id=envelope.session_instance_id,
            transaction_id=envelope.transaction_id,
            result=result,
        )
    if (
        int(mask) != 0
        or int(payload_op) != 0
        or int(payload_cause) != 0
        or int(counter_op) != 0
        or int(counter_cause) != 0
    ):
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            "clean release tags require inactive failure entries",
        )
    status = {
        DelegatedReleaseReplyTag.RELEASED: ProviderReleaseStatus.RELEASED,
        DelegatedReleaseReplyTag.ALREADY_GONE: ProviderReleaseStatus.ALREADY_GONE,
        DelegatedReleaseReplyTag.UNKNOWN_RESOURCE: ProviderReleaseStatus.UNKNOWN_RESOURCE,
    }[tag]
    result = ProviderReleaseResult(provider_resource_id=int(resource_id), status=status)
    return DelegatedReleaseReply(
        tag=tag,
        session_instance_id=envelope.session_instance_id,
        transaction_id=envelope.transaction_id,
        result=result,
    )


def _decode_cleanup_incomplete_failures(
    mask: int,
    payload_op: int,
    payload_cause: int,
    counter_op: int,
    counter_cause: int,
) -> list[ProviderCleanupFailure]:
    if mask & ~(FAILURE_MASK_PAYLOAD | FAILURE_MASK_COUNTER):
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, "failure mask has unknown bits")
    if mask == 0:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            "CLEANUP_INCOMPLETE requires a nonzero failure mask",
        )
    failures: list[ProviderCleanupFailure] = []
    if mask & FAILURE_MASK_PAYLOAD:
        failures.append(
            ProviderCleanupFailure(
                part=RegionPartKind.PAYLOAD,
                backend_operation=_require_enum(RegionOperationKind, payload_op),
                typed_cause=_require_enum(RegionCleanupCause, payload_cause),
            )
        )
    elif payload_op != 0 or payload_cause != 0:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            "unselected PAYLOAD failure entry must be zero",
        )
    if mask & FAILURE_MASK_COUNTER:
        failures.append(
            ProviderCleanupFailure(
                part=RegionPartKind.COUNTER,
                backend_operation=_require_enum(RegionOperationKind, counter_op),
                typed_cause=_require_enum(RegionCleanupCause, counter_cause),
            )
        )
    elif counter_op != 0 or counter_cause != 0:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            "unselected COUNTER failure entry must be zero",
        )
    return failures


def _encode_buffer_descriptor(view: bytearray, offset: int, descriptor: BufferDescriptor) -> None:
    try:
        wire = bytes(_encode_buffer_descriptor_wire(descriptor))
    except Exception as exc:
        raise RegionControlError(
            RegionControlErrorKind.INTERNAL_INVARIANT,
            str(exc) or "canonical BufferDescriptor encode failed",
        ) from exc
    if len(wire) != BUFFER_DESCRIPTOR_WIRE_BYTES:
        raise RegionControlError(
            RegionControlErrorKind.INTERNAL_INVARIANT,
            "canonical BufferDescriptor wire must be 88 bytes",
        )
    view[offset : offset + BUFFER_DESCRIPTOR_WIRE_BYTES] = wire


def _decode_buffer_descriptor(view: bytes, offset: int) -> BufferDescriptor:
    raw = bytes(view[offset : offset + BUFFER_DESCRIPTOR_WIRE_BYTES])
    if len(raw) != BUFFER_DESCRIPTOR_WIRE_BYTES:
        raise RegionControlError(RegionControlErrorKind.BAD_MESSAGE_SIZE, "BufferDescriptor slot must be 88 bytes")
    try:
        return _decode_buffer_descriptor_wire(raw)
    except Exception as exc:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            str(exc) or "BufferDescriptor is invalid",
        ) from exc


def _validate_decoded_descriptor_pair(descriptor: RegionExportDescriptor) -> None:
    payload = descriptor.payload
    counter = descriptor.counter
    if payload.access is not AccessMode.READWRITE or counter.access is not AccessMode.READWRITE:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            "region BufferDescriptors must be READWRITE",
        )
    if int(payload.identity.generation) != 1 or int(counter.identity.generation) != 1:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            "region BufferDescriptor generation must be 1",
        )
    if payload.identity == counter.identity:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            "PAYLOAD and COUNTER identities must differ",
        )
    if bytes(payload.identity.owner_instance_id) != bytes(counter.identity.owner_instance_id):
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            "PAYLOAD and COUNTER must share one owner nonce",
        )


def _encode_local_view(view: bytearray, offset: int, local_view: RegionPartLocalView) -> None:
    _LOCAL_VIEW.pack_into(
        view,
        offset,
        int(local_view.part),
        0,
        int(local_view.local_base),
        int(local_view.logical_bytes),
    )


def _decode_local_view(view: bytes, offset: int, expected: RegionPartKind) -> RegionPartLocalView:
    part, reserved, local_base, logical_bytes = _LOCAL_VIEW.unpack_from(view, offset)
    if int(reserved) != 0:
        raise RegionControlError(RegionControlErrorKind.RESERVED_NONZERO, "local-view reserved must be zero")
    typed_part = _require_enum(RegionPartKind, int(part))
    if typed_part is not expected:
        raise RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, "local-view part does not match slot")
    try:
        return RegionPartLocalView(part=expected, local_base=int(local_base), logical_bytes=int(logical_bytes))
    except (TypeError, ValueError) as exc:
        raise RegionControlError(
            RegionControlErrorKind.INVALID_FIELD_VALUE,
            str(exc) or "local-view fields are invalid",
        ) from exc


def _require_zero_span(view: bytes, offset: int, length: int) -> None:
    if any(view[offset : offset + length]):
        raise RegionControlError(RegionControlErrorKind.RESERVED_NONZERO, "reserved bytes must be zero")


class ProviderTransactionState(str, Enum):
    CREATING = "CREATING"
    ACTIVE = "ACTIVE"
    RELEASING = "RELEASING"
    TERMINAL_FAILURE = "TERMINAL_FAILURE"


@dataclass
class _ProviderTransactionRecord:
    session_instance_id: bytes
    transaction_id: int
    provider_path: bytes
    body_bytes: bytes
    state: ProviderTransactionState
    provider_resource_id: int = 0
    allocated_reply: bytes | None = None
    terminal_operation: DelegatedRegionOperation | None = None
    terminal_reply: bytes | None = None


class _ProviderRegionStoreOps(Protocol):
    def allocate_and_export(self, spec: RegionAllocationSpec) -> RegionAllocationResult: ...

    def local_view(self, provider_resource_id: int, part: RegionPartKind) -> RegionPartLocalView: ...

    def release(self, provider_resource_id: int) -> ProviderReleaseResult: ...


class ProviderTransactionTable:
    def __init__(self) -> None:
        self._records: dict[tuple[bytes, int], _ProviderTransactionRecord] = {}
        self._waterline: dict[bytes, int] = {}

    def execute(
        self, request: DelegatedAllocateRequest | DelegatedReleaseRequest, store: _ProviderRegionStoreOps
    ) -> bytes:
        if isinstance(request, DelegatedAllocateRequest):
            return self._execute_allocate(request, store)
        if isinstance(request, DelegatedReleaseRequest):
            return self._execute_release(request, store)
        raise TypeError("request must be DelegatedAllocateRequest or DelegatedReleaseRequest")

    def _key(self, session_instance_id: bytes, transaction_id: int) -> tuple[bytes, int]:
        return (session_instance_id, transaction_id)

    def _execute_allocate(self, request: DelegatedAllocateRequest, store: _ProviderRegionStoreOps) -> bytes:
        key = self._key(request.session_instance_id, request.transaction_id)
        body = request.projection_bytes + request.initiator_path
        record = self._records.get(key)
        if record is not None:
            return self._replay_or_conflict(record, request, body)
        waterline = self._waterline.get(request.session_instance_id, 0)
        if request.transaction_id <= waterline:
            return encode_reply(
                DelegatedAllocateReply(
                    tag=DelegatedAllocateReplyTag.ERROR,
                    session_instance_id=request.session_instance_id,
                    transaction_id=request.transaction_id,
                    error_kind=RegionControlErrorKind.STORE_LIFECYCLE,
                )
            )
        self._waterline[request.session_instance_id] = request.transaction_id
        record = _ProviderTransactionRecord(
            session_instance_id=request.session_instance_id,
            transaction_id=request.transaction_id,
            provider_path=request.provider_path,
            body_bytes=body,
            state=ProviderTransactionState.CREATING,
        )
        self._records[key] = record
        try:
            result = store.allocate_and_export(request.spec)
        except RegionAllocationError as exc:
            return self._finish_allocate_error(key, record, request, exc)
        except RegionControlError as exc:
            return self._finish_allocate_control_error(key, record, request, exc)
        record.provider_resource_id = int(result.provider_resource_id)
        try:
            payload_view = store.local_view(result.provider_resource_id, RegionPartKind.PAYLOAD)
            counter_view = store.local_view(result.provider_resource_id, RegionPartKind.COUNTER)
            committed = encode_reply(
                DelegatedAllocateReply(
                    tag=DelegatedAllocateReplyTag.ALLOCATED,
                    session_instance_id=request.session_instance_id,
                    transaction_id=request.transaction_id,
                    result=result,
                    payload_view=payload_view,
                    counter_view=counter_view,
                )
            )
        except BaseException:
            return self._compensate_before_allocated_tag(key, record, request, store)
        record.state = ProviderTransactionState.ACTIVE
        record.allocated_reply = committed
        return committed

    def _replay_or_conflict(
        self, record: _ProviderTransactionRecord, request: DelegatedAllocateRequest, body: bytes
    ) -> bytes:
        if record.provider_path != request.provider_path or record.body_bytes != body:
            return encode_reply(
                DelegatedAllocateReply(
                    tag=DelegatedAllocateReplyTag.ERROR,
                    session_instance_id=request.session_instance_id,
                    transaction_id=request.transaction_id,
                    error_kind=RegionControlErrorKind.INVALID_FIELD_VALUE,
                )
            )
        if record.state is ProviderTransactionState.ACTIVE:
            if record.allocated_reply is None:
                raise RegionControlError(
                    RegionControlErrorKind.INTERNAL_INVARIANT,
                    "ACTIVE record is missing the committed allocate reply cache",
                )
            return record.allocated_reply
        if record.state is ProviderTransactionState.TERMINAL_FAILURE:
            return self._terminal_reply_for_operation(record, DelegatedRegionOperation.DELEGATED_ALLOCATE)
        raise RegionControlError(
            RegionControlErrorKind.INTERNAL_INVARIANT,
            f"{record.state.value} is not observable across serial provider-agent calls",
        )

    def _terminal_reply_for_operation(
        self, record: _ProviderTransactionRecord, operation: DelegatedRegionOperation
    ) -> bytes:
        if record.terminal_operation is None or record.terminal_reply is None:
            raise RegionControlError(
                RegionControlErrorKind.INTERNAL_INVARIANT,
                "TERMINAL_FAILURE record is missing its diagnostic reply",
            )
        if operation is record.terminal_operation:
            return record.terminal_reply
        if operation is DelegatedRegionOperation.DELEGATED_ALLOCATE:
            return encode_reply(
                DelegatedAllocateReply(
                    tag=DelegatedAllocateReplyTag.ERROR,
                    session_instance_id=record.session_instance_id,
                    transaction_id=record.transaction_id,
                    error_kind=RegionControlErrorKind.STORE_LIFECYCLE,
                )
            )
        if operation is DelegatedRegionOperation.DELEGATED_RELEASE:
            return encode_reply(
                DelegatedReleaseReply(
                    tag=DelegatedReleaseReplyTag.ERROR,
                    session_instance_id=record.session_instance_id,
                    transaction_id=record.transaction_id,
                    error_kind=RegionControlErrorKind.STORE_LIFECYCLE,
                )
            )
        raise RegionControlError(
            RegionControlErrorKind.INTERNAL_INVARIANT,
            f"unsupported terminal replay operation {int(operation)}",
        )

    def _enter_terminal_failure(
        self,
        record: _ProviderTransactionRecord,
        operation: DelegatedRegionOperation,
        committed: bytes,
    ) -> bytes:
        record.terminal_operation = operation
        record.terminal_reply = committed
        record.state = ProviderTransactionState.TERMINAL_FAILURE
        return committed

    def _finish_allocate_error(
        self,
        key: tuple[bytes, int],
        record: _ProviderTransactionRecord,
        request: DelegatedAllocateRequest,
        error: RegionAllocationError,
    ) -> bytes:
        committed = encode_reply(
            DelegatedAllocateReply(
                tag=DelegatedAllocateReplyTag.ERROR,
                session_instance_id=request.session_instance_id,
                transaction_id=request.transaction_id,
                error=error,
            )
        )
        if error.control_kind is RegionControlErrorKind.BACKEND_FAILURE and not error.cleanup_debt_remaining:
            del self._records[key]
            return committed
        record.provider_resource_id = int(error.provisional_resource_id)
        return self._enter_terminal_failure(record, DelegatedRegionOperation.DELEGATED_ALLOCATE, committed)

    def _finish_allocate_control_error(
        self,
        key: tuple[bytes, int],
        record: _ProviderTransactionRecord,
        request: DelegatedAllocateRequest,
        error: RegionControlError,
    ) -> bytes:
        committed = encode_reply(
            DelegatedAllocateReply(
                tag=DelegatedAllocateReplyTag.ERROR,
                session_instance_id=request.session_instance_id,
                transaction_id=request.transaction_id,
                error_kind=error.kind
                if error.kind in _ALLOCATE_PROTOCOL_ERROR_KINDS
                else RegionControlErrorKind.INTERNAL_INVARIANT,
            )
        )
        return self._enter_terminal_failure(record, DelegatedRegionOperation.DELEGATED_ALLOCATE, committed)

    def _compensate_before_allocated_tag(
        self,
        key: tuple[bytes, int],
        record: _ProviderTransactionRecord,
        request: DelegatedAllocateRequest,
        store: _ProviderRegionStoreOps,
    ) -> bytes:
        try:
            result = store.release(record.provider_resource_id)
        except RegionControlError:
            committed = encode_reply(
                DelegatedAllocateReply(
                    tag=DelegatedAllocateReplyTag.ERROR,
                    session_instance_id=request.session_instance_id,
                    transaction_id=request.transaction_id,
                    error=RegionAllocationError(
                        provisional_resource_id=record.provider_resource_id,
                        control_kind=RegionControlErrorKind.INTERNAL_INVARIANT,
                        failed_part=RegionPartKind.INVALID,
                        failed_operation=RegionOperationKind.LOCAL_VIEW,
                        cleanup_debt_remaining=True,
                    ),
                )
            )
            return self._enter_terminal_failure(record, DelegatedRegionOperation.DELEGATED_ALLOCATE, committed)
        if result.status in (ProviderReleaseStatus.RELEASED, ProviderReleaseStatus.ALREADY_GONE):
            del self._records[key]
            return encode_reply(
                DelegatedAllocateReply(
                    tag=DelegatedAllocateReplyTag.ERROR,
                    session_instance_id=request.session_instance_id,
                    transaction_id=request.transaction_id,
                    error_kind=RegionControlErrorKind.INTERNAL_INVARIANT,
                )
            )
        committed = encode_reply(
            DelegatedAllocateReply(
                tag=DelegatedAllocateReplyTag.ERROR,
                session_instance_id=request.session_instance_id,
                transaction_id=request.transaction_id,
                error=RegionAllocationError(
                    provisional_resource_id=record.provider_resource_id,
                    control_kind=RegionControlErrorKind.INTERNAL_INVARIANT,
                    failed_part=RegionPartKind.INVALID,
                    failed_operation=RegionOperationKind.LOCAL_VIEW,
                    cleanup_debt_remaining=True,
                ),
            )
        )
        return self._enter_terminal_failure(record, DelegatedRegionOperation.DELEGATED_ALLOCATE, committed)

    def _execute_release(self, request: DelegatedReleaseRequest, store: _ProviderRegionStoreOps) -> bytes:
        key = self._key(request.session_instance_id, request.transaction_id)
        record = self._records.get(key)
        if record is None:
            return encode_reply(
                DelegatedReleaseReply(
                    tag=DelegatedReleaseReplyTag.UNKNOWN_TRANSACTION,
                    session_instance_id=request.session_instance_id,
                    transaction_id=request.transaction_id,
                )
            )
        if record.provider_path != request.provider_path:
            return encode_reply(
                DelegatedReleaseReply(
                    tag=DelegatedReleaseReplyTag.ERROR,
                    session_instance_id=request.session_instance_id,
                    transaction_id=request.transaction_id,
                    error_kind=RegionControlErrorKind.INVALID_FIELD_VALUE,
                )
            )
        if record.state is ProviderTransactionState.TERMINAL_FAILURE:
            return self._terminal_reply_for_operation(record, DelegatedRegionOperation.DELEGATED_RELEASE)
        if record.state is not ProviderTransactionState.ACTIVE:
            raise RegionControlError(
                RegionControlErrorKind.INTERNAL_INVARIANT,
                f"{record.state.value} is not observable across serial provider-agent calls",
            )
        record.state = ProviderTransactionState.RELEASING
        try:
            result = store.release(record.provider_resource_id)
        except RegionControlError as exc:
            kind = exc.kind if exc.kind in _RELEASE_ERROR_KINDS else RegionControlErrorKind.INTERNAL_INVARIANT
            committed = encode_reply(
                DelegatedReleaseReply(
                    tag=DelegatedReleaseReplyTag.ERROR,
                    session_instance_id=request.session_instance_id,
                    transaction_id=request.transaction_id,
                    error_kind=kind,
                )
            )
            return self._enter_terminal_failure(record, DelegatedRegionOperation.DELEGATED_RELEASE, committed)
        if result.status in (ProviderReleaseStatus.RELEASED, ProviderReleaseStatus.ALREADY_GONE):
            committed = encode_reply(
                DelegatedReleaseReply(
                    tag=(
                        DelegatedReleaseReplyTag.RELEASED
                        if result.status is ProviderReleaseStatus.RELEASED
                        else DelegatedReleaseReplyTag.ALREADY_GONE
                    ),
                    session_instance_id=request.session_instance_id,
                    transaction_id=request.transaction_id,
                    result=result,
                )
            )
            del self._records[key]
            return committed
        tag = {
            ProviderReleaseStatus.UNKNOWN_RESOURCE: DelegatedReleaseReplyTag.UNKNOWN_RESOURCE,
            ProviderReleaseStatus.CLEANUP_INCOMPLETE: DelegatedReleaseReplyTag.CLEANUP_INCOMPLETE,
        }[result.status]
        committed = encode_reply(
            DelegatedReleaseReply(
                tag=tag,
                session_instance_id=request.session_instance_id,
                transaction_id=request.transaction_id,
                result=result,
            )
        )
        return self._enter_terminal_failure(record, DelegatedRegionOperation.DELEGATED_RELEASE, committed)


def handle_terminal_delegated_region(
    payload: memoryview, table: ProviderTransactionTable, store: _ProviderRegionStoreOps
) -> None:
    view = _as_writable_bytes(payload)
    try:
        request = parse_request(view).decode_terminal()
    except RegionControlError as exc:
        _publish_invalid_terminal_request(view, exc)
        return
    publish_reply(view, table.execute(request, store))


def _publish_invalid_terminal_request(view: memoryview, error: RegionControlError) -> None:
    raw = view.tobytes()
    if len(raw) < REQUEST_HEADER_BYTES:
        return
    magic, _request_bytes, operation, session, transaction_id, _provider, _initiator = _REQUEST_HEADER.unpack_from(
        raw, 0
    )
    if int(magic) != DELEGATED_REGION_CTRL_MAGIC_VERSION:
        return
    try:
        op = _require_operation(int(operation))
        session_id = _require_session(session)
        tx = _require_transaction_id(int(transaction_id))
    except RegionControlError:
        return
    if view.nbytes < _operation_fixed_reply_bytes(op):
        return
    if op is DelegatedRegionOperation.DELEGATED_ALLOCATE:
        kind = (
            error.kind if error.kind in _ALLOCATE_PROTOCOL_ERROR_KINDS else RegionControlErrorKind.INVALID_FIELD_VALUE
        )
        committed = encode_reply(
            DelegatedAllocateReply(
                tag=DelegatedAllocateReplyTag.ERROR,
                session_instance_id=session_id,
                transaction_id=tx,
                error_kind=kind,
            )
        )
    else:
        kind = error.kind if error.kind in _RELEASE_ERROR_KINDS else RegionControlErrorKind.INVALID_FIELD_VALUE
        committed = encode_reply(
            DelegatedReleaseReply(
                tag=DelegatedReleaseReplyTag.ERROR,
                session_instance_id=session_id,
                transaction_id=tx,
                error_kind=kind,
            )
        )
    publish_reply(view, committed)
