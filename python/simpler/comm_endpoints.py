# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Endpoint selectors, registry resolution, and backend planning."""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Protocol

from _task_interface import OWNER_INSTANCE_ID_BYTES, BackendKind  # pyright: ignore[reportMissingImports]

EndpointId = int
NodeScopeId = int


class EndpointDeployment(str, Enum):
    HOST_CPU = "HOST_CPU"
    DEVICE_AICORE = "DEVICE_AICORE"
    DEVICE_AICPU = "DEVICE_AICPU"


HOST_CPU = EndpointDeployment.HOST_CPU
DEVICE_AICORE = EndpointDeployment.DEVICE_AICORE
DEVICE_AICPU = EndpointDeployment.DEVICE_AICPU


class EndpointDeploymentKind(IntEnum):
    INVALID = 0
    HOST_CPU = 1
    DEVICE_AICORE = 2
    DEVICE_AICPU = 3


class RegionTopologyKind(IntEnum):
    INVALID = 0
    SINGLE_OWNER = 1


class EndpointSelectorKind(str, Enum):
    AT = "AT"
    UNDER = "UNDER"


@dataclass(frozen=True)
class EndpointSelector:
    kind: EndpointSelectorKind
    path: str
    deployment: EndpointDeployment

    def __post_init__(self) -> None:
        kind, path, deployment = _normalize_selector_values(self.kind, self.path, self.deployment)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "deployment", deployment)


def at(path: str, deployment: EndpointDeployment | str) -> EndpointSelector:
    kind, normalized_path, normalized_deployment = _normalize_selector_values(
        EndpointSelectorKind.AT,
        path,
        deployment,
    )
    return EndpointSelector(kind, normalized_path, normalized_deployment)


def under(path: str, deployment: EndpointDeployment | str) -> EndpointSelector:
    kind, normalized_path, normalized_deployment = _normalize_selector_values(
        EndpointSelectorKind.UNDER,
        path,
        deployment,
    )
    return EndpointSelector(kind, normalized_path, normalized_deployment)


def _normalize_selector_values(
    kind: EndpointSelectorKind | str, path: str, deployment: EndpointDeployment | str
) -> tuple[EndpointSelectorKind, str, EndpointDeployment]:
    try:
        kind_value = EndpointSelectorKind(kind)
    except ValueError as exc:
        raise ValueError(f"invalid endpoint selector kind {kind!r}") from exc
    if not isinstance(path, str) or not path:
        raise ValueError("endpoint path must be a non-empty string")
    if any(segment == "" for segment in path.split("/")):
        raise ValueError(f"endpoint path must not contain empty segments: {path!r}")
    try:
        deployment_value = EndpointDeployment(deployment)
    except ValueError as exc:
        raise ValueError(f"invalid endpoint deployment {deployment!r}") from exc
    return kind_value, path, deployment_value


@dataclass(frozen=True)
class EndpointPathSegment:
    level: int
    index: int | None = None


@dataclass(frozen=True)
class ParsedEndpointPath:
    segments: tuple[EndpointPathSegment, ...]
    text: str

    @property
    def sort_key(self) -> tuple[tuple[int, int], ...]:
        return tuple((segment.level, -1 if segment.index is None else segment.index) for segment in self.segments)


_PATH_SEGMENT_RE = re.compile(r"^L(?P<level>[0-9]+)(?:\[(?P<index>[0-9]+)\])?$")


def parse_endpoint_path(path: str, *, root_level: int) -> ParsedEndpointPath:
    if not isinstance(path, str) or not path:
        raise ValueError("endpoint path must be a non-empty string")
    raw_segments = path.split("/")
    if any(segment == "" for segment in raw_segments):
        raise ValueError(f"invalid endpoint path {path!r}: empty segment")
    segments: list[EndpointPathSegment] = []
    for i, raw in enumerate(raw_segments):
        match = _PATH_SEGMENT_RE.match(raw)
        if match is None:
            raise ValueError(f"invalid endpoint path segment {raw!r} in {path!r}")
        level = int(match.group("level"))
        index_s = match.group("index")
        index = None if index_s is None else int(index_s)
        if i == 0:
            if level != root_level or index is not None:
                raise ValueError(f"endpoint path {path!r} must start with root L{root_level}")
        elif index is None:
            raise ValueError(f"child endpoint path segment {raw!r} in {path!r} must include an index")
        segments.append(EndpointPathSegment(level=level, index=index))
    return ParsedEndpointPath(segments=tuple(segments), text=path)


def _format_worker_path(level: int, *, parent_path: str | None = None, index: int | None = None) -> str:
    segment = f"L{int(level)}" if index is None else f"L{int(level)}[{int(index)}]"
    return segment if parent_path is None else f"{parent_path}/{segment}"


def _normalize_node_identity(host: str) -> str:
    normalized = str(host).strip().lower()
    if normalized == "localhost":
        return "local"
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return normalized
    if address.is_loopback:
        return "local"
    return str(address).lower()


def _require_owner_instance_id(nonce: object) -> bytes:
    if not isinstance(nonce, (bytes, bytearray)):
        raise ValueError("owner nonce must be bytes")
    value = bytes(nonce)
    if len(value) != OWNER_INSTANCE_ID_BYTES:
        raise ValueError(f"owner nonce must be {OWNER_INSTANCE_ID_BYTES} bytes")
    if value == b"\x00" * OWNER_INSTANCE_ID_BYTES:
        raise ValueError("owner nonce must be nonzero")
    return value


@dataclass(frozen=True)
class _EndpointTopologyEntry:
    path: str
    deployment: EndpointDeployment
    node_identity: str
    #: Endpoint incarnation nonce. Local HOST_CPU/AICPU/AICORE entries carry one; Remote/MPI
    #: entries leave it absent. Allocator presence is a separate frozen startup fact.
    owner_instance_id: bytes | None = None
    has_buffer_identity_allocator: bool | None = None


@dataclass(frozen=True)
class _EndpointTopologySnapshot:
    root_level: int
    session_instance_id: bytes
    entries: tuple[_EndpointTopologyEntry, ...]


@dataclass(frozen=True)
class SingleOwner:
    provider: EndpointSelector | None = None


@dataclass(frozen=True)
class EndpointIdentity:
    session_instance_id: bytes
    registry_epoch: int
    endpoint_id: EndpointId


@dataclass(frozen=True)
class EndpointRecord:
    identity: EndpointIdentity
    path: str
    deployment: EndpointDeployment
    node_scope_id: NodeScopeId

    @property
    def endpoint_id(self) -> EndpointId:
        return self.identity.endpoint_id


@dataclass(frozen=True)
class ResolvedSingleOwner:
    provider_endpoint: EndpointIdentity | None


@dataclass(frozen=True)
class ResolvedRegionSpec:
    members: tuple[EndpointRecord, ...]
    topology: ResolvedSingleOwner


class EndpointResolveReason(str, Enum):
    PATH_NOT_FOUND = "PATH_NOT_FOUND"
    ENDPOINT_NOT_REGISTERED = "ENDPOINT_NOT_REGISTERED"
    EMPTY_UNDER_SELECTOR = "EMPTY_UNDER_SELECTOR"
    DUPLICATE_ENDPOINT = "DUPLICATE_ENDPOINT"
    INVALID_PATH = "INVALID_PATH"
    PROVIDER_NOT_SINGLE_ENDPOINT = "PROVIDER_NOT_SINGLE_ENDPOINT"
    OWNER_NOT_REGISTERED = "OWNER_NOT_REGISTERED"


class EndpointResolveError(ValueError):
    def __init__(self, reason: EndpointResolveReason, message: str, offending: Sequence[str] = ()) -> None:
        self.reason = reason
        self.message = message
        self.offending = tuple(offending)
        super().__init__(message)


class EndpointRegistry:
    def __init__(
        self,
        *,
        root_level: int,
        session_instance_id: bytes,
        registry_epoch: int,
        records: Sequence[EndpointRecord],
        owner_bindings: Mapping[bytes, EndpointId] = MappingProxyType({}),
        allocator_states: Mapping[EndpointId, bool | None] = MappingProxyType({}),
    ) -> None:
        self.root_level = int(root_level)
        self.session_instance_id = bytes(session_instance_id)
        self.registry_epoch = int(registry_epoch)
        self._records = tuple(records)
        self._by_id = {record.endpoint_id: record for record in self._records}
        self._by_identity = {record.identity: record for record in self._records}
        self._by_key = {(record.path, record.deployment): record for record in self._records}
        self._known_paths = {record.path for record in self._records}
        self._parsed_paths = {path: self._parse(path) for path in self._known_paths}
        self._by_owner_instance_id = {
            bytes(nonce): self._by_id[endpoint_id] for nonce, endpoint_id in owner_bindings.items()
        }
        self._buffer_identity_allocator_states = {
            int(endpoint_id): None if state is None else bool(state) for endpoint_id, state in allocator_states.items()
        }

    @classmethod
    def from_snapshot(
        cls,
        snapshot: _EndpointTopologySnapshot,
        *,
        registry_epoch: int,
    ) -> EndpointRegistry:
        entries = tuple(snapshot.entries)
        session_instance_id = bytes(snapshot.session_instance_id)
        node_scopes = {"local": 0}
        next_node_scope_id = 1
        records: list[EndpointRecord] = []
        seen: set[tuple[str, EndpointDeployment]] = set()
        owner_bindings: dict[bytes, EndpointId] = {}
        allocator_states: dict[EndpointId, bool | None] = {}
        for endpoint_id, entry in enumerate(entries):
            deployment = EndpointDeployment(entry.deployment)
            key = (entry.path, deployment)
            if key in seen:
                raise ValueError(f"duplicate endpoint topology entry: {entry.path} {deployment.value}")
            seen.add(key)
            node_identity = _normalize_node_identity(entry.node_identity)
            node_scope_id = node_scopes.get(node_identity)
            if node_scope_id is None:
                node_scope_id = next_node_scope_id
                node_scopes[node_identity] = node_scope_id
                next_node_scope_id += 1
            identity = EndpointIdentity(
                session_instance_id=session_instance_id,
                registry_epoch=int(registry_epoch),
                endpoint_id=endpoint_id,
            )
            allocator_state = entry.has_buffer_identity_allocator
            if allocator_state is not None:
                allocator_state = bool(allocator_state)
            if entry.owner_instance_id is not None:
                nonce = _require_owner_instance_id(entry.owner_instance_id)
                bound = owner_bindings.get(nonce)
                if bound is not None:
                    raise ValueError(
                        f"owner nonce bound to two endpoints: {entry.path} {deployment.value} "
                        f"and {records[bound].path} {records[bound].deployment.value}"
                    )
                owner_bindings[nonce] = endpoint_id
            elif allocator_state is True:
                raise ValueError(f"allocator presence requires an owner nonce: {entry.path} {deployment.value}")
            allocator_states[endpoint_id] = allocator_state
            records.append(
                EndpointRecord(
                    identity=identity,
                    path=entry.path,
                    deployment=deployment,
                    node_scope_id=node_scope_id,
                )
            )
        return cls(
            root_level=int(snapshot.root_level),
            session_instance_id=session_instance_id,
            registry_epoch=int(registry_epoch),
            records=records,
            owner_bindings=owner_bindings,
            allocator_states=allocator_states,
        )

    @property
    def records(self) -> tuple[EndpointRecord, ...]:
        return self._records

    def resolve_members(self, selectors: Sequence[EndpointSelector]) -> tuple[EndpointRecord, ...]:
        resolved: list[EndpointRecord] = []
        seen: dict[EndpointIdentity, EndpointRecord] = {}
        for selector in selectors:
            for record in self._resolve_selector(selector):
                duplicate = seen.get(record.identity)
                if duplicate is not None:
                    label = _endpoint_label(record)
                    raise EndpointResolveError(
                        EndpointResolveReason.DUPLICATE_ENDPOINT,
                        f"duplicate endpoint in region members: {label}",
                        (label,),
                    )
                seen[record.identity] = record
                resolved.append(record)
        return tuple(resolved)

    def resolve_region_spec(self, members: Sequence[EndpointSelector], topology: SingleOwner) -> ResolvedRegionSpec:
        resolved_members = self.resolve_members(members)
        if not isinstance(topology, SingleOwner):
            raise TypeError("Region planning supports SingleOwner topology only")
        provider_endpoint = None
        if topology.provider is not None:
            provider_records = self._resolve_provider(topology.provider)
            provider_endpoint = provider_records[0].identity
        return ResolvedRegionSpec(
            members=resolved_members,
            topology=ResolvedSingleOwner(provider_endpoint=provider_endpoint),
        )

    def same_node(
        self, a: EndpointRecord | EndpointIdentity | EndpointId, b: EndpointRecord | EndpointIdentity | EndpointId
    ) -> bool:
        return self._record_for(a).node_scope_id == self._record_for(b).node_scope_id

    def record_for(self, endpoint: EndpointRecord | EndpointIdentity | EndpointId) -> EndpointRecord:
        """Return this registry's canonical record for a record, identity, or endpoint id."""
        return self._record_for(endpoint)

    def owner_endpoint(self, owner_instance_id: bytes) -> EndpointRecord:
        """The endpoint that minted buffers carrying `owner_instance_id`.

        The one direction a `BufferDescriptor` cannot answer for itself: `owner_instance_id` is an
        opaque nonce, `owner_worker_path_id` is diagnostic by contract, and `address_space`
        distinguishes only HOST from DEVICE — none of them names a card. Resolving the owner is
        therefore a registry judgement, and this is the binding that makes it one, so a consumer
        deciding whether a device reference may reach it has a deployment to test against.

        Raises when the nonce belongs to a Worker this registry cannot see — a remote Worker mints
        in its own process, and its binding has to arrive over a session channel rather than be
        inferred here.
        """
        record = self._by_owner_instance_id.get(bytes(owner_instance_id))
        if record is None:
            raise EndpointResolveError(
                EndpointResolveReason.OWNER_NOT_REGISTERED,
                f"no endpoint in this registry minted buffers under owner {bytes(owner_instance_id).hex()}",
                (bytes(owner_instance_id).hex(),),
            )
        return record

    def _buffer_identity_allocator_state(self, endpoint: EndpointRecord | EndpointIdentity | EndpointId) -> bool | None:
        record = self._record_for(endpoint)
        return self._buffer_identity_allocator_states.get(record.endpoint_id)

    def all_same_node(self, members: Sequence[EndpointRecord]) -> bool:
        if len(members) <= 1:
            return True
        first = members[0].node_scope_id
        return all(member.node_scope_id == first for member in members)

    def _resolve_provider(self, selector: EndpointSelector) -> tuple[EndpointRecord, ...]:
        try:
            records = self._resolve_selector(selector)
        except EndpointResolveError as exc:
            raise self._provider_error(selector) from exc
        if len(records) != 1:
            raise self._provider_error(selector, records)
        return records

    def _provider_error(
        self, selector: EndpointSelector, records: Sequence[EndpointRecord] = ()
    ) -> EndpointResolveError:
        labels = tuple(_endpoint_label(record) for record in records) or (_selector_label(selector),)
        return EndpointResolveError(
            EndpointResolveReason.PROVIDER_NOT_SINGLE_ENDPOINT,
            f"SingleOwner provider must resolve to exactly one endpoint: {_selector_label(selector)}",
            labels,
        )

    def _resolve_selector(self, selector: EndpointSelector) -> tuple[EndpointRecord, ...]:
        if not isinstance(selector, EndpointSelector):
            raise TypeError("region members must be EndpointSelector values")
        parsed = self._parse_or_error(selector.path)
        if selector.path not in self._known_paths:
            raise EndpointResolveError(
                EndpointResolveReason.PATH_NOT_FOUND,
                f"endpoint path not found: {_selector_label(selector)}",
                (_selector_label(selector),),
            )
        if selector.kind is EndpointSelectorKind.AT:
            record = self._by_key.get((selector.path, selector.deployment))
            if record is None:
                raise EndpointResolveError(
                    EndpointResolveReason.ENDPOINT_NOT_REGISTERED,
                    f"endpoint is not registered: {_selector_label(selector)}",
                    (_selector_label(selector),),
                )
            return (record,)
        if selector.kind is EndpointSelectorKind.UNDER:
            records = [
                record
                for record in self._records
                if record.deployment is selector.deployment and self._is_descendant(record.path, parsed)
            ]
            records.sort(key=lambda record: self._parsed_paths[record.path].sort_key)
            if not records:
                raise EndpointResolveError(
                    EndpointResolveReason.EMPTY_UNDER_SELECTOR,
                    f"endpoint selector expanded to no endpoints: {_selector_label(selector)}",
                    (_selector_label(selector),),
                )
            return tuple(records)
        raise TypeError(f"unsupported endpoint selector kind: {selector.kind!r}")

    def _is_descendant(self, path: str, parent: ParsedEndpointPath) -> bool:
        parsed = self._parsed_paths[path]
        parent_len = len(parent.segments)
        return len(parsed.segments) > parent_len and parsed.segments[:parent_len] == parent.segments

    def _record_for(self, endpoint: EndpointRecord | EndpointIdentity | EndpointId) -> EndpointRecord:
        if isinstance(endpoint, EndpointRecord):
            return endpoint
        if isinstance(endpoint, EndpointIdentity):
            record = self._by_identity.get(endpoint)
            if record is None:
                raise ValueError(
                    "unknown EndpointIdentity in registry "
                    f"session={self.session_instance_id!r} epoch={self.registry_epoch}"
                )
            return record
        try:
            return self._by_id[int(endpoint)]
        except KeyError as exc:
            raise ValueError(f"unknown endpoint_id {endpoint!r} in registry epoch {self.registry_epoch}") from exc

    def _parse(self, path: str) -> ParsedEndpointPath:
        return parse_endpoint_path(path, root_level=self.root_level)

    def _parse_or_error(self, path: str) -> ParsedEndpointPath:
        try:
            return self._parse(path)
        except ValueError as exc:
            raise EndpointResolveError(
                EndpointResolveReason.INVALID_PATH,
                f"invalid endpoint path for root L{self.root_level}: {path!r}",
                (path,),
            ) from exc


class RegionPartKind(str, Enum):
    PAYLOAD = "PAYLOAD"
    COUNTER = "COUNTER"


# These three enums plus the numeric tables below are a wire contract. Global CommDomain and
# delegated-region control both encode them as little-endian u32. A new enumerator needs a
# new id; a rename or renumber is a wire break.
class AttachmentRole(str, Enum):
    PROVIDER = "PROVIDER"
    CONSUMER = "CONSUMER"


class AdapterKind(str, Enum):
    DIRECT_MAP = "DIRECT_MAP"
    DEVICE_PEER = "DEVICE_PEER"
    OWNER_DELEGATED_COPY = "OWNER_DELEGATED_COPY"
    EXPLICIT_TRANSFER = "EXPLICIT_TRANSFER"
    COLLECTIVE = "COLLECTIVE"


class AdapterProfile(str, Enum):
    """How a consumer reaches a Buffer — the mechanism, not the wire backend tag."""

    # `HOST_SVM_MAP` currently has no implemented producer. It remains a named mechanism so the
    # service can reject it explicitly instead of conflating it with an unknown profile.
    HOST_SVM_MAP = "HOST_SVM_MAP"
    HOST_VMM_COPY = "HOST_VMM_COPY"
    DEVICE_VMM_PEER_IMPORT = "DEVICE_VMM_PEER_IMPORT"
    DEVICE_FABRIC_V2_PEER_IMPORT = "DEVICE_FABRIC_V2_PEER_IMPORT"
    REMOTE_COPY = "REMOTE_COPY"
    HOST_SHM_MAP = "HOST_SHM_MAP"
    # These profiles do not currently travel in a `GlobalDomainCommand`, which is why
    # `global_comm_domain._ADAPTER_PROFILE_IDS` does not number them. They still enter the same
    # Buffer candidate/service judgment as the profiles used by region planning.
    FORK_INHERITED_VA = "FORK_INHERITED_VA"
    DEVICE_LOCAL = "DEVICE_LOCAL"
    # A host endpoint reaching a chip-owned `DEVICE_MALLOC` Buffer by asking that chip to copy. The
    # planner has no counterpart because `_backend_kind_for_provider` never gives a region a
    # `DEVICE_MALLOC` Buffer; the `VMM_WINDOW` half of the same relation is `HOST_VMM_COPY`.
    OWNER_DEVICE_COPY = "OWNER_DEVICE_COPY"


_ADAPTER_KIND_IDS = {
    AdapterKind.DIRECT_MAP: 1,
    AdapterKind.DEVICE_PEER: 2,
    AdapterKind.OWNER_DELEGATED_COPY: 3,
    AdapterKind.EXPLICIT_TRANSFER: 4,
    AdapterKind.COLLECTIVE: 5,
}
_ADAPTER_KIND_BY_ID = {value: key for key, value in _ADAPTER_KIND_IDS.items()}
_ADAPTER_PROFILE_IDS = {
    AdapterProfile.HOST_SVM_MAP: 1,
    AdapterProfile.HOST_VMM_COPY: 2,
    AdapterProfile.DEVICE_VMM_PEER_IMPORT: 3,
    AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT: 4,
    AdapterProfile.HOST_SHM_MAP: 5,
    AdapterProfile.REMOTE_COPY: 6,
}
_ADAPTER_PROFILE_BY_ID = {value: key for key, value in _ADAPTER_PROFILE_IDS.items()}


def _adapter_kind_id(kind: AdapterKind | None) -> int:
    if kind is None:
        return 0
    try:
        return _ADAPTER_KIND_IDS[AdapterKind(kind)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"adapter_kind is unknown: {kind!r}") from exc


def _adapter_kind_from_id(value: int) -> AdapterKind | None:
    if value == 0:
        return None
    try:
        return _ADAPTER_KIND_BY_ID[int(value)]
    except KeyError as exc:
        raise ValueError(f"adapter_kind id is unknown: {value}") from exc


def _adapter_profile_id(profile: AdapterProfile | None) -> int:
    if profile is None:
        return 0
    try:
        return _ADAPTER_PROFILE_IDS[AdapterProfile(profile)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"adapter_profile is unknown: {profile!r}") from exc


def _adapter_profile_from_id(value: int) -> AdapterProfile | None:
    if value == 0:
        return None
    try:
        return _ADAPTER_PROFILE_BY_ID[int(value)]
    except KeyError as exc:
        raise ValueError(f"adapter_profile id is unknown: {value}") from exc


class RegionAccessReasonCode(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED_BACKEND_KIND = "UNSUPPORTED_BACKEND_KIND"
    UNSUPPORTED_ENDPOINT_RELATION = "UNSUPPORTED_ENDPOINT_RELATION"
    NO_IMPLEMENTED_DIRECT_MAP_PROBE = "NO_IMPLEMENTED_DIRECT_MAP_PROBE"
    NO_COPY_BACKEND = "NO_COPY_BACKEND"
    STATIC_UNSUPPORTED = "STATIC_UNSUPPORTED"


@dataclass(frozen=True)
class RegionLayoutSpec:
    payload_bytes: int
    counter_bytes: int


@dataclass(frozen=True)
class BufferAccessQuery:
    """Facts needed to judge one candidate adapter for one Buffer kind.

    Provider selection, region layout, and concrete Buffer authorization happen before this query.
    ``same_endpoint`` distinguishes a Buffer already local to the consumer from a same-node peer;
    backend plus deployment alone cannot distinguish DEVICE_LOCAL from DEVICE_PEER.
    """

    backend_kind: BackendKind
    consumer_deployment: EndpointDeployment
    same_node: bool
    same_endpoint: bool = False
    platform: str | None = None
    runtime: str | None = None


@dataclass(frozen=True)
class RegionAccessDiagnostics:
    reason_code: RegionAccessReasonCode
    message: str
    provider_label: str | None = None
    consumer_label: str | None = None
    platform: str | None = None
    runtime: str | None = None
    backend_kind: BackendKind | None = None
    consumer_deployment: EndpointDeployment | None = None
    same_node: bool | None = None
    same_endpoint: bool | None = None
    part: RegionPartKind | None = None
    adapter_kind: AdapterKind | None = None
    adapter_profile: AdapterProfile | None = None


@dataclass(frozen=True)
class RegionAccessDecision:
    supported: bool
    reason: str | None = None
    diagnostics: RegionAccessDiagnostics | None = None


@dataclass(frozen=True)
class MemberAttachmentPlan:
    member: EndpointIdentity
    role: AttachmentRole
    adapter_kind: AdapterKind | None
    adapter_profile: AdapterProfile | None


@dataclass(frozen=True)
class RegionPartPlan:
    part: RegionPartKind
    backend_kind: BackendKind
    attachments: tuple[MemberAttachmentPlan, ...]


@dataclass(frozen=True)
class SingleOwnerPlan:
    provider_endpoint: EndpointIdentity


@dataclass(frozen=True)
class BackendPlan:
    ordered_members: tuple[EndpointIdentity, ...]
    payload: RegionPartPlan
    counter: RegionPartPlan
    topology_plan: SingleOwnerPlan


class RegionAccessService(Protocol):
    def evaluate_buffer_access(
        self,
        query: BufferAccessQuery,
        candidate: _AdapterCandidate,
    ) -> RegionAccessDecision: ...


class BackendUnsupportedReason(str, Enum):
    PROVIDER_NOT_IN_MEMBERS = "PROVIDER_NOT_IN_MEMBERS"
    NO_DEFAULT_PROVIDER = "NO_DEFAULT_PROVIDER"
    ADAPTER_UNSUPPORTED = "ADAPTER_UNSUPPORTED"
    UNSUPPORTED_DEPLOYMENT_COMBINATION = "UNSUPPORTED_DEPLOYMENT_COMBINATION"


@dataclass(frozen=True)
class AdapterAttempt:
    part: RegionPartKind
    member: EndpointRecord
    backend_kind: BackendKind
    adapter_kind: AdapterKind
    adapter_profile: AdapterProfile
    reason: str | None = None


@dataclass(frozen=True)
class UnsupportedRegionPlan:
    reason: BackendUnsupportedReason
    message: str
    offending_endpoints: tuple[EndpointRecord, ...] = ()
    attempted_adapters: tuple[AdapterAttempt, ...] = ()


@dataclass(frozen=True)
class _AdapterCandidate:
    kind: AdapterKind
    profile: AdapterProfile
    unsupported_reason: str | None = None


class DefaultRegionAccessService:
    def evaluate_buffer_access(
        self,
        query: BufferAccessQuery,
        candidate: _AdapterCandidate,
    ) -> RegionAccessDecision:
        if query.backend_kind is BackendKind.REMOTE_SIDECAR:
            return _region_access_unsupported(
                RegionAccessReasonCode.UNSUPPORTED_BACKEND_KIND,
                candidate.unsupported_reason or "REMOTE_SIDECAR is not a locally materializable backend",
                query,
                candidate,
            )
        # Mechanisms that dereference a peer's memory. Whether one works is a platform fact that
        # would have to be probed, and no probe exists — nor does a materializer that could carry
        # either of them out, so admitting one would produce a plan nothing can honour.
        if candidate.profile in (AdapterProfile.HOST_SVM_MAP, AdapterProfile.DEVICE_VMM_PEER_IMPORT):
            return _region_access_unsupported(
                RegionAccessReasonCode.NO_IMPLEMENTED_DIRECT_MAP_PROBE,
                "direct map probe is not implemented for this region access profile",
                query,
                candidate,
            )
        offered = any(
            offered.kind is candidate.kind and offered.profile is candidate.profile
            for offered in buffer_adapter_candidates(query)
        )
        if not offered:
            return _region_access_unsupported(
                RegionAccessReasonCode.UNSUPPORTED_ENDPOINT_RELATION,
                "adapter does not match this backend and consumer relation",
                query,
                candidate,
            )
        # Availability is a service verdict, never an assertion carried by the candidate. A caller
        # cannot turn an unimplemented adapter into a supported one by rebuilding the same
        # kind/profile without its diagnostic `unsupported_reason`.
        if candidate.profile in (AdapterProfile.REMOTE_COPY, AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT):
            return _region_access_unsupported(
                RegionAccessReasonCode.NO_COPY_BACKEND,
                candidate.unsupported_reason or f"{candidate.profile.value} is not implemented for this endpoint",
                query,
                candidate,
            )
        # Every profile admitted here is carried out by something: `HOST_VMM_COPY` and
        # `OWNER_DEVICE_COPY` by the owner's control-plane copy, the three map profiles by a
        # resolver in `buffer._RESOLVERS`. A mechanism with no implementation belongs above, not
        # here — admitting one lets a plan through that no materializer can honour.
        if candidate.profile in (
            AdapterProfile.HOST_VMM_COPY,
            AdapterProfile.HOST_SHM_MAP,
            AdapterProfile.FORK_INHERITED_VA,
            AdapterProfile.DEVICE_LOCAL,
            AdapterProfile.OWNER_DEVICE_COPY,
        ):
            return _region_access_supported(query, candidate)
        return _region_access_unsupported(
            RegionAccessReasonCode.STATIC_UNSUPPORTED,
            "region access profile is not supported by the default service",
            query,
            candidate,
        )


class StaticRegionAccessService:
    def __init__(
        self,
        decisions: dict[tuple[BackendKind, AdapterKind, AdapterProfile], RegionAccessDecision | bool] | None = None,
    ) -> None:
        self._decisions: dict[tuple[BackendKind, AdapterKind, AdapterProfile], RegionAccessDecision] = {}
        for key, decision in (decisions or {}).items():
            normalized = _normalize_region_access_key(key)
            if isinstance(decision, RegionAccessDecision):
                self._decisions[normalized] = decision
            else:
                self._decisions[normalized] = RegionAccessDecision(bool(decision))

    def evaluate_buffer_access(
        self,
        query: BufferAccessQuery,
        candidate: _AdapterCandidate,
    ) -> RegionAccessDecision:
        key = (query.backend_kind, candidate.kind, candidate.profile)
        decision = self._decisions.get(key)
        if decision is not None:
            if decision.supported and decision.diagnostics is None:
                return _region_access_supported(query, candidate)
            if not decision.supported and decision.diagnostics is None:
                return _region_access_unsupported(
                    RegionAccessReasonCode.STATIC_UNSUPPORTED,
                    decision.reason or "region access is not supported by the static service",
                    query,
                    candidate,
                )
            return decision
        return _region_access_unsupported(
            RegionAccessReasonCode.STATIC_UNSUPPORTED,
            "region access is not supported by the static service",
            query,
            candidate,
        )


class BackendResolver:
    def __init__(
        self,
        registry: EndpointRegistry,
        region_access: RegionAccessService,
    ) -> None:
        self._registry = registry
        self._region_access = region_access

    def plan(self, resolved: ResolvedRegionSpec, layout: RegionLayoutSpec) -> BackendPlan | UnsupportedRegionPlan:
        self._validate_layout(layout)
        members = resolved.members
        provider = self._choose_provider(resolved)
        if isinstance(provider, UnsupportedRegionPlan):
            return provider
        backend_kind = _backend_kind_for_provider(provider)
        payload = self._plan_part(RegionPartKind.PAYLOAD, backend_kind, provider, members, layout)
        if isinstance(payload, UnsupportedRegionPlan):
            return payload
        counter = self._plan_part(RegionPartKind.COUNTER, backend_kind, provider, members, layout)
        if isinstance(counter, UnsupportedRegionPlan):
            return counter
        return BackendPlan(
            ordered_members=tuple(member.identity for member in members),
            payload=payload,
            counter=counter,
            # Only SingleOwnerPlan is defined here; topology-specific plans extend this field.
            topology_plan=SingleOwnerPlan(provider_endpoint=provider.identity),
        )

    def _validate_layout(self, layout: RegionLayoutSpec) -> None:
        if not isinstance(layout, RegionLayoutSpec):
            raise TypeError("BackendResolver.plan expects RegionLayoutSpec")
        if int(layout.payload_bytes) < 0 or int(layout.counter_bytes) < 0:
            raise ValueError("RegionLayoutSpec byte sizes must be non-negative")

    def _choose_provider(self, resolved: ResolvedRegionSpec) -> EndpointRecord | UnsupportedRegionPlan:
        members = resolved.members
        by_identity = {member.identity: member for member in members}
        provider_identity = resolved.topology.provider_endpoint
        if provider_identity is not None:
            provider = by_identity.get(provider_identity)
            if provider is None:
                try:
                    provider_record = self._registry._record_for(provider_identity)
                except ValueError:
                    provider_record = None
                return _unsupported(
                    BackendUnsupportedReason.PROVIDER_NOT_IN_MEMBERS,
                    "SingleOwner provider is not included in region members",
                    () if provider_record is None else (provider_record,),
                )
            return provider
        for deployment in (DEVICE_AICORE, DEVICE_AICPU, HOST_CPU):
            for member in members:
                if member.deployment is deployment:
                    return member
        return _unsupported(
            BackendUnsupportedReason.NO_DEFAULT_PROVIDER,
            "no default SingleOwner provider is available",
        )

    def _plan_part(
        self,
        part: RegionPartKind,
        backend_kind: BackendKind,
        provider: EndpointRecord,
        members: Sequence[EndpointRecord],
        layout: RegionLayoutSpec,
    ) -> RegionPartPlan | UnsupportedRegionPlan:
        attachments: list[MemberAttachmentPlan] = []
        for member in members:
            if member.identity == provider.identity:
                attachments.append(
                    MemberAttachmentPlan(
                        member=member.identity,
                        role=AttachmentRole.PROVIDER,
                        adapter_kind=None,
                        adapter_profile=None,
                    )
                )
                continue
            attachment = self._consumer_attachment(part, backend_kind, provider, member, layout)
            if isinstance(attachment, UnsupportedRegionPlan):
                return attachment
            attachments.append(attachment)
        return RegionPartPlan(part=part, backend_kind=backend_kind, attachments=tuple(attachments))

    def _consumer_attachment(
        self,
        part: RegionPartKind,
        backend_kind: BackendKind,
        provider: EndpointRecord,
        member: EndpointRecord,
        layout: RegionLayoutSpec,
    ) -> MemberAttachmentPlan | UnsupportedRegionPlan:
        attempts: list[AdapterAttempt] = []
        same_node = self._registry.same_node(provider, member)
        query = BufferAccessQuery(
            backend_kind=backend_kind,
            consumer_deployment=member.deployment,
            same_node=same_node,
            # Provider attachments are skipped above. Every consumer considered here therefore
            # needs either a peer/map/copy edge rather than the Buffer-local mechanism.
            same_endpoint=False,
        )
        for candidate in buffer_adapter_candidates(query):
            decision = self._region_access.evaluate_buffer_access(query, candidate)
            if decision.diagnostics is not None:
                decision = replace(
                    decision,
                    diagnostics=replace(
                        decision.diagnostics,
                        provider_label=_endpoint_label(provider),
                        consumer_label=_endpoint_label(member),
                        part=part,
                    ),
                )
            if decision.supported:
                return MemberAttachmentPlan(
                    member=member.identity,
                    role=AttachmentRole.CONSUMER,
                    adapter_kind=candidate.kind,
                    adapter_profile=candidate.profile,
                )
            attempts.append(_attempt(part, member, backend_kind, candidate, decision.reason))
        if attempts:
            return _unsupported(
                BackendUnsupportedReason.ADAPTER_UNSUPPORTED,
                "no adapter can attach region member to provider",
                (provider, member),
                attempted_adapters=attempts,
            )
        return _unsupported(
            BackendUnsupportedReason.UNSUPPORTED_DEPLOYMENT_COMBINATION,
            "unsupported endpoint deployment combination for SingleOwner region",
            (provider, member),
        )


def buffer_adapter_candidates(query: BufferAccessQuery) -> tuple[_AdapterCandidate, ...]:
    """Ordered adapter candidates for one backing and consumer relation.

    All wire ``BackendKind`` values enter this function, including combinations whose only
    candidate is currently unimplemented. That guarantees the service owns the structured verdict
    instead of an empty enumeration silently bypassing it.

    Offering a candidate is not asserting it works. A mechanism with no materializer is offered
    *with* an ``unsupported_reason`` and refused by the service, because a plan admitting one names
    an attachment nothing can carry out — the refusal then moves from planning, where it is
    structured and carries the alternatives tried, to materialization or to nowhere at all. Where
    the exclusion is structural rather than merely unimplemented it stays here instead, so that an
    injected service cannot override it.
    """
    if query.backend_kind is BackendKind.REMOTE_SIDECAR:
        return _explicit_transfer_candidates(
            "REMOTE_SIDECAR is resolved by its remote session, never by a local materializer"
        )
    if query.consumer_deployment is HOST_CPU:
        return _host_consumer_candidates(query.backend_kind, query.same_node)
    if query.consumer_deployment in (DEVICE_AICORE, DEVICE_AICPU):
        return _device_consumer_candidates(query.backend_kind, query.same_node, query.same_endpoint)
    return _explicit_transfer_candidates(f"unsupported consumer deployment {query.consumer_deployment!r}")


def _host_consumer_candidates(backend_kind: BackendKind, same_node: bool) -> tuple[_AdapterCandidate, ...]:
    """What a host consumer can reach a backing by: a local mechanism, or the owner copying for it.

    Off-node the answer does not depend on the backend at all — no mapping survives the hop, so
    every backing offers the same copy pair. Only the same-node mechanism is backend-specific.
    """
    if not same_node:
        return _remote_copy_candidates()
    if backend_kind in (BackendKind.FORK_SHM, BackendKind.FORK_COW):
        return (_AdapterCandidate(AdapterKind.DIRECT_MAP, AdapterProfile.FORK_INHERITED_VA),)
    if backend_kind is BackendKind.POSIX_SHM:
        return (_AdapterCandidate(AdapterKind.DIRECT_MAP, AdapterProfile.HOST_SHM_MAP),)
    if backend_kind is BackendKind.DEVICE_MALLOC:
        return (_AdapterCandidate(AdapterKind.OWNER_DELEGATED_COPY, AdapterProfile.OWNER_DEVICE_COPY),)
    if backend_kind in (BackendKind.VMM_WINDOW, BackendKind.VMM_SHAREABLE):
        # A VMM VA cannot be host-registered, so HOST_SVM_MAP is deliberately absent.
        return (_AdapterCandidate(AdapterKind.OWNER_DELEGATED_COPY, AdapterProfile.HOST_VMM_COPY),)
    return _explicit_transfer_candidates(f"unsupported backend {backend_kind!r}")


def _device_consumer_candidates(
    backend_kind: BackendKind, same_node: bool, same_endpoint: bool
) -> tuple[_AdapterCandidate, ...]:
    """What a device consumer can reach a backing by.

    ``same_endpoint`` is what separates a backing already local to this chip from a same-node peer's:
    backend and deployment alone cannot tell ``DEVICE_LOCAL`` from ``DEVICE_PEER``.
    """
    if same_endpoint and backend_kind in (
        BackendKind.DEVICE_MALLOC,
        BackendKind.VMM_WINDOW,
        BackendKind.VMM_SHAREABLE,
    ):
        return (_AdapterCandidate(AdapterKind.DIRECT_MAP, AdapterProfile.DEVICE_LOCAL),)
    if backend_kind in (BackendKind.VMM_WINDOW, BackendKind.VMM_SHAREABLE):
        if same_node:
            return (
                _AdapterCandidate(
                    AdapterKind.DEVICE_PEER,
                    AdapterProfile.DEVICE_VMM_PEER_IMPORT,
                    unsupported_reason="device VMM peer import materializer is not implemented yet",
                ),
            )
        return (
            _AdapterCandidate(
                AdapterKind.DEVICE_PEER,
                AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT,
                unsupported_reason="device fabric peer import is not available for this endpoint",
            ),
            *_remote_copy_candidates(),
        )
    if backend_kind is BackendKind.DEVICE_MALLOC:
        return _explicit_transfer_candidates("DEVICE_MALLOC cannot be imported by a peer endpoint")
    if backend_kind in (BackendKind.FORK_SHM, BackendKind.FORK_COW):
        return _explicit_transfer_candidates("fork-inherited VA is unavailable at this endpoint")
    if backend_kind is BackendKind.POSIX_SHM:
        return _explicit_transfer_candidates("explicit transfer materializer is not implemented yet")
    return _explicit_transfer_candidates(f"unsupported backend {backend_kind!r}")


def _explicit_transfer_candidates(reason: str) -> tuple[_AdapterCandidate, ...]:
    return (_AdapterCandidate(AdapterKind.EXPLICIT_TRANSFER, AdapterProfile.REMOTE_COPY, unsupported_reason=reason),)


def _remote_copy_candidates() -> tuple[_AdapterCandidate, ...]:
    return (
        _AdapterCandidate(
            AdapterKind.OWNER_DELEGATED_COPY,
            AdapterProfile.REMOTE_COPY,
            unsupported_reason="remote copy materializer is not implemented yet",
        ),
        _AdapterCandidate(
            AdapterKind.EXPLICIT_TRANSFER,
            AdapterProfile.REMOTE_COPY,
            unsupported_reason="explicit transfer materializer is not implemented yet",
        ),
    )


def _backend_kind_for_provider(provider: EndpointRecord) -> BackendKind:
    """The backing family the provider owns, shared by both parts of the region.

    Payload and counter carry their own `RegionPartPlan.backend_kind` so they can diverge, but one
    provider names one backing here: a device provider's control buffer wants a non-VMM,
    host-mappable backing that no platform path allocates yet. Until it does, a device-provided
    counter is VMM-backed and therefore not host direct-mappable — the constraint
    `buffer_adapter_candidates` encodes.
    """
    if provider.deployment in (DEVICE_AICORE, DEVICE_AICPU):
        return BackendKind.VMM_SHAREABLE
    if provider.deployment is HOST_CPU:
        return BackendKind.POSIX_SHM
    raise ValueError(f"unsupported provider deployment: {_endpoint_label(provider)}")


def _normalize_region_access_key(
    key: tuple[BackendKind | str, AdapterKind | str, AdapterProfile | str],
) -> tuple[BackendKind, AdapterKind, AdapterProfile]:
    backend_kind, adapter_kind, adapter_profile = key
    return BackendKind(backend_kind), AdapterKind(adapter_kind), AdapterProfile(adapter_profile)


def _region_access_supported(query: BufferAccessQuery, candidate: _AdapterCandidate) -> RegionAccessDecision:
    diagnostics = _region_access_diagnostics(
        RegionAccessReasonCode.SUPPORTED,
        "region access is supported",
        query,
        candidate,
    )
    return RegionAccessDecision(True, diagnostics=diagnostics)


def _region_access_unsupported(
    reason_code: RegionAccessReasonCode,
    message: str,
    query: BufferAccessQuery,
    candidate: _AdapterCandidate,
) -> RegionAccessDecision:
    diagnostics = _region_access_diagnostics(reason_code, message, query, candidate)
    return RegionAccessDecision(False, message, diagnostics)


def _region_access_diagnostics(
    reason_code: RegionAccessReasonCode,
    message: str,
    query: BufferAccessQuery,
    candidate: _AdapterCandidate,
) -> RegionAccessDiagnostics:
    return RegionAccessDiagnostics(
        reason_code=reason_code,
        message=message,
        platform=query.platform,
        runtime=query.runtime,
        backend_kind=query.backend_kind,
        consumer_deployment=query.consumer_deployment,
        same_node=query.same_node,
        same_endpoint=query.same_endpoint,
        adapter_kind=candidate.kind,
        adapter_profile=candidate.profile,
    )


def _attempt(
    part: RegionPartKind,
    member: EndpointRecord,
    backend_kind: BackendKind,
    candidate: _AdapterCandidate,
    reason: str | None,
) -> AdapterAttempt:
    return AdapterAttempt(
        part=part,
        member=member,
        backend_kind=backend_kind,
        adapter_kind=candidate.kind,
        adapter_profile=candidate.profile,
        reason=reason,
    )


def _unsupported(
    reason: BackendUnsupportedReason,
    message: str,
    offending: Sequence[EndpointRecord] = (),
    *,
    attempted_adapters: Sequence[AdapterAttempt] = (),
) -> UnsupportedRegionPlan:
    unique_offending = _unique_endpoints(offending)
    attempts = tuple(attempted_adapters)
    if reason is not BackendUnsupportedReason.ADAPTER_UNSUPPORTED and attempts:
        raise ValueError("attempted_adapters are only valid for ADAPTER_UNSUPPORTED")
    if unique_offending:
        message = f"{message}: {', '.join(_endpoint_label(endpoint) for endpoint in unique_offending)}"
    return UnsupportedRegionPlan(
        reason=reason,
        message=message,
        offending_endpoints=unique_offending,
        attempted_adapters=attempts,
    )


def _unique_endpoints(endpoints: Sequence[EndpointRecord]) -> tuple[EndpointRecord, ...]:
    unique: list[EndpointRecord] = []
    seen: set[EndpointIdentity] = set()
    for endpoint in endpoints:
        if endpoint.identity in seen:
            continue
        seen.add(endpoint.identity)
        unique.append(endpoint)
    return tuple(unique)


def _endpoint_label(record: EndpointRecord) -> str:
    return f"{record.path} {record.deployment.value}"


def _selector_label(selector: EndpointSelector) -> str:
    return f"{selector.path} {selector.deployment.value}"


__all__ = [
    "AdapterAttempt",
    "AdapterKind",
    "AdapterProfile",
    "AttachmentRole",
    "BackendKind",
    "BackendPlan",
    "BackendResolver",
    "BackendUnsupportedReason",
    "BufferAccessQuery",
    "DEVICE_AICORE",
    "DEVICE_AICPU",
    "EndpointDeployment",
    "EndpointDeploymentKind",
    "EndpointId",
    "EndpointIdentity",
    "EndpointPathSegment",
    "EndpointRecord",
    "EndpointRegistry",
    "EndpointResolveError",
    "EndpointResolveReason",
    "EndpointSelector",
    "EndpointSelectorKind",
    "HOST_CPU",
    "MemberAttachmentPlan",
    "NodeScopeId",
    "ParsedEndpointPath",
    "RegionAccessDecision",
    "RegionAccessDiagnostics",
    "RegionAccessReasonCode",
    "RegionAccessService",
    "RegionLayoutSpec",
    "RegionPartKind",
    "RegionPartPlan",
    "RegionTopologyKind",
    "ResolvedRegionSpec",
    "ResolvedSingleOwner",
    "SingleOwner",
    "SingleOwnerPlan",
    "StaticRegionAccessService",
    "UnsupportedRegionPlan",
    "at",
    "buffer_adapter_candidates",
    "parse_endpoint_path",
    "under",
]
