# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Neutral provider-region values, typed codes, store, and structured results.

This module is the provider-agent value surface for CPU-NPU comm regions. It
does not import worker-chip compatibility types, Worker, mailbox transport, or
delegated-region transaction identity.
"""

from __future__ import annotations

import ctypes
import errno
import logging
import threading
import uuid
from dataclasses import dataclass
from enum import Enum, IntEnum
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable, Protocol, TypeVar, Union

from _task_interface import (  # pyright: ignore[reportMissingImports]
    AccessMode,
    AddressSpace,
    BackendKind,
    BufferDescriptor,
    CanonicalIdentity,
    _region_vmm_allocate_export,
    _region_vmm_begin,
    _region_vmm_granularity,
    _region_vmm_release,
    _region_vmm_zero_bytes,
)

from .buffer import Buffer, _wrap_vmm_shareable, intern_worker_path

_UINT64_MAX = (1 << 64) - 1
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_COUNTER_LOGICAL_ALIGNMENT = 4
_COUNTER_BASE_ALIGNMENT = 64
POSIX_SHM_TOKEN_MAX_BYTES = 32


class RegionPartKind(IntEnum):
    INVALID = 0
    PAYLOAD = 1
    COUNTER = 2


class RegionEnvironmentKind(str, Enum):
    SIM = "SIM"
    ONBOARD = "ONBOARD"


class ProviderRegionStoreState(str, Enum):
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSE_FAILED = "CLOSE_FAILED"
    CLOSED = "CLOSED"


class ProviderRegionResourceState(str, Enum):
    CREATING = "CREATING"
    ACTIVE = "ACTIVE"
    CLEANUP_PENDING = "CLEANUP_PENDING"


class ProviderPartResourceState(str, Enum):
    SHELL = "SHELL"
    MATERIALIZING = "MATERIALIZING"
    READY = "READY"
    CLEANUP_PENDING = "CLEANUP_PENDING"
    RELEASED = "RELEASED"


class ProviderReleaseStatus(IntEnum):
    RELEASED = 1
    ALREADY_GONE = 2
    UNKNOWN_RESOURCE = 3
    CLEANUP_INCOMPLETE = 4


class RegionControlErrorKind(IntEnum):
    NONE = 0
    BAD_MAGIC_VERSION = 1
    BAD_MESSAGE_SIZE = 2
    INVALID_ENUM_VALUE = 3
    RESERVED_NONZERO = 4
    INVALID_FIELD_VALUE = 5
    STORE_LIFECYCLE = 6
    INTERNAL_INVARIANT = 7
    BACKEND_FAILURE = 8


class RegionOperationKind(IntEnum):
    NONE = 0
    MATERIALIZE = 1
    ZERO_BYTES = 2
    DESCRIBE = 3
    LOCAL_VIEW = 4
    RELEASE = 5


class RegionCleanupCause(IntEnum):
    NONE = 0
    BACKEND_ERROR = 1
    INTERRUPTED = 2
    BACKEND_STATE_MISMATCH = 3


REGION_PARTS = (RegionPartKind.PAYLOAD, RegionPartKind.COUNTER)
_ALLOCATION_ERROR_KINDS = (
    RegionControlErrorKind.BACKEND_FAILURE,
    RegionControlErrorKind.INTERNAL_INVARIANT,
)
_BACKEND_FAILURE_OPERATIONS = (
    RegionOperationKind.MATERIALIZE,
    RegionOperationKind.ZERO_BYTES,
    RegionOperationKind.DESCRIBE,
    RegionOperationKind.LOCAL_VIEW,
)


def _require_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    return value


def _require_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")
    return value


def _require_uint64(name: str, value: object) -> int:
    number = _require_int(name, value)
    if number < 0 or number > _UINT64_MAX:
        raise ValueError(f"{name} overflowed uint64")
    return number


def _require_positive_uint64(name: str, value: object) -> int:
    number = _require_uint64(name, value)
    if number < 1:
        raise ValueError(f"{name} must be positive")
    return number


def _require_nonzero_uint64(name: str, value: object) -> int:
    return _require_positive_uint64(name, value)


def _require_int32(name: str, value: object) -> int:
    number = _require_int(name, value)
    if number < _INT32_MIN or number > _INT32_MAX:
        raise ValueError(f"{name} overflowed int32")
    return number


_EnumT = TypeVar("_EnumT", bound=Enum)


def _require_enum(enum_cls: type[_EnumT], value: object, name: str) -> _EnumT:
    try:
        return enum_cls(value)
    except ValueError as exc:
        raise ValueError(f"unknown {name} {value!r}") from exc


def _require_backend_kind(value: object) -> BackendKind:
    return _require_enum(BackendKind, value, "planned backing kind")


def _require_part(value: object, *, allow_invalid: bool = False) -> RegionPartKind:
    part = _require_enum(RegionPartKind, value, "region part")
    if part is RegionPartKind.INVALID and not allow_invalid:
        raise ValueError("region part must be PAYLOAD or COUNTER")
    return part


def _require_posix_shm_token(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("POSIX shm token must be str")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("POSIX shm token must be ASCII") from exc
    if not encoded or len(encoded) > POSIX_SHM_TOKEN_MAX_BYTES:
        raise ValueError("POSIX shm token length must be in 1..32 bytes")
    if b"\x00" in encoded:
        raise ValueError("POSIX shm token must not contain NUL")
    return value


def _checked_add_u64(lhs: int, rhs: int, name: str) -> int:
    total = lhs + rhs
    if total > _UINT64_MAX:
        raise ValueError(f"{name} overflowed uint64")
    return total


def _require_counter_logical_bytes(value: object) -> int:
    logical_bytes = _require_positive_uint64("COUNTER logical_bytes", value)
    if logical_bytes % _COUNTER_LOGICAL_ALIGNMENT != 0:
        raise ValueError("COUNTER logical_bytes must be a multiple of 4")
    return logical_bytes


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return (int(value) + alignment - 1) // alignment * alignment


def _require_owner_nonce(nonce: object) -> bytes:
    if not isinstance(nonce, (bytes, bytearray)):
        raise TypeError("owner_instance_id must be bytes")
    value = bytes(nonce)
    if len(value) != 8 or value == b"\x00" * 8:
        raise ValueError("owner_instance_id must be a nonzero 8-byte nonce")
    return value


class EndpointBufferIdentityAllocator(Protocol):
    @property
    def owner_instance_id(self) -> bytes: ...

    def burn_identity(self) -> CanonicalIdentity: ...


class LocalEndpointBufferIdentityAllocator:
    """Thread-safe endpoint-scoped Buffer identity allocator."""

    def __init__(self, owner_instance_id: bytes) -> None:
        self._owner_instance_id = _require_owner_nonce(owner_instance_id)
        self._lock = threading.Lock()
        self._next_buffer_id = 1

    @property
    def owner_instance_id(self) -> bytes:
        return self._owner_instance_id

    def burn_identity(self) -> CanonicalIdentity:
        with self._lock:
            buffer_id = self._next_buffer_id
            if buffer_id > _UINT64_MAX:
                raise RegionControlError(
                    RegionControlErrorKind.INTERNAL_INVARIANT,
                    "buffer identity space exhausted",
                )
            self._next_buffer_id = buffer_id + 1
        return CanonicalIdentity(self._owner_instance_id, buffer_id, 1)


@dataclass(frozen=True)
class DeviceAllocationTarget:
    device_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "device_id", _require_int32("device_id", self.device_id))


@dataclass(frozen=True)
class HostAllocationTarget:
    pass


AllocationTarget = Union[DeviceAllocationTarget, HostAllocationTarget]


@dataclass(frozen=True)
class RegionAllocationContext:
    environment_kind: RegionEnvironmentKind
    target: AllocationTarget

    def __post_init__(self) -> None:
        environment = _require_enum(RegionEnvironmentKind, self.environment_kind, "environment kind")
        object.__setattr__(self, "environment_kind", environment)
        if not isinstance(self.target, (DeviceAllocationTarget, HostAllocationTarget)):
            raise TypeError("allocation target must be DeviceAllocationTarget or HostAllocationTarget")


@dataclass(frozen=True)
class RegionPartAllocationSpec:
    planned_backing_kind: BackendKind
    logical_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "planned_backing_kind", _require_backend_kind(self.planned_backing_kind))
        object.__setattr__(self, "logical_bytes", _require_positive_uint64("logical_bytes", self.logical_bytes))


@dataclass(frozen=True)
class RegionAllocationSpec:
    payload: RegionPartAllocationSpec
    counter: RegionPartAllocationSpec

    def __post_init__(self) -> None:
        if not isinstance(self.payload, RegionPartAllocationSpec):
            raise TypeError("payload spec must be RegionPartAllocationSpec")
        if not isinstance(self.counter, RegionPartAllocationSpec):
            raise TypeError("counter spec must be RegionPartAllocationSpec")
        _require_counter_logical_bytes(self.counter.logical_bytes)

    def part(self, kind: RegionPartKind) -> RegionPartAllocationSpec:
        part = _require_part(kind)
        if part is RegionPartKind.PAYLOAD:
            return self.payload
        return self.counter


@dataclass(frozen=True)
class VmmShareableHandleImport:
    device_id: int
    shareable_handle: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "device_id", _require_int32("device_id", self.device_id))
        object.__setattr__(
            self,
            "shareable_handle",
            _require_uint64("shareable_handle", self.shareable_handle),
        )


@dataclass(frozen=True)
class PosixShmImport:
    shm_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "shm_name", _require_posix_shm_token(self.shm_name))


ImportCapability = Union[VmmShareableHandleImport, PosixShmImport]


def _posix_token_from_descriptor(descriptor: BufferDescriptor) -> str:
    if descriptor.backend_kind is not BackendKind.POSIX_SHM:
        raise ValueError("POSIX shm token requires POSIX_SHM")
    return _require_posix_shm_token(bytes(descriptor.body).decode("ascii"))


def _posix_object_size(token: str) -> int:
    shm = SharedMemory(name=_posix_shm_create_name(token), create=False)
    try:
        return int(shm.size)
    finally:
        shm.close()


def _vmm_shareable_facts(descriptor: BufferDescriptor) -> tuple[int, int, int]:
    if descriptor.backend_kind is not BackendKind.VMM_SHAREABLE:
        raise ValueError("VMM shareable facts require VMM_SHAREABLE")
    body = bytes(descriptor.body)
    if len(body) != 24:
        raise ValueError("VMM_SHAREABLE body must be 24 bytes")
    device_id = int.from_bytes(body[0:4], "little", signed=True)
    shareable_handle = int.from_bytes(body[8:16], "little")
    mapping_bytes = int.from_bytes(body[16:24], "little")
    return device_id, shareable_handle, mapping_bytes


@dataclass(frozen=True)
class RegionExportDescriptor:
    payload: BufferDescriptor
    counter: BufferDescriptor

    def __post_init__(self) -> None:
        if not isinstance(self.payload, BufferDescriptor):
            raise TypeError("payload export descriptor must be BufferDescriptor")
        if not isinstance(self.counter, BufferDescriptor):
            raise TypeError("counter export descriptor must be BufferDescriptor")
        _require_counter_logical_bytes(self.counter.nbytes)


@dataclass(frozen=True)
class RegionPartLocalView:
    part: RegionPartKind
    local_base: int
    logical_bytes: int

    def __post_init__(self) -> None:
        part = _require_part(self.part)
        local_base = _require_uint64("local_base", self.local_base)
        if part is RegionPartKind.COUNTER:
            logical_bytes = _require_counter_logical_bytes(self.logical_bytes)
            if local_base % _COUNTER_BASE_ALIGNMENT != 0:
                raise ValueError("COUNTER local_base must be 64-byte aligned")
        else:
            logical_bytes = _require_positive_uint64("logical_bytes", self.logical_bytes)
        _checked_add_u64(local_base, logical_bytes, f"{part.name} local span")
        object.__setattr__(self, "part", part)
        object.__setattr__(self, "local_base", local_base)
        object.__setattr__(self, "logical_bytes", logical_bytes)

    @property
    def local_end(self) -> int:
        return self.local_base + self.logical_bytes


def validate_independent_local_views(
    payload: RegionPartLocalView, counter: RegionPartLocalView
) -> tuple[RegionPartLocalView, RegionPartLocalView]:
    if not isinstance(payload, RegionPartLocalView) or payload.part is not RegionPartKind.PAYLOAD:
        raise ValueError("payload local view must use part PAYLOAD")
    if not isinstance(counter, RegionPartLocalView) or counter.part is not RegionPartKind.COUNTER:
        raise ValueError("counter local view must use part COUNTER")
    if payload.local_base < counter.local_end and counter.local_base < payload.local_end:
        raise ValueError("PAYLOAD and COUNTER local spans must not overlap")
    return payload, counter


@dataclass(frozen=True)
class RegionAllocationResult:
    provider_resource_id: int
    export_descriptor: RegionExportDescriptor

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_resource_id",
            _require_nonzero_uint64("provider_resource_id", self.provider_resource_id),
        )
        if not isinstance(self.export_descriptor, RegionExportDescriptor):
            raise TypeError("export_descriptor must be RegionExportDescriptor")


@dataclass(frozen=True)
class ProviderCleanupFailure:
    part: RegionPartKind
    backend_operation: RegionOperationKind
    typed_cause: RegionCleanupCause

    def __post_init__(self) -> None:
        part = _require_part(self.part)
        operation = _require_enum(RegionOperationKind, self.backend_operation, "backend operation")
        cause = _require_enum(RegionCleanupCause, self.typed_cause, "cleanup cause")
        if operation is RegionOperationKind.NONE:
            raise ValueError("cleanup failure requires a nonzero backend operation")
        if cause is RegionCleanupCause.NONE:
            raise ValueError("cleanup failure requires a nonzero typed cause")
        object.__setattr__(self, "part", part)
        object.__setattr__(self, "backend_operation", operation)
        object.__setattr__(self, "typed_cause", cause)


@dataclass(frozen=True)
class ProviderReleaseResult:
    provider_resource_id: int
    status: ProviderReleaseStatus
    failures: tuple[ProviderCleanupFailure, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_resource_id",
            _require_nonzero_uint64("provider_resource_id", self.provider_resource_id),
        )
        status = _require_enum(ProviderReleaseStatus, self.status, "release status")
        object.__setattr__(self, "status", status)
        failures = tuple(self.failures)
        if any(not isinstance(failure, ProviderCleanupFailure) for failure in failures):
            raise TypeError("release failures must be ProviderCleanupFailure values")
        parts = [failure.part for failure in failures]
        if len(parts) != len(set(parts)):
            raise ValueError("ProviderReleaseResult allows at most one failure per part")
        if status is ProviderReleaseStatus.CLEANUP_INCOMPLETE:
            if not failures:
                raise ValueError("CLEANUP_INCOMPLETE requires structured per-part failures")
        elif failures:
            raise ValueError(f"{status.name} must not carry cleanup failures")
        object.__setattr__(self, "failures", failures)


class RegionProviderError(Exception):
    pass


class RegionControlError(RegionProviderError):
    def __init__(
        self,
        kind: RegionControlErrorKind | int,
        message: str = "",
        *,
        failed_part: RegionPartKind | int = RegionPartKind.INVALID,
        failed_operation: RegionOperationKind | int = RegionOperationKind.NONE,
    ) -> None:
        control_kind = _require_enum(RegionControlErrorKind, kind, "control error kind")
        if control_kind is RegionControlErrorKind.NONE:
            raise ValueError("RegionControlError requires a nonzero control kind")
        part = _require_part(failed_part, allow_invalid=True)
        operation = _require_enum(RegionOperationKind, failed_operation, "failed operation")
        self.kind = control_kind
        self.failed_part = part
        self.failed_operation = operation
        self.message = str(message)
        super().__init__(self.message or control_kind.name)


class RegionAllocationError(RegionProviderError):
    def __init__(
        self,
        *,
        provisional_resource_id: int,
        control_kind: RegionControlErrorKind,
        failed_part: RegionPartKind | int,
        failed_operation: RegionOperationKind | int,
        cleanup_debt_remaining: bool,
        message: str = "",
    ) -> None:
        resource_id = _require_nonzero_uint64("provisional_resource_id", provisional_resource_id)
        kind = _require_enum(RegionControlErrorKind, control_kind, "control error kind")
        if kind not in _ALLOCATION_ERROR_KINDS:
            raise ValueError("RegionAllocationError control kind must be BACKEND_FAILURE or INTERNAL_INVARIANT")
        part = _require_part(failed_part, allow_invalid=True)
        operation = _require_enum(RegionOperationKind, failed_operation, "failed operation")
        if kind is RegionControlErrorKind.BACKEND_FAILURE and operation not in _BACKEND_FAILURE_OPERATIONS:
            raise ValueError("BACKEND_FAILURE requires a create-path operation")
        if operation is RegionOperationKind.RELEASE:
            raise ValueError("RegionAllocationError must not use RELEASE as the failed operation")
        self.provisional_resource_id = resource_id
        self.control_kind = kind
        self.failed_part = part
        self.failed_operation = operation
        self.cleanup_debt_remaining = _require_bool("cleanup_debt_remaining", cleanup_debt_remaining)
        self.message = str(message)
        super().__init__(self.message or kind.name)


class RegionPartAllocation(Protocol):
    """Store-private allocation mechanism for one PAYLOAD or COUNTER backing."""

    def materialize(self, identity: CanonicalIdentity, diagnostics: Any = None) -> Buffer: ...

    def zero_bytes(self, offset: int, nbytes: int) -> None: ...

    def release_once(self) -> ProviderCleanupFailure | None: ...


ShellFactory = Callable[..., Any]


def _generate_posix_shm_token() -> str:
    return _require_posix_shm_token("smp_" + uuid.uuid4().hex[:24])


def _posix_shm_create_name(token: str) -> str:
    return token[1:] if token.startswith("/") else token


def _unlink_posix_shm_token(token: str) -> None:
    import _posixshmem  # noqa: PLC0415
    from multiprocessing import resource_tracker  # noqa: PLC0415

    name = token if token.startswith("/") else f"/{token}"
    try:
        _posixshmem.shm_unlink(name)
    except FileNotFoundError:
        return
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            return
        raise
    try:
        resource_tracker.unregister(name, "shared_memory")
    except Exception as exc:
        logging.getLogger("simpler").warning(
            "POSIX shm resource-tracker unregister failed: token=%s error=%s",
            token,
            exc,
        )


def _require_shm_buf(shm: SharedMemory) -> memoryview:
    buf = shm.buf
    if buf is None:
        raise RuntimeError("shared memory mapping is missing")
    return buf


def _shm_local_base(shm: SharedMemory) -> int:
    exported = ctypes.c_char.from_buffer(_require_shm_buf(shm))
    try:
        return ctypes.addressof(exported)
    finally:
        del exported


class SimPosixShmAllocation:
    """One POSIX shm object for a single PAYLOAD or COUNTER part."""

    def __init__(
        self,
        context: RegionAllocationContext,
        part: RegionPartKind,
        spec: RegionPartAllocationSpec,
        *,
        candidate_name: str | None = None,
        shm_cls: Any = SharedMemory,
    ) -> None:
        del context
        self._part = _require_part(part)
        if not isinstance(spec, RegionPartAllocationSpec):
            raise TypeError("spec must be RegionPartAllocationSpec")
        self._spec = spec
        self._shm_cls = shm_cls
        self.candidate_name = _require_posix_shm_token(candidate_name or _generate_posix_shm_token())
        self._name_ownership_known = False
        self._shm_object_installed = False
        self._mapping_available = False
        self._shm: SharedMemory | None = None
        self._local_base: int | None = None
        self._physical_bytes = (
            _align_up(self._spec.logical_bytes, _COUNTER_BASE_ALIGNMENT)
            if self._part is RegionPartKind.COUNTER
            else self._spec.logical_bytes
        )
        self._close_attempted = False
        self._close_complete = False
        self._unlink_attempted = False
        self._unlink_complete = False
        self._release_once_count = 0
        self._first_cleanup_failure: ProviderCleanupFailure | None = None
        self.local_cleanup_details: list[tuple[str, BaseException]] = []

    def materialize(self, identity: CanonicalIdentity, diagnostics: Any = None) -> Buffer:
        try:
            shm = self._shm_cls(
                name=_posix_shm_create_name(self.candidate_name),
                create=True,
                size=self._physical_bytes,
            )
        except FileExistsError as exc:
            raise RegionControlError(
                RegionControlErrorKind.BACKEND_FAILURE,
                "POSIX shm name collision",
                failed_part=self._part,
                failed_operation=RegionOperationKind.MATERIALIZE,
            ) from exc
        except OSError as exc:
            self._name_ownership_known = True
            raise RegionControlError(
                RegionControlErrorKind.BACKEND_FAILURE,
                str(exc) or "POSIX shm create failed",
                failed_part=self._part,
                failed_operation=RegionOperationKind.MATERIALIZE,
            ) from exc
        except BaseException:
            self._name_ownership_known = True
            raise
        self._shm = shm
        self._name_ownership_known = True
        self._shm_object_installed = True
        self._local_base = _shm_local_base(shm)
        self._mapping_available = True
        if self._part is RegionPartKind.COUNTER and self._local_base % _COUNTER_BASE_ALIGNMENT != 0:
            raise RegionControlError(
                RegionControlErrorKind.INTERNAL_INVARIANT,
                "COUNTER POSIX mapping base must be 64-byte aligned",
                failed_part=self._part,
                failed_operation=RegionOperationKind.MATERIALIZE,
            )
        owner_worker_path = ""
        if diagnostics is not None:
            owner_worker_path = str(getattr(diagnostics, "owner_worker_path", "") or "")
        return Buffer(
            identity=identity,
            owner_worker_path_id=intern_worker_path(owner_worker_path),
            address_space=AddressSpace.HOST,
            access=AccessMode.READWRITE,
            backend_kind=BackendKind.POSIX_SHM,
            nbytes=self._spec.logical_bytes,
            body=self.candidate_name.encode("ascii"),
            shm=None,
            base=self._local_base,
        )

    def zero_bytes(self, offset: int, nbytes: int) -> None:
        self._require_mapping("zero_bytes")
        if type(offset) is not int or type(nbytes) is not int:
            raise TypeError("zero_bytes offset and nbytes must be int")
        if offset < 0 or nbytes < 0:
            raise ValueError("zero_bytes offset and nbytes must be non-negative")
        end = offset + nbytes
        if end > self._spec.logical_bytes:
            raise ValueError("zero_bytes range exceeds logical_bytes")
        assert self._shm is not None
        _require_shm_buf(self._shm)[offset:end] = b"\x00" * nbytes

    def release_once(self) -> ProviderCleanupFailure | None:
        self._release_once_count += 1
        if self._release_once_count > 1:
            return self._first_cleanup_failure
        if self._shm_object_installed and not self._close_attempted:
            self._close_attempted = True
            try:
                if self._shm is not None:
                    self._shm.close()
                self._close_complete = True
            except BaseException as exc:
                self._record_cleanup_failure("close", exc)
        if self._name_ownership_known and not self._unlink_attempted:
            self._unlink_attempted = True
            try:
                self._unlink_owned_name()
                self._unlink_complete = True
            except BaseException as exc:
                self._record_cleanup_failure("unlink", exc)
        return self._first_cleanup_failure

    def _unlink_owned_name(self) -> None:
        if self._shm is not None:
            try:
                self._shm.unlink()
                return
            except FileNotFoundError:
                return
            except OSError as exc:
                if exc.errno == errno.ENOENT:
                    return
                raise
        _unlink_posix_shm_token(self.candidate_name)

    def _record_cleanup_failure(self, step: str, exc: BaseException) -> None:
        cause = RegionCleanupCause.INTERRUPTED if _interrupt_like(exc) else RegionCleanupCause.BACKEND_ERROR
        failure = ProviderCleanupFailure(
            part=self._part,
            backend_operation=RegionOperationKind.RELEASE,
            typed_cause=cause,
        )
        self.local_cleanup_details.append((step, exc))
        if self._first_cleanup_failure is None:
            self._first_cleanup_failure = failure

    def _require_mapping(self, operation: str) -> None:
        if not self._mapping_available or self._shm is None:
            raise RegionControlError(
                RegionControlErrorKind.INTERNAL_INVARIANT,
                f"{operation} requires a materialized POSIX shm mapping",
                failed_part=self._part,
                failed_operation=RegionOperationKind.ZERO_BYTES,
            )


class VmmAllocation:
    """One native VMM owner record for a single PAYLOAD or COUNTER part."""

    def __init__(
        self,
        context: RegionAllocationContext,
        part: RegionPartKind,
        spec: RegionPartAllocationSpec,
    ) -> None:
        self._part = _require_part(part)
        if not isinstance(spec, RegionPartAllocationSpec):
            raise TypeError("spec must be RegionPartAllocationSpec")
        if not isinstance(context.target, DeviceAllocationTarget):
            raise TypeError("VmmAllocation requires DeviceAllocationTarget")
        self._spec = spec
        self._device_id = context.target.device_id
        self._handle: int | None = None
        self._device_addr: int | None = None
        self._mapping_bytes: int | None = None
        self._shareable_handle: int | None = None
        self._mapping_available = False
        self._release_once_count = 0
        self._first_cleanup_failure: ProviderCleanupFailure | None = None
        self.local_cleanup_details: list[tuple[str, BaseException]] = []

    @property
    def registry_handle(self) -> int | None:
        return self._handle

    def materialize(self, identity: CanonicalIdentity, diagnostics: Any = None) -> Buffer:
        try:
            handle = _region_vmm_begin(self._device_id)
            self._handle = handle
            export = _region_vmm_allocate_export(handle, self._spec.logical_bytes)
        except RegionControlError:
            raise
        except BaseException as exc:
            raise RegionControlError(
                RegionControlErrorKind.BACKEND_FAILURE,
                str(exc) or "VMM materialize failed",
                failed_part=self._part,
                failed_operation=RegionOperationKind.MATERIALIZE,
            ) from exc
        device_addr = int(export.device_addr)
        mapping_bytes = int(export.mapping_bytes)
        shareable_handle = int(export.shareable_handle)
        try:
            granularity = int(_region_vmm_granularity(self._device_id))
        except BaseException as exc:
            raise RegionControlError(
                RegionControlErrorKind.BACKEND_FAILURE,
                str(exc) or "VMM granularity query failed",
                failed_part=self._part,
                failed_operation=RegionOperationKind.MATERIALIZE,
            ) from exc
        if granularity < 1 or mapping_bytes % granularity != 0:
            raise RegionControlError(
                RegionControlErrorKind.INTERNAL_INVARIANT,
                "VMM mapping_bytes must be a positive multiple of runtime granularity",
                failed_part=self._part,
                failed_operation=RegionOperationKind.MATERIALIZE,
            )
        if mapping_bytes < self._spec.logical_bytes:
            raise RegionControlError(
                RegionControlErrorKind.INTERNAL_INVARIANT,
                "VMM mapping_bytes must cover logical_bytes",
                failed_part=self._part,
                failed_operation=RegionOperationKind.MATERIALIZE,
            )
        if self._part is RegionPartKind.COUNTER:
            if mapping_bytes < _align_up(self._spec.logical_bytes, _COUNTER_BASE_ALIGNMENT):
                raise RegionControlError(
                    RegionControlErrorKind.INTERNAL_INVARIANT,
                    "COUNTER VMM mapping_bytes must cover 64-byte alignment",
                    failed_part=self._part,
                    failed_operation=RegionOperationKind.MATERIALIZE,
                )
            if device_addr % _COUNTER_BASE_ALIGNMENT != 0:
                raise RegionControlError(
                    RegionControlErrorKind.INTERNAL_INVARIANT,
                    "COUNTER VMM base must be 64-byte aligned",
                    failed_part=self._part,
                    failed_operation=RegionOperationKind.MATERIALIZE,
                )
        self._device_addr = device_addr
        self._mapping_bytes = mapping_bytes
        self._shareable_handle = shareable_handle
        self._mapping_available = True
        owner_worker_path = ""
        owner_worker_id = 0
        if diagnostics is not None:
            owner_worker_path = str(getattr(diagnostics, "owner_worker_path", "") or "")
            owner_worker_id = int(getattr(diagnostics, "owner_worker_id", 0) or 0)
        return _wrap_vmm_shareable(
            device_addr,
            self._spec.logical_bytes,
            self._device_id,
            mapping_bytes,
            shareable_handle,
            identity,
            owner_worker_path=owner_worker_path,
            owner_worker_id=owner_worker_id,
        )

    def zero_bytes(self, offset: int, nbytes: int) -> None:
        self._require_mapping("zero_bytes")
        if type(offset) is not int or type(nbytes) is not int:
            raise TypeError("zero_bytes offset and nbytes must be int")
        if offset < 0 or nbytes < 0:
            raise ValueError("zero_bytes offset and nbytes must be non-negative")
        end = offset + nbytes
        if end > self._spec.logical_bytes:
            raise ValueError("zero_bytes range exceeds logical_bytes")
        assert self._handle is not None
        try:
            _region_vmm_zero_bytes(self._handle, offset, nbytes)
        except RegionControlError:
            raise
        except BaseException as exc:
            raise RegionControlError(
                RegionControlErrorKind.BACKEND_FAILURE,
                str(exc) or "VMM zero_bytes failed",
                failed_part=self._part,
                failed_operation=RegionOperationKind.ZERO_BYTES,
            ) from exc

    def release_once(self) -> ProviderCleanupFailure | None:
        self._release_once_count += 1
        if self._release_once_count > 1:
            return self._first_cleanup_failure
        if self._handle is None:
            return None
        try:
            _region_vmm_release(self._handle)
        except BaseException as exc:
            self._record_cleanup_failure("release", exc)
            return self._first_cleanup_failure
        self._handle = None
        self._mapping_available = False
        return self._first_cleanup_failure

    def _record_cleanup_failure(self, step: str, exc: BaseException) -> None:
        cause = RegionCleanupCause.INTERRUPTED if _interrupt_like(exc) else RegionCleanupCause.BACKEND_ERROR
        failure = ProviderCleanupFailure(
            part=self._part,
            backend_operation=RegionOperationKind.RELEASE,
            typed_cause=cause,
        )
        self.local_cleanup_details.append((step, exc))
        if self._first_cleanup_failure is None:
            self._first_cleanup_failure = failure

    def _require_mapping(self, operation: str) -> None:
        if not self._mapping_available or self._handle is None:
            raise RegionControlError(
                RegionControlErrorKind.INTERNAL_INVARIANT,
                f"{operation} requires a materialized VMM mapping",
                failed_part=self._part,
                failed_operation=RegionOperationKind.ZERO_BYTES,
            )


def _closed_part_dispatcher(
    context: RegionAllocationContext,
    part: RegionPartKind,
    spec: RegionPartAllocationSpec,
) -> RegionPartAllocation:
    if spec.planned_backing_kind is not BackendKind.VMM_SHAREABLE or not isinstance(
        context.target, DeviceAllocationTarget
    ):
        raise RegionControlError(
            RegionControlErrorKind.INTERNAL_INVARIANT,
            "admitted allocation combination has no provider backend",
            failed_part=part,
            failed_operation=RegionOperationKind.NONE,
        )
    if context.environment_kind is RegionEnvironmentKind.SIM:
        return SimPosixShmAllocation(context, part, spec)
    if context.environment_kind is RegionEnvironmentKind.ONBOARD:
        return VmmAllocation(context, part, spec)
    raise RegionControlError(
        RegionControlErrorKind.INTERNAL_INVARIANT,
        "admitted allocation combination has no provider backend",
        failed_part=part,
        failed_operation=RegionOperationKind.NONE,
    )


def _initialize_counter_storage(allocation: RegionPartAllocation, logical_bytes: int) -> None:
    allocation.zero_bytes(0, logical_bytes)


def _store_lifecycle_error(state: ProviderRegionStoreState) -> RegionControlError:
    return RegionControlError(
        RegionControlErrorKind.STORE_LIFECYCLE,
        f"store is {state.name}",
    )


def _invalid_resource_id_error(message: str) -> RegionControlError:
    return RegionControlError(RegionControlErrorKind.INVALID_FIELD_VALUE, message)


def _interrupt_like(exc: BaseException) -> bool:
    return isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit))


def _cleanup_failure_from_exception(part: RegionPartKind, exc: BaseException) -> ProviderCleanupFailure:
    cause = RegionCleanupCause.INTERRUPTED if _interrupt_like(exc) else RegionCleanupCause.BACKEND_ERROR
    return ProviderCleanupFailure(
        part=part,
        backend_operation=RegionOperationKind.RELEASE,
        typed_cause=cause,
    )


def _coerce_part_cleanup_failure(kind: RegionPartKind, returned: object) -> ProviderCleanupFailure:
    if isinstance(returned, ProviderCleanupFailure):
        if returned.part is kind:
            return returned
        return ProviderCleanupFailure(
            part=kind,
            backend_operation=returned.backend_operation,
            typed_cause=returned.typed_cause,
        )
    return ProviderCleanupFailure(
        part=kind,
        backend_operation=RegionOperationKind.RELEASE,
        typed_cause=RegionCleanupCause.BACKEND_ERROR,
    )


class _AllocateStage:
    def __init__(self) -> None:
        self.part = RegionPartKind.INVALID
        self.operation = RegionOperationKind.NONE


class _ProviderPartRecord:
    def __init__(
        self,
        kind: RegionPartKind,
        spec: RegionPartAllocationSpec,
        identity: CanonicalIdentity,
        allocation: RegionPartAllocation,
    ) -> None:
        self.kind = kind
        self.spec = spec
        self.identity = identity
        self.allocation = allocation
        self.buffer: Buffer | None = None
        self.state = ProviderPartResourceState.SHELL
        self.cleanup_failure: ProviderCleanupFailure | None = None


class ProviderRegionResource:
    """Store-private logical record for one provider resource ID."""

    def __init__(self, provider_resource_id: int, spec: RegionAllocationSpec) -> None:
        self.provider_resource_id = provider_resource_id
        self.spec = spec
        self.state = ProviderRegionResourceState.CREATING
        self.parts: dict[RegionPartKind, _ProviderPartRecord] = {}
        self.export_descriptor: RegionExportDescriptor | None = None
        self.local_views: dict[RegionPartKind, RegionPartLocalView] = {}


class ProviderRegionStore:
    """Provider-agent physical ownership authority for one store incarnation."""

    def __init__(
        self,
        context: RegionAllocationContext,
        identity_allocator: EndpointBufferIdentityAllocator,
        *,
        _shell_factory: ShellFactory | None = None,
    ) -> None:
        if not isinstance(context, RegionAllocationContext):
            raise TypeError("context must be RegionAllocationContext")
        if identity_allocator is None:
            raise TypeError("identity_allocator is required")
        owner_nonce = _require_owner_nonce(identity_allocator.owner_instance_id)
        self._context = context
        self._identity_allocator = identity_allocator
        self._allocator_nonce = owner_nonce
        self._state = ProviderRegionStoreState.OPEN
        self._next_provider_resource_id = 1
        self._resources: dict[int, ProviderRegionResource] = {}
        self._shell_factory = _closed_part_dispatcher if _shell_factory is None else _shell_factory

    @property
    def context(self) -> RegionAllocationContext:
        return self._context

    @property
    def state(self) -> ProviderRegionStoreState:
        return self._state

    def _require_allocator(self) -> EndpointBufferIdentityAllocator:
        allocator = self._identity_allocator
        if allocator is None:
            raise RegionControlError(
                RegionControlErrorKind.INTERNAL_INVARIANT,
                "provider identity allocator is missing",
            )
        nonce = _require_owner_nonce(allocator.owner_instance_id)
        if nonce != self._allocator_nonce:
            raise RegionControlError(
                RegionControlErrorKind.INTERNAL_INVARIANT,
                "provider identity allocator nonce changed",
            )
        return allocator

    def allocate_and_export(self, spec: RegionAllocationSpec) -> RegionAllocationResult:
        self._require_open()
        if not isinstance(spec, RegionAllocationSpec):
            raise TypeError("spec must be RegionAllocationSpec")
        allocator = self._require_allocator()
        resource_id = self._burn_id()
        resource = ProviderRegionResource(resource_id, spec)
        self._resources[resource_id] = resource
        stage = _AllocateStage()
        try:
            for kind in REGION_PARTS:
                stage.part = kind
                stage.operation = RegionOperationKind.NONE
                identity = allocator.burn_identity()
                allocation = self._shell_factory(self._context, kind, spec.part(kind))
                resource.parts[kind] = _ProviderPartRecord(kind, spec.part(kind), identity, allocation)
            for kind in REGION_PARTS:
                stage.part = kind
                stage.operation = RegionOperationKind.MATERIALIZE
                part = resource.parts[kind]
                part.state = ProviderPartResourceState.MATERIALIZING
                part.buffer = part.allocation.materialize(part.identity)
            stage.part = RegionPartKind.COUNTER
            stage.operation = RegionOperationKind.ZERO_BYTES
            _initialize_counter_storage(resource.parts[RegionPartKind.COUNTER].allocation, spec.counter.logical_bytes)
            descriptor = self._freeze_descriptor(resource, stage)
            local_views = self._freeze_local_views(resource, stage)
            for part in resource.parts.values():
                part.state = ProviderPartResourceState.READY
            resource.export_descriptor = descriptor
            resource.local_views = local_views
            resource.state = ProviderRegionResourceState.ACTIVE
            return RegionAllocationResult(resource_id, descriptor)
        except BaseException as primary:
            if isinstance(primary, RegionControlError):
                if primary.failed_part is not RegionPartKind.INVALID:
                    stage.part = primary.failed_part
                if primary.failed_operation is not RegionOperationKind.NONE:
                    stage.operation = primary.failed_operation
            resource.state = ProviderRegionResourceState.CLEANUP_PENDING
            complete = self._release_installed_parts(resource)
            if complete:
                del self._resources[resource_id]
            else:
                self._state = ProviderRegionStoreState.CLOSE_FAILED
            error = self._allocation_error(primary, resource_id, stage.part, stage.operation, debt=not complete)
            raise error from primary

    def describe(self, provider_resource_id: int) -> RegionExportDescriptor:
        self._require_open()
        resource = self._require_active_resource(provider_resource_id)
        if resource.export_descriptor is None:
            raise RegionControlError(
                RegionControlErrorKind.INTERNAL_INVARIANT,
                "ACTIVE resource is missing a frozen export descriptor",
            )
        return resource.export_descriptor

    def local_view(self, provider_resource_id: int, part: RegionPartKind | int) -> RegionPartLocalView:
        self._require_open()
        kind = _require_part(part)
        resource = self._require_active_resource(provider_resource_id)
        view = resource.local_views.get(kind)
        if view is None:
            raise RegionControlError(
                RegionControlErrorKind.INTERNAL_INVARIANT,
                "ACTIVE resource is missing a frozen local view",
                failed_part=kind,
                failed_operation=RegionOperationKind.LOCAL_VIEW,
            )
        return view

    def release(self, provider_resource_id: int) -> ProviderReleaseResult:
        self._require_open()
        resource_id = self._require_resource_id(provider_resource_id)
        classified = self._classify_absent_id(resource_id)
        if classified is not None:
            return ProviderReleaseResult(provider_resource_id=resource_id, status=classified)
        resource = self._resources[resource_id]
        resource.state = ProviderRegionResourceState.CLEANUP_PENDING
        resource.export_descriptor = None
        resource.local_views = {}
        complete = self._release_installed_parts(resource)
        if complete:
            del self._resources[resource_id]
            return ProviderReleaseResult(
                provider_resource_id=resource_id,
                status=ProviderReleaseStatus.RELEASED,
            )
        self._state = ProviderRegionStoreState.CLOSE_FAILED
        return self._incomplete_result(resource)

    def sweep(self) -> tuple[ProviderReleaseResult, ...]:
        if self._state is ProviderRegionStoreState.CLOSED:
            return ()
        self._state = ProviderRegionStoreState.CLOSING
        results: list[ProviderReleaseResult] = []
        for resource_id in sorted(self._resources):
            resource = self._resources[resource_id]
            if resource.state in (
                ProviderRegionResourceState.ACTIVE,
                ProviderRegionResourceState.CREATING,
            ):
                resource.state = ProviderRegionResourceState.CLEANUP_PENDING
                resource.export_descriptor = None
                resource.local_views = {}
                complete = self._release_installed_parts(resource)
                if complete:
                    del self._resources[resource_id]
                    results.append(
                        ProviderReleaseResult(
                            provider_resource_id=resource_id,
                            status=ProviderReleaseStatus.RELEASED,
                        )
                    )
                else:
                    results.append(self._incomplete_result(resource))
            else:
                results.append(self._incomplete_result(resource))
        self._state = ProviderRegionStoreState.CLOSE_FAILED if self._resources else ProviderRegionStoreState.CLOSED
        return tuple(results)

    def _require_open(self) -> None:
        if self._state is not ProviderRegionStoreState.OPEN:
            raise _store_lifecycle_error(self._state)

    def _burn_id(self) -> int:
        resource_id = self._next_provider_resource_id
        if resource_id > _UINT64_MAX:
            raise RegionControlError(
                RegionControlErrorKind.INTERNAL_INVARIANT,
                "provider_resource_id space exhausted",
            )
        self._next_provider_resource_id = resource_id + 1
        return resource_id

    def _require_resource_id(self, provider_resource_id: object) -> int:
        resource_id = _require_uint64("provider_resource_id", provider_resource_id)
        if resource_id == 0:
            raise _invalid_resource_id_error("provider_resource_id must be nonzero")
        return resource_id

    def _classify_absent_id(self, resource_id: int) -> ProviderReleaseStatus | None:
        if resource_id in self._resources:
            return None
        if resource_id < self._next_provider_resource_id:
            return ProviderReleaseStatus.ALREADY_GONE
        return ProviderReleaseStatus.UNKNOWN_RESOURCE

    def _require_active_resource(self, provider_resource_id: object) -> ProviderRegionResource:
        resource_id = self._require_resource_id(provider_resource_id)
        resource = self._resources.get(resource_id)
        if resource is None:
            if resource_id < self._next_provider_resource_id:
                raise _invalid_resource_id_error("provider_resource_id is already gone")
            raise _invalid_resource_id_error("provider_resource_id is unknown")
        if resource.state is not ProviderRegionResourceState.ACTIVE:
            raise RegionControlError(
                RegionControlErrorKind.STORE_LIFECYCLE,
                f"resource is {resource.state.name}",
            )
        return resource

    def _freeze_descriptor(self, resource: ProviderRegionResource, stage: _AllocateStage) -> RegionExportDescriptor:
        exports: dict[RegionPartKind, BufferDescriptor] = {}
        for kind in REGION_PARTS:
            stage.part = kind
            stage.operation = RegionOperationKind.DESCRIBE
            part = resource.parts[kind]
            buffer = part.buffer
            if buffer is None:
                raise RegionControlError(
                    RegionControlErrorKind.INTERNAL_INVARIANT,
                    "READY part is missing its canonical Buffer",
                    failed_part=kind,
                    failed_operation=RegionOperationKind.DESCRIBE,
                )
            try:
                exports[kind] = buffer.to_descriptor()
            except RegionControlError:
                raise
            except Exception as exc:
                raise RegionControlError(
                    RegionControlErrorKind.INTERNAL_INVARIANT,
                    str(exc) or "export descriptor construction failed",
                    failed_part=kind,
                    failed_operation=RegionOperationKind.DESCRIBE,
                ) from exc
            descriptor = exports[kind]
            if descriptor.identity != part.identity:
                raise RegionControlError(
                    RegionControlErrorKind.INTERNAL_INVARIANT,
                    "Buffer identity must match the burned part identity",
                    failed_part=kind,
                    failed_operation=RegionOperationKind.DESCRIBE,
                )
            if int(descriptor.nbytes) != int(part.spec.logical_bytes):
                raise RegionControlError(
                    RegionControlErrorKind.INTERNAL_INVARIANT,
                    "Buffer nbytes must match the admitted logical_bytes",
                    failed_part=kind,
                    failed_operation=RegionOperationKind.DESCRIBE,
                )
        return RegionExportDescriptor(
            payload=exports[RegionPartKind.PAYLOAD],
            counter=exports[RegionPartKind.COUNTER],
        )

    def _freeze_local_views(
        self, resource: ProviderRegionResource, stage: _AllocateStage
    ) -> dict[RegionPartKind, RegionPartLocalView]:
        views: dict[RegionPartKind, RegionPartLocalView] = {}
        for kind in REGION_PARTS:
            stage.part = kind
            stage.operation = RegionOperationKind.LOCAL_VIEW
            part = resource.parts[kind]
            buffer = part.buffer
            if buffer is None:
                raise RegionControlError(
                    RegionControlErrorKind.INTERNAL_INVARIANT,
                    "READY part is missing its canonical Buffer",
                    failed_part=kind,
                    failed_operation=RegionOperationKind.LOCAL_VIEW,
                )
            try:
                views[kind] = RegionPartLocalView(
                    part=kind,
                    local_base=int(buffer.base),
                    logical_bytes=int(buffer.nbytes),
                )
            except RegionControlError:
                raise
            except Exception as exc:
                raise RegionControlError(
                    RegionControlErrorKind.INTERNAL_INVARIANT,
                    str(exc) or "local view construction failed",
                    failed_part=kind,
                    failed_operation=RegionOperationKind.LOCAL_VIEW,
                ) from exc
        try:
            validate_independent_local_views(views[RegionPartKind.PAYLOAD], views[RegionPartKind.COUNTER])
        except ValueError as exc:
            raise RegionControlError(
                RegionControlErrorKind.INTERNAL_INVARIANT,
                str(exc),
                failed_part=RegionPartKind.INVALID,
                failed_operation=RegionOperationKind.LOCAL_VIEW,
            ) from exc
        return views

    def _logical_close_part_buffers(self, resource: ProviderRegionResource) -> None:
        for kind in REGION_PARTS:
            part = resource.parts.get(kind)
            if part is None or part.buffer is None:
                continue
            try:
                part.buffer.close()
            except BaseException:
                part.buffer.closed = True

    def _release_installed_parts(self, resource: ProviderRegionResource) -> bool:
        self._logical_close_part_buffers(resource)
        for kind in REGION_PARTS:
            part = resource.parts.get(kind)
            if part is None or part.state is ProviderPartResourceState.RELEASED:
                continue
            part.state = ProviderPartResourceState.CLEANUP_PENDING
            try:
                returned = part.allocation.release_once()
            except BaseException as exc:
                part.cleanup_failure = _cleanup_failure_from_exception(kind, exc)
                continue
            if returned is None:
                part.state = ProviderPartResourceState.RELEASED
                part.cleanup_failure = None
            else:
                part.cleanup_failure = _coerce_part_cleanup_failure(kind, returned)
        return all(part.state is ProviderPartResourceState.RELEASED for part in resource.parts.values())

    def _incomplete_result(self, resource: ProviderRegionResource) -> ProviderReleaseResult:
        failures: list[ProviderCleanupFailure] = []
        for kind in REGION_PARTS:
            part = resource.parts.get(kind)
            if part is None or part.state is ProviderPartResourceState.RELEASED:
                continue
            if part.cleanup_failure is None:
                part.cleanup_failure = ProviderCleanupFailure(
                    part=kind,
                    backend_operation=RegionOperationKind.RELEASE,
                    typed_cause=RegionCleanupCause.BACKEND_STATE_MISMATCH,
                )
            failures.append(part.cleanup_failure)
        return ProviderReleaseResult(
            provider_resource_id=resource.provider_resource_id,
            status=ProviderReleaseStatus.CLEANUP_INCOMPLETE,
            failures=tuple(failures),
        )

    def _allocation_error(
        self,
        primary: BaseException,
        resource_id: int,
        part: RegionPartKind,
        operation: RegionOperationKind,
        *,
        debt: bool,
    ) -> RegionAllocationError:
        kind = RegionControlErrorKind.INTERNAL_INVARIANT
        if isinstance(primary, RegionControlError) and primary.kind in _ALLOCATION_ERROR_KINDS:
            kind = primary.kind
        if kind is RegionControlErrorKind.BACKEND_FAILURE and operation not in _BACKEND_FAILURE_OPERATIONS:
            kind = RegionControlErrorKind.INTERNAL_INVARIANT
            operation = RegionOperationKind.NONE
        return RegionAllocationError(
            provisional_resource_id=resource_id,
            control_kind=kind,
            failed_part=part,
            failed_operation=operation,
            cleanup_debt_remaining=debt,
            message=str(primary),
        )
