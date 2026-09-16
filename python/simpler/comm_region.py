# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Private region materialization helpers."""

from __future__ import annotations

import ctypes
import itertools
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any

from _task_interface import (  # pyright: ignore[reportMissingImports]
    _host_vmm_copy_from,
    _host_vmm_copy_to,
    _region_counter_notify,
    _region_counter_test,
    _region_counter_wait,
    _worker_host_mapped_region_close,
)

from .buffer import AddressSpace, Buffer
from .comm_endpoints import (
    DEVICE_AICPU,
    HOST_CPU,
    AdapterKind,
    AdapterProfile,
    AttachmentRole,
    BackendKind,
    BackendPlan,
    EndpointDeploymentKind,
    EndpointRecord,
    EndpointRegistry,
    MemberAttachmentPlan,
    RegionLayoutSpec,
    RegionPartPlan,
    RegionTopologyKind,
    SingleOwnerPlan,
    UnsupportedRegionPlan,
    parse_endpoint_path,
)
from .comm_provider import (
    PosixShmImport,
    ProviderReleaseStatus,
    RegionAllocationError,
    RegionAllocationResult,
    RegionAllocationSpec,
    RegionControlError,
    RegionControlErrorKind,
    RegionPartAllocationSpec,
    RegionPartKind,
    RegionPartLocalView,
    VmmShareableHandleImport,
    validate_independent_local_views,
)
from .comm_provider_control import (
    ALLOCATE_REPLY_BYTES,
    ALLOCATE_REQUEST_HARD_CEILING,
    DelegatedAllocateReply,
    DelegatedAllocateReplyTag,
    DelegatedAllocateRequest,
    encode_request,
    parse_reply,
)

_GENERATION_COUNTER = itertools.count(1)
_MAX_SIGNED_CHRONO_TIMEOUT_NS = 2**63 - 1
_WAIT_STATUS_TIMEOUT = -1
_WAIT_ERROR_SIGNAL_TIMEOUT = 7
_UINT64_MAX = (1 << 64) - 1
_SESSION_INSTANCE_ID_BYTES = 8


class NotifyOp(IntEnum):
    Set = 0
    Add = 1


class WaitCmp(IntEnum):
    EQ = 0
    NE = 1
    GT = 2
    GE = 3
    LT = 4
    LE = 5


@dataclass(frozen=True)
class SignalTestResult:
    matched: bool
    observed: int


@dataclass(frozen=True)
class RegionPartSpan:
    offset: int
    nbytes: int

    def validate_range(self, offset: int, nbytes: int, label: str) -> None:
        offset = int(offset)
        nbytes = int(nbytes)
        if offset < 0 or nbytes <= 0:
            raise ValueError(f"{label} offset must be non-negative and nbytes must be positive")
        if offset + nbytes > int(self.nbytes):
            raise ValueError(f"{label} range [{offset}, {offset + nbytes}) exceeds part size {int(self.nbytes)}")


class _PinnedBuffer:
    def __init__(self, obj: Any, *, writable: bool = False) -> None:
        self._keepalive: Any = obj
        if isinstance(obj, Buffer):
            if obj.address_space != AddressSpace.HOST:
                raise ValueError("region payload buffer must be host storage, not device storage")
            self.addr = int(obj.base)
            self.nbytes = int(obj.nbytes)
            return

        try:
            view = memoryview(obj)
        except TypeError as exc:
            raise ValueError("region payload buffer must be a contiguous host-accessible byte span") from exc
        if not view.c_contiguous:
            raise ValueError("region payload buffer must be contiguous")
        try:
            byte_view = view if view.itemsize == 1 and view.format in {"B", "b", "c"} else view.cast("B")
        except (TypeError, ValueError) as exc:
            raise ValueError("region payload buffer must be viewable as bytes") from exc
        if writable and byte_view.readonly:
            raise ValueError("region payload read destination must be writable")
        self.nbytes = int(byte_view.nbytes)
        if byte_view.readonly:
            staging = ctypes.create_string_buffer(byte_view.tobytes())
            self._keepalive = staging
            self.addr = ctypes.addressof(staging)
            return
        exported = ctypes.c_char.from_buffer(byte_view)
        self._keepalive = (byte_view, exported)
        self.addr = ctypes.addressof(exported)

    def close(self) -> None:
        self._keepalive = None

    def __enter__(self) -> _PinnedBuffer:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class HostVmmCopyAccess:
    def __init__(self, handle: Any) -> None:
        self._handle = handle

    @classmethod
    def from_mapping(cls, mapping: Any) -> HostVmmCopyAccess:
        return cls(getattr(mapping, "handle", mapping))

    @property
    def handle(self) -> Any:
        return self._handle

    def write_bytes(self, span: RegionPartSpan, offset: int, host_ptr: int, nbytes: int) -> None:
        span.validate_range(offset, nbytes, "region byte write")
        _host_vmm_copy_to(int(self._handle), int(span.offset) + int(offset), int(host_ptr), int(nbytes))

    def read_bytes(self, span: RegionPartSpan, offset: int, host_ptr: int, nbytes: int) -> None:
        span.validate_range(offset, nbytes, "region byte read")
        _host_vmm_copy_from(int(self._handle), int(span.offset) + int(offset), int(host_ptr), int(nbytes))


class PayloadPart:
    def __init__(self, span: RegionPartSpan, access: Any) -> None:
        self._span = span
        self._access = access

    @property
    def span(self) -> RegionPartSpan:
        return self._span

    def write(self, offset: int, host_buffer: Any, nbytes: int | None = None) -> None:
        with _PinnedBuffer(host_buffer) as pinned:
            size = pinned.nbytes if nbytes is None else int(nbytes)
            self._validate_payload_range(offset, size, pinned.nbytes)
            self._access.write_bytes(self._span, int(offset), pinned.addr, size)

    def read(self, offset: int, host_buffer: Any, nbytes: int | None = None) -> None:
        with _PinnedBuffer(host_buffer, writable=True) as pinned:
            size = pinned.nbytes if nbytes is None else int(nbytes)
            self._validate_payload_range(offset, size, pinned.nbytes)
            self._access.read_bytes(self._span, int(offset), pinned.addr, size)

    def _validate_payload_range(self, offset: int, nbytes: int, buffer_nbytes: int) -> None:
        if int(nbytes) > int(buffer_nbytes):
            raise ValueError(f"region payload nbytes={int(nbytes)} exceeds host buffer size {int(buffer_nbytes)}")
        self._span.validate_range(offset, nbytes, "region payload")


class CounterPart:
    def __init__(self, span: RegionPartSpan, access: Any, handle: Any | None = None) -> None:
        self._span = span
        self._access = access
        resolved_handle = getattr(access, "handle", handle)
        if resolved_handle is None:
            raise ValueError("region counter access requires a native mapped-region handle")
        self._handle = int(resolved_handle)

    @property
    def span(self) -> RegionPartSpan:
        return self._span

    def counter(self, offset: int) -> RegionCounter:
        self._validate_counter_offset(offset)
        return RegionCounter(self, int(offset))

    def notify(self, offset: int, value: int, op: NotifyOp) -> None:
        self._validate_counter_offset(offset)
        op = NotifyOp(op)
        _region_counter_notify(self._handle, self._mapping_offset(offset), int(value), int(op))

    def test(self, offset: int, cmp_value: int, cmp: WaitCmp) -> SignalTestResult:
        self._validate_counter_offset(offset)
        cmp = WaitCmp(cmp)
        matched, observed = _region_counter_test(self._handle, self._mapping_offset(offset), int(cmp_value), int(cmp))
        return SignalTestResult(matched=bool(matched), observed=int(observed))

    def wait(self, offset: int, cmp_value: int, cmp: WaitCmp, timeout: float) -> int:
        self._validate_counter_offset(offset)
        cmp = WaitCmp(cmp)
        if timeout is None or float(timeout) <= 0:
            raise ValueError("region counter wait requires a positive timeout")
        timeout_ns = min(int(float(timeout) * 1_000_000_000), _MAX_SIGNED_CHRONO_TIMEOUT_NS)
        status, error_kind, observed, _matched, message = _region_counter_wait(
            self._handle, self._mapping_offset(offset), int(cmp_value), int(cmp), int(timeout_ns)
        )
        if int(status) == 0:
            return int(observed)
        msg = str(message) if message else "region counter wait timed out"
        if int(status) == _WAIT_STATUS_TIMEOUT and int(error_kind) == _WAIT_ERROR_SIGNAL_TIMEOUT:
            raise TimeoutError(f"{msg}; observed={int(observed)}")
        raise AssertionError(f"unexpected region counter wait result status={int(status)} error_kind={int(error_kind)}")

    def _mapping_offset(self, offset: int) -> int:
        return int(self._span.offset) + int(offset)

    def _validate_counter_offset(self, offset: int) -> None:
        offset = int(offset)
        if offset < 0 or offset % 4 != 0 or offset + 4 > int(self._span.nbytes):
            raise ValueError("region counter offset must be 4-byte aligned and inside the counter range")


class RegionCounter:
    def __init__(self, part: CounterPart, offset: int) -> None:
        self._part = part
        self._offset = int(offset)

    @property
    def offset(self) -> int:
        return self._offset

    def notify(self, value: int, op: NotifyOp = NotifyOp.Set) -> None:
        self._part.notify(self._offset, int(value), op)

    def test(self, cmp_value: int, cmp: WaitCmp) -> SignalTestResult:
        return self._part.test(self._offset, int(cmp_value), cmp)

    def wait(self, cmp_value: int, cmp: WaitCmp, timeout: float) -> int:
        return self._part.wait(self._offset, int(cmp_value), cmp, timeout)


class RegionInstanceState(str, Enum):
    LIVE = "LIVE"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    CLOSE_FAILED = "CLOSE_FAILED"


class RefusalReason(str, Enum):
    UNSUPPORTED_PLAN = "UNSUPPORTED_PLAN"
    NEEDS_DELEGATION = "NEEDS_DELEGATION"
    UNSUPPORTED_MEMBER_SHAPE = "UNSUPPORTED_MEMBER_SHAPE"
    UNSUPPORTED_PROVIDER_DEPLOYMENT = "UNSUPPORTED_PROVIDER_DEPLOYMENT"
    UNSUPPORTED_BACKEND_KIND = "UNSUPPORTED_BACKEND_KIND"
    UNSUPPORTED_ATTACHMENT = "UNSUPPORTED_ATTACHMENT"
    REGISTRY_MISMATCH = "REGISTRY_MISMATCH"


class MaterializationError(RuntimeError):
    pass


class MaterializationRefusal(MaterializationError):
    def __init__(self, reason: RefusalReason, message: str) -> None:
        self.reason = reason
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class MaterializationContext:
    worker: Any
    registry: EndpointRegistry
    plan: BackendPlan | UnsupportedRegionPlan
    layout: RegionLayoutSpec


@dataclass(frozen=True)
class SingleOwnerRegionShape:
    provider: EndpointRecord
    consumer: EndpointRecord
    worker_id: int


@dataclass(frozen=True)
class DelegatedSingleOwnerRegionShape:
    provider: EndpointRecord
    consumer: EndpointRecord
    initiator_path: bytes
    provider_path: bytes
    first_hop_child_id: int
    provider_device_id: int

    @property
    def worker_id(self) -> int:
        return int(self.first_hop_child_id)


def _require_session_instance_id(value: object) -> bytes:
    if isinstance(value, memoryview):
        raw = value.tobytes()
    elif isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    else:
        raise MaterializationRefusal(
            RefusalReason.REGISTRY_MISMATCH,
            "session_instance_id must be 8 opaque bytes",
        )
    if len(raw) != _SESSION_INSTANCE_ID_BYTES:
        raise MaterializationRefusal(
            RefusalReason.REGISTRY_MISMATCH,
            "session_instance_id must be 8 opaque bytes",
        )
    return raw


class RegionInstanceRegistry:
    """Strong reachability for consumer instances. Not an ID-addressed data path."""

    def __init__(self) -> None:
        self._instances: dict[int, RegionInstance] = {}
        self._run_scopes: dict[int, Any] = {}
        self._delegated_session_instance_id: bytes | None = None
        self._next_delegated_transaction_id: int = 1
        self._delegated_admission_closed: bool = False
        self._delegated_allocate_dispatch_lock = threading.Lock()

    def close_delegated_admission(self) -> None:
        self._delegated_admission_closed = True

    def track(self, instance: RegionInstance, run_scope: Any) -> None:
        key = id(instance)
        if key in self._instances:
            raise MaterializationError("region instance is already tracked")
        self._instances[key] = instance
        self._run_scopes[key] = run_scope

    def _require_tracked(self, instance: RegionInstance) -> None:
        if id(instance) not in self._instances:
            raise MaterializationError("region instance must be tracked before delegated dispatch")

    @contextmanager
    def _delegated_allocate_dispatch(
        self, *, registry: object, expected_registry_epoch: int
    ) -> Iterator[tuple[bytes, int]]:
        with self._delegated_allocate_dispatch_lock:
            session = _require_session_instance_id(getattr(registry, "session_instance_id", None))
            if self._delegated_session_instance_id is None:
                self._delegated_session_instance_id = session
            elif self._delegated_session_instance_id != session:
                raise MaterializationRefusal(
                    RefusalReason.REGISTRY_MISMATCH,
                    "delegated allocate observed a different session nonce on this Worker incarnation",
                )
            if int(getattr(registry, "registry_epoch", -1)) != int(expected_registry_epoch):
                raise MaterializationRefusal(
                    RefusalReason.REGISTRY_MISMATCH,
                    "delegated allocate observed a registry epoch mismatch",
                )
            if self._delegated_admission_closed or self._next_delegated_transaction_id > _UINT64_MAX:
                self._delegated_admission_closed = True
                raise MaterializationError("delegated transaction id space is exhausted")
            transaction_id = int(self._next_delegated_transaction_id)
            if transaction_id < 1:
                self._delegated_admission_closed = True
                raise MaterializationError("delegated transaction_id 0 is illegal")
            self._next_delegated_transaction_id = transaction_id + 1
            if self._next_delegated_transaction_id > _UINT64_MAX:
                self._delegated_admission_closed = True
            yield session, transaction_id

    def close(self, instance: RegionInstance) -> None:
        try:
            instance._close_owned(poison_on_error=True)
        finally:
            self._settle(instance)

    def cleanup_run(self, run_scope: Any) -> None:
        errors: list[BaseException] = []
        for instance in self._iter_run(run_scope):
            if instance._close_attempted or instance._state in (
                RegionInstanceState.CLOSED,
                RegionInstanceState.CLOSE_FAILED,
            ):
                self._settle(instance)
                continue
            try:
                self.close(instance)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            raise errors[0]

    def sweep(self) -> None:
        errors: list[BaseException] = []
        for instance in tuple(self._instances.values()):
            if not instance._close_attempted and instance._state not in (
                RegionInstanceState.CLOSED,
                RegionInstanceState.CLOSE_FAILED,
            ):
                try:
                    instance._close_owned(poison_on_error=True)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)
            self._retire(id(instance))
        if errors:
            raise errors[0]

    def record_data_plane_failure(self, run_scope: Any, resource_id: int, error: BaseException) -> None:
        resource_id = int(resource_id)
        matches = [
            instance for instance in self._iter_run(run_scope) if int(instance.provider_resource_id) == resource_id
        ]
        if not matches:
            raise MaterializationError(f"no region instance for resource {resource_id}")
        if len(matches) > 1:
            raise MaterializationError(f"duplicate region instances for resource {resource_id}")
        instance = matches[0]
        if instance._data_plane_error is None:
            instance._data_plane_error = error

    def record_data_plane_failure_by_allocation_identity(
        self,
        run_scope: object,
        session_instance_id: bytes,
        transaction_id: int,
        error: BaseException,
    ) -> None:
        session = bytes(session_instance_id)
        transaction = int(transaction_id)
        matches: list[RegionInstance] = []
        for instance in self._iter_run(run_scope):
            try:
                identity = instance._allocation_identity
            except MaterializationError:
                continue
            if identity == (session, transaction):
                matches.append(instance)
        if not matches:
            raise MaterializationError("no region instance for allocation identity")
        if len(matches) > 1:
            raise MaterializationError("duplicate region instances for allocation identity")
        instance = matches[0]
        if instance._data_plane_error is None:
            instance._data_plane_error = error

    def _iter_run(self, run_scope: Any) -> tuple[RegionInstance, ...]:
        return tuple(instance for key, instance in self._instances.items() if self._run_scopes[key] is run_scope)

    def _settle(self, instance: RegionInstance) -> None:
        if id(instance) not in self._instances:
            return
        if instance._retains_cleanup_only_reachability():
            return
        self._retire(id(instance))

    def _retire(self, key: int) -> None:
        self._instances.pop(key, None)
        self._run_scopes.pop(key, None)


class RegionInstance:
    def __init__(self, ctx: MaterializationContext, shape: SingleOwnerRegionShape) -> None:
        self.plan = ctx.plan
        self.layout = ctx.layout
        self.provider = shape.provider
        self.consumer = shape.consumer
        self.worker_id = int(shape.worker_id)
        self.generation = next(_GENERATION_COUNTER)
        self.diagnostic_label = (
            f"{self.consumer.path} {self.consumer.deployment.value} -> "
            f"{self.provider.path} {self.provider.deployment.value} "
            f"payload={int(self.layout.payload_bytes)} counter={int(self.layout.counter_bytes)}"
        )
        self._worker = ctx.worker
        self._cleanup_resources = getattr(ctx.worker, "_building_run_resources", None)
        self._payload_mapping: Any | None = None
        self._counter_mapping: Any | None = None
        self._payload_part: PayloadPart | None = None
        self._counter_part: CounterPart | None = None
        self._payload_local_view: RegionPartLocalView | None = None
        self._counter_local_view: RegionPartLocalView | None = None
        self._provider_resource_id = 0
        self._delegated_session_instance_id: bytes | None = None
        self._delegated_transaction_id: int = 0
        self._delegated_provider_path: bytes | None = None
        self._delegated_allocation_committed: bool = False
        self._delegated_release_edge: bool = False
        self._delegated_provider_device_id: int | None = None
        self._state: RegionInstanceState | None = None
        self._cleanup_error: BaseException | None = None
        self._close_attempted = False
        self._ever_live = False
        self._provider_release_committed = False
        self._data_plane_error: BaseException | None = None

    @property
    def state(self) -> RegionInstanceState:
        if self._state is None:
            raise MaterializationError("region instance is not published")
        return self._state

    @property
    def provider_resource_id(self) -> int:
        return int(self._provider_resource_id)

    @property
    def _allocation_identity(self) -> tuple[bytes, int]:
        session = self._delegated_session_instance_id
        if not isinstance(session, (bytes, bytearray)) or len(bytes(session)) != _SESSION_INSTANCE_ID_BYTES:
            raise MaterializationError("region allocation identity is incomplete")
        transaction_id = self._delegated_transaction_id
        if type(transaction_id) is not int or transaction_id < 1 or transaction_id > _UINT64_MAX:
            raise MaterializationError("region allocation identity is incomplete")
        return bytes(session), int(transaction_id)

    @property
    def data_plane_error(self) -> BaseException | None:
        return self._data_plane_error

    def local_view(self, part: RegionPartKind) -> RegionPartLocalView | None:
        if self._state is not RegionInstanceState.LIVE:
            return None
        selected = RegionPartKind(part)
        if selected is RegionPartKind.PAYLOAD:
            return self._payload_local_view
        if selected is RegionPartKind.COUNTER:
            return self._counter_local_view
        raise ValueError("region part must be PAYLOAD or COUNTER")

    @classmethod
    def planned(
        cls, ctx: MaterializationContext, shape: SingleOwnerRegionShape | DelegatedSingleOwnerRegionShape
    ) -> RegionInstance:
        if isinstance(shape, DelegatedSingleOwnerRegionShape):
            instance = cls(
                ctx,
                SingleOwnerRegionShape(
                    provider=shape.provider,
                    consumer=shape.consumer,
                    worker_id=int(shape.first_hop_child_id),
                ),
            )
            instance._delegated_provider_device_id = int(shape.provider_device_id)
            return instance
        return cls(ctx, shape)

    def payload_write(self, offset: int, host_buffer: Any, nbytes: int | None = None) -> None:
        self._ensure_live()
        self._worker._require_region_control_context("region_instance.payload_write")
        payload_part = self._payload_part
        assert payload_part is not None
        payload_part.write(offset, host_buffer, nbytes)

    def payload_read(self, offset: int, host_buffer: Any, nbytes: int | None = None) -> None:
        self._ensure_live()
        self._worker._require_region_control_context("region_instance.payload_read")
        payload_part = self._payload_part
        assert payload_part is not None
        payload_part.read(offset, host_buffer, nbytes)

    def counter(self, offset: int):
        self._ensure_live()
        self._worker._require_region_control_context("region_instance.counter")
        counter_part = self._counter_part
        assert counter_part is not None
        return counter_part.counter(offset)

    def close(self) -> None:
        if self._state is RegionInstanceState.CLOSED:
            return
        if self._state is RegionInstanceState.CLOSE_FAILED and self._cleanup_error is not None:
            raise self._cleanup_error
        if self._state is None and self._provider_resource_id == 0 and self._payload_mapping is None:
            self._state = RegionInstanceState.CLOSED
            self._worker._region_instance_registry._settle(self)
            return
        self._worker._require_region_control_before_submit("region_instance.close")
        self._worker._region_instance_registry.close(self)

    def _bind_delegated_identity(self, session_instance_id: bytes, transaction_id: int, provider_path: bytes) -> None:
        self._worker._region_instance_registry._require_tracked(self)
        session = _require_session_instance_id(session_instance_id)
        if type(transaction_id) is not int or transaction_id < 1 or transaction_id > _UINT64_MAX:
            raise MaterializationError("delegated transaction_id 0 is illegal")
        path = bytes(provider_path)
        if not path:
            raise MaterializationError("delegated provider path must be non-empty")
        if self._delegated_session_instance_id is not None or self._delegated_transaction_id != 0:
            raise MaterializationError("delegated identity is already bound")
        self._delegated_session_instance_id = session
        self._delegated_transaction_id = transaction_id
        self._delegated_provider_path = path

    def _commit_delegated_allocation(self, provider_resource_id: int) -> None:
        resource_id = int(provider_resource_id)
        if resource_id <= 0:
            raise MaterializationError("ALLOCATED requires a nonzero provider resource id")
        if self._delegated_session_instance_id is None or self._delegated_transaction_id < 1:
            raise MaterializationError("delegated identity must be bound before ALLOCATED commit")
        if self._delegated_provider_path is None:
            raise MaterializationError("delegated provider path must be bound before ALLOCATED commit")
        self._provider_resource_id = resource_id
        self._delegated_allocation_committed = True
        self._delegated_release_edge = True

    def _abort_materialization(self, cause: BaseException) -> None:
        if isinstance(cause, RegionAllocationError):
            poison = bool(cause.cleanup_debt_remaining)
        else:
            poison = False
        close_error: BaseException | None = None
        try:
            self._close_owned(poison_on_error=False)
        except BaseException as exc:  # noqa: BLE001
            close_error = exc
        if not poison:
            if close_error is None and self._state is not RegionInstanceState.CLOSE_FAILED:
                self._state = RegionInstanceState.CLOSED
            elif close_error is not None:
                self._cleanup_error = close_error
                self._state = RegionInstanceState.CLOSE_FAILED
            return
        if close_error is not None and close_error.__cause__ is None:
            close_error.__cause__ = cause
        self._cleanup_error = self._worker._record_unreclaimable(
            f"region instance: committed allocation could not be published on worker {self.worker_id}; "
            "no further work is admitted",
            close_error or cause,
        )
        self._state = RegionInstanceState.CLOSE_FAILED

    def _close_mapping_leases(self) -> list[BaseException]:
        errors: list[BaseException] = []
        for lease in (self._payload_mapping, self._counter_mapping):
            if lease is None:
                continue
            try:
                closer = getattr(lease, "close", None)
                if closer is not None:
                    closer()
                else:
                    _worker_host_mapped_region_close(int(lease))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        return errors

    def _release_provider_resource(self) -> BaseException | None:
        if self._delegated_release_edge:
            return self._release_delegated_resource()
        return None

    def _release_delegated_resource(self) -> BaseException | None:
        if (
            self._delegated_session_instance_id is None
            or self._delegated_transaction_id < 1
            or self._delegated_provider_path is None
        ):
            return RuntimeError("delegated release edge is missing identity")
        dispatcher = getattr(self._worker, "_dispatch_delegated_release", None)
        if not callable(dispatcher):
            return RuntimeError("delegated release requires Worker dispatch")
        try:
            result = dispatcher(
                session_instance_id=self._delegated_session_instance_id,
                transaction_id=self._delegated_transaction_id,
                provider_path=self._delegated_provider_path,
            )
        except BaseException as exc:  # noqa: BLE001
            return exc
        return self._interpret_provider_release(result)

    def _interpret_provider_release(self, result: object) -> BaseException | None:
        status = getattr(result, "status", None)
        if status in (ProviderReleaseStatus.RELEASED, ProviderReleaseStatus.ALREADY_GONE):
            self._provider_release_committed = True
            self._delegated_release_edge = False
            return None
        if status is ProviderReleaseStatus.CLEANUP_INCOMPLETE:
            return RuntimeError(
                f"region instance: provider cleanup incomplete for resource {self._provider_resource_id}"
            )
        if status is ProviderReleaseStatus.UNKNOWN_RESOURCE:
            return RuntimeError(f"region instance: provider resource {self._provider_resource_id} is unknown")
        return RuntimeError(
            f"region instance: delegated release failed for transaction {self._delegated_transaction_id}"
        )

    def _close_owned(self, *, poison_on_error: bool) -> None:
        if self._state is RegionInstanceState.CLOSED:
            return
        if self._state is RegionInstanceState.CLOSE_FAILED and self._cleanup_error is not None:
            raise self._cleanup_error
        if self._close_attempted:
            if self._cleanup_error is not None:
                raise self._cleanup_error
            return
        self._close_attempted = True
        if self._state is RegionInstanceState.LIVE:
            self._state = RegionInstanceState.CLOSING
        errors = self._close_mapping_leases()
        release_error = self._release_provider_resource()
        if release_error is not None:
            errors.append(release_error)
        if not errors:
            self._state = RegionInstanceState.CLOSED
            return
        self._cleanup_error = errors[0]
        self._state = RegionInstanceState.CLOSE_FAILED
        if poison_on_error:
            self._worker._record_unreclaimable(
                f"region instance: resource {self._provider_resource_id} on worker {self.worker_id} "
                "could not be fully reclaimed; no further work is admitted",
                self._cleanup_error,
            )
        raise self._cleanup_error

    def _fail_terminal(self, error: BaseException) -> None:
        self._cleanup_error = error
        self._state = RegionInstanceState.CLOSE_FAILED

    def _retains_cleanup_only_reachability(self) -> bool:
        if self._state is RegionInstanceState.CLOSED:
            return False
        if self._state is RegionInstanceState.CLOSE_FAILED:
            return not (self._ever_live and self._provider_release_committed)
        return True

    def _ensure_live(self) -> None:
        if self._state is not RegionInstanceState.LIVE:
            label = "unpublished" if self._state is None else self._state.value
            raise MaterializationError(f"region instance is not live: {label}")


def project_region_allocation_spec(plan: object, layout: object) -> RegionAllocationSpec:
    if not isinstance(plan, BackendPlan):
        raise TypeError("project_region_allocation_spec requires a BackendPlan")
    if not isinstance(layout, RegionLayoutSpec):
        raise TypeError("project_region_allocation_spec requires a RegionLayoutSpec")
    return RegionAllocationSpec(
        payload=RegionPartAllocationSpec(
            planned_backing_kind=plan.payload.backend_kind,
            logical_bytes=int(layout.payload_bytes),
        ),
        counter=RegionPartAllocationSpec(
            planned_backing_kind=plan.counter.backend_kind,
            logical_bytes=int(layout.counter_bytes),
        ),
    )


def validate_single_owner_region_shape(ctx: MaterializationContext) -> DelegatedSingleOwnerRegionShape:
    _validate_registry_matches_worker(ctx)
    plan = ctx.plan
    if isinstance(plan, UnsupportedRegionPlan):
        raise MaterializationRefusal(RefusalReason.UNSUPPORTED_PLAN, plan.message)
    if not isinstance(plan, BackendPlan):
        raise MaterializationRefusal(RefusalReason.UNSUPPORTED_PLAN, "materializer expects a BackendPlan")
    if not isinstance(plan.topology_plan, SingleOwnerPlan):
        raise MaterializationRefusal(RefusalReason.UNSUPPORTED_PLAN, "Only SingleOwner region plans are supported")
    root_path = f"L{int(ctx.registry.root_level)}"
    provider = _record_for(ctx, plan.topology_plan.provider_endpoint)
    if provider.deployment is not DEVICE_AICPU:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_PROVIDER_DEPLOYMENT,
            "Only DEVICE_AICPU providers are supported for worker-chip regions",
        )
    member_records = tuple(_record_for(ctx, member) for member in plan.ordered_members)
    if len(member_records) != 2:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
            "Only one host consumer and one device provider are supported",
        )
    host_consumers = [member for member in member_records if member.deployment is HOST_CPU]
    if len(host_consumers) != 1 or host_consumers[0].path != root_path:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
            "The current registry-root HOST_CPU endpoint must be the only consumer",
        )
    consumer = host_consumers[0]
    if provider.identity not in plan.ordered_members or consumer.identity not in plan.ordered_members:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
            "Ordered members must contain the provider and consumer endpoints",
        )
    if not ctx.registry.same_node(provider, consumer):
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
            "Worker-chip region materialization only supports local endpoints",
        )
    _validate_part(plan.payload, provider, consumer)
    _validate_part(plan.counter, provider, consumer)
    first_hop_child_id, provider_device_id = _resolve_delegated_provider_route(ctx.worker, provider.path)
    return DelegatedSingleOwnerRegionShape(
        provider=provider,
        consumer=consumer,
        initiator_path=consumer.path.encode("ascii"),
        provider_path=provider.path.encode("ascii"),
        first_hop_child_id=first_hop_child_id,
        provider_device_id=provider_device_id,
    )


def _resolve_delegated_provider_route(worker: Any, provider_path: str) -> tuple[int, int]:
    root_level = int(getattr(worker, "level", -1))
    try:
        parsed = parse_endpoint_path(provider_path, root_level=root_level)
    except ValueError as exc:
        raise MaterializationRefusal(
            RefusalReason.NEEDS_DELEGATION,
            "provider is not a descendant of the current registry root",
        ) from exc
    if len(parsed.segments) < 2:
        raise MaterializationRefusal(
            RefusalReason.NEEDS_DELEGATION,
            "provider is not a descendant of the current registry root",
        )
    hop = parsed.segments[1]
    if hop.index is None:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
            "first hop must include a child index",
        )
    first_hop_child_id = int(hop.index)
    if first_hop_child_id in tuple(int(worker_id) for worker_id in getattr(worker, "_remote_worker_ids", ())):
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
            "remote child is not supported",
        )
    if root_level == 3:
        if len(parsed.segments) != 2 or hop.level != 2:
            raise MaterializationRefusal(RefusalReason.NEEDS_DELEGATION, "provider is not a local L3/L2 endpoint")
        device_ids = list(worker._config.get("device_ids", []))
        if first_hop_child_id < 0 or first_hop_child_id >= len(device_ids):
            raise MaterializationRefusal(
                RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
                f"provider worker_id {first_hop_child_id} is outside the current L3 device list",
            )
        return first_hop_child_id, int(device_ids[first_hop_child_id])
    children = {
        int(worker_id): child
        for worker_id, child in zip(
            getattr(worker, "_next_level_worker_ids", ()),
            getattr(worker, "_next_level_workers", ()),
        )
    }
    child = children.get(first_hop_child_id)
    if child is None:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
            f"first-hop child {first_hop_child_id} is not in the current Worker tree",
        )
    child_root = f"L{int(child.level)}"
    remainder = [child_root]
    for segment in parsed.segments[2:]:
        if segment.index is None:
            raise MaterializationRefusal(
                RefusalReason.UNSUPPORTED_MEMBER_SHAPE,
                "child provider path segments must include an index",
            )
        remainder.append(f"L{int(segment.level)}[{int(segment.index)}]")
    _ignored_child_hop, provider_device_id = _resolve_delegated_provider_route(child, "/".join(remainder))
    return first_hop_child_id, provider_device_id


def _deployment_kind(record: EndpointRecord) -> EndpointDeploymentKind:
    mapping = {
        HOST_CPU: EndpointDeploymentKind.HOST_CPU,
        DEVICE_AICPU: EndpointDeploymentKind.DEVICE_AICPU,
    }
    try:
        return mapping[record.deployment]
    except KeyError as exc:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_PROVIDER_DEPLOYMENT,
            f"unsupported endpoint deployment {record.deployment.value}",
        ) from exc


def _consumer_adapter(part: RegionPartPlan, consumer: EndpointRecord) -> tuple[AdapterKind, AdapterProfile]:
    matches = [
        attachment
        for attachment in part.attachments
        if attachment.member == consumer.identity and attachment.role is AttachmentRole.CONSUMER
    ]
    if len(matches) != 1 or matches[0].adapter_kind is None or matches[0].adapter_profile is None:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_ATTACHMENT,
            "single-owner first shape requires a consumer adapter on each part",
        )
    return matches[0].adapter_kind, matches[0].adapter_profile


def _delegated_allocate_request(
    shape: DelegatedSingleOwnerRegionShape,
    plan: BackendPlan,
    layout: RegionLayoutSpec,
    session_instance_id: bytes,
    transaction_id: int,
) -> DelegatedAllocateRequest:
    payload_kind, payload_profile = _consumer_adapter(plan.payload, shape.consumer)
    counter_kind, counter_profile = _consumer_adapter(plan.counter, shape.consumer)
    return DelegatedAllocateRequest(
        session_instance_id=session_instance_id,
        transaction_id=transaction_id,
        initiator_path=shape.initiator_path,
        provider_path=shape.provider_path,
        payload_logical_bytes=int(layout.payload_bytes),
        counter_logical_bytes=int(layout.counter_bytes),
        topology=RegionTopologyKind.SINGLE_OWNER,
        initiator_deployment=_deployment_kind(shape.consumer),
        provider_deployment=_deployment_kind(shape.provider),
        payload_backend_kind=plan.payload.backend_kind,
        counter_backend_kind=plan.counter.backend_kind,
        payload_consumer_adapter_kind=payload_kind,
        payload_consumer_adapter_profile=payload_profile,
        counter_consumer_adapter_kind=counter_kind,
        counter_consumer_adapter_profile=counter_profile,
    )


def materialize_region_instance(ctx: MaterializationContext) -> RegionInstance:
    shape = validate_single_owner_region_shape(ctx)
    spec = project_region_allocation_spec(ctx.plan, ctx.layout)
    instance = RegionInstance.planned(ctx, shape)
    resources = getattr(ctx.worker, "_building_run_resources", None)
    ctx.worker._region_instance_registry.track(instance, resources)
    if resources is not None:
        resources.requires_ordered_cleanup = True
    dispatcher = getattr(ctx.worker, "_dispatch_delegated_allocate", None)
    if not callable(dispatcher):
        ctx.worker._region_instance_registry._settle(instance)
        raise MaterializationError("delegated allocate requires Worker dispatch")
    try:
        with ctx.worker._region_instance_registry._delegated_allocate_dispatch(
            registry=ctx.registry,
            expected_registry_epoch=int(ctx.registry.registry_epoch),
        ) as (session_instance_id, transaction_id):
            instance._bind_delegated_identity(session_instance_id, transaction_id, shape.provider_path)
            if not isinstance(ctx.plan, BackendPlan):
                raise MaterializationRefusal(
                    RefusalReason.UNSUPPORTED_PLAN, "materialized region requires a BackendPlan"
                )
            request = _delegated_allocate_request(shape, ctx.plan, ctx.layout, session_instance_id, transaction_id)
            staged = encode_request(request, staged_capacity=max(ALLOCATE_REQUEST_HARD_CEILING, ALLOCATE_REPLY_BYTES))
            raw_reply = dispatcher(memoryview(staged))
            if isinstance(raw_reply, (bytes, bytearray, memoryview)):
                reply_payload = raw_reply
            else:
                reply_payload = staged
            outcome = parse_reply(reply_payload).decode_outcome()
        if isinstance(outcome, DelegatedAllocateReply) and outcome.tag is DelegatedAllocateReplyTag.ALLOCATED:
            try:
                if outcome.result is None or outcome.payload_view is None or outcome.counter_view is None:
                    raise MaterializationError("delegated ALLOCATED reply is missing result or local views")
                validate_committed_region_allocation(
                    ctx.plan,
                    spec,
                    outcome.result,
                    outcome.payload_view,
                    outcome.counter_view,
                    expected_capability_type=ctx.worker._provider_import_capability_type(),
                    expected_device_id=int(shape.provider_device_id),
                )
            except BaseException as exc:
                ctx.worker._latch_delegated_session_fatal(exc)
                raise
            instance._commit_delegated_allocation(int(outcome.result.provider_resource_id))
            payload_lease = ctx.worker._import_region_part_lease(
                instance.worker_id, instance._provider_resource_id, outcome.result.export_descriptor.payload
            )
            instance._payload_mapping = payload_lease
            counter_lease = ctx.worker._import_region_part_lease(
                instance.worker_id, instance._provider_resource_id, outcome.result.export_descriptor.counter
            )
            instance._counter_mapping = counter_lease
            instance._payload_part = PayloadPart(
                RegionPartSpan(offset=0, nbytes=int(spec.payload.logical_bytes)),
                _select_host_vmm_copy_access(ctx.plan.payload, instance.provider, instance.consumer, payload_lease),
            )
            instance._counter_part = CounterPart(
                RegionPartSpan(offset=0, nbytes=int(spec.counter.logical_bytes)),
                _select_host_vmm_copy_access(ctx.plan.counter, instance.provider, instance.consumer, counter_lease),
            )
            instance._payload_local_view = outcome.payload_view
            instance._counter_local_view = outcome.counter_view
            instance._ever_live = True
            instance._state = RegionInstanceState.LIVE
            return instance
        if isinstance(outcome, DelegatedAllocateReply) and outcome.tag is DelegatedAllocateReplyTag.ERROR:
            if outcome.error is None:
                missing = RegionControlError(
                    RegionControlErrorKind.INTERNAL_INVARIANT,
                    "delegated allocate ERROR is missing its typed error",
                )
                ctx.worker._latch_delegated_session_fatal(missing)
                raise missing
            raise outcome.error
        unexpected = RegionControlError(
            RegionControlErrorKind.INTERNAL_INVARIANT,
            "delegated allocate returned an unexpected reply tag",
        )
        ctx.worker._latch_delegated_session_fatal(unexpected)
        raise unexpected
    except BaseException as exc:
        if instance._state is None:
            instance._abort_materialization(exc)
        ctx.worker._region_instance_registry._settle(instance)
        raise


def validate_committed_region_allocation(  # noqa: PLR0912
    plan: BackendPlan,
    spec: RegionAllocationSpec,
    result: RegionAllocationResult,
    payload_view: RegionPartLocalView,
    counter_view: RegionPartLocalView,
    *,
    expected_capability_type: type,
    expected_device_id: int | None,
) -> None:
    payload_export = result.export_descriptor.payload
    counter_export = result.export_descriptor.counter
    if payload_export.planned_backing_kind is not plan.payload.backend_kind:
        raise RuntimeError("committed PAYLOAD planned backing does not match the admitted plan")
    if counter_export.planned_backing_kind is not plan.counter.backend_kind:
        raise RuntimeError("committed COUNTER planned backing does not match the admitted plan")
    if payload_export.planned_backing_kind is not spec.payload.planned_backing_kind:
        raise RuntimeError("committed PAYLOAD planned backing does not match the admitted spec")
    if counter_export.planned_backing_kind is not spec.counter.planned_backing_kind:
        raise RuntimeError("committed COUNTER planned backing does not match the admitted spec")
    if int(payload_export.logical_bytes) != int(spec.payload.logical_bytes):
        raise RuntimeError("committed PAYLOAD logical_bytes do not match the admitted spec")
    if int(counter_export.logical_bytes) != int(spec.counter.logical_bytes):
        raise RuntimeError("committed COUNTER logical_bytes do not match the admitted spec")
    if not isinstance(payload_export.import_capability, expected_capability_type) or not isinstance(
        counter_export.import_capability, expected_capability_type
    ):
        raise RuntimeError("committed import capability does not match the current execution environment")
    if expected_capability_type is VmmShareableHandleImport:
        payload_cap = payload_export.import_capability
        counter_cap = counter_export.import_capability
        assert isinstance(payload_cap, VmmShareableHandleImport)
        assert isinstance(counter_cap, VmmShareableHandleImport)
        if int(payload_cap.shareable_handle) == int(counter_cap.shareable_handle):
            raise RuntimeError("committed VMM shareable handles must be distinct")
        if expected_device_id is not None and (
            int(payload_cap.device_id) != int(expected_device_id)
            or int(counter_cap.device_id) != int(expected_device_id)
        ):
            raise RuntimeError("committed VMM device_id is outside this worker's device namespace")
    elif expected_capability_type is PosixShmImport:
        payload_cap = payload_export.import_capability
        counter_cap = counter_export.import_capability
        assert isinstance(payload_cap, PosixShmImport)
        assert isinstance(counter_cap, PosixShmImport)
        if payload_cap.shm_name == counter_cap.shm_name:
            raise RuntimeError("committed POSIX shm tokens must be distinct")
    if payload_view.part is not RegionPartKind.PAYLOAD or counter_view.part is not RegionPartKind.COUNTER:
        raise RuntimeError("committed local views must be PAYLOAD then COUNTER")
    if int(payload_view.logical_bytes) != int(spec.payload.logical_bytes):
        raise RuntimeError("committed PAYLOAD local view bytes do not match the admitted spec")
    if int(counter_view.logical_bytes) != int(spec.counter.logical_bytes):
        raise RuntimeError("committed COUNTER local view bytes do not match the admitted spec")
    try:
        validate_independent_local_views(payload_view, counter_view)
    except ValueError as exc:
        raise RuntimeError(str(exc) or "committed local views are not independent") from exc


def _record_for(ctx: MaterializationContext, endpoint: Any) -> EndpointRecord:
    try:
        return ctx.registry.record_for(endpoint)
    except ValueError as exc:
        raise MaterializationRefusal(RefusalReason.REGISTRY_MISMATCH, str(exc)) from exc


def _select_host_vmm_copy_access(
    part: RegionPartPlan,
    provider: EndpointRecord,
    consumer: EndpointRecord,
    mapping: Any,
) -> HostVmmCopyAccess:
    _validate_part(part, provider, consumer)
    return HostVmmCopyAccess.from_mapping(mapping)


def _validate_registry_matches_worker(ctx: MaterializationContext) -> None:
    worker_instance_id = getattr(ctx.worker, "_owner_instance_id", None)
    worker_epoch = getattr(ctx.worker, "_endpoint_registry_epoch", None)
    if worker_instance_id != ctx.registry.session_instance_id or worker_epoch != ctx.registry.registry_epoch:
        raise MaterializationRefusal(
            RefusalReason.REGISTRY_MISMATCH,
            "Region materialization requires a registry from the current worker endpoint epoch",
        )


def _validate_part(part: RegionPartPlan, provider: EndpointRecord, consumer: EndpointRecord) -> None:
    if part.backend_kind is not BackendKind.VMM_WINDOW:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_BACKEND_KIND,
            "Only VMM_WINDOW-backed worker-chip region parts are supported",
        )
    attachments = {attachment.member: attachment for attachment in part.attachments}
    if len(attachments) != len(part.attachments) or set(attachments) != {provider.identity, consumer.identity}:
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_ATTACHMENT,
            "Part attachments must match exactly the provider and host consumer",
        )
    _validate_provider_attachment(attachments[provider.identity])
    _validate_consumer_attachment(attachments[consumer.identity])


def _validate_provider_attachment(attachment: MemberAttachmentPlan) -> None:
    if (
        attachment.role is not AttachmentRole.PROVIDER
        or attachment.adapter_kind is not None
        or attachment.adapter_profile is not None
    ):
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_ATTACHMENT,
            "Provider attachment must be a bare PROVIDER attachment",
        )


def _validate_consumer_attachment(attachment: MemberAttachmentPlan) -> None:
    if (
        attachment.role is not AttachmentRole.CONSUMER
        or attachment.adapter_kind is not AdapterKind.OWNER_DELEGATED_COPY
        or attachment.adapter_profile is not AdapterProfile.HOST_VMM_COPY
    ):
        raise MaterializationRefusal(
            RefusalReason.UNSUPPORTED_ATTACHMENT,
            "Host consumer attachment must use OWNER_DELEGATED_COPY/HOST_VMM_COPY",
        )
