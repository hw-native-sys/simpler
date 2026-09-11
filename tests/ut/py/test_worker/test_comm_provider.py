# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Unit tests for neutral comm-provider values and typed errors."""

from __future__ import annotations

import ast
import dataclasses
import logging
import sys
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import pytest
import simpler.comm_provider as comm_provider_module
from simpler.buffer import (
    AccessMode,
    AddressSpace,
    BackendKind,
    Buffer,
    BufferDescriptor,
    CanonicalIdentity,
    intern_worker_path,
)
from simpler.comm_provider import (
    POSIX_SHM_TOKEN_MAX_BYTES,
    DeviceAllocationTarget,
    HostAllocationTarget,
    ImportCapability,
    PosixShmImport,
    ProviderCleanupFailure,
    ProviderPartResourceState,
    ProviderRegionResourceState,
    ProviderRegionStore,
    ProviderRegionStoreState,
    ProviderReleaseResult,
    ProviderReleaseStatus,
    RegionAllocationContext,
    RegionAllocationError,
    RegionAllocationResult,
    RegionAllocationSpec,
    RegionCleanupCause,
    RegionControlError,
    RegionControlErrorKind,
    RegionEnvironmentKind,
    RegionExportDescriptor,
    RegionOperationKind,
    RegionPartAllocation,
    RegionPartAllocationSpec,
    RegionPartKind,
    RegionPartLocalView,
    SimPosixShmAllocation,
    VmmAllocation,
    VmmShareableHandleImport,
    validate_independent_local_views,
)

_COMM_PROVIDER_PATH = Path(comm_provider_module.__file__).resolve()
_UINT64_MAX = (1 << 64) - 1
_INT32_MAX = (1 << 31) - 1


def _payload_spec(logical_bytes: int = 64, backing=BackendKind.VMM_WINDOW) -> RegionPartAllocationSpec:
    return RegionPartAllocationSpec(planned_backing_kind=backing, logical_bytes=logical_bytes)


def _counter_spec(logical_bytes: int = 8, backing=BackendKind.VMM_WINDOW) -> RegionPartAllocationSpec:
    return RegionPartAllocationSpec(planned_backing_kind=backing, logical_bytes=logical_bytes)


def _posix_descriptor(*, owner_nonce: bytes, buffer_id: int, nbytes: int, token: str) -> BufferDescriptor:
    return BufferDescriptor(
        CanonicalIdentity(bytes(owner_nonce), int(buffer_id), 1),
        AddressSpace.HOST,
        AccessMode.READWRITE,
        BackendKind.POSIX_SHM,
        int(nbytes),
        token.encode("ascii"),
    )


def test_comm_provider_module_does_not_import_worker_chip_or_w5a():
    tree = ast.parse(_COMM_PROVIDER_PATH.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden_substrings = (
        "worker_chip",
        "worker.py",
        "mailbox",
        "w5a",
        "orchestrator",
        "comm_region",
    )
    for name in imported:
        lowered = name.replace("_", "").lower()
        assert "workerchip" not in lowered
        assert not any(token in name for token in forbidden_substrings)
        assert name not in {"simpler.worker", "worker"}


@pytest.mark.parametrize(
    ("enum_cls", "name", "value"),
    [
        (RegionPartKind, "INVALID", 0),
        (RegionPartKind, "PAYLOAD", 1),
        (RegionPartKind, "COUNTER", 2),
        (RegionControlErrorKind, "NONE", 0),
        (RegionControlErrorKind, "BAD_MAGIC_VERSION", 1),
        (RegionControlErrorKind, "BAD_MESSAGE_SIZE", 2),
        (RegionControlErrorKind, "INVALID_ENUM_VALUE", 3),
        (RegionControlErrorKind, "RESERVED_NONZERO", 4),
        (RegionControlErrorKind, "INVALID_FIELD_VALUE", 5),
        (RegionControlErrorKind, "STORE_LIFECYCLE", 6),
        (RegionControlErrorKind, "INTERNAL_INVARIANT", 7),
        (RegionControlErrorKind, "BACKEND_FAILURE", 8),
        (RegionOperationKind, "NONE", 0),
        (RegionOperationKind, "MATERIALIZE", 1),
        (RegionOperationKind, "ZERO_BYTES", 2),
        (RegionOperationKind, "DESCRIBE", 3),
        (RegionOperationKind, "LOCAL_VIEW", 4),
        (RegionOperationKind, "RELEASE", 5),
        (RegionCleanupCause, "NONE", 0),
        (RegionCleanupCause, "BACKEND_ERROR", 1),
        (RegionCleanupCause, "INTERRUPTED", 2),
        (RegionCleanupCause, "BACKEND_STATE_MISMATCH", 3),
        (ProviderReleaseStatus, "RELEASED", 1),
        (ProviderReleaseStatus, "ALREADY_GONE", 2),
        (ProviderReleaseStatus, "UNKNOWN_RESOURCE", 3),
        (ProviderReleaseStatus, "CLEANUP_INCOMPLETE", 4),
    ],
)
def test_typed_code_values_match_design_tables(enum_cls, name, value):
    member = enum_cls[name]
    assert int(member) == value
    assert enum_cls(value) is member


@pytest.mark.parametrize(
    "enum_cls",
    [
        RegionPartKind,
        RegionControlErrorKind,
        RegionOperationKind,
        RegionCleanupCause,
        ProviderReleaseStatus,
        BackendKind,
    ],
)
def test_unknown_nonzero_enum_values_are_rejected(enum_cls):
    with pytest.raises(ValueError):
        enum_cls(99)


def test_state_enum_names_are_exactly_the_accepted_set():
    assert [state.name for state in ProviderRegionStoreState] == [
        "OPEN",
        "CLOSING",
        "CLOSE_FAILED",
        "CLOSED",
    ]
    assert [state.name for state in ProviderRegionResourceState] == [
        "CREATING",
        "ACTIVE",
        "CLEANUP_PENDING",
    ]
    assert [state.name for state in ProviderPartResourceState] == [
        "SHELL",
        "MATERIALIZING",
        "READY",
        "CLEANUP_PENDING",
        "RELEASED",
    ]


def test_allocation_spec_accepts_independent_part_sizes_and_backing_kinds():
    spec = RegionAllocationSpec(
        payload=_payload_spec(96, BackendKind.VMM_WINDOW),
        counter=_counter_spec(12, BackendKind.POSIX_SHM),
    )
    assert spec.part(RegionPartKind.PAYLOAD).logical_bytes == 96
    assert spec.part(RegionPartKind.COUNTER).logical_bytes == 12
    assert spec.payload.planned_backing_kind is BackendKind.VMM_WINDOW
    assert spec.counter.planned_backing_kind is BackendKind.POSIX_SHM


@pytest.mark.parametrize("logical_bytes", [0, -1, _UINT64_MAX + 1, True])
def test_part_spec_rejects_non_positive_or_overflow_sizes(logical_bytes):
    with pytest.raises((ValueError, TypeError)):
        RegionPartAllocationSpec(planned_backing_kind=BackendKind.VMM_WINDOW, logical_bytes=logical_bytes)


@pytest.mark.parametrize("logical_bytes", [1, 2, 3, 5, 6, 7, 9])
def test_counter_spec_requires_multiple_of_four(logical_bytes):
    payload = _payload_spec()
    counter = RegionPartAllocationSpec(planned_backing_kind=BackendKind.VMM_WINDOW, logical_bytes=logical_bytes)
    with pytest.raises(ValueError, match="multiple of 4"):
        RegionAllocationSpec(payload=payload, counter=counter)


def test_allocation_context_preserves_typed_targets():
    sim = RegionAllocationContext(
        environment_kind=RegionEnvironmentKind.SIM,
        target=DeviceAllocationTarget(device_id=3),
    )
    reserved = RegionAllocationContext(
        environment_kind=RegionEnvironmentKind.ONBOARD,
        target=HostAllocationTarget(),
    )
    assert sim.target.device_id == 3
    assert isinstance(reserved.target, HostAllocationTarget)
    with pytest.raises(ValueError):
        DeviceAllocationTarget(device_id=_INT32_MAX + 1)


def test_export_descriptor_holds_two_buffer_descriptors():
    nonce = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    payload = _posix_descriptor(owner_nonce=nonce, buffer_id=1, nbytes=64, token="pto_payload_a")
    counter = _posix_descriptor(owner_nonce=nonce, buffer_id=2, nbytes=8, token="pto_counter_a")
    descriptor = RegionExportDescriptor(payload=payload, counter=counter)
    field_names = {field.name for field in dataclasses.fields(RegionExportDescriptor)}
    assert field_names == {"payload", "counter"}
    assert "local_base" not in field_names
    assert "local_addr" not in field_names
    assert descriptor.payload.identity != descriptor.counter.identity
    result = RegionAllocationResult(provider_resource_id=1, export_descriptor=descriptor)
    assert result.provider_resource_id == 1
    with pytest.raises(ValueError):
        RegionAllocationResult(provider_resource_id=0, export_descriptor=descriptor)


@pytest.mark.parametrize("shm_name", ["", "a" * (POSIX_SHM_TOKEN_MAX_BYTES + 1), "npu\u4e00", "name\x00x"])
def test_posix_shm_token_rejects_empty_overlong_non_ascii_and_nul(shm_name):
    with pytest.raises((ValueError, TypeError)):
        PosixShmImport(shm_name=shm_name)


def test_posix_shm_token_accepts_bounded_ascii():
    token = "p" * POSIX_SHM_TOKEN_MAX_BYTES
    capability = PosixShmImport(shm_name=token)
    assert capability.shm_name == token


def test_local_view_accepts_base_zero_and_rejects_uint64_span_overflow():
    view = RegionPartLocalView(part=RegionPartKind.PAYLOAD, local_base=0, logical_bytes=8)
    assert view.local_base == 0
    with pytest.raises(ValueError, match="overflowed uint64"):
        RegionPartLocalView(part=RegionPartKind.PAYLOAD, local_base=_UINT64_MAX, logical_bytes=1)
    with pytest.raises(ValueError, match="64-byte aligned"):
        RegionPartLocalView(part=RegionPartKind.COUNTER, local_base=32, logical_bytes=8)


def test_independent_span_rules_allow_noncontiguous_nonoverlapping_bases():
    payload = RegionPartLocalView(part=RegionPartKind.PAYLOAD, local_base=0x1000, logical_bytes=100)
    counter = RegionPartLocalView(part=RegionPartKind.COUNTER, local_base=0x2000, logical_bytes=8)
    validate_independent_local_views(payload, counter)


def test_independent_span_rules_reject_overlap_and_swapped_parts():
    payload = RegionPartLocalView(part=RegionPartKind.PAYLOAD, local_base=0, logical_bytes=128)
    counter = RegionPartLocalView(part=RegionPartKind.COUNTER, local_base=64, logical_bytes=8)
    with pytest.raises(ValueError, match="must not overlap"):
        validate_independent_local_views(payload, counter)
    with pytest.raises(ValueError, match="part PAYLOAD"):
        validate_independent_local_views(counter, payload)


def test_adjacent_spans_do_not_overlap():
    payload = RegionPartLocalView(part=RegionPartKind.PAYLOAD, local_base=0, logical_bytes=64)
    counter = RegionPartLocalView(part=RegionPartKind.COUNTER, local_base=64, logical_bytes=8)
    validate_independent_local_views(payload, counter)


def test_release_result_allows_at_most_one_failure_per_part():
    payload_failure = ProviderCleanupFailure(
        part=RegionPartKind.PAYLOAD,
        backend_operation=RegionOperationKind.RELEASE,
        typed_cause=RegionCleanupCause.BACKEND_ERROR,
    )
    counter_failure = ProviderCleanupFailure(
        part=RegionPartKind.COUNTER,
        backend_operation=RegionOperationKind.RELEASE,
        typed_cause=RegionCleanupCause.INTERRUPTED,
    )
    result = ProviderReleaseResult(
        provider_resource_id=9,
        status=ProviderReleaseStatus.CLEANUP_INCOMPLETE,
        failures=(payload_failure, counter_failure),
    )
    assert len(result.failures) == 2
    with pytest.raises(ValueError, match="at most one failure per part"):
        ProviderReleaseResult(
            provider_resource_id=9,
            status=ProviderReleaseStatus.CLEANUP_INCOMPLETE,
            failures=(payload_failure, payload_failure),
        )
    with pytest.raises(ValueError):
        ProviderReleaseResult(
            provider_resource_id=9,
            status=ProviderReleaseStatus.RELEASED,
            failures=(payload_failure,),
        )


def test_allocation_error_carries_debt_bit_and_no_cleanup_cause():
    error = RegionAllocationError(
        provisional_resource_id=4,
        control_kind=RegionControlErrorKind.BACKEND_FAILURE,
        failed_part=RegionPartKind.COUNTER,
        failed_operation=RegionOperationKind.ZERO_BYTES,
        cleanup_debt_remaining=True,
    )
    assert error.provisional_resource_id == 4
    assert error.cleanup_debt_remaining is True
    assert error.failed_part is RegionPartKind.COUNTER
    assert error.failed_operation is RegionOperationKind.ZERO_BYTES
    assert not hasattr(error, "typed_cause")
    assert not hasattr(error, "cleanup_cause")
    with pytest.raises(ValueError):
        RegionAllocationError(
            provisional_resource_id=0,
            control_kind=RegionControlErrorKind.BACKEND_FAILURE,
            failed_part=RegionPartKind.PAYLOAD,
            failed_operation=RegionOperationKind.MATERIALIZE,
            cleanup_debt_remaining=False,
        )
    with pytest.raises(ValueError):
        RegionAllocationError(
            provisional_resource_id=1,
            control_kind=RegionControlErrorKind.BAD_MESSAGE_SIZE,
            failed_part=RegionPartKind.PAYLOAD,
            failed_operation=RegionOperationKind.MATERIALIZE,
            cleanup_debt_remaining=False,
        )


def test_control_error_rejects_none_kind_and_unknown_values():
    error = RegionControlError(RegionControlErrorKind.STORE_LIFECYCLE, "store closed")
    assert error.kind is RegionControlErrorKind.STORE_LIFECYCLE
    assert error.failed_part is RegionPartKind.INVALID
    with pytest.raises(ValueError):
        RegionControlError(RegionControlErrorKind.NONE)
    with pytest.raises(ValueError):
        RegionControlError(99)


class FakeRegionPartAllocation:
    """Deterministic RegionPartAllocation shell for store tests."""

    def __init__(
        self,
        part: RegionPartKind,
        spec: RegionPartAllocationSpec,
        *,
        world: FakeShellWorld,
        local_base: int,
        shm_name: str,
        mapping_bytes: int | None = None,
    ) -> None:
        self.part = part
        self.spec = spec
        self.world = world
        self.calls: list[str] = []
        self.local_ledger: list[object] = []
        self.materialized = False
        self.release_count = 0
        self.zero_calls: list[tuple[int, int]] = []
        self.fail_materialize: BaseException | type[BaseException] | None = None
        self.fail_zero: BaseException | type[BaseException] | None = None
        self.fail_mapping_bytes: BaseException | type[BaseException] | None = None
        self.fail_import_capability: BaseException | type[BaseException] | None = None
        self.fail_local_base: BaseException | type[BaseException] | None = None
        self.raise_on_release: BaseException | type[BaseException] | None = None
        self.release_step_failures: list[ProviderCleanupFailure] = []
        self._local_base = local_base
        self._mapping_bytes = spec.logical_bytes if mapping_bytes is None else mapping_bytes
        self._import_capability: ImportCapability = PosixShmImport(shm_name=shm_name)
        self._buffer: Buffer | None = None

    def _raise_configured(self, spec: BaseException | type[BaseException] | None) -> None:
        if spec is None:
            return
        if isinstance(spec, BaseException):
            raise spec
        raise spec()

    def materialize(self, identity, diagnostics=None) -> Buffer:
        self.calls.append("materialize")
        self.world.record(self.part, "materialize")
        if self.release_count != 0:
            self.world.materialize_called_release = True
        self.world.constructed_at_first_materialize.setdefault(self.part, tuple(self.world.constructed_parts))
        self._raise_configured(self.fail_materialize)
        self.materialized = True
        self.world.side_effects.append((self.part, "materialize"))
        owner_worker_path = ""
        if diagnostics is not None:
            owner_worker_path = str(getattr(diagnostics, "owner_worker_path", "") or "")
        token = self._import_capability.shm_name.lstrip("/")
        buffer = Buffer(
            identity=identity,
            owner_worker_path_id=intern_worker_path(owner_worker_path),
            address_space=AddressSpace.HOST,
            access=AccessMode.READWRITE,
            backend_kind=BackendKind.POSIX_SHM,
            nbytes=int(self.spec.logical_bytes),
            body=token.encode("ascii"),
            shm=None,
            base=int(self._local_base),
        )
        self._buffer = buffer
        return buffer

    def mapping_bytes(self) -> int:
        self.calls.append("mapping_bytes")
        self.world.record(self.part, "mapping_bytes")
        if not self.materialized:
            raise RegionControlError(
                RegionControlErrorKind.INTERNAL_INVARIANT,
                "mapping_bytes requires a materialized shell",
                failed_part=self.part,
                failed_operation=RegionOperationKind.DESCRIBE,
            )
        self._raise_configured(self.fail_mapping_bytes)
        return self._mapping_bytes

    def import_capability(self) -> ImportCapability:
        self.calls.append("import_capability")
        self.world.record(self.part, "import_capability")
        if not self.materialized:
            raise RegionControlError(
                RegionControlErrorKind.INTERNAL_INVARIANT,
                "import_capability requires a materialized shell",
                failed_part=self.part,
                failed_operation=RegionOperationKind.DESCRIBE,
            )
        self._raise_configured(self.fail_import_capability)
        return self._import_capability

    def local_base(self) -> int:
        self.calls.append("local_base")
        self.world.record(self.part, "local_base")
        if not self.materialized:
            raise RegionControlError(
                RegionControlErrorKind.INTERNAL_INVARIANT,
                "local_base requires a materialized shell",
                failed_part=self.part,
                failed_operation=RegionOperationKind.LOCAL_VIEW,
            )
        self._raise_configured(self.fail_local_base)
        return self._local_base

    def zero_bytes(self, offset: int, nbytes: int) -> None:
        self.calls.append("zero_bytes")
        self.world.record(self.part, "zero_bytes")
        self.zero_calls.append((offset, nbytes))
        if not self.materialized:
            raise RegionControlError(
                RegionControlErrorKind.INTERNAL_INVARIANT,
                "zero_bytes requires a materialized shell",
                failed_part=self.part,
                failed_operation=RegionOperationKind.ZERO_BYTES,
            )
        self._raise_configured(self.fail_zero)

    def release_once(self) -> ProviderCleanupFailure | None:
        self.calls.append("release_once")
        self.world.record(self.part, "release_once")
        self.release_count += 1
        if self.release_count > 1:
            self.world.duplicate_releases.append(self.part)
        self._raise_configured(self.raise_on_release)
        for failure in self.release_step_failures:
            self.local_ledger.append(failure)
        if self.release_step_failures:
            return self.release_step_failures[0]
        return None


class FakeShellWorld:
    def __init__(self) -> None:
        self.calls: list[tuple[RegionPartKind, str]] = []
        self.constructed_parts: list[RegionPartKind] = []
        self.side_effects: list[tuple[RegionPartKind, str]] = []
        self.constructed_at_first_materialize: dict[RegionPartKind, tuple[RegionPartKind, ...]] = {}
        self.duplicate_releases: list[RegionPartKind] = []
        self.materialize_called_release = False

    def record(self, part: RegionPartKind, name: str) -> None:
        self.calls.append((part, name))


class FakeShellFactory:
    def __init__(self) -> None:
        self.world = FakeShellWorld()
        self.payloads: list[FakeRegionPartAllocation] = []
        self.counters: list[FakeRegionPartAllocation] = []
        self.fail_construct: dict[RegionPartKind, BaseException | type[BaseException]] = {}
        self._seq = 0

    def __call__(
        self,
        context: RegionAllocationContext,
        part: RegionPartKind,
        spec: RegionPartAllocationSpec,
    ) -> RegionPartAllocation:
        del context
        self.world.record(part, "construct")
        if part in self.fail_construct:
            spec_or_exc = self.fail_construct[part]
            if isinstance(spec_or_exc, BaseException):
                raise spec_or_exc
            raise spec_or_exc()
        self._seq += 1
        local_base = 0x1000 if part is RegionPartKind.PAYLOAD else 0x2000
        shell = FakeRegionPartAllocation(
            part,
            spec,
            world=self.world,
            local_base=local_base,
            shm_name=f"pto{part.name[0].lower()}{self._seq}",
        )
        self.world.constructed_parts.append(part)
        if part is RegionPartKind.PAYLOAD:
            self.payloads.append(shell)
        else:
            self.counters.append(shell)
        return shell


def _sim_context() -> RegionAllocationContext:
    return RegionAllocationContext(
        environment_kind=RegionEnvironmentKind.SIM,
        target=DeviceAllocationTarget(device_id=0),
    )


def _allocation_spec() -> RegionAllocationSpec:
    return RegionAllocationSpec(payload=_payload_spec(), counter=_counter_spec())


def _open_store(factory: FakeShellFactory | None = None) -> tuple[ProviderRegionStore, FakeShellFactory]:
    factory = FakeShellFactory() if factory is None else factory
    store = ProviderRegionStore(_sim_context(), _shell_factory=factory)
    return store, factory


def _mutating_shell_factory(factory: FakeShellFactory, mutate):
    def _factory(context, kind, spec):
        shell = FakeShellFactory.__call__(factory, context, kind, spec)
        mutate(shell, kind)
        return shell

    return _factory


def _backend_failure(part: RegionPartKind) -> ProviderCleanupFailure:
    return ProviderCleanupFailure(
        part=part,
        backend_operation=RegionOperationKind.RELEASE,
        typed_cause=RegionCleanupCause.BACKEND_ERROR,
    )


def test_empty_sweep_closes_an_open_store_and_is_idempotent():
    store, _factory = _open_store()
    assert store.state is ProviderRegionStoreState.OPEN
    assert store.sweep() == ()
    assert store.state is ProviderRegionStoreState.CLOSED
    assert store.sweep() == ()
    assert store.state is ProviderRegionStoreState.CLOSED
    with pytest.raises(RegionControlError) as exc_info:
        store.allocate_and_export(_allocation_spec())
    assert exc_info.value.kind is RegionControlErrorKind.STORE_LIFECYCLE
    assert not hasattr(store, "retry_cleanup")


def test_allocate_and_export_installs_both_shells_before_materialize():
    store, factory = _open_store()
    result = store.allocate_and_export(_allocation_spec())
    assert result.provider_resource_id == 1
    assert factory.world.constructed_at_first_materialize[RegionPartKind.PAYLOAD] == (
        RegionPartKind.PAYLOAD,
        RegionPartKind.COUNTER,
    )
    assert factory.world.calls[:4] == [
        (RegionPartKind.PAYLOAD, "construct"),
        (RegionPartKind.COUNTER, "construct"),
        (RegionPartKind.PAYLOAD, "materialize"),
        (RegionPartKind.COUNTER, "materialize"),
    ]
    assert factory.payloads[0].zero_calls == []
    assert factory.counters[0].zero_calls == [(0, 8)]
    assert factory.world.materialize_called_release is False
    payload = store.local_view(1, RegionPartKind.PAYLOAD)
    counter = store.local_view(1, RegionPartKind.COUNTER)
    validate_independent_local_views(payload, counter)
    descriptor = store.describe(1)
    assert isinstance(descriptor.payload.import_capability, PosixShmImport)
    assert isinstance(descriptor.counter.import_capability, PosixShmImport)
    assert descriptor.payload.import_capability.shm_name != descriptor.counter.import_capability.shm_name


def test_ids_are_monotonic_nonzero_and_never_reused_after_burned_create_failure():
    store, factory = _open_store()
    first = store.allocate_and_export(_allocation_spec())
    assert first.provider_resource_id == 1
    factory.fail_construct[RegionPartKind.COUNTER] = RuntimeError("construct")
    with pytest.raises(RegionAllocationError) as exc_info:
        store.allocate_and_export(_allocation_spec())
    error = exc_info.value
    assert error.provisional_resource_id == 2
    assert error.cleanup_debt_remaining is False
    assert error.control_kind is RegionControlErrorKind.INTERNAL_INVARIANT
    assert error.failed_part is RegionPartKind.COUNTER
    assert factory.payloads[-1].release_count == 1
    assert factory.payloads[-1].materialized is False
    assert store.state is ProviderRegionStoreState.OPEN
    factory.fail_construct.clear()
    third = store.allocate_and_export(_allocation_spec())
    assert third.provider_resource_id == 3
    released = store.release(1)
    assert released.status is ProviderReleaseStatus.RELEASED
    gone = store.release(1)
    assert gone.status is ProviderReleaseStatus.ALREADY_GONE
    unknown = store.release(99)
    assert unknown.status is ProviderReleaseStatus.UNKNOWN_RESOURCE
    with pytest.raises(RegionControlError) as id_exc:
        store.release(0)
    assert id_exc.value.kind is RegionControlErrorKind.INVALID_FIELD_VALUE


def test_allocate_after_successful_release_uses_a_new_id_and_cleans_once():
    store, factory = _open_store()
    first = store.allocate_and_export(_allocation_spec())
    assert store.release(first.provider_resource_id).status is ProviderReleaseStatus.RELEASED
    assert store.state is ProviderRegionStoreState.OPEN
    assert factory.payloads[0].release_count == 1
    assert factory.counters[0].release_count == 1
    second = store.allocate_and_export(_allocation_spec())
    assert second.provider_resource_id == 2
    assert store.release(second.provider_resource_id).status is ProviderReleaseStatus.RELEASED
    assert factory.payloads[1].release_count == 1
    assert factory.counters[1].release_count == 1
    assert factory.world.duplicate_releases == []
    assert store.state is ProviderRegionStoreState.OPEN


def test_release_attempts_both_parts_and_enters_close_failed_on_first_cleanup_debt():
    store, factory = _open_store()
    store.allocate_and_export(_allocation_spec())
    factory.payloads[0].release_step_failures = [_backend_failure(RegionPartKind.PAYLOAD)]
    result = store.release(1)
    assert result.status is ProviderReleaseStatus.CLEANUP_INCOMPLETE
    assert result.failures[0].part is RegionPartKind.PAYLOAD
    assert factory.payloads[0].release_count == 1
    assert factory.counters[0].release_count == 1
    assert store.state is ProviderRegionStoreState.CLOSE_FAILED
    with pytest.raises(RegionControlError) as exc_info:
        store.describe(1)
    assert exc_info.value.kind is RegionControlErrorKind.STORE_LIFECYCLE
    with pytest.raises(RegionControlError):
        store.local_view(1, RegionPartKind.PAYLOAD)
    with pytest.raises(RegionControlError):
        store.release(1)
    with pytest.raises(RegionControlError):
        store.allocate_and_export(_allocation_spec())


def test_sweep_first_cleans_untouched_resources_and_does_not_reissue_failed_cleanup():
    store, factory = _open_store()
    store.allocate_and_export(_allocation_spec())
    store.allocate_and_export(_allocation_spec())
    factory.payloads[0].release_step_failures = [_backend_failure(RegionPartKind.PAYLOAD)]
    first = store.release(1)
    assert first.status is ProviderReleaseStatus.CLEANUP_INCOMPLETE
    assert factory.payloads[1].release_count == 0
    results = store.sweep()
    assert [item.provider_resource_id for item in results] == [1, 2]
    assert results[0].status is ProviderReleaseStatus.CLEANUP_INCOMPLETE
    assert results[1].status is ProviderReleaseStatus.RELEASED
    assert factory.payloads[0].release_count == 1
    assert factory.counters[0].release_count == 1
    assert factory.payloads[1].release_count == 1
    assert factory.counters[1].release_count == 1
    assert store.state is ProviderRegionStoreState.CLOSE_FAILED
    again = store.sweep()
    assert [item.provider_resource_id for item in again] == [1]
    assert again[0].status is ProviderReleaseStatus.CLEANUP_INCOMPLETE
    assert factory.payloads[0].release_count == 1
    assert factory.world.duplicate_releases == []


def test_counter_materialize_failure_cleans_both_installed_shells_and_stays_open():
    factory = FakeShellFactory()

    def _mutate(shell, kind):
        if kind is RegionPartKind.COUNTER:
            shell.fail_materialize = RuntimeError("counter materialize")

    store = ProviderRegionStore(_sim_context(), _shell_factory=_mutating_shell_factory(factory, _mutate))
    with pytest.raises(RegionAllocationError) as exc_info:
        store.allocate_and_export(_allocation_spec())
    error = exc_info.value
    assert error.control_kind is RegionControlErrorKind.BACKEND_FAILURE
    assert error.failed_part is RegionPartKind.COUNTER
    assert error.failed_operation is RegionOperationKind.MATERIALIZE
    assert error.cleanup_debt_remaining is False
    assert factory.payloads[0].materialized is True
    assert factory.counters[0].materialized is False
    assert factory.payloads[0].release_count == 1
    assert factory.counters[0].release_count == 1
    assert store.state is ProviderRegionStoreState.OPEN


@pytest.mark.parametrize(
    ("attr", "exc", "part", "operation"),
    [
        ("fail_zero", RuntimeError("zero"), RegionPartKind.COUNTER, RegionOperationKind.ZERO_BYTES),
        ("fail_mapping_bytes", RuntimeError("map"), RegionPartKind.PAYLOAD, RegionOperationKind.DESCRIBE),
        ("fail_mapping_bytes", RuntimeError("map"), RegionPartKind.COUNTER, RegionOperationKind.DESCRIBE),
        ("fail_import_capability", RuntimeError("import"), RegionPartKind.PAYLOAD, RegionOperationKind.DESCRIBE),
        ("fail_import_capability", RuntimeError("import"), RegionPartKind.COUNTER, RegionOperationKind.DESCRIBE),
        ("fail_local_base", RuntimeError("base"), RegionPartKind.PAYLOAD, RegionOperationKind.LOCAL_VIEW),
        ("fail_local_base", RuntimeError("base"), RegionPartKind.COUNTER, RegionOperationKind.LOCAL_VIEW),
        ("fail_materialize", KeyboardInterrupt(), RegionPartKind.PAYLOAD, RegionOperationKind.MATERIALIZE),
    ],
)
def test_create_path_failures_are_classified_and_cleanup_runs_once(attr, exc, part, operation):
    factory = FakeShellFactory()

    def _mutate(shell, kind):
        if kind is part:
            setattr(shell, attr, exc)

    store = ProviderRegionStore(_sim_context(), _shell_factory=_mutating_shell_factory(factory, _mutate))
    with pytest.raises(RegionAllocationError) as exc_info:
        store.allocate_and_export(_allocation_spec())
    error = exc_info.value
    assert error.failed_part is part
    assert error.failed_operation is operation
    assert error.control_kind is RegionControlErrorKind.BACKEND_FAILURE
    assert error.cleanup_debt_remaining is False
    assert factory.payloads[0].release_count == 1
    assert factory.counters[0].release_count == 1
    assert store.state is ProviderRegionStoreState.OPEN


def test_create_cleanup_failure_retains_record_and_close_failed():
    factory = FakeShellFactory()

    def _mutate(shell, kind):
        if kind is RegionPartKind.COUNTER:
            shell.fail_materialize = RuntimeError("counter materialize")
        if kind is RegionPartKind.PAYLOAD:
            shell.release_step_failures = [_backend_failure(RegionPartKind.PAYLOAD)]

    store = ProviderRegionStore(_sim_context(), _shell_factory=_mutating_shell_factory(factory, _mutate))
    with pytest.raises(RegionAllocationError) as exc_info:
        store.allocate_and_export(_allocation_spec())
    error = exc_info.value
    assert error.cleanup_debt_remaining is True
    assert error.failed_part is RegionPartKind.COUNTER
    assert store.state is ProviderRegionStoreState.CLOSE_FAILED
    assert factory.payloads[0].release_count == 1
    assert factory.counters[0].release_count == 1
    snapshot = store.sweep()
    assert snapshot[0].status is ProviderReleaseStatus.CLEANUP_INCOMPLETE
    assert factory.payloads[0].release_count == 1


def test_fake_release_keeps_later_failures_local_and_summarizes_the_first():
    store, factory = _open_store()
    store.allocate_and_export(_allocation_spec())
    first = _backend_failure(RegionPartKind.PAYLOAD)
    later = ProviderCleanupFailure(
        part=RegionPartKind.PAYLOAD,
        backend_operation=RegionOperationKind.RELEASE,
        typed_cause=RegionCleanupCause.INTERRUPTED,
    )
    factory.payloads[0].release_step_failures = [first, later]
    result = store.release(1)
    assert result.failures == (first,)
    assert factory.payloads[0].local_ledger == [first, later]


def test_overlapping_local_views_are_an_internal_invariant_and_are_cleaned():
    factory = FakeShellFactory()

    def _mutate(shell, _kind):
        shell._local_base = 0

    store = ProviderRegionStore(_sim_context(), _shell_factory=_mutating_shell_factory(factory, _mutate))
    with pytest.raises(RegionAllocationError) as exc_info:
        store.allocate_and_export(_allocation_spec())
    error = exc_info.value
    assert error.control_kind is RegionControlErrorKind.INTERNAL_INVARIANT
    assert error.failed_operation is RegionOperationKind.LOCAL_VIEW
    assert error.cleanup_debt_remaining is False
    assert factory.payloads[0].release_count == 1
    assert factory.counters[0].release_count == 1


def test_closed_dispatcher_is_an_internal_invariant_and_burns_the_id():
    store = ProviderRegionStore(
        RegionAllocationContext(
            environment_kind=RegionEnvironmentKind.SIM,
            target=HostAllocationTarget(),
        )
    )
    with pytest.raises(RegionAllocationError) as exc_info:
        store.allocate_and_export(_allocation_spec())
    error = exc_info.value
    assert error.provisional_resource_id == 1
    assert error.control_kind is RegionControlErrorKind.INTERNAL_INVARIANT
    assert error.cleanup_debt_remaining is False
    assert store.state is ProviderRegionStoreState.OPEN
    with pytest.raises(RegionAllocationError) as second:
        store.allocate_and_export(_allocation_spec())
    assert second.value.provisional_resource_id == 2


def test_id_exhaustion_fails_before_a_side_effect_and_does_not_wrap():
    store, factory = _open_store()
    store._next_provider_resource_id = _UINT64_MAX
    result = store.allocate_and_export(_allocation_spec())
    assert result.provider_resource_id == _UINT64_MAX
    with pytest.raises(RegionControlError) as exc_info:
        store.allocate_and_export(_allocation_spec())
    assert exc_info.value.kind is RegionControlErrorKind.INTERNAL_INVARIANT
    assert factory.world.constructed_parts == [RegionPartKind.PAYLOAD, RegionPartKind.COUNTER]
    gone = store.release(1)
    assert gone.status is ProviderReleaseStatus.ALREADY_GONE


def test_release_interrupt_on_payload_still_attempts_counter():
    store, factory = _open_store()
    store.allocate_and_export(_allocation_spec())
    factory.payloads[0].raise_on_release = KeyboardInterrupt()
    result = store.release(1)
    assert result.status is ProviderReleaseStatus.CLEANUP_INCOMPLETE
    assert result.failures[0].typed_cause is RegionCleanupCause.INTERRUPTED
    assert factory.counters[0].release_count == 1
    assert store.state is ProviderRegionStoreState.CLOSE_FAILED


def test_describe_and_local_view_reject_gone_and_unknown_ids():
    store, _factory = _open_store()
    store.allocate_and_export(_allocation_spec())
    store.release(1)
    with pytest.raises(RegionControlError) as gone:
        store.describe(1)
    assert gone.value.kind is RegionControlErrorKind.INVALID_FIELD_VALUE
    with pytest.raises(RegionControlError) as unknown:
        store.local_view(8, RegionPartKind.PAYLOAD)
    assert unknown.value.kind is RegionControlErrorKind.INVALID_FIELD_VALUE
    with pytest.raises(RegionControlError) as zero:
        store.describe(0)
    assert zero.value.kind is RegionControlErrorKind.INVALID_FIELD_VALUE


def test_unmaterialized_fake_facts_fail_and_materialize_does_not_release():
    factory = FakeShellFactory()
    spec = _allocation_spec()
    payload = factory(_sim_context(), RegionPartKind.PAYLOAD, spec.payload)
    with pytest.raises(RegionControlError) as exc_info:
        payload.mapping_bytes()
    assert exc_info.value.kind is RegionControlErrorKind.INTERNAL_INVARIANT
    payload.materialize()
    assert payload.release_count == 0
    assert payload.mapping_bytes() == spec.payload.logical_bytes


def test_successful_sweep_releases_active_resources_in_id_order():
    store, factory = _open_store()
    store.allocate_and_export(_allocation_spec())
    store.allocate_and_export(_allocation_spec())
    results = store.sweep()
    assert [item.status for item in results] == [
        ProviderReleaseStatus.RELEASED,
        ProviderReleaseStatus.RELEASED,
    ]
    assert [item.provider_resource_id for item in results] == [1, 2]
    assert store.state is ProviderRegionStoreState.CLOSED
    assert factory.payloads[0].release_count == 1
    assert factory.payloads[1].release_count == 1
    with pytest.raises(RegionControlError):
        store.describe(1)


def test_sim_posix_store_creates_two_distinct_named_objects_and_zeros_only_counter():
    store = ProviderRegionStore(_sim_context())
    result = store.allocate_and_export(_allocation_spec())
    try:
        from simpler.comm_provider import _posix_shm_create_name

        payload_name = result.export_descriptor.payload.import_capability.shm_name
        counter_name = result.export_descriptor.counter.import_capability.shm_name
        assert payload_name != counter_name
        payload_shm = SharedMemory(name=_posix_shm_create_name(payload_name))
        counter_shm = SharedMemory(name=_posix_shm_create_name(counter_name))
        try:
            assert payload_shm.buf is not None and payload_shm.buf.nbytes >= 64
            assert counter_shm.buf is not None and counter_shm.buf.nbytes >= 8
            assert bytes(counter_shm.buf[:8]) == b"\x00" * 8
            payload_view = store.local_view(result.provider_resource_id, RegionPartKind.PAYLOAD)
            counter_view = store.local_view(result.provider_resource_id, RegionPartKind.COUNTER)
            validate_independent_local_views(payload_view, counter_view)
        finally:
            payload_shm.close()
            counter_shm.close()
    finally:
        released = store.release(result.provider_resource_id)
        assert released.status is ProviderReleaseStatus.RELEASED


def test_sim_posix_collision_does_not_open_or_unlink_the_existing_object():
    from simpler.comm_provider import _generate_posix_shm_token, _posix_shm_create_name

    token = _generate_posix_shm_token()
    existing = SharedMemory(name=_posix_shm_create_name(token), create=True, size=8)
    try:
        assert existing.buf is not None
        existing.buf[:8] = b"KEEPKEEP"
        shell = SimPosixShmAllocation(
            _sim_context(),
            RegionPartKind.PAYLOAD,
            _payload_spec(8),
            candidate_name=token,
        )
        with pytest.raises(RegionControlError) as exc_info:
            shell.materialize()
        assert exc_info.value.kind is RegionControlErrorKind.BACKEND_FAILURE
        assert shell.release_once() is None
        still_there = SharedMemory(name=_posix_shm_create_name(token))
        try:
            assert still_there.buf is not None
            assert bytes(still_there.buf[:8]) == b"KEEPKEEP"
        finally:
            still_there.close()
    finally:
        existing.close()
        existing.unlink()


def test_sim_posix_overlong_token_is_rejected_before_creation():
    with pytest.raises(ValueError):
        SimPosixShmAllocation(
            _sim_context(),
            RegionPartKind.PAYLOAD,
            _payload_spec(),
            candidate_name="n" * (POSIX_SHM_TOKEN_MAX_BYTES + 1),
        )


def test_sim_posix_interrupted_create_unlinks_owned_name_once():
    from simpler.comm_provider import _generate_posix_shm_token, _posix_shm_create_name

    token = _generate_posix_shm_token()
    created: dict[str, SharedMemory] = {}

    class _CreateThenInterrupt:
        def __init__(self, name, *, create, size, **kwargs):
            created["shm"] = SharedMemory(name=name, create=create, size=size)
            raise KeyboardInterrupt()

    shell = SimPosixShmAllocation(
        _sim_context(),
        RegionPartKind.COUNTER,
        _counter_spec(),
        candidate_name=token,
        shm_cls=_CreateThenInterrupt,
    )
    with pytest.raises(KeyboardInterrupt):
        shell.materialize()
    failure = shell.release_once()
    assert failure is None
    assert shell.release_once() is None
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=_posix_shm_create_name(token))
    if "shm" in created:
        created["shm"].close()


def test_sim_posix_close_failure_still_unlinks_and_keeps_later_detail():
    from simpler.comm_provider import _posix_shm_create_name

    shell = SimPosixShmAllocation(_sim_context(), RegionPartKind.PAYLOAD, _payload_spec(8))
    shell.materialize()
    token = shell.candidate_name

    def _boom_close():
        raise OSError("close failed")

    assert shell._shm is not None
    shell._shm.close = _boom_close  # type: ignore[method-assign]
    failure = shell.release_once()
    assert failure is not None
    assert failure.typed_cause is RegionCleanupCause.BACKEND_ERROR
    assert [step for step, _exc in shell.local_cleanup_details] == ["close"]
    assert shell.release_once() is failure
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=_posix_shm_create_name(token))


def test_sim_posix_unlink_failure_is_summarized_after_successful_close():
    shell = SimPosixShmAllocation(_sim_context(), RegionPartKind.COUNTER, _counter_spec())
    shell.materialize()

    def _boom_unlink():
        raise OSError("unlink failed")

    assert shell._shm is not None
    shell._shm.unlink = _boom_unlink  # type: ignore[method-assign]
    failure = shell.release_once()
    assert failure is not None
    assert [step for step, _exc in shell.local_cleanup_details] == ["unlink"]
    shell._name_ownership_known = False
    try:
        if shell._shm is not None:
            SharedMemory.unlink(shell._shm)
    except FileNotFoundError:
        pass


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_posix_token_has_no_slash_on_linux_and_macos(monkeypatch, platform):
    from simpler.comm_provider import _generate_posix_shm_token, _posix_shm_create_name

    monkeypatch.setattr(sys, "platform", platform)
    token = _generate_posix_shm_token()
    assert token.startswith("smp_")
    assert "/" not in token
    assert _posix_shm_create_name(token) == token
    shell = SimPosixShmAllocation(_sim_context(), RegionPartKind.PAYLOAD, _payload_spec(8, BackendKind.POSIX_SHM))
    try:
        shell.materialize()
        capability = shell.import_capability()
        assert "/" not in shell.candidate_name
        assert capability.shm_name == shell.candidate_name
        assert "/" not in capability.shm_name
        consumer = SharedMemory(name=_posix_shm_create_name(capability.shm_name))
        try:
            assert consumer.buf is not None
        finally:
            consumer.close()
    finally:
        assert shell.release_once() is None


def test_posix_token_does_not_use_shared_memory_prepend_attr():
    assert "_prepend_leading_slash" not in _COMM_PROVIDER_PATH.read_text()


def test_posix_unlink_and_tracker_use_the_same_single_leading_slash(monkeypatch):
    from multiprocessing import resource_tracker

    from simpler.comm_provider import _unlink_posix_shm_token

    unlinks: list[str] = []
    unregisters: list[tuple[str, str]] = []

    class _FakePosix:
        @staticmethod
        def shm_unlink(name: str) -> None:
            unlinks.append(name)

    monkeypatch.setitem(sys.modules, "_posixshmem", _FakePosix)
    monkeypatch.setattr(
        resource_tracker,
        "unregister",
        lambda name, kind: unregisters.append((name, kind)),
    )
    _unlink_posix_shm_token("smp_deadbeefcafebabe01234567")
    assert unlinks == ["/smp_deadbeefcafebabe01234567"]
    assert unregisters == [("/smp_deadbeefcafebabe01234567", "shared_memory")]


def test_posix_repeated_unlink_treats_enoent_as_done(monkeypatch):
    from multiprocessing import resource_tracker

    from simpler.comm_provider import _unlink_posix_shm_token

    class _FakePosix:
        calls = 0

        @staticmethod
        def shm_unlink(name: str) -> None:
            _FakePosix.calls += 1
            if _FakePosix.calls > 1:
                raise FileNotFoundError(name)

    monkeypatch.setitem(sys.modules, "_posixshmem", _FakePosix)
    monkeypatch.setattr(resource_tracker, "unregister", lambda *_args: None)
    _unlink_posix_shm_token("smp_deadbeefcafebabe01234567")
    _unlink_posix_shm_token("smp_deadbeefcafebabe01234567")
    assert _FakePosix.calls == 2


def test_posix_unregister_failure_warns_without_changing_result(monkeypatch, caplog):
    from multiprocessing import resource_tracker

    from simpler.comm_provider import _unlink_posix_shm_token

    class _FakePosix:
        @staticmethod
        def shm_unlink(name: str) -> None:
            del name

    monkeypatch.setitem(sys.modules, "_posixshmem", _FakePosix)
    monkeypatch.setattr(
        resource_tracker,
        "unregister",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("tracker boom")),
    )
    caplog.set_level(logging.WARNING, logger="simpler")
    _unlink_posix_shm_token("smp_deadbeefcafebabe01234567")
    assert "smp_deadbeefcafebabe01234567" in caplog.text
    assert "tracker boom" in caplog.text


def test_sim_posix_first_cleanup_failure_is_close_when_both_steps_fail():
    from simpler.comm_provider import _posix_shm_create_name

    shell = SimPosixShmAllocation(_sim_context(), RegionPartKind.PAYLOAD, _payload_spec(8))
    shell.materialize()
    token = shell.candidate_name
    assert shell._shm is not None
    real_unlink = shell._shm.unlink

    def _boom_close():
        raise OSError("close failed")

    def _boom_unlink():
        raise OSError("unlink failed")

    shell._shm.close = _boom_close  # type: ignore[method-assign]
    shell._shm.unlink = _boom_unlink  # type: ignore[method-assign]
    failure = shell.release_once()
    assert failure is not None
    assert [step for step, _exc in shell.local_cleanup_details] == ["close", "unlink"]
    try:
        real_unlink()
    except FileNotFoundError:
        pass
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=_posix_shm_create_name(token))


def _onboard_context(device_id: int = 0) -> RegionAllocationContext:
    return RegionAllocationContext(
        environment_kind=RegionEnvironmentKind.ONBOARD,
        target=DeviceAllocationTarget(device_id=device_id),
    )


@pytest.fixture
def fake_vmm():
    from _task_interface import (  # pyright: ignore[reportMissingImports]
        _region_vmm_inspect,
        _region_vmm_test_live_handles,
        _region_vmm_test_reset_hooks,
        _region_vmm_test_use_fake_driver,
    )

    _region_vmm_test_use_fake_driver()
    _region_vmm_test_reset_hooks()
    before = set(_region_vmm_test_live_handles())
    yield
    _region_vmm_test_reset_hooks()
    leftover = [handle for handle in _region_vmm_test_live_handles() if handle not in before]
    unclean = [handle for handle in leftover if not _region_vmm_inspect(handle).release_once_done]
    assert unclean == [], f"fake VMM leaked live handles: {unclean}"


def test_region_vmm_begin_has_no_physical_side_effect(fake_vmm):
    from _task_interface import (  # pyright: ignore[reportMissingImports]
        _region_vmm_begin,
        _region_vmm_inspect,
        _region_vmm_release,
        _region_vmm_test_issued_ops,
        _region_vmm_test_live_handles,
    )

    before = set(_region_vmm_test_live_handles())
    handle = _region_vmm_begin(3)
    inspect = _region_vmm_inspect(handle)
    try:
        assert handle not in before
        assert inspect.present is True
        assert inspect.device_id == 3
        assert inspect.physical_allocated is False
        assert inspect.va_reserved is False
        assert inspect.mapped is False
        assert inspect.exported is False
        assert inspect.mapping_bytes == 0
        assert _region_vmm_test_issued_ops() == []
        assert handle in _region_vmm_test_live_handles()
    finally:
        _region_vmm_release(handle)
    assert _region_vmm_inspect(handle).present is False


@pytest.mark.parametrize(
    ("stage", "when", "allocated", "reserved", "mapped", "exported"),
    [
        ("physical_alloc", "before", False, False, False, False),
        ("physical_alloc", "after", True, False, False, False),
        ("va_reserve", "before", True, False, False, False),
        ("va_reserve", "after", True, True, False, False),
        ("map", "before", True, True, False, False),
        ("map", "after", True, True, True, False),
        ("set_access", "before", True, True, True, False),
        ("set_access", "after", True, True, True, False),
        ("export", "before", True, True, True, False),
        ("export", "after", True, True, True, True),
    ],
)
def test_region_vmm_allocate_failure_retains_record_without_rollback(
    fake_vmm, stage, when, allocated, reserved, mapped, exported
):
    from _task_interface import (  # pyright: ignore[reportMissingImports]
        _region_vmm_allocate_export,
        _region_vmm_begin,
        _region_vmm_inspect,
        _region_vmm_release,
        _region_vmm_test_fail_stage,
        _region_vmm_test_issued_ops,
        _region_vmm_test_reset_hooks,
    )

    handle = _region_vmm_begin(0)
    _region_vmm_test_reset_hooks()
    _region_vmm_test_fail_stage(stage, when)
    with pytest.raises(RuntimeError, match="injected region VMM"):
        _region_vmm_allocate_export(handle, 8)
    inspect = _region_vmm_inspect(handle)
    issued = _region_vmm_test_issued_ops()
    try:
        assert inspect.present is True
        assert inspect.physical_allocated is allocated
        assert inspect.va_reserved is reserved
        assert inspect.mapped is mapped
        assert inspect.exported is exported
        if when == "before":
            assert stage not in issued
        else:
            assert stage in issued
    finally:
        _region_vmm_test_reset_hooks()
        _region_vmm_release(handle)
    assert _region_vmm_inspect(handle).present is False


def test_region_vmm_failed_unmap_skips_va_and_physical(fake_vmm):
    from _task_interface import (  # pyright: ignore[reportMissingImports]
        _region_vmm_allocate_export,
        _region_vmm_begin,
        _region_vmm_inspect,
        _region_vmm_release,
        _region_vmm_test_fail_stage,
        _region_vmm_test_issued_ops,
        _region_vmm_test_reset_hooks,
    )

    handle = _region_vmm_begin(0)
    _region_vmm_allocate_export(handle, 8)
    _region_vmm_test_reset_hooks()
    _region_vmm_test_fail_stage("unmap", "before")
    with pytest.raises(RuntimeError, match="before unmap"):
        _region_vmm_release(handle)
    inspect = _region_vmm_inspect(handle)
    issued = _region_vmm_test_issued_ops()
    assert inspect.present is True
    assert inspect.mapped is True
    assert inspect.va_reserved is True
    assert inspect.physical_allocated is True
    assert inspect.unmap_attempted is True
    assert inspect.unmap_complete is False
    assert inspect.release_once_done is True
    assert "unmap" not in issued
    assert "va_release" not in issued
    assert "physical_free" not in issued
    _region_vmm_test_reset_hooks()
    with pytest.raises(RuntimeError, match="before unmap"):
        _region_vmm_release(handle)
    assert _region_vmm_test_issued_ops() == []
    assert _region_vmm_inspect(handle).mapped is True


def test_region_vmm_successful_unmap_attempts_va_and_physical_independently(fake_vmm):
    from _task_interface import (  # pyright: ignore[reportMissingImports]
        _region_vmm_allocate_export,
        _region_vmm_begin,
        _region_vmm_inspect,
        _region_vmm_release,
        _region_vmm_test_fail_stage,
        _region_vmm_test_issued_ops,
        _region_vmm_test_reset_hooks,
    )

    handle = _region_vmm_begin(0)
    _region_vmm_allocate_export(handle, 8)
    _region_vmm_test_reset_hooks()
    _region_vmm_test_fail_stage("va_release", "before")
    with pytest.raises(RuntimeError, match="before va_release"):
        _region_vmm_release(handle)
    inspect = _region_vmm_inspect(handle)
    issued = _region_vmm_test_issued_ops()
    assert inspect.present is True
    assert inspect.mapped is False
    assert inspect.unmap_complete is True
    assert inspect.va_reserved is True
    assert inspect.physical_allocated is False
    assert inspect.physical_free_complete is True
    assert inspect.first_cleanup_failure
    assert inspect.local_cleanup_details == [inspect.first_cleanup_failure]
    assert issued.count("unmap") == 1
    assert "va_release" not in issued
    assert issued.count("physical_free") == 1
    _region_vmm_test_reset_hooks()
    with pytest.raises(RuntimeError, match="before va_release"):
        _region_vmm_release(handle)
    assert _region_vmm_test_issued_ops() == []


def test_region_vmm_already_gone_unmap_still_releases_va_and_physical(fake_vmm):
    from _task_interface import (  # pyright: ignore[reportMissingImports]
        _region_vmm_allocate_export,
        _region_vmm_begin,
        _region_vmm_inspect,
        _region_vmm_release,
        _region_vmm_test_issued_ops,
        _region_vmm_test_reset_hooks,
        _region_vmm_test_set_unmap_already_gone,
    )

    handle = _region_vmm_begin(0)
    _region_vmm_allocate_export(handle, 8)
    _region_vmm_test_reset_hooks()
    _region_vmm_test_set_unmap_already_gone()
    _region_vmm_release(handle)
    assert _region_vmm_inspect(handle).present is False
    assert _region_vmm_test_issued_ops() == ["bind_device", "unmap", "va_release", "physical_free"]


def test_region_vmm_two_handles_are_independent(fake_vmm):
    from _task_interface import (  # pyright: ignore[reportMissingImports]
        _region_vmm_allocate_export,
        _region_vmm_begin,
        _region_vmm_inspect,
        _region_vmm_release,
    )

    first = _region_vmm_allocate_export(_region_vmm_begin(1), 8)
    second = _region_vmm_allocate_export(_region_vmm_begin(2), 64)
    try:
        assert first.registry_handle != second.registry_handle
        assert first.shareable_handle != second.shareable_handle
        assert first.device_addr != second.device_addr
        assert first.mapping_bytes == 64
        assert second.mapping_bytes == 64
        assert _region_vmm_inspect(first.registry_handle).device_id == 1
        assert _region_vmm_inspect(second.registry_handle).device_id == 2
    finally:
        _region_vmm_release(first.registry_handle)
        _region_vmm_release(second.registry_handle)


def test_region_vmm_two_handles_use_independent_mapping_sizes(fake_vmm):
    from _task_interface import (  # pyright: ignore[reportMissingImports]
        _region_vmm_allocate_export,
        _region_vmm_begin,
        _region_vmm_release,
    )

    first = _region_vmm_allocate_export(_region_vmm_begin(1), 64)
    second = _region_vmm_allocate_export(_region_vmm_begin(2), 128)
    try:
        assert first.registry_handle != second.registry_handle
        assert first.shareable_handle != second.shareable_handle
        assert first.mapping_bytes == 64
        assert second.mapping_bytes == 128
    finally:
        _region_vmm_release(first.registry_handle)
        _region_vmm_release(second.registry_handle)


def test_region_vmm_later_independent_errors_stay_local(fake_vmm):
    from _task_interface import (  # pyright: ignore[reportMissingImports]
        _region_vmm_allocate_export,
        _region_vmm_begin,
        _region_vmm_inspect,
        _region_vmm_release,
        _region_vmm_test_fail_stage,
        _region_vmm_test_issued_ops,
        _region_vmm_test_reset_hooks,
    )

    handle = _region_vmm_begin(0)
    _region_vmm_allocate_export(handle, 8)
    _region_vmm_test_reset_hooks()
    _region_vmm_test_fail_stage("va_release", "after")
    with pytest.raises(RuntimeError, match="after va_release"):
        _region_vmm_release(handle)
    inspect = _region_vmm_inspect(handle)
    issued = _region_vmm_test_issued_ops()
    assert inspect.present is True
    assert inspect.unmap_complete is True
    assert inspect.va_release_complete is True
    assert inspect.physical_free_complete is True
    assert inspect.physical_allocated is False
    assert inspect.first_cleanup_failure
    assert inspect.local_cleanup_details == [inspect.first_cleanup_failure]
    assert "after va_release" in inspect.first_cleanup_failure
    assert issued.count("unmap") == 1
    assert issued.count("va_release") == 1
    assert issued.count("physical_free") == 1
    _region_vmm_test_reset_hooks()
    with pytest.raises(RuntimeError, match="after va_release"):
        _region_vmm_release(handle)
    assert _region_vmm_test_issued_ops() == []


def test_region_vmm_physical_free_failure_retains_record_without_reissue(fake_vmm):
    from _task_interface import (  # pyright: ignore[reportMissingImports]
        _region_vmm_allocate_export,
        _region_vmm_begin,
        _region_vmm_inspect,
        _region_vmm_release,
        _region_vmm_test_fail_stage,
        _region_vmm_test_issued_ops,
        _region_vmm_test_reset_hooks,
    )

    handle = _region_vmm_begin(0)
    _region_vmm_allocate_export(handle, 8)
    _region_vmm_test_reset_hooks()
    _region_vmm_test_fail_stage("physical_free", "before")
    with pytest.raises(RuntimeError, match="before physical_free"):
        _region_vmm_release(handle)
    inspect = _region_vmm_inspect(handle)
    issued = _region_vmm_test_issued_ops()
    assert inspect.present is True
    assert inspect.unmap_complete is True
    assert inspect.va_reserved is False
    assert inspect.physical_allocated is True
    assert inspect.physical_free_attempted is True
    assert inspect.physical_free_complete is False
    assert "physical_free" not in issued
    assert issued.count("unmap") == 1
    assert issued.count("va_release") == 1
    _region_vmm_test_reset_hooks()
    with pytest.raises(RuntimeError, match="before physical_free"):
        _region_vmm_release(handle)
    assert _region_vmm_test_issued_ops() == []
    assert _region_vmm_inspect(handle).physical_allocated is True


def test_region_vmm_release_bind_failure_keeps_ledger_and_does_not_retry(fake_vmm):
    from _task_interface import (  # pyright: ignore[reportMissingImports]
        _region_vmm_allocate_export,
        _region_vmm_begin,
        _region_vmm_inspect,
        _region_vmm_release,
        _region_vmm_test_fail_stage,
        _region_vmm_test_issued_ops,
        _region_vmm_test_reset_hooks,
    )

    handle = _region_vmm_begin(3)
    _region_vmm_allocate_export(handle, 8)
    _region_vmm_test_reset_hooks()
    _region_vmm_test_fail_stage("bind_device", "before")
    with pytest.raises(RuntimeError, match="bind_device failed:") as first:
        _region_vmm_release(handle)
    inspect = _region_vmm_inspect(handle)
    issued = _region_vmm_test_issued_ops()
    assert inspect.present is True
    assert inspect.mapped is True
    assert inspect.va_reserved is True
    assert inspect.physical_allocated is True
    assert inspect.unmap_attempted is False
    assert inspect.va_release_attempted is False
    assert inspect.physical_free_attempted is False
    assert inspect.release_once_done is True
    assert inspect.first_cleanup_failure == str(first.value)
    assert inspect.first_cleanup_failure.startswith("bind_device failed:")
    assert inspect.local_cleanup_details == [inspect.first_cleanup_failure]
    assert issued == ["bind_device"]
    _region_vmm_test_reset_hooks()
    with pytest.raises(RuntimeError, match="bind_device failed:") as second:
        _region_vmm_release(handle)
    assert str(second.value) == str(first.value)
    assert _region_vmm_test_issued_ops() == []
    later = _region_vmm_inspect(handle)
    assert later.mapped is True
    assert later.va_reserved is True
    assert later.physical_allocated is True
    assert later.first_cleanup_failure == inspect.first_cleanup_failure


def test_region_vmm_zero_binds_record_device_before_zero(fake_vmm):
    from _task_interface import (  # pyright: ignore[reportMissingImports]
        _region_vmm_allocate_export,
        _region_vmm_begin,
        _region_vmm_inspect,
        _region_vmm_release,
        _region_vmm_test_issued_ops,
        _region_vmm_test_reset_hooks,
        _region_vmm_zero_bytes,
    )

    handle = _region_vmm_begin(4)
    _region_vmm_allocate_export(handle, 8)
    other = _region_vmm_allocate_export(_region_vmm_begin(9), 8)
    _region_vmm_test_reset_hooks()
    _region_vmm_zero_bytes(handle, 0, 8)
    assert _region_vmm_test_issued_ops() == ["bind_device", "zero_bytes"]
    assert _region_vmm_inspect(handle).device_id == 4
    assert _region_vmm_inspect(other.registry_handle).device_id == 9
    _region_vmm_release(handle)
    _region_vmm_release(other.registry_handle)


def test_region_vmm_zero_bind_failure_does_not_issue_zero(fake_vmm):
    from _task_interface import (  # pyright: ignore[reportMissingImports]
        _region_vmm_allocate_export,
        _region_vmm_begin,
        _region_vmm_inspect,
        _region_vmm_release,
        _region_vmm_test_fail_stage,
        _region_vmm_test_issued_ops,
        _region_vmm_test_reset_hooks,
        _region_vmm_zero_bytes,
    )

    handle = _region_vmm_begin(4)
    _region_vmm_allocate_export(handle, 8)
    _region_vmm_test_reset_hooks()
    _region_vmm_test_fail_stage("bind_device", "before")
    with pytest.raises(RuntimeError, match="injected region VMM before bind_device"):
        _region_vmm_zero_bytes(handle, 0, 8)
    assert _region_vmm_test_issued_ops() == ["bind_device"]
    inspect = _region_vmm_inspect(handle)
    assert inspect.present is True
    assert inspect.mapped is True
    _region_vmm_test_reset_hooks()
    _region_vmm_release(handle)


def test_vmm_allocation_stores_handle_before_allocate_and_zeros_only_logical_bytes(fake_vmm):
    from _task_interface import (  # pyright: ignore[reportMissingImports]
        _region_vmm_inspect,
        _region_vmm_test_fail_stage,
        _region_vmm_test_fake_bytes,
        _region_vmm_test_reset_hooks,
    )

    shell = VmmAllocation(_onboard_context(4), RegionPartKind.COUNTER, _counter_spec(8))
    assert shell.registry_handle is None
    _region_vmm_test_fail_stage("export", "before")
    with pytest.raises(RegionControlError) as exc_info:
        shell.materialize()
    assert exc_info.value.kind is RegionControlErrorKind.BACKEND_FAILURE
    handle = shell.registry_handle
    assert handle is not None
    inspect = _region_vmm_inspect(handle)
    assert inspect.present is True
    assert inspect.mapped is True
    assert inspect.exported is False
    _region_vmm_test_reset_hooks()
    assert shell.release_once() is None
    assert _region_vmm_inspect(handle).present is False
    assert shell.release_once() is None

    ready = VmmAllocation(_onboard_context(4), RegionPartKind.COUNTER, _counter_spec(8))
    ready.materialize()
    try:
        ready.zero_bytes(0, 8)
        backing = bytes(_region_vmm_test_fake_bytes(ready.registry_handle))
        assert backing[:8] == b"\x00" * 8
        assert backing[8:] == b"\xff" * (len(backing) - 8)
        capability = ready.import_capability()
        assert capability.device_id == 4
        assert ready.mapping_bytes() == 64
        assert ready.local_base() == int(_region_vmm_inspect(ready.registry_handle).device_addr)
    finally:
        assert ready.release_once() is None


def test_onboard_store_creates_two_independent_vmm_handles_and_zeros_only_counter(fake_vmm):
    from _task_interface import (  # pyright: ignore[reportMissingImports]
        _region_vmm_inspect,
        _region_vmm_test_fake_bytes,
    )

    store = ProviderRegionStore(_onboard_context(5))
    result = store.allocate_and_export(_allocation_spec())
    resource = store._resources[result.provider_resource_id]
    payload_shell = resource.parts[RegionPartKind.PAYLOAD].allocation
    counter_shell = resource.parts[RegionPartKind.COUNTER].allocation
    assert isinstance(payload_shell, VmmAllocation)
    assert isinstance(counter_shell, VmmAllocation)
    payload_handle = payload_shell.registry_handle
    counter_handle = counter_shell.registry_handle
    try:
        payload = result.export_descriptor.payload.import_capability
        counter = result.export_descriptor.counter.import_capability
        assert isinstance(payload, VmmShareableHandleImport)
        assert isinstance(counter, VmmShareableHandleImport)
        assert payload.shareable_handle != counter.shareable_handle
        assert payload.device_id == 5
        assert payload_handle != counter_handle
        payload_view = store.local_view(result.provider_resource_id, RegionPartKind.PAYLOAD)
        counter_view = store.local_view(result.provider_resource_id, RegionPartKind.COUNTER)
        validate_independent_local_views(payload_view, counter_view)
        payload_bytes = bytes(_region_vmm_test_fake_bytes(payload_handle))
        counter_bytes = bytes(_region_vmm_test_fake_bytes(counter_handle))
        assert payload_bytes[:64] == b"\xff" * 64
        assert counter_bytes[:8] == b"\x00" * 8
        assert counter_bytes[8:] == b"\xff" * (len(counter_bytes) - 8)
    finally:
        released = store.release(result.provider_resource_id)
    assert released.status is ProviderReleaseStatus.RELEASED
    assert _region_vmm_inspect(payload_handle).present is False
    assert _region_vmm_inspect(counter_handle).present is False


def test_closed_dispatcher_routes_onboard_vmm_window_to_vmm_allocation():
    payload = comm_provider_module._closed_part_dispatcher(
        _onboard_context(),
        RegionPartKind.PAYLOAD,
        _payload_spec(),
    )
    sim = comm_provider_module._closed_part_dispatcher(
        _sim_context(),
        RegionPartKind.PAYLOAD,
        _payload_spec(),
    )
    assert isinstance(payload, VmmAllocation)
    assert isinstance(sim, SimPosixShmAllocation)
    assert payload.registry_handle is None
