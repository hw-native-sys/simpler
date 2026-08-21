# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for W2 endpoint selectors, registry resolution, and planning."""

import dataclasses

import pytest
from simpler import comm_endpoints as ce
from simpler.buffer import BackendKind as BufferBackendKind
from simpler.worker import RemoteWorkerSpec, Worker, _Lifecycle, _MpiDirectWorkerSpec


def _ready(worker: Worker) -> Worker:
    worker._lifecycle = _Lifecycle.READY
    return worker


def _l3(device_ids=(), *, num_sub_workers: int = 0) -> Worker:
    return _ready(Worker(level=3, device_ids=list(device_ids), num_sub_workers=num_sub_workers))


def _l4_with_local_l3(device_ids=()) -> Worker:
    l3 = Worker(level=3, device_ids=list(device_ids), num_sub_workers=0)
    w4 = Worker(level=4, num_sub_workers=0)
    w4.add_worker(l3)
    return _ready(w4)


def _l4_with_remote(*specs: RemoteWorkerSpec) -> Worker:
    worker = Worker(level=4, num_sub_workers=0)
    for spec in specs:
        worker.add_remote_worker(spec)
    return _ready(worker)


def _l4_with_mpi_direct(*, hosts: tuple[str, ...], device_ids_by_rank: tuple[tuple[int, ...], ...]) -> Worker:
    worker = Worker(level=4, num_sub_workers=0)
    for worker_id, (host, device_ids) in enumerate(zip(hosts, device_ids_by_rank)):
        worker._add_mpi_direct_worker(
            _MpiDirectWorkerSpec(
                worker_id=worker_id,
                mpi_rank=worker_id + 1,
                session_id=worker_id + 1,
                host=host,
                comm_profile="sim",
                platform="a2a3sim",
                runtime="sim",
                device_ids=device_ids,
                global_device_ranks=(),
                hub=object(),  # type: ignore[arg-type]
                attach_timeout_s=1.0,
                runtime_timeout_s=1.0,
            )
        )
    return _ready(worker)


def _record(worker: Worker, path: str, deployment: ce.EndpointDeployment) -> ce.EndpointRecord:
    return worker._resolve_region_spec([ce.at(path, deployment)], ce.SingleOwner()).members[0]


def _access_key(
    backend_kind: ce.BackendKind,
    part: ce.RegionPartKind,
    adapter_kind: ce.AdapterKind,
    adapter_profile: ce.AdapterProfile,
):
    return (backend_kind, part, adapter_kind, adapter_profile)


def _supported_parts(
    backend_kind: ce.BackendKind,
    adapter_kind: ce.AdapterKind,
    adapter_profile: ce.AdapterProfile,
):
    return {
        _access_key(backend_kind, part, adapter_kind, adapter_profile): True
        for part in (ce.RegionPartKind.PAYLOAD, ce.RegionPartKind.COUNTER)
    }


def _plan(worker: Worker, members, topology=None, access=None):
    if access is not None:
        worker._region_access_service = ce.StaticRegionAccessService(access)
    return worker._plan_region(
        members,
        topology or ce.SingleOwner(),
        ce.RegionLayoutSpec(payload_bytes=64, counter_bytes=8),
    )


def _attachments_by_member(part: ce.RegionPartPlan) -> dict[ce.EndpointIdentity, ce.MemberAttachmentPlan]:
    return {attachment.member: attachment for attachment in part.attachments}


def test_selector_constructors_validate_shape_and_preserve_hashability():
    selector = ce.at("L3/L2[0]", "DEVICE_AICORE")
    assert selector == ce.EndpointSelector(ce.EndpointSelectorKind.AT, "L3/L2[0]", ce.DEVICE_AICORE)
    assert hash(selector) == hash(ce.at("L3/L2[0]", ce.DEVICE_AICORE))
    assert ce.under("L3", ce.DEVICE_AICPU).kind is ce.EndpointSelectorKind.UNDER
    with pytest.raises(ValueError, match="invalid endpoint deployment"):
        ce.at("L3", "DEVICE")
    with pytest.raises(ValueError, match="non-empty"):
        ce.at("", ce.HOST_CPU)
    with pytest.raises(ValueError, match="empty segments"):
        ce.EndpointSelector(ce.EndpointSelectorKind.AT, "L3//L2[0]", ce.DEVICE_AICORE)


def test_snapshot_is_frozen_flat_and_uses_worker_owner_instance_id():
    worker = _l3(device_ids=[8, 9])
    snapshot = worker._endpoint_topology_snapshot()
    assert dataclasses.is_dataclass(snapshot)
    assert snapshot.root_level == 3
    assert snapshot.session_instance_id == worker._owner_instance_id
    assert isinstance(snapshot.entries, tuple)
    assert [entry.path for entry in snapshot.entries] == [
        "L3",
        "L3/L2[0]",
        "L3/L2[0]",
        "L3/L2[1]",
        "L3/L2[1]",
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.entries[0].path = "L3/changed"


def test_owner_nonce_resolves_to_the_endpoint_that_mints_under_it():
    """A `BufferDescriptor` names its owner only by an opaque nonce, so resolving that back to a
    deployment is a registry judgement — this is the binding that supplies it."""
    worker = _l4_with_local_l3(device_ids=[4])
    child = worker._next_level_workers[0]
    registry = worker._get_endpoint_registry()

    assert registry.owner_endpoint(worker._owner_instance_id) == _record(worker, "L4", ce.HOST_CPU)
    # The child mints its own nonce, and resolves to its own endpoint rather than the root's.
    assert child._owner_instance_id != worker._owner_instance_id
    assert registry.owner_endpoint(child._owner_instance_id) == _record(worker, "L4/L3[0]", ce.HOST_CPU)

    # A device endpoint is a view of a chip, not a buffer owner, so it binds no nonce.
    device = _record(worker, "L4/L3[0]/L2[0]", ce.DEVICE_AICORE)
    assert device not in [registry.owner_endpoint(w._owner_instance_id) for w in (worker, child)]


def test_unknown_owner_nonce_is_a_typed_refusal_not_a_guess():
    """A remote Worker mints in its own process; its binding has to arrive over a session channel
    rather than be inferred, so an unresolvable nonce fails loudly."""
    worker = _l4_with_remote(RemoteWorkerSpec(endpoint="10.0.0.7:1234", platform="a2a3", device_ids=(6,)))
    registry = worker._get_endpoint_registry()
    with pytest.raises(ce.EndpointResolveError) as excinfo:
        registry.owner_endpoint(b"\x00" * 8)
    assert excinfo.value.reason is ce.EndpointResolveReason.OWNER_NOT_REGISTERED
    assert "0000000000000000" in excinfo.value.message


def test_registry_only_builds_from_snapshot_and_identity_binds_epoch_and_session():
    worker = _l3(device_ids=[0])
    snapshot = worker._endpoint_topology_snapshot()
    assert not hasattr(ce.EndpointRegistry, "from_worker")
    registry0 = ce.EndpointRegistry.from_snapshot(snapshot, registry_epoch=0)
    registry1 = ce.EndpointRegistry.from_snapshot(snapshot, registry_epoch=1)
    record0 = registry0.resolve_members([ce.at("L3/L2[0]", ce.DEVICE_AICORE)])[0]
    record1 = registry1.resolve_members([ce.at("L3/L2[0]", ce.DEVICE_AICORE)])[0]
    other_session = dataclasses.replace(snapshot, session_instance_id=b"other-session")
    record_other = ce.EndpointRegistry.from_snapshot(other_session, registry_epoch=0).resolve_members(
        [ce.at("L3/L2[0]", ce.DEVICE_AICORE)]
    )[0]
    root = registry0.resolve_members([ce.at("L3", ce.HOST_CPU)])[0]
    assert record0.endpoint_id == record0.identity.endpoint_id
    assert record0.identity != record1.identity
    assert record0.identity != record_other.identity
    assert record0.identity != root.identity


def test_backend_kind_reuses_buffer_abi_vocabulary():
    assert ce.BackendKind is BufferBackendKind
    assert ce.BackendKind.VMM_WINDOW is BufferBackendKind.VMM_WINDOW


def test_path_grammar_requires_root_and_uses_numeric_canonical_sort():
    worker = _l3(device_ids=range(11))
    with pytest.raises(ce.EndpointResolveError) as excinfo:
        worker._resolve_region_spec([ce.at("L3[0]/L2[0]", ce.DEVICE_AICORE)], ce.SingleOwner())
    assert excinfo.value.reason is ce.EndpointResolveReason.INVALID_PATH

    members = worker._resolve_region_spec([ce.under("L3", ce.DEVICE_AICORE)], ce.SingleOwner()).members
    paths = [record.path for record in members]
    assert paths[2] == "L3/L2[2]"
    assert paths[10] == "L3/L2[10]"


def test_l3_registry_registers_host_and_device_views_but_not_l2_host_cpu():
    worker = _l3(device_ids=[8, 9])
    assert _record(worker, "L3", ce.HOST_CPU).path == "L3"
    assert _record(worker, "L3/L2[0]", ce.DEVICE_AICORE).deployment is ce.DEVICE_AICORE
    assert _record(worker, "L3/L2[1]", ce.DEVICE_AICPU).deployment is ce.DEVICE_AICPU
    with pytest.raises(ce.EndpointResolveError) as excinfo:
        worker._resolve_region_spec([ce.at("L3/L2[0]", ce.HOST_CPU)], ce.SingleOwner())
    assert excinfo.value.reason is ce.EndpointResolveReason.ENDPOINT_NOT_REGISTERED


def test_l4_local_registry_registers_child_l3_devices_on_same_node():
    worker = _l4_with_local_l3(device_ids=[4, 7])
    root = _record(worker, "L4", ce.HOST_CPU)
    child_host = _record(worker, "L4/L3[0]", ce.HOST_CPU)
    child_aicore = _record(worker, "L4/L3[0]/L2[1]", ce.DEVICE_AICORE)
    child_aicpu = _record(worker, "L4/L3[0]/L2[1]", ce.DEVICE_AICPU)
    registry = worker._get_endpoint_registry()
    assert registry.same_node(root, child_host)
    assert registry.same_node(child_host, child_aicore)
    assert registry.same_node(child_aicore, child_aicpu)


def test_remote_registry_normalizes_node_identity_by_host_not_remote_status():
    worker = _l4_with_remote(
        RemoteWorkerSpec(endpoint="127.0.0.1:1234", platform="a2a3", device_ids=(6,)),
        RemoteWorkerSpec(endpoint="10.0.0.7:1234", platform="a2a3", device_ids=(7,)),
        RemoteWorkerSpec(endpoint="10.0.0.7:2345", platform="a2a3", device_ids=(8,)),
        RemoteWorkerSpec(endpoint="10.0.0.8:1234", platform="a2a3", device_ids=(9,)),
    )
    registry = worker._get_endpoint_registry()
    root = _record(worker, "L4", ce.HOST_CPU)
    loopback = _record(worker, "L4/L3[0]", ce.HOST_CPU)
    remote_a = _record(worker, "L4/L3[1]", ce.HOST_CPU)
    remote_b = _record(worker, "L4/L3[2]", ce.HOST_CPU)
    remote_c = _record(worker, "L4/L3[3]", ce.HOST_CPU)
    assert registry.same_node(root, loopback)
    assert registry.same_node(remote_a, remote_b)
    assert not registry.same_node(root, remote_a)
    assert not registry.same_node(remote_a, remote_c)


def test_mpi_direct_registry_registers_rank_hosts_and_devices():
    worker = _l4_with_mpi_direct(
        hosts=("localhost", "10.0.0.7", "10.0.0.7"),
        device_ids_by_rank=((12, 13), (14,), (15,)),
    )
    registry = worker._get_endpoint_registry()
    root = _record(worker, "L4", ce.HOST_CPU)
    local_device = _record(worker, "L4/L3[0]/L2[1]", ce.DEVICE_AICORE)
    remote_a = _record(worker, "L4/L3[1]", ce.HOST_CPU)
    remote_device = _record(worker, "L4/L3[1]/L2[0]", ce.DEVICE_AICORE)
    remote_b = _record(worker, "L4/L3[2]", ce.HOST_CPU)

    assert registry.same_node(root, local_device)
    assert registry.same_node(remote_a, remote_device)
    assert registry.same_node(remote_a, remote_b)
    assert not registry.same_node(root, remote_a)


def test_at_missing_path_reports_path_not_found():
    worker = _l3(device_ids=[0])
    with pytest.raises(ce.EndpointResolveError) as excinfo:
        worker._resolve_region_spec([ce.at("L3/L2[9]", ce.DEVICE_AICORE)], ce.SingleOwner())
    assert excinfo.value.reason is ce.EndpointResolveReason.PATH_NOT_FOUND
    assert "L3/L2[9] DEVICE_AICORE" in excinfo.value.message


def test_under_excludes_self_and_empty_expansion_is_error():
    worker = _l4_with_local_l3(device_ids=[])
    members = worker._resolve_region_spec([ce.under("L4", ce.HOST_CPU)], ce.SingleOwner()).members
    assert [member.path for member in members] == ["L4/L3[0]"]

    with pytest.raises(ce.EndpointResolveError) as excinfo:
        worker._resolve_region_spec([ce.under("L4/L3[0]", ce.HOST_CPU)], ce.SingleOwner())
    assert excinfo.value.reason is ce.EndpointResolveReason.EMPTY_UNDER_SELECTOR


def test_duplicate_member_expansion_is_rejected():
    worker = _l3(device_ids=[0, 1])
    with pytest.raises(ce.EndpointResolveError) as excinfo:
        worker._resolve_region_spec(
            [ce.under("L3", ce.DEVICE_AICORE), ce.at("L3/L2[0]", ce.DEVICE_AICORE)], ce.SingleOwner()
        )
    assert excinfo.value.reason is ce.EndpointResolveReason.DUPLICATE_ENDPOINT


def test_provider_resolve_is_registry_api_boundary():
    worker = _l3(device_ids=[0, 1])
    resolved = worker._resolve_region_spec(
        [ce.at("L3/L2[0]", ce.DEVICE_AICORE)],
        ce.SingleOwner(provider=ce.at("L3/L2[0]", ce.DEVICE_AICORE)),
    )
    assert resolved.topology.provider_endpoint == resolved.members[0].identity

    with pytest.raises(ce.EndpointResolveError) as excinfo:
        worker._resolve_region_spec(
            [ce.at("L3/L2[0]", ce.DEVICE_AICORE)],
            ce.SingleOwner(provider=ce.under("L3", ce.DEVICE_AICORE)),
        )
    assert excinfo.value.reason is ce.EndpointResolveReason.PROVIDER_NOT_SINGLE_ENDPOINT


def test_provider_not_in_members_is_backend_resolver_error():
    worker = _l3(device_ids=[0])
    plan = _plan(
        worker,
        [ce.at("L3", ce.HOST_CPU)],
        ce.SingleOwner(provider=ce.at("L3/L2[0]", ce.DEVICE_AICORE)),
    )
    assert isinstance(plan, ce.UnsupportedRegionPlan)
    assert plan.reason is ce.BackendUnsupportedReason.PROVIDER_NOT_IN_MEMBERS
    assert not plan.attempted_adapters


def test_default_provider_order_aicore_then_aicpu_then_host_cpu():
    worker = _l3(device_ids=[0])
    plan = _plan(
        worker,
        [ce.at("L3/L2[0]", ce.DEVICE_AICPU), ce.at("L3/L2[0]", ce.DEVICE_AICORE)],
        access=_supported_parts(
            ce.BackendKind.VMM_WINDOW,
            ce.AdapterKind.DEVICE_PEER,
            ce.AdapterProfile.DEVICE_VMM_PEER_IMPORT,
        ),
    )
    assert isinstance(plan, ce.BackendPlan)
    assert _record(worker, "L3/L2[0]", ce.DEVICE_AICORE).identity == plan.topology_plan.provider_endpoint

    worker = _l3(device_ids=[0])
    plan = _plan(
        worker,
        [ce.at("L3", ce.HOST_CPU), ce.at("L3/L2[0]", ce.DEVICE_AICPU)],
    )
    assert isinstance(plan, ce.BackendPlan)
    assert _record(worker, "L3/L2[0]", ce.DEVICE_AICPU).identity == plan.topology_plan.provider_endpoint

    worker = _l4_with_local_l3()
    plan = _plan(
        worker,
        [ce.at("L4", ce.HOST_CPU), ce.at("L4/L3[0]", ce.HOST_CPU)],
        access=_supported_parts(
            ce.BackendKind.POSIX_SHM,
            ce.AdapterKind.DIRECT_MAP,
            ce.AdapterProfile.HOST_SHM_MAP,
        ),
    )
    assert isinstance(plan, ce.BackendPlan)
    assert _record(worker, "L4", ce.HOST_CPU).identity == plan.topology_plan.provider_endpoint


def test_default_provider_missing_for_empty_members():
    worker = _l3()
    plan = _plan(worker, [])
    assert isinstance(plan, ce.UnsupportedRegionPlan)
    assert plan.reason is ce.BackendUnsupportedReason.NO_DEFAULT_PROVIDER
    assert not plan.attempted_adapters


def test_device_backend_default_host_consumer_uses_copy_for_payload_and_counter():
    worker = _l3(device_ids=[0])
    members = [ce.at("L3", ce.HOST_CPU), ce.at("L3/L2[0]", ce.DEVICE_AICORE)]
    plan = _plan(worker, members)
    assert isinstance(plan, ce.BackendPlan)
    assert not hasattr(plan, "required_capabilities")
    assert not hasattr(ce, "BackingKind")
    assert not hasattr(ce, "MaterializationMode")
    assert not hasattr(ce, "PlatformCapability")
    assert not hasattr(ce, "CapabilityResult")
    assert not hasattr(ce, "PlatformCapabilityCache")
    assert not hasattr(ce, "StaticPlatformCapabilityCache")
    resolved = worker._resolve_region_spec(members, ce.SingleOwner()).members
    assert plan.ordered_members == tuple(record.identity for record in resolved)
    assert plan.payload.part is ce.RegionPartKind.PAYLOAD
    assert plan.counter.part is ce.RegionPartKind.COUNTER
    assert plan.payload.backend_kind is ce.BackendKind.VMM_WINDOW
    assert plan.counter.backend_kind is ce.BackendKind.VMM_WINDOW

    host = _record(worker, "L3", ce.HOST_CPU)
    provider = _record(worker, "L3/L2[0]", ce.DEVICE_AICORE)
    payload_by_member = _attachments_by_member(plan.payload)
    assert payload_by_member[provider.identity].role is ce.AttachmentRole.PROVIDER
    assert payload_by_member[provider.identity].adapter_kind is None
    assert payload_by_member[host.identity].role is ce.AttachmentRole.CONSUMER
    assert payload_by_member[host.identity].adapter_kind is ce.AdapterKind.OWNER_DELEGATED_COPY
    assert payload_by_member[host.identity].adapter_profile is ce.AdapterProfile.HOST_VMM_COPY

    counter_host = _attachments_by_member(plan.counter)[host.identity]
    assert counter_host.adapter_kind is ce.AdapterKind.OWNER_DELEGATED_COPY
    assert counter_host.adapter_profile is ce.AdapterProfile.HOST_VMM_COPY


def test_host_direct_map_is_never_offered_over_a_vmm_backing():
    """`halHostRegister` refuses a VMM VA, so no host consumer may direct-map a VMM_WINDOW
    backing — for either part.

    The exclusion is structural rather than a service verdict: a service that would happily
    admit the pairing must still never see it, because a plan it accepted could not be
    materialized. Injecting exactly that permissive service is how this test proves the
    candidate is absent from enumeration and not merely rejected downstream.
    """
    worker = _l3(device_ids=[0])
    members = [ce.at("L3", ce.HOST_CPU), ce.at("L3/L2[0]", ce.DEVICE_AICORE)]
    permissive = _supported_parts(
        ce.BackendKind.VMM_WINDOW,
        ce.AdapterKind.DIRECT_MAP,
        ce.AdapterProfile.HOST_SVM_MAP,
    )
    plan = _plan(worker, members, access=permissive)

    # Every candidate was refused, and none of the refused ones was the direct map.
    assert isinstance(plan, ce.UnsupportedRegionPlan)
    assert plan.reason is ce.BackendUnsupportedReason.ADAPTER_UNSUPPORTED
    assert ce.AdapterProfile.HOST_SVM_MAP not in [attempt.adapter_profile for attempt in plan.attempted_adapters]
    assert ce.AdapterKind.DIRECT_MAP not in [attempt.adapter_kind for attempt in plan.attempted_adapters]

    provider = _record(worker, "L3/L2[0]", ce.DEVICE_AICORE)
    host = _record(worker, "L3", ce.HOST_CPU)
    resolver = ce.BackendResolver(worker._get_endpoint_registry(), worker._get_region_access_service())
    for part in (ce.RegionPartKind.PAYLOAD, ce.RegionPartKind.COUNTER):
        offered = resolver._adapter_candidates(  # pyright: ignore[reportPrivateUsage]
            part, ce.BackendKind.VMM_WINDOW, provider, host
        )
        assert [(candidate.kind, candidate.profile) for candidate in offered] == [
            (ce.AdapterKind.OWNER_DELEGATED_COPY, ce.AdapterProfile.HOST_VMM_COPY)
        ]


def test_device_backend_attempts_are_recorded_when_direct_and_copy_are_absent():
    worker = _l3(device_ids=[0])
    plan = _plan(worker, [ce.at("L3", ce.HOST_CPU), ce.at("L3/L2[0]", ce.DEVICE_AICORE)], access={})
    assert isinstance(plan, ce.UnsupportedRegionPlan)
    assert plan.reason is ce.BackendUnsupportedReason.ADAPTER_UNSUPPORTED
    assert [attempt.adapter_profile for attempt in plan.attempted_adapters] == [
        ce.AdapterProfile.HOST_VMM_COPY,
    ]
    assert all(attempt.part is ce.RegionPartKind.PAYLOAD for attempt in plan.attempted_adapters)
    assert all(attempt.backend_kind is ce.BackendKind.VMM_WINDOW for attempt in plan.attempted_adapters)
    assert all(attempt.member.path == "L3" for attempt in plan.attempted_adapters)
    assert plan.message.count("L3 HOST_CPU") == 1


def test_default_service_refuses_host_svm_map_when_asked_directly():
    """`_adapter_candidates` no longer offers the direct map over a VMM backing, but the default
    service's own refusal still has to hold — it is the backstop if some other backing ever
    routes a `HOST_SVM_MAP` candidate here.
    """
    worker = _l3(device_ids=[0])
    provider = _record(worker, "L3/L2[0]", ce.DEVICE_AICORE)
    host = _record(worker, "L3", ce.HOST_CPU)
    query = ce.RegionAccessQuery(
        topology="SingleOwner",
        part=ce.RegionPartKind.COUNTER,
        backend_kind=ce.BackendKind.VMM_WINDOW,
        provider=provider,
        consumer=host,
        layout=ce.RegionLayoutSpec(payload_bytes=64, counter_bytes=8),
        same_node=True,
        platform=worker._config.get("platform"),
        runtime=worker._config.get("runtime"),
    )
    direct_candidate = ce._AdapterCandidate(  # pyright: ignore[reportPrivateUsage]
        ce.AdapterKind.DIRECT_MAP, ce.AdapterProfile.HOST_SVM_MAP
    )
    decision = worker._get_region_access_service().evaluate_region_access(query, direct_candidate)
    assert not decision.supported
    assert decision.diagnostics is not None
    assert decision.diagnostics.reason_code is ce.RegionAccessReasonCode.NO_IMPLEMENTED_DIRECT_MAP_PROBE


def test_consumer_attachment_passes_part_to_candidate_selection():
    worker = _l3(device_ids=[0])
    registry = worker._get_endpoint_registry()
    provider = _record(worker, "L3/L2[0]", ce.DEVICE_AICORE)
    host = _record(worker, "L3", ce.HOST_CPU)
    resolver = ce.BackendResolver(
        registry,
        ce.StaticRegionAccessService(
            {
                _access_key(
                    ce.BackendKind.VMM_WINDOW,
                    ce.RegionPartKind.COUNTER,
                    ce.AdapterKind.OWNER_DELEGATED_COPY,
                    ce.AdapterProfile.HOST_VMM_COPY,
                ): True
            }
        ),
    )
    candidate = resolver._adapter_candidates(  # pyright: ignore[reportPrivateUsage]
        ce.RegionPartKind.COUNTER,
        ce.BackendKind.VMM_WINDOW,
        provider,
        host,
    )[0]
    seen_parts = []

    def candidate_order(part, backend_kind, provider_record, member_record):
        seen_parts.append(part)
        assert backend_kind is ce.BackendKind.VMM_WINDOW
        assert provider_record == provider
        assert member_record == host
        return (candidate,)

    resolver._adapter_candidates = candidate_order  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
    attachment = resolver._consumer_attachment(  # pyright: ignore[reportPrivateUsage]
        ce.RegionPartKind.COUNTER,
        ce.BackendKind.VMM_WINDOW,
        provider,
        host,
        ce.RegionLayoutSpec(payload_bytes=64, counter_bytes=8),
    )
    assert isinstance(attachment, ce.MemberAttachmentPlan)
    assert seen_parts == [ce.RegionPartKind.COUNTER]


def test_host_shm_plan_and_host_provider_device_member_attempts():
    worker = _l4_with_local_l3()
    plan = _plan(
        worker,
        [ce.at("L4", ce.HOST_CPU), ce.at("L4/L3[0]", ce.HOST_CPU)],
        access=_supported_parts(
            ce.BackendKind.POSIX_SHM,
            ce.AdapterKind.DIRECT_MAP,
            ce.AdapterProfile.HOST_SHM_MAP,
        ),
    )
    assert isinstance(plan, ce.BackendPlan)
    child_host = _record(worker, "L4/L3[0]", ce.HOST_CPU)
    attachment = _attachments_by_member(plan.payload)[child_host.identity]
    assert attachment.adapter_kind is ce.AdapterKind.DIRECT_MAP
    assert attachment.adapter_profile is ce.AdapterProfile.HOST_SHM_MAP
    assert plan.payload.backend_kind is ce.BackendKind.POSIX_SHM

    worker = _l3(device_ids=[0])
    plan = _plan(
        worker,
        [ce.at("L3", ce.HOST_CPU), ce.at("L3/L2[0]", ce.DEVICE_AICORE)],
        ce.SingleOwner(provider=ce.at("L3", ce.HOST_CPU)),
    )
    assert isinstance(plan, ce.UnsupportedRegionPlan)
    assert plan.reason is ce.BackendUnsupportedReason.ADAPTER_UNSUPPORTED
    assert [(attempt.adapter_kind, attempt.adapter_profile, attempt.reason) for attempt in plan.attempted_adapters] == [
        (
            ce.AdapterKind.EXPLICIT_TRANSFER,
            ce.AdapterProfile.REMOTE_COPY,
            "explicit transfer materializer is not implemented yet",
        )
    ]


def test_cross_node_resolves_then_reports_adapter_attempts_without_hard_reject():
    worker = _l4_with_remote(RemoteWorkerSpec(endpoint="10.0.0.7:1234", platform="a2a3", device_ids=(6,)))
    resolved = worker._resolve_region_spec(
        [ce.at("L4", ce.HOST_CPU), ce.at("L4/L3[0]/L2[0]", ce.DEVICE_AICORE)],
        ce.SingleOwner(provider=ce.at("L4/L3[0]/L2[0]", ce.DEVICE_AICORE)),
    )
    assert [record.path for record in resolved.members] == ["L4", "L4/L3[0]/L2[0]"]

    plan = _plan(
        worker,
        [ce.at("L4", ce.HOST_CPU), ce.at("L4/L3[0]/L2[0]", ce.DEVICE_AICORE)],
        ce.SingleOwner(provider=ce.at("L4/L3[0]/L2[0]", ce.DEVICE_AICORE)),
    )
    assert isinstance(plan, ce.UnsupportedRegionPlan)
    assert plan.reason is ce.BackendUnsupportedReason.ADAPTER_UNSUPPORTED
    assert not hasattr(ce.BackendUnsupportedReason, "CROSS_NODE_UNSUPPORTED")
    assert [(attempt.adapter_kind, attempt.adapter_profile, attempt.reason) for attempt in plan.attempted_adapters] == [
        (
            ce.AdapterKind.OWNER_DELEGATED_COPY,
            ce.AdapterProfile.REMOTE_COPY,
            "remote copy materializer is not implemented yet",
        ),
        (
            ce.AdapterKind.EXPLICIT_TRANSFER,
            ce.AdapterProfile.REMOTE_COPY,
            "explicit transfer materializer is not implemented yet",
        ),
    ]


def test_cross_node_device_member_attempt_order_includes_fabric_before_remote_copy():
    worker = _l4_with_remote(
        RemoteWorkerSpec(endpoint="10.0.0.7:1234", platform="a2a3", device_ids=(6,)),
        RemoteWorkerSpec(endpoint="10.0.0.8:1234", platform="a2a3", device_ids=(7,)),
    )
    plan = _plan(
        worker,
        [ce.at("L4/L3[0]/L2[0]", ce.DEVICE_AICORE), ce.at("L4/L3[1]/L2[0]", ce.DEVICE_AICORE)],
        ce.SingleOwner(provider=ce.at("L4/L3[0]/L2[0]", ce.DEVICE_AICORE)),
    )
    assert isinstance(plan, ce.UnsupportedRegionPlan)
    assert [attempt.adapter_profile for attempt in plan.attempted_adapters] == [
        ce.AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT,
        ce.AdapterProfile.REMOTE_COPY,
        ce.AdapterProfile.REMOTE_COPY,
    ]
    assert plan.attempted_adapters[0].reason == "device fabric peer import is not available for this endpoint"


def test_worker_region_planning_uses_lease_admission_and_close_invalidates_epoch():
    worker = Worker(level=3, device_ids=[0])
    with pytest.raises(RuntimeError, match="READY"):
        worker._resolve_region_spec([ce.at("L3", ce.HOST_CPU)], ce.SingleOwner())

    _ready(worker)
    registry = worker._get_endpoint_registry()
    record = _record(worker, "L3", ce.HOST_CPU)
    assert registry.registry_epoch == 0
    assert record.identity.registry_epoch == 0
    worker.close()
    assert worker._endpoint_registry is None
    assert worker._region_access_service is None
    assert worker._endpoint_registry_epoch == 1
    with pytest.raises(RuntimeError, match="READY"):
        worker._resolve_region_spec([ce.at("L3", ce.HOST_CPU)], ce.SingleOwner())
