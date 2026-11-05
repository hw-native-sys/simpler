# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Direct MPI session views and registry invalidation boundaries."""

from __future__ import annotations

import dataclasses
import time
from typing import Any, cast
from unittest.mock import Mock

import pytest
import simpler.worker as worker_mod
from _task_interface import _MpiDirectTransportHub
from simpler import comm_endpoints as ce
from simpler.comm_region import MaterializationContext, MaterializationRefusal, _validate_registry_matches_worker
from simpler.mpi_direct_topology import MpiDirectTopology
from test_mpi_direct import _topology_dict


@pytest.fixture
def session():
    worker = worker_mod.Worker(level=4)
    topology = MpiDirectTopology.from_dict(_topology_dict())
    hub = _MpiDirectTransportHub(topology.max_pending_frame_bytes)
    session_id = (1 << 64) - 1
    for spec in topology.executors:
        hub.register_route(spec.worker_id, spec.rank, session_id, spec.comm_profile)
    worker._configure_mpi_direct(topology, session_id, hub)
    worker._worker = cast(Any, Mock())
    worker._activate_mpi_direct_workers(time.monotonic() + 5)
    worker._lifecycle = worker_mod._Lifecycle.READY
    try:
        yield worker, hub
    finally:
        worker._worker = None
        worker.close()
        hub.close()


def test_shared_uint64_session_binds_host_and_device_views(session):
    worker, _hub = session
    registry = worker._get_endpoint_registry()
    views = [registry.mpi_session_for_endpoint(record) for record in registry.records]
    views = [view for view in views if view is not None]
    assert {view.session_id for view in views} == {(1 << 64) - 1}
    assert {view.rank for view in views} == {1, 2}
    assert len(views) == 6
    worker._validate_endpoint_registry(registry, registry.records)
    with pytest.raises(ValueError, match="registry identity"):
        ce.EndpointRegistry(
            root_level=registry.root_level,
            session_instance_id=b"x" * len(registry.session_instance_id),
            registry_epoch=registry.registry_epoch,
            records=registry.records,
            mpi_sessions={
                record.endpoint_id: view
                for record in registry.records
                if (view := registry.mpi_session_for_endpoint(record)) is not None
            },
        )
    snapshot = worker._endpoint_topology_snapshot()
    entries = tuple(
        dataclasses.replace(entry, mpi_session=dataclasses.replace(entry.mpi_session, session_epoch=2))
        if entry.mpi_session is not None and entry.mpi_session.worker_id == 1
        else entry
        for entry in snapshot.entries
    )
    with pytest.raises(ValueError, match="inconsistent epoch"):
        ce.EndpointRegistry.from_snapshot(dataclasses.replace(snapshot, entries=entries), registry_epoch=1)


def test_session_epoch_is_checked_independently_of_directory_version(session):
    worker, _hub = session
    registry = worker._get_endpoint_registry()
    worker._invalidate_endpoint_registry()
    current = worker._get_endpoint_registry()
    assert worker._mpi_direct_session.session_epoch == 1
    with pytest.raises(ce.EndpointResolveError) as directory:
        worker._validate_endpoint_registry(registry, registry.records)
    assert directory.value.reason is ce.EndpointResolveReason.REGISTRY_MISMATCH
    provider = ce.at("L4/L3[0]/L2[0]", ce.DEVICE_AICPU)
    resolved = current.resolve_region_spec(
        [ce.at("L4/L3[0]", ce.HOST_CPU), provider], ce.SingleOwner(provider=provider)
    )
    layout = ce.RegionLayoutSpec(payload_bytes=64, counter_bytes=0)
    plan = worker._plan_current_region(current, resolved, layout)
    assert isinstance(plan, ce.BackendPlan)
    worker._mpi_direct_session.session_epoch += 1
    with pytest.raises(ce.EndpointResolveError) as epoch:
        worker._validate_endpoint_registry(current, current.records)
    assert epoch.value.reason is ce.EndpointResolveReason.MPI_SESSION_UNAVAILABLE
    with pytest.raises(MaterializationRefusal):
        _validate_registry_matches_worker(
            MaterializationContext(worker=worker, registry=current, plan=plan, layout=layout)
        )


@pytest.mark.parametrize("terminal", ["failure", "close", "route"])
def test_observed_native_changes_revoke_cached_views_once(session, terminal):
    worker, hub = session
    registry = worker._get_endpoint_registry()
    state = worker._mpi_direct_session
    assert state is not None
    if terminal == "failure":
        hub.fail("injected progress failure")
    elif terminal == "close":
        hub.close()
    else:
        state.hub = cast(Any, Mock(session_snapshot=lambda: (False, "", [0])))
    current = worker._get_endpoint_registry()
    assert current is not registry and state.session_epoch == 2
    worker._observe_mpi_sessions()
    assert state.session_epoch == 2
    with pytest.raises(ce.EndpointResolveError):
        worker._validate_endpoint_registry(registry, registry.records)
    failed = current.resolve_region_spec([ce.at("L4/L3[0]", ce.HOST_CPU)], ce.SingleOwner())
    with pytest.raises(ce.EndpointResolveError) as unavailable:
        worker._validate_endpoint_registry(current, failed.members)
    assert unavailable.value.reason is ce.EndpointResolveReason.MPI_SESSION_UNAVAILABLE
    if terminal == "route":
        healthy = current.resolve_region_spec([ce.at("L4/L3[1]", ce.HOST_CPU)], ce.SingleOwner())
        worker._validate_endpoint_registry(current, healthy.members)
    with worker._operation_lease("ordinary admission"):
        pass


def test_late_snapshot_cannot_overwrite_retirement_or_replacement(session):
    worker, _hub = session
    state = worker._mpi_direct_session
    assert state is not None

    def retire_while_observing():
        with worker._hierarchical_start_cv:
            worker._retire_mpi_direct_session_locked()
        return False, "", []

    state.hub = cast(Any, Mock(session_snapshot=retire_while_observing))
    worker._observe_mpi_sessions()
    assert state.retired and not state.active and state.session_epoch == 2
    with worker._hierarchical_start_cv:
        worker._retire_mpi_direct_session_locked()
    assert state.session_epoch == 2
    replacement = dataclasses.replace(state, retired=False, active=True, session_epoch=3)
    state.retired = False

    def replace_while_observing():
        worker._mpi_direct_session = replacement
        return False, "old failure", []

    state.hub = cast(Any, Mock(session_snapshot=replace_while_observing))
    worker._observe_mpi_sessions()
    assert worker._mpi_direct_session is replacement
    assert replacement.session_epoch == 3 and not replacement.failure and replacement.active


def test_attach_failure_retires_unpublished_session(session):
    worker, _hub = session
    state = worker._mpi_direct_session
    assert state is not None
    state.active = False
    worker._worker = cast(Any, Mock())
    worker._worker.add_remote_l3_mpi.side_effect = [None, RuntimeError("attach failed")]
    with pytest.raises(RuntimeError, match="attach failed"):
        worker._activate_mpi_direct_workers(time.monotonic() + 5)
    assert state.retired and not state.active and state.session_epoch == 2
