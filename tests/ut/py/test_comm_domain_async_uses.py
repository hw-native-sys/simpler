# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Existing communication owners remain retained without native domain metadata.

Allocation, imports and run ownership use production paths. Group submission
intercepts only the final sink; no communication kernel executes here.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import simpler.orchestrator as orch_module
from simpler.comm_domain import CommDomainRef
from simpler.global_comm_domain import (
    GlobalDomainBuffer,
    GlobalDomainCommand,
    GlobalDomainMember,
    GlobalDomainPhase,
)
from simpler.orchestrator import Orchestrator
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs
from simpler.worker import Worker

pytestmark = pytest.mark.timeout(30)


@pytest.fixture
def owner():
    worker = Worker(
        level=3,
        platform="a2a3sim",
        runtime="tensormap_and_ringbuffer",
        device_ids=[0, 1],
        num_sub_workers=0,
        comm_profile="sim",
        startup_timeout_s=5,
        py_control_timeout_s=5,
    )
    try:
        worker.init()
        yield worker
    finally:
        worker.close()


def _allocate(owner, name, members=(0,)):
    created = []
    owner.run(
        lambda orch, _args, _cfg: created.append(
            orch.allocate_domain(
                name=name,
                members=list(members),
                window_size=4096,
                buffers=[CommBufferSpec("payload", "uint8", 64, 64)],
                retain_after_run=True,
            )
        )
    )
    return created[0]


def _tensor_args(context):
    args = TaskArgs()
    args.add_tensor(context.buffers["payload"].tensor([64], DataType.UINT8))
    return args


def _run(owner, action):
    owner.run(lambda orch, _args, _cfg: action(orch))


def _import_global(owner, domain_id=901, generation=1):
    """Exercise the receiving node's real prepare/import/commit control path."""
    command = GlobalDomainCommand(
        phase=GlobalDomainPhase.PREPARE_EXPORT,
        domain_id=domain_id,
        generation=generation,
        name="imported-metadata",
        profile="sim",
        window_size=4096,
        members=(GlobalDomainMember(0, 0, 0, 0),),
        buffers=(GlobalDomainBuffer("payload", 64, "uint8", 64),),
        owner_instance_id=owner._owner_instance_id,
    )
    descriptors = owner._prepare_global_domain_node(command, 0)
    imported = replace(command, phase=GlobalDomainPhase.IMPORT, descriptors=descriptors)
    owner._import_global_domain_node(imported, 0)
    owner._commit_global_domain_node(replace(imported, phase=GlobalDomainPhase.COMMIT))
    return CommDomainRef(owner._owner_instance_id, domain_id, generation)


def test_canonical_tensor_enrolls_its_existing_domain_in_the_current_run(owner):
    domain = _allocate(owner, "selected")
    args = _tensor_args(domain[0])

    def inspect(_orch):
        resources = owner._building_run_resources
        assert resources.live_domains == {}
        owner._record_touched_identities(args)
        assert resources.live_domains[domain.name] is domain
        assert resources.requires_ordered_cleanup

    _run(owner, inspect)


def test_scalar_only_device_context_records_the_same_live_owner(owner):
    domain = _allocate(owner, "scalar")

    def inspect(_orch):
        args = TaskArgs()
        args.add_scalar(domain.device_context(0))
        assert args.tensor_count() == 0
        owner._record_touched_identities(args)
        assert owner._building_run_resources.live_domains[domain.name] is domain

    _run(owner, inspect)


@pytest.mark.parametrize("source", ["tensor", "scalar"])
def test_first_local_domain_zero_is_retained_without_native_metadata(source):
    worker = Worker(
        level=3,
        platform="a2a3sim",
        runtime="tensormap_and_ringbuffer",
        device_ids=[0],
        num_sub_workers=0,
        startup_timeout_s=5,
        py_control_timeout_s=5,
    )
    try:
        worker.init()
        domain = _allocate(worker, "first-local")
        assert domain.allocation_id == 0

        def inspect(_orch):
            if source == "tensor":
                args = _tensor_args(domain[0])
            else:
                args = TaskArgs()
                args.add_scalar(domain.device_context(0))
            worker._record_touched_identities(args)
            resources = worker._building_run_resources
            assert resources is not None
            assert resources.live_domains[domain.name] is domain

        _run(worker, inspect)
    finally:
        worker.close()


def test_wrong_worker_and_an_unselected_live_domain_do_not_gain_authority(owner):
    selected = _allocate(owner, "selected", (0,))
    unselected = _allocate(owner, "not-used", (1,))
    args = _tensor_args(selected[0])

    def inspect(_orch):
        owner._record_touched_identities(args)
        assert owner._building_run_resources.live_domains[selected.name] is selected
        assert unselected.name not in owner._building_run_resources.live_domains
        with owner._child_prov_lock, pytest.raises(ValueError, match="wrong worker, or stale"):
            owner._child_prov_check_dispatch_locked(args, 1, api="async-domain-test")

    _run(owner, inspect)


def test_a_live_domain_outside_a_building_run_is_not_a_selected_run_owner(owner):
    domain = _allocate(owner, "outside")
    owner._record_touched_identities(_tensor_args(domain[0]))
    assert owner._building_run_resources is None
    assert not domain.released


def test_group_submission_retains_the_existing_owner_without_a_domain_sidecar(owner, monkeypatch):
    domain = _allocate(owner, "group", (0, 1))
    unselected = _allocate(owner, "group-not-used", (0,))
    calls = []

    class SubmissionSink:
        def submit_next_level_group(self, *arguments):
            calls.append(arguments)
            return "captured"

    # Only callable lookup and final task execution are intercepted.
    # Production provenance checks and owner enrollment remain in the path.
    monkeypatch.setattr(
        orch_module, "_require_handle", lambda *_args, **_kw: (bytes(32), "CHIP_CALLABLE", "LOCAL_CHIP", (0, 1))
    )

    def inspect(_orch):
        dispatch = Orchestrator(SubmissionSink(), owner)
        assert (
            dispatch.submit_next_level_group(
                object(), [_tensor_args(domain[1]), _tensor_args(domain[0])], workers=[1, 0]
            )
            == "captured"
        )
        assert len(calls[0]) == 8
        assert owner._building_run_resources.live_domains[domain.name] is domain
        assert unselected.name not in owner._building_run_resources.live_domains

    _run(owner, inspect)


@pytest.mark.parametrize("source", ["tensor", "scalar"])
def test_imported_global_domain_uses_its_committed_owner_without_a_release_duty(owner, source):
    reference = _import_global(owner)
    state = owner._global_node_domains[reference.domain_id]

    def inspect(orch):
        if source == "scalar":
            borrowed = orch.get_domain(reference)
            assert borrowed.borrowed
            with pytest.raises(RuntimeError, match="borrowed domain handles cannot release"):
                borrowed.release()
            args = TaskArgs()
            args.add_scalar(borrowed.device_context(0))
        else:
            args = _tensor_args(state.view[0])
        owner._record_touched_identities(args)
        resources = owner._building_run_resources
        assert owner._global_node_domains[reference.domain_id] is state
        assert state.phase is GlobalDomainPhase.COMMIT
        assert not resources.live_domains
        assert not resources.pending_release_domains

    _run(owner, inspect)
    assert owner._global_node_domains[reference.domain_id] is state
    assert state.phase is GlobalDomainPhase.COMMIT
