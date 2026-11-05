# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Selected-domain metadata from real sim owners; no chip callable executes.

Allocation, import, provenance registration, callback scopes and cleanup are
production paths. Only the final submission sink is intercepted when checking
group metadata, so ownership is never supplied by a fabricated registry row.
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
    GlobalDomainReleaseCommand,
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
        assert owner._domain_uses_for_dispatch(args, 0) == [(1, domain.allocation_id, domain[0].device_ctx)]
        assert resources.live_domains[domain.name] is domain
        assert resources.requires_ordered_cleanup

    _run(owner, inspect)


def test_scalar_only_device_context_records_the_same_live_owner(owner):
    domain = _allocate(owner, "scalar")

    def inspect(_orch):
        args = TaskArgs()
        args.add_scalar(domain.device_context(0))
        assert args.tensor_count() == 0
        assert owner._domain_uses_for_dispatch(args, 0) == [(1, domain.allocation_id, domain[0].device_ctx)]
        assert owner._building_run_resources.live_domains[domain.name] is domain

    _run(owner, inspect)


def test_wrong_worker_and_an_unselected_live_domain_do_not_gain_authority(owner):
    selected = _allocate(owner, "selected", (0,))
    unselected = _allocate(owner, "not-used", (1,))
    args = _tensor_args(selected[0])

    def inspect(_orch):
        assert owner._domain_uses_for_dispatch(TaskArgs(), 1) == []
        assert owner._domain_uses_for_dispatch(args, 0) == [(1, selected.allocation_id, selected[0].device_ctx)]
        assert owner._domain_uses_for_dispatch(args, 1) == []
        assert unselected.name not in owner._building_run_resources.live_domains
        with owner._child_prov_lock, pytest.raises(ValueError, match="wrong worker, or stale"):
            owner._child_prov_check_dispatch_locked(args, 1, api="async-domain-test")

    _run(owner, inspect)


def test_a_live_domain_outside_a_building_run_is_not_a_selected_run_owner(owner):
    domain = _allocate(owner, "outside")
    assert owner._domain_uses_for_dispatch(_tensor_args(domain[0]), 0) == []
    assert not domain.released


def test_group_submission_preserves_each_members_actual_domain_context(owner, monkeypatch):
    domain = _allocate(owner, "group", (0, 1))
    unselected = _allocate(owner, "group-not-used", (0,))
    calls = []

    class SubmissionSink:
        def submit_next_level_group(self, *arguments):
            calls.append(arguments)
            return "captured"

    # Callable lookup and final task execution are outside this metadata test.
    # Real provenance checks and _domain_uses_for_dispatch remain in the path.
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
        assert calls[0][-1] == [
            [(1, domain.allocation_id, domain[1].device_ctx)],
            [(1, domain.allocation_id, domain[0].device_ctx)],
        ]
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
            args = TaskArgs()
            args.add_scalar(borrowed.device_context(0))
        else:
            args = _tensor_args(state.view[0])
        assert owner._domain_uses_for_dispatch(args, 0) == [(2, reference.domain_id, state.contexts[0].device_ctx)]
        resources = owner._building_run_resources
        assert reference.domain_id in resources.borrowed_domain_uses
        assert not resources.live_domains
        assert not resources.pending_release_domains

    _run(owner, inspect)
    assert owner._global_node_domains[reference.domain_id] is state
    assert state.phase is GlobalDomainPhase.COMMIT


@pytest.mark.parametrize("replacement", [False, True], ids=["released", "replaced"])
def test_a_selected_borrowed_owner_cannot_be_reused_after_release_or_replacement(owner, replacement):
    reference = _import_global(owner)

    def inspect(orch):
        borrowed = orch.get_domain(reference)
        assert owner._domain_uses_for_dispatch(TaskArgs(), 0)
        owner._release_global_domain_node(GlobalDomainReleaseCommand(reference.domain_id, reference.generation))
        assert borrowed.released
        if replacement:
            _import_global(owner, reference.domain_id, reference.generation + 1)
        message = "owner was replaced" if replacement else "released before dispatch"
        with pytest.raises(RuntimeError, match=message):
            owner._domain_uses_for_dispatch(TaskArgs(), 0)

    _run(owner, inspect)
