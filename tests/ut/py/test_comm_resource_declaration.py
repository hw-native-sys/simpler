# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit communication access contracts at the orchestration boundary."""

import pytest
from simpler.buffer import AddressSpace
from simpler.comm_domain import DomainConfig
from simpler.orchestrator import Orchestrator, _callback_run
from simpler.task_interface import CommDomainHandle
from simpler.worker import Worker, _RunResources


@pytest.fixture
def declaration():
    worker = Worker(level=3, platform="a2a3", runtime="tensormap_and_ringbuffer", device_ids=[0, 1])
    resources = _RunResources()
    worker._building_run_resources = resources
    domain = CommDomainHandle(
        name="owned", workers=(0, 1), allocation_id=0, retain_after_run=True, _release_fn=lambda _h: None
    )
    domain._config = DomainConfig.capture((0, 1), 64, (), True, ())
    worker._live_domains[domain.name] = domain
    try:
        yield worker, Orchestrator(None, worker), resources, domain
    finally:
        worker._building_run_resources = None
        worker._live_domains.clear()


def test_declaration_requires_current_worker_callback(declaration):
    worker, orch, _resources, domain = declaration
    with pytest.raises(RuntimeError, match="callback"):
        orch.declare_comm_domains([domain])
    with _callback_run(1, object()), pytest.raises(RuntimeError, match="callback"):
        orch.declare_comm_domains([domain])


def test_declaration_retains_scalar_only_domain_until_run_fence(declaration):
    worker, orch, resources, domain = declaration
    with _callback_run(1, worker):
        orch.declare_comm_domains([domain])
        orch.declare_comm_domains([])
    assert resources.declared_comm_domains == {0: domain}
    assert domain._run_uses[id(resources)] is resources
    assert resources.live_domains[domain.name] is domain


def test_declaration_rejects_foreign_domain_without_claiming_it(declaration):
    worker, orch, resources, _domain = declaration
    foreign = CommDomainHandle(name="other", workers=(0, 1), retain_after_run=True, _release_fn=lambda _h: None)
    with _callback_run(1, worker), pytest.raises(ValueError, match="belong"):
        orch.declare_comm_domains([foreign])
    assert resources.declared_comm_domains is None
    assert not foreign._run_uses


@pytest.mark.parametrize("unsupported", ["host", "workspace", "nonretained", "released"])
def test_declaration_rejects_unsupported_domain(declaration, unsupported):
    worker, orch, resources, domain = declaration
    if unsupported == "host":
        domain._config = DomainConfig.capture((0, 1), 64, (), True, (), memory_source=AddressSpace.HOST)
    elif unsupported == "workspace":
        domain._config = DomainConfig.capture((0, 1), 64, (), True, (), workspace_engines=("sdma",))
    elif unsupported == "nonretained":
        domain.retain_after_run = False
    else:
        domain._released = True
    with _callback_run(1, worker), pytest.raises((ValueError, RuntimeError)):
        orch.declare_comm_domains([domain])
    assert resources.declared_comm_domains is None
