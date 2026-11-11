# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import threading
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import simpler.worker as worker_mod
from simpler.ccu import ChipCcuBinding
from simpler.task_interface import ChipCallable, CoreCallable
from simpler.worker import Worker


@pytest.mark.parametrize("index", [0, 1])
def test_extracted_child_repackages_without_padding_or_ccu_package(index):
    children = [(i, CoreCallable.build(signature=[], binary=bytes([i + 1]))) for i in range(2)]
    parent = ChipCallable.build(
        signature=[], func_name="parent", binary=b"orchestration", children=children, ccu_modules=[b"module"]
    )
    extracted = parent.child(index)
    rebuilt = ChipCallable.build(signature=[], func_name="rebuilt", binary=b"new", children=[(0, extracted)])
    assert rebuilt.child_count == 1
    assert rebuilt.child(0).binary_size == 1


def test_prepare_successor_preserves_active_generation_and_launch_waits_for_stop():
    operations = []
    native = SimpleNamespace(
        prepare_ccu_run=lambda comm, registration, gen, dispatch, allocation: operations.append(
            ("prepare", gen, dispatch)
        ),
        control_ccu_run=lambda comm, registration, op, gen: operations.append((op, gen)) or 0,
    )
    binding = ChipCcuBinding(native, 10, 9, 1)
    first = binding.prepare(100, 7)
    first.begin()
    second = binding.prepare(101, 7)
    assert operations == [("prepare", 1, 100), (1, 1), ("prepare", 2, 101)]
    with pytest.raises(RuntimeError, match="confirmed stop"):
        second.begin()
    assert operations[-1] == ("prepare", 2, 101)
    first.finish()
    second.begin()
    second.finish()
    assert operations[-5:] == [(3, 1), (7, 1), (1, 2), (3, 2), (7, 2)]


def test_unconfirmed_stop_blocks_prepared_successor():
    def control(comm, registration, op, gen):
        if op == 3:
            raise RuntimeError("stop unconfirmed")
        return 0

    binding = ChipCcuBinding(SimpleNamespace(prepare_ccu_run=lambda *args: 0, control_ccu_run=control), 10, 9, 1)
    first, second = binding.prepare(100, 7), binding.prepare(101, 7)
    first.begin()
    with pytest.raises(RuntimeError, match="stop unconfirmed"):
        first.finish()
    with pytest.raises(RuntimeError, match="confirmed stop"):
        second.begin()


def test_exit_timeout_aborts_without_discarding_or_admitting_successor(monkeypatch):
    operations = []

    def control(_comm, _registration, operation, generation):
        operations.append((operation, generation))
        return int(operation == 3)

    times = iter((0.0, 0.5, 1.0))
    monkeypatch.setattr("simpler.ccu.time", SimpleNamespace(monotonic=lambda: next(times)))
    binding = ChipCcuBinding(SimpleNamespace(prepare_ccu_run=lambda *_: 0, control_ccu_run=control), 10, 9, 1)
    first, successor = binding.prepare(100, 7), binding.prepare(101, 7)
    first.begin()

    with pytest.raises(TimeoutError, match="CCU exit protocol timed out.*generation 1"):
        first.finish()
    assert operations == [(1, 1), (3, 1), (3, 1), (4, 1)]
    assert binding.active
    with pytest.raises(RuntimeError, match="confirmed stop"):
        successor.begin()


def _ccu_worker(domains):
    resources = worker_mod._RunResources(live_domains={domain.name: domain for domain in domains})
    observed = []
    state = SimpleNamespace(target=SimpleNamespace(ccu_module_count=1), eligible_worker_ids=(0, 1, 2))

    class Native:
        def control_payload(self, _kind, chip, _command, payload, _timeout):
            _, _, operation, digest, _ = worker_mod._CCU_DOMAIN_REQUEST.unpack_from(payload)
            ranks = tuple(range(len(worker._config["device_ids"])))
            if operation == 5:
                tail = worker_mod._CCU_DOMAIN_REQUEST.size
                _, count, _ = worker_mod.struct.unpack_from("<QII", payload, tail)
                ranks = worker_mod.struct.unpack_from(f"<{count}I", payload, tail + 16)
            elif operation == 6:
                ranks = next(iter(worker._building_run_resources.live_domains.values())).workers
            if chip == ranks[0]:
                observed.append((ranks, operation, digest))
            reply = bytearray(payload)
            worker_mod.struct.pack_into("<i", reply, worker_mod._CCU_DOMAIN_REQUEST.size - 4, chip + 7)
            return reply

    worker: Any = SimpleNamespace(
        level=3,
        _config={"platform": "a5", "device_ids": [3, 4, 5]},
        _next_level_worker_ids=[],
        _building_run_resources=resources,
        _ccu_run_registrations={},
        _ccu_context_registrations={},
        _ccu_registration_collectives={},
        _worker=Native(),
        _alloc_id_lock=threading.Lock(),
        _next_alloc_id=0,
        _py_control_timeout_s=1,
        _record_touched_identities=lambda _: None,
        _enroll_domain_uses=lambda _: None,
        _resolve_handle=lambda *args, **kwargs: state,
        _control_admission=lambda _: nullcontext(),
        _ensure_comm_base=lambda: None,
    )
    worker._ccu_registration_collective = lambda digest: Worker._ccu_registration_collective(worker, digest)
    worker._ccu_domain_collective = lambda *args: Worker._ccu_domain_collective(worker, *args)
    return worker, observed


def _domain(allocation=7, engines=("ccu",)) -> Any:
    return SimpleNamespace(
        name=f"w-{allocation}",
        allocation_id=allocation,
        workers=(2, 0),
        released=False,
        _config=SimpleNamespace(workspace_engines=engines),
        contexts={2: SimpleNamespace(device_ctx=2000 + allocation), 0: SimpleNamespace(device_ctx=1000 + allocation)},
    )


def _args(context):
    return SimpleNamespace(scalar_count=lambda: 1, scalar=lambda index: context)


@pytest.mark.parametrize("domains", [[], [_domain(engines=())]])
def test_unenabled_ccu_rejected_before_registration(domains):
    worker, observed = _ccu_worker(domains)
    with pytest.raises(ValueError, match="CCU-enabled"):
        Worker._prepare_ccu_submission(worker, SimpleNamespace(digest=b"a" * 32), [2], [_args(2007)])
    assert observed == []


def test_distinct_windows_share_full_context_registration():
    domain = _domain()
    worker, observed = _ccu_worker([domain])
    target = SimpleNamespace(digest=b"a" * 32)
    Worker._ensure_ccu_registration(worker, target.digest)
    Worker._prepare_ccu_submission(worker, target, [2], [_args(2007)])
    Worker._prepare_ccu_submission(worker, target, [0], [_args(1007)])
    second = _domain(8)
    worker._building_run_resources = worker_mod._RunResources(live_domains={second.name: second})
    Worker._prepare_ccu_submission(worker, target, [2], [_args(2008)])
    registrations = [entry for entry in observed if entry[1] == 5]
    assert registrations == [((0, 1, 2), 5, target.digest)]


def test_context_recreation_and_distinct_callable_do_not_reuse_old_registration():
    domain = _domain()
    worker, observed = _ccu_worker([domain])
    first, second = b"a" * 32, b"b" * 32
    Worker._ensure_ccu_registration(worker, first)
    Worker._ensure_ccu_registration(worker, second)
    worker._worker = type(worker._worker)()
    Worker._ensure_ccu_registration(worker, first)
    assert [entry[2] for entry in observed if entry[1] == 5] == [first, second, first]


def test_tensor_domains_are_enrolled_before_context_selection():
    domain = _domain()
    worker, observed = _ccu_worker([])
    order = []
    worker._record_touched_identities = lambda args: order.append("record")

    def enroll(resources):
        assert order == ["record"]
        resources.live_domains[domain.name] = domain
        order.append("enroll")

    worker._enroll_domain_uses = enroll
    Worker._ensure_ccu_registration(worker, b"a" * 32)
    Worker._prepare_ccu_submission(worker, SimpleNamespace(digest=b"a" * 32), [2], [_args(2007)])
    assert order == ["record", "enroll"]
    assert worker._building_run_resources.ccu_domains[domain.allocation_id] is domain
    assert len([entry for entry in observed if entry[1] == 5]) == 1


def test_submission_requires_selected_context_and_one_ccu_domain_per_run():
    first, second = _domain(), _domain(8)
    worker, observed = _ccu_worker([first, second])
    target = SimpleNamespace(digest=b"a" * 32)
    Worker._ensure_ccu_registration(worker, target.digest)
    with pytest.raises(ValueError, match="context"):
        Worker._prepare_ccu_submission(worker, target, [2], [_args(0)])
    assert observed == [((0, 1, 2), 5, target.digest)]
    Worker._prepare_ccu_submission(worker, target, [2], [_args(2007)])
    with pytest.raises(ValueError, match="one CCU domain"):
        Worker._prepare_ccu_submission(worker, target, [2], [_args(2008)])
    assert len(observed) == 2


def test_domain_allocation_does_not_register_ccu_callable():
    domain = _domain()
    worker, observed = _ccu_worker([domain])
    assert observed == []
    assert worker._ccu_context_registrations == {}


def test_submit_requires_explicit_ccu_registration():
    domain = _domain()
    worker, observed = _ccu_worker([domain])
    with pytest.raises(ValueError, match="explicitly registered"):
        Worker._prepare_ccu_submission(worker, SimpleNamespace(digest=b"a" * 32), [2], [_args(2007)])
    assert observed == []


def test_ccu_submission_rejects_callable_missing_a_context_chip():
    worker, observed = _ccu_worker([_domain()])
    worker._resolve_handle = lambda *args, **kwargs: SimpleNamespace(
        target=SimpleNamespace(ccu_module_count=1), eligible_worker_ids=(0, 2)
    )
    with pytest.raises(ValueError, match="every L3 chip"):
        Worker._prepare_ccu_submission(worker, SimpleNamespace(digest=b"a" * 32), [2], [_args(2007)])
    assert observed == []
