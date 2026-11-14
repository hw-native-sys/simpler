# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit per-rank domain binding admission without a device."""

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast

import pytest
import simpler.worker as worker_module
from simpler.ccu import CcuRegistrationHandle, ChipCcuRunSet
from simpler.task_interface import CommDomainHandle, TaskArgs
from simpler.worker import CallableHandle, Worker, _RunResources


def worker():
    owner = SimpleNamespace(
        level=3,
        _config={"platform": "a5"},
        _building_run_resources=_RunResources(),
        _resolve_handle=lambda *a, **k: SimpleNamespace(
            target=SimpleNamespace(ccu_module_count=1), eligible_worker_ids=(0, 2)
        ),
        _use_domain=lambda domain: None,
        _record_touched_identities=lambda args: None,
        _enroll_domain_uses=lambda resources: None,
        _control_admission=lambda api: nullcontext(),
        _operation_lease=lambda *a, **k: nullcontext(),
    )
    return owner


def domain(allocation=7, members=(2, 0)):
    result = CommDomainHandle(name="domain", workers=members, allocation_id=allocation, _release_fn=lambda h: None)
    result._config = SimpleNamespace(workspace_engines=("ccu",))
    return result


def registration(owner, dom, ccu_id=0, native_id=10, worker_id=2):
    return CcuRegistrationHandle(
        dom,
        owner,
        SimpleNamespace(digest=b"a" * 32),
        ccu_id,
        worker_id,
        native_id + worker_id,
        dom.allocation_id,
        dom.generation,
    )


def test_multidomain_bindings_ignore_repeated_scalar_contexts_and_preserve_arguments():
    owner = worker()
    first, second = domain(), domain(8, (0, 2))
    bindings = (registration(owner, first), registration(owner, second, native_id=20))
    args = TaskArgs()
    args.add_scalar(99)
    args.add_scalar(99)
    actual = Worker._prepare_ccu_submission(cast(Worker, owner), object(), [2], [args], ccu_bindings=bindings)[0]
    assert actual.ccu_bindings() == [(7, 12, 0), (8, 22, 0)]
    assert actual.scalar_count() == 2
    assert actual.scalar(0) == actual.scalar(1) == 99
    assert args.ccu_bindings() == []
    assert set(owner._building_run_resources.ccu_domains) == {7, 8}


def test_rank_plan_can_use_a_subset_without_launching_other_members():
    owner = worker()
    binding = registration(owner, domain())
    actual = Worker._prepare_ccu_submission(
        cast(Worker, owner), object(), [2, 0], [TaskArgs(), TaskArgs()], ccu_bindings={2: (binding, binding), 0: ()}
    )
    assert actual[0].ccu_bindings() == [(7, 12, 0)]
    assert actual[1].ccu_bindings() == []


@pytest.mark.parametrize("failure", ["released", "owner", "generation", "rank", "other_member"])
def test_invalid_binding_is_rejected_before_encoding(failure):
    owner = worker()
    dom = domain()
    binding = registration(owner, dom)
    chip = 2
    if failure == "released":
        dom._released = True
    elif failure == "owner":
        owner = worker()
    elif failure == "generation":
        dom.generation += 1
    elif failure == "rank":
        chip = 1
    else:
        chip = 0
    with pytest.raises((ValueError, RuntimeError)):
        Worker._prepare_ccu_submission(cast(Worker, owner), object(), [chip], [TaskArgs()], ccu_bindings=(binding,))
    assert owner._building_run_resources.ccu_domains == {}


def test_missing_explicit_binding_is_not_guessed_from_scalars():
    owner = worker()
    with pytest.raises(ValueError, match="explicit ccu_bindings"):
        Worker._prepare_ccu_submission(cast(Worker, owner), object(), [2], [TaskArgs()])


def test_registration_cache_is_owned_by_domain_worker_and_definition():
    owner = worker()
    observed = []
    owner._register_domain_ccu = lambda dom, digest, ccu_id, worker_id: (
        observed.append((dom.allocation_id, digest, ccu_id, worker_id)) or 10 + worker_id
    )
    first, recreated = domain(), domain(8)
    target = cast(CallableHandle, SimpleNamespace(digest=b"a" * 32))
    initial = Worker._get_or_register_domain_ccu(cast(Worker, owner), first, target, ccu_id=0, worker=2)
    assert Worker._get_or_register_domain_ccu(cast(Worker, owner), first, target, ccu_id=0, worker=2) is initial
    assert Worker._get_or_register_domain_ccu(cast(Worker, owner), first, target, ccu_id=0, worker=0) is not initial
    assert Worker._get_or_register_domain_ccu(cast(Worker, owner), first, target, ccu_id=1, worker=2) is not initial
    assert Worker._get_or_register_domain_ccu(cast(Worker, owner), recreated, target, ccu_id=0, worker=2) is not initial
    assert len(observed) == 4
    assert initial.worker == 2
    assert initial.registration_id == 12


def test_run_set_does_not_discard_when_stop_is_unconfirmed():
    calls = []

    def control(comm, run_set, op):
        calls.append(op)
        if op == 9:
            raise RuntimeError("stop unconfirmed")
        return int(op == 3)

    native = SimpleNamespace(prepare_ccu_run_set=lambda *a: 8, control_ccu_run_set=control)
    run = ChipCcuRunSet(native, 1, [10, 20], 3, 3, natural_exit_wait=0)
    run.begin()
    with pytest.raises(RuntimeError, match="stop unconfirmed"):
        run.finish()
    assert calls == [1, 3, 9]
    assert run.started


def test_concurrent_registration_publishes_one_complete_handle():
    owner = worker()
    dom = domain()
    entered, release = threading.Event(), threading.Event()
    attempts = []

    def register(_domain, _digest, _ccu_id, _worker):
        attempts.append(1)
        entered.set()
        assert release.wait(5), "registration release was not acknowledged"
        return 12

    owner._register_domain_ccu = register
    dom._ccu_register_fn = lambda *a, **k: Worker._get_or_register_domain_ccu(cast(Worker, owner), *a, **k)
    target = cast(CallableHandle, SimpleNamespace(digest=b"a" * 32))
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(dom.get_or_register_ccu, target, ccu_id=0, worker=2)
        assert entered.wait(5), "first registration did not enter native operation"
        second = executor.submit(dom.get_or_register_ccu, target, ccu_id=0, worker=2)
        assert not dom._ccu_registrations
        release.set()
        assert first.result() is second.result()
    assert attempts == [1]


def attach_native_registration(owner, control):
    owner._worker = SimpleNamespace(control_payload=control)
    owner._ensure_comm_base = lambda: None
    owner._alloc_id_lock = threading.Lock()
    owner._next_alloc_id = 100
    owner._py_control_timeout_s = 5
    owner._register_domain_ccu = lambda *a: Worker._register_domain_ccu(cast(Worker, owner), *a)


def test_registration_contacts_only_target_and_preserves_full_domain_topology():
    owner = worker()
    dom = domain()
    observed = []

    def control(_kind, rank, _command, payload, _timeout):
        tail = worker_module._CCU_DOMAIN_REQUEST.size
        context_id, count, logical_rank, ccu_id = worker_module.struct.unpack_from("<QIII", payload, tail)
        members = worker_module.struct.unpack_from(f"<{count}I", payload, tail + 20)
        observed.append((rank, context_id, members, logical_rank, ccu_id))
        reply = bytearray(payload)
        worker_module.struct.pack_into("<i", reply, tail - 4, 12)
        return reply

    attach_native_registration(owner, control)
    result = Worker._register_domain_ccu(cast(Worker, owner), dom, b"a" * 32, 0, 0)
    assert result == 12
    assert observed == [(0, 100, (2, 0), 1, 0)]


def test_unknown_registration_result_revokes_domain_reuse():
    owner = worker()
    dom = domain()
    recipients = []

    def control(_kind, rank, _command, _payload, _timeout):
        recipients.append(rank)
        raise RuntimeError("registration outcome unknown")

    attach_native_registration(owner, control)
    with pytest.raises(RuntimeError, match="outcome unknown"):
        Worker._register_domain_ccu(cast(Worker, owner), dom, b"a" * 32, 0, 0)
    assert recipients == [0]
    assert dom._ccu_registration_failed
    with pytest.raises(RuntimeError, match="cannot be reused"):
        Worker._get_or_register_domain_ccu(
            cast(Worker, owner), dom, SimpleNamespace(digest=b"a" * 32), ccu_id=0, worker=0
        )


def test_run_set_accepts_natural_exit_without_forcing_stop():
    calls = []
    native = SimpleNamespace(
        prepare_ccu_run_set=lambda *a: 8,
        control_ccu_run_set=lambda comm, run, op: calls.append(op) or 0,
    )
    run = ChipCcuRunSet(native, 1, [10, 20], 3, 3)
    run.begin()
    run.finish()
    assert calls == [1, 3, 7]
    assert not run.started


def test_run_set_bounded_natural_wait_then_confirmed_resident_stop(monkeypatch):
    calls = []
    clock = iter((10.0, 10.05, 10.1))
    monkeypatch.setattr("simpler.ccu.time.monotonic", lambda: next(clock))
    native = SimpleNamespace(
        prepare_ccu_run_set=lambda *a: 8,
        control_ccu_run_set=lambda comm, run, op: calls.append(op) or int(op == 3),
    )
    run = ChipCcuRunSet(native, 1, [10, 20], 3, 3)
    run.begin()
    run.finish()
    assert calls == [1, 3, 3, 9, 7]
    assert not run.started


@pytest.mark.parametrize("ccu_id", [-1, 2**32 - 1, 2**32, True, 0.5])
def test_registration_rejects_invalid_definition_ids_before_native_control(ccu_id):
    owner = worker()
    dom = domain()
    with pytest.raises(ValueError, match="ccu_id"):
        Worker._get_or_register_domain_ccu(
            cast(Worker, owner), dom, SimpleNamespace(digest=b"a" * 32), ccu_id=ccu_id, worker=2
        )
    assert not dom._ccu_registrations


def test_clean_registration_failure_can_retry_without_contacting_other_ranks():
    owner = worker()
    dom = domain()
    recipients = []
    reject = True

    def control(_kind, rank, _command, payload, _timeout):
        recipients.append(rank)
        reply = bytearray(payload)
        result = -1 if reject else 10 + rank
        worker_module.struct.pack_into("<i", reply, worker_module._CCU_DOMAIN_REQUEST.size - 4, result)
        return reply

    attach_native_registration(owner, control)
    dom._ccu_register_fn = lambda *a, **k: Worker._get_or_register_domain_ccu(cast(Worker, owner), *a, **k)
    target = cast(CallableHandle, SimpleNamespace(digest=b"a" * 32))
    with pytest.raises(RuntimeError, match="cleanup confirmed"):
        dom.get_or_register_ccu(target, ccu_id=0, worker=0)
    assert not dom._ccu_registration_failed
    assert not dom._ccu_registrations
    reject = False
    handle = dom.get_or_register_ccu(target, ccu_id=0, worker=0)
    assert handle.worker == 0 and handle.registration_id == 10
    assert dom.get_or_register_ccu(target, ccu_id=0, worker=0) is handle
    assert recipients == [0, 0]


@pytest.mark.parametrize("target_worker", [1, -1, True, 0.5, "0"])
def test_invalid_registration_worker_rejected_before_native_control(target_worker):
    owner = worker()
    dom = domain()
    with pytest.raises(ValueError, match="worker"):
        Worker._get_or_register_domain_ccu(
            cast(Worker, owner), dom, SimpleNamespace(digest=b"a" * 32), ccu_id=0, worker=target_worker
        )
    assert not dom._ccu_registrations


def test_callable_only_needs_to_be_available_on_registration_target():
    owner = worker()
    owner._resolve_handle = lambda *a, **k: SimpleNamespace(
        target=SimpleNamespace(ccu_module_count=1), eligible_worker_ids=(0,)
    )
    owner._register_domain_ccu = lambda *args: 10
    dom = domain()
    target = cast(CallableHandle, SimpleNamespace(digest=b"a" * 32))
    assert Worker._get_or_register_domain_ccu(cast(Worker, owner), dom, target, ccu_id=0, worker=0).worker == 0
    with pytest.raises(ValueError, match="target worker"):
        Worker._get_or_register_domain_ccu(cast(Worker, owner), dom, target, ccu_id=0, worker=2)


def test_registration_holds_worker_until_reply_and_cache_publication(monkeypatch):
    owner = Worker(level=3, num_sub_workers=0)
    owner._config["platform"] = "a5"
    owner._lifecycle = worker_module._Lifecycle.READY
    dom = domain()
    target = cast(CallableHandle, SimpleNamespace(digest=b"a" * 32))
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    class Transport:
        def control_payload(self, *_args):
            entered.set()
            assert release.wait(5)
            assert not closed.is_set()
            return worker_module._CCU_DOMAIN_REQUEST.pack(dom.allocation_id, dom.generation, 9, target.digest, 12)

        def close(self):
            assert dom._ccu_registrations, "transport closed before registration publication"
            closed.set()

    owner._worker = Transport()
    monkeypatch.setattr(owner, "_ensure_comm_base", lambda: None)
    monkeypatch.setattr(
        owner,
        "_resolve_handle",
        lambda *a, **k: SimpleNamespace(target=SimpleNamespace(ccu_module_count=1), eligible_worker_ids=(0, 2)),
    )
    wait = owner._hierarchical_start_cv.wait

    def release_registration_when_close_drains(timeout=None):
        release.set()
        return wait(timeout)

    monkeypatch.setattr(owner._hierarchical_start_cv, "wait", release_registration_when_close_drains)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(owner._get_or_register_domain_ccu, dom, target, ccu_id=0, worker=2)
        try:
            assert entered.wait(5)
            owner.close()
        finally:
            release.set()
        binding = future.result(timeout=5)
    assert closed.is_set()
    assert binding.registration_id == 12
    with pytest.raises(RuntimeError, match="READY"):
        owner._get_or_register_domain_ccu(dom, target, ccu_id=0, worker=2)
