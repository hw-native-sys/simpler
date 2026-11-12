# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU coverage for rank joins, READY admission, and retained stop failures."""

import threading
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import simpler.worker as worker_mod
from simpler.ccu import ChipCcuBinding
from simpler.comm_domain import DomainConfig
from simpler.orchestrator import _callback_run
from simpler.task_interface import CommDomainHandle
from simpler.worker import Worker


def test_registration_collective_carries_full_context_without_a_domain():
    observed = []

    def control(_kind, chip, _command, payload, _timeout):
        allocation, generation, operation, digest, result = worker_mod._CCU_DOMAIN_REQUEST.unpack_from(payload)
        assert (allocation, generation, operation, digest, result) == (8, 0, 5, b"c" * 32, 0)
        tail = worker_mod._CCU_DOMAIN_REQUEST.size
        assert worker_mod.struct.unpack_from("<QII3I", payload, tail) == (0, 3, chip, 0, 1, 2)
        assert len(payload) == tail + 28
        observed.append(chip)
        reply = bytearray(payload)
        worker_mod.struct.pack_into("<i", reply, tail - 4, 17)
        return reply

    native = type("Native", (), {"control_payload": staticmethod(control)})()
    worker: Any = SimpleNamespace(
        _worker=native,
        _py_control_timeout_s=1,
        _alloc_id_lock=threading.Lock(),
        _next_alloc_id=0,
        _ccu_registration_collectives={},
        _ccu_run_registrations={},
        _config={"device_ids": [3, 4, 5]},
        _next_level_worker_ids=(),
        _ensure_comm_base=lambda: None,
    )
    assert Worker._ccu_registration_collective(worker, b"c" * 32) == [17, 17, 17]
    assert sorted(observed) == [0, 1, 2]


def test_registration_collective_identity_survives_different_rank_histories():
    requests = []
    local_caches = {rank: {} for rank in range(3)}

    def control(_kind, chip, _command, payload, _timeout):
        tail = worker_mod._CCU_DOMAIN_REQUEST.size
        collective, count, domain_rank = worker_mod.struct.unpack_from("<QII", payload, tail)
        members = worker_mod.struct.unpack_from(f"<{count}I", payload, tail + 16)
        assert members[domain_rank] == chip
        requests.append((chip, collective, members))
        cache = local_caches[chip]
        registration = cache.setdefault(collective, len(cache) + 1)
        reply = bytearray(payload)
        worker_mod.struct.pack_into("<i", reply, tail - 4, registration)
        return reply

    native = type("Native", (), {"control_payload": staticmethod(control)})()
    worker: Any = SimpleNamespace(
        _worker=native,
        _py_control_timeout_s=1,
        _alloc_id_lock=threading.Lock(),
        _next_alloc_id=7,
        _ccu_registration_collectives={},
        _ccu_run_registrations={},
        _config={"device_ids": [3, 4, 5]},
        _next_level_worker_ids=(),
        _ccu_context_registrations={},
        _ensure_comm_base=lambda: None,
    )
    worker._ccu_registration_collective = lambda digest: Worker._ccu_registration_collective(worker, digest)
    Worker._ensure_ccu_registration(worker, b"c" * 32)
    assert {identity for _, identity, _ in requests} == {7}
    assert {ranks for _, _, ranks in requests} == {(0, 1, 2)}
    assert worker._next_alloc_id == 8


def _domain(allocation_id=9):
    handle = CommDomainHandle(name="ccu", workers=(0, 1), allocation_id=allocation_id, _release_fn=lambda _: None)
    handle._config = DomainConfig((0, 1), 4096, (), True, (), ("ccu",))
    handle.contexts = {i: SimpleNamespace(device_ctx=100 + i) for i in range(2)}
    return handle


@pytest.mark.parametrize("members", [(0, 1, 2), (2, 0), (0, 2)])
def test_domain_controls_select_full_context_registration_by_worker(members):
    observed = []
    digest = b"c" * 32
    registrations = [17, 29, 43]

    def control(_kind, chip, _command, payload, _timeout):
        resource, generation, operation, callable_digest, _ = worker_mod._CCU_DOMAIN_REQUEST.unpack(payload)
        observed.append((chip, resource))
        assert (generation, operation, callable_digest) == (0, 8, digest)
        return payload

    native = type("Native", (), {"control_payload": staticmethod(control)})()
    worker: Any = SimpleNamespace(
        _worker=native,
        _py_control_timeout_s=1,
        _ccu_context_registrations={(native, digest): registrations},
    )
    domain = CommDomainHandle(name="subset", workers=members, allocation_id=9, _release_fn=lambda _: None)
    Worker._ccu_domain_collective(worker, domain, 8, digest)
    assert sorted(observed) == sorted((chip, registrations[chip]) for chip in members)


def test_collective_joins_other_ranks_before_raising():
    seen = []

    def control(_kind, rank, _command, payload, _timeout):
        seen.append(rank)
        if rank == 0:
            raise RuntimeError("rank zero refused")
        return payload

    native = type("Native", (), {"control_payload": staticmethod(control)})()
    worker: Any = SimpleNamespace(
        _worker=native,
        _py_control_timeout_s=1,
        _ccu_run_registrations={(9, bytes(32)): [1, 2]},
        _ccu_context_registrations={(native, bytes(32)): [1, 2]},
    )
    domain = _domain()
    with pytest.raises(RuntimeError, match="rank zero"):
        Worker._ccu_domain_collective(worker, domain, 1)
    assert sorted(seen) == [0, 1]
    assert domain._ccu_run_generation == 1


def test_ccu_collective_drains_all_ranks_after_start_interruption(monkeypatch):
    seen = []

    def control(_kind, rank, _command, payload, _timeout):
        seen.append(rank)
        return payload

    native = type("Native", (), {"control_payload": staticmethod(control)})()
    worker: Any = SimpleNamespace(
        _worker=native,
        _py_control_timeout_s=1,
        _ccu_run_registrations={(9, bytes(32)): [1, 2]},
        _ccu_context_registrations={(native, bytes(32)): [1, 2]},
    )
    real_thread = threading.Thread

    class StartInterrupt(real_thread):
        armed = True

        def start(self):
            super().start()
            if StartInterrupt.armed:
                StartInterrupt.armed = False
                raise KeyboardInterrupt

    monkeypatch.setattr(worker_mod.threading, "Thread", StartInterrupt)
    with pytest.raises(KeyboardInterrupt):
        Worker._ccu_domain_collective(worker, _domain(), 1)
    assert sorted(seen) == [0, 1]


def test_use_domain_only_records_keepalive_without_launch():
    worker: Any = SimpleNamespace(level=3, _building_run_resources=worker_mod._RunResources())
    observed = []
    domain = _domain()

    def collective(*_):
        pytest.fail("borrowing a domain must not prepare or launch CCU")

    worker._control_admission = lambda _api: nullcontext()
    worker._ccu_domain_collective = collective
    worker._release_domain_handle = lambda *_: None
    worker._abort_ccu_domain = lambda _: pytest.fail("unexpected abort")
    with _callback_run(1, worker):
        Worker._use_domain(worker, domain)
        Worker._use_domain(worker, domain)
    assert observed == []
    assert worker._building_run_resources.ccu_domains == {}
    assert worker._building_run_resources.requires_ordered_cleanup


def test_task_failure_aborts_all_used_ccu_domains():
    first, second = _domain(), _domain(allocation_id=10)
    first.name, second.name = "first", "second"
    aborted = []
    worker: Any = SimpleNamespace(_abort_ccu_domain=lambda handle: aborted.append(handle.name))
    resources = worker_mod._RunResources(ccu_domains={first.allocation_id: first, second.allocation_id: second})
    error = RuntimeError("ordinary task failed")
    with pytest.raises(RuntimeError, match="CCU run aborted") as result:
        Worker._stop_run_ccu_domains(worker, resources, error)
    assert result.value.__cause__ is error
    assert aborted == ["first", "second"]
    assert list(resources.ccu_domains) == [first.allocation_id, second.allocation_id]


def test_abort_marks_unconfirmed_resources():
    observed = []

    def collective(_handle, operation, digest):
        assert operation == 4
        raise RuntimeError("stream stop unconfirmed")

    worker: Any = SimpleNamespace(
        _ccu_domain_collective=collective,
        _ccu_run_registrations={(9, bytes(32)): [1, 2]},
        _record_unreclaimable=lambda message, error: observed.append((message, error)),
    )
    Worker._abort_ccu_domain(worker, _domain())
    assert "retained" in observed[0][0]
    assert str(observed[0][1]) == "stream stop unconfirmed"


@pytest.mark.parametrize("first_error", [RuntimeError("stop unconfirmed"), KeyboardInterrupt()])
@pytest.mark.parametrize("already_poisoned", [False, True])
def test_abort_attempts_every_callable_and_preserves_first_failure(first_error, already_poisoned):
    attempted = []
    digests = [bytes([i]) * 32 for i in range(4)]
    prior_error = RuntimeError("prior failure") if already_poisoned else None
    worker: Any = SimpleNamespace(
        _ccu_run_registrations={
            (9, digests[0]): [1, 2],
            (10, digests[1]): [3, 4],
            (9, digests[2]): [5, 6],
            (9, digests[3]): [7, 8],
        },
        _hierarchical_start_cv=threading.Condition(),
        _ordered_cleanup_error=prior_error,
        _require_no_delegated_session_fatal=lambda _: None,
    )

    def collective(handle, operation, digest):
        assert (handle.allocation_id, operation) == (9, 4)
        attempted.append(digest)
        if digest == digests[0]:
            raise first_error
        assert worker._ordered_cleanup_error is not None
        if digest == digests[2]:
            raise RuntimeError("second stop unconfirmed")
        return [0, 0]

    worker._ccu_domain_collective = collective
    worker._record_unreclaimable = lambda message, error: Worker._record_unreclaimable(worker, message, error)
    Worker._abort_ccu_domain(worker, _domain())
    assert attempted == [digests[0], digests[2], digests[3]]
    assert len(worker._ccu_run_registrations) == 4
    if prior_error is not None:
        assert worker._ordered_cleanup_error is prior_error
    else:
        assert worker._ordered_cleanup_error.__cause__ is first_error
    with pytest.raises(RuntimeError, match="no further work is admitted"):
        Worker._require_no_ordered_cleanup_failure(worker, "submit")


def test_prepare_registers_selected_domain_and_reuses_without_launch():
    domain = _domain()
    resources = worker_mod._RunResources(live_domains={domain.name: domain})
    state = SimpleNamespace(target=SimpleNamespace(ccu_module_count=1), eligible_worker_ids=(0, 1))
    observed = []
    worker: Any = SimpleNamespace(
        level=3,
        _config={"platform": "a5", "device_ids": [4, 5]},
        _next_level_worker_ids=[],
        _ccu_run_registrations={},
        _ccu_context_registrations={},
        _worker=object(),
        _building_run_resources=resources,
        _record_touched_identities=lambda _: None,
        _enroll_domain_uses=lambda _: None,
        _resolve_handle=lambda *_args, **_kwargs: state,
        _control_admission=lambda _: nullcontext(),
        _ccu_domain_collective=lambda handle, operation, digest: observed.append((handle.workers, operation, digest))
        or [1, 2],
    )
    callable_handle = SimpleNamespace(digest=b"c" * 32)
    worker._ccu_context_registrations[(worker._worker, callable_handle.digest)] = [1, 2]

    def args(chip):
        return SimpleNamespace(scalar_count=lambda: 1, scalar=lambda _: 100 + chip)

    Worker._prepare_ccu_submission(worker, callable_handle, [0], [args(0)])
    Worker._prepare_ccu_submission(worker, callable_handle, [1], [args(1)])
    assert observed == [((0, 1), 6, callable_handle.digest)]
    assert resources.ccu_domains[9] is domain
    with pytest.raises(ValueError, match="explicitly registered"):
        Worker._prepare_ccu_submission(worker, SimpleNamespace(digest=b"d" * 32), [0], [args(0)])
    assert len(observed) == 1


@pytest.mark.parametrize("domain_count", [0, 2])
def test_ccu_submission_rejects_missing_or_ambiguous_context(domain_count):
    domains = {str(i): _domain() for i in range(domain_count)}
    worker: Any = SimpleNamespace(
        level=3,
        _config={"platform": "a5", "device_ids": [4, 5]},
        _next_level_worker_ids=[],
        _ccu_run_registrations={},
        _ccu_context_registrations={},
        _worker=object(),
        _control_admission=lambda _: nullcontext(),
        _record_touched_identities=lambda _: None,
        _enroll_domain_uses=lambda _: None,
        _ccu_domain_collective=lambda *_: [1, 1],
        _building_run_resources=worker_mod._RunResources(live_domains=domains),
        _resolve_handle=lambda *_args, **_kwargs: SimpleNamespace(
            target=SimpleNamespace(ccu_module_count=1), eligible_worker_ids=(0, 1)
        ),
    )
    worker._ccu_context_registrations[(worker._worker, b"c" * 32)] = [1, 2]
    args = SimpleNamespace(scalar_count=lambda: 1, scalar=lambda _: 100)
    with pytest.raises(ValueError, match="CCU-enabled|context"):
        Worker._prepare_ccu_submission(worker, SimpleNamespace(digest=b"c" * 32), [0], [args])


def test_l2_launches_only_on_begin_and_confirms_stop_before_reuse():
    operations = []
    worker = SimpleNamespace(
        control_ccu_run=lambda comm, domain, op, gen: operations.append((op, gen)) or 0, prepare_ccu_run=lambda *_: 0
    )
    binding = ChipCcuBinding(worker, 10, 9, 1)
    assert operations == []
    first, second = binding.prepare(1, 9), binding.prepare(2, 9)
    first.begin()
    with pytest.raises(RuntimeError, match="confirmed stop"):
        second.begin()
    first.finish()
    second.begin()
    second.finish()
    assert operations == [(1, 1), (3, 1), (7, 1), (1, 2), (3, 2), (7, 2)]
    assert not binding.active


@pytest.mark.parametrize("failed_operation", [1, 3])
def test_l2_failure_aborts_and_keeps_binding_active(failed_operation):
    operations = []

    def control(comm, domain, operation, generation):
        operations.append(operation)
        if operation == failed_operation:
            raise RuntimeError("original failure")
        if operation == 4:
            raise RuntimeError("abort unconfirmed")
        return 0

    binding = ChipCcuBinding(SimpleNamespace(control_ccu_run=control, prepare_ccu_run=lambda *_: 0), 10, 9, 1)
    run = binding.prepare(1, 9)
    with pytest.raises(RuntimeError, match="original failure"):
        run.begin()
        run.finish()
    assert operations[-1] == 4
    assert binding.active
