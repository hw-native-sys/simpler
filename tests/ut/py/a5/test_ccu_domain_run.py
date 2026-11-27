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
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

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
        _domain_control_mu=threading.Lock(),
        _next_alloc_id=0,
        _ccu_registration_collectives={},
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
        _domain_control_mu=threading.Lock(),
        _next_alloc_id=7,
        _ccu_registration_collectives={},
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


def test_ccu_collective_drains_all_ranks_after_start_interruption(monkeypatch):
    seen = []

    def control(_kind, rank, _command, payload, _timeout):
        seen.append(rank)
        return payload

    native = type("Native", (), {"control_payload": staticmethod(control)})()
    worker: Any = SimpleNamespace(
        _worker=native,
        _py_control_timeout_s=1,
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
        Worker._confirm_ccu_terminal(worker, worker_mod._CCU_CONFIRM_PRODUCERS_STOPPED, {0, 1})
    assert sorted(seen) == [0, 1]


def test_use_domain_only_records_keepalive_without_launch():
    worker: Any = SimpleNamespace(level=3, _building_run_resources=worker_mod._RunResources())
    worker._release_domain_handle = lambda *_: None
    domain = _domain()
    with _callback_run(1, worker):
        Worker._use_domain(worker, domain)
        Worker._use_domain(worker, domain)
    assert worker._building_run_resources.live_domains == {domain.name: domain}
    assert worker._building_run_resources.ccu_domains == {}
    assert worker._building_run_resources.requires_ordered_cleanup


@pytest.mark.parametrize("failed_phase", [11, 12])
def test_ccu_terminal_cleanup_joins_all_peers_and_retries_only_unfinished_phases(failed_phase):
    completed = {phase: set() for phase in (11, 12)}
    failures = [True]

    def control(_kind, chip, _command, payload, _timeout):
        _, _, phase, _, _ = worker_mod._CCU_DOMAIN_REQUEST.unpack(payload)
        if phase > 11:
            assert completed[phase - 1] == {0, 1, 2}
        if phase == failed_phase and chip == 1 and failures:
            failures.pop()
            raise RuntimeError("SDK stop is unconfirmed")
        completed[phase].add(chip)
        return payload

    worker = Worker.__new__(Worker)
    mailboxes = [bytearray(worker_mod.MAILBOX_FRAME_SIZE) for _ in range(3)]
    worker._chip_shms = [SimpleNamespace(buf=mailbox) for mailbox in mailboxes]
    worker._worker = SimpleNamespace(control_payload=control)
    worker._config = {"device_ids": [0, 1, 2]}
    worker._py_control_timeout_s = 1
    worker._ccu_terminal_error = None
    worker._ccu_terminal_lock = threading.Lock()
    worker._cleanup_journal = worker_mod.CleanupJournal()
    first, second = _domain(), _domain(allocation_id=10)
    resources = worker_mod._RunResources(ccu_domains={first.allocation_id: first, second.allocation_id: second})
    original = RuntimeError("original native task failure")
    with pytest.raises(RuntimeError, match="SDK stop"):
        worker._stop_run_ccu_domains(resources, original)
    assert not worker._cleanup_journal.empty
    assert worker._ccu_terminal_error is original
    assert completed[failed_phase] == {0, 2}
    assert all(not completed[phase] for phase in range(failed_phase + 1, 13))
    assert all(mailbox[worker_mod._OFF_CCU_PEERS_STOPPED] == 0 for mailbox in mailboxes)
    worker._finish_ccu_terminal_cleanup()
    assert all(mailbox[worker_mod._OFF_CCU_PEERS_STOPPED] == 1 for mailbox in mailboxes)
    assert worker._cleanup_journal.empty
    assert worker._ccu_terminal_error is original
    assert completed == {11: {0, 1, 2}, 12: {0, 1, 2}}


def test_close_resumes_ccu_barrier_after_failed_run_and_failed_first_close():
    parent = Worker(level=3, num_sub_workers=0)
    parent._config["device_ids"] = [0, 1]
    native = Mock()
    parent._worker = native
    parent._chip_shms = [Mock(buf=bytearray(worker_mod.MAILBOX_FRAME_SIZE)) for _ in range(2)]
    calls = {0: [], 1: []}

    def control(_kind, chip, _command, payload, _timeout):
        phase = worker_mod._CCU_DOMAIN_REQUEST.unpack(payload)[2]
        calls[chip].append(phase)
        if chip == 1 and phase == 12 and calls[1].count(12) < 3:
            raise RuntimeError("awaiting stream completion")
        return payload

    native.control_payload.side_effect = control
    domain = _domain()
    resources = worker_mod._RunResources(ccu_domains={domain.allocation_id: domain})
    original = RuntimeError("execution failed")
    with pytest.raises(RuntimeError, match="awaiting stream completion"):
        parent._stop_run_ccu_domains(resources, original)
    with pytest.raises(RuntimeError, match="awaiting stream completion"):
        parent.close()
    native.close.assert_not_called()
    assert not parent._cleanup_journal.empty
    assert calls == {0: [11, 12], 1: [11, 12, 12]}
    parent.close()
    assert calls == {0: [11, 12], 1: [11, 12, 12, 12]}
    assert parent._ccu_terminal_error is original
    assert parent._cleanup_journal.empty
    native.close.assert_called_once()


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


def test_l2_confirmed_abort_releases_active_use_but_keeps_admission_closed():
    binding = ChipCcuBinding(SimpleNamespace(control_ccu_run=lambda *_: 0, prepare_ccu_run=lambda *_: 0), 10, 9, 1)
    run = binding.prepare(1, 9)
    run.begin()
    assert run.abort()
    assert not binding.active
    with pytest.raises(RuntimeError, match="admission is closed"):
        binding.prepare(2, 9)
    with pytest.raises(RuntimeError, match="admission is closed"):
        run.begin()


@pytest.mark.parametrize("failed_operation", [1, 3, 7])
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
