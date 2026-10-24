# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import errno
import json
import socket
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from simpler.mpi_direct_cleanup import (
    CLEANUP_SCHEMA,
    MAX_FAILURE_BYTES,
    MAX_FAILURES,
    MAX_REPORT_BYTES,
    MpiDirectCleanupLedger,
    MpiDirectCleanupReport,
    _CleanupReceiver,
    _print_cleanup_reports,
    _RankCleanupReporter,
    cleanup_failure,
)
from simpler.mpi_direct_protocol import _gate_recv, _gate_send
from simpler.mpi_direct_topology import MpiDirectTopology


@pytest.fixture
def ledger():
    topology = MpiDirectTopology.from_dict(
        {
            "controller_host": "127.0.0.1",
            "executor_ranks": [
                {"rank": rank, "worker_id": rank - 1, "host": "127.0.0.1", "device_ids": [], "platform": "a2a3sim"}
                for rank in (1, 2)
            ],
        }
    )
    return MpiDirectCleanupLedger(topology, "job", 91)


def _record(rank=1, **changes):
    return replace(
        MpiDirectCleanupReport("job", 91, rank, rank - 1 if rank else None, "L3" if rank else "L4"),
        **{
            "sequence": 2,
            "command": "observed" if rank else "not_observed",
            "cleanup": "succeeded",
            **changes,
        },
    ).to_dict()


def test_rank_report_keeps_missing_and_incomplete_results_unknown(ledger):
    assert set(ledger.snapshot()) == {0, 1, 2}
    payload = _record()
    payload.update(sequence=1, cleanup="unknown")
    ledger.accept(payload)
    assert ledger.snapshot()[1].command == "observed"
    assert ledger.snapshot()[1].cleanup == "unknown"
    assert ledger.snapshot()[2].command == "unknown"
    ledger.accept(_record())
    assert ledger.snapshot()[1].cleanup == "succeeded"
    assert ledger.snapshot()[2].cleanup == "unknown"
    assert "process_exit" not in ledger.snapshot()[1].to_dict()
    snapshot = ledger.snapshot()
    snapshot.clear()
    assert len(ledger.snapshot()) == 3


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_each_rank_can_report_failed_release_without_claiming_process_exit(ledger, rank):
    assert ledger.accept(_record(rank, failures=("close failed",), cleanup="failed"))
    assert ledger.snapshot()[rank].failures == ("close failed",)


def test_duplicate_reports_are_idempotent_and_conflicts_do_not_overwrite(ledger):
    final = _record()
    assert ledger.accept(final)
    assert not ledger.accept(final)
    for changed in ({"cleanup": "failed", "failures": ["failure"]}, {"sequence": 1, "cleanup": "unknown"}):
        with pytest.raises(ValueError, match="stale or conflicting"):
            ledger.accept({**final, **changed})
    assert ledger.snapshot()[1].cleanup == "succeeded"


@pytest.mark.parametrize(
    "changes",
    [
        {"schema": CLEANUP_SCHEMA + 1},
        {"schema": True},
        {"session_id": 92},
        {"session_id": True},
        {"job_id": "old-job"},
        {"rank": 4},
        {"rank": True},
        {"worker_id": 1},
        {"worker_id": False},
        {"role": "L4"},
        {"command": "unknown"},
        {"command": "sent"},
        {"sequence": 0},
        {"sequence": True},
        {"sequence": 1},
        {"cleanup": "failed"},
        {"cleanup": "unknown"},
        {"failures": ["failure"]},
        {"failures": "failure"},
        {"failures": [1]},
        {"failures": ["failure"] * (MAX_FAILURES + 1)},
        {"failures": ["x" * (MAX_FAILURE_BYTES + 1)]},
        {"job_id": "x" * MAX_REPORT_BYTES},
        {"process_exit": "exited"},
    ],
)
def test_rejects_invalid_stale_and_oversized_records_atomically(ledger, changes):
    original = ledger.snapshot()
    with pytest.raises(ValueError):
        ledger.accept({**_record(), **changes})
    assert ledger.snapshot() == original


@pytest.mark.parametrize("field", list(_record()))
def test_incomplete_record_is_not_partially_committed(ledger, field):
    payload = _record()
    del payload[field]
    with pytest.raises(ValueError):
        ledger.accept(payload)
    assert ledger.snapshot()[1].sequence == 0


def test_final_cannot_change_acknowledged_command(ledger):
    payload = _record()
    ledger.accept({**payload, "sequence": 1, "cleanup": "unknown"})
    with pytest.raises(ValueError, match="conflicting"):
        ledger.accept({**payload, "command": "not_observed"})


def test_external_receiver_accepts_all_ranks_independently(ledger):
    with _CleanupReceiver("127.0.0.1", "token", ledger) as receiver:
        reporters = [
            _RankCleanupReporter("127.0.0.1", receiver.port, "token", identity)
            for identity in ledger.snapshot().values()
        ]
        threads = []
        for rank, reporter in enumerate(reporters):

            def report(reporter=reporter, rank=rank):
                reporter.begin(command="not_observed" if rank == 0 else "observed")
                reporter.finish(["release failed"] if rank == 2 else [])

            threads.append(threading.Thread(target=report))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
            assert not thread.is_alive()
    assert [item.cleanup for item in ledger.snapshot().values()] == ["succeeded", "succeeded", "failed"]


@pytest.mark.parametrize("accept_errno", [errno.ECONNABORTED, errno.EMFILE])
def test_receiver_recovers_from_transient_accept_error(ledger, accept_errno):
    receiver = _CleanupReceiver("127.0.0.1", "token", ledger)
    listener = receiver._listener
    attempts = 0

    def accept():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(accept_errno, "injected accept failure")
        return listener.accept()

    receiver._listener = SimpleNamespace(accept=accept, fileno=listener.fileno, close=listener.close)
    with receiver:
        reporter = _RankCleanupReporter("127.0.0.1", receiver.port, "token", ledger.snapshot()[1])
        reporter.begin(command="observed")
        reporter.finish([])
    assert ledger.snapshot()[1].cleanup == "succeeded"
    assert attempts >= 3


@pytest.mark.parametrize("closed", [False, True])
def test_receiver_accept_error_waits_interruptibly_unless_listener_closed(closed):
    receiver = _CleanupReceiver.__new__(_CleanupReceiver)
    stopped = False
    waits = []

    def accept():
        raise OSError(errno.EMFILE, "injected persistent accept failure")

    def wait(delay):
        nonlocal stopped
        waits.append(delay)
        stopped = True

    receiver._listener = SimpleNamespace(accept=accept, fileno=lambda: -1 if closed else 10)
    receiver._stop = SimpleNamespace(is_set=lambda: stopped, wait=wait)
    receiver._run()
    if closed:
        assert not waits
    else:
        assert len(waits) == 1
        assert 0 < waits[0] <= 0.1


def test_receiver_rejects_bad_authentication_and_incomplete_input_then_recovers(ledger):
    with _CleanupReceiver("127.0.0.1", "token", ledger) as receiver:
        for message in ({"token": "wrong", "report": _record()}, {"token": "token", "report": {}}):
            with socket.create_connection(("127.0.0.1", receiver.port), timeout=2) as peer:
                _gate_send(peer, message)
                with pytest.raises((OSError, ConnectionError)):
                    _gate_recv(peer, time.monotonic() + 2)
        with socket.create_connection(("127.0.0.1", receiver.port), timeout=2) as peer:
            peer.sendall(b"\x00\x00")
        reporter = _RankCleanupReporter("127.0.0.1", receiver.port, "token", ledger.snapshot()[1])
        reporter.begin(command="observed")
        reporter.finish([])
    assert ledger.snapshot()[1].cleanup == "succeeded"
    assert ledger.snapshot()[2].cleanup == "unknown"


def test_unreachable_report_channel_does_not_raise_or_fabricate_receipt(ledger):
    with socket.socket() as peer:
        peer.bind(("127.0.0.1", 0))
        port = peer.getsockname()[1]
    reporter = _RankCleanupReporter("127.0.0.1", port, "token", ledger.snapshot()[1])
    reporter.begin(command="observed")
    reporter.finish([])
    assert ledger.snapshot()[1].sequence == 0


def test_reporter_resolves_host_at_startup_and_reuses_numeric_address(ledger, monkeypatch):
    calls = []

    def resolve(host):
        calls.append(host)
        return "127.0.0.1"

    monkeypatch.setattr(socket, "gethostbyname", resolve)
    with _CleanupReceiver("127.0.0.1", "token", ledger) as receiver:
        reporter = _RankCleanupReporter("controller", receiver.port, "token", ledger.snapshot()[1])
        reporter.begin(command="observed")
        reporter.finish([])
    assert calls == ["controller"]
    assert ledger.snapshot()[1].cleanup == "succeeded"


def test_reporter_does_not_wait_forever_for_receiver_acknowledgement(ledger):
    stop = threading.Event()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()

        def no_ack():
            with listener.accept()[0] as peer:
                _gate_recv(peer, time.monotonic() + 2)
                stop.wait(3)

        thread = threading.Thread(target=no_ack)
        thread.start()
        reporter = _RankCleanupReporter("127.0.0.1", listener.getsockname()[1], "token", ledger.snapshot()[1])
        started = time.monotonic()
        try:
            reporter.begin(command="observed")
        finally:
            stop.set()
            thread.join(3)
        assert time.monotonic() - started < 2.5
        assert not thread.is_alive()
        assert ledger.snapshot()[1].sequence == 0


def test_summary_output_failure_does_not_mask_job_result(ledger, monkeypatch):
    def fail_print(*_args, **_kwargs):
        raise BrokenPipeError("summary output closed")

    monkeypatch.setattr("builtins.print", fail_print)
    _print_cleanup_reports(ledger)


def test_failure_text_stays_inside_byte_and_record_limits(ledger):
    failure = cleanup_failure("close", RuntimeError("\u9519\x00" * 1000))
    assert len(failure.encode()) <= MAX_FAILURE_BYTES
    payload = _record(cleanup="failed", failures=(failure,) * MAX_FAILURES)
    assert len(json.dumps(payload).encode()) <= MAX_REPORT_BYTES
    ledger.accept(payload)
