# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Session cleanup record codec and report semantics."""

from __future__ import annotations

import json

import pytest
from simpler.session_cleanup import (
    MAX_FAILURE_BYTES,
    MAX_FAILURES,
    MAX_RECORD_BYTES,
    CleanupOutcome,
    ProcessExit,
    SessionCleanupReport,
    ShutdownCommand,
    decode_runner_record,
    encode_runner_record,
    format_cleanup_failure,
)


def _record(**overrides) -> bytes:
    payload = {
        "schema": 1,
        "worker_id": 7,
        "rank": 0,
        "command": int(ShutdownCommand.OBSERVED),
        "cleanup": int(CleanupOutcome.SUCCEEDED),
        "failures": [],
    }
    payload.update(overrides)
    return json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n"


def test_round_trip_keeps_the_three_events_apart():
    raw = encode_runner_record(
        worker_id=7,
        command=ShutdownCommand.OBSERVED,
        cleanup=CleanupOutcome.FAILED,
        failures=["unlink failed"],
        rank=0,
    )
    report = decode_runner_record(raw)
    assert report is not None
    assert report.committed is True
    assert report.command is ShutdownCommand.OBSERVED
    assert report.cleanup is CleanupOutcome.FAILED
    assert report.failures == ("unlink failed",)
    # The runner never reports its own exit; only its launcher observes that.
    assert report.process_exit is ProcessExit.UNKNOWN
    assert report.exit_status is None
    assert report.notes == ()


def test_a_record_ends_with_a_newline_so_it_can_share_a_stream():
    raw = encode_runner_record(worker_id=1, command=ShutdownCommand.OBSERVED, cleanup=CleanupOutcome.SUCCEEDED)
    assert raw.endswith(b"\n")
    assert b"\n" not in raw[:-1]


@pytest.mark.parametrize(
    "command,cleanup,failures",
    [
        (ShutdownCommand.UNKNOWN, CleanupOutcome.SUCCEEDED, ()),
        (ShutdownCommand.OBSERVED, CleanupOutcome.UNKNOWN, ()),
    ],
)
def test_a_runner_may_not_write_unknown_about_itself(command, cleanup, failures):
    # A runner that reached its teardown knows how its lane ended and what its
    # release returned; UNKNOWN would claim a record while asserting nothing.
    with pytest.raises(ValueError):
        encode_runner_record(worker_id=1, command=command, cleanup=cleanup, failures=failures)


@pytest.mark.parametrize(
    "cleanup,failures",
    [
        (CleanupOutcome.FAILED, ()),
        (CleanupOutcome.SUCCEEDED, ("boom",)),
    ],
)
def test_failures_and_outcome_must_agree_on_the_way_out(cleanup, failures):
    with pytest.raises(ValueError):
        encode_runner_record(worker_id=1, command=ShutdownCommand.OBSERVED, cleanup=cleanup, failures=failures)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"not json",
        b"[]",
        _record(schema=99),
        _record(command=7),
        _record(cleanup=7),
        _record(command=int(ShutdownCommand.UNKNOWN)),
        _record(cleanup=int(CleanupOutcome.UNKNOWN)),
        _record(cleanup=int(CleanupOutcome.FAILED), failures=[]),
        _record(failures=["ok", 5]),
        _record(failures="not a list"),
    ],
    ids=[
        "empty",
        "not-json",
        "not-an-object",
        "wrong-schema",
        "unknown-command-value",
        "unknown-cleanup-value",
        "command-unknown",
        "cleanup-unknown",
        "failed-without-failures",
        "non-string-failure",
        "failures-not-a-list",
    ],
)
def test_anything_short_of_wholly_valid_reads_as_absent(raw):
    # Absent is the honest answer: the launcher reports committed false rather
    # than half-trusting a record it does not understand.
    assert decode_runner_record(raw) is None


def test_a_failed_record_without_its_failure_list_is_rejected_both_ways():
    # The identity holds on the way in as a ValueError and on the way out as an
    # absent record, so neither side can publish a FAILED claim with no detail.
    with pytest.raises(ValueError):
        encode_runner_record(worker_id=1, command=ShutdownCommand.OBSERVED, cleanup=CleanupOutcome.FAILED)
    assert decode_runner_record(_record(cleanup=int(CleanupOutcome.FAILED), failures=[])) is None


def test_a_record_stays_inside_the_bound_a_reader_will_take():
    raw = encode_runner_record(
        worker_id=1,
        command=ShutdownCommand.OBSERVED,
        cleanup=CleanupOutcome.FAILED,
        failures=["x" * (MAX_FAILURE_BYTES * 4)] * (MAX_FAILURES * 4),
    )
    assert len(raw) <= MAX_RECORD_BYTES
    report = decode_runner_record(raw)
    assert report is not None
    assert len(report.failures) == MAX_FAILURES
    assert all(len(failure) <= MAX_FAILURE_BYTES for failure in report.failures)


def test_a_failure_summary_is_one_bounded_line():
    summary = format_cleanup_failure("release", RuntimeError("first\nsecond " + "y" * 4096))
    assert "\n" not in summary
    assert summary.startswith("release: RuntimeError: first second")
    assert len(summary.encode("utf-8")) <= MAX_FAILURE_BYTES


def test_cleanup_proven_separates_no_answer_from_a_bad_one():
    assert SessionCleanupReport().cleanup_proven is False
    assert SessionCleanupReport(cleanup=CleanupOutcome.FAILED).cleanup_proven is True
    assert SessionCleanupReport(cleanup=CleanupOutcome.SUCCEEDED).cleanup_proven is True


def test_describe_names_all_three_events_and_both_detail_lists():
    line = SessionCleanupReport(
        worker_id=4,
        committed=True,
        command=ShutdownCommand.NOT_OBSERVED,
        process_exit=ProcessExit.EXITED,
        exit_status=9,
        cleanup=CleanupOutcome.FAILED,
        failures=("release failed",),
        notes=("could not unlink mailbox",),
        rank=1,
    ).describe()
    assert "worker 4" in line
    assert "command=not_observed" in line
    assert "process=exited(9)" in line
    assert "cleanup=failed@rank1" in line
    assert "release failed" in line
    assert "could not unlink mailbox" in line


def test_a_default_report_asserts_nothing():
    report = SessionCleanupReport(worker_id=2)
    assert report.committed is False
    assert report.command is ShutdownCommand.UNKNOWN
    assert report.process_exit is ProcessExit.UNKNOWN
    assert report.cleanup is CleanupOutcome.UNKNOWN
    assert report.failures == ()
    assert report.notes == ()
    assert report.rank == -1
