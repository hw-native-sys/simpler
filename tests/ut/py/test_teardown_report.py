# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The parent's reading of a chip child's device-teardown record.

The record exists because a single teardown error code cannot distinguish "no
reset ran", "the reset call failed" and "the reset call returned 0 but the
post-reset probe did not confirm". These cases pin the decode side of that: the
four real shapes, the commit-last rule that makes a killed child read as
unknown rather than as a partly populated success, and the validity identities
the producer is obliged to maintain.

They exercise the codec and the mailbox layout mirrors, not a live child: a
real teardown needs hardware, which these do not use.
"""

from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest
import simpler.worker as worker_module
from simpler.task_interface import (
    MAILBOX_ARGS_CAPACITY,
    MAILBOX_OFF_TEARDOWN_REPORT,
    MAILBOX_TASK_PROTOCOL_VERSION,
    SIMPLER_TEARDOWN_REPORT_BYTES,
    TEARDOWN_REPORT_SCHEMA,
)
from simpler.teardown_report import (
    TeardownFlags,
    TeardownPath,
    TeardownReport,
    TeardownResetApi,
    TeardownResetStage,
    decode_teardown_report,
    encode_teardown_report_payload,
)

_WIRE = struct.Struct("=IiiBBBBiiiHHQ")
_PID = 4242

_API_RC_VALID = int(TeardownFlags.LAST_RESET_API_RC_VALID)
_SEQ_RC_VALID = int(TeardownFlags.RECOVERY_SEQUENCE_RC_VALID)
_PROBE_CONFIRMED = int(TeardownFlags.PROBE_CONFIRMED)


def _record(  # noqa: PLR0913 -- one keyword per wire field, so a case names exactly what it varies
    *,
    schema: int = TEARDOWN_REPORT_SCHEMA,
    child_pid: int = _PID,
    device_id: int = 3,
    path: int = int(TeardownPath.NORMAL),
    stage: int = int(TeardownResetStage.NOT_ATTEMPTED),
    api: int = int(TeardownResetApi.NONE),
    flags: int = 0,
    last_reset_api_rc: int = 0,
    recovery_sequence_rc: int = 0,
    teardown_rc: int = 0,
    invocations: int = 0,
    attempts: int = 0,
    reserved: int = 0,
) -> bytes:
    return _WIRE.pack(
        schema,
        child_pid,
        device_id,
        path,
        stage,
        api,
        flags,
        last_reset_api_rc,
        recovery_sequence_rc,
        teardown_rc,
        invocations,
        attempts,
        reserved,
    )


def _normal_reset_ok() -> bytes:
    """The rt-path reset returning 0, which no probe follows."""
    return _record(
        path=int(TeardownPath.NORMAL),
        stage=int(TeardownResetStage.API_OK_NO_PROBE),
        api=int(TeardownResetApi.RT_DEVICE_RESET),
        flags=_API_RC_VALID,
        last_reset_api_rc=0,
        invocations=1,
    )


# --------------------------------------------------------------------------
# The four real shapes
# --------------------------------------------------------------------------


def test_normal_reset_success_reports_no_probe_rather_than_confirmation():
    report = decode_teardown_report(_normal_reset_ok(), expected_pid=_PID)

    assert report.committed
    assert report.path is TeardownPath.NORMAL
    assert report.reset_stage is TeardownResetStage.API_OK_NO_PROBE
    assert report.reset_api is TeardownResetApi.RT_DEVICE_RESET
    assert report.last_reset_api_rc == 0
    assert report.reset_api_invoked
    # The distinction the stage exists for: a reset that returned 0 with
    # nothing checked afterwards is not a confirmed device generation.
    assert not report.probe_confirmed
    # No recovery wrapper ran, so its return is absent rather than a zero that
    # would read as a verdict nobody produced.
    assert report.recovery_sequence_rc is None
    assert report.recovery_attempts_total == 0


def test_early_return_reports_that_no_reset_api_ran():
    raw = _record(
        stage=int(TeardownResetStage.NOT_ATTEMPTED),
        api=int(TeardownResetApi.NONE),
        teardown_rc=-1001,
    )
    report = decode_teardown_report(raw, expected_pid=_PID)

    assert report.committed
    assert report.reset_stage is TeardownResetStage.NOT_ATTEMPTED
    assert report.reset_api is TeardownResetApi.NONE
    assert not report.reset_api_invoked
    # Absent, not zero: "no call" and "a call that returned 0" are different.
    assert report.last_reset_api_rc is None
    assert report.teardown_rc == -1001


def test_api_failure_then_preamble_failure_keeps_the_earlier_call():
    raw = _record(
        path=int(TeardownPath.FATAL),
        stage=int(TeardownResetStage.PREAMBLE_FAILED),
        api=int(TeardownResetApi.ACL_RESET_DEVICE_FORCE),
        flags=_API_RC_VALID | _SEQ_RC_VALID,
        last_reset_api_rc=507899,
        recovery_sequence_rc=-1001,
        teardown_rc=-1001,
        invocations=1,
        attempts=2,
    )
    report = decode_teardown_report(raw, expected_pid=_PID)

    assert report.committed
    # The stage is the last attempt's, which never reached a reset call...
    assert report.reset_stage is TeardownResetStage.PREAMBLE_FAILED
    # ...and the cumulative fields are what stop that from reading as "no reset
    # API ever ran in this teardown".
    assert report.reset_api_invocations_total == 1
    assert report.last_reset_api_rc == 507899
    assert report.recovery_attempts_total == 2
    assert report.recovery_sequence_rc == -1001


def test_reset_api_success_with_probe_failure_keeps_both_values():
    raw = _record(
        path=int(TeardownPath.FATAL),
        stage=int(TeardownResetStage.API_OK_PROBE_RUN),
        api=int(TeardownResetApi.ACL_RESET_DEVICE_FORCE),
        flags=_API_RC_VALID | _SEQ_RC_VALID,
        last_reset_api_rc=0,
        recovery_sequence_rc=507899,
        teardown_rc=507899,
        invocations=1,
        attempts=1,
    )
    report = decode_teardown_report(raw, expected_pid=_PID)

    assert report.committed
    assert report.last_reset_api_rc == 0
    assert report.recovery_sequence_rc == 507899
    assert not report.probe_confirmed


def test_confirmed_probe_is_the_only_probe_backed_value():
    raw = _record(
        path=int(TeardownPath.FATAL),
        stage=int(TeardownResetStage.API_OK_PROBE_RUN),
        api=int(TeardownResetApi.ACL_RESET_DEVICE_FORCE),
        flags=_API_RC_VALID | _SEQ_RC_VALID | _PROBE_CONFIRMED,
        invocations=1,
        attempts=1,
    )
    report = decode_teardown_report(raw, expected_pid=_PID)

    assert report.committed
    assert report.probe_confirmed
    assert report.recovery_sequence_rc == 0


# --------------------------------------------------------------------------
# Commit-last and generation
# --------------------------------------------------------------------------


def test_uncommitted_schema_reads_as_unknown():
    # A child killed before releasing the schema word leaves a payload behind.
    # It must never surface as a partly populated success.
    raw = _record(schema=0, stage=int(TeardownResetStage.API_OK_NO_PROBE), invocations=1, flags=_API_RC_VALID)
    report = decode_teardown_report(raw, expected_pid=_PID)

    assert report == TeardownReport()
    assert not report.committed


def test_unrecognised_schema_reads_as_unknown():
    raw = _record(schema=TEARDOWN_REPORT_SCHEMA + 1)
    assert not decode_teardown_report(raw, expected_pid=_PID).committed


def test_a_record_from_another_child_is_not_read_as_this_one():
    assert not decode_teardown_report(_normal_reset_ok(), expected_pid=_PID + 1).committed


def test_short_record_reads_as_unknown():
    assert not decode_teardown_report(_normal_reset_ok()[:-1], expected_pid=_PID).committed


def test_payload_is_written_with_the_schema_word_cleared_and_the_pid_stamped():
    payload = encode_teardown_report_payload(_record(schema=TEARDOWN_REPORT_SCHEMA, child_pid=0), child_pid=_PID)

    assert len(payload) == SIMPLER_TEARDOWN_REPORT_BYTES
    assert payload[:4] == b"\x00\x00\x00\x00"
    assert struct.unpack_from("=i", payload, 4)[0] == _PID
    # Only committing the schema afterwards makes it readable.
    assert not decode_teardown_report(payload, expected_pid=_PID).committed
    committed = struct.pack("=I", TEARDOWN_REPORT_SCHEMA) + payload[4:]
    assert decode_teardown_report(committed, expected_pid=_PID).committed


def test_encode_rejects_a_wrongly_sized_record():
    with pytest.raises(ValueError, match="teardown report must be"):
        encode_teardown_report_payload(b"\x00" * 8, child_pid=_PID)


# --------------------------------------------------------------------------
# Validity identities the producer owes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        (
            "reserved bytes set",
            _record(reserved=1),
        ),
        (
            "reserved flag bit set",
            _record(flags=1 << 3),
        ),
        (
            "unknown path value",
            _record(path=9),
        ),
        (
            "unknown stage value",
            _record(stage=9),
        ),
        (
            "unknown reset api value",
            _record(api=9),
        ),
        (
            "api rc valid without an invocation",
            _record(flags=_API_RC_VALID, invocations=0),
        ),
        (
            "invocation without a valid api rc",
            _record(api=int(TeardownResetApi.RT_DEVICE_RESET), invocations=1),
        ),
        (
            "named api with no invocation",
            _record(api=int(TeardownResetApi.RT_DEVICE_RESET), flags=_API_RC_VALID, invocations=0),
        ),
        (
            "probe confirmed without the probe stage",
            _record(
                path=int(TeardownPath.FATAL),
                stage=int(TeardownResetStage.API_FAILED),
                api=int(TeardownResetApi.ACL_RESET_DEVICE_FORCE),
                flags=_API_RC_VALID | _SEQ_RC_VALID | _PROBE_CONFIRMED,
                invocations=1,
                attempts=1,
            ),
        ),
        (
            "probe confirmed with a non-zero recovery rc",
            _record(
                path=int(TeardownPath.FATAL),
                stage=int(TeardownResetStage.API_OK_PROBE_RUN),
                api=int(TeardownResetApi.ACL_RESET_DEVICE_FORCE),
                flags=_API_RC_VALID | _SEQ_RC_VALID | _PROBE_CONFIRMED,
                recovery_sequence_rc=7,
                invocations=1,
                attempts=1,
            ),
        ),
        (
            "recovery attempts on the normal path",
            _record(path=int(TeardownPath.NORMAL), attempts=1),
        ),
        (
            "ok stage with a non-zero api rc",
            _record(
                stage=int(TeardownResetStage.API_OK_NO_PROBE),
                api=int(TeardownResetApi.RT_DEVICE_RESET),
                flags=_API_RC_VALID,
                last_reset_api_rc=5,
                invocations=1,
            ),
        ),
    ],
)
def test_a_record_breaking_an_identity_reads_as_unknown(name, raw):
    # Not "partly usable": a producer this reader does not understand is a
    # reader that knows nothing, so the whole record is unknown.
    assert not decode_teardown_report(raw, expected_pid=_PID).committed, name


# --------------------------------------------------------------------------
# Mailbox layout mirrors
# --------------------------------------------------------------------------


def test_python_and_cpp_agree_on_the_record_placement():
    assert worker_module._OFF_TEARDOWN_REPORT == MAILBOX_OFF_TEARDOWN_REPORT
    assert worker_module._MAILBOX_ARGS_CAPACITY == MAILBOX_ARGS_CAPACITY
    assert worker_module._TASK_PROTOCOL_VERSION == MAILBOX_TASK_PROTOCOL_VERSION


def test_the_record_is_reserved_below_the_shutdown_word_and_above_the_args_blob():
    off = worker_module._OFF_TEARDOWN_REPORT
    # Reserved on every frame, like the shutdown word, so a task-args blob can
    # never reach it.
    assert worker_module._OFF_TASK_ARGS_BLOB + worker_module._MAILBOX_ARGS_CAPACITY == off
    assert off + SIMPLER_TEARDOWN_REPORT_BYTES <= worker_module._OFF_SHUTDOWN
    assert off % 8 == 0


def test_the_record_costs_the_args_blob_forty_eight_bytes():
    # The user-visible cost of the trailer: an args blob within this much of
    # the old limit is now rejected by the existing capacity check.
    assert worker_module._OFF_SHUTDOWN - worker_module._OFF_TEARDOWN_REPORT == 48


# --------------------------------------------------------------------------
# Publication never masks the teardown outcome
# --------------------------------------------------------------------------


class _RaisingWorker:
    def teardown_report_bytes(self):
        raise RuntimeError("module has no usable report")


class _EmptyWorker:
    def teardown_report_bytes(self):
        return None


class _ShortWorker:
    def teardown_report_bytes(self):
        return b"\x00" * 8


@pytest.mark.parametrize("fake", [_RaisingWorker(), _EmptyWorker(), _ShortWorker()])
def test_publication_swallows_its_own_failures(fake):
    # The record must never displace the teardown outcome the parent is about
    # to be told about, so a capture or a write that fails leaves the mailbox
    # untouched and raises nothing.
    buf = memoryview(bytearray(worker_module.MAILBOX_FRAME_SIZE))
    worker_module._publish_teardown_report(buf, fake)

    off = worker_module._OFF_TEARDOWN_REPORT
    assert bytes(buf[off : off + SIMPLER_TEARDOWN_REPORT_BYTES]) == b"\x00" * SIMPLER_TEARDOWN_REPORT_BYTES


# --------------------------------------------------------------------------
# Production exit paths: one finalize, publication in a finally
# --------------------------------------------------------------------------


class _RecordingWorker:
    """A chip worker whose teardown either returns or raises, counted."""

    def __init__(self, raises: BaseException | None = None):
        self.finalize_calls = 0
        self._raises = raises

    def finalize(self):
        self.finalize_calls += 1
        if self._raises is not None:
            raise self._raises

    def teardown_report_bytes(self):
        return _record(child_pid=0)


def test_child_exit_publishes_after_exactly_one_finalize():
    buf = memoryview(bytearray(worker_module.MAILBOX_FRAME_SIZE))
    cw = _RecordingWorker()

    worker_module._finalize_chip_worker_and_publish(buf, cast(Any, cw))

    # Both child exits that own a teardown share this helper, so the count is
    # what keeps the startup-failure path from gaining a second reset.
    assert cw.finalize_calls == 1
    off = worker_module._OFF_TEARDOWN_REPORT
    raw = bytes(buf[off : off + SIMPLER_TEARDOWN_REPORT_BYTES])
    assert decode_teardown_report(raw, expected_pid=os.getpid()).committed


def test_a_raising_teardown_still_publishes_and_still_raises():
    buf = memoryview(bytearray(worker_module.MAILBOX_FRAME_SIZE))
    boom = RuntimeError("device teardown failed")
    cw = _RecordingWorker(raises=boom)

    with pytest.raises(RuntimeError) as caught:
        worker_module._finalize_chip_worker_and_publish(buf, cast(Any, cw))

    # The teardown's own error reaches the caller unchanged, and the record it
    # captured before throwing is still published.
    assert caught.value is boom
    assert cw.finalize_calls == 1
    off = worker_module._OFF_TEARDOWN_REPORT
    raw = bytes(buf[off : off + SIMPLER_TEARDOWN_REPORT_BYTES])
    assert decode_teardown_report(raw, expected_pid=os.getpid()).committed


# --------------------------------------------------------------------------
# Late reap through the cleanup journal
# --------------------------------------------------------------------------


class _FakeShm:
    """Enough of SharedMemory for the journal's reap-then-release sequence."""

    def __init__(self, payload: bytes | None):
        self._store = bytearray(worker_module.MAILBOX_FRAME_SIZE)
        if payload is not None:
            off = worker_module._OFF_TEARDOWN_REPORT
            self._store[off : off + SIMPLER_TEARDOWN_REPORT_BYTES] = payload
        self._view = memoryview(self._store)
        self.closed = False
        self.unlinked = False

    @property
    def buf(self):
        return None if self.closed else self._view

    def close(self):
        self.closed = True

    def unlink(self):
        self.unlinked = True


def _journal_one_chip_child(shm, pid, reports):
    journal = worker_module.CleanupJournal()
    worker_module._journal_child_survivors(journal, [], [], [shm], [pid], [], [], set(), reports)
    return journal


def test_a_child_reaped_by_the_journal_still_yields_its_report(monkeypatch):
    # The child outlived the close deadline, so the first reap pass never read
    # its mailbox. The journal retry owns that read.
    pid = 31337
    shm = _FakeShm(_record(child_pid=pid, stage=int(TeardownResetStage.NOT_ATTEMPTED)))
    reports: dict[int, TeardownReport] = {}
    monkeypatch.setattr(worker_module.os, "waitpid", lambda _pid, _flags: (pid, 0))

    assert _journal_one_chip_child(shm, pid, reports).drive() is None

    assert reports[pid].committed
    assert reports[pid].child_pid == pid
    # Read before release, which is the only order in which the record exists.
    assert shm.closed
    assert shm.unlinked


def test_a_journal_reaped_child_that_published_nothing_is_unknown_not_absent(monkeypatch):
    pid = 31338
    shm = _FakeShm(None)
    reports: dict[int, TeardownReport] = {}
    monkeypatch.setattr(worker_module.os, "waitpid", lambda _pid, _flags: (pid, 0))

    assert _journal_one_chip_child(shm, pid, reports).drive() is None

    # Present and uncommitted: it was reaped, so "never reaped" would be wrong.
    assert pid in reports
    assert not reports[pid].committed


def test_a_still_live_child_leaves_no_entry_and_keeps_its_shm(monkeypatch):
    pid = 31339
    shm = _FakeShm(_record(child_pid=pid))
    reports: dict[int, TeardownReport] = {}
    monkeypatch.setattr(worker_module.os, "waitpid", lambda _pid, _flags: (0, 0))

    assert _journal_one_chip_child(shm, pid, reports).drive() is not None

    # Absent means never reaped, and the mailbox is untouched for the retry.
    assert pid not in reports
    assert not shm.closed
    assert not shm.unlinked


def test_a_journal_retry_after_a_close_failure_keeps_the_first_read(monkeypatch):
    pid = 31340
    shm = _FakeShm(_record(child_pid=pid, device_id=5))
    reports: dict[int, TeardownReport] = {}
    monkeypatch.setattr(worker_module.os, "waitpid", lambda _pid, _flags: (pid, 0))

    close_failure = OSError("shm close failed")
    original_close = shm.close
    calls = {"n": 0}

    def _flaky_close():
        calls["n"] += 1
        if calls["n"] == 1:
            raise close_failure
        original_close()

    shm.close = _flaky_close
    journal = _journal_one_chip_child(shm, pid, reports)

    assert journal.drive() is close_failure
    first = reports[pid]
    assert first.committed
    assert first.device_id == 5

    # The retry must not replace what the first read established, and the close
    # error still surfaces exactly as it did before this record existed.
    off = worker_module._OFF_TEARDOWN_REPORT
    shm._store[off : off + 4] = b"\x00\x00\x00\x00"
    assert journal.drive() is None
    assert reports[pid] is first


def test_non_chip_survivors_get_no_report_entry(monkeypatch):
    pid = 31341
    shm = _FakeShm(_record(child_pid=pid))
    reports: dict[int, TeardownReport] = {}
    monkeypatch.setattr(worker_module.os, "waitpid", lambda _pid, _flags: (pid, 0))

    journal = worker_module.CleanupJournal()
    # A sub worker runs no device teardown, so its mailbox is never read for one.
    worker_module._journal_child_survivors(journal, [shm], [pid], [], [], [], [], set(), reports)
    assert journal.drive() is None

    assert reports == {}
    assert shm.closed


# --------------------------------------------------------------------------
# Which runtime publishes a report
# --------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# The onboard platform runner is shared, so the accessor is exported by every
# onboard module. What decides coverage is which module binds the capability
# strongly: only the approved runtime overrides the weak default.
_PUBLISHING_MODULE = ("a2a3", "host_build_graph")
_NON_PUBLISHING_MODULES = [
    ("a2a3", "tensormap_and_ringbuffer"),
    ("a5", "host_build_graph"),
    ("a5", "tensormap_and_ringbuffer"),
]


def _symbol_bindings(arch: str, runtime: str) -> dict[str, str]:
    library = _PROJECT_ROOT / "build" / "lib" / arch / "onboard" / runtime / "libhost_runtime.so"
    if not library.exists():
        pytest.skip(f"{arch}/onboard/{runtime} is not built in this tree")
    result = subprocess.run(["nm", "-D", "--defined-only", str(library)], check=True, capture_output=True, text=True)
    bindings = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            bindings[parts[-1]] = parts[-2]
    return bindings


def test_only_the_approved_runtime_binds_the_teardown_capability_strongly():
    arch, runtime = _PUBLISHING_MODULE
    bindings = _symbol_bindings(arch, runtime)
    assert bindings.get("get_teardown_report") == "T"
    # "T" is the runtime's own override; "W" would be the weak default that
    # answers unsupported.
    assert bindings.get("teardown_report_supported_impl") == "T"


@pytest.mark.parametrize(("arch", "runtime"), _NON_PUBLISHING_MODULES)
def test_other_onboard_runtimes_keep_the_unsupported_default(arch, runtime):
    bindings = _symbol_bindings(arch, runtime)
    # The accessor exists because the platform TU is shared; what makes the
    # answer unsupported is that the capability is still the weak default.
    assert bindings.get("get_teardown_report") == "T"
    assert bindings.get("teardown_report_supported_impl") == "W"
