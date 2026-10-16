# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A remote L3 session's shutdown, as the launcher reads it back.

A session ends by three separable events: the peer reads the shutdown command,
its process exits, and the resources that process owns are released. A single
host-side outcome folds all three together, so "the command never arrived",
"the process is gone" and "the release failed" collapse into one value — and a
release failure nobody recorded is indistinguishable from success.

This module is the record that keeps them apart, plus the JSON codec a session
runner writes its own half with and its launcher reads back.

**Observation only.** No field asserts that a peer's resources are gone.
``CleanupOutcome.SUCCEEDED`` is what that process's own ``Worker.close()``
returned, nothing more, and a record that never arrived asserts nothing at all.

Each route reaches only as far as its launcher can see. ``rank`` names whose
release the cleanup half describes, because on an MPI group only rank 0 shares a
filesystem with the launcher: the other ranks' releases are visible solely as
the group's exit status.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum

__all__ = [
    "CleanupOutcome",
    "GROUP_CLEANUP_RECORD_NAME",
    "MAX_FAILURES",
    "MAX_FAILURE_BYTES",
    "MAX_RECORD_BYTES",
    "ProcessExit",
    "RECORD_SCHEMA",
    "SessionCleanupReport",
    "ShutdownCommand",
    "decode_runner_record",
    "encode_runner_record",
    "format_cleanup_failure",
]

# Bumped whenever a field changes meaning. A reader that does not recognise the
# schema reads the record as absent rather than as a weaker claim.
RECORD_SCHEMA = 1

# An MPI group carries its record as a file beside the group manifest, in the
# directory the launcher created and still owns. Rank 0 is the only rank
# guaranteed to share that filesystem — it is the rank that reads the manifest —
# so it is the only one that can write there, and the file name is fixed so the
# launcher can look for it without the rank reporting a path back.
GROUP_CLEANUP_RECORD_NAME = "cleanup-rank0.json"

# A runner writes its record on a teardown path, so the record has to stay
# small enough to land in one write whatever the failure looked like. A
# traceback is truncated to fit rather than dropped.
MAX_FAILURES = 16
MAX_FAILURE_BYTES = 1024
# What a reader will take from a record's carrier before giving up on it. The
# bound is the failure list plus room for the JSON around it, so a record this
# module could have written always fits and a file that grew some other way is
# rejected rather than parsed.
MAX_RECORD_BYTES = 2 * MAX_FAILURES * MAX_FAILURE_BYTES


class ShutdownCommand(IntEnum):
    """Whether the peer is known to have read the shutdown command.

    ``NOT_OBSERVED`` is a positive fact, not a weaker ``UNKNOWN``: the peer's
    command loop ended some other way — the transport reached EOF — so no
    command reached it. ``UNKNOWN`` means no record came back at all.
    """

    UNKNOWN = 0
    NOT_OBSERVED = 1
    OBSERVED = 2


class ProcessExit(IntEnum):
    """Whether the launcher reaped the session process.

    ``RUNNING`` is what the launcher saw when it stopped waiting, so the
    process may have exited since; ``exit_status`` is meaningful only for
    ``EXITED``.
    """

    UNKNOWN = 0
    RUNNING = 1
    EXITED = 2


class CleanupOutcome(IntEnum):
    """What the session process's own resource release returned."""

    UNKNOWN = 0
    SUCCEEDED = 1
    FAILED = 2


@dataclass(frozen=True)
class SessionCleanupReport:
    """One session's shutdown, with the three events kept apart.

    ``committed`` false is the unknown answer for the runner-written half: the
    launcher looked where the record belongs and found nothing valid — the
    process died before writing it, the route has no way to carry it back, or
    the record did not decode. ``command`` and ``cleanup`` are then
    ``UNKNOWN``. ``process_exit`` is the launcher's own observation and stands
    whether or not that half arrived, which is what makes a process that was
    reaped without leaving a record distinguishable from one still running.

    A worker absent from the ledger is a session this Worker never closed,
    which is a different thing from a committed-false record.
    """

    worker_id: int = -1
    committed: bool = False
    command: ShutdownCommand = ShutdownCommand.UNKNOWN
    process_exit: ProcessExit = ProcessExit.UNKNOWN
    exit_status: int | None = None
    cleanup: CleanupOutcome = CleanupOutcome.UNKNOWN
    failures: tuple[str, ...] = ()
    # What the launcher itself hit while closing this session — a command it
    # could not deliver, a process it could not reap, a mailbox it could not
    # unlink. Kept apart from ``failures``, which is only ever what the peer's
    # own release reported.
    notes: tuple[str, ...] = ()
    # This session's rank inside its group, or -1 on a route that runs one
    # process per session. Only rank 0 of an MPI group carries a release
    # result; see ``Worker.session_cleanup_reports``.
    rank: int = -1

    @property
    def cleanup_proven(self) -> bool:
        """Whether a release was reported, successful or not.

        False covers both "nothing came back" and "no record was written", so
        it is the test for "this session's release is unaccounted for" rather
        than for success.
        """
        return self.cleanup is not CleanupOutcome.UNKNOWN

    def describe(self) -> str:
        """One line naming all three events, for a warning or an error."""
        parts = [
            f"worker {self.worker_id}",
            f"command={self.command.name.lower()}",
            f"process={self.process_exit.name.lower()}" + ("" if self.exit_status is None else f"({self.exit_status})"),
            f"cleanup={self.cleanup.name.lower()}" + ("" if self.rank < 0 else f"@rank{self.rank}"),
        ]
        line = " ".join(parts)
        detail = "; ".join(self.failures + self.notes)
        return line if not detail else f"{line}: {detail}"


def format_cleanup_failure(prefix: str, exc: BaseException) -> str:
    """A bounded one-line summary of a release failure.

    The record is a ledger entry rather than a diagnostic: the full traceback
    belongs on the runner's stderr, which the launcher's own log already
    captures, so this keeps only what identifies the failure.
    """
    text = f"{prefix}: {type(exc).__name__}: {exc}".replace("\n", " ").strip()
    encoded = text.encode("utf-8", errors="replace")[:MAX_FAILURE_BYTES]
    return encoded.decode("utf-8", errors="replace")


def _bounded_failures(failures: Iterable[str]) -> tuple[str, ...]:
    """Trim a failure list to what a single record may carry."""
    trimmed: list[str] = []
    for item in tuple(failures)[:MAX_FAILURES]:
        encoded = str(item).replace("\n", " ").strip().encode("utf-8", errors="replace")[:MAX_FAILURE_BYTES]
        trimmed.append(encoded.decode("utf-8", errors="replace"))
    return tuple(trimmed)


def encode_runner_record(
    *,
    worker_id: int,
    command: ShutdownCommand,
    cleanup: CleanupOutcome,
    failures: Iterable[str] = (),
    rank: int = -1,
) -> bytes:
    """The half a session runner writes about itself, as one newline-ended line.

    The runner never reports its own exit; only its launcher can observe that.
    ``UNKNOWN`` is rejected for both events because a runner that reached this
    point knows how its loop ended and what its release returned — writing
    ``UNKNOWN`` would claim a record exists while asserting nothing.
    """
    command = ShutdownCommand(command)
    cleanup = CleanupOutcome(cleanup)
    if command is ShutdownCommand.UNKNOWN or cleanup is CleanupOutcome.UNKNOWN:
        raise ValueError("a session runner's own cleanup record cannot report UNKNOWN")
    bounded = _bounded_failures(failures)
    if (cleanup is CleanupOutcome.FAILED) != bool(bounded):
        raise ValueError("a session cleanup record must list a failure exactly when it reports FAILED")
    payload = {
        "schema": RECORD_SCHEMA,
        "worker_id": int(worker_id),
        "rank": int(rank),
        "command": int(command),
        "cleanup": int(cleanup),
        "failures": list(bounded),
    }
    return json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n"


def decode_runner_record(raw: bytes) -> SessionCleanupReport | None:
    """Validate and decode one runner-written record, or ``None`` if unusable.

    Anything short of wholly valid reads as absent: the launcher then reports
    ``committed`` false, which is the honest answer, rather than a record whose
    fields it half-trusts.
    """
    try:
        payload = json.loads(bytes(raw).decode("utf-8"))
    except BaseException:  # noqa: BLE001 -- any malformed record is simply absent
        return None
    if not isinstance(payload, dict) or payload.get("schema") != RECORD_SCHEMA:
        return None
    try:
        command = ShutdownCommand(int(payload["command"]))
        cleanup = CleanupOutcome(int(payload["cleanup"]))
        worker_id = int(payload["worker_id"])
        rank = int(payload.get("rank", -1))
        raw_failures = payload.get("failures", [])
    except BaseException:  # noqa: BLE001
        return None
    if not isinstance(raw_failures, list) or any(not isinstance(item, str) for item in raw_failures):
        return None
    if command is ShutdownCommand.UNKNOWN or cleanup is CleanupOutcome.UNKNOWN:
        return None
    failures = _bounded_failures(raw_failures)
    # The identity the writer is obliged to maintain; a record that breaks it
    # is a producer this reader does not understand.
    if (cleanup is CleanupOutcome.FAILED) != bool(failures):
        return None
    return SessionCleanupReport(
        worker_id=worker_id,
        committed=True,
        command=command,
        cleanup=cleanup,
        failures=failures,
        rank=rank,
    )
