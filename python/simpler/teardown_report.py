# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The chip child's device-teardown observation, as the parent reads it back.

A ``Worker`` closes by shutting its chip children down, and the child process
is what actually calls the device reset. Only that process sees which reset
entry ran, what it returned, whether a recovery wrapper with a post-reset probe
sat above it, and what the teardown as a whole returned; a single host-side
error code folds all four together, so "no reset ran", "the reset failed" and
"the reset returned zero but the probe did not confirm" become one value.

This module is the wire codec and the reader's validation for the record that
keeps them apart.

**Observation only.** No field asserts that device work has stopped, and a
confirmed reset invalidates that device generation's allocations rather than
making an old device pointer reusable.

Only a2a3 `host_build_graph` onboard publishes a teardown report today. The
onboard platform runner is shared across runtimes, so the runtime itself opts
in; one that has not answers "nothing recorded", which reads back as an
uncommitted record.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum, IntFlag

from .task_interface import SIMPLER_TEARDOWN_REPORT_BYTES, TEARDOWN_REPORT_SCHEMA

__all__ = [
    "TeardownPath",
    "TeardownResetStage",
    "TeardownResetApi",
    "TeardownFlags",
    "TeardownReport",
    "decode_teardown_report",
    "encode_teardown_report_payload",
]


class TeardownPath(IntEnum):
    """Which teardown branch ``finalize_device`` took — a branch, not a severity."""

    UNKNOWN = 0
    NORMAL = 1
    FATAL = 2


class TeardownResetStage(IntEnum):
    """How far the **last** reset attempt got.

    ``API_OK_NO_PROBE`` and ``API_OK_PROBE_RUN`` are orthogonal facts rather
    than an ordering: the first is a reset call that returned 0 with nothing
    checked afterwards, the second is one with a post-reset probe behind it.

    ``PREAMBLE_FAILED`` describes that one attempt. Whether any attempt reached
    a reset call is ``reset_api_invocations_total`` / ``last_reset_api_rc``.
    """

    UNKNOWN = 0
    NOT_ATTEMPTED = 1
    REFUSED = 2
    PREAMBLE_FAILED = 3
    API_FAILED = 4
    API_OK_NO_PROBE = 5
    API_OK_PROBE_RUN = 6


class TeardownResetApi(IntEnum):
    """Which reset entry the recorded invocations used."""

    NONE = 0
    RT_DEVICE_RESET = 1
    ACL_RESET_DEVICE = 2
    ACL_RESET_DEVICE_FORCE = 3


class TeardownFlags(IntFlag):
    LAST_RESET_API_RC_VALID = 1 << 0
    RECOVERY_SEQUENCE_RC_VALID = 1 << 1
    PROBE_CONFIRMED = 1 << 2


_RESERVED_FLAG_MASK = ~0x7 & 0xFF

# schema u32 | child_pid i32 | device_id i32 | path u8 | stage u8 | api u8 |
# flags u8 | last_reset_api_rc i32 | recovery_sequence_rc i32 | teardown_rc i32 |
# invocations u16 | attempts u16 | reserved u64
_WIRE = struct.Struct("=IiiBBBBiiiHHQ")
assert _WIRE.size == SIMPLER_TEARDOWN_REPORT_BYTES, "teardown report codec disagrees with the C++ wire size"

_SCHEMA_BYTES = 4


@dataclass(frozen=True)
class TeardownReport:
    """One chip child's teardown observation, or the fact that there is none.

    ``committed`` false is the unknown answer: the child was reaped and its
    mailbox read, but nothing valid had been published — it died before or
    during the write, or the backend records no teardown. Every other field is
    then at its zero value and means nothing. A child the parent never reaped
    produces no entry at all, which is a different thing from this.
    """

    committed: bool = False
    child_pid: int = 0
    device_id: int = -1
    path: TeardownPath = TeardownPath.UNKNOWN
    reset_stage: TeardownResetStage = TeardownResetStage.UNKNOWN
    reset_api: TeardownResetApi = TeardownResetApi.NONE
    last_reset_api_rc: int | None = None
    recovery_sequence_rc: int | None = None
    teardown_rc: int = 0
    reset_api_invocations_total: int = 0
    recovery_attempts_total: int = 0
    probe_confirmed: bool = False

    @property
    def reset_api_invoked(self) -> bool:
        """Whether any reset entry actually ran, which ``last_reset_api_rc is None`` also states."""
        return self.reset_api_invocations_total > 0


def encode_teardown_report_payload(report_bytes: bytes, *, child_pid: int) -> bytes:
    """The record with this child's pid stamped in and its schema word cleared.

    The producer writes this first and releases the schema word last, so a
    reader either sees the whole record or none of it. The pid is stamped here
    rather than recorded by the runner because it is the transport's
    generation key, not something a device teardown observes.
    """
    if len(report_bytes) != SIMPLER_TEARDOWN_REPORT_BYTES:
        raise ValueError(f"teardown report must be {SIMPLER_TEARDOWN_REPORT_BYTES} bytes, got {len(report_bytes)}")
    stamped = bytearray(report_bytes)
    stamped[0:_SCHEMA_BYTES] = b"\x00" * _SCHEMA_BYTES
    struct.pack_into("=i", stamped, _SCHEMA_BYTES, child_pid)
    return bytes(stamped)


def decode_teardown_report(raw: bytes, *, expected_pid: int) -> TeardownReport:
    """Validate and decode one record; anything short of wholly valid is unknown.

    ``expected_pid`` is the generation key: the parent knows which child it
    reaped, and a record carrying any other pid belongs to a different child,
    so it is not read as this one's.
    """
    unknown = TeardownReport()
    if len(raw) != SIMPLER_TEARDOWN_REPORT_BYTES:
        return unknown
    (
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
    ) = _WIRE.unpack(raw)

    if schema != TEARDOWN_REPORT_SCHEMA or reserved != 0:
        return unknown
    if child_pid != expected_pid:
        return unknown
    if flags & _RESERVED_FLAG_MASK:
        return unknown
    try:
        path_value = TeardownPath(path)
        stage_value = TeardownResetStage(stage)
        api_value = TeardownResetApi(api)
    except ValueError:
        return unknown

    api_rc_valid = bool(flags & TeardownFlags.LAST_RESET_API_RC_VALID)
    sequence_rc_valid = bool(flags & TeardownFlags.RECOVERY_SEQUENCE_RC_VALID)
    probe_confirmed = bool(flags & TeardownFlags.PROBE_CONFIRMED)

    # The identities the producer is obliged to maintain. A record that breaks
    # one is not partly usable: it is a producer this reader does not
    # understand, so it reads as unknown rather than as a weaker claim.
    if api_rc_valid != (invocations > 0):
        return unknown
    if (api_value is TeardownResetApi.NONE) != (invocations == 0):
        return unknown
    if probe_confirmed and not (
        stage_value is TeardownResetStage.API_OK_PROBE_RUN and sequence_rc_valid and recovery_sequence_rc == 0
    ):
        return unknown
    if attempts > 0 and path_value is not TeardownPath.FATAL:
        return unknown
    if stage_value in (TeardownResetStage.API_OK_NO_PROBE, TeardownResetStage.API_OK_PROBE_RUN):
        if not api_rc_valid or last_reset_api_rc != 0:
            return unknown

    return TeardownReport(
        committed=True,
        child_pid=child_pid,
        device_id=device_id,
        path=path_value,
        reset_stage=stage_value,
        reset_api=api_value,
        last_reset_api_rc=last_reset_api_rc if api_rc_valid else None,
        recovery_sequence_rc=recovery_sequence_rc if sequence_rc_valid else None,
        teardown_rc=teardown_rc,
        reset_api_invocations_total=invocations,
        recovery_attempts_total=attempts,
        probe_confirmed=probe_confirmed,
    )
