#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Place device-clock records on the Host timeline by span containment.

Every level of the runtime wraps the level below it in a Host span, which makes
one statement that needs no clock calibration: *the inner work happened inside
the outer window*. For a Chip Swimlane capture the chain is

    chip.run.runner_run                     Host CLOCK_MONOTONIC
      └─ chip.run.runner_run.device_wall    device sys-counter, ts = 0
           └─ the capture's records         device sys-counter, absolute cycles

so an event's placement error is the outer window's *slack* — how much wider it
is than the work it brackets — and that term is measured, never estimated:

    slack     = outer_duration - device_extent
    placement in [outer_start, outer_start + slack]

Nothing here reads a Host/Device clock anchor. Two unknowns stand between a raw
cycle and the Host axis, and containment bounds both:

``origin_cycles``
    The cycle that device-phase time zero sits on. The host log carries the
    ``device_wall`` sub-phase windows on a relative device-phase timeline while
    the capture carries the same windows in absolute cycles, so each such pair
    brackets the offset. The pairs are intersected, which both narrows the
    result and checks it: an empty intersection means the two artifacts do not
    describe the same run.

``placement_ns``
    The Host ns that device-phase zero sits on, bounded by ``runner_run``
    containing everything drawn.

Same-host Ranks need nothing further: ``runner_run`` endpoints are
CLOCK_MONOTONIC and same-host cross-process comparable
(``docs/dfx/host-trace.md``), so each Rank's window is already on one axis.

The bound cannot be wrong, only loose. A reader who needs to know whether two
events are separable compares their gap against the sum of their slacks; a gap
below that sum is undecided, not zero.
"""

import math
import sys
from bisect import bisect_right
from dataclasses import dataclass, replace
from functools import cached_property
from itertools import permutations
from typing import Optional

from simpler_setup.tools.strace_timing import node_span_leaf

RUN_SPAN = "chip.run"
RUNNER_SPAN = "chip.run.runner_run"
DEVICE_WALL_SPAN = "chip.run.runner_run.device_wall"

_PHASE_PREFIX = DEVICE_WALL_SPAN + "."

# The dispatch this invocation served, as the root `chip.run` span reports it
# (`c_api_shared.cpp`, `emit_native_run_host_wall`) and as the capture's
# `dispatch_identity.json` sidecar reports the same four numbers under its own
# names. This is what actually names one artifact in the other, so it is the
# pairing's first choice and the window fit only covers what it cannot answer.
_HOST_IDENTITY_FIELDS = ("run_id", "dispatch_id", "slot_id", "generation")
_CAPTURE_IDENTITY_FIELDS = ("run_id", "endpoint_dispatch_id", "pipeline_slot", "pipeline_generation")

# The `device_wall` sub-phases a capture also records in absolute cycles, each
# paired with the stream that covers the same window. Most-constraining first:
# `sched` brackets the whole scheduler dispatch window and its record stream
# covers nearly all of it, while the orchestrator's per-submit records occupy
# only part of `orch`.
_JOIN_STREAMS = (
    ("sched", "aicpu_scheduler_phases"),
    ("orch", "aicpu_orchestrator_phases"),
)

_NS_PER_S = 1_000_000_000

# What two counters' rates can differ by, relative. This is the one term
# containment does not measure: it handles a clock's offset and not its rate,
# and `#2134` states the magnitude - 10-100 ppm between two machines - and asks
# for it in the published error rather than left implicit. The loose end is
# taken, because a bound drawn tighter than the parts would not hold.
MAX_RELATIVE_RATE = 100e-6

# The window fit scores whole assignments, so its cost is the number of
# *assignments*, `P(candidate windows, unidentified captures)` — not the number
# of captures. A Host log holds every invocation of the run, so that product
# grows with the run's length and has to be bounded here rather than by a limit
# on Ranks: 4 captures against 100 windows is 94 million assignments, which was
# measured at ~231 s before this cap existed. Above the cap the pairing refuses
# instead of grinding, because the identity join below makes the fit a fallback
# for captures that predate the sidecar, not the normal path.
_MAX_ASSIGNMENTS = 200_000


class ContainmentError(ValueError):
    """Raised when the two artifacts cannot describe one contained run."""


def _ns_to_cycles(ns, frequency_hz):
    return ns * frequency_hz / _NS_PER_S


def _cycles_to_ns(cycles, frequency_hz):
    return cycles * _NS_PER_S / frequency_hz


@dataclass(frozen=True)
class HostWindow:
    """One invocation's Host-side bracket around its device execution.

    ``phases`` holds the ``device_wall`` sub-phase spans as
    ``name -> (ts_ns, dur_ns)`` on the device-phase timeline, whose zero is the
    earliest sub-phase start (``DeviceRunnerBase::read_device_wall_ns``).
    ``device_wall`` itself is emitted at ``ts = 0`` by convention and its own
    start is not published, so it bounds a duration and never a position.

    ``identity`` is the dispatch the root ``chip.run`` span names, or ``None``
    for an invocation that carries none — a run outside the prepared-native
    path, or a log from before the attributes existed.
    """

    pid: int
    inv: int
    start_ns: int
    duration_ns: int
    device_wall_ns: int
    phases: dict[str, tuple[int, int]]
    identity: Optional[tuple[int, ...]] = None

    @property
    def end_ns(self):
        return self.start_ns + self.duration_ns


@dataclass(frozen=True)
class CaptureWindows:
    """The device-clock extent of one Chip Swimlane capture, in raw cycles."""

    frequency_hz: int
    extent: tuple[int, int]
    windows: dict[str, tuple[int, int]]

    @property
    def extent_cycles(self):
        return self.extent[1] - self.extent[0]


@dataclass(frozen=True)
class Join:
    """Where the capture's absolute cycles sit on the device-phase timeline."""

    origin_cycles: float
    interval_cycles: tuple[float, float]
    sources: tuple[str, ...]

    @property
    def residual_ns(self):
        """Width of the surviving offset interval — the join's own error bar."""
        return self.interval_cycles[1] - self.interval_cycles[0]

    def metadata(self, frequency_hz):
        return {
            "sources": list(self.sources),
            "origin_cycles": int(round(self.origin_cycles)),
            "residual_ns": int(round(_cycles_to_ns(self.residual_ns, frequency_hz))),
        }


@dataclass(frozen=True)
class Placement:
    """One capture's provable position on the Host CLOCK_MONOTONIC axis."""

    host: HostWindow
    capture: Optional[CaptureWindows]
    join: Optional[Join]
    lo_phase_ns: float
    hi_phase_ns: float

    @property
    def head_phase_ns(self):
        """The device-phase instant the placed block starts at.

        Zero is the earliest sub-phase start, so device activity begins there
        even when the first thing drawn comes later. A capture whose records
        precede every sub-phase moves it earlier.
        """
        return min(0.0, self.lo_phase_ns)

    @property
    def extent_ns(self):
        """How much device time must fit inside the Host window.

        ``device_wall`` brackets the whole on-NPU run and so is a lower bound on
        the extent even when less than that is drawn; what is drawn bounds it
        from the other side whenever a record sits past the run wall's own end.
        """
        return max(self.host.device_wall_ns, self.hi_phase_ns) - self.head_phase_ns

    @property
    def slack_ns(self):
        return self.host.duration_ns - self.extent_ns

    @property
    def place_lo_ns(self):
        """Earliest Host ns the placed block can start at — the placement used."""
        return float(self.host.start_ns)

    @property
    def place_hi_ns(self):
        return self.host.start_ns + self.slack_ns

    def phase_ns_to_host_ns(self, phase_ns):
        return self.place_lo_ns + (phase_ns - self.head_phase_ns)

    def map_cycles_to_host_ns(self, cycles):
        """Place one raw device cycle on the Host axis, at its lower bound."""
        if self.capture is None or self.join is None:
            raise ContainmentError("this placement carries no capture, so it maps no raw cycles")
        phase_ns = _cycles_to_ns(cycles - self.join.origin_cycles, self.capture.frequency_hz)
        return self.phase_ns_to_host_ns(phase_ns)

    def metadata(self):
        out = {
            "method": "span_containment_v1",
            "outer_span": RUNNER_SPAN,
            "outer_pid": self.host.pid,
            "outer_inv": self.host.inv,
            "outer_start_ns": self.host.start_ns,
            "outer_duration_ns": self.host.duration_ns,
            "device_wall_ns": self.host.device_wall_ns,
            "device_extent_ns": int(round(self.extent_ns)),
            "slack_ns": int(round(self.slack_ns)),
            "place_lo_ns": int(round(self.place_lo_ns)),
            "place_hi_ns": int(round(self.place_hi_ns)),
        }
        if self.join is not None and self.capture is not None:
            out["join"] = self.join.metadata(self.capture.frequency_hz)
        return out


def _host_identity(span):
    """The dispatch a root ``chip.run`` span names, or ``None`` if it names none.

    The four fields are written as one ``snprintf`` and are therefore all
    present or all absent; a partial set means the log is not what it claims and
    is treated as no identity rather than as a half key.
    """
    if span is None:
        return None
    found = {}
    for item in span.attrs.split():
        key, separator, value = item.partition("=")
        if separator and key in _HOST_IDENTITY_FIELDS:
            try:
                found[key] = int(value)
            except ValueError:
                return None
    if len(found) != len(_HOST_IDENTITY_FIELDS):
        return None
    return tuple(found[field] for field in _HOST_IDENTITY_FIELDS)


def capture_identity(dispatch_identity):
    """The same dispatch key, read off a capture's ``dispatch_identity.json``.

    The sidecar spells the four numbers differently — the Host log's
    ``dispatch_id`` / ``slot_id`` / ``generation`` are the frame's
    ``endpoint_dispatch_id`` / ``pipeline_slot`` / ``pipeline_generation``
    (``worker._write_dispatch_identity_sidecar``) — so the rename is undone here
    and nowhere else.
    """
    if not isinstance(dispatch_identity, dict):
        return None
    try:
        return tuple(int(dispatch_identity[field]) for field in _CAPTURE_IDENTITY_FIELDS)
    except (KeyError, TypeError, ValueError):
        return None


def host_windows(spans):
    """Group parsed ``[STRACE]`` spans into per-invocation Host windows.

    One invocation contributes a window only when it has both the outer
    ``runner_run`` span and the ``device_wall`` it brackets: an invocation that
    launched nothing has no device work to place.
    """
    by_invocation = {}
    for span in spans:
        key = (span.pid, span.inv)
        by_invocation.setdefault(key, {})[span.name] = span

    windows = []
    for (pid, inv), named in sorted(by_invocation.items()):
        runner = named.get(RUNNER_SPAN)
        device_wall = named.get(DEVICE_WALL_SPAN)
        if runner is None or device_wall is None:
            continue
        phases = {
            name[len(_PHASE_PREFIX) :]: (span.ts, span.dur)
            for name, span in named.items()
            if name.startswith(_PHASE_PREFIX) and span.dur > 0
        }
        windows.append(
            HostWindow(
                pid=pid,
                inv=inv,
                start_ns=runner.ts,
                duration_ns=runner.dur,
                device_wall_ns=device_wall.dur,
                phases=phases,
                identity=_host_identity(named.get(RUN_SPAN)),
            )
        )
    return windows


def _phase_records(raw):
    """Flatten a phase stream that is either per-thread lists or one list."""
    for entry in raw or []:
        if isinstance(entry, dict):
            yield entry
        elif isinstance(entry, list):
            for record in entry:
                if isinstance(record, dict):
                    yield record


def _cycle_bounds(values):
    kept = [value for value in values if value > 0]
    if not kept:
        return None
    return min(kept), max(kept)


def _stream_bounds(data, stream):
    return _cycle_bounds(
        [
            int(record.get(field, 0))
            for record in _phase_records(data.get(stream))
            for field in ("start_cycles", "end_cycles")
        ]
    )


def capture_windows(data):
    """Read one ``chip_swimlane_records.json`` document's device-cycle extent.

    Takes the raw document rather than the converter's decoded form: placement
    is decided before decoding, and the raw cycles are what the join needs.
    """
    metadata = data.get("metadata") or {}
    frequency_hz = int(metadata.get("clock_freq_hz") or 0)
    if frequency_hz <= 0:
        raise ContainmentError("capture metadata has no usable clock_freq_hz")

    # Everything the converter draws on the device clock has to be in here, or
    # the guarantee this module exists for — every record inside the window —
    # holds only for the streams that happened to be counted.
    cycles = []
    for row in data.get("aicore_tasks") or []:
        start_cycles, end_cycles = int(row[3]), int(row[4])
        # A v3 row's bar opens at its receive time, which is earlier than the
        # kernel start by `receive_to_start_cycles` (v2 rows have no column 5).
        receive_to_start_cycles = int(row[5]) if len(row) > 5 else 0
        cycles.extend((start_cycles - receive_to_start_cycles, start_cycles, end_cycles))
    for row in data.get("aicpu_tasks") or []:
        cycles.extend(int(value) for value in row[2:4])
    scheduler_tasks = data.get("scheduler_tasks") or {}
    for row in scheduler_tasks.get("records") or []:
        cycles.extend(int(value) for value in row[2:4])
    for stream in ("aicpu_scheduler_phases", "aicpu_orchestrator_phases"):
        for record in _phase_records(data.get(stream)):
            cycles.extend(int(record.get(field, 0)) for field in ("start_cycles", "end_cycles"))
    # Lifecycle records name their instants one field per event rather than as
    # a start/end pair, and the set grows with the control plane, so this reads
    # the suffix instead of a field list that would drift from the decoder's.
    for record in data.get("aicpu_lifecycle_records") or []:
        if not isinstance(record, dict):
            continue
        cycles.extend(
            value
            for key, value in record.items()
            # `bool` is an `int`, and a JSON `true` would otherwise read as cycle 1.
            if key.endswith("_cycles") and isinstance(value, int) and not isinstance(value, bool) and value > 0
        )

    extent = _cycle_bounds(cycles)
    if extent is None:
        raise ContainmentError("capture holds no device-clock records to place")

    windows = {}
    for short_name, stream in _JOIN_STREAMS:
        bounds = _stream_bounds(data, stream)
        if bounds is not None:
            windows[short_name] = bounds
    return CaptureWindows(frequency_hz=frequency_hz, extent=extent, windows=windows)


def join_origin(host, capture):
    """Bound the cycle that device-phase time zero sits on.

    Each ``(host-log window, capture stream)`` pair states that the records lie
    inside the phase window, which brackets the offset from both sides. The
    pairs are intersected; ``device_wall`` bounds the whole capture and is
    always available, so a level-1 capture with no phase streams still joins,
    just loosely.
    """
    frequency_hz = capture.frequency_hz
    lo = -math.inf
    hi = math.inf
    sources = []

    def narrow(window_ns, record_cycles, source):
        nonlocal lo, hi
        start_ns, duration_ns = window_ns
        first, last = record_cycles
        # records ⊆ window: first is at or after the window start, last at or
        # before its end. Both are statements about `origin`.
        hi = min(hi, first - _ns_to_cycles(start_ns, frequency_hz))
        lo = max(lo, last - _ns_to_cycles(start_ns + duration_ns, frequency_hz))
        sources.append(source)

    for short_name, _stream in _JOIN_STREAMS:
        window = host.phases.get(short_name)
        records = capture.windows.get(short_name)
        if window is not None and records is not None:
            narrow(window, records, short_name)

    # device_wall covers the whole run, but its own start is not published, so
    # it bounds only through the sub-phase timeline it contains: everything
    # drawn must fit inside a window of that length.
    narrow((0, host.device_wall_ns), capture.extent, "device_wall")

    if lo > hi:
        raise ContainmentError(
            f"pid {host.pid} inv {host.inv} and this capture cannot describe one run: "
            f"the device windows they report do not overlap"
        )
    return Join(origin_cycles=(lo + hi) / 2.0, interval_cycles=(lo, hi), sources=tuple(sources))


def place(host, capture=None, join=None):
    """Bound a run's device timeline inside its Host window.

    With a ``capture``, that capture's records are joined onto the same
    device-phase timeline and placed alongside the ``device_wall`` sub-phases
    the Host log carries. The cross-Rank merge is the only caller and always
    passes one; the capture-less form bounds the Host log's own ``clk=dev``
    spans and is exercised by the unit tests alone.
    """
    if capture is not None and join is None:
        join = join_origin(host, capture)

    drawn = []
    if capture is not None and join is not None:
        origin = join.origin_cycles
        drawn.extend(_cycles_to_ns(cycles - origin, capture.frequency_hz) for cycles in capture.extent)
    for start_ns, duration_ns in host.phases.values():
        drawn.extend((float(start_ns), float(start_ns + duration_ns)))
    # A run with neither records nor sub-phases still has its own wall, which is
    # the whole of what it did.
    lo_phase_ns = min(drawn) if drawn else 0.0
    hi_phase_ns = max(drawn) if drawn else float(host.device_wall_ns)

    placement = Placement(
        host=host,
        capture=capture,
        join=join,
        lo_phase_ns=lo_phase_ns,
        hi_phase_ns=hi_phase_ns,
    )
    if placement.slack_ns < 0:
        raise ContainmentError(
            f"pid {host.pid} inv {host.inv} draws {placement.extent_ns / 1000.0:.1f} us of device work "
            f"but its {RUNNER_SPAN} window is only {host.duration_ns / 1000.0:.1f} us wide"
        )
    return placement


def _pairing_cost(host, capture):
    """How much wider the Host log's windows are than the capture's own.

    Containment makes this non-negative for the true pairing, and the record
    stream covers nearly all of its phase window, so the true pairing is the
    small one. A pairing that is not even feasible costs infinity.
    """
    cost = 0.0
    matched = 0
    for short_name in capture.windows:
        window = host.phases.get(short_name)
        records = capture.windows.get(short_name)
        if window is None or records is None:
            continue
        window_ns = window[1]
        records_ns = _cycles_to_ns(records[1] - records[0], capture.frequency_hz)
        if records_ns > window_ns:
            return math.inf
        cost += window_ns - records_ns
        matched += 1
    if not matched:
        # Nothing but device_wall to go on: fall back to the same comparison on
        # the whole run, which still orders candidates but discriminates far
        # less. `pair_captures` reports the margin, so a weak pairing is visible.
        extent_ns = _cycles_to_ns(capture.extent_cycles, capture.frequency_hz)
        if extent_ns > host.device_wall_ns:
            return math.inf
        cost = host.device_wall_ns - extent_ns
    return cost


def _resolve_pins(hosts, keys, forced):
    """Turn ``--rank-pid`` values into Host-window indexes.

    A pin names either a pid or a ``(pid, inv)``. The bare form is the one a
    reader can write from memory, so it stays, but a process runs many
    invocations and only one of them is the capture's — so it resolves only
    when the pid contributed exactly one placeable window, and otherwise says
    which form to use instead of failing blankly.
    """
    forced = dict(forced or {})
    unknown = sorted(set(forced) - set(keys))
    if unknown:
        raise ContainmentError(f"pinned pairing names a capture that is not being merged: {unknown}")

    by_pid = {}
    by_invocation = {}
    for index, host in enumerate(hosts):
        by_pid.setdefault(host.pid, []).append(index)
        by_invocation[(host.pid, host.inv)] = index

    pinned = {}
    for key, target in sorted(forced.items()):
        pid, inv = target if isinstance(target, tuple) else (target, None)
        if inv is not None:
            index = by_invocation.get((pid, inv))
            if index is None:
                raise ContainmentError(
                    f"pinned pairing {key}={pid}:{inv} names an invocation with no placeable "
                    f"{RUNNER_SPAN} window in the Host logs"
                )
            pinned[key] = index
            continue
        candidates = by_pid.get(pid) or []
        if len(candidates) != 1:
            invocations = ", ".join(str(hosts[index].inv) for index in candidates) or "none"
            raise ContainmentError(
                f"pinned pairing {key}={pid} names a pid with {len(candidates)} placeable invocations "
                f"(inv {invocations}); name the invocation too, as {key}={pid}:INV"
            )
        pinned[key] = candidates[0]
    if len(set(pinned.values())) != len(pinned):
        raise ContainmentError("pinned pairing sends two captures to one Host invocation")
    return pinned


def _sidecar_candidates(hosts, keys, identities, host_pids, taken):
    """Narrow each capture to the windows its own sidecar can be describing.

    Two independent keys, and each answers half of the pairing:

    ``host_pid``
        Which *process*. Written by the ChipWorker child that served the
        dispatch, so it is that child's own pid — the one whose
        ``host.<pid>.log`` holds the window. The members of one group share
        every other key, so this is the only one that separates them.

    ``identity``
        Which *invocation* of that process. ``dispatch_id`` counts per worker,
        so a group's members all report the same one, but a process's
        successive dispatches do not.

    Together they name exactly one window. Either alone narrows; neither leaves
    the capture to the window fit. Returns ``{key: ([index, ...], (source, ...))}``
    for the captures that carried at least one.
    """
    candidates = {}
    for key in keys:
        matched = None
        sources = []
        host_pid = (host_pids or {}).get(key)
        if host_pid is not None:
            matched = {index for index, host in enumerate(hosts) if host.pid == host_pid}
            sources.append("host_pid")
        identity = (identities or {}).get(key)
        if identity is not None:
            by_identity = {index for index, host in enumerate(hosts) if host.identity == tuple(identity)}
            matched = by_identity if matched is None else matched & by_identity
            sources.append("dispatch")
        if matched is None:
            continue
        remaining = sorted(matched - taken)
        if not remaining and matched:
            raise ContainmentError(
                f"capture {key}'s sidecar names a Host invocation that a pinned pairing already claims; the pin "
                "and the capture disagree about which invocation ran it"
            )
        if remaining:
            candidates[key] = (remaining, tuple(sources))
    # Two captures claiming one window is a contradiction, not a tie: drop the
    # sidecar route for those and let the fit decide, which at least reports a
    # margin. Distinct Ranks carry distinct `host_pid`s, so this is a
    # malformed-sidecar path rather than an expected one.
    claimed = {}
    for key, (matched, _sources) in candidates.items():
        if len(matched) == 1:
            claimed.setdefault(matched[0], []).append(key)
    for contested in (keys for _index, keys in claimed.items() if len(keys) > 1):
        for key in contested:
            candidates.pop(key, None)
    return candidates


def pair_captures(hosts, captures, *, forced=None, identities=None, host_pids=None, margin_ratio=4.0):
    """Pair each capture with the Host invocation that ran it.

    The capture's ``dispatch_identity.json`` names both halves of the answer —
    ``host_pids`` the process, ``identities`` the invocation of it — so where
    the sidecar is present the pairing is exact and the rest of this function
    never runs. Without it, an old capture falls back to reading the pairing
    out of the timing: a capture's phase windows are the same windows the Host
    log recorded, to within the few µs between "the phase opened" and "the
    first record inside it", while every other candidate is off by the
    difference between two Ranks' workloads. That fallback cannot separate two
    Ranks of the same shape, and says so.

    Returns ``(pairs, diagnostics)`` where ``pairs`` maps each capture key to
    its ``HostWindow``.
    """
    keys = sorted(captures)
    if len(keys) > len(hosts):
        raise ContainmentError(f"{len(keys)} captures but only {len(hosts)} Host invocations to place them in")

    pinned = _resolve_pins(hosts, keys, forced)
    taken = set(pinned.values())
    narrowed = _sidecar_candidates(hosts, [key for key in keys if key not in pinned], identities, host_pids, taken)
    by_sidecar = {key: matched[0] for key, (matched, _sources) in narrowed.items() if len(matched) == 1}
    taken |= set(by_sidecar.values())

    settled = {**pinned, **by_sidecar}
    free_keys = [key for key in keys if key not in settled]

    # A capture the sidecar narrowed to a few windows is scored against those
    # only; one that carried no sidecar is scored against everything still free.
    def candidates_for(key):
        narrowing = narrowed.get(key)
        return range(len(hosts)) if narrowing is None else narrowing[0]

    free_indexes = sorted({index for key in free_keys for index in candidates_for(key) if index not in taken})
    costs = {(key, index): _pairing_cost(hosts[index], captures[key]) for key in free_keys for index in free_indexes}
    allowed = {key: set(candidates_for(key)) & set(free_indexes) for key in free_keys}

    assignment_count = math.perm(len(free_indexes), len(free_keys))
    if assignment_count > _MAX_ASSIGNMENTS:
        raise ContainmentError(
            f"{len(free_keys)} capture(s) carry no dispatch_identity.json and the Host logs hold "
            f"{len(free_indexes)} candidate {RUNNER_SPAN} windows, which is {assignment_count} ways of matching "
            "them — too many to score. Re-run so the captures carry the sidecar, narrow the logs with --host-log, "
            "or pin the pairing with --rank-pid RANK=PID:INV."
        )

    # Score whole assignments, not one capture at a time: what a reader needs to
    # trust is that no *other* way of matching the two artifacts fits nearly as
    # well.
    def total(assignment):
        return sum(costs[(key, index)] for key, index in assignment.items())

    ranked_assignments = []
    for candidate in permutations(free_indexes, len(free_keys)):
        assignment = dict(zip(free_keys, candidate))
        if any(index not in allowed[key] for key, index in assignment.items()):
            continue
        ranked_assignments.append((total(assignment), assignment))
    ranked_assignments.sort(key=lambda item: item[0])

    if not ranked_assignments or math.isinf(ranked_assignments[0][0]):
        raise ContainmentError(
            "no way of matching these captures to the Host log's invocations puts every capture inside a window "
            "wide enough to hold it; the log and the captures are probably from different runs"
        )
    best_cost, best = ranked_assignments[0]
    runner_up = next((cost for cost, other in ranked_assignments[1:] if other != best), math.inf)
    # `max(..., best_cost)` so a tie is ambiguous even when both fit perfectly:
    # a zero-cost best would otherwise beat a zero-cost runner-up on arithmetic.
    if runner_up <= max(best_cost * margin_ratio, best_cost):
        raise ContainmentError(
            f"the captures pair ambiguously with the Host log's invocations: the best match is off by "
            f"{best_cost / 1000.0:.1f} us of window and the next by {runner_up / 1000.0:.1f} us, which is too "
            "close to tell apart. Ranks running the same shape leave nothing in the timing that names the other, "
            "so re-run so the captures carry dispatch_identity.json with its host_pid, or pin the pairing "
            "explicitly (--rank-pid RANK=PID:INV)."
        )

    sources = dict.fromkeys(pinned, "pinned")
    sources.update(dict.fromkeys(by_sidecar, "capture_sidecar"))
    sources.update(dict.fromkeys(free_keys, "device_window_fit"))
    best.update(settled)
    pairs = {key: hosts[index] for key, index in best.items()}
    diagnostics = {
        key: {
            "pid": hosts[index].pid,
            "inv": hosts[index].inv,
            "source": sources[key],
            # Which of the sidecar's two keys did the narrowing, so a pairing
            # that leaned on only one of them is visible rather than implied.
            "sidecar_keys": list(narrowed[key][1]) if key in by_sidecar else None,
            "window_excess_ns": int(round(costs[(key, index)])) if key in free_keys else None,
            "assignment_margin_ns": None if math.isinf(runner_up) else int(round(runner_up - best_cost)),
        }
        for key, index in sorted(best.items())
    }
    return pairs, diagnostics


def sum_two_widest_slacks(slacks):
    """Worst-case error on an interval read between two placed captures.

    Each end carries its own capture's slack, so the two largest bound any pair.
    ``None`` means fewer than two were placed — never that a comparison is
    exact, and never one capture's own slack, which bounds nothing across
    captures. Taken as a slack list rather than as placements because the
    document and the line printed beside it must not answer this differently.
    """
    widest = sorted(slacks)
    if len(widest) < 2:
        return None
    return widest[-1] + widest[-2]


def cross_uncertainty_ns(placements):
    """``sum_two_widest_slacks`` over the placements themselves."""
    return sum_two_widest_slacks(int(round(placement.slack_ns)) for placement in placements)


# The frame header a remote dispatch sent, and that the peer serving it wrote
# onto its own span, as one `frame=<session>:<worker>:<sequence>` token. Two
# hosts' CLOCK_MONOTONIC axes have no common zero, so this is the only name
# that means the same thing in both logs.
#
# One field rather than three, because a record's attributes are capped and a
# dispatch's already run most of the way to that cap: three names overran it,
# and what a reader got back was a key that looked present with its last part
# cut off. Joined, the three arrive together or not at all.
_FRAME_FIELD = "frame"
# The three parts are one token, so what joins them is part of the grammar
# rather than a detail of whoever wrote the record.
_FRAME_SEPARATOR = ":"

# What closes a dispatch span against the completion span for the same handoff,
# inside one process. Every field names a run or a slot on that host, which is
# exactly why none of them can also serve as the cross-host key.
_DISPATCH_FIELDS = ("run_id", "task_slot", "group_index", "worker_id", "dispatch_id")

DISPATCH_LEAF = "dispatch"
COMPLETE_LEAF = "complete"
REMOTE_TASK_LEAF = "remote_task"


@dataclass(frozen=True)
class RemoteWindow:
    """One remote dispatch, and the peer window it provably contains.

    The caller blocks from publishing the frame until the completion for it
    arrives, so everything the peer did for that frame happened inside
    ``[caller_start_ns, caller_start_ns + caller_duration_ns)``. Both ends of
    that interval are read on the caller's clock and both ends of the peer's
    are read on the peer's, so the difference of the two *durations* is a real
    quantity while no instant is ever compared across the two hosts.

    What the subtraction assumes is that the two machines agree on how long a
    second is, not on when it started, and that is the one thing containment
    does not measure: it handles a clock's offset and not its rate. Writing the
    true elapsed times ``T_c`` and ``T_p`` with relative rate errors ``e_c`` and
    ``e_p``, the true slack and the measured one differ by
    ``T_p(e_p - e_c) - S·e_c``, which ``T_c × MAX_RELATIVE_RATE`` bounds. Only
    the *difference* between the two rates carries: two clocks off by the same
    amount leave an error proportional to the slack itself. ``slack_ns``
    therefore carries that term as well as the measured width, and
    ``rate_bound_ns`` publishes it apart, since only the measured half shrinks
    when a window tightens.
    """

    frame: tuple[int, int, int]
    caller_pid: int
    caller_start_ns: int
    caller_duration_ns: int
    peer_pid: int
    peer_start_ns: int
    peer_duration_ns: int

    @property
    def slack_ns(self):
        """Spare width of the caller's window, plus what two clocks can add."""
        return self.caller_duration_ns - self.peer_duration_ns + self.rate_bound_ns

    @property
    def rate_bound_ns(self):
        """How far two counters' rates can pull the measured width apart."""
        return self.caller_duration_ns * MAX_RELATIVE_RATE

    @property
    def same_axis(self):
        """Whether the peer's own timestamps already fall in the caller's window.

        They do when the two logs share a clock, and also when two machines
        booted within this window's spare width of each other; nothing here
        separates those readings. What both allow is leaving the peer where it
        was recorded, since under either the recorded instant is within
        ``slack_ns`` of the truth. That keeps the observed gap between the
        dispatch and the peer picking the frame up, which placing would
        collapse to zero — and the bound is published all the same, running
        both ways rather than forward only.
        """
        return (
            self.caller_start_ns <= self.peer_start_ns
            and self.peer_start_ns + self.peer_duration_ns <= self.caller_start_ns + self.caller_duration_ns
        )

    @property
    def place_lo_ns(self):
        """Earliest caller-clock ns the peer block can start at.

        The bound the window states, which holds whether the block was moved
        onto that window or left where it was recorded; ``drawn_at_ns`` says
        which of those happened.
        """
        return float(self.caller_start_ns)

    @property
    def place_hi_ns(self):
        return self.caller_start_ns + self.slack_ns

    @property
    def drawn_at_ns(self):
        """Where the block is actually drawn, which is not always the bound's floor."""
        return float(self.peer_start_ns if self.same_axis else self.caller_start_ns)

    def peer_ns_to_host_ns(self, peer_ns):
        """Place one peer-clock instant on the caller's axis, at its lower bound.

        The offset inside the peer's own window is exact — one clock, one rate —
        so only the block's origin carries the bound.
        """
        return self.place_lo_ns + (peer_ns - self.peer_start_ns)

    def metadata(self):
        return {
            "method": "span_containment_v1",
            "frame_session": self.frame[0],
            "frame_worker": self.frame[1],
            "frame_sequence": self.frame[2],
            "caller_pid": self.caller_pid,
            "caller_start_ns": self.caller_start_ns,
            "caller_duration_ns": self.caller_duration_ns,
            "peer_pid": self.peer_pid,
            "peer_duration_ns": self.peer_duration_ns,
            "slack_ns": int(round(self.slack_ns)),
            # Published apart from the total: only the measured half of the
            # bound shrinks when a window tightens, and `#2134` asks for the
            # rate term to be stated rather than folded away.
            "rate_bound_ns": int(round(self.rate_bound_ns)),
            "place_lo_ns": int(round(self.place_lo_ns)),
            "place_hi_ns": int(round(self.place_hi_ns)),
            # Where the bound is and where the block sits are two facts: a peer
            # already reading on this clock keeps its own instant.
            "drawn_at_ns": int(round(self.drawn_at_ns)),
            "observed": self.same_axis,
        }


@dataclass(frozen=True)
class RemoteChain:
    """Every peer process's path back to the axis a trace is drawn on.

    A peer can itself have dispatched to a further peer, so placing one instant
    takes as many hops as the pile holds: each window puts it on the clock of
    the process that dispatched, and the next takes it from there. The slacks
    add, because the freedom one window grants is freedom the windows above it
    cannot take back - which is the sum the chain's bound is made of.
    """

    windows: tuple
    axis_pids: frozenset

    @cached_property
    def peer_pids(self):
        return frozenset(window.peer_pid for window in self.windows)

    @cached_property
    def observed_pids(self):
        """Peers left where they were recorded rather than moved onto the axis.

        Their bound runs both ways, where a placed block is drawn at the
        earliest position its window allows and can only be later, so a reader
        has to be told which kind of position a lane carries.
        """
        return frozenset(window.peer_pid for window in self.windows if window.same_axis)

    @cached_property
    def _by_peer(self):
        """Each peer's windows in start order, with the widest of them.

        A long run dispatches thousands of frames and a trace holds thousands
        of spans, so finding one instant's window by scanning would make
        placing a log quadratic in what the log holds. The widest duration
        bounds how far back a search has to look once it has bisected, which is
        one step for the frames a process served in sequence.
        """
        index = {}
        for window in self.windows:
            index.setdefault(window.peer_pid, []).append(window)
        found = {}
        for pid, group in index.items():
            group.sort(key=lambda window: window.peer_start_ns)
            widest = max(window.peer_duration_ns for window in group)
            found[pid] = ([window.peer_start_ns for window in group], group, widest)
        return found

    def place(self, pid, ns):
        """One instant on the drawn axis, with the bound it arrives under.

        ``(ns, 0)`` for a process ``axis_pids`` names, which is the only
        evidence that a timestamp is already on the axis — a pid alone is not,
        since two machines number their processes independently. ``None``
        otherwise: for a peer instant belonging to no frame this pile holds,
        and for a process nothing places, both of which are dropped rather than
        drawn where nothing supports them.
        """
        slack = 0.0
        for _ in range(len(self.windows) + 1):
            window = self._window_holding(pid, ns)
            if window is None:
                return (ns, slack) if pid in self.axis_pids else None
            # A window whose peer already reads on this clock moves nothing;
            # what it contributes is the bound, which holds under either
            # reading of why the two timestamps line up.
            if not window.same_axis:
                ns = window.peer_ns_to_host_ns(ns)
            slack += window.slack_ns
            pid = window.caller_pid
        raise ContainmentError("the remote windows form a cycle, so no process in them reaches the drawn axis")

    def _window_holding(self, pid, ns):
        """The window an instant falls in, the window being half-open.

        An instant at ``peer_start_ns + peer_duration_ns`` is the first one
        after the frame rather than the last one in it, so it belongs to
        whatever the peer did next and not to a frame that has closed.
        """
        found = self._by_peer.get(pid)
        if found is None:
            return None
        starts, group, widest = found
        for index in range(bisect_right(starts, ns) - 1, -1, -1):
            window = group[index]
            if ns - window.peer_start_ns > widest:
                return None
            if ns < window.peer_start_ns + window.peer_duration_ns:
                return window
        return None


def _dispatchers_of(spans, device_windows):
    """The processes that dispatched the device work this merge is of.

    A device window's own process is the one that ran the work, never the one
    that asked for it — the scheduler above it is a different process with its
    own log. What names one in the other is the dispatch identity both write
    down, the same four fields the capture pairing already joins on.

    Identity alone does not settle it: the fields count per run, so two
    machines each starting at one write the same numbers, and a pile holding
    both logs holds that identity twice. The dispatcher is then the one that
    was running when the window opened. That compares no clock across a
    boundary — it asks only whether a process's own log covers an instant, and
    a process whose whole recorded life excludes it cannot have dispatched what
    began there.
    """
    wanted = {}
    for window in device_windows:
        if window.identity is not None:
            wanted.setdefault(window.identity, set()).add(window.start_ns)
    if not wanted:
        return frozenset()

    extent = {}
    for span in spans:
        low, high = extent.get(span.pid, (span.ts, span.ts + span.dur))
        extent[span.pid] = (min(low, span.ts), max(high, span.ts + span.dur))

    found = set()
    for span in spans:
        if node_span_leaf(span.name) != DISPATCH_LEAF or span.pid in found:
            continue
        starts = wanted.get(_attr_ints(span, _HOST_IDENTITY_FIELDS))
        if not starts:
            continue
        low, high = extent[span.pid]
        if any(low <= start <= high for start in starts):
            found.add(span.pid)
    return frozenset(found)


def remote_chain(spans, log_pids):
    """The chain the remote dispatches in ``spans`` form.

    ``log_pids`` is every process the logs hold, and which of them are on the
    axis is decided rather than assumed. A span names its process by pid alone,
    so a pile holding two machines' logs would otherwise read a second
    machine's raw timestamps as this axis's. With no window in the pile nothing
    is spliced and every process is on the one axis there is.

    With a window, a process is on the axis when the pile can name what put it
    there: the top of a chain, a process whose own device window this merge is
    of, or one that dispatched such a window. The three cover a chain from end
    to end, which the first two alone do not — a scheduler between the root and
    a chip owns no device window and is nobody's root, and dropping it draws a
    trace whose middle level is missing.

    Everything else is left out, because a peer's machine writes a log per
    process and no window names those — and no rule over the timestamps
    themselves separates them, since two machines' clocks can read however
    close to one another.
    """
    windows = tuple(remote_windows(spans))
    if not windows:
        return RemoteChain((), frozenset(log_pids))

    peers = frozenset(window.peer_pid for window in windows)
    roots = frozenset(window.caller_pid for window in windows) - peers
    device = tuple(host_windows(spans))
    # Peers are placed, not vouched for: a peer that also has a device window
    # of its own would otherwise be drawn twice, once down each path.
    vouched = roots | frozenset(window.pid for window in device) | _dispatchers_of(spans, device)
    return RemoteChain(windows, (vouched - peers) & frozenset(log_pids))


def rebase_windows(windows, chain):
    """Put every Host window a peer recorded onto the axis the trace is drawn on.

    A Rank whose ``runner_run`` sits in a peer's log is bounded twice: once by
    that window around its device work, and again by the frame that dispatched
    to the peer. Moving the window's origin and widening its spare width by the
    frame's own makes every reader of the resulting placement correct without
    having to know a second hop happened.

    A window the chain cannot place belongs to no frame this trace holds, and
    is dropped for the same reason an unplaceable span is.
    """
    rebased = []
    for window in windows:
        placed = chain.place(window.pid, window.start_ns)
        if placed is None:
            continue
        start_ns, slack = placed
        if not slack and start_ns == window.start_ns:
            rebased.append(window)
            continue
        rebased.append(
            replace(window, start_ns=int(round(start_ns)), duration_ns=int(round(window.duration_ns + slack)))
        )
    return rebased


def format_frame_key(frame):
    """The attribute a span carries to name one frame, as `_frame_key` reads it.

    The grammar lives beside its parser so that a writer and a reader cannot
    come to describe different things — which is what spelling it out a second
    time, in a test or anywhere else, would eventually allow.
    """
    return _FRAME_FIELD + "=" + _FRAME_SEPARATOR.join(str(part) for part in frame)


def _frame_key(span):
    """The frame header one span names, or ``None`` if it names none.

    A span whose attributes the record could not fit says so with a trailing
    `~`, and that is reported rather than read as "this dispatch carried no
    frame": a key the log cut short is a record problem, and pairing nothing
    with no reason given is how it would otherwise present.
    """
    found = None
    for item in span.attrs.split():
        key, separator, value = item.partition("=")
        if not separator or key != _FRAME_FIELD:
            continue
        parts = value.split(_FRAME_SEPARATOR)
        if len(parts) == 3:
            try:
                found = tuple(int(part) for part in parts)
            except ValueError:
                found = None
        break
    if found is None and span.attrs.endswith("~"):
        print(
            f"warning: pid {span.pid} {span.name} at ts={span.ts} was cut short by the record's attribute "
            "capacity, so any frame header it carried is unreadable",
            file=sys.stderr,
        )
    return found


def _attr_ints(span, fields):
    """The named attributes of one span as ints, or ``None`` if any is missing.

    A partial set means the record is not what it claims, which is treated as no
    key at all rather than as a key with holes.
    """
    found = {}
    for item in span.attrs.split():
        key, separator, value = item.partition("=")
        if separator and key in fields:
            try:
                found[key] = int(value)
            except ValueError:
                return None
    if len(found) != len(fields):
        return None
    return tuple(found[field] for field in fields)


def _close_dispatch(dispatch, completions):
    """The completion that closed one dispatch, or ``None`` if the log holds none.

    Keyed on the handoff the two spans both name, then taken in time order: a
    process that dispatched the same slot again reuses every one of those
    fields, so the key alone selects a set and the ordering selects within it.
    """
    for completion in completions:
        if completion.ts + completion.dur >= dispatch.ts + dispatch.dur:
            return completion
    return None


def remote_windows(spans):
    """Every remote dispatch paired with the peer window it contains.

    A dispatch is remote exactly when it recorded a frame header; a peer window
    is the ``remote_task`` span carrying the same one. Both sides therefore
    select themselves out of one undifferentiated pile of logs, and a pile
    holding only one side yields nothing rather than a guess.

    A pair whose peer window is wider than the caller's describes no
    containment and is dropped with its reason, because that is a fact about
    one dispatch: the usual cause is two logs from different runs, where the
    sequence numbers collide by coincidence.
    """
    dispatches = {}
    completions = {}
    peers = {}
    for span in spans:
        leaf = node_span_leaf(span.name)
        if leaf == REMOTE_TASK_LEAF:
            frame = _frame_key(span)
            if frame is not None:
                peers.setdefault(frame, []).append(span)
            continue
        if leaf not in (DISPATCH_LEAF, COMPLETE_LEAF):
            continue
        handoff = _attr_ints(span, _DISPATCH_FIELDS)
        if handoff is None:
            continue
        if leaf == COMPLETE_LEAF:
            completions.setdefault((span.pid, handoff), []).append(span)
            continue
        frame = _frame_key(span)
        if frame is None:
            continue
        if frame in dispatches:
            raise ContainmentError(f"frame {frame} is dispatched by two spans; the logs are from more than one run")
        dispatches[frame] = (span, handoff)

    for group in completions.values():
        group.sort(key=lambda span: span.ts)

    windows = []
    for frame, (dispatch, handoff) in sorted(dispatches.items()):
        peer_group = peers.get(frame)
        if not peer_group:
            continue
        if len(peer_group) > 1:
            raise ContainmentError(
                f"frame {frame} is served by {len(peer_group)} peer windows; the logs are from more than one run"
            )
        completion = _close_dispatch(dispatch, completions.get((dispatch.pid, handoff), []))
        if completion is None:
            continue
        peer = peer_group[0]
        window = RemoteWindow(
            frame=frame,
            caller_pid=dispatch.pid,
            caller_start_ns=dispatch.ts,
            caller_duration_ns=completion.ts + completion.dur - dispatch.ts,
            peer_pid=peer.pid,
            peer_start_ns=peer.ts,
            peer_duration_ns=peer.dur,
        )
        if window.slack_ns < 0:
            continue
        windows.append(window)
    return windows
