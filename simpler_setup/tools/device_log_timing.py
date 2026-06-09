#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Parse Total / Orch / Sched device timing from a CANN device log.

The orchestrator + scheduler print one block of ``LOG_INFO_V9`` timing lines
per on-device run (gated by the compile-time ``PTO2_PROFILING`` macro, on by
default — NOT by ``--enable-l2-swimlane``). A multi-round run appends one block
per round to the same device log file, so this module groups the log into
rounds (one ``orch_start`` per round) and reports per-round Total / Orch /
Sched plus a trimmed average, mirroring ``scene_test._log_round_timings``.

Marker lines (wrapped in the CANN dlog prefix, matched anywhere on the line):

    Thread N: orch_start=<cyc> orch_end=<cyc> orch_cost=<us>us
    Thread N: sched_start=<cyc> sched_end[(timeout)]=<cyc> sched_cost=<us>us
    Thread N: Scheduler summary: total_time=<us>us, loops=<n>, tasks_scheduled=<n>
    PTO2 total submitted tasks = <n>[, already executed <n> tasks]

Per round:
    Orch  = orch span                       (== orch_cost)
    Sched = max(sched_end) - min(sched_start)
    Total = max(all ends) - min(all starts) across orch + sched threads

Cycle counts are converted to microseconds using a frequency self-calibrated
from each round's ``orch_cost`` (``cycles / cost_us``), so no per-platform clock
constant is needed; ``benchmark_rounds.sh`` hard-codes the same 50 MHz for a2a3.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_ORCH_RE = re.compile(r"Thread\s+(\d+):\s+orch_start=(\d+)\s+orch_end=(\d+)\s+orch_cost=([\d.]+)us")
_SCHED_RE = re.compile(r"Thread\s+(\d+):\s+sched_start=(\d+)\s+sched_end(?:\(timeout\))?=(\d+)\s+sched_cost=([\d.]+)us")
_TASKS_RE = re.compile(r"PTO2 total submitted tasks = (\d+)")


class RoundTiming:
    """Total / Orch / Sched (microseconds) parsed for one run/round."""

    def __init__(self, total_us: float, orch_us: float, sched_us: float, tasks: int | None):
        self.total_us = total_us
        self.orch_us = orch_us
        self.sched_us = sched_us
        self.tasks = tasks


def _finalize_round(orch, scheds, tasks) -> RoundTiming | None:
    """Build a RoundTiming from one round's collected orch/sched events.

    ``orch`` is ``(start, end, cost_us)`` or None; ``scheds`` is a list of the
    same. A round needs the orch line (carries the freq calibration) plus at
    least one sched line to bound the Total span; otherwise it is dropped.
    """
    if orch is None or not scheds:
        return None
    orch_start, orch_end, orch_cost_us = orch
    orch_cycles = orch_end - orch_start
    if orch_cost_us <= 0 or orch_cycles <= 0:
        return None
    freq = orch_cycles / orch_cost_us  # cycles per microsecond

    starts = [orch_start] + [s[0] for s in scheds]
    ends = [orch_end] + [s[1] for s in scheds]
    sched_starts = [s[0] for s in scheds]
    sched_ends = [s[1] for s in scheds]

    total_us = (max(ends) - min(starts)) / freq
    sched_us = (max(sched_ends) - min(sched_starts)) / freq
    return RoundTiming(total_us=total_us, orch_us=orch_cost_us, sched_us=sched_us, tasks=tasks)


def parse_device_log_timing(log_paths, offsets=None) -> list[RoundTiming]:
    """Parse a device log into per-round Total / Orch / Sched timings.

    Rounds are split on each ``orch_start`` line (one orchestrator block per
    on-device run). Returns the rounds in log order; incomplete trailing blocks
    are dropped.

    ``log_paths`` is a single path or an iterable of paths. CANN rotates the
    device log at 20 MB (``ASCEND_HOST_LOG_FILE_NUM`` files retained), so a
    long multi-round run can split its blocks across several files; pass them
    in chronological order and they are concatenated before parsing.

    ``offsets`` optionally maps ``str(path) -> start byte``. A file present in
    the map is read only from that offset onward, so callers can snapshot the
    pre-run byte size of an already-open log file and parse only the bytes this
    run appended — otherwise a process that ran earlier work into the same file
    (e.g. a prior case on a reused worker) would have its stale rounds counted.

    A file that disappears between discovery and read (log rotation / cleanup)
    is skipped rather than aborting the whole parse.
    """
    if isinstance(log_paths, (str, Path)):
        log_paths = [log_paths]
    chunks: list[str] = []
    for p in log_paths:
        p = Path(p)
        start = (offsets or {}).get(str(p), 0)
        try:
            with p.open("rb") as f:
                if start:
                    f.seek(start)
                chunks.append(f.read().decode(errors="replace"))
        except OSError:
            continue
    text = "\n".join(chunks)

    rounds: list[RoundTiming] = []
    cur_orch = None
    cur_scheds: list[tuple[int, int, float]] = []
    cur_tasks: int | None = None

    for line in text.splitlines():
        m = _ORCH_RE.search(line)
        if m:
            # New orchestrator block → close the previous round.
            done = _finalize_round(cur_orch, cur_scheds, cur_tasks)
            if done is not None:
                rounds.append(done)
            cur_orch = (int(m.group(2)), int(m.group(3)), float(m.group(4)))
            cur_scheds = []
            cur_tasks = None
            continue
        m = _SCHED_RE.search(line)
        if m:
            cur_scheds.append((int(m.group(2)), int(m.group(3)), float(m.group(4))))
            continue
        m = _TASKS_RE.search(line)
        if m:
            cur_tasks = int(m.group(1))

    done = _finalize_round(cur_orch, cur_scheds, cur_tasks)
    if done is not None:
        rounds.append(done)
    return rounds


def format_device_log_timing(rounds: list[RoundTiming], *, source=None) -> str:
    """Render per-round Total / Orch / Sched table + summary + trimmed average.

    Style mirrors ``scene_test._log_round_timings`` (user-facing ``print``
    artifact). ``source`` is an optional device-log path shown in the header.
    """
    lines: list[str] = []
    if source is not None:
        lines.append(f"Device-log timing (from {source})")
    if not rounds:
        lines.append("  (no orch/sched timing blocks found — was the runtime built with PTO2_PROFILING?)")
        return "\n".join(lines)

    n = len(rounds)
    tasks = next((r.tasks for r in rounds if r.tasks is not None), None)
    if tasks is not None:
        lines.append(f"  tasks = {tasks}")

    header = f"  {'Round':<6}  {'Total (us)':>12}  {'Orch (us)':>12}  {'Sched (us)':>12}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for i, r in enumerate(rounds):
        lines.append(f"  {i:<6d}  {r.total_us:>12.1f}  {r.orch_us:>12.1f}  {r.sched_us:>12.1f}")

    def _avg(vals):
        return sum(vals) / len(vals)

    totals = [r.total_us for r in rounds]
    orchs = [r.orch_us for r in rounds]
    scheds = [r.sched_us for r in rounds]
    lines.append(
        f"  Avg  Total: {_avg(totals):.1f} us  |  Orch: {_avg(orchs):.1f} us  |  "
        f"Sched: {_avg(scheds):.1f} us  ({n} rounds)"
    )

    trim = 10
    if n > 2 * trim:
        tc = n - 2 * trim

        def _trimmed(vals):
            s = sorted(vals)
            return sum(s[trim:-trim]) / tc

        lines.append(
            f"  Trimmed Avg  Total: {_trimmed(totals):.1f} us  |  Orch: {_trimmed(orchs):.1f} us  |  "
            f"Sched: {_trimmed(scheds):.1f} us  (dropped {trim} low + {trim} high, {tc} rounds used)"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    """Standalone CLI: print Total / Orch / Sched from a CANN device log.

    The no-swimlane counterpart to ``sched_overhead_analysis`` (which reads the
    richer swimlane JSON): this reads only the ``PTO2_PROFILING`` orch/sched
    markers, so it works for a plain run or an external workload that never
    produced an ``l2_swimlane_records.json``.
    """
    from .device_log_resolver import resolve_device_log_paths  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Per-round Total / Orch / Sched from a CANN device log (PTO2_PROFILING markers).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Examples:\n  %(prog)s --device-log '~/ascend/log/debug/device-0/device-*.log'\n  %(prog)s -d 0\n"),
    )
    parser.add_argument(
        "--device-log",
        help="Path / dir / glob of a CANN device log. A glob parses all matched (rotated) files.",
    )
    parser.add_argument(
        "-d",
        "--device-id",
        help="Device id: auto-pick the newest log under device-<id>/.",
    )
    args = parser.parse_args(argv)
    if args.device_log is None and args.device_id is None:
        parser.error("one of --device-log or -d/--device-id is required")

    if os.environ.get("ASCEND_SLOG_PRINT_TO_STDOUT") == "1":
        # No device log is written in this mode; any .log on disk is from an
        # older run, so parsing it would report stale, misleading numbers.
        print(
            "Error: ASCEND_SLOG_PRINT_TO_STDOUT=1 routes CANN logs to stdout, not to a log file; "
            "no device log to parse. Unset it (or set 0) and re-run the workload.",
            file=sys.stderr,
        )
        return 1

    logs, strategy = resolve_device_log_paths(device_id=args.device_id, device_log=args.device_log)
    if not logs:
        print(f"Error: Failed to resolve device log ({strategy})", file=sys.stderr)
        return 1
    source = logs[0] if len(logs) == 1 else strategy
    print(format_device_log_timing(parse_device_log_timing(logs), source=source))
    return 0


if __name__ == "__main__":
    sys.exit(main())
