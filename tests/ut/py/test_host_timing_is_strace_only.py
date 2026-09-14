#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""`[STRACE]` is the only timeline format the host emits.

Three layers, and this file pins the boundary between the first two:

- **`TIMING` is a log level** — the envelope every host record rides in.
- **`[STRACE]` is the timing system built on it**, and the only format that
  carries an *interval*. Everything a reader can place on a timeline goes
  through it, whether it was measured by an RAII scope or read back from a
  fixed-slot capture buffer and re-emitted (`STRACE_HOST_SPAN_AT`,
  `STRACE_DEV_SPAN_AT`).
- **A summed cost share is not an interval** and stays an ordinary `LOG_TIMING`
  line. Sub-operations that nest inside each other cannot honestly be drawn as
  bars, so they report a total and a count instead.

The concrete failure this prevents: the host prepare path once printed
`bind phase=<p> start_ns=<n> dur_ns=<n>` at `LOG_TIMING`, which is an interval in
a second format. It grew its own regex in `strace_timing.py`, two dedicated
tools, and a grouping heuristic that existed only because the line carried no
`pid`/`inv` — the keys a `[STRACE]` record has by construction.

A start plus a duration is the signature of that mistake, so that is what this
test looks for.
"""

from __future__ import annotations

import pathlib
import re

# A start timestamp in a log line is the marker of an interval reported outside
# `[STRACE]`. `[STRACE]` itself writes `ts=`, and it is not emitted through
# LOG_TIMING format strings — `HostLogger::log_host_span` builds the record.
_INTERVAL_START_KEYS = ("start_ns=", "start_us=", "begin_ns=")

# The shapes a LOG_TIMING line may legitimately carry, as a reader's map rather
# than as an assertion: summed cost shares (`total_ns=`/`count=`), the
# orchestrator's own step totals, the writer's drop counters, and queue-occupancy
# probes.
_ALLOWED_SHAPES = ("total_ns=", "count=", "ns=", "dropped=", "queue")

_SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[3] / "src"
_STRING_LITERAL = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _log_timing_format_strings(text: str) -> list[str]:
    """Every `LOG_TIMING(...)` call's format string, adjacent literals joined.

    C++ concatenates adjacent string literals, so a format string wrapped across
    lines is several literals in the source and one string to the compiler. The
    scan therefore takes the whole call — balanced to its closing paren — and
    joins every literal in it.
    """
    formats = []
    for match in re.finditer(r"\bLOG_TIMING\s*\(", text):
        depth = 0
        end = len(text)
        for index in range(match.end() - 1, len(text)):
            char = text[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        formats.append("".join(literal.group(1) for literal in _STRING_LITERAL.finditer(text[match.end() : end])))
    return formats


def _sources(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(path for pattern in ("**/*.cpp", "**/*.h") for path in root.glob(pattern))


def _interval_carrying_lines(root: pathlib.Path) -> list[str]:
    """Every LOG_TIMING format string under `root` that carries a start timestamp."""
    offenders = []
    for path in _sources(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        for fmt in _log_timing_format_strings(text):
            if any(key in fmt for key in _INTERVAL_START_KEYS):
                offenders.append(f"{path.relative_to(root.parent)}: {fmt}")
    return offenders


def test_no_log_timing_line_reports_an_interval():
    """An interval outside `[STRACE]` is a second timeline format."""
    offenders = _interval_carrying_lines(_SOURCE_ROOT)

    assert offenders == [], (
        "these LOG_TIMING lines carry a start timestamp, so they report an interval in a second "
        "timeline format. Emit a span instead — STRACE_HOST_SPAN_AT_A takes an already-measured "
        "start and duration, which is what the host prepare path uses for its bind segments:\n  "
        + "\n  ".join(offenders)
    )


def test_the_scan_actually_reaches_the_log_timing_call_sites():
    """A positive control: an empty scan would pass the assertion above vacuously.

    Both halves matter — that files are being read at all, and that the format
    strings inside them are recovered — because either failing silently turns the
    invariant into a test that can never fail.
    """
    sources = _sources(_SOURCE_ROOT)
    assert len(sources) > 100, f"only {len(sources)} sources found under {_SOURCE_ROOT}"

    formats = [fmt for path in sources for fmt in _log_timing_format_strings(path.read_text(errors="replace"))]
    assert len(formats) > 10, f"only {len(formats)} LOG_TIMING format strings recovered"
    assert any(any(shape in fmt for shape in _ALLOWED_SHAPES) for fmt in formats), (
        "no LOG_TIMING line matched any known shape, so the scan is probably not recovering format strings correctly"
    )


def test_multi_line_format_strings_are_joined_before_the_check():
    """The scan has to see one string where the source has three literals.

    A format string split across lines is how the offending line was written, so
    a scan that tested each literal on its own would miss `start_ns=` sitting in
    the second one.
    """
    source = """
    LOG_TIMING(
        "bind phase=%s "
        "start_ns=%llu "
        "dur_ns=%llu",
        name, start, dur
    );
    """

    assert _log_timing_format_strings(source) == ["bind phase=%s start_ns=%llu dur_ns=%llu"]


def test_a_reintroduced_interval_line_is_caught(tmp_path):
    """The invariant is green today, so prove it can go red.

    Without this the assertion above would keep passing if the scan silently
    stopped finding anything — and a guard that cannot fail is not a guard. The
    summed-share line beside the offender is the other half: the check has to
    admit what a LOG_TIMING line may legitimately carry.
    """
    source_root = tmp_path / "src"
    (source_root / "runtime").mkdir(parents=True)
    (source_root / "runtime" / "clean.cpp").write_text(
        'LOG_TIMING("host-orch phase=%s total_ns=%llu count=%llu", name, total, count);\n', encoding="utf-8"
    )
    (source_root / "runtime" / "offender.cpp").write_text(
        'LOG_TIMING(\n    "bind phase=%s "\n    "start_ns=%llu dur_ns=%llu",\n    name, start, dur\n);\n',
        encoding="utf-8",
    )

    offenders = _interval_carrying_lines(source_root)

    assert offenders == ["src/runtime/offender.cpp: bind phase=%s start_ns=%llu dur_ns=%llu"]
