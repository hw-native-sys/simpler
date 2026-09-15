#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Per-phase statistics from the `chip.run.bind.*` spans of a host_build_graph run.

Reads the log a `SIMPLER_HBG_BIND_BREAKDOWN_ENABLE=1 --rounds N` run leaves
behind and reports each bind segment's minimum, median and maximum across the
warm binds, plus the subtotal historically labelled "control plane". This sum
does not include uninstrumented Definition preparation or A5 scheduler work;
matching segment names across commits does not establish matching coverage.

The spread is the point: a change smaller than the range beside it cannot be
demonstrated. `strace_timing --tree` gives the median alone, and its default
table a mean, so neither answers "is this difference real".

What the phases are, and how to compare two runs:
docs/dfx/hbg-bind-phases.md.
"""

import argparse
import re
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path

from simpler_setup.tools.strace_timing import expand_log_source, parse_spans

# The stage the segments subdivide. A span one level below it is a segment; one
# two levels below (`chip.run.bind.host_orch.<op>`) is an orchestrator operation
# from the per-event artifact, which is not a segment of the stage.
BIND_SPAN = "chip.run.bind"

# The recipe's first log line, recording the command and the commit the run used.
# Two runs are comparable only if it matches, so it is echoed above the table.
STAMP_LINE = re.compile(r"^\[stamp\] (.*)$")
_JSON_STRING = r'"(?:\\.|[^"\\])*"'
TORCH_AUTOLOAD_LINE = re.compile(
    rf"(torch_backend_autoload setting=\S+ "
    rf"(?:raw=(?:null|{_JSON_STRING}) raw_truncated=(?:true|false) )?effective=\S+ "
    r"torch_imported=(?:true|false) torch_npu_loaded=(?:true|false))"
)

# Instrumented subtotal, not the full interval before device execution.
# Definition preparation and A5 scheduler work have no segment in this sum.
# `args` is a per-byte staging cost. `host_view_close` remains excluded for
# comparison with historical mapped-view logs; current binds close no mappings.
CONTROL_PLANE = ("host_orch", "graph_upload", "arena_h2d")

# Historical display order; spans carry timestamps for the actual execution order.
PHASE_ORDER = (
    "args",
    "arena_build",
    "runtime_init",
    "host_orch",
    "graph_upload",
    "static_arena",
    "shared_mem",
    "gm_heap",
    "arena_h2d",
    "host_view_close",
)


def parse_binds(paths: Sequence[Path]) -> list[dict]:
    """Group bind-segment spans into binds, in milliseconds.

    A span carries `pid` and `inv`, so a bind is exactly the segments sharing one
    `(pid, inv)` — the run epoch the enclosing `chip.run.bind` allocated. Nothing
    is inferred from emission order or from which segment came last, and
    concurrent ranks writing one stream cannot be confused for each other.

    Each bind is `{"pid", "inv", "ts", <segment>: ms, ...}`; `ts` is its earliest
    segment start, which orders the binds a rank ran.
    """
    prefix = f"{BIND_SPAN}."
    binds: dict[tuple, dict] = {}
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for span in parse_spans(handle):
                if span.is_device or not span.name.startswith(prefix):
                    continue
                phase = span.name[len(prefix) :]
                if "." in phase:
                    # An orchestrator operation inside host_orch, not a segment.
                    continue
                bind = binds.setdefault((span.pid, span.inv), {"pid": span.pid, "inv": span.inv, "ts": span.ts})
                bind[phase] = span.dur / 1e6
                bind["ts"] = min(bind["ts"], span.ts)
    ordered = sorted(binds.values(), key=lambda b: (b["pid"], b["ts"]))
    # A group without `host_orch` is not a bind.
    return [b for b in ordered if "host_orch" in b]


def cold_binds(binds: Sequence[dict]) -> set[tuple]:
    """The `(pid, inv)` of each rank's first bind, which is warm-up.

    One per pid, and exactly the first one that rank ran — where the old
    line-based tool had to be told a rank count and drop that many from the
    front of a single interleaved stream.
    """
    first: dict[int, dict] = {}
    for bind in binds:
        current = first.get(bind["pid"])
        if current is None or bind["ts"] < current["ts"]:
            first[bind["pid"]] = bind
    return {(b["pid"], b["inv"]) for b in first.values()}


def parse_stamp(paths: Sequence[Path]) -> str:
    """The run's environment stamp, or "" for a log this script did not produce."""
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = STAMP_LINE.match(line.rstrip("\n"))
                if match is not None:
                    return match.group(1)
    return ""


def parse_torch_autoload(paths: Sequence[Path]) -> list[str]:
    """Unique torch backend autoload records in log order."""
    records: list[str] = []
    seen: set[str] = set()
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = TORCH_AUTOLOAD_LINE.search(line)
                if match is None:
                    continue
                record = match.group(1)
                if record not in seen:
                    seen.add(record)
                    records.append(record)
    return records


def spread(values: Sequence[float]) -> tuple[float, float, float]:
    """Minimum, median and maximum. The median averages the two central values."""
    return min(values), statistics.median(values), max(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "log",
        nargs="+",
        help="run log(s) carrying the `chip.run.bind.*` spans, or a directory of host.*.log files",
    )
    parser.add_argument("--keep-first", action="store_true", help="keep each rank's cold bind in the statistics")
    args = parser.parse_args()

    paths = [path for source in args.log for path in expand_log_source(source)]

    binds = parse_binds(paths)
    if not binds:
        print(
            f"{' '.join(args.log)}: no `chip.run.bind.<segment>` spans. Either "
            "SIMPLER_HBG_BIND_BREAKDOWN_ENABLE=1 was not set, or an output run moved the whole "
            "host log into the process-session spool -- parse those host.*.log files instead. "
            "The log level is not a "
            "cause: TIMING is the default. See docs/dfx/hbg-bind-phases.md.",
            file=sys.stderr,
        )
        return 1

    cold = set() if args.keep_first else cold_binds(binds)
    warm = [b for b in binds if (b["pid"], b["inv"]) not in cold]
    ranks = len({b["pid"] for b in binds})
    if not warm:
        print(
            f"{' '.join(args.log)}: {len(binds)} bind(s) over {ranks} rank(s), all of them cold. The "
            "first bind of each rank is warm-up, so a warm bind needs --rounds 2 or more on the run; "
            "--keep-first reports the cold binds instead.",
            file=sys.stderr,
        )
        return 1

    print(f"{' '.join(args.log)}")
    stamp = parse_stamp(paths)
    if stamp:
        print(f"  {stamp}")
    else:
        print("  (no `[stamp]` line: the command and commit behind these numbers have")
        print("   to be established by hand before comparing them to anything)")
    torch_autoload = parse_torch_autoload(paths)
    if torch_autoload:
        for record in torch_autoload:
            print(f"  {record}")
    else:
        print("  (no `torch_backend_autoload` record: backend-autoload state")
        print("   must be established before comparing this log)")
    print(f"  {len(binds)} binds, {len(warm)} warm ({len(cold)} cold dropped, ranks={ranks})\n")
    print(f"  {'phase':<18}{'min':>10}{'median':>10}{'max':>10}   n")
    for phase in PHASE_ORDER:
        values = [b[phase] for b in warm if phase in b]
        if not values:
            continue
        low, mid, high = spread(values)
        mark = "*" if phase in CONTROL_PLANE else " "
        print(f" {mark}{phase:<18}{low:>10.3f}{mid:>10.3f}{high:>10.3f}  {len(values):3d}")

    known = set(PHASE_ORDER) | {"pid", "inv", "ts"}
    unknown = sorted({k for b in warm for k in b} - known)
    if unknown:
        print(f"\n  segments this tool does not know about: {', '.join(unknown)}")
        print("  (add them to PHASE_ORDER, and to CONTROL_PLANE if a dispatch change can move them)")

    # The control-plane set is not fixed: a change can retire a segment outright,
    # so the total covers the segments this run has and names the ones absent from
    # every bind. Absent from only some binds means those binds lost records, and
    # would understate the total, so they are excluded and warned about.
    present = [k for k in CONTROL_PLANE if any(k in b for b in warm)]
    partial = [k for k in present if not all(k in b for b in warm)]
    retired = [k for k in CONTROL_PLANE if k not in present]
    complete = [b for b in warm if all(k in b for k in present)]
    if complete:
        totals = [sum(b[k] for k in present) for b in complete]
        low, mid, high = spread(totals)
        print("\n  * = control plane, summed within each bind and then:")
        print(f"    total{'':<13}{low:>10.3f}{mid:>10.3f}{high:>10.3f}  {len(totals):3d}   (ms)")
        if retired:
            print(f"    over {len(present)} of {len(CONTROL_PLANE)} segments; absent from every")
            print(f"    bind: {', '.join(retired)}")
        if partial:
            print(f"    WARNING: {', '.join(partial)} is missing from some binds but not all;")
            print("    those binds are excluded, and the total may not describe the run")
        print("\n  Compare by min, over the same segment set, against a log with the same")
        print("  stamp and torch autoload state; see docs/dfx/hbg-bind-phases.md 'Comparing two branches'.")
    else:
        print(f"\n  no bind carries any of {CONTROL_PLANE}; the control-plane total is not computable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
