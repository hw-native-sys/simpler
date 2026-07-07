#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Parse simpler host-side trace markers (``[STRACE]``) into per-stage timing.

The host runtime emits one ``[STRACE]`` line per span on scope exit (RAII
markers in ``src/common/log/include/common/strace.h``), gated by the
compile-time ``SIMPLER_HOST_STRACE`` macro (on by default) and emitted at
``LOG_INFO_V9``. Device-domain phases (AICPU subdivision of the on-NPU wall)
are emitted by the host after readback as ``clk=dev`` spans nested under
``simpler_run.runner_run.device_wall``.

Marker grammar (matched anywhere on the line, so the CANN/host log prefix is
ignored)::

    [STRACE] v=1 pid=<n> tid=<n> inv=<n> hid=<hex> depth=<n> name=<dotted> ts=<ns> dur=<ns> [k=v ...]

Grouping:
    * ``(pid, inv)`` identifies one ``simpler_run`` invocation — all its spans
      share these. ``inv`` is a process-wide id (atomic-allocated, so unique even
      across concurrent calls), NOT a token index.
    * ``hid`` is the callable's content hash (stable across slot reuse / runs).
      The most-frequently-seen hid bucket is the decode callable (one
      invocation per token); a once-seen hid is prefill.
    * ``depth`` rebuilds the call tree per invocation (no timestamp-containment
      guessing): a span at depth d is a child of the most recent span at d-1.

Outputs:
    * a per-callable TPOT table (each invocation's simpler_run dur + the mean
      of each sub-stage across invocations), and
    * optionally a Chrome-trace / Perfetto JSON (``--trace-out``): one ``ph:"X"``
      event per span, lane = pid, so the host call tree renders as nested
      slices (L3 parent and each L2 child get their own pid lane).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field

_STRACE_RE = re.compile(
    r"\[STRACE\]\s+v=(?P<v>\d+)\s+pid=(?P<pid>\d+)\s+tid=(?P<tid>\d+)\s+"
    r"inv=(?P<inv>\d+)\s+hid=(?P<hid>[0-9a-fA-F]+)\s+depth=(?P<depth>\d+)\s+"
    r"name=(?P<name>\S+)\s+ts=(?P<ts>\d+)\s+dur=(?P<dur>\d+)(?P<attrs>.*)"
)


@dataclass
class Span:
    pid: int
    tid: int
    inv: int
    hid: str
    depth: int
    name: str
    ts: int
    dur: int
    attrs: str

    @property
    def is_device(self) -> bool:
        return "clk=dev" in self.attrs


@dataclass
class Invocation:
    """All spans emitted by one simpler_run call (one (pid, inv) group)."""

    pid: int
    inv: int
    hid: str
    spans: list = field(default_factory=list)

    def root(self):
        """The depth-0 span (simpler_run), or None if absent."""
        for s in self.spans:
            if s.depth == 0:
                return s
        return None

    def by_name(self):
        m = {}
        for s in self.spans:
            m.setdefault(s.name, s)
        return m


def parse_spans(lines):
    """Yield Span for every matching [STRACE] line."""
    for line in lines:
        m = _STRACE_RE.search(line)
        if not m:
            continue
        yield Span(
            pid=int(m["pid"]),
            tid=int(m["tid"]),
            inv=int(m["inv"]),
            hid=m["hid"].lower(),
            depth=int(m["depth"]),
            name=m["name"],
            ts=int(m["ts"]),
            dur=int(m["dur"]),
            attrs=m["attrs"].strip(),
        )


def group_invocations(spans):
    """Group spans into Invocation objects keyed by (pid, inv)."""
    groups: dict = {}
    for s in spans:
        key = (s.pid, s.inv)
        inv = groups.get(key)
        if inv is None:
            inv = Invocation(pid=s.pid, inv=s.inv, hid=s.hid)
            groups[key] = inv
        inv.spans.append(s)
    # Stable order: by pid then inv.
    return [groups[k] for k in sorted(groups)]


def bucket_by_hid(invocations):
    """Map hid -> [Invocation], ordered by inv within each bucket."""
    buckets: dict = defaultdict(list)
    for inv in invocations:
        buckets[inv.hid].append(inv)
    for bucket in buckets.values():
        bucket.sort(key=lambda i: i.inv)
    return buckets


def _fmt_us(ns: int) -> str:
    return f"{ns / 1000.0:.1f}"


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def print_tpot_table(buckets, label_for_hid=None, stream=sys.stdout):
    """Print a per-callable TPOT table. The most-invoked bucket is decode."""
    if not buckets:
        print("No [STRACE] markers found.", file=stream)
        return

    ordered = sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)
    for hid, invs in ordered:
        label = (label_for_hid or {}).get(hid, "")
        header = f"callable hid={hid}"
        if label:
            header += f" ({label})"
        header += f" — {len(invs)} invocation(s)"
        print(header, file=stream)

        roots = [i.root() for i in invs if i.root() is not None]
        durs = [r.dur for r in roots]
        if durs:
            print(
                f"  simpler_run: mean={_fmt_us(int(_mean(durs)))}us "
                f"min={_fmt_us(min(durs))}us max={_fmt_us(max(durs))}us",
                file=stream,
            )

        # Mean of each sub-stage across invocations (by span name).
        stage_durs: dict = defaultdict(list)
        for inv in invs:
            for name, span in inv.by_name().items():
                if span.depth == 0:
                    continue
                stage_durs[name].append(span.dur)
        for name in sorted(stage_durs, key=lambda n: (-len(stage_durs[n]), n)):
            ds = stage_durs[name]
            indent = "  " + "  " * name.count(".")
            print(f"{indent}{name}: mean={_fmt_us(int(_mean(ds)))}us (n={len(ds)})", file=stream)
        print(file=stream)


_ROUNDS_TABLE_NAMES = {
    "host": "simpler_run",
    "device": "simpler_run.runner_run.device_wall",
    "orch": "simpler_run.runner_run.device_wall.orch",
    "sched": "simpler_run.runner_run.device_wall.sched",
}

# Per-round table columns, in print order. "Effective" is the orch∪sched merged
# window (the old device-log "Total"), recomputed here purely from the orch/sched
# markers' device-domain ts+dur — no device log needed. label is the column
# header / "Avg <label>".
_ROUNDS_TABLE_COLUMNS = ("Host", "Device", "Effective", "Orch", "Sched")


def _round_metrics(inv):
    """Return one round's (Host, Device, Effective, Orch, Sched) in µs from spans.

    Host/Device/Orch/Sched are span durations; Effective =
    ``max(orch_end, sched_end) - min(orch_start, sched_start)`` from the orch/sched
    spans' device-domain ``ts``/``dur`` (0 when neither is present). All values in
    µs. Column order matches ``_ROUNDS_TABLE_COLUMNS``.
    """
    names = inv.by_name()

    def _dur(key):
        span = names.get(_ROUNDS_TABLE_NAMES[key])
        return span.dur / 1000.0 if span is not None else 0.0

    orch = names.get(_ROUNDS_TABLE_NAMES["orch"])
    sched = names.get(_ROUNDS_TABLE_NAMES["sched"])
    windows = [s for s in (orch, sched) if s is not None]
    if windows:
        start = min(s.ts for s in windows)
        end = max(s.ts + s.dur for s in windows)
        effective = (end - start) / 1000.0
    else:
        effective = 0.0

    return (_dur("host"), _dur("device"), effective, _dur("orch"), _dur("sched"))


def print_rounds_table(buckets, stream=sys.stdout):
    """Print a per-round Host/Device/Effective/Orch/Sched table (µs) for the busiest hid.

    This renders the per-round benchmark table that ``scene_test`` used to print
    inline. The most-invoked hid bucket is treated as the rounds (one row per
    invocation, ordered by ``inv``); each row's metrics come from
    :func:`_round_metrics`. A column is hidden when every row read 0 (e.g.
    device/orch/sched/effective are 0 on a SIMPLER_HOST_STRACE-off build or on sim,
    where the device-domain subdivision is not captured).

    The output format is consumed by ``tools/benchmark_rounds.sh``'s
    framework-table parser (header ``Round  Host (us) …``, ``Avg Host:``
    terminator).
    """
    if not buckets:
        print("No [STRACE] markers found.", file=stream)
        return

    # Busiest hid = the rounds (decode emits one invocation per token; a static
    # L2 example emits one per --rounds repetition).
    _, invs = max(buckets.items(), key=lambda kv: len(kv[1]))
    invs = sorted(invs, key=lambda i: i.inv)
    rows = [_round_metrics(inv) for inv in invs]

    if not rows:
        print("No [STRACE] markers found.", file=stream)
        return

    n = len(rows)
    # Host (col 0) is always captured → averaged over all rounds. Every other
    # column is shown only if some round captured it, and averaged over nonzero.
    host_vals = sorted(r[0] for r in rows)
    host_avg = sum(host_vals) / n

    nz = {}  # col idx -> sorted nonzero values (cols 1..N)
    for idx in range(1, len(_ROUNDS_TABLE_COLUMNS)):
        vals = sorted(r[idx] for r in rows if r[idx] > 0.0)
        if vals:
            nz[idx] = vals
    shown = [0] + [idx for idx in range(1, len(_ROUNDS_TABLE_COLUMNS)) if idx in nz]

    def _avg(idx):
        return sum(nz[idx]) / len(nz[idx])

    header = f"  {'Round':<6}"
    for idx in shown:
        header += f"  {_ROUNDS_TABLE_COLUMNS[idx] + ' (us)':>12}"
    print(header, file=stream)
    print("  " + "-" * (len(header) - 2), file=stream)
    for i, r in enumerate(rows):
        line = f"  {i:<6d}"
        for idx in shown:
            line += f"  {r[idx]:>12.1f}"
        print(line, file=stream)

    summary = f"  Avg Host: {host_avg:.1f} us"
    for idx in shown[1:]:
        summary += f"  |  Avg {_ROUNDS_TABLE_COLUMNS[idx]}: {_avg(idx):.1f} us"
        if idx == 1:  # device gets a capture-count annotation
            summary += f" [{len(nz[1])}/{n}]"
    summary += f"  ({n} rounds)"
    print(summary, file=stream)

    trim = 10
    if n > 2 * trim:
        tc = n - 2 * trim
        host_trim = sum(host_vals[trim:-trim]) / tc
        msg = f"  Trimmed Avg Host: {host_trim:.1f} us"
        if 1 in nz and len(nz[1]) > 2 * trim:
            dev = nz[1]
            msg += f"  |  Trimmed Avg Device: {sum(dev[trim:-trim]) / (len(dev) - 2 * trim):.1f} us"
        msg += f"  (dropped {trim} low + {trim} high, {tc} rounds used)"
        print(msg, file=stream)


def _bucket_label(buckets, hid):
    """Short human label for an hid: 'decode' (busiest bucket) / 'prefill' (once) / hid prefix."""
    if not buckets:
        return hid[:8]
    ordered = sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)
    if hid == ordered[0][0] and len(ordered[0][1]) > 1:
        return "decode"
    if len(buckets.get(hid, [])) == 1:
        return "prefill"
    return hid[:8]


def to_chrome_trace(invocations, buckets=None):
    """Build a Chrome-trace / Perfetto event list with readable nested tracks.

    Each invocation gets its own named process lane ("decode inv=3" /
    "prefill inv=1"), and within it host spans and device (``clk=dev``) spans go
    to two separate threads — because host ``ts`` is steady_clock while device
    ``ts`` is a device-clock offset, the two are NOT on a common timeline and
    must not share a track. Within each track the spans nest by their own
    ``ts``/``dur`` (Perfetto renders containment as nested slices), and ``depth``
    is carried so the structure is unambiguous.
    """
    events = []
    lane_map = {}
    for inv in invocations:
        label = _bucket_label(buckets, inv.hid) if buckets else inv.hid[:8]
        # One process lane per invocation; host vs device on separate tracks.
        # Key by (pid, inv): `inv` is only unique within a pid, so distinct
        # processes (L3 parent + L2 children) can share inv values — mapping the
        # pair to a dense lane id keeps their lanes from merging in Perfetto.
        key = (inv.pid, inv.inv)
        if key not in lane_map:
            lane_map[key] = len(lane_map) + 1
        lane = lane_map[key]
        host_tid, dev_tid = 0, 1
        events.append(
            {
                "ph": "M",
                "name": "process_name",
                "pid": lane,
                "tid": host_tid,
                "args": {"name": f"{label} inv={inv.inv} (pid={inv.pid})"},
            }
        )
        events.append({"ph": "M", "name": "thread_name", "pid": lane, "tid": host_tid, "args": {"name": "host"}})
        events.append(
            {"ph": "M", "name": "thread_name", "pid": lane, "tid": dev_tid, "args": {"name": "device (clk=dev)"}}
        )
        for s in inv.spans:
            events.append(
                {
                    "name": s.name,
                    "ph": "X",
                    "ts": s.ts / 1000.0,  # Chrome trace ts is microseconds
                    "dur": s.dur / 1000.0,
                    "pid": lane,
                    "tid": dev_tid if s.is_device else host_tid,
                    "args": {"inv": s.inv, "hid": s.hid, "depth": s.depth, "attrs": s.attrs},
                }
            )
    return {"traceEvents": events, "displayTimeUnit": "ms"}


def _print_inv_tree(inv, stream=sys.stdout):
    """Print one invocation's spans as a nested tree built from the dotted span
    names (so e.g. ``simpler_run.bind.args`` nests under ``simpler_run.bind``),
    NOT from depth+ts — host (steady_clock) and device (``clk=dev``) spans live
    on different clocks, so timestamp containment across domains is meaningless;
    the dotted name is the unambiguous parent link. Siblings are ordered by
    ``ts``. Device spans are tagged ``[dev]``; durations are µs."""
    by_name = {s.name: s for s in inv.spans}
    # children[parent_name] = [child span...]; a span's parent is its name minus
    # the last dotted segment (if that prefix is itself a known span).
    children = {}
    roots = []
    for s in inv.spans:
        parent = s.name.rsplit(".", 1)[0] if "." in s.name else None
        if parent is not None and parent in by_name:
            children.setdefault(parent, []).append(s)
        else:
            roots.append(s)

    def emit(s, indent):
        tag = " [dev]" if s.is_device else ""
        leaf = s.name.rsplit(".", 1)[-1] if "." in s.name else s.name
        stream.write(f"{'  ' * indent}{leaf:<22}{tag:>6}  {s.dur / 1000.0:>12.1f} us\n")
        kids = sorted(children.get(s.name, []), key=lambda x: x.ts)
        # orch and sched run concurrently (see docs/dfx/device-phases.md): render
        # them on ONE line, left = orch, right = sched, under their merged window
        # `Effective = orch ∪ sched`, instead of as two sequential-looking rows.
        has_sched = any(k.name.rsplit(".", 1)[-1] == "sched" for k in kids)
        has_orch = any(k.name.rsplit(".", 1)[-1] == "orch" for k in kids)
        for c in kids:
            cleaf = c.name.rsplit(".", 1)[-1]
            if cleaf == "orch" and has_sched:
                sched = next(k for k in kids if k.name.rsplit(".", 1)[-1] == "sched")
                eff = (max(c.ts + c.dur, sched.ts + sched.dur) - min(c.ts, sched.ts)) / 1000.0
                base = "  " * (indent + 1)
                # Effective = the merged orch ∪ sched window, with the two
                # concurrent children shown side by side on the indented line
                # below it (see docs/dfx/device-phases.md).
                stream.write(f"{base}{'Effective':<22} [dev]  {eff:>12.1f} us\n")
                stream.write(f"{base}  orch {c.dur / 1000.0:.1f}  ∥  sched {sched.dur / 1000.0:.1f}   (concurrent)\n")
            elif cleaf == "sched" and has_orch:
                continue  # shown beside orch on the Effective line above
            else:
                emit(c, indent + 1)

    for r in sorted(roots, key=lambda x: x.ts):
        emit(r, 0)


def print_tree(buckets, stream=sys.stdout):
    """Per-callable, per-invocation indented tree of spans (the nested view)."""
    if not buckets:
        print("No [STRACE] markers found.", file=stream)
        return
    ordered = sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)
    for hid, invs in ordered:
        label = _bucket_label(buckets, hid)
        print(f"callable hid={hid} ({label}) — {len(invs)} invocation(s)", file=stream)
        # Show the first invocation's tree as representative (they share structure).
        _print_inv_tree(invs[0], stream=stream)
        print(file=stream)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="path to a host/CANN log containing [STRACE] lines (or '-' for stdin)")
    ap.add_argument(
        "--trace-out", help="write a Chrome-trace/Perfetto JSON here (load in chrome://tracing or perfetto)"
    )
    ap.add_argument(
        "--rounds-table",
        action="store_true",
        help="print a per-round Host/Device/Orch/Sched table (the format tools/benchmark_rounds.sh "
        "parses) instead of the per-callable TPOT table",
    )
    ap.add_argument(
        "--tree",
        action="store_true",
        help="print an indented nested span tree per callable (device_wall → sub-phases), "
        "instead of the per-callable TPOT table",
    )
    args = ap.parse_args(argv)

    if args.log == "-":
        lines = sys.stdin.readlines()
    else:
        with open(args.log, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

    spans = list(parse_spans(lines))
    invocations = group_invocations(spans)
    buckets = bucket_by_hid(invocations)

    if args.rounds_table:
        print_rounds_table(buckets)
    elif args.tree:
        print_tree(buckets)
    else:
        print_tpot_table(buckets)

    if args.trace_out:
        with open(args.trace_out, "w", encoding="utf-8") as f:
            json.dump(to_chrome_trace(invocations, buckets), f)
        print(f"Wrote Chrome trace: {args.trace_out} ({len(spans)} spans)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
