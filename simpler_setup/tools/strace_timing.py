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
``LOG_TIMING``. Device-domain phases (AICPU subdivision of the on-NPU wall)
are emitted by the host after readback as ``clk=dev`` spans nested under
``chip.run.runner_run.device_wall``.

Runtimes emit only the device spans they implement. Both current runtimes emit
``device_wall``; the finer orch/sched phase subdivision is TMR-specific.

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
    * a per-callable TPOT table (each invocation's chip.run dur + the mean
      of each sub-stage across invocations), and
    * optionally a Chrome-trace / Perfetto JSON (``--trace-out``): one ``ph:"X"``
      event per span on a synthetic per-invocation lane, so each host call tree
      renders as nested slices; host events also carry wall time when the log
      contains a matching ``CLOCK_ANCHOR``, or
    * a host scheduler swimlane (``--swimlane``) whose lanes are the real OS
      pid/tid, except that a thread which interleaved runs is split into one lane
      per pipeline slot so each lane reads as a sequence; cross-thread handoffs
      are Chrome flow events and host events carry matching anchor wall time.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

# A record's attribute list runs to the end of its line, so what bounds it is the
# lookahead for the next record's log prefix — which is why this pattern exists at
# all. Two complete records do share one physical line: ranks forked by an L3
# share the capture fd. The func segment excludes ':' because `LOG_TIMING` passes
# `__FUNCTION__`, an unqualified name. A qualified name containing '::' stops
# this alternative from matching, leaving `[STRACE]` to bound the record.
_HOST_LOG_TIME = r"\[mono_ns=\d+\]"
_HOST_LOG_PREFIX = _HOST_LOG_TIME + r"\[T0x[0-9a-fA-F]+\]\[[A-Z]+\]\s+[^:\r\n]+:\s+"
# MULTILINE anchors the `$` alternative at every line end, so a caller passing a
# multi-line blob rather than one line per item keeps every record but the last.
_STRACE_RE = re.compile(
    r"\[STRACE\]\s+v=(?P<v>\d+)\s+pid=(?P<pid>\d+)\s+tid=(?P<tid>\d+)\s+"
    r"inv=(?P<inv>\d+)\s+hid=(?P<hid>[0-9a-fA-F]+)\s+depth=(?P<depth>\d+)\s+"
    r"name=(?P<name>\S+)\s+ts=(?P<ts>\d+)\s+dur=(?P<dur>\d+)(?P<attrs>.*?)"
    rf"(?={_HOST_LOG_PREFIX}|\[STRACE\]|\r?$)",
    re.MULTILINE,
)
# A record start, matched independently of whether the rest of that record
# survived the write that emitted it.
_STRACE_HEAD_RE = re.compile(r"\[STRACE\]\s+v=\d+")
# The writer's own loss report, emitted at each quiescent boundary when the drop
# total has grown. The counters live in process memory and die with it, so this
# record is the only way a reader holding just the log learns that records are
# missing — and the breakdown says which knob is wrong.
_DROP_SUMMARY_RE = re.compile(
    r"\[HOSTLOG_DROPS\]\s+v=(?P<v>\d+)\s+pid=(?P<pid>\d+)\s+new=(?P<new>\d+)\s+total=(?P<total>\d+)\s+"
    r"queue_full=(?P<queue_full>\d+)\s+claim_exhausted=(?P<claim_exhausted>\d+)\s+"
    r"output_failed=(?P<output_failed>\d+)\s+not_admitted=(?P<not_admitted>\d+)",
)
_CLOCK_ANCHOR_RE = re.compile(
    r"\[mono_ns=\d+\]\[T0x[0-9a-fA-F]+\]\[TIMING\]\s+clock_anchor:\s+"
    r"\[CLOCK_ANCHOR\]\s+v=(?P<v>\d+)\s+pid=(?P<pid>\d+)\s+"
    r"mono_ns=(?P<mono_ns>\d+)\s+wall_ns=(?P<wall_ns>\d+)[ \t]*\r?$",
    re.MULTILINE,
)
# The emitter percent-encodes any byte that would otherwise be record grammar —
# see `encode_host_span_field` in src/common/log/host_log.cpp.
_PERCENT_ESCAPE_RE = re.compile(r"%([0-9A-Fa-f]{2})")
# One file per process, written under a run's `output_prefix` when the host logger
# writes to files rather than stderr. It holds everything that logger emits, so a
# process's spans and its `[CLOCK_ANCHOR]` are in the same file. See
# docs/dfx/host-trace.md.
_LOG_FILE_GLOB = "host.*.log"


def expand_log_source(source):
    """Resolve one CLI input to the files to read.

    A directory expands to its per-process log files, so a run's
    ``output_prefix`` can be passed as-is instead of being globbed by the caller.
    Sorted, because a reader comparing two runs should not have to care that the
    shell and the filesystem disagree about order.

    Public because the other log readers in this package take the same input and
    must resolve it the same way.
    """
    path = pathlib.Path(source)
    if not path.is_dir():
        return [path]
    log_files = sorted(path.glob(_LOG_FILE_GLOB))
    if not log_files:
        raise SystemExit(f"{source} is a directory but holds no {_LOG_FILE_GLOB} files")
    return log_files


def decode_field(text):
    """Reverse the emitter's percent-encoding of a name or attribute value.

    A field the emitter truncated ends in ``~``, which is left in place: it is a
    marker that the value is incomplete, not an encoded byte.
    """
    return _PERCENT_ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), text)


@dataclass
class ClockAnchor:
    pid: int
    mono_ns: int
    wall_ns: int

    def to_wall_ns(self, monotonic_ns):
        return self.wall_ns + monotonic_ns - self.mono_ns


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


class NativeOverlapError(ValueError):
    """Raised when pipeline markers do not prove native preparation overlap."""


@dataclass(frozen=True)
class NativeDispatchIdentity:
    pid: int
    run_id: int
    dispatch_id: int
    run_epoch: int
    slot_id: int
    generation: int

    @property
    def sequence(self):
        """Submission order within a process.

        ``dispatch_id`` is the scheduler's, and is zero on the direct-chip lane,
        which allocates none; ``run_epoch`` is a per-process monotonic counter
        that is always set. Neither field stands in for the other in the record —
        the choice is made here, where it is visible.
        """
        return self.dispatch_id or self.run_epoch


@dataclass(frozen=True)
class NativeOverlapCheck:
    predecessor: NativeDispatchIdentity
    successor: NativeDispatchIdentity


@dataclass
class Invocation:
    """All spans emitted by one simpler_run call (one (pid, inv) group)."""

    pid: int
    inv: int
    hid: str
    spans: list = field(default_factory=list)

    def root(self):
        """The depth-0 span (chip.run), or None if absent."""
        for s in self.spans:
            if s.depth == 0:
                return s
        return None

    def by_name(self):
        m = {}
        for s in self.spans:
            previous = m.get(s.name)
            if previous is None or s.ts < previous.ts:
                m[s.name] = s
        return m


def count_record_heads(lines):
    """Count ``[STRACE]`` record starts, torn ones included.

    Pairs with :func:`parse_spans`, which yields only records that survived
    intact. A shortfall between the two counts is instrumentation loss, and
    without it a torn record is indistinguishable from a real measurement.
    """
    return sum(len(_STRACE_HEAD_RE.findall(line)) for line in lines)


def parse_drop_summaries(lines):
    """Return the cumulative loss report per process, keyed by pid.

    A process reports a growth at every quiescent boundary, so the last record
    for a pid carries its running totals. Keyed by pid because each process has
    its own queue and its own counters.
    """
    latest = {}
    for line in lines:
        for match in _DROP_SUMMARY_RE.finditer(line):
            if int(match["v"]) != 1:
                continue
            pid = int(match["pid"])
            total = int(match["total"])
            previous = latest.get(pid)
            if previous is None or total >= previous["total"]:
                latest[pid] = {
                    key: int(match[key])
                    for key in ("total", "queue_full", "claim_exhausted", "output_failed", "not_admitted")
                }
    return latest


def warn_about_lost_records(lines, spans):
    """Report both ways a log can be incomplete, before any timing is derived.

    They are separate channels and only one is visible in the records: a torn
    record leaves a header behind, while a dropped one leaves nothing at all and
    is knowable only from the writer's own summary.
    """
    heads = count_record_heads(lines)
    if heads > len(spans):
        print(
            f"warning: {heads - len(spans)} of {heads} [STRACE] records are incomplete and are "
            "excluded from the timing below",
            file=sys.stderr,
        )
    for pid, counts in sorted(parse_drop_summaries(lines).items()):
        print(
            f"warning: pid {pid} dropped {counts['total']} host-log record(s) before they reached the "
            f"destination (queue_full={counts['queue_full']} claim_exhausted={counts['claim_exhausted']} "
            f"output_failed={counts['output_failed']} not_admitted={counts['not_admitted']}); "
            "the timing below is computed from an incomplete log",
            file=sys.stderr,
        )


def parse_clock_anchors(lines):
    """Yield the per-process monotonic-to-wall mappings in a log."""
    for line in lines:
        for match in _CLOCK_ANCHOR_RE.finditer(line):
            if int(match["v"]) != 1:
                continue
            yield ClockAnchor(
                pid=int(match["pid"]),
                mono_ns=int(match["mono_ns"]),
                wall_ns=int(match["wall_ns"]),
            )


def parse_spans(lines):
    """Yield every complete span, including adjacent records on one line."""
    for line in lines:
        for m in _STRACE_RE.finditer(line):
            yield Span(
                pid=int(m["pid"]),
                tid=int(m["tid"]),
                inv=int(m["inv"]),
                hid=m["hid"].lower(),
                depth=int(m["depth"]),
                name=decode_field(m["name"]),
                ts=int(m["ts"]),
                dur=int(m["dur"]),
                attrs=m["attrs"].strip(),
            )


def _format_wall_time(wall_ns):
    seconds, nanoseconds = divmod(wall_ns, 1_000_000_000)
    prefix = datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{prefix}.{nanoseconds:09d}Z"


def _wall_time_args(pid, monotonic_ns, anchors_by_pid):
    anchor = anchors_by_pid.get(pid)
    if anchor is None or anchor.mono_ns > monotonic_ns:
        return {}
    wall_ns = anchor.to_wall_ns(monotonic_ns)
    return {"wall_ts_ns": str(wall_ns), "wall_time": _format_wall_time(wall_ns)}


def _trace_document(events, anchors_by_pid, **extra):
    document = {"traceEvents": events, "displayTimeUnit": "ms", **extra}
    if anchors_by_pid:
        document["clockAnchors"] = [
            {"pid": anchor.pid, "mono_ns": str(anchor.mono_ns), "wall_ns": str(anchor.wall_ns)}
            for anchor in sorted(anchors_by_pid.values(), key=lambda item: item.pid)
        ]
    return document


# A span name leads with the word for the level that produced it. The words come
# from `simpler.worker_level.WorkerLevel`, and this parser cannot import the
# runtime package, so this is a second copy of that ladder.
# `test_every_level_word_the_ladder_names_is_a_word_this_parser_knows` compares
# the two as sets: a level added there and not here makes `span_family` answer
# `unknown` for a word the runtime emits, which puts per-task spans into
# invocation grouping instead of excluding them.
_CHIP_WORD = "chip"
_CORE_WORD = "core"
_NODE_WORDS = ("node", "network1", "network2", "network3")

# Reserved for producers outside simpler: `ext.<producer>.<span>`. Without a
# reserved word a caller's span called `host.foo` would parse as one of ours.
_EXTERNAL_WORD = "ext"


def span_family(name):
    """Classify `name` by the producer its leading word names.

    Returns ``"chip"``, ``"core"``, ``"node"``, ``"external"``, or ``"unknown"``.
    Every level at or above L3 answers ``"node"`` — the family takes its lowest
    member's name — because they run the same orchestrator and scheduler code and
    so form one family whichever word a given process resolved to.
    """
    head = name.split(".", 1)[0]
    if head == _EXTERNAL_WORD:
        return "external"
    if head == _CHIP_WORD:
        return "chip"
    if head == _CORE_WORD:
        return "core"
    if head in _NODE_WORDS:
        return "node"
    return "unknown"


def node_span_leaf(name):
    """The part of a node-family span name after its level word, else ``None``.

    Call sites match on the leaf rather than the whole name because the level
    word varies by the process that emitted it — ``node.submit`` from an L3 and
    ``network1.submit`` from an L4 are the same decision point.
    """
    if span_family(name) != "node":
        return None
    _, _, leaf = name.partition(".")
    return leaf


def external_producer(name):
    """The producer segment of an ``ext.<producer>.<span>`` name, else ``None``.

    All three segments must be present and non-empty: ``ext.pypto.decode_layer``
    attributes to ``pypto``, while ``ext.foo`` names no span of its own,
    ``ext..foo`` names no producer, and ``ext.pypto.`` names a producer but no
    span. None of those resolve, so a malformed name lands in an unattributed
    lane instead of another producer's.

    Attribution is separate from :func:`span_family`, which answers only whether
    a name is ours. A name under ``ext.`` is never ours whatever follows it.
    """
    if span_family(name) != "external":
        return None
    parts = name.split(".")
    if len(parts) < 3 or not parts[1] or not parts[2]:
        return None
    return parts[1]


def invocation_spans(spans):
    """Return the spans the invocation-keyed views may consume.

    Those views key on ``(pid, inv)``, and ``inv`` is a native run epoch. Two
    families carry none, so admitting either would group all of its spans into
    one forged invocation: the per-task ``node``/``network*`` scheduler family,
    and anything an external producer emitted under ``ext.``.

    Everything else is kept, **including a name this parser does not recognize**.
    Dropping an unfamiliar family silently is how ``chip.prewarm.build`` once
    vanished from the tables.
    """
    return [span for span in spans if span_family(span.name) not in ("node", "external")]


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


# The spans one native run contributes to the overlap proof. All three already
# exist in the `chip.run` tree; only `claim_release` was added for it.
_PREPARE_SPAN = "chip.run.bind"
_DEVICE_SPAN = "chip.run.runner_run"
_RELEASE_SPAN = "chip.run.claim_release"
_NATIVE_REQUIRED_SPANS = (_PREPARE_SPAN, _DEVICE_SPAN, _RELEASE_SPAN)
_PIPELINE_IDENTITY_FIELDS = ("run_id", "dispatch_id", "run_epoch", "slot_id", "generation")


def _native_dispatches(invocations):
    """Map each native run's identity to its ``{span name: span}``.

    The identity rides on the root ``chip.run`` span, and every sub-span of
    that run shares its ``(pid, inv)`` — so grouping by invocation is what joins
    the windows to the identity. An invocation whose root carries no identity is
    not a phased native run and is skipped.
    """
    dispatches = {}
    for invocation in invocations:
        root = invocation.root()
        if root is None:
            continue
        attrs = _parsed_attrs(root)
        if not any(field in attrs for field in _PIPELINE_IDENTITY_FIELDS):
            continue
        missing = [field for field in _PIPELINE_IDENTITY_FIELDS if field not in attrs]
        if missing:
            raise NativeOverlapError(
                f"{root.name} (pid={root.pid} inv={root.inv}) is missing identity field(s): {', '.join(missing)}"
            )
        try:
            identity = NativeDispatchIdentity(
                pid=root.pid,
                run_id=int(attrs["run_id"]),
                dispatch_id=int(attrs["dispatch_id"]),
                run_epoch=int(attrs["run_epoch"]),
                slot_id=int(attrs["slot_id"]),
                generation=int(attrs["generation"]),
            )
        except (TypeError, ValueError) as exc:
            raise NativeOverlapError(f"{root.name} has a non-integer identity") from exc
        if identity.sequence == 0:
            continue
        if identity in dispatches:
            raise NativeOverlapError(f"duplicate identity for pid={identity.pid} sequence={identity.sequence}")
        dispatches[identity] = invocation.by_name()
    return dispatches


def assert_native_overlap(spans, *, require_hidden=False):
    """Prove native prepare overlap and FIFO launch ordering for adjacent runs.

    Two properties per adjacent pair on one process lane:

    * ``bind(N+1)`` overlaps ``runner_run(N)`` — the intervals intersect, which
      is what makes the successor's preparation concurrent with the
      predecessor's device work.
    * ``runner_run(N+1)`` does not start before ``claim_release(N)``.

    ``bind`` is the successor's own arena build and host orchestration and sits
    inside its prepare, so reading it is conservative: an overlap it reports is
    one the prepare certainly had.

    ``require_hidden`` additionally demands that the preparation *finish* inside
    the predecessor's device window — fully hidden rather than merely
    overlapping. That is a claim about pipeline depth and is sensitive to host
    scheduling, so it is opt-in: a bind that runs long on a contended machine
    still overlaps.
    """
    dispatches = _native_dispatches(group_invocations(invocation_spans(spans)))
    by_pid = defaultdict(list)
    for identity, by_name in dispatches.items():
        missing = [name for name in _NATIVE_REQUIRED_SPANS if name not in by_name]
        if missing:
            raise NativeOverlapError(f"sequence={identity.sequence} is missing span(s): {', '.join(missing)}")
        by_pid[identity.pid].append((identity, by_name))

    checks = []
    for lane in by_pid.values():
        lane.sort(key=lambda item: item[1][_DEVICE_SPAN].ts)
        for (predecessor, pred_spans), (successor, succ_spans) in zip(lane, lane[1:]):
            if successor.sequence <= predecessor.sequence:
                raise NativeOverlapError(
                    f"launch order is not monotonic: predecessor={predecessor.sequence} successor={successor.sequence}"
                )
            pred_device = pred_spans[_DEVICE_SPAN]
            pred_release = pred_spans[_RELEASE_SPAN]
            succ_prepare = succ_spans[_PREPARE_SPAN]
            succ_device = succ_spans[_DEVICE_SPAN]
            pred_device_end = pred_device.ts + pred_device.dur
            succ_prepare_end = succ_prepare.ts + succ_prepare.dur
            if succ_prepare.ts >= pred_device_end or succ_prepare_end <= pred_device.ts:
                raise NativeOverlapError(
                    f"preparation did not overlap: sequence={successor.sequence} prepare="
                    f"[{succ_prepare.ts},{succ_prepare_end}] predecessor_device="
                    f"[{pred_device.ts},{pred_device_end}]"
                )
            if require_hidden and succ_prepare_end > pred_device_end:
                raise NativeOverlapError(
                    f"preparation was not fully hidden: sequence={successor.sequence} prepare_end="
                    f"{succ_prepare_end} predecessor_device_end={pred_device_end}"
                )
            if succ_device.ts < pred_release.ts:
                raise NativeOverlapError(
                    f"device execution reordered before the claim release: sequence={successor.sequence} "
                    f"device_start={succ_device.ts} predecessor_release={pred_release.ts}"
                )
            checks.append(NativeOverlapCheck(predecessor=predecessor, successor=successor))
    if not checks:
        raise NativeOverlapError("need at least two complete native runs on one process lane")
    return checks


def _fmt_us(ns: int) -> str:
    return f"{ns / 1000.0:.1f}"


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _median(values):
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


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
                f"  chip.run: mean={_fmt_us(int(_mean(durs)))}us min={_fmt_us(min(durs))}us max={_fmt_us(max(durs))}us",
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
    "run": "chip.run",
    "device": "chip.run.runner_run.device_wall",
    "orch": "chip.run.runner_run.device_wall.orch",
    "sched": "chip.run.runner_run.device_wall.sched",
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

    return (_dur("run"), _dur("device"), effective, _dur("orch"), _dur("sched"))


def print_rounds_table(buckets, stream=sys.stdout):
    """Print a per-round Host/Device/Effective/Orch/Sched table (µs) for the busiest hid.

    This renders the per-round benchmark table that ``scene_test`` used to print
    inline. The most-invoked hid bucket is treated as the rounds (one row per
    invocation, ordered by ``inv``); each row's metrics come from
    :func:`_round_metrics`. A column is hidden when every row read 0 (e.g.
    device/orch/sched/effective are 0 when their marker is absent; for example,
    HBG emits device wall but has no device-side orch/sched windows).

    The output format is consumed by ``tools/benchmark_rounds.sh``'s
    framework-table parser (header ``Round  Host (us) …``, ``Avg Host:``
    terminator).
    """
    if not buckets:
        print("No [STRACE] markers found.", file=stream)
        return

    # Only a depth-0 chip.run root identifies an official round. Internal
    # prewarm invocations reuse chip.run.* child spans, so matching children
    # would add one fake round and contaminate benchmark averages.
    run_buckets = {
        hid: [inv for inv in invs if (root := inv.root()) is not None and root.name == _ROUNDS_TABLE_NAMES["run"]]
        for hid, invs in buckets.items()
    }
    run_buckets = {hid: invs for hid, invs in run_buckets.items() if invs}
    if not run_buckets:
        print("No official [STRACE] rounds found.", file=stream)
        return

    # Busiest official hid = the rounds (decode emits one invocation per token;
    # a static L2 example emits one per --rounds repetition).
    _, invs = max(run_buckets.items(), key=lambda kv: len(kv[1]))
    invs = sorted(invs, key=lambda i: i.inv)
    rows = [_round_metrics(inv) for inv in invs]

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


def to_chrome_trace(invocations, buckets=None, anchors=None):
    """Build a Chrome-trace / Perfetto event list with readable nested tracks.

    Each invocation gets its own named process lane ("decode inv=3" /
    "prefill inv=1"), and within it host spans and device (``clk=dev``) spans go
    to two separate threads — because host ``ts`` is steady_clock while device
    ``ts`` is a device-clock offset, the two are NOT on a common timeline and
    must not share a track. Within each track the spans nest by their own
    ``ts``/``dur`` (Perfetto renders containment as nested slices), and ``depth``
    is carried so the structure is unambiguous. A matching clock anchor adds
    wall time to the event arguments without changing that monotonic axis.
    """
    anchors_by_pid = {anchor.pid: anchor for anchor in anchors or ()}
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
            event_args = {"inv": s.inv, "hid": s.hid, "depth": s.depth, "attrs": s.attrs}
            if not s.is_device:
                event_args.update(_wall_time_args(s.pid, s.ts, anchors_by_pid))
            events.append(
                {
                    "name": s.name,
                    "ph": "X",
                    "ts": s.ts / 1000.0,  # Chrome trace ts is microseconds
                    "dur": s.dur / 1000.0,
                    "pid": lane,
                    "tid": dev_tid if s.is_device else host_tid,
                    "args": event_args,
                }
            )
    return _trace_document(events, anchors_by_pid)


def _parsed_attrs(span):
    attrs = {}
    for attribute in span.attrs.split():
        key, separator, value = attribute.partition("=")
        if not separator:
            continue
        if re.fullmatch(r"-?\d+", value):
            attrs[key] = int(value)
        else:
            attrs[key] = decode_field(value)
    return attrs


# Highest-precedence match wins. One OS thread emits spans of several roles: the
# scheduler loop is the sole caller of both `dispatch_ready` and
# `manager->progress`, so it emits `node.dispatch` (role=scheduler) alongside
# `node.frame_submit` / `node.activate` / `node.complete`, whose `role=worker` names
# the worker a dispatch targets rather than the thread doing the work.
_HOST_THREAD_ROLES = ("facade", "scheduler", "worker")


def _roots_overlap(entries):
    """Whether two runs were in flight at once on this thread.

    A depth-0 span is one run's whole lifetime, so two of them overlapping means
    the thread interleaved runs. Nesting *within* a run is wanted; nesting one
    run's spans inside another's is an artifact of flattening them onto one lane.
    """
    roots = sorted((span.ts, span.ts + span.dur) for span, _ in entries if span.depth == 0)
    return any(a[1] > b[0] for a, b in zip(roots, roots[1:]))


def _slot_by_invocation(entries):
    """Map ``(pid, inv)`` to the pipeline slot that run held.

    Only the root span carries the identity; its children carry no attributes at
    all. They share the root's ``(pid, inv)``, so that is the join — the same one
    :func:`assert_native_overlap` uses.
    """
    slots = {}
    for span, attrs in entries:
        if "slot_id" in attrs:
            slots[(span.pid, span.inv)] = attrs["slot_id"]
    return slots


def _slot_lanes(entries, slot_by_invocation):
    """Split one thread's spans by pipeline slot, or None to leave it on its tid.

    The pipeline slot is what a run holds exclusively, so runs sharing a slot
    cannot overlap — which is what makes a per-slot lane render as a plain
    sequence instead of false containment. Returning None keeps the real-tid
    lane: splitting a thread whose runs never overlapped would only fragment it,
    and the L3 scheduler thread carries a slot while running strictly
    sequentially.
    """
    if not _roots_overlap(entries):
        return None
    by_slot = defaultdict(list)
    for span, attrs in entries:
        slot_id = slot_by_invocation.get((span.pid, span.inv))
        if slot_id is None:
            return None
        by_slot[slot_id].append((span, attrs))
    if len(by_slot) < 2 or any(_roots_overlap(group) for group in by_slot.values()):
        return None
    return dict(sorted(by_slot.items()))


def _lane_name(entries):
    """Name one OS thread's lane from every span it emitted.

    `entries` are that thread's (span, parsed attributes) pairs.

    Role inference reads only our own spans. `role` is an ordinary attribute key,
    so an external producer is free to use it for its own meaning; letting one
    reach the loop below would name its lane after one of our roles.
    """
    phase_threads = {attrs.get("host_phase_thread") for _, attrs in entries}
    if "graph_record_worker" in phase_threads:
        return "graph record worker"
    if "graph_submit_main" in phase_threads:
        return "graph submit main"

    roles = set()
    worker_ids = set()
    external = [(span, attrs) for span, attrs in entries if span_family(span.name) == "external"]
    for span, attrs in entries:
        if span_family(span.name) == "external":
            continue
        role = attrs.get("role")
        leaf = node_span_leaf(span.name)
        if role == "facade" or leaf in {"graph_build", "submit"}:
            roles.add("facade")
        elif role in ("scheduler", "worker"):
            roles.add(role)
        elif leaf is not None:
            roles.add("worker")
        if role == "worker":
            worker_ids.add(attrs.get("worker_id"))

    for role in _HOST_THREAD_ROLES:
        if role not in roles:
            continue
        if role == "facade":
            return "orchestrator / facade"
        if role == "scheduler":
            return "scheduler"
        worker_id = worker_ids.pop() if len(worker_ids) == 1 else None
        return f"worker {worker_id}" if worker_id is not None else "worker"

    if any(span_family(span.name) == "chip" for span, _ in entries):
        return "chip child"
    if len(external) == len(entries):
        producers = sorted({producer for span, _ in external if (producer := external_producer(span.name))})
        if producers:
            return f"ext {'/'.join(producers)}"
    return f"tid {entries[0][0].tid}"


def _flow_key(span, attrs):
    run_id = attrs.get("run_id")
    task_slot = attrs.get("task_slot", attrs.get("slot"))
    if run_id is None or task_slot is None:
        return None
    return span.pid, run_id, task_slot


def _assign_lanes(non_device_entries, non_device_threads):
    """Choose a lane per span, and a name per lane.

    A thread that interleaved runs is split by pipeline slot (see
    :func:`_slot_lanes`); every other thread keeps its real tid. Synthetic lane
    ids start past the observed tid space so a split lane cannot collide with a
    real thread's.

    Only our own spans shape the split. A pipeline slot is our concept and
    `slot_id` / `depth` are ordinary record fields, so an external producer that
    writes them would otherwise decide whether one of our threads is split and
    into how many lanes. Its spans stay on the thread's real tid.
    """
    lane_of = {}
    lane_names = {}
    next_synthetic_tid = max(tid for _, tid in non_device_threads) + 1
    ours = [entry for entry in non_device_entries if span_family(entry[0].name) != "external"]
    slot_by_invocation = _slot_by_invocation(ours)
    for pid, tid in non_device_threads:
        on_thread = [entry for entry in non_device_entries if entry[0].pid == pid and entry[0].tid == tid]
        ours_on_thread = [entry for entry in on_thread if span_family(entry[0].name) != "external"]
        slot_lanes = _slot_lanes(ours_on_thread, slot_by_invocation)
        if slot_lanes is None:
            lane_names[(pid, tid)] = _lane_name(on_thread)
            for span, _ in on_thread:
                lane_of[id(span)] = tid
            continue
        external_on_thread = [entry for entry in on_thread if span_family(entry[0].name) == "external"]
        if external_on_thread:
            lane_names[(pid, tid)] = _lane_name(external_on_thread)
            for span, _ in external_on_thread:
                lane_of[id(span)] = tid
        for slot_id, group in slot_lanes.items():
            lane_tid = next_synthetic_tid
            next_synthetic_tid += 1
            lane_names[(pid, lane_tid)] = f"pipeline slot {slot_id} (tid {tid})"
            for span, _ in group:
                lane_of[id(span)] = lane_tid
    return lane_of, lane_names


def load_host_phase_records(paths):
    """Read host phase records from ``host_phase_records.jsonl`` files.

    One JSON object per prepare pass, as written by a runtime with a host prepare
    path. Malformed lines are skipped rather than fatal: the artifact is appended
    to while a run is in flight, so a truncated last line is an ordinary outcome,
    not a broken file. A line that parses but is not an object, or whose
    ``records`` is not a list, is malformed in the same sense and skipped here so
    no consumer has to re-check it.
    """
    passes = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    one_pass = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(one_pass, dict) or not isinstance(one_pass.get("records", []), list):
                    continue
                passes.append(one_pass)
    return passes


# The bind stage's own segments, each one interval inside it. Everything else a
# pass records is an orchestrator operation nested inside the host_orch segment.
_BIND_PHASE_NAMES = frozenset(
    {
        "args",
        "arena_build",
        "static_arena",
        "gm_heap",
        "shared_mem",
        "runtime_init",
        "host_orch",
        "graph_upload",
        "arena_h2d",
        "host_view_close",
    }
)

# Phases a recorder worker emits, so a record carrying a tid of its own belongs
# on the recorder lane rather than the main one. "record_node" is the name the
# runtime emitted for an in-graph task before it was renamed; logs and the
# archived runs cited in docs/investigations/ still carry it, and an unknown
# phase name here is silently attributed to host_main rather than rejected, so
# both spellings stay accepted.
_RECORD_WORKER_PHASE_NAMES = frozenset({"record_in_graph_task", "record_node", "build_definition"})


def host_record_spans(spans, passes):
    """Turn phase records into spans nested under their pass's ``bind``.

    A record carries only ``(pid, inv)`` and a host-clock interval; the thread and
    the tree position come from the ``chip.run.bind`` span of the same
    invocation, which is the stage the records subdivide. Records whose pass has
    no such span are dropped — without it there is no lane to draw them on and no
    parent to nest them under.

    A bind segment sits directly under ``bind``; an orchestrator operation sits a
    level deeper, under the ``host_orch`` segment it happened inside.

    The clock needs no conversion: both sides are the same CLOCK_MONOTONIC axis.

    **A segment the log already carries as a span is skipped here.** The runtime
    emits each bind segment as a `[STRACE]` span, and that span carries the
    segment's attributes — byte and tensor counts, fault and CPU counters — where
    this artifact carries only ``detail``. So the log wins and the artifact fills
    in the segments it has that the log does not: a run whose breakdown switch was
    off collects records without emitting spans, and then the artifact is the only
    source. Orchestrator operations have no span form at all and always come from
    here.

    Returns the spans, the number of passes with no matching ``bind`` span, and
    the number of records skipped as already present in the log.
    """
    bind_by_key = {(span.pid, span.inv): span for span in spans if span.name == _PREPARE_SPAN and not span.is_device}
    already_in_log = {(span.pid, span.inv, span.name) for span in spans if not span.is_device}
    out = []
    dropped_passes = 0
    skipped_records = 0
    for one_pass in passes:
        key = (one_pass.get("pid"), one_pass.get("inv"))
        parent = bind_by_key.get(key)
        if parent is None:
            dropped_passes += 1
            continue
        for record in one_pass.get("records", []):
            try:
                start = int(record["start_ns"])
                end = int(record["end_ns"])
                phase = str(record["phase"])
            except (KeyError, TypeError, ValueError):
                continue
            raw_tid = record.get("tid", parent.tid)
            try:
                record_tid = int(raw_tid)
            except (TypeError, ValueError):
                record_tid = parent.tid
            if record_tid <= 0:
                record_tid = parent.tid
            is_record_worker = "tid" in record and record_tid != parent.tid
            if phase in _BIND_PHASE_NAMES:
                name = f"{_PREPARE_SPAN}.{phase}"
                if (parent.pid, parent.inv, name) in already_in_log:
                    skipped_records += 1
                    continue
                depth = parent.depth + 1
                record_tid = parent.tid
                phase_thread = "host_main"
            else:
                name = f"{_PREPARE_SPAN}.host_orch.{phase}"
                depth = parent.depth + 2
                if phase in _RECORD_WORKER_PHASE_NAMES and is_record_worker:
                    phase_thread = "graph_record_worker"
                elif phase == "graph_submit":
                    phase_thread = "graph_submit_main"
                else:
                    phase_thread = "host_main"
            out.append(
                Span(
                    pid=parent.pid,
                    tid=record_tid,
                    inv=parent.inv,
                    hid=parent.hid,
                    depth=depth,
                    name=name,
                    ts=start,
                    dur=max(0, end - start),
                    attrs=(f"detail={record.get('detail', 0)} src=host_phase_records host_phase_thread={phase_thread}"),
                )
            )
    return out, dropped_passes, skipped_records


def _process_label(pid, process_spans):
    """Name one process from the families of the spans it emitted.

    Any span of ours makes the process ours, and the host family wins over the
    chip one. Sharing is the common case rather than the exception: a producer
    calling the public tracing API emits from inside our own host process, so
    `ext.` spans alongside ours say nothing about whose process it is. Only a
    process that emitted external spans and nothing else belongs to a producer,
    and then the label carries no `simpler` prefix — it is not our process.
    """
    families = {span_family(span.name) for span in process_spans}
    if "node" in families:
        return f"simpler node (pid={pid})"
    if families == {"external"}:
        producers = sorted({producer for span in process_spans if (producer := external_producer(span.name))})
        named = "/".join(producers) if producers else "unattributed"
        return f"external producer {named} (pid={pid})"
    return f"simpler chip child (pid={pid})"


def to_host_swimlane(spans, anchors=None):
    """Build a real-pid/tid host scheduling timeline for Perfetto.

    Host timestamps remain on their shared CLOCK_MONOTONIC axis. Chrome Trace
    JSON has one timestamp axis, so raw ``clk=dev`` events cannot be rendered
    alongside host events without either a false clock alignment or a huge
    empty interval. Keep those raw events in ``unalignedDeviceSpans`` for
    inspection, but do not add them to Perfetto's visible ``traceEvents``.
    Matching anchors add wall time as host-event metadata only.
    """
    anchors_by_pid = {anchor.pid: anchor for anchor in anchors or ()}
    # (span, parsed attributes) pairs, so the attributes travel with their span
    # through every partition below. `Span` is an unhashable dataclass, so a
    # side table would have to be keyed on identity.
    entries = [(span, _parsed_attrs(span)) for span in spans]
    events = []

    non_device_entries = [entry for entry in entries if not entry[0].is_device]
    device_entries = [entry for entry in entries if entry[0].is_device]
    non_device_pids = sorted({span.pid for span, _ in non_device_entries})
    non_device_threads = sorted({(span.pid, span.tid) for span, _ in non_device_entries})

    lane_of, lane_names = _assign_lanes(non_device_entries, non_device_threads)

    for pid in non_device_pids:
        # Every span the process emitted, device-clock ones included: a `clk=dev`
        # span is ours too, so it is evidence about whose process this is even
        # though it never reaches the visible timeline.
        process_spans = [span for span in spans if span.pid == pid]
        events.append(
            {
                "ph": "M",
                "name": "process_name",
                "pid": pid,
                "tid": 0,
                "args": {"name": _process_label(pid, process_spans)},
            }
        )
    for (pid, lane_tid), name in sorted(lane_names.items()):
        events.append(
            {
                "ph": "M",
                "name": "thread_name",
                "pid": pid,
                "tid": lane_tid,
                "args": {"name": name},
            }
        )
    for span, parsed in sorted(
        non_device_entries, key=lambda item: (item[0].ts, item[0].pid, item[0].tid, item[0].name)
    ):
        event_args = {
            "inv": span.inv,
            "hid": span.hid,
            "depth": span.depth,
            "attrs": span.attrs,
            "os_tid": span.tid,
            **parsed,
        }
        event_args.update(_wall_time_args(span.pid, span.ts, anchors_by_pid))
        events.append(
            {
                "name": span.name,
                "ph": "X",
                "ts": span.ts / 1000.0,
                "dur": span.dur / 1000.0,
                "pid": span.pid,
                "tid": lane_of[id(span)],
                "args": event_args,
            }
        )

    submits = defaultdict(list)
    for span, attrs in non_device_entries:
        if node_span_leaf(span.name) != "submit":
            continue
        key = _flow_key(span, attrs)
        if key is not None:
            submits[key].append(span)
    for candidates in submits.values():
        candidates.sort(key=lambda item: item.ts)

    dispatches = []
    for span, attrs in non_device_entries:
        if node_span_leaf(span.name) != "dispatch":
            continue
        key = _flow_key(span, attrs)
        source = None
        if key is not None:
            for candidate in submits.get(key, []):
                if candidate.ts > span.ts:
                    break
                source = candidate
        if source is None:
            continue
        dispatches.append((source, span, attrs))

    for flow_id, (source, destination, attrs) in enumerate(sorted(dispatches, key=lambda item: item[1].ts), start=1):
        dispatch_key = (
            f"dispatch:{source.pid}:{attrs['run_id']}:{attrs.get('task_slot', attrs.get('slot'))}:"
            f"{attrs.get('group_index', -1)}:{attrs.get('worker_id', -1)}:{attrs.get('dispatch_id', 0)}"
        )
        source_ts = min(source.ts + source.dur, destination.ts)
        source_args = {"dispatch_key": dispatch_key, **_wall_time_args(source.pid, source_ts, anchors_by_pid)}
        destination_args = {
            "dispatch_key": dispatch_key,
            **_wall_time_args(destination.pid, destination.ts, anchors_by_pid),
        }
        events.append(
            {
                "name": "task dispatch",
                "cat": "host.scheduler",
                "ph": "s",
                "id": flow_id,
                "ts": source_ts / 1000.0,
                "pid": source.pid,
                "tid": lane_of[id(source)],
                "args": source_args,
            }
        )
        events.append(
            {
                "name": "task dispatch",
                "cat": "host.scheduler",
                "ph": "f",
                "id": flow_id,
                "ts": destination.ts / 1000.0,
                "pid": destination.pid,
                "tid": lane_of[id(destination)],
                "args": destination_args,
            }
        )

    unaligned_device_spans = []
    for span, attrs in sorted(
        device_entries, key=lambda item: (item[0].pid, item[0].inv, item[0].ts, item[0].tid, item[0].name)
    ):
        unaligned_device_spans.append(
            {
                "name": span.name,
                "ts_ns": span.ts,
                "dur_ns": span.dur,
                "pid": span.pid,
                "tid": span.tid,
                "inv": span.inv,
                "hid": span.hid,
                "depth": span.depth,
                "attrs": {"raw": span.attrs, **attrs},
            }
        )

    return _trace_document(events, anchors_by_pid, unalignedDeviceSpans=unaligned_device_spans)


def _print_agg_tree(invs, stream=sys.stdout):
    """Print a callable's spans as a nested tree built from the dotted span
    names (so e.g. ``chip.run.bind.args`` nests under ``chip.run.bind``),
    NOT from depth+ts — host (steady_clock) and device (``clk=dev``) spans live
    on different clocks, so timestamp containment across domains is meaningless;
    the dotted name is the unambiguous parent link. Device spans are tagged
    ``[dev]``; durations are µs.

    Each node's duration is the **median across every invocation** of this
    callable, not one invocation's value. A single-invocation tree would mislead
    on a callable whose invocations differ in cost — e.g. qwen3 decode, where the
    pypto-serving profile warmup dispatches a tiny-KV decode step (seq_len≈257)
    before the real steps: its Effective (~28 ms) is far below the steady-state
    (~40 ms at 3.5k context). The median is robust to that warmup outlier."""
    # Per-span-name median duration across all invocations. by_name() rebuilds
    # its dict per call, so materialize each invocation's map once and reuse it
    # for both the medians here and the per-inv Effective loop below.
    by_names = [inv.by_name() for inv in invs]
    dur_samples: dict = defaultdict(list)
    for bn in by_names:
        for name, span in bn.items():
            dur_samples[name].append(span.dur)
    med = {name: _median(ds) for name, ds in dur_samples.items()}

    # Structure + ts ordering from the LAST invocation — for qwen decode the
    # warmup step is inv 0, so the last is a steady-state one; either way the
    # dotted-name tree shape is identical across invocations.
    ref = invs[-1]
    by_name = {s.name: s for s in ref.spans}
    children = {}
    roots = []
    for s in ref.spans:
        parent = s.name.rsplit(".", 1)[0] if "." in s.name else None
        if parent is not None and parent in by_name:
            children.setdefault(parent, []).append(s)
        else:
            roots.append(s)

    def emit(s, indent):
        tag = " [dev]" if s.is_device else ""
        leaf = s.name.rsplit(".", 1)[-1] if "." in s.name else s.name
        stream.write(f"{'  ' * indent}{leaf:<22}{tag:>6}  {med[s.name] / 1000.0:>12.1f} us\n")
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
                # Effective is per-invocation (orch ∪ sched depends on both
                # markers' overlap in that inv), so take the median of the
                # per-inv Effective values rather than combining the two medians.
                effs = []
                for bn in by_names:
                    o, sc = bn.get(c.name), bn.get(sched.name)
                    if o is not None and sc is not None:
                        effs.append(max(o.ts + o.dur, sc.ts + sc.dur) - min(o.ts, sc.ts))
                eff = _median(effs) / 1000.0
                base = "  " * (indent + 1)
                # Effective = the merged orch ∪ sched window, with the two
                # concurrent children shown side by side on the indented line
                # below it (see docs/dfx/device-phases.md).
                stream.write(f"{base}{'Effective':<22} [dev]  {eff:>12.1f} us\n")
                stream.write(
                    f"{base}  orch {med[c.name] / 1000.0:.1f}  ∥  sched {med[sched.name] / 1000.0:.1f}   (concurrent)\n"
                )
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
        n = len(invs)
        suffix = f" — median of {n} invocation(s)" if n > 1 else " — 1 invocation"
        print(f"callable hid={hid} ({label}){suffix}", file=stream)
        _print_agg_tree(invs, stream=stream)
        print(file=stream)


def write_host_swimlane(args, spans, anchors):
    """Write the host swimlane, adding whatever prepare-path detail is available.

    A runtime's bind segments are `[STRACE]` spans and are already in ``spans``.
    The per-event artifact adds what has no span form — the orchestrator
    operations inside ``host_orch`` — and stands in for the segments themselves on
    a run that collected records without emitting their spans.
    """
    lane_spans = spans
    record_count = 0
    skipped = 0
    if args.host_phase_records:
        passes = load_host_phase_records(args.host_phase_records)
        extra, orphaned, skipped = host_record_spans(spans, passes)
        lane_spans = lane_spans + extra
        record_count = len(extra)
        if orphaned:
            print(
                f"warning: {orphaned} phase-record pass(es) had no matching chip.run.bind span "
                "in this log and were dropped — is the log from the same run as the artifact?",
                file=sys.stderr,
            )
    with open(args.swimlane, "w", encoding="utf-8") as f:
        json.dump(to_host_swimlane(lane_spans, anchors=anchors), f)
    host_count = sum(not span.is_device for span in spans)
    extras = []
    if record_count:
        extras.append(f"{record_count} host phase records")
    if args.host_phase_records and skipped:
        extras.append(f"{skipped} segment(s) already in the log")
    suffix = (", " + ", ".join(extras)) if extras else ""
    print(
        f"Wrote host swimlane: {args.swimlane} "
        f"({host_count} host spans, {len(spans) - host_count} unaligned device spans{suffix})"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "log",
        nargs="+",
        help="one or more host/CANN logs containing [STRACE] lines, '-' for stdin, or a directory holding "
        f"{_LOG_FILE_GLOB} files (a run's output_prefix). Several inputs are concatenated: records carry their "
        "own pid, so a whole run's per-process logs, or logs from several runs, can be passed together.",
    )
    ap.add_argument(
        "--trace-out", help="write a Chrome-trace/Perfetto JSON here (load in chrome://tracing or perfetto)"
    )
    ap.add_argument(
        "--swimlane",
        help="write a real-pid/tid L3/L4 host swimlane JSON here (load in chrome://tracing or perfetto)",
    )
    ap.add_argument(
        "--host-phase-records",
        action="append",
        metavar="PATH",
        help="a host_phase_records.jsonl from the same run; its per-event orchestrator operations are "
        "drawn inside the matching chip.run.bind, along with any bind segment the log does not "
        "already carry as a span. Repeatable. Only affects --swimlane, because the summed host-orch "
        "timing lines are cost shares and cannot be placed on a timeline",
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
    ap.add_argument(
        "--assert-native-overlap",
        action="store_true",
        help="fail unless adjacent dispatches prove prepare(N+1) overlaps device(N) and preserve ordered launch",
    )
    ap.add_argument(
        "--require-hidden-prepare",
        action="store_true",
        help="with --assert-native-overlap, also require prepare(N+1) to finish inside device(N) "
        "(fully hidden, not merely overlapping — sensitive to host scheduling)",
    )
    args = ap.parse_args(argv)

    lines = []
    for source in args.log:
        if source == "-":
            lines.extend(sys.stdin.readlines())
            continue
        for path in expand_log_source(source):
            with open(path, encoding="utf-8", errors="replace") as f:
                lines.extend(f.readlines())

    spans = list(parse_spans(lines))
    anchors = list(parse_clock_anchors(lines))
    anchor_counts = defaultdict(int)
    for anchor in anchors:
        anchor_counts[anchor.pid] += 1
    for pid, count in sorted(anchor_counts.items()):
        if count > 1:
            print(
                f"warning: multiple [CLOCK_ANCHOR] records found for pid {pid} ({count} records); using the last one",
                file=sys.stderr,
            )
    # Without an anchor a pid's records stay monotonic-only, and every renderer
    # degrades to relative time without saying so. A process writes its anchor
    # ahead of its first record, into whichever stream it is logging to, so a pid
    # with spans and no anchor means that stream reached us incomplete.
    unanchored = sorted({span.pid for span in spans} - set(anchor_counts))
    if unanchored:
        print(
            f"warning: no [CLOCK_ANCHOR] record for pid(s) {', '.join(str(pid) for pid in unanchored)} that emitted "
            "spans; their timestamps stay monotonic-only. Each process writes its anchor before its first record, so "
            "check that every input is complete and that no process's stream is missing.",
            file=sys.stderr,
        )
    warn_about_lost_records(lines, spans)
    keyed = invocation_spans(spans)
    invocations = group_invocations(keyed)
    buckets = bucket_by_hid(invocations)

    if args.assert_native_overlap:
        try:
            checks = assert_native_overlap(spans, require_hidden=args.require_hidden_prepare)
        except NativeOverlapError as exc:
            print(f"native overlap assertion failed: {exc}", file=sys.stderr)
            return 2
        print(f"Native overlap verified for {len(checks)} adjacent dispatch pair(s).")
    elif args.rounds_table:
        print_rounds_table(buckets)
    elif args.tree:
        print_tree(buckets)
    else:
        print_tpot_table(buckets)

    if args.trace_out:
        with open(args.trace_out, "w", encoding="utf-8") as f:
            json.dump(to_chrome_trace(invocations, buckets, anchors=anchors), f)
        print(f"Wrote Chrome trace: {args.trace_out} ({len(keyed)} spans)")

    if args.swimlane:
        write_host_swimlane(args, spans, anchors)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
