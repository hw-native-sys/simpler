#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
from collections import defaultdict
from io import StringIO

import pytest
from simpler.worker_level import WorkerLevel

from simpler_setup.tools.strace_timing import (
    _CHIP_WORD,
    _CORE_WORD,
    _NODE_WORDS,
    NativeOverlapError,
    assert_native_overlap,
    bucket_by_hid,
    count_record_heads,
    external_producer,
    group_invocations,
    host_record_spans,
    invocation_spans,
    load_host_phase_records,
    main,
    node_span_leaf,
    parse_clock_anchors,
    parse_drop_summaries,
    parse_spans,
    print_rounds_table,
    span_family,
    to_chrome_trace,
    to_host_swimlane,
)


def _metadata(trace, kind):
    """The `args.name` of every `kind` metadata event, in lane order."""
    return [event["args"]["name"] for event in trace["traceEvents"] if event["ph"] == "M" and event["name"] == kind]


def _lanes(trace):
    """Lane tid -> lane name."""
    return {
        event["tid"]: event["args"]["name"]
        for event in trace["traceEvents"]
        if event["ph"] == "M" and event["name"] == "thread_name"
    }


def _record(pid, inv, name, attrs="", *, depth=0, ts=100, dur=20):
    """One current host-log record in the shape `HostLogger::emit` writes it."""
    return (
        f"[mono_ns={1_000_000 + pid}][T0x{pid}][TIMING] emit_host_span: "
        f"[STRACE] v=1 pid={pid} tid={pid} inv={inv} hid=abc depth={depth} "
        f"name={name} ts={ts} dur={dur} {attrs}"
    )


def _anchor_record(pid, mono_ns, wall_ns):
    return (
        f"[mono_ns={mono_ns}][T0x{pid}][TIMING] clock_anchor: "
        f"[CLOCK_ANCHOR] v=1 pid={pid} mono_ns={mono_ns} wall_ns={wall_ns}\n"
    )


def _span_record(
    *,
    pid: int,
    tid: int,
    inv: int,
    name: str,
    ts: int,
    dur: int,
    attrs: str = "",
    hid: str = "abc",
    depth: int = 0,
) -> str:
    return f"[STRACE] v=1 pid={pid} tid={tid} inv={inv} hid={hid} depth={depth} name={name} ts={ts} dur={dur} {attrs}\n"


def test_parse_spans_finds_adjacent_records_on_one_physical_line():
    line = (
        _record(1, 1, "chip.run", "rank=0") + _record(2, 2, "chip.run.runner_run.device_wall", "clk=dev rank=1") + "\n"
    )

    spans = list(parse_spans([line]))

    assert [(span.pid, span.inv, span.name) for span in spans] == [
        (1, 1, "chip.run"),
        (2, 2, "chip.run.runner_run.device_wall"),
    ]
    assert spans[0].attrs == "rank=0"
    assert spans[1].attrs == "clk=dev rank=1"


def test_parse_spans_keeps_every_record_of_a_multi_line_blob():
    blob = _record(1, 1, "chip.run", "rank=0") + "\n" + _record(2, 2, "chip.run.bind", "rank=1") + "\n"

    spans = list(parse_spans([blob]))

    assert [(span.pid, span.inv, span.name) for span in spans] == [
        (1, 1, "chip.run"),
        (2, 2, "chip.run.bind"),
    ]
    assert spans[0].attrs == "rank=0"
    assert spans[1].attrs == "rank=1"


def test_parse_spans_splits_two_records_sharing_one_physical_line():
    """The log prefix has to bound a record, not just precede one.

    Ranks forked by an L3 share the capture fd, so two complete records land on
    one physical line often enough that pypto's reader re-splits on ``[STRACE]``
    before parsing. A record's attribute list runs to the end of the line, so
    what stops it is the lookahead for the *next* record's log prefix — remove
    that lookahead and the first record's ``attrs`` silently absorbs the
    second's whole prefix rather than failing to match.
    """
    blob = _record(1, 1, "chip.run", "rank=0") + _record(2, 2, "chip.run.bind", "rank=1")

    spans = list(parse_spans([blob]))

    assert [(span.pid, span.inv, span.name) for span in spans] == [
        (1, 1, "chip.run"),
        (2, 2, "chip.run.bind"),
    ]
    assert spans[0].attrs == "rank=0"
    assert spans[1].attrs == "rank=1"


def test_parse_spans_preserves_64_bit_invocation_id():
    invocation_id = 2**32 + 7

    spans = list(parse_spans([_record(41, invocation_id, "chip.run")]))

    assert len(spans) == 1
    assert spans[0].inv == invocation_id


def test_parse_clock_anchor_maps_monotonic_to_wall_time():
    lines = [
        "worker-3: [mono_ns=1005][T0x1][TIMING] clock_anchor: "
        "[CLOCK_ANCHOR] v=1 pid=41 mono_ns=1000 wall_ns=1700000000000000000\n"
    ]

    anchors = list(parse_clock_anchors(lines))

    assert len(anchors) == 1
    assert anchors[0].pid == 41
    assert anchors[0].mono_ns == 1000
    assert anchors[0].wall_ns == 1_700_000_000_000_000_000
    assert anchors[0].to_wall_ns(1250) == 1_700_000_000_000_000_250


def test_parse_clock_anchor_rejects_payloads_outside_complete_anchor_records():
    payload = "[CLOCK_ANCHOR] v=1 pid=41 mono_ns=1000 wall_ns=1700000000000000000"
    lines = [
        payload + "\n",
        f"[mono_ns=1005][T0x1][INFO] message: copied {payload}\n",
        f"[mono_ns=1005][T0x1][TIMING] clock_anchor: {payload} trailing\n",
    ]

    assert list(parse_clock_anchors(lines)) == []


def test_trace_renderers_add_wall_time_without_replacing_monotonic_timestamps():
    wall_ns = 1_700_000_000_000_000_000
    anchors = list(parse_clock_anchors([_anchor_record(41, 1_000, wall_ns)]))
    spans = list(
        parse_spans(
            [
                _span_record(pid=41, tid=410, inv=7, name="chip.run", ts=1_250, dur=20),
                _span_record(
                    pid=41,
                    tid=410,
                    inv=7,
                    name="chip.run.runner_run.device_wall",
                    ts=300,
                    dur=40,
                    attrs="clk=dev",
                    depth=1,
                ),
            ]
        )
    )

    invocations = group_invocations(spans)
    chrome_trace = to_chrome_trace(invocations, bucket_by_hid(invocations), anchors=anchors)
    host_swimlane = to_host_swimlane(spans, anchors=anchors)

    for trace in (chrome_trace, host_swimlane):
        host_event = next(event for event in trace["traceEvents"] if event.get("name") == "chip.run")
        assert host_event["ts"] == 1.25
        assert host_event["args"]["wall_ts_ns"] == "1700000000000000250"
        assert host_event["args"]["wall_time"] == "2023-11-14T22:13:20.000000250Z"
        assert trace["clockAnchors"] == [{"pid": 41, "mono_ns": "1000", "wall_ns": "1700000000000000000"}]

    device_event = next(event for event in chrome_trace["traceEvents"] if event.get("name", "").endswith("device_wall"))
    assert "wall_ts_ns" not in device_event["args"]
    assert "wall_time" not in device_event["args"]


def test_wall_time_uses_the_matching_anchor_for_each_pid():
    anchors = list(
        parse_clock_anchors(
            [
                _anchor_record(41, 1_000, 1_700_000_000_000_000_000),
                _anchor_record(52, 1_000, 1_900_000_000_000_000_000),
                _anchor_record(64, 2_000, 2_000_000_000_000_000_000),
            ]
        )
    )
    spans = list(
        parse_spans(
            [
                _span_record(pid=41, tid=410, inv=1, name="node.submit", ts=1_500, dur=10),
                _span_record(pid=52, tid=520, inv=1, name="node.submit", ts=1_500, dur=10),
                _span_record(pid=63, tid=630, inv=1, name="node.submit", ts=1_500, dur=10),
                _span_record(pid=64, tid=640, inv=1, name="node.submit", ts=1_500, dur=10),
            ]
        )
    )

    trace = to_host_swimlane(spans, anchors=anchors)
    events = [event for event in trace["traceEvents"] if event.get("ph") == "X"]

    assert [event["args"].get("wall_ts_ns") for event in events] == [
        "1700000000000000500",
        "1900000000000000500",
        None,
        None,
    ]


def test_count_record_heads_sees_a_torn_record_that_parse_spans_drops():
    intact = _record(1, 1, "chip.run", "rank=0")
    torn = intact[: intact.index(" ts=")]
    lines = [intact + "\n", torn + "\n"]

    assert count_record_heads(lines) == 2
    assert len(list(parse_spans(lines))) == 1


def _drop_summary(pid, new, total, queue_full=0, claim_exhausted=0, output_failed=0, not_admitted=0):
    return (
        f"[mono_ns=1000][T0xabc][ERROR] host_log_drops: [HOSTLOG_DROPS] v=1 pid={pid} new={new} "
        f"total={total} queue_full={queue_full} claim_exhausted={claim_exhausted} "
        f"output_failed={output_failed} not_admitted={not_admitted}\n"
    )


def test_parse_drop_summaries_keeps_the_running_total_per_process():
    # A process reports a growth at every quiescent boundary, so the reader must
    # keep the last (largest) figure rather than summing the reports.
    lines = [
        _drop_summary(71, new=2, total=2, queue_full=2),
        _drop_summary(72, new=1, total=1, output_failed=1),
        _drop_summary(71, new=3, total=5, queue_full=5),
    ]

    summaries = parse_drop_summaries(lines)
    assert summaries[71] == {
        "total": 5,
        "queue_full": 5,
        "claim_exhausted": 0,
        "output_failed": 0,
        "not_admitted": 0,
    }
    assert summaries[72]["total"] == 1
    assert summaries[72]["output_failed"] == 1


def test_parse_drop_summaries_ignores_an_unknown_grammar_version():
    lines = [_drop_summary(71, new=1, total=1).replace("v=1", "v=2")]

    assert parse_drop_summaries(lines) == {}


def test_main_warns_that_dropped_records_make_the_timing_incomplete(tmp_path, capsys):
    log_file = tmp_path / "host.71.log"
    log_file.write_text(_record(1, 1, "chip.run", "rank=0") + "\n" + _drop_summary(71, new=4, total=4, queue_full=4))

    main([str(log_file)])

    stderr = capsys.readouterr().err
    assert "pid 71 dropped 4 host-log record(s)" in stderr
    assert "queue_full=4" in stderr
    assert "incomplete log" in stderr


def test_host_swimlane_keeps_real_host_lanes_and_builds_dispatch_flow():
    lines = [
        _span_record(
            pid=41,
            tid=410,
            inv=7,
            name="node.graph_build",
            ts=1_000,
            dur=900,
            attrs="run_id=7 role=facade",
        ),
        _span_record(
            pid=41,
            tid=410,
            inv=7,
            name="node.submit",
            ts=1_100,
            dur=100,
            attrs="run_id=7 task_slot=12 group_index=0 worker_id=3 role=facade",
        ),
        _span_record(
            pid=41,
            tid=411,
            inv=7,
            name="node.dispatch",
            ts=1_400,
            dur=80,
            attrs=(
                "run_id=7 task_slot=12 group_index=0 worker_id=3 dispatch_id=99 "
                "endpoint_kind=local_mailbox role=scheduler"
            ),
        ),
    ]

    trace = to_host_swimlane(list(parse_spans(lines)))
    events = trace["traceEvents"]
    slices = [event for event in events if event["ph"] == "X"]
    flows = [event for event in events if event["ph"] in {"s", "f"}]

    assert {(event["pid"], event["tid"]) for event in slices} == {(41, 410), (41, 411)}
    assert [(event["ph"], event["pid"], event["tid"]) for event in flows] == [
        ("s", 41, 410),
        ("f", 41, 411),
    ]
    assert flows[0]["id"] == flows[1]["id"]
    assert isinstance(flows[0]["id"], int)
    assert slices[-1]["args"]["dispatch_id"] == 99
    thread_names = {
        (event["pid"], event["tid"]): event["args"]["name"]
        for event in events
        if event["ph"] == "M" and event["name"] == "thread_name"
    }
    assert thread_names == {(41, 410): "orchestrator / facade", (41, 411): "scheduler"}


def test_host_swimlane_names_the_scheduler_lane_from_every_role_it_emits():
    """The scheduler thread emits `role=worker` spans before its first `role=scheduler` one.

    `scheduler.cpp`'s loop is the only caller of both `dispatch_ready()` and
    `manager->progress()`, and inside `submit_dispatch` the `node.frame_submit`
    scope closes within `submit_progress` — before the `node.dispatch` record
    emitted after the admission lock is released. Naming the lane from the
    first span it emitted therefore labels the scheduler `worker 3`.
    """
    lines = [
        _span_record(
            pid=41,
            tid=411,
            inv=7,
            name="node.frame_submit",
            ts=1_410,
            dur=40,
            attrs="run_id=7 task_slot=12 group_index=0 worker_id=3 dispatch_id=1 role=worker",
        ),
        _span_record(
            pid=41,
            tid=411,
            inv=7,
            name="node.dispatch",
            ts=1_400,
            dur=80,
            attrs="run_id=7 task_slot=12 group_index=0 worker_id=3 dispatch_id=1 role=scheduler",
        ),
        _span_record(
            pid=41,
            tid=411,
            inv=7,
            name="node.complete",
            ts=2_000,
            dur=30,
            attrs="run_id=7 task_slot=12 group_index=0 worker_id=3 dispatch_id=1 role=worker",
        ),
    ]

    trace = to_host_swimlane(list(parse_spans(lines)))
    thread_names = {
        (event["pid"], event["tid"]): event["args"]["name"]
        for event in trace["traceEvents"]
        if event["ph"] == "M" and event["name"] == "thread_name"
    }

    assert thread_names == {(41, 411): "scheduler"}


def test_a_scheduler_loop_span_lands_on_the_scheduler_lane_and_no_invocation():
    """It is the one host span that belongs to no run, and both views must cope.

    A loop iteration drains completions and dispatches tasks belonging to
    whichever runs happened to be ready, so it carries no `run_id` and its `inv`
    is 0 rather than a run epoch. The swimlane still has to place it — `role`
    names the lane, which is why the attribute is there and not inferred from the
    name — and the invocation-keyed views still have to leave it out, or every
    one of them gains a forged invocation 0.
    """
    lines = [
        _span_record(
            pid=41,
            tid=411,
            inv=0,
            name="node.scheduler_loop",
            ts=1_400,
            dur=120,
            attrs="role=scheduler drained=2 dispatched=1 drain_ns=40 spins=917",
        )
    ]
    spans = list(parse_spans(lines))

    assert invocation_spans(spans) == []

    trace = to_host_swimlane(spans)
    thread_names = {
        (event["pid"], event["tid"]): event["args"]["name"]
        for event in trace["traceEvents"]
        if event["ph"] == "M" and event["name"] == "thread_name"
    }
    assert thread_names == {(41, 411): "scheduler"}

    slices = [event for event in trace["traceEvents"] if event["ph"] == "X"]
    assert [event["name"] for event in slices] == ["node.scheduler_loop"]
    assert slices[0]["args"]["spins"] == 917
    assert slices[0]["args"]["drained"] == 2


def test_parse_spans_decodes_percent_escaped_name_and_attribute_values():
    """`encode_host_span_field` escapes whatever would otherwise be record grammar."""
    lines = [
        _span_record(
            pid=41,
            tid=410,
            inv=7,
            name="host.odd%20name",
            ts=100,
            dur=10,
            attrs="run_id=7 reason=submit%20failed%3A%20%5Bfatal%5D role=facade",
        )
    ]

    span = next(iter(parse_spans(lines)))
    assert span.name == "host.odd name"

    trace = to_host_swimlane([span])
    args = next(event["args"] for event in trace["traceEvents"] if event["ph"] == "X")
    assert args["reason"] == "submit failed: [fatal]"
    # The raw field stays verbatim: it is the record as written.
    assert "%5Bfatal%5D" in args["attrs"]


def test_host_swimlane_pairs_dispatch_with_latest_preceding_submit():
    lines = [
        _span_record(
            pid=41,
            tid=410,
            inv=7,
            name="node.submit",
            ts=100,
            dur=20,
            attrs="run_id=7 task_slot=12 group_index=0 worker_id=3 role=facade",
        ),
        _span_record(
            pid=41,
            tid=411,
            inv=7,
            name="node.dispatch",
            ts=200,
            dur=10,
            attrs="run_id=7 task_slot=12 group_index=0 worker_id=3 dispatch_id=1 role=scheduler",
        ),
        _span_record(
            pid=41,
            tid=410,
            inv=7,
            name="node.submit",
            ts=300,
            dur=20,
            attrs="run_id=7 task_slot=12 group_index=0 worker_id=3 role=facade",
        ),
        _span_record(
            pid=41,
            tid=411,
            inv=7,
            name="node.dispatch",
            ts=400,
            dur=10,
            attrs="run_id=7 task_slot=12 group_index=0 worker_id=3 dispatch_id=2 role=scheduler",
        ),
    ]

    trace = to_host_swimlane(list(parse_spans(lines)))
    flows = [event for event in trace["traceEvents"] if event["ph"] in {"s", "f"}]

    assert [event["id"] for event in flows] == [1, 1, 2, 2]
    assert [event["ts"] for event in flows if event["ph"] == "s"] == [0.12, 0.32]
    assert [event["args"]["dispatch_key"] for event in flows] == [
        "dispatch:41:7:12:0:3:1",
        "dispatch:41:7:12:0:3:1",
        "dispatch:41:7:12:0:3:2",
        "dispatch:41:7:12:0:3:2",
    ]


def test_host_swimlane_keeps_unaligned_device_clock_out_of_visible_timeline():
    spans = list(
        parse_spans(
            [
                _span_record(
                    pid=41,
                    tid=410,
                    inv=7,
                    name="node.graph_build",
                    ts=1_000_000_000,
                    dur=900,
                    attrs="run_id=7 role=facade",
                ),
                _span_record(
                    pid=52,
                    tid=520,
                    inv=8,
                    name="chip.run.runner_run.device_wall",
                    ts=300,
                    dur=40,
                    attrs="clk=dev rank=1",
                ),
            ]
        )
    )

    trace = to_host_swimlane(spans)
    visible_slices = [event for event in trace["traceEvents"] if event.get("ph") == "X"]

    assert [(event["name"], event["ts"], event["dur"]) for event in visible_slices] == [
        ("node.graph_build", 1_000_000.0, 0.9)
    ]
    assert trace["unalignedDeviceSpans"] == [
        {
            "name": "chip.run.runner_run.device_wall",
            "ts_ns": 300,
            "dur_ns": 40,
            "pid": 52,
            "tid": 520,
            "inv": 8,
            "hid": "abc",
            "depth": 0,
            "attrs": {"raw": "clk=dev rank=1", "clk": "dev", "rank": 1},
        }
    ]


def test_invocation_views_ignore_host_swimlane_markers():
    """The current vocabulary: the host family stays out of the invocation views."""
    old = list(
        parse_spans(
            [
                _span_record(pid=61, tid=610, inv=3, name="chip.run", ts=1_000, dur=500),
                _span_record(pid=61, tid=610, inv=3, name="chip.run.bind", ts=1_100, dur=50, depth=1),
                _span_record(pid=61, tid=610, inv=4, name="chip.prewarm.build", ts=1_200, dur=75),
            ]
        )
    )
    mixed = old + list(
        parse_spans(
            [
                _span_record(
                    pid=61,
                    tid=611,
                    inv=9,
                    name="node.dispatch",
                    ts=900,
                    dur=20,
                    attrs="run_id=9 task_slot=4 group_index=0 worker_id=0 dispatch_id=1",
                )
            ]
        )
    )

    old_invocations = group_invocations(invocation_spans(old))
    mixed_invocations = group_invocations(invocation_spans(mixed))

    assert [span.name for span in invocation_spans(mixed)] == [
        "chip.run",
        "chip.run.bind",
        "chip.prewarm.build",
    ]
    assert to_chrome_trace(old_invocations, bucket_by_hid(old_invocations)) == to_chrome_trace(
        mixed_invocations, bucket_by_hid(mixed_invocations)
    )


def test_span_family_classifies_the_level_words_and_reserves_ext():
    assert span_family("chip.run") == "chip"
    assert span_family("chip.run.runner_run.device_wall") == "chip"
    assert span_family("core.pipe") == "core"
    # Every level at or above L3 runs the same orchestrator and scheduler code,
    # so they answer as one family whichever word the process resolved to.
    for level in WorkerLevel:
        if level.value >= WorkerLevel.node.value:
            assert span_family(f"{level.name}.dispatch") == "node"
    # A caller cannot land in ours, which is what `ext.` is reserved for.
    assert span_family("ext.pypto.decode_layer") == "external"
    assert span_family("something_else.foo") == "unknown"
    # The leaf is what the swimlane's flow pairing matches on, so it resolves for
    # every host level word and for nothing else.
    assert node_span_leaf("node.dispatch") == "dispatch"
    assert node_span_leaf("network1.submit") == "submit"
    assert node_span_leaf("chip.run") is None


def test_every_level_word_the_ladder_names_is_a_word_this_parser_knows():
    """`WorkerLevel` is the source of truth; the parser carries a second copy.

    The parser cannot import the runtime package, so the two lists are written
    twice and only a test can hold them together. Nothing else would: a level
    added to the ladder makes the runtime emit a word the parser does not know,
    `span_family` answers ``unknown``, and — because unknown is deliberately
    kept rather than dropped — those per-task spans enter invocation grouping
    under a forged ``(pid, 0)`` key. No error anywhere, just wrong tables.
    """
    ladder = {level.name for level in WorkerLevel}
    parser = set(_NODE_WORDS) | {_CHIP_WORD, _CORE_WORD}

    assert ladder == parser, (
        f"the ladder and this parser disagree: ladder-only={sorted(ladder - parser)}, "
        f"parser-only={sorted(parser - ladder)}"
    )
    # Each word also has to reach the family the ladder position implies, not
    # merely be present in some list.
    for level in WorkerLevel:
        expected = "node" if level.value >= WorkerLevel.node.value else level.name
        assert span_family(f"{level.name}.something") == expected
        # A level word is never mistaken for the external namespace.
        assert external_producer(f"{level.name}.something") is None


def test_invocation_by_name_uses_earliest_timestamp_not_input_order():
    spans = list(
        parse_spans(
            [
                _span_record(pid=61, tid=610, inv=3, name="chip.run.bind", ts=1_200, dur=20, depth=1),
                _span_record(pid=61, tid=611, inv=3, name="chip.run.bind", ts=1_100, dur=30, depth=1),
            ]
        )
    )

    invocation = group_invocations(spans)[0]

    assert invocation.by_name()["chip.run.bind"].ts == 1_100


def test_swimlane_cli_writes_trace(tmp_path):
    log_path = tmp_path / "run.log"
    output_path = tmp_path / "host_swimlane.json"
    log_path.write_text(
        _anchor_record(71, 50, 1_700_000_000_000_000_000)
        + _span_record(pid=71, tid=710, inv=2, name="node.graph_build", ts=100, dur=25, attrs="run_id=2 role=facade"),
        encoding="utf-8",
    )

    assert main([str(log_path), "--swimlane", str(output_path)]) == 0

    trace = json.loads(output_path.read_text(encoding="utf-8"))
    event = next(event for event in trace["traceEvents"] if event.get("name") == "node.graph_build")
    assert event["args"]["wall_ts_ns"] == "1700000000000000050"


def test_cli_reads_every_input_so_a_run_needs_no_manual_merge(tmp_path, capsys):
    """A run's per-process logs, passed together.

    Records carry their own pid, so concatenating inputs is all the merge that
    was ever needed — but the tool used to take one path, which left the merge to
    whoever ran it.
    """
    first = tmp_path / "host.71.log"
    second = tmp_path / "host.72.log"
    # Each process's own file carries its anchor ahead of its records.
    first.write_text(
        _anchor_record(71, 50, 1_700_000_000_000_000_000)
        + _span_record(pid=71, tid=710, inv=1, name="node.submit", ts=100, dur=10),
        encoding="utf-8",
    )
    second.write_text(
        _anchor_record(72, 60, 1_700_000_000_000_000_010)
        + _span_record(pid=72, tid=720, inv=1, name="chip.run", ts=200, dur=20),
        encoding="utf-8",
    )
    output_path = tmp_path / "swimlane.json"

    assert main([str(first), str(second), "--swimlane", str(output_path)]) == 0

    trace = json.loads(output_path.read_text(encoding="utf-8"))
    assert {event["pid"] for event in trace["traceEvents"] if event.get("ph") == "X"} == {71, 72}
    assert {int(anchor["pid"]) for anchor in trace["clockAnchors"]} == {71, 72}
    assert "no [CLOCK_ANCHOR] record" not in capsys.readouterr().err


def test_cli_expands_a_run_directory_to_its_per_process_log_files(tmp_path):
    """A run's output_prefix can be passed as-is."""
    prefix = tmp_path / "case_20260824"
    prefix.mkdir()
    (prefix / "host.71.log").write_text(
        _span_record(pid=71, tid=710, inv=1, name="node.submit", ts=100, dur=10), encoding="utf-8"
    )
    (prefix / "host.72.log").write_text(
        _span_record(pid=72, tid=720, inv=1, name="chip.run", ts=200, dur=20), encoding="utf-8"
    )
    # A sibling artifact must not be read as a log.
    (prefix / "pmu.csv").write_text("not,a,log\n", encoding="utf-8")
    output_path = tmp_path / "swimlane.json"

    assert main([str(prefix), "--swimlane", str(output_path)]) == 0

    trace = json.loads(output_path.read_text(encoding="utf-8"))
    assert {event["pid"] for event in trace["traceEvents"] if event.get("ph") == "X"} == {71, 72}


def test_cli_rejects_a_directory_with_no_span_files(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(SystemExit, match="holds no host\\."):
        main([str(empty)])


def test_cli_warns_when_a_pid_emitted_spans_without_a_clock_anchor(tmp_path, capsys):
    """An incomplete input leaves those pids monotonic-only.

    `clockAnchors` is then simply absent from the output rather than wrong, which
    is exactly the kind of loss nobody notices.
    """
    log_file = tmp_path / "host.71.log"
    log_file.write_text(
        _span_record(pid=71, tid=710, inv=1, name="node.submit", ts=100, dur=10)
        + _span_record(pid=72, tid=720, inv=1, name="chip.run", ts=200, dur=20),
        encoding="utf-8",
    )

    assert main([str(log_file)]) == 0

    err = capsys.readouterr().err
    assert "no [CLOCK_ANCHOR] record for pid(s) 71, 72" in err
    assert "check that every input is complete" in err


def test_cli_warns_when_one_pid_has_multiple_clock_anchors(tmp_path, capsys):
    log_path = tmp_path / "run.log"
    log_path.write_text(
        _anchor_record(71, 50, 1_700_000_000_000_000_000)
        + _anchor_record(71, 75, 1_700_000_000_000_000_025)
        + _span_record(pid=71, tid=710, inv=2, name="chip.run", ts=100, dur=25),
        encoding="utf-8",
    )

    assert main([str(log_path)]) == 0

    assert "warning: multiple [CLOCK_ANCHOR] records found for pid 71" in capsys.readouterr().err


def test_rounds_table_omits_tmr_only_columns_when_only_host_and_device_exist():
    lines = []
    for inv, host_dur, device_dur in ((1, 100_000, 20_000), (2, 120_000, 24_000)):
        lines.append(
            _record(1, inv, "chip.run", dur=host_dur)
            + _record(
                1,
                inv,
                "chip.run.runner_run.device_wall",
                "clk=dev",
                depth=2,
                dur=device_dur,
            )
            + "\n"
        )
    buckets = bucket_by_hid(group_invocations(parse_spans(lines)))
    output = StringIO()

    print_rounds_table(buckets, stream=output)

    rendered = output.getvalue()
    assert "Host (us)" in rendered
    assert "Device (us)" in rendered
    assert "Effective (us)" not in rendered
    assert "Orch (us)" not in rendered
    assert "Sched (us)" not in rendered
    assert "Avg Host: 110.0 us" in rendered
    assert "Avg Device: 22.0 us [2/2]" in rendered


def test_rounds_table_excludes_internal_prewarm_invocation():
    prewarm = (
        _record(1, 1, "chip.run.runner_run.device_wall", "clk=dev", depth=2, dur=300_000)
        + _record(1, 1, "chip.prewarm.run", depth=0, dur=400_000)
        + "\n"
    )
    official = []
    for inv, host_dur, device_dur in ((2, 100_000, 20_000), (3, 120_000, 24_000)):
        official.append(
            _record(1, inv, "chip.run", dur=host_dur)
            + _record(1, inv, "chip.run.runner_run.device_wall", "clk=dev", depth=2, dur=device_dur)
            + "\n"
        )
    buckets = bucket_by_hid(group_invocations(parse_spans([prewarm, *official])))
    output = StringIO()

    print_rounds_table(buckets, stream=output)

    rendered = output.getvalue()
    assert "Avg Host: 110.0 us" in rendered
    assert "Avg Device: 22.0 us [2/2]" in rendered
    assert "(2 rounds)" in rendered
    assert "300.0" not in rendered


def test_rounds_table_distinguishes_prewarm_only_from_no_markers():
    lines = [_record(1, 1, "chip.run.bind", depth=1) + _record(1, 1, "chip.prewarm.run", depth=0, dur=400_000) + "\n"]
    buckets = bucket_by_hid(group_invocations(parse_spans(lines)))
    output = StringIO()

    print_rounds_table(buckets, stream=output)

    assert output.getvalue() == "No official [STRACE] rounds found.\n"


def _run_records(*, run_epoch, prepare, device, release, run_id=0, dispatch_id=0, slot_id=0, pid=7, inv=None):
    """One phased native run's spans, as the `chip.run` tree carries them.

    `prepare` and `device` are (ts, dur). The root span carries the identity and
    spans the whole lifetime, so it must enclose the children.
    """
    invocation = run_epoch if inv is None else inv
    identity = f"run_id={run_id} dispatch_id={dispatch_id} slot_id={slot_id} generation=1 run_epoch={run_epoch}"
    root_end = release + 10
    return [
        _record(pid, invocation, "chip.run.bind", depth=1, ts=prepare[0], dur=prepare[1]),
        _record(pid, invocation, "chip.run.runner_run", depth=1, ts=device[0], dur=device[1]),
        _record(pid, invocation, "chip.run.claim_release", depth=1, ts=release, dur=5),
        _record(pid, invocation, "chip.run", identity, depth=0, ts=prepare[0], dur=root_end - prepare[0]),
    ]


def test_native_overlap_reads_identity_from_the_invocation_root():
    """The identity is on the root; the windows are on its children.

    `(pid, inv)` is what joins them — the same grouping the TPOT views use.
    """
    lines = _run_records(run_epoch=1, prepare=(50, 20), device=(100, 100), release=200)
    lines += _run_records(run_epoch=2, prepare=(150, 30), device=(240, 100), release=340)

    checks = assert_native_overlap(parse_spans(lines))

    assert len(checks) == 1
    assert (checks[0].predecessor.sequence, checks[0].successor.sequence) == (1, 2)


def test_native_overlap_accepts_a_prepare_that_outlasts_the_device_window():
    """Overlap means the intervals intersect, not that prepare is contained.

    A prepare that starts inside the predecessor's device window and finishes
    after it still ran concurrently with device work, which is the property being
    proved. Demanding containment turns a slow host into a failure whose message
    blames the wrong thing.
    """
    lines = _run_records(run_epoch=1, prepare=(50, 20), device=(100, 100), release=200)
    lines += _run_records(run_epoch=2, prepare=(190, 40), device=(240, 100), release=340)

    assert len(assert_native_overlap(parse_spans(lines))) == 1

    with pytest.raises(NativeOverlapError, match="not fully hidden"):
        assert_native_overlap(parse_spans(lines), require_hidden=True)


def test_native_overlap_rejects_a_prepare_after_the_device_window():
    lines = _run_records(run_epoch=1, prepare=(50, 20), device=(100, 100), release=200)
    lines += _run_records(run_epoch=2, prepare=(210, 20), device=(240, 100), release=340)

    with pytest.raises(NativeOverlapError, match="did not overlap"):
        assert_native_overlap(parse_spans(lines))


def test_native_overlap_rejects_device_work_before_the_predecessor_release():
    lines = _run_records(run_epoch=1, prepare=(50, 20), device=(100, 100), release=250)
    lines += _run_records(run_epoch=2, prepare=(150, 30), device=(240, 100), release=400)

    with pytest.raises(NativeOverlapError, match="reordered before the claim release"):
        assert_native_overlap(parse_spans(lines))


def test_native_overlap_orders_by_run_epoch_when_dispatch_id_is_absent():
    """`_submit_chip_run_direct` defaults run_id and dispatch_id to 0.

    `run_epoch` is the only identity those runs carry, so it is what orders them.
    """
    lines = _run_records(run_epoch=1, prepare=(50, 20), device=(100, 100), release=200)
    lines += _run_records(run_epoch=2, prepare=(150, 30), device=(240, 100), release=340)

    identities = [check.predecessor for check in assert_native_overlap(parse_spans(lines))]

    assert identities[0].dispatch_id == 0
    assert identities[0].run_epoch == 1
    assert identities[0].sequence == 1


def test_native_overlap_orders_by_dispatch_id_when_the_scheduler_allocates_one():
    lines = _run_records(run_epoch=1, dispatch_id=10, run_id=5, prepare=(50, 20), device=(100, 100), release=200)
    lines += _run_records(run_epoch=2, dispatch_id=11, run_id=6, prepare=(150, 30), device=(240, 100), release=340)

    checks = assert_native_overlap(parse_spans(lines))

    assert (checks[0].predecessor.sequence, checks[0].successor.sequence) == (10, 11)


def test_native_overlap_skips_an_invocation_that_is_not_a_phased_run():
    """A lexical `chip.run` carries no identity attrs, so it orders nothing."""
    lines = _run_records(run_epoch=1, prepare=(50, 20), device=(100, 100), release=200)
    lines += _run_records(run_epoch=2, prepare=(150, 30), device=(240, 100), release=340)
    lines += [_record(7, 99, "chip.run", "", depth=0, ts=10, dur=5)]

    assert len(assert_native_overlap(parse_spans(lines))) == 1


def test_native_overlap_needs_two_runs_on_one_lane():
    lines = _run_records(run_epoch=1, prepare=(50, 20), device=(100, 100), release=200)

    with pytest.raises(NativeOverlapError, match="at least two complete native runs"):
        assert_native_overlap(parse_spans(lines))


def test_claim_release_span_stays_inside_the_invocation_tree():
    """It is a `chip.run` child, so the existing views absorb it.

    A separate family would have to be filtered out of every view; a depth-1
    child of the root is what the tree already knows how to render — and it
    cannot displace the root the way a second depth-0 span would.
    """
    lines = _run_records(run_epoch=1, prepare=(50, 20), device=(100, 100), release=200)

    invocation = group_invocations(invocation_spans(list(parse_spans(lines))))[0]

    assert invocation.root().name == "chip.run"
    assert invocation.by_name()["chip.run.claim_release"].depth == 1
    # The root still reports the whole lifetime, not a sub-stage's duration.
    assert invocation.root().dur == 210 - 50


def test_swimlane_splits_an_interleaved_thread_into_one_lane_per_pipeline_slot():
    """A K-deep pipeline runs on one OS thread, so tid is not the unit of concurrency.

    Flattening overlapping runs onto one lane makes Perfetto nest run N+1's spans
    inside run N's root by timestamp containment — false nesting that hides the
    very overlap the lane is meant to show. The pipeline slot is what a run holds
    exclusively, so runs sharing one cannot overlap.
    """
    lines = _run_records(run_epoch=1, slot_id=0, prepare=(50, 20), device=(100, 100), release=200)
    lines += _run_records(run_epoch=2, slot_id=1, prepare=(150, 30), device=(240, 100), release=340)
    lines += _run_records(run_epoch=3, slot_id=0, prepare=(300, 30), device=(380, 100), release=480)

    trace = to_host_swimlane(list(parse_spans(lines)))
    lanes = {
        event["tid"]: event["args"]["name"]
        for event in trace["traceEvents"]
        if event["ph"] == "M" and event["name"] == "thread_name"
    }

    assert sorted(lanes.values()) == ["pipeline slot 0 (tid 7)", "pipeline slot 1 (tid 7)"]
    # Two runs shared slot 0, so that lane holds both — and neither overlaps.
    roots = defaultdict(list)
    for event in trace["traceEvents"]:
        if event["ph"] == "X" and event["args"]["depth"] == 0:
            roots[event["tid"]].append((event["ts"], event["ts"] + event["dur"]))
    assert sorted(len(windows) for windows in roots.values()) == [1, 2]
    for windows in roots.values():
        windows.sort()
        assert all(a[1] <= b[0] for a, b in zip(windows, windows[1:]))
    # The real thread stays recoverable from the slice.
    slices = [event for event in trace["traceEvents"] if event["ph"] == "X"]
    assert {event["args"]["os_tid"] for event in slices} == {7}


def test_swimlane_keeps_a_sequential_thread_on_its_own_tid():
    """Splitting a thread whose runs never overlapped would only fragment it.

    The L3 scheduler thread carries a pipeline slot but runs strictly
    sequentially, so it must keep its real tid.
    """
    lines = _run_records(run_epoch=1, slot_id=0, prepare=(50, 20), device=(100, 100), release=200)
    lines += _run_records(run_epoch=2, slot_id=1, prepare=(300, 20), device=(350, 100), release=460)

    trace = to_host_swimlane(list(parse_spans(lines)))
    lanes = {
        event["tid"]: event["args"]["name"]
        for event in trace["traceEvents"]
        if event["ph"] == "M" and event["name"] == "thread_name"
    }

    assert list(lanes) == [7]


def test_chrome_trace_cli_writes_wall_time(tmp_path):
    log_path = tmp_path / "run.log"
    output_path = tmp_path / "strace.json"
    log_path.write_text(
        _anchor_record(71, 50, 1_700_000_000_000_000_000)
        + _span_record(pid=71, tid=710, inv=2, name="chip.run", ts=100, dur=25),
        encoding="utf-8",
    )

    assert main([str(log_path), "--trace-out", str(output_path)]) == 0

    trace = json.loads(output_path.read_text(encoding="utf-8"))
    event = next(event for event in trace["traceEvents"] if event.get("name") == "chip.run")
    assert event["ts"] == 0.1
    assert event["args"]["wall_ts_ns"] == "1700000000000000050"


def test_host_phase_records_loader_keeps_only_well_formed_passes(tmp_path):
    """Every line a consumer would choke on is dropped by the loader.

    The artifact is appended to while a run is in flight, so a truncated tail is
    ordinary. A line that parses but is not an object is malformed in the same
    sense — the consumer indexes it as a mapping — so it is dropped here rather
    than left for each consumer to re-check.
    """
    artifact = tmp_path / "host_phase_records.jsonl"
    artifact.write_text(
        "\n".join(
            [
                json.dumps({"pid": 7, "inv": 1, "records": [{"phase": "args", "start_ns": 10, "end_ns": 20}]}),
                "null",
                "[1, 2, 3]",
                "42",
                '"a string"',
                json.dumps({"pid": 7, "inv": 2, "records": "not-a-list"}),
                '{"pid": 7, "inv": 3, "records": [',
                "",
                json.dumps({"pid": 7, "inv": 4, "records": []}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    passes = load_host_phase_records([str(artifact)])

    assert [one_pass["inv"] for one_pass in passes] == [1, 4]


def test_host_record_spans_nest_bind_segments_and_orchestrator_operations(tmp_path):
    """A record's kind decides its depth: a bind segment sits under the stage, an
    orchestrator operation one level further in, under the segment it ran inside."""
    spans = list(parse_spans([_span_record(pid=9, tid=9, inv=5, name="chip.run.bind", ts=1_000, dur=500)]))
    passes = [
        {
            "pid": 9,
            "inv": 5,
            "records": [
                {"phase": "args", "start_ns": 1_000, "end_ns": 1_100, "detail": 4096, "tid": 9},
                {"phase": "record_in_graph_task", "start_ns": 1_150, "end_ns": 1_230, "detail": 4, "tid": 42},
                {"phase": "graph_submit", "start_ns": 1_200, "end_ns": 1_250, "detail": 77, "tid": 9},
            ],
        }
    ]

    out, orphaned, skipped = host_record_spans(spans, passes)

    assert orphaned == 0
    assert skipped == 0
    by_name = {span.name: span for span in out}
    bind_depth = spans[0].depth
    assert by_name["chip.run.bind.args"].depth == bind_depth + 1
    assert by_name["chip.run.bind.host_orch.graph_submit"].depth == bind_depth + 2
    assert by_name["chip.run.bind.args"].ts == 1_000
    assert by_name["chip.run.bind.args"].dur == 100
    assert by_name["chip.run.bind.host_orch.record_in_graph_task"].tid == 42
    assert by_name["chip.run.bind.host_orch.graph_submit"].tid == 9

    trace = to_host_swimlane(spans + out)
    lane_names = {
        event["tid"]: event["args"]["name"]
        for event in trace["traceEvents"]
        if event["ph"] == "M" and event["name"] == "thread_name"
    }
    assert lane_names[9] == "graph submit main"
    assert lane_names[42] == "graph record worker"


def test_host_record_spans_put_the_pre_rename_record_phase_on_the_recorder_lane():
    """A log written before the phase was renamed still lands on the recorder lane.

    The runtime emitted `record_node` for what is now `record_in_graph_task`, and
    an unrecognised phase name is attributed to host_main rather than rejected —
    so dropping the old spelling would silently redraw every archived log's
    recorder work onto the main lane.
    """
    spans = list(parse_spans([_span_record(pid=9, tid=9, inv=5, name="chip.run.bind", ts=900, dur=500)]))
    passes = [
        {
            "pid": 9,
            "inv": 5,
            "records": [
                {"phase": "record_node", "start_ns": 1_000, "end_ns": 1_080, "detail": 4, "tid": 42},
            ],
        }
    ]

    out, orphaned, _ = host_record_spans(spans, passes)

    assert orphaned == 0
    by_name = {span.name: span for span in out}
    assert by_name["chip.run.bind.host_orch.record_node"].tid == 42
    trace = to_host_swimlane(spans + out)
    lane_names = {
        event["tid"]: event["args"]["name"]
        for event in trace["traceEvents"]
        if event["ph"] == "M" and event["name"] == "thread_name"
    }
    assert lane_names[42] == "graph record worker"


def test_host_record_spans_keep_legacy_recording_on_main_lane():
    spans = list(parse_spans([_span_record(pid=9, tid=9, inv=5, name="chip.run.bind", ts=900, dur=500)]))
    passes = [
        {
            "pid": 9,
            "inv": 5,
            "records": [
                {"phase": "record_in_graph_task", "start_ns": 1_000, "end_ns": 1_080, "detail": 4},
                {"phase": "build_definition", "start_ns": 1_100, "end_ns": 1_180, "detail": 4},
                {"phase": "graph_submit", "start_ns": 1_200, "end_ns": 1_250, "detail": 77},
            ],
        }
    ]

    out, dropped, _ = host_record_spans(spans, passes)

    assert dropped == 0
    assert {span.tid for span in out} == {9}
    trace = to_host_swimlane(spans + out)
    lane_names = {
        event["tid"]: event["args"]["name"]
        for event in trace["traceEvents"]
        if event["ph"] == "M" and event["name"] == "thread_name"
    }
    assert lane_names[9] == "graph submit main"


def test_host_record_spans_drop_passes_with_no_matching_bind(tmp_path):
    spans = list(parse_spans([_span_record(pid=9, tid=9, inv=5, name="chip.run.bind", ts=1_000, dur=500)]))
    passes = [{"pid": 9, "inv": 999, "records": [{"phase": "args", "start_ns": 1_000, "end_ns": 1_100}]}]

    out, orphaned, skipped = host_record_spans(spans, passes)

    assert out == []
    assert orphaned == 1
    assert skipped == 0


def test_a_bind_segment_the_log_already_carries_is_not_drawn_twice(tmp_path):
    """Both channels describe the same segment, and the log's copy is the one kept.

    The runtime emits each bind segment as a span, and that span carries the
    segment's own attributes — byte counts, fault and CPU counters — where the
    artifact record carries only `detail`. Drawing both would put two bars on one
    interval, and keeping the artifact's would lose the attributes.
    """
    spans = list(
        parse_spans(
            [
                _span_record(pid=9, tid=9, inv=5, name="chip.run.bind", ts=1_000, dur=500),
                _span_record(
                    pid=9, tid=9, inv=5, name="chip.run.bind.args", ts=1_000, dur=100, depth=2, attrs="bytes=4096"
                ),
            ]
        )
    )
    passes = [
        {
            "pid": 9,
            "inv": 5,
            "records": [
                {"phase": "args", "start_ns": 1_000, "end_ns": 1_100, "detail": 4096, "tid": 9},
                {"phase": "graph_submit", "start_ns": 1_200, "end_ns": 1_250, "detail": 77, "tid": 9},
            ],
        }
    ]

    out, orphaned, skipped = host_record_spans(spans, passes)

    assert orphaned == 0
    assert skipped == 1
    assert [span.name for span in out] == ["chip.run.bind.host_orch.graph_submit"]

    drawn = [span for span in spans + out if span.name == "chip.run.bind.args"]
    assert len(drawn) == 1
    assert "bytes=4096" in drawn[0].attrs


def test_a_bind_segment_only_the_artifact_has_is_still_drawn():
    """A run may collect records without emitting the segments' spans.

    The breakdown switch and the record pool are separate conditions, so a
    chip-swimlane capture arms the pool with the switch off. Then the artifact is
    the only source for the segments and must still be admitted.
    """
    spans = list(parse_spans([_span_record(pid=9, tid=9, inv=5, name="chip.run.bind", ts=1_000, dur=500)]))
    passes = [{"pid": 9, "inv": 5, "records": [{"phase": "args", "start_ns": 1_000, "end_ns": 1_100, "tid": 9}]}]

    out, orphaned, skipped = host_record_spans(spans, passes)

    assert (orphaned, skipped) == (0, 0)
    assert [span.name for span in out] == ["chip.run.bind.args"]


# ---------------------------------------------------------------------------
# The `ext.` reserved namespace — the contract an external producer codes to.
#
# Every level word belongs to simpler, so a producer outside this repo emits
# under `ext.<producer>.<span>`. The tests below are the executable form of that
# contract: what a well-formed name looks like, which views the spans reach, and
# what a producer provably cannot do to our views. A repository adapting to it
# (pypto, pypto-serving, a user script) can read these as the specification and
# mirror the assertions against its own emitter.
# ---------------------------------------------------------------------------


def test_ext_name_attributes_a_producer_and_refuses_a_malformed_one():
    """``ext.<producer>.<span>`` — three present, non-empty segments to attribute.

    Family and attribution answer different questions. A name under ``ext.`` is
    never ours whatever follows it, so a malformed one still classifies as
    external; it simply lands in an unattributed lane rather than another
    producer's. A present-but-empty segment counts as malformed either side of
    the producer: ``ext..foo`` names nobody, and ``ext.pypto.`` names a producer
    with no span of its own.
    """
    assert external_producer("ext.pypto.decode_layer") == "pypto"
    assert external_producer("ext.pypto.attention.qk") == "pypto"

    for malformed in ("ext.pypto", "ext.", "ext..decode_layer", "ext.pypto.", "ext.pypto..detail"):
        assert span_family(malformed) == "external"
        assert external_producer(malformed) is None

    # Not external at all, so no producer.
    assert external_producer("node.dispatch") is None
    assert external_producer("chip.run") is None


def test_ext_name_cannot_impersonate_one_of_our_level_words():
    """The reason the namespace exists: a caller's ``node.foo`` must not parse as ours.

    Every level word is reachable as a producer segment, and none of them
    promotes the span into our family. ``host`` is included precisely because it
    is *not* a level word — it names a processor, so a producer may legitimately
    be called that, and it must be treated as any other producer name.
    """
    for word in [level.name for level in WorkerLevel] + ["host"]:
        assert span_family(f"ext.{word}.foo") == "external"
        assert external_producer(f"ext.{word}.foo") == word
        assert node_span_leaf(f"ext.{word}.foo") is None


def test_ext_spans_stay_out_of_the_invocation_keyed_views():
    """A producer has no ``inv``, so admitting its spans would forge one lane for all of them.

    ``inv`` is our correlation key (a native run epoch) and the public surface
    does not expose it, so every external record carries 0. Grouping on
    ``(pid, inv)`` would collapse unrelated producer spans into a single
    invocation and contaminate the rounds and TPOT tables computed from it.
    """
    lines = [
        _span_record(pid=41, tid=410, inv=7, name="chip.run", ts=100, dur=500, attrs="run_id=7 slot_id=0"),
        _span_record(pid=41, tid=410, inv=0, name="ext.pypto.decode_layer", ts=120, dur=80),
        _span_record(pid=41, tid=410, inv=0, name="ext.pypto.attention", ts=140, dur=20),
    ]
    spans = list(parse_spans(lines))

    assert [span.name for span in invocation_spans(spans)] == ["chip.run"]
    invocations = group_invocations(invocation_spans(spans))
    assert [(inv.pid, inv.inv) for inv in invocations] == [(41, 7)]

    trace = to_chrome_trace(invocations, bucket_by_hid(invocations))
    assert "ext.pypto.decode_layer" not in {event.get("name") for event in trace["traceEvents"]}


def test_ext_spans_appear_on_the_swimlane_attributed_to_their_producer():
    """The swimlane is where an external span is visible, and the only such view.

    A process that emitted nothing but external spans is not ours, so its label
    carries no ``simpler`` prefix.
    """
    lines = [
        _span_record(pid=90, tid=900, inv=0, name="ext.pypto.decode_layer", ts=100, dur=80),
        _span_record(pid=90, tid=900, inv=0, name="ext.pypto.attention", ts=110, dur=20),
    ]

    trace = to_host_swimlane(list(parse_spans(lines)))

    assert {event["name"] for event in trace["traceEvents"] if event["ph"] == "X"} == {
        "ext.pypto.decode_layer",
        "ext.pypto.attention",
    }
    assert _metadata(trace, "process_name") == ["external producer pypto (pid=90)"]
    assert _metadata(trace, "thread_name") == ["ext pypto"]


def test_a_shared_process_stays_ours_when_a_producer_emits_into_it():
    """The common case: a producer calls the public API from inside our host process.

    Sharing therefore says nothing about whose process it is, so any span of ours
    keeps the process labelled as ours.
    """
    lines = [
        _span_record(pid=41, tid=410, inv=7, name="node.graph_build", ts=100, dur=200, attrs="run_id=7 role=facade"),
        _span_record(pid=41, tid=411, inv=0, name="ext.pypto.decode_layer", ts=120, dur=60),
    ]

    trace = to_host_swimlane(list(parse_spans(lines)))

    assert _metadata(trace, "process_name") == ["simpler node (pid=41)"]
    assert _lanes(trace) == {410: "orchestrator / facade", 411: "ext pypto"}


def test_a_device_clock_span_still_proves_the_process_is_ours():
    """A `clk=dev` span never reaches the visible timeline, but it is still ours.

    So it is evidence about whose process this is. Classifying from the visible
    spans alone would label a process that emitted one of our device spans and
    one external host span as the producer's.
    """
    lines = [
        _span_record(pid=41, tid=410, inv=0, name="ext.pypto.decode_layer", ts=120, dur=60),
        _span_record(
            pid=41, tid=410, inv=7, name="chip.run.runner_run.device_wall", ts=300, dur=40, attrs="clk=dev", depth=1
        ),
    ]

    trace = to_host_swimlane(list(parse_spans(lines)))

    assert _metadata(trace, "process_name") == ["simpler chip child (pid=41)"]
    # The device span stays off the visible timeline all the same.
    assert {event["name"] for event in trace["traceEvents"] if event["ph"] == "X"} == {"ext.pypto.decode_layer"}


def test_ext_attributes_cannot_take_over_one_of_our_lane_names():
    """``role`` is an ordinary attribute key, so a producer may use it for its own meaning.

    Lane naming infers our roles, so it must read only our spans — otherwise a
    producer writing ``role=facade`` renames its lane to one of ours.
    """
    lines = [
        _span_record(pid=90, tid=900, inv=0, name="ext.pypto.decode_layer", ts=100, dur=80, attrs="role=facade"),
        _span_record(pid=90, tid=901, inv=0, name="ext.pypto.step", ts=100, dur=80, attrs="role=scheduler worker_id=3"),
    ]

    trace = to_host_swimlane(list(parse_spans(lines)))

    assert _lanes(trace) == {900: "ext pypto", 901: "ext pypto"}


def test_ext_spans_cannot_reshape_our_lanes_or_our_dispatch_flow():
    """A pipeline slot is ours, and ``slot_id`` / ``depth`` / ``run_id`` are plain fields.

    An interleaved thread splits into one lane per slot. A producer emitting on
    that same thread must not decide whether the split happens, into how many
    lanes, or which spans a dispatch flow arrow connects.
    """
    lines = _run_records(run_epoch=1, slot_id=0, prepare=(50, 20), device=(100, 100), release=200)
    lines += _run_records(run_epoch=2, slot_id=1, prepare=(150, 30), device=(240, 100), release=340)
    lines += _run_records(run_epoch=3, slot_id=0, prepare=(300, 30), device=(380, 100), release=480)
    baseline = to_host_swimlane(list(parse_spans(lines)))

    # The same log, plus a producer on that thread claiming a slot of its own, a
    # root-looking depth, and the run/slot pair the flow pairing keys on.
    lines += [
        _span_record(
            pid=7,
            tid=7,
            inv=0,
            name="ext.pypto.decode_layer",
            ts=60,
            dur=400,
            attrs="run_id=0 task_slot=12 slot_id=99 role=facade",
        )
    ]
    mixed = to_host_swimlane(list(parse_spans(lines)))

    def flows(trace):
        return [event for event in trace["traceEvents"] if event["ph"] in ("s", "f")]

    assert _metadata(baseline, "thread_name") == ["pipeline slot 0 (tid 7)", "pipeline slot 1 (tid 7)"]
    assert _metadata(mixed, "thread_name") == ["ext pypto", *_metadata(baseline, "thread_name")]
    assert flows(mixed) == flows(baseline)
    # The producer's span is still on the timeline, on the thread's real tid.
    external = next(event for event in mixed["traceEvents"] if event.get("name") == "ext.pypto.decode_layer")
    assert external["tid"] == 7


def test_ext_spans_reach_the_swimlane_but_not_the_tables_through_the_cli(tmp_path, capsys):
    """End to end: one log carrying both, through the same CLI a downstream repo runs."""
    log_path = tmp_path / "run.log"
    swimlane_path = tmp_path / "host_swimlane.json"
    trace_path = tmp_path / "strace.json"
    log_path.write_text(
        _span_record(pid=41, tid=410, inv=7, name="chip.run", ts=100, dur=500, attrs="run_id=7 slot_id=0")
        + _span_record(pid=41, tid=411, inv=0, name="ext.pypto.decode_layer", ts=120, dur=80),
        encoding="utf-8",
    )

    assert main([str(log_path), "--swimlane", str(swimlane_path), "--trace-out", str(trace_path)]) == 0
    capsys.readouterr()

    swimlane = json.loads(swimlane_path.read_text(encoding="utf-8"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))

    assert "ext.pypto.decode_layer" in {event.get("name") for event in swimlane["traceEvents"]}
    assert "ext.pypto.decode_layer" not in {event.get("name") for event in trace["traceEvents"]}
