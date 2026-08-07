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

from simpler_setup.tools.strace_timing import (
    bucket_by_hid,
    count_record_heads,
    group_invocations,
    legacy_spans,
    main,
    parse_spans,
    to_chrome_trace,
    to_host_swimlane,
)


def _record(pid, inv, name, attrs=""):
    """One host-log record in the shape `HostLogger::emit` writes it.

    `LOG_TIMING` prepends `[<file>:<line>] ` to the caller's format string, so
    the marker never sits flush against the `<func>: ` separator on stderr.
    """
    return (
        f"[2026-08-04 10:00:00.00000{pid}][T0x{pid}][TIMING] emit_span: [strace.h:132] "
        f"[STRACE] v=1 pid={pid} tid={pid} inv={inv} hid=abc depth=0 name={name} ts=100 dur=20 {attrs}"
    )


def test_parse_spans_finds_adjacent_records_on_one_physical_line():
    line = (
        _record(1, 1, "simpler_run", "rank=0")
        + _record(2, 2, "simpler_run.runner_run.device_wall", "clk=dev rank=1")
        + "\n"
    )

    spans = list(parse_spans([line]))

    assert [(span.pid, span.inv, span.name) for span in spans] == [
        (1, 1, "simpler_run"),
        (2, 2, "simpler_run.runner_run.device_wall"),
    ]
    assert spans[0].attrs == "rank=0"
    assert spans[1].attrs == "clk=dev rank=1"


def test_parse_spans_keeps_every_record_of_a_multi_line_blob():
    blob = _record(1, 1, "simpler_run", "rank=0") + "\n" + _record(2, 2, "simpler_run.bind", "rank=1") + "\n"

    spans = list(parse_spans([blob]))

    assert [(span.pid, span.inv, span.name) for span in spans] == [
        (1, 1, "simpler_run"),
        (2, 2, "simpler_run.bind"),
    ]
    assert spans[0].attrs == "rank=0"
    assert spans[1].attrs == "rank=1"


def test_count_record_heads_sees_a_torn_record_that_parse_spans_drops():
    intact = _record(1, 1, "simpler_run", "rank=0")
    torn = intact[: intact.index(" ts=")]
    lines = [intact + "\n", torn + "\n"]

    assert count_record_heads(lines) == 2
    assert len(list(parse_spans(lines))) == 1


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


def test_host_swimlane_keeps_real_host_lanes_and_builds_dispatch_flow():
    lines = [
        _span_record(
            pid=41,
            tid=410,
            inv=7,
            name="l3.graph_build",
            ts=1_000,
            dur=900,
            attrs="run_id=7 role=facade",
        ),
        _span_record(
            pid=41,
            tid=410,
            inv=7,
            name="l3.submit",
            ts=1_100,
            dur=100,
            attrs="run_id=7 task_slot=12 group_index=0 worker_id=3 role=facade",
        ),
        _span_record(
            pid=41,
            tid=411,
            inv=7,
            name="l3.dispatch",
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
    assert slices[-1]["args"]["dispatch_id"] == 99
    thread_names = {
        (event["pid"], event["tid"]): event["args"]["name"]
        for event in events
        if event["ph"] == "M" and event["name"] == "thread_name"
    }
    assert thread_names == {(41, 410): "orchestrator / facade", (41, 411): "scheduler"}


def test_host_swimlane_keeps_unaligned_device_clock_out_of_visible_timeline():
    spans = list(
        parse_spans(
            [
                _span_record(
                    pid=41,
                    tid=410,
                    inv=7,
                    name="l3.graph_build",
                    ts=1_000_000_000,
                    dur=900,
                    attrs="run_id=7 role=facade",
                ),
                _span_record(
                    pid=52,
                    tid=520,
                    inv=8,
                    name="simpler_run.runner_run.device_wall",
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
        ("l3.graph_build", 1_000_000.0, 0.9)
    ]
    assert trace["unalignedDeviceSpans"] == [
        {
            "name": "simpler_run.runner_run.device_wall",
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


def test_legacy_trace_output_ignores_host_swimlane_markers():
    old = list(
        parse_spans(
            [
                _span_record(pid=61, tid=610, inv=3, name="simpler_run", ts=1_000, dur=500),
                _span_record(pid=61, tid=610, inv=3, name="simpler_run.bind", ts=1_100, dur=50, depth=1),
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
                    name="l3.dispatch",
                    ts=900,
                    dur=20,
                    attrs="run_id=9 task_slot=4 group_index=0 worker_id=0 dispatch_id=1",
                )
            ]
        )
    )

    old_invocations = group_invocations(legacy_spans(old))
    mixed_invocations = group_invocations(legacy_spans(mixed))

    assert to_chrome_trace(old_invocations, bucket_by_hid(old_invocations)) == to_chrome_trace(
        mixed_invocations, bucket_by_hid(mixed_invocations)
    )


def test_swimlane_cli_writes_trace(tmp_path):
    log_path = tmp_path / "run.log"
    output_path = tmp_path / "host_swimlane.json"
    log_path.write_text(
        _span_record(
            pid=71,
            tid=710,
            inv=2,
            name="l3.graph_build",
            ts=100,
            dur=25,
            attrs="run_id=2 role=facade",
        ),
        encoding="utf-8",
    )

    assert main([str(log_path), "--swimlane", str(output_path)]) == 0

    trace = json.loads(output_path.read_text(encoding="utf-8"))
    assert any(event.get("name") == "l3.graph_build" for event in trace["traceEvents"])
