# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Percounter nested execution intervals retain exact accounting and runtime identity."""

import json

import pytest
from _capture_builder import Capture

from simpler_setup.tools.percounter import scheduler_view
from simpler_setup.tools.swimlane_converter import generate_chrome_trace_json, read_perf_data


@pytest.mark.parametrize(("level", "memory_event"), [(3, None), (4, "bypass_load"), (4, "ssbuf_store"), (4, "dsb_all")])
def test_nested_kernel_in_step_exports_without_double_counting(tmp_path, level, memory_event):
    capture = Capture(runtime="percounter", level=level, scheduler_producer="aicore")
    capture.task(task_id=7, core_id=0, start=40, end=60, run_epoch=0)
    phases = [
        ("scalar_window", 10, 100, 0),
        ("scalar_percounter_step", 20, 80, 1),
        ("scalar_percounter_run_task", 30, 70, 2),
    ]
    if memory_event:
        phases.append((f"scalar_{memory_event}", 32, 35, 3))
    for kind, begin, end, depth in phases:
        metrics = {"scalar_depth": depth, "scalar_calls": 1}
        if depth == 0:
            metrics.update(scalar_schema=2, function_min_cycles=1 if level == 3 else 0)
        capture.aicore_sched_phase(phase=kind, start=begin, end=end, metrics=metrics)
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(capture.build()))
    data = read_perf_data(raw)
    profile = data["scalar_profiles"][0]
    assert profile["complete"]
    assert profile["kernel_cycles"] == 20
    assert profile["exclusive_cycles"]["scalar_percounter_run_task"] == (17 if memory_event else 20)
    assert profile["exclusive_cycles"]["scalar_percounter_step"] == 20
    assert sum(profile["exclusive_cycles"].values()) == 90
    trace = tmp_path / "trace.json"
    generate_chrome_trace_json(
        data["tasks"],
        str(trace),
        runtime_name="percounter",
        scheduler_phases=data["scheduler_records"],
        scheduler_streams=data["scheduler_streams"],
        scalar_profiles=data["scalar_profiles"],
        scheduler_task_producer="aicore",
        chip_swimlane_level=level,
    )
    output = json.loads(trace.read_text())
    assert output["metadata"]["runtime"] == "percounter"
    assert output["metadata"]["scalar_function_capture"]["memory_operations"] == (
        "per_call" if level == 4 else "disabled"
    )
    names = {e["name"] for e in output["traceEvents"] if e["ph"] == "X" and e["pid"] == 2}
    assert {"percounter_step", "percounter_run_task"} <= names
    if memory_event:
        assert memory_event in names
    assert not any(name.startswith("scheduler_") for name in names)


@pytest.mark.parametrize("level", [2, 3, 4])
def test_calendar_scheduler_tracks_preserve_role_and_defer_actor(tmp_path, level):
    capture = Capture(
        runtime="percounter", level=level, scheduler_producer="aicore", core_types=["aic", "aic", "aiv", "aiv"]
    )
    capture.task(task_id=7, core_id=0, start=40, end=55, run_epoch=0)
    capture.task(task_id=7, core_id=2, start=44, end=62, run_epoch=0)
    raw = capture.build()
    raw["percounter_task_events"] = {
        "dropped": 0,
        "unmatched_deferred_updates": 0,
        "folded_task_ids": [],
        "inline_completed_task_ids": [],
        "range_records": [
            {
                "kind": kind,
                "cycles": cycles,
                "run_epoch": 0,
                "worker_id": 0,
                "cluster_index": 0,
                "cluster_lane": 0,
                "range_index": 2,
                "graph_shell_id": 5,
                **extra,
            }
            for kind, cycles, extra in [
                ("range_enter_begin", 6, {}),
                ("graph_ready_observed", 13, {"watermark": 16}),
                ("range_entered", 15, {}),
            ]
        ],
        "records": [
            {
                "task_id": 7,
                "block_index": 0,
                "block_count": 1,
                "kind": kind,
                "cycles": cycles,
                "run_epoch": 0,
                "worker_id": worker,
                "cluster_index": cluster,
                "cluster_lane": lane,
                **extra,
            }
            for kind, cycles, worker, cluster, lane, extra in [
                ("row_begin", 16, 0, 0, 0, {}),
                ("row_begin", 20, 2, 0, 1, {}),
                ("inputs_ready", 26, 0, 0, 0, {}),
                ("ready", 30, 0, 0, 0, {}),
                ("ready", 35, 2, 0, 1, {}),
                ("kernel_end", 55, 0, 0, 0, dict(dispatch_cycles=32, kernel_start_cycles=40, finish_cycles=58)),
                ("kernel_end", 62, 2, 0, 1, dict(dispatch_cycles=38, kernel_start_cycles=44, finish_cycles=64)),
                ("defer_registered", 61, 0, 0, 0, dict(deferred_group=4)),
                ("local_retired", 64, 0, 0, 0, {}),
                ("completion_issue", 65, 2, 0, 1, dict(contribution=1)),
                ("local_retired", 66, 2, 0, 1, {}),
                ("defer_poll_begin", 70, 1, 1, 0, dict(deferred_group=4, source_worker_id=0)),
                ("defer_signal_ready", 85, 1, 1, 0, dict(deferred_group=4, source_worker_id=0)),
                ("deferred_completion_issue", 90, 1, 1, 0, dict(deferred_group=4, source_worker_id=0, contribution=1)),
            ]
        ],
    }
    raw["aicpu_lifecycle_records"] = [
        {
            "aicpu_thread_id": 1,
            "graph_shell_id": 5,
            "materialize_start_cycles": 10,
            "materialize_end_cycles": 20,
        },
        {
            "aicpu_thread_id": 1,
            "graph_shell_id": 5,
            "graph_watermark_published_cycles": 12,
            "watermark": 16,
        },
    ]
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(raw))
    data = read_perf_data(path)
    trace = generate_chrome_trace_json(
        data["tasks"],
        None,
        runtime_name="percounter",
        scheduler_task_producer="aicore",
        chip_swimlane_level=level,
        percounter_task_events=data["percounter_task_events"],
        aicpu_lifecycle_records=data["aicpu_lifecycle_records"],
    )
    events = trace["traceEvents"]
    scheduler = [e for e in events if e["pid"] == 3]
    workers = [e for e in events if e["pid"] == 4 and e["ph"] == "X"]
    assert len(workers) == 2
    assert all(e["args"]["block_idx"] == 0 and e["args"]["block_num"] == 1 for e in workers)
    assert workers[0]["dur"] == pytest.approx(15e6 / raw["metadata"]["clock_freq_hz"])
    assert sum(e["ph"] == "X" and "Output fence" in e["name"] for e in scheduler) == 2
    assert all("Kernel execution" not in e["name"] for e in scheduler)
    ready = [e for e in scheduler if e["ph"] == "i" and e["args"]["kind"] == "ready"]
    assert len({e["tid"] for e in ready}) == 2
    register = next(e for e in scheduler if e["ph"] == "i" and e["args"]["kind"] == "defer_registered")
    done = next(e for e in scheduler if e["ph"] == "i" and e["args"]["kind"] == "deferred_completion_issue")
    assert register["tid"] != done["tid"]
    assert done["args"]["source_worker_id"] == register["args"]["worker_id"]
    assert any(e.get("pid") == 6 and e["ph"] == "X" and e["args"]["graph_shell_id"] == 5 for e in events)
    assert any(e.get("pid") == 6 and e["name"] == "Graph watermark published" for e in events)
    published = next(e for e in events if e.get("pid") == 6 and e["name"] == "Graph watermark published")
    observed = next(
        e for e in scheduler if e.get("cat") == "percounter_range" and e["args"]["kind"] == "graph_ready_observed"
    )
    assert observed["ts"] > published["ts"]
    assert observed["args"]["watermark"] == 16


@pytest.mark.parametrize("mte3", [False, True])
def test_calendar_peer_dmb_markers_keep_target_token_and_observed_count(mte3):
    identity = {"run_epoch": 1, "block_index": 0, "block_count": 1, "cluster_index": 0}
    sender = {**identity, "kind": "peer_dmb_published", "task_id": 7, "worker_id": 0, "cluster_lane": 0, "cycles": 10}
    receiver = {**identity, "kind": "peer_dmb_observed", "task_id": 8, "worker_id": 1, "cluster_lane": 1, "cycles": 11}
    section = {
        "records": [
            {**sender, "kind": "peer_dmb_write_begin", "target_worker": 1, "token": 2, "cycles": 8},
            {**sender, "target_worker": 1, "token": 2, **({"transport": "MTE3"} if mte3 else {})},
            {**receiver, "token": 2, "ready_inputs": 1},
            {**receiver, "token": 3, "ready_inputs": 2, "cycles": 12},
            {**receiver, "kind": "peer_dmb_wait_begin", "input_count": 2, "cycles": 6},
        ],
        "range_records": [],
        "dropped": 0,
        "unmatched_deferred_updates": 0,
        "folded_task_ids": [],
        "inline_completed_task_ids": [],
    }
    capture = scheduler_view.decode_capture(scheduler_view.parse_capture(section), float)
    events = list(scheduler_view.trace_events(capture, 3, str))
    ticks = [e for e in events if e["ph"] == "i"]
    assert [e["args"]["token"] for e in ticks] == [2, 2, 3]
    assert ticks[0]["args"]["target_worker"] == 1
    assert [e["args"]["ready_inputs"] for e in ticks[1:]] == [1, 2]
    assert ticks[0]["tid"] != ticks[1]["tid"] == ticks[2]["tid"]
    write, wait = (e for e in events if e["cat"] == "percounter_dmb")
    assert (write["tid"], write["ts"], write["dur"]) == (ticks[0]["tid"], 8, 2)
    assert (wait["tid"], wait["ts"], wait["dur"]) == (ticks[1]["tid"], 6, 6)
    assert write["name"].endswith("Issue MTE3 DMB write → 8" if mte3 else "Write DMB → 8")
    assert write["args"]["transport"] == ("MTE3" if mte3 else "Scalar")
    assert wait["args"]["observed_tokens"] == wait["args"]["input_count"] == 2
    partial = {**capture, "records": [r for r in capture["records"] if r.get("ready_inputs") != 2]}
    assert not any("Wait on local DMB" in e["name"] for e in scheduler_view.trace_events(partial, 3, str))
    with pytest.raises(ValueError, match="completion count"):
        scheduler_view.parse_capture({**section, "records": [{**receiver, "token": 2}]})
    with pytest.raises(ValueError, match="input count"):
        scheduler_view.parse_capture(
            {**section, "records": [{**receiver, "kind": "peer_dmb_wait_begin", "input_count": 0}]}
        )


def test_dmb_register_observation_precedes_completion_check_on_receiver_lane():
    identity = dict(run_epoch=1, task_id=8, worker_id=1, block_index=0, block_count=1, cluster_index=0, cluster_lane=1)
    records = [
        dict(identity, kind="peer_dmb_mov_at_entry", cycles=12, token=7, sequence=0),
        dict(identity, kind="peer_dmb_mov_changed", cycles=25, token=10, sequence=13, previous_poll_cycles=22),
        dict(identity, kind="peer_dmb_observed", cycles=38, token=10, ready_inputs=2),
    ]
    section = dict(
        records=records,
        range_records=[],
        dropped=0,
        unmatched_deferred_updates=0,
        folded_task_ids=[],
        inline_completed_task_ids=[],
    )
    capture = scheduler_view.decode_capture(scheduler_view.parse_capture(section), float)
    marks = [e for e in scheduler_view.trace_events(capture, 3, str) if e["ph"] == "i"]
    assert [e["args"]["kind"] for e in marks] == [r["kind"] for r in records]
    assert [e["ts"] for e in marks] == [12, 25, 38]
    assert len({e["tid"] for e in marks}) == 1
    assert marks[1]["args"]["previous_poll_cycles"] == 22
    assert marks[1]["args"]["sequence"] == 13
    records[0]["previous_poll_cycles"] = 11
    with pytest.raises(ValueError, match="previous sample time"):
        scheduler_view.parse_capture(section)


def test_periodic_count_releases_wait_without_a_new_dmb_token():
    row = dict(run_epoch=1, task_id=3, worker_id=0, block_index=0, block_count=1, cluster_index=0, cluster_lane=0)
    records = [
        dict(row, kind="peer_dmb_wait_begin", cycles=1, input_count=2),
        dict(row, kind="peer_dmb_initial_proof", cycles=12, token=9, ready_inputs=1),
        dict(row, kind="peer_dmb_periodic_ready", cycles=336, cycles_since_last_check=310, ready_inputs=2),
    ]
    section = dict(
        records=records,
        range_records=[],
        dropped=0,
        unmatched_deferred_updates=0,
        folded_task_ids=[],
        inline_completed_task_ids=[],
    )
    capture = scheduler_view.decode_capture(scheduler_view.parse_capture(section), float)
    events = list(scheduler_view.trace_events(capture, 3, str))
    (wait,) = [e for e in events if e["cat"] == "percounter_dmb" and e["ph"] == "X"]
    assert (wait["ts"], wait["dur"]) == (1, 335)
    assert wait["args"]["observed_tokens"] == 0
    assert wait["args"]["input_count"] == wait["args"]["ready_inputs"] == 2
    assert wait["args"]["cycles_since_last_check"] == 310
    assert "token" not in wait["args"]
    assert any(e["ph"] == "i" and e["args"].get("kind") == "peer_dmb_periodic_ready" for e in events)


@pytest.mark.parametrize("transport", ["Scalar", "MTE3"])
def test_dmb_batch_shows_one_first_to_last_issue_span(transport):
    identity = dict(
        run_epoch=1,
        task_id=1,
        worker_id=32,
        block_index=0,
        block_count=32,
        cluster_index=0,
        cluster_lane=1,
        target_count=64,
        sequence=7,
    )
    begin = dict(identity, kind="peer_dmb_batch_begin", cycles=12)
    end = dict(identity, kind="peer_dmb_batch_issued", transport=transport, cycles=25)
    if transport == "MTE3":
        end["command_count"] = 9
    section = dict(
        records=[begin, end],
        range_records=[],
        dropped=0,
        unmatched_deferred_updates=0,
        folded_task_ids=[],
        inline_completed_task_ids=[],
    )
    capture = scheduler_view.decode_capture(scheduler_view.parse_capture(section), float)
    (span,) = [e for e in scheduler_view.trace_events(capture, 3, str) if e["cat"] == "percounter_dmb"]
    assert (span["ts"], span["dur"], span["args"]["transport"], span["args"]["target_count"]) == (12, 13, transport, 64)
    if transport == "MTE3":
        assert span["args"]["command_count"] == 9
