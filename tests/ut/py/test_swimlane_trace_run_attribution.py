# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run attribution in the rendered trace, not just in the parsed rows.

Two runs of one graph reuse every per-run token: the same ``task_id`` on the
same ``core_id``, and ``reg_task_id`` restarting at 0. Timestamps differ, so a
test that only counts arrows or checks that none spans the two runs passes
while every arrow still points at the wrong slice. These cases therefore follow
``bind_id`` to the event it names and compare *that* event's run, and read the
epoch back off the final trace.

CPU-only: they call the converter directly on in-memory rows. Export → parse is
covered separately in ``test_swimlane_export_run_identity.py``, which drives the
real C++ exporter.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from simpler_setup.tools._runtime_dispatch import TMR_RUNTIME
from simpler_setup.tools.swimlane_converter import generate_chrome_trace_json

RUN_A = 7
RUN_B = 8
# The two runs sit far apart so a mixed endpoint is unambiguous in the output.
BASE_A = 10.0
BASE_B = 100.0


def _task(epoch, task_id, core_id, start):
    return {
        "task_id": task_id,
        "func_id": 0,
        "core_id": core_id,
        "core_type": "aiv",
        "start_time_us": start,
        "end_time_us": start + 1.0,
        "duration_us": 1.0,
        "dispatch_time_us": start - 0.5,
        "finish_time_us": start + 1.2,
        "receive_time_us": start - 0.2,
        "local_setup_us": 0.2,
        "run_epoch": epoch,
    }


def _phase(kind, epoch, start, end, *, task_id=None, processed=0):
    record = {
        "phase": kind,
        "start_time_us": start,
        "end_time_us": end,
        "run_epoch": epoch,
        "loop_iter": 0,
        "tasks_processed": processed,
    }
    if task_id is not None:
        record["task_id"] = task_id
    return record


def _render(tmp_path, tasks, **kwargs):
    out = Path(tmp_path) / "trace.json"
    # The trace states the TaskId layout its labels follow. These tests are about run
    # attribution -- which epoch a record belongs to -- which no runtime decides, so
    # they take tmr and assert nothing that depends on it: flipping this to
    # HBG_RUNTIME leaves the file green.
    kwargs.setdefault("runtime_name", TMR_RUNTIME)
    generate_chrome_trace_json(tasks, str(out), **kwargs)
    return json.loads(out.read_text())["traceEvents"]


def _epoch_by_event_id(events):
    """Map every X event id to the run domain it belongs to."""
    by_id = {}
    for event in events:
        if event.get("ph") == "X" and event.get("id") is not None:
            by_id[event["id"]] = (event.get("args") or {}).get("run_epoch")
    return by_id


def _flow_pairs(events, name=("dependency", "hb_violation")):
    grouped = defaultdict(dict)
    for event in events:
        if event.get("cat") == "flow" and event.get("name") in name:
            grouped[event["id"]][event["ph"]] = event
    return [(p["s"], p["f"]) for p in grouped.values() if "s" in p and "f" in p]


def _assert_flow_binds_within_one_run(events, *, expected_runs, label, per_view_per_run=1):
    """Every arrow's two ends must name X events belonging to the same run.

    Checking timestamps alone cannot see this: endpoints can carry correct times
    while ``bind_id`` names the other run's slice, because the event index is
    keyed by identity rather than by time. One logical edge renders once in each
    view that has data, so counts are asserted per (view, run) rather than in
    total.
    """
    epoch_of = _epoch_by_event_id(events)
    pairs = _flow_pairs(events)
    assert pairs, f"{label}: no dependency arrows were emitted, so this proves nothing"

    per_view_run = defaultdict(int)
    for start_event, finish_event in pairs:
        src_bind = start_event.get("bind_id")
        dst_bind = finish_event.get("bind_id")
        assert src_bind is not None and dst_bind is not None, (
            f"{label}: arrow {start_event.get('id')} has an unbound endpoint ({src_bind}, {dst_bind})"
        )
        assert src_bind in epoch_of and dst_bind in epoch_of, (
            f"{label}: bind_id ({src_bind}, {dst_bind}) does not resolve to an emitted X event"
        )
        src_epoch = epoch_of[src_bind]
        dst_epoch = epoch_of[dst_bind]
        assert src_epoch == dst_epoch, (
            f"{label}: arrow binds run {src_epoch} to run {dst_epoch} — the endpoints name slices in different runs"
        )
        per_view_run[(start_event.get("pid"), src_epoch)] += 1

    seen_runs = {run for _, run in per_view_run}
    assert seen_runs == set(expected_runs), (
        f"{label}: rendered runs {sorted(map(str, seen_runs))}, expected {sorted(map(str, expected_runs))}"
    )
    for key, count in sorted(per_view_run.items(), key=lambda kv: str(kv[0])):
        assert count == per_view_per_run, (
            f"{label}: view pid={key[0]} run {key[1]} rendered {count} arrows, expected {per_view_per_run}"
        )
    return seen_runs


def test_two_runs_with_complete_phases_and_fanin_keep_their_own_anchors(tmp_path):
    """F1: completion anchors and complete-phase candidates are both per run.

    Both runs execute A -> B on core 0 with the same task ids and their own
    complete phases. Selecting anchors from a map merged across runs either
    returns both runs' rows or, when the merged rows look like two SPMD subtasks
    of one task, hands run B the row belonging to run A.
    """
    tasks = [
        _task(RUN_A, 0x101, 0, BASE_A),
        _task(RUN_A, 0x102, 0, BASE_A + 2.0),
        _task(RUN_B, 0x101, 0, BASE_B),
        _task(RUN_B, 0x102, 0, BASE_B + 2.0),
    ]
    sched = [
        [
            _phase("complete", RUN_A, BASE_A + 1.3, BASE_A + 1.6, processed=1),
            _phase("complete", RUN_A, BASE_A + 3.3, BASE_A + 3.6, processed=1),
            _phase("complete", RUN_B, BASE_B + 1.3, BASE_B + 1.6, processed=1),
            _phase("complete", RUN_B, BASE_B + 3.3, BASE_B + 3.6, processed=1),
        ]
    ]
    events = _render(
        tmp_path,
        tasks,
        scheduler_phases=sched,
        core_to_thread=[0],
        deps_edges={0x101: [0x102]},
    )

    _assert_flow_binds_within_one_run(events, expected_runs={RUN_A, RUN_B}, label="complete+fanin")

    # The completion arrow for each run must start inside that run's own kernel
    # slice: a merged anchor puts run B's completion flow on run A's bar.
    epoch_of = _epoch_by_event_id(events)
    completes = [e for e in events if e.get("cat") == "flow" and e.get("name") == "complete" and e.get("ph") == "s"]
    assert completes, "no completion arrows were emitted"
    for event in completes:
        bind = event.get("bind_id")
        if bind is None:
            continue
        anchor_epoch = epoch_of.get(bind)
        assert anchor_epoch is not None, f"completion arrow binds to {bind}, which carries no identity"
        run_window = (BASE_A, BASE_A + 10.0) if anchor_epoch == RUN_A else (BASE_B, BASE_B + 10.0)
        assert run_window[0] <= event["ts"] <= run_window[1], (
            f"completion arrow for run {anchor_epoch} starts at {event['ts']}, outside that run's window {run_window}"
        )


def test_completion_fallback_does_not_step_into_a_later_run(tmp_path):
    """F1: the "next-starting complete" fallback must not cross a run boundary.

    When a run's own complete phase is missing, attribution falls back to the
    next complete that starts after the finish time. Searching one flat
    per-thread list makes that fallback step into whatever run comes next: run
    A's task then binds its completion to a phase that executed in run B.

    Run A here has no complete phase at all, so the fallback is the only path
    available; run B has one. Run A's task must end up with no completion arrow
    rather than one pointing into run B.
    """
    tasks = [_task(RUN_A, 0x101, 0, BASE_A), _task(RUN_B, 0x101, 0, BASE_B)]
    sched = [[_phase("complete", RUN_B, BASE_B + 1.3, BASE_B + 1.6, processed=1)]]
    events = _render(tmp_path, tasks, scheduler_phases=sched, core_to_thread=[0])

    epoch_of = _epoch_by_event_id(events)
    grouped = defaultdict(dict)
    for event in events:
        if event.get("cat") == "flow" and event.get("name") == "complete":
            grouped[event["id"]][event["ph"]] = event

    run_windows = {RUN_A: (BASE_A - 5.0, BASE_A + 50.0), RUN_B: (BASE_B - 5.0, BASE_B + 50.0)}
    seen_source_runs = set()
    for pair in grouped.values():
        if "s" not in pair or "f" not in pair:
            continue
        source_epoch = epoch_of.get(pair["s"].get("bind_id"))
        if source_epoch is None:
            continue
        seen_source_runs.add(source_epoch)
        low, high = run_windows[source_epoch]
        assert low <= pair["f"]["ts"] <= high, (
            f"run {source_epoch}'s completion arrow lands at {pair['f']['ts']}, outside that run's window "
            f"({low}, {high}) — the fallback selected a complete phase from another run"
        )

    assert RUN_A not in seen_source_runs, (
        "run A has no complete phase of its own, so it must get no completion arrow; "
        "one was produced, which means a phase from another run was selected"
    )
    assert RUN_B in seen_source_runs, "run B has its own complete phase and should still be attributed"


def test_repeated_dummy_to_kernel_does_not_cross_runs(tmp_path):
    """F2: the dummy fallback anchor is queried within one run.

    Each run has dummy D -> kernel A. A fallback map keyed by bare task_id hands
    both runs' D anchors to each run's A, turning two legal edges into four and
    reporting the backwards one as a happens-before violation.
    """
    tasks = [_task(RUN_A, 0x101, 0, BASE_A), _task(RUN_B, 0x101, 0, BASE_B)]
    sched = [
        [
            _phase("dummy_task", RUN_A, BASE_A - 1.0, BASE_A - 0.8, task_id=0x100),
            _phase("complete", RUN_A, BASE_A + 1.3, BASE_A + 1.6, processed=1),
            _phase("dummy_task", RUN_B, BASE_B - 1.0, BASE_B - 0.8, task_id=0x100),
            _phase("complete", RUN_B, BASE_B + 1.3, BASE_B + 1.6, processed=1),
        ]
    ]
    events = _render(
        tmp_path,
        tasks,
        scheduler_phases=sched,
        core_to_thread=[0],
        deps_edges={0x100: [0x101]},
    )

    _assert_flow_binds_within_one_run(events, expected_runs={RUN_A, RUN_B}, label="dummy->kernel")
    violations = [e for e in events if e.get("name") == "hb_violation"]
    assert not violations, f"cross-run dummy pairing produced {len(violations)} bogus happens-before violation(s)"


def test_repeated_alloc_to_kernel_does_not_cross_runs(tmp_path):
    """F2, the orchestrator-side fallback: alloc -> kernel in two runs."""
    tasks = [_task(RUN_A, 0x101, 0, BASE_A), _task(RUN_B, 0x101, 0, BASE_B)]
    orch = [
        [
            {
                "phase": "orch_submit",
                "submit_idx": 0,
                "task_id": 0x100,
                "start_time_us": BASE_A - 2.0,
                "end_time_us": BASE_A - 1.8,
                "run_epoch": RUN_A,
            },
            {
                "phase": "orch_submit",
                "submit_idx": 1,
                "task_id": 0x100,
                "start_time_us": BASE_B - 2.0,
                "end_time_us": BASE_B - 1.8,
                "run_epoch": RUN_B,
            },
        ]
    ]
    events = _render(
        tmp_path,
        tasks,
        scheduler_phases=[[]],
        orchestrator_phases=orch,
        core_to_thread=[0],
        deps_edges={0x100: [0x101]},
    )

    _assert_flow_binds_within_one_run(events, expected_runs={RUN_A, RUN_B}, label="alloc->kernel")


def test_phase_only_run_still_renders_its_dependency(tmp_path):
    """F2 boundary: a run whose only rows are AICPU phases must still be walked.

    Run B contributes no AICore task row at all. Deriving the set of runs from
    the task map alone skips it entirely, so its dummy -> dummy edge disappears;
    collapsing every run into one unkeyed bucket instead mixes the two.
    """
    tasks = [_task(RUN_A, 0x101, 0, BASE_A)]
    sched = [
        [
            _phase("dummy_task", RUN_A, BASE_A - 1.0, BASE_A - 0.8, task_id=0x100),
            _phase("dummy_task", RUN_B, BASE_B - 1.0, BASE_B - 0.8, task_id=0x100),
            _phase("dummy_task", RUN_B, BASE_B, BASE_B + 0.2, task_id=0x101),
        ]
    ]
    events = _render(
        tmp_path,
        tasks,
        scheduler_phases=sched,
        core_to_thread=[0],
        deps_edges={0x100: [0x101]},
    )

    # A phase-only run contributes no arrow at all when the run set is derived
    # from the task map alone.
    _assert_flow_binds_within_one_run(events, expected_runs={RUN_A, RUN_B}, label="phase-only run")


def test_worker_event_ids_are_distinct_per_run(tmp_path):
    """F3: the event index is keyed by execution, so no run overwrites another.

    Run A and run B put the same task ids on the same core. With a
    (task_id, core_id) index the later run's event id replaces the earlier
    one's, and run A's arrows then bind to run B's slices.
    """
    tasks = [
        _task(RUN_A, 0x101, 0, BASE_A),
        _task(RUN_A, 0x102, 0, BASE_A + 2.0),
        _task(RUN_B, 0x101, 0, BASE_B),
        _task(RUN_B, 0x102, 0, BASE_B + 2.0),
    ]
    events = _render(tmp_path, tasks, core_to_thread=[0], deps_edges={0x101: [0x102]})

    worker_bars = [
        e for e in events if e.get("ph") == "X" and e.get("pid") == 4 and e.get("cat") == "event" and "id" in e
    ]
    assert len(worker_bars) == 4, f"expected one Worker View bar per execution, got {len(worker_bars)}"
    ids = [e["id"] for e in worker_bars]
    assert len(set(ids)) == 4, f"Worker View bars share event ids across runs: {ids}"

    by_run = defaultdict(set)
    for event in worker_bars:
        by_run[event["args"]["run_epoch"]].add(event["args"]["taskId"])
    assert by_run == {RUN_A: {0x101, 0x102}, RUN_B: {0x101, 0x102}}

    _assert_flow_binds_within_one_run(events, expected_runs={RUN_A, RUN_B}, label="event-id identity")


def test_final_events_carry_their_run_epoch(tmp_path):
    """F3: identity the parser preserved must survive into the emitted trace."""
    tasks = [_task(RUN_A, 0x101, 0, BASE_A), _task(RUN_B, 0x101, 0, BASE_B)]
    sched = [
        [
            _phase("complete", RUN_A, BASE_A + 1.3, BASE_A + 1.6, processed=1),
            _phase("complete", RUN_B, BASE_B + 1.3, BASE_B + 1.6, processed=1),
        ]
    ]
    events = _render(tmp_path, tasks, scheduler_phases=sched, core_to_thread=[0])

    worker_epochs = {
        e["args"].get("run_epoch")
        for e in events
        if e.get("ph") == "X" and e.get("pid") == 4 and e.get("cat") == "event"
    }
    assert worker_epochs == {RUN_A, RUN_B}, f"Worker View bars lost their run identity: {worker_epochs}"

    sched_epochs = {
        e["args"].get("run_epoch")
        for e in events
        if e.get("ph") == "X" and e.get("pid") == 2 and (e.get("args") or {}).get("phase") == "complete"
    }
    assert sched_epochs == {RUN_A, RUN_B}, f"scheduler phase bars lost their run identity: {sched_epochs}"


@pytest.mark.parametrize("epoch", [RUN_A, RUN_B])
def test_single_run_renders_one_arrow_per_edge(epoch, tmp_path):
    """Guard against the run-scoping splitting or duplicating a normal capture."""
    base = BASE_A if epoch == RUN_A else BASE_B
    tasks = [_task(epoch, 0x101, 0, base), _task(epoch, 0x102, 0, base + 2.0)]
    events = _render(tmp_path, tasks, core_to_thread=[0], deps_edges={0x101: [0x102]})
    _assert_flow_binds_within_one_run(events, expected_runs={epoch}, label=f"single run {epoch}")


RUN_C = 9


def _complete_bars(events):
    """Scheduler complete bars, keyed by the run they belong to."""
    by_run = {}
    for event in events:
        args = event.get("args") or {}
        if event.get("ph") == "X" and event.get("pid") == 2 and args.get("phase") == "complete":
            by_run.setdefault(args.get("run_epoch"), []).append(event)
    return by_run


def test_finish_attribution_counts_only_its_own_run(tmp_path):
    """R1: the finish counter search is scoped by run, like the arrow search.

    Run A finishes at BASE_A+1.2 with no complete phase of its own; run B has
    one. A per-thread search with a next-start fallback and no run input hands
    both finishes to B's phase, so B reports two attributed finishes while
    having drained one.

    The arrow assertions cannot see this — the arrows are already scoped — so
    this reads the rendered count off the phase bar.
    """
    tasks = [_task(RUN_A, 0x101, 0, BASE_A), _task(RUN_B, 0x101, 0, BASE_B)]
    sched = [[_phase("complete", RUN_B, BASE_B + 1.3, BASE_B + 1.6, processed=1)]]
    events = _render(tmp_path, tasks, scheduler_phases=sched, core_to_thread=[0])

    bars = _complete_bars(events)
    assert list(bars) == [RUN_B], f"expected one complete bar, for run B; got runs {sorted(map(str, bars))}"
    (bar,) = bars[RUN_B]
    assert bar["args"]["finish_rows_attributed"] == 1, (
        f"run B's complete phase claims {bar['args']['finish_rows_attributed']} attributed finishes; "
        "only one finish belongs to run B"
    )
    # The raw runtime count is untouched by attribution.
    assert bar["args"]["finishes_processed"] == 1


def test_synthetic_complete_without_raw_count_falls_back_within_its_run(tmp_path):
    """R1, the fallback branch: a record with no ``tasks_processed``.

    Such a record displays the attributed row count directly, so a cross-run
    attribution becomes the number shown on the bar rather than a side field.
    """
    tasks = [_task(RUN_A, 0x101, 0, BASE_A), _task(RUN_B, 0x101, 0, BASE_B)]
    synthetic = {
        "phase": "complete",
        "start_time_us": BASE_B + 1.3,
        "end_time_us": BASE_B + 1.6,
        "run_epoch": RUN_B,
        "loop_iter": 0,
    }
    events = _render(tmp_path, tasks, scheduler_phases=[[synthetic]], core_to_thread=[0])

    bars = _complete_bars(events)
    (bar,) = bars[RUN_B]
    assert bar["args"]["finishes_processed"] == 1, (
        f"a synthetic complete displayed {bar['args']['finishes_processed']} finishes; only one is run B's"
    )
    assert bar["args"]["tasks_processed"] == 1
    assert "(1)" in bar["name"], f"the bar label shows a cross-run count: {bar['name']!r}"


def test_three_runs_execute_execute_skip_render_without_crashing(tmp_path):
    """R2: a phase-only run alongside repeated ordinary executions.

    Runs A and B each execute T -> U once on one core, reusing task ids. Run C
    contributes only predicated-skip phases and has no AICore row at all.

    Inferring SPMD from a map merged across runs classifies T and U as SPMD
    because the merged map holds two rows each. The fan count for run C's
    fallback-anchored endpoints then dereferences a row that run never had, and
    conversion dies with KeyError instead of rendering the run.
    """
    tasks = [
        _task(RUN_A, 0x101, 0, BASE_A),
        _task(RUN_A, 0x102, 0, BASE_A + 2.0),
        _task(RUN_B, 0x101, 0, BASE_B),
        _task(RUN_B, 0x102, 0, BASE_B + 2.0),
    ]
    sched = [
        [
            _phase("complete", RUN_A, BASE_A + 1.3, BASE_A + 1.6, processed=1),
            _phase("complete", RUN_B, BASE_B + 1.3, BASE_B + 1.6, processed=1),
            _phase("predicated_skip", RUN_C, 200.0, 200.2, task_id=0x101),
            _phase("predicated_skip", RUN_C, 202.0, 202.2, task_id=0x102),
        ]
    ]
    events = _render(
        tmp_path,
        tasks,
        scheduler_phases=sched,
        core_to_thread=[0],
        deps_edges={0x101: [0x102]},
    )

    runs = _assert_flow_binds_within_one_run(events, expected_runs={RUN_A, RUN_B, RUN_C}, label="execute/execute/skip")
    assert runs == {RUN_A, RUN_B, RUN_C}


def test_repeated_single_core_task_is_not_labelled_spmd(tmp_path):
    """R2: executing once per run is not SPMD, however many runs are present.

    Row count in a merged map is the number of runs, not a block count, so a
    merged inference both mislabels the task and reports that number as its
    dependency fan.
    """
    tasks = [
        _task(RUN_A, 0x101, 0, BASE_A),
        _task(RUN_A, 0x102, 0, BASE_A + 2.0),
        _task(RUN_B, 0x101, 0, BASE_B),
        _task(RUN_B, 0x102, 0, BASE_B + 2.0),
    ]
    events = _render(tmp_path, tasks, core_to_thread=[0], deps_edges={0x101: [0x102]})

    names = {e["name"] for e in events if e.get("ph") == "X" and e.get("pid") == 4 and e.get("cat") == "event"}
    assert not any("spmd" in name for name in names), f"a once-per-run task was labelled SPMD: {sorted(names)}"

    for start_event, _finish in _flow_pairs(events):
        assert start_event.get("output_task_count") == 1, (
            f"dependency fan count is {start_event.get('output_task_count')}, "
            "which is the number of runs rather than a block count"
        )
        assert start_event.get("input_task_count") == 1


def test_authoritative_block_map_still_wins(tmp_path):
    """R2 guard: static block metadata is not overridden by per-run observation.

    A genuine SPMD task executes as several rows within one run; ``block_num``
    from deps.json remains the reported fan count.
    """
    tasks = [
        _task(RUN_A, 0x101, 0, BASE_A),
        {**_task(RUN_A, 0x101, 1, BASE_A), "core_id": 1},
        _task(RUN_A, 0x102, 0, BASE_A + 2.0),
    ]
    events = _render(
        tmp_path,
        tasks,
        core_to_thread=[0, 0],
        deps_edges={0x101: [0x102]},
        deps_block_map={0x101: 4},
    )

    pairs = _flow_pairs(events)
    assert pairs, "no dependency arrows were emitted"
    for start_event, _finish in pairs:
        assert start_event.get("output_task_count") == 4, (
            f"authoritative block_num=4 was replaced by {start_event.get('output_task_count')}"
        )


def test_genuine_spmd_task_skipped_in_a_later_run_does_not_crash(tmp_path):
    """R2: the fan count must be defined for a run that has no row for the task.

    Run A executes T as a real SPMD task — two subtask rows on two cores of the
    same type — so T is genuinely in the SPMD set. Run B reaches T only as a
    predicated skip and has no AICore row for it at all.

    The fan count for run B's endpoint therefore looks T up in a map that does
    not contain it. Dereferencing directly raises KeyError and aborts the whole
    conversion; a fallback-anchored endpoint contributes one logical task.
    """
    tasks = [
        _task(RUN_A, 0x101, 0, BASE_A),
        {**_task(RUN_A, 0x101, 1, BASE_A), "core_id": 1},
        _task(RUN_A, 0x102, 0, BASE_A + 2.0),
    ]
    sched = [
        [
            _phase("complete", RUN_A, BASE_A + 1.3, BASE_A + 1.6, processed=2),
            _phase("predicated_skip", RUN_B, BASE_B, BASE_B + 0.2, task_id=0x101),
            _phase("predicated_skip", RUN_B, BASE_B + 2.0, BASE_B + 2.2, task_id=0x102),
        ]
    ]
    events = _render(
        tmp_path,
        tasks,
        scheduler_phases=sched,
        core_to_thread=[0, 0],
        deps_edges={0x101: [0x102]},
    )

    _assert_flow_binds_within_one_run(events, expected_runs={RUN_A, RUN_B}, label="spmd skipped in later run")

    # Run A's endpoint keeps its observed multiplicity; run B's fallback-only
    # endpoint counts as a single logical task rather than crashing.
    epoch_of = _epoch_by_event_id(events)
    fan_by_run = {}
    for start_event, _finish in _flow_pairs(events):
        fan_by_run.setdefault(epoch_of[start_event["bind_id"]], set()).add(start_event.get("output_task_count"))
    assert fan_by_run[RUN_A] == {2}, f"run A's SPMD fan count is {fan_by_run[RUN_A]}, expected 2"
    assert fan_by_run[RUN_B] == {1}, f"run B's fallback-only fan count is {fan_by_run[RUN_B]}, expected 1"


class _CountingCompletePhase(dict):
    """A complete-phase record that counts reads of the key the search bisects.

    The completion searches locate a phase by binary-searching its start times.
    Whether that index is built once or rebuilt per query is invisible in the
    output and in wall-clock time on small inputs, but it is exactly visible in
    how often each record's ``start_time_us`` is read.
    """

    __slots__ = ("counter",)

    def __init__(self, mapping, counter):
        super().__init__(mapping)
        self.counter = counter

    def __getitem__(self, key):
        if key == "start_time_us":
            self.counter[0] += 1
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "start_time_us":
            self.counter[0] += 1
        return super().get(key, default)


def _count_start_reads(tmp_path, n, label):
    """Render n tasks against n complete phases; return start_time_us reads."""
    counter = [0]
    tasks = []
    phases = []
    for i in range(n):
        start = 10.0 + i * 10.0
        tasks.append(_task(RUN_A, 0x100 + i, 0, start))
        phases.append(
            _CountingCompletePhase(
                {
                    "phase": "complete",
                    "start_time_us": start + 1.1,
                    "end_time_us": start + 1.5,
                    "run_epoch": RUN_A,
                    "loop_iter": 0,
                    "tasks_processed": 1,
                },
                counter,
            )
        )
    out = Path(tmp_path) / f"{label}.json"
    generate_chrome_trace_json(tasks, str(out), scheduler_phases=[phases], core_to_thread=[0], runtime_name=TMR_RUNTIME)
    return counter[0]


def test_completion_lookup_does_not_rescan_the_phase_list_per_task(tmp_path):
    """The complete-phase start index is built once, not rebuilt per lookup.

    Both the finish counter and the completion arrows query the selector once
    per task. If the selector materialises the start-time list on each call,
    every query walks the whole run's phases, so the work grows with
    tasks × phases instead of tasks × log(phases).

    This is asserted as an access count, not a duration: a wall-clock threshold
    on an input small enough for a unit test would be noise, while the number
    of reads is exact and machine-independent. Doubling the input doubles the
    reads when the index is reused and roughly quadruples them when it is not.
    """
    small = _count_start_reads(tmp_path, 30, "small")
    large = _count_start_reads(tmp_path, 60, "large")

    assert small > 0, "the counting record never saw a read, so this measures nothing"
    # Linear in the input: doubling n doubles the reads. Rebuilding the index
    # per query lands near 4x. 3x sits between the two and is not a timing
    # threshold — the counts are deterministic.
    assert large <= 3 * small, (
        f"start_time_us reads grew from {small} to {large} when the input doubled "
        f"({large / small:.1f}x); a reused index grows about 2x, a per-query rebuild about 4x"
    )
    # Absolute guard, with headroom over the observed linear factor.
    assert large <= 16 * 60, f"{large} reads for 60 tasks / 60 phases is above the linear budget"
