#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest

from simpler_setup.tools import containment
from simpler_setup.tools.strace_timing import parse_spans

_GHZ = 1_000_000_000


def _host_log(*, pid=42, inv=1, runner=(1_000, 5_000), wall_ns=2_000, phases=(("sched", 700, 100),), dispatch=None):
    """One invocation's `[STRACE]` lines, as `emit_native_run_host_wall` writes them.

    ``dispatch`` is the ``(run_id, dispatch_id, slot_id, generation)`` the root
    ``chip.run`` span carries; without it the invocation names no dispatch, as
    a log from before the attributes existed does.
    """
    prefix = "[mono_ns=1000][T0x1][TIMING] emit_host_span: "
    head = f"[STRACE] v=1 pid={pid} tid={pid} inv={inv} hid=abc"
    lines = []
    if dispatch is not None:
        run_id, dispatch_id, slot_id, generation = dispatch
        lines.append(
            f"{prefix}{head} depth=0 name=chip.run ts={runner[0] - 500} dur={runner[1] + 900} "
            f"run_id={run_id} dispatch_id={dispatch_id} slot_id={slot_id} generation={generation} run_epoch=7"
        )
    lines += [
        f"{prefix}{head} depth=1 name=chip.run.runner_run ts={runner[0]} dur={runner[1]} ",
        f"{prefix}{head} depth=2 name=chip.run.runner_run.device_wall ts=0 dur={wall_ns} clk=dev",
    ]
    lines += [
        f"{prefix}{head} depth=3 name=chip.run.runner_run.device_wall.{name} ts={ts} dur={dur} clk=dev"
        for name, ts, dur in phases
    ]
    return lines


def _sidecar(*, run_id, endpoint_dispatch_id, pipeline_slot=0, pipeline_generation=1):
    """The fields of ``dispatch_identity.json`` the pairing reads."""
    return {
        "schema_version": 1,
        "run_id": run_id,
        "endpoint_dispatch_id": endpoint_dispatch_id,
        "pipeline_slot": pipeline_slot,
        "pipeline_generation": pipeline_generation,
    }


def _capture(*, base=0, sched=(1_900, 1_950), tasks=((2_100, 2_200),), frequency_hz=_GHZ, **extra):
    return {
        "metadata": {"clock_freq_hz": frequency_hz},
        "aicore_tasks": [[0, 7, 1, base + start, base + end, 0] for start, end in tasks],
        "aicpu_scheduler_phases": [
            [{"kind": "dispatch", "start_cycles": base + sched[0], "end_cycles": base + sched[1]}]
        ],
        **extra,
    }


def _window(**kwargs):
    (window,) = containment.host_windows(parse_spans(_host_log(**kwargs)))
    return window


def test_host_windows_need_both_ends_of_the_bracket():
    """An invocation that launched nothing brackets nothing."""
    lines = _host_log()
    assert containment.host_windows(parse_spans(lines))

    without_wall = [line for line in lines if "device_wall ts=" not in line]
    assert containment.host_windows(parse_spans(without_wall)) == []


def test_join_narrows_the_offset_to_the_window_the_records_do_not_fill():
    """The offset is bounded by the window, not solved for.

    A 100 ns `sched` window holding 50 ns of records leaves exactly 50 ns of
    freedom, and the midpoint of that interval is what gets used.
    """
    join = containment.join_origin(_window(), containment.capture_windows(_capture()))

    assert join.sources == ("sched", "device_wall")
    assert join.interval_cycles == (1_150, 1_200)
    assert join.residual_ns == 50
    assert join.origin_cycles == 1_175


def test_join_refuses_artifacts_that_cannot_be_one_run():
    """Records wider than the window that supposedly held them are not that run."""
    capture = containment.capture_windows(_capture(sched=(1_900, 2_100)))

    with pytest.raises(containment.ContainmentError, match="do not overlap"):
        containment.join_origin(_window(), capture)


def test_placement_publishes_the_window_it_could_not_narrow():
    """Slack is the outer window minus the device wall it brackets."""
    placement = containment.place(_window(), containment.capture_windows(_capture()))

    assert placement.slack_ns == 3_000
    assert (placement.place_lo_ns, placement.place_hi_ns) == (1_000, 4_000)
    # Cycle 2_100 sits 925 ns past the joined origin, drawn from the window start.
    assert placement.map_cycles_to_host_ns(2_100) == 1_925
    assert placement.metadata()["device_extent_ns"] == 2_000


def test_placement_keeps_every_record_inside_its_window():
    """The property the whole mechanism exists to guarantee."""
    capture = containment.capture_windows(_capture())
    placement = containment.place(_window(), capture)

    first, last = (placement.map_cycles_to_host_ns(cycles) for cycles in capture.extent)
    assert placement.host.start_ns <= first <= last <= placement.host.end_ns


def test_placement_without_a_capture_places_the_host_logs_own_device_spans():
    """A host swimlane has no capture, only the `clk=dev` spans in the log."""
    placement = containment.place(_window(phases=(("sched", 700, 100), ("preamble", 0, 300))))

    assert placement.capture is None
    assert placement.slack_ns == 3_000
    assert placement.phase_ns_to_host_ns(700) == 1_700
    assert "join" not in placement.metadata()


def test_placement_refuses_a_window_too_narrow_for_its_own_run():
    with pytest.raises(containment.ContainmentError, match="only 1.5 us wide"):
        containment.place(_window(runner=(1_000, 1_500)), containment.capture_windows(_capture()))


def test_pairing_reads_the_rank_out_of_the_window_each_capture_fills():
    """Neither artifact names the other, so the device windows do the matching."""
    hosts = [
        _window(pid=10, phases=(("sched", 700, 100),)),
        _window(pid=11, runner=(50_000, 5_000), phases=(("sched", 700, 300),)),
    ]
    captures = {
        0: containment.capture_windows(_capture(sched=(1_900, 1_950))),
        1: containment.capture_windows(_capture(base=100_000, sched=(1_900, 2_150))),
    }

    pairs, diagnostics = containment.pair_captures(hosts, captures)

    assert [pairs[rank].pid for rank in (0, 1)] == [10, 11]
    assert diagnostics[0]["window_excess_ns"] == 50
    assert diagnostics[1]["source"] == "device_window_fit"


def test_pairing_refuses_to_guess_between_look_alike_ranks():
    """Two Ranks of the same shape are not distinguishable, and guessing is wrong."""
    hosts = [_window(pid=10), _window(pid=11, runner=(50_000, 5_000))]
    captures = {
        0: containment.capture_windows(_capture()),
        1: containment.capture_windows(_capture(base=100_000)),
    }

    with pytest.raises(containment.ContainmentError, match="pair ambiguously"):
        containment.pair_captures(hosts, captures)

    pairs, diagnostics = containment.pair_captures(hosts, captures, forced={0: 11, 1: 10})
    assert [pairs[rank].pid for rank in (0, 1)] == [11, 10]
    assert diagnostics[0]["source"] == "pinned"


def test_cross_uncertainty_sums_the_two_widest_slacks():
    """Either end of a cross-Rank read carries its own placement's slack."""
    placements = [
        containment.place(_window(), containment.capture_windows(_capture())),
        containment.place(_window(runner=(1_000, 9_000)), containment.capture_windows(_capture())),
    ]

    assert containment.cross_uncertainty_ns(placements) == 3_000 + 7_000
    assert containment.cross_uncertainty_ns(placements[:1]) is None


def test_one_captures_own_slack_is_not_a_cross_rank_bound():
    """The rule the document and the line printed beside it must share.

    A single placement bounds nothing across Ranks, so the answer is "unknown"
    rather than that capture's own slack — which a `[-2:]` slice would return.
    """
    assert containment.sum_two_widest_slacks([6_000]) is None
    assert containment.sum_two_widest_slacks([]) is None
    assert containment.sum_two_widest_slacks([1_000, 6_000, 4_000]) == 10_000


def _repeated_invocations(pid, count, *, dispatch_of=None):
    """One process running the same shape `count` times, as any real run does."""
    lines = []
    for inv in range(1, count + 1):
        lines += _host_log(
            pid=pid,
            inv=inv,
            runner=(1_000 + inv * 10_000, 5_000),
            dispatch=None if dispatch_of is None else dispatch_of(inv),
        )
    return lines


def test_identity_pairs_one_capture_among_a_process_repeated_invocations():
    """The window fit cannot see which run of many a capture came from; the dispatch can.

    Every invocation here has the same shape, so every window fits the capture
    equally well — which is exactly the case the fit reports as ambiguous.
    """
    lines = _repeated_invocations(10, 4, dispatch_of=lambda inv: (17, inv, 0, 1))
    hosts = containment.host_windows(parse_spans(lines))
    captures = {0: containment.capture_windows(_capture())}

    with pytest.raises(containment.ContainmentError, match="pair ambiguously"):
        containment.pair_captures(hosts, captures)

    identity = containment.capture_identity(_sidecar(run_id=17, endpoint_dispatch_id=3))
    pairs, diagnostics = containment.pair_captures(hosts, captures, identities={0: identity})

    assert (pairs[0].pid, pairs[0].inv) == (10, 3)
    assert diagnostics[0]["source"] == "capture_sidecar"
    assert diagnostics[0]["sidecar_keys"] == ["dispatch"]


def test_identity_is_the_same_four_numbers_under_two_spellings():
    """The sidecar renames three of them; nothing else may know that."""
    (window,) = containment.host_windows(parse_spans(_host_log(dispatch=(17, 4, 2, 9))))

    assert window.identity == (17, 4, 2, 9)
    assert (
        containment.capture_identity(
            _sidecar(run_id=17, endpoint_dispatch_id=4, pipeline_slot=2, pipeline_generation=9)
        )
        == window.identity
    )


def test_a_log_without_the_identity_attributes_still_pairs_by_window_fit():
    """Old logs keep working; they just fall back to the looser route."""
    hosts = [
        _window(pid=10, phases=(("sched", 700, 100),)),
        _window(pid=11, runner=(50_000, 5_000), phases=(("sched", 700, 300),)),
    ]
    captures = {
        0: containment.capture_windows(_capture(sched=(1_900, 1_950))),
        1: containment.capture_windows(_capture(base=100_000, sched=(1_900, 2_150))),
    }

    pairs, diagnostics = containment.pair_captures(hosts, captures, identities={})

    assert [pairs[rank].pid for rank in (0, 1)] == [10, 11]
    assert [diagnostics[rank]["source"] for rank in (0, 1)] == ["device_window_fit"] * 2


def test_a_pin_names_the_invocation_when_the_process_ran_more_than_once():
    """`RANK=PID` cannot name one of four runs, and says so rather than guessing."""
    hosts = containment.host_windows(parse_spans(_repeated_invocations(10, 4)))
    captures = {0: containment.capture_windows(_capture())}

    with pytest.raises(containment.ContainmentError, match=r"4 placeable invocations \(inv 1, 2, 3, 4\)"):
        containment.pair_captures(hosts, captures, forced={0: 10})

    pairs, diagnostics = containment.pair_captures(hosts, captures, forced={0: (10, 2)})

    assert (pairs[0].pid, pairs[0].inv) == (10, 2)
    assert diagnostics[0]["source"] == "pinned"


def test_the_window_fit_refuses_a_search_it_cannot_finish():
    """A Host log holds every invocation of the run, not just the captured one.

    Scoring whole assignments costs `P(windows, captures)`, so without a
    dispatch identity to key on this has to refuse rather than grind: the same
    shape at 100 windows was measured at ~231 s.
    """
    hosts = containment.host_windows(parse_spans(_repeated_invocations(10, 40)))
    captures = {rank: containment.capture_windows(_capture()) for rank in range(4)}

    with pytest.raises(containment.ContainmentError, match="too many to score"):
        containment.pair_captures(hosts, captures)


def test_the_extent_counts_every_record_the_converter_draws():
    """Slack bounds only what it measured, so anything drawn has to be in it.

    A lifecycle record and an AICore task's receive point are both drawn on the
    device clock, and both sit outside the task/phase cycles the streams above
    report.
    """
    plain = containment.capture_windows(_capture())
    assert plain.extent == (1_900, 2_200)

    # `receive_to_start_cycles` opens the task bar 300 cycles before the kernel.
    with_receive = containment.capture_windows(_capture(aicore_tasks=[[0, 7, 1, 2_100, 2_200, 300]]))
    assert with_receive.extent == (1_800, 2_200)

    with_lifecycle = containment.capture_windows(
        _capture(aicpu_lifecycle_records=[{"aicpu_thread_id": 0, "exit_wait_end_cycles": 3_400}])
    )
    assert with_lifecycle.extent == (1_900, 3_400)


def test_a_record_outside_the_run_wall_is_caught_rather_than_drawn_outside_it():
    """The join is what enforces containment, so the extent has to feed it."""
    capture = containment.capture_windows(
        _capture(aicpu_lifecycle_records=[{"aicpu_thread_id": 0, "exit_wait_end_cycles": 9_000}])
    )

    with pytest.raises(containment.ContainmentError, match="do not overlap"):
        containment.join_origin(_window(), capture)


def test_host_pid_is_what_separates_two_ranks_of_one_group():
    """A group's members share every other key the sidecar carries.

    `dispatch_id` counts per worker, so both Ranks of one round report the same
    one and the dispatch key narrows to both windows. The device windows match
    too, because the Ranks run the same shape. Only the pid the ChipWorker
    child wrote into its own sidecar tells them apart.
    """
    hosts = containment.host_windows(
        parse_spans(_host_log(pid=10, dispatch=(1, 1, 0, 1)) + _host_log(pid=11, dispatch=(1, 1, 0, 1)))
    )
    captures = {rank: containment.capture_windows(_capture()) for rank in (0, 1)}
    identities = {rank: containment.capture_identity(_sidecar(run_id=1, endpoint_dispatch_id=1)) for rank in (0, 1)}

    with pytest.raises(containment.ContainmentError, match="pair ambiguously"):
        containment.pair_captures(hosts, captures, identities=identities)

    pairs, diagnostics = containment.pair_captures(hosts, captures, identities=identities, host_pids={0: 11, 1: 10})

    assert [pairs[rank].pid for rank in (0, 1)] == [11, 10]
    assert diagnostics[0]["sidecar_keys"] == ["host_pid", "dispatch"]


def test_host_pid_alone_still_needs_the_dispatch_when_the_process_ran_twice():
    """Which process and which invocation are two questions, one key each."""
    lines = _repeated_invocations(10, 3, dispatch_of=lambda inv: (1, inv, 0, 1))
    lines += _repeated_invocations(11, 3, dispatch_of=lambda inv: (1, inv, 0, 1))
    hosts = containment.host_windows(parse_spans(lines))
    captures = {0: containment.capture_windows(_capture())}

    with pytest.raises(containment.ContainmentError, match="pair ambiguously"):
        containment.pair_captures(hosts, captures, host_pids={0: 11})

    pairs, diagnostics = containment.pair_captures(
        hosts,
        captures,
        identities={0: containment.capture_identity(_sidecar(run_id=1, endpoint_dispatch_id=2))},
        host_pids={0: 11},
    )

    assert (pairs[0].pid, pairs[0].inv) == (11, 2)
    assert diagnostics[0]["source"] == "capture_sidecar"
