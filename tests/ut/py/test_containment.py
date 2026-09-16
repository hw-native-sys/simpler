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
_RECORD_PREFIX = "[mono_ns=1000][T0x1][TIMING] emit_host_span: "


def _span_line(name, ts, dur, *, pid=100, inv=1, depth=0, attrs=""):
    """One `[STRACE]` record, as the host logger writes it.

    The record grammar is spelled here and nowhere else, so a test that means
    to describe a span cannot end up describing the format instead, and a
    change to the format moves one line rather than every log a test builds.
    """
    line = (
        f"{_RECORD_PREFIX}[STRACE] v=1 pid={pid} tid={pid} inv={inv} hid=abc "
        f"depth={depth} name={name} ts={ts} dur={dur}"
    )
    return f"{line} {attrs}" if attrs else line


def _host_log(*, pid=42, inv=1, runner=(1_000, 5_000), wall_ns=2_000, phases=(("sched", 700, 100),), dispatch=None):
    """One invocation's `[STRACE]` lines, as `emit_native_run_host_wall` writes them.

    ``dispatch`` is the ``(run_id, dispatch_id, slot_id, generation)`` the root
    ``chip.run`` span carries; without it the invocation names no dispatch, as
    a log from before the attributes existed does.
    """
    prefix = _RECORD_PREFIX
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


_HANDOFF = "run_id=5 task_slot=2 group_index=0 worker_id=1 dispatch_id=9"


def _caller_log(
    *,
    pid=100,
    level="network1",
    frame=(1026041362850311600, 1, 17),
    dispatch=(10_000, 200),
    complete=(23_000, 300),
    handoff=_HANDOFF,
):
    """The dispatching host's two spans for one remote handoff.

    The window they bracket is ``dispatch.start .. complete.end``: the caller
    publishes the frame inside the first and handles the peer's completion
    inside the second, so everything the peer did sits between them.
    """
    prefix = _RECORD_PREFIX
    head = f"[STRACE] v=1 pid={pid} tid={pid} inv=1 hid=abc"
    frame_attrs = "" if frame is None else " " + containment.format_frame_key(frame)
    return [
        f"{prefix}{head} depth=1 name={level}.dispatch ts={dispatch[0]} dur={dispatch[1]} "
        f"{handoff} endpoint_kind=remote_l3 role=scheduler{frame_attrs}",
        f"{prefix}{head} depth=1 name={level}.complete ts={complete[0]} dur={complete[1]} "
        f"{handoff} endpoint_kind=remote_l3 role=worker outcome=0",
    ]


def _peer_log(*, pid=200, level="node", frame=(1026041362850311600, 1, 17), window=(8_000_000_000, 11_000)):
    """The serving host's span for one frame, on that host's own clock."""
    prefix = _RECORD_PREFIX
    head = f"[STRACE] v=1 pid={pid} tid={pid} inv=1 hid=abc"
    return [
        f"{prefix}{head} depth=0 name={level}.remote_task ts={window[0]} dur={window[1]} "
        f"{containment.format_frame_key(frame)}"
    ]


def _windows(*log_groups):
    spans = list(parse_spans([line for group in log_groups for line in group]))
    return containment.remote_windows(spans)


def test_remote_window_slack_is_the_caller_window_less_the_peer_window():
    """The bound is a difference of two durations, each timed on one host."""
    (window,) = _windows(_caller_log(), _peer_log())

    assert window.frame == (1026041362850311600, 1, 17)
    assert window.caller_pid == 100
    assert window.peer_pid == 200
    # dispatch opens at 10_000 and complete ends at 23_300, so the caller held
    # the frame for 13_300 ns while the peer reported 11_000 of work.
    assert window.caller_duration_ns == 13_300
    assert window.peer_duration_ns == 11_000
    # 2_300 ns of spare window, and what two counters' rates can add over the
    # 13.3 us the caller held the frame for.
    assert window.rate_bound_ns == pytest.approx(13_300 * containment.MAX_RELATIVE_RATE)
    assert window.slack_ns == pytest.approx(2_300 + window.rate_bound_ns)
    assert (window.place_lo_ns, window.place_hi_ns) == (10_000.0, 10_000 + window.slack_ns)


def test_no_instant_is_compared_across_the_two_hosts():
    """Moving the peer's whole clock changes nothing the placement reports.

    This is the property the cross-host case rests on: the two machines'
    CLOCK_MONOTONIC axes have no common zero, so a method that reads one of
    them against the other would move here. Seventeen days is the real gap
    measured between two hosts' boot times.
    """
    seventeen_days_ns = 17 * 24 * 3_600 * 1_000_000_000
    near = _windows(_caller_log(), _peer_log(window=(8_000_000_000, 11_000)))[0]
    far = _windows(_caller_log(), _peer_log(window=(8_000_000_000 + seventeen_days_ns, 11_000)))[0]

    assert near.slack_ns == far.slack_ns
    assert near.place_lo_ns == far.place_lo_ns
    assert near.place_hi_ns == far.place_hi_ns
    # An event 4 us into the peer's window lands at the same caller instant.
    assert near.peer_ns_to_host_ns(8_000_000_000 + 4_000) == far.peer_ns_to_host_ns(
        8_000_000_000 + seventeen_days_ns + 4_000
    )


def test_offsets_inside_the_peer_window_are_exact():
    """One clock and one rate inside a host, so only the block's origin is bounded."""
    (window,) = _windows(_caller_log(), _peer_log(window=(8_000_000_000, 11_000)))

    assert window.peer_ns_to_host_ns(8_000_000_000) == 10_000.0
    assert window.peer_ns_to_host_ns(8_000_000_000 + 11_000) == 21_000.0


def test_one_side_of_the_wire_alone_yields_no_window():
    """A pile holding only the caller, or only the peer, produces no pairing."""
    assert _windows(_caller_log()) == []
    assert _windows(_peer_log()) == []


def test_a_frame_the_peer_never_served_is_not_paired():
    assert (
        _windows(_caller_log(frame=(1026041362850311600, 1, 17)), _peer_log(frame=(1026041362850311600, 1, 18))) == []
    )


def test_a_local_dispatch_carries_no_frame_and_is_ignored():
    """Only a dispatch that published a frame can name a peer window."""
    local = _caller_log(frame=None, handoff=_HANDOFF + " endpoint_kind=local_mailbox")
    assert _windows(local, _peer_log()) == []


def test_a_frame_key_missing_a_part_is_no_key():
    """The three arrive as one token, so two of them is a malformed record."""
    prefix = _RECORD_PREFIX
    head = "[STRACE] v=1 pid=100 tid=100 inv=1 hid=abc depth=1"
    # The token with its last part missing, which is what a reader sees when
    # three values were meant and two arrived.
    whole = containment.format_frame_key((41, 1, 17))
    missing_a_part = whole.rsplit(containment._FRAME_SEPARATOR, 1)[0]
    half = [
        f"{prefix}{head} name=network1.dispatch ts=10000 dur=200 {_HANDOFF} {missing_a_part}",
        f"{prefix}{head} name=network1.complete ts=23000 dur=300 {_HANDOFF} outcome=0",
    ]
    assert _windows(half, _peer_log()) == []


def test_a_truncated_dispatch_says_so_instead_of_pairing_nothing(capsys):
    """A record the attribute capacity cut short is a record problem, not a local dispatch.

    The log marks it with a trailing `~`. Reading that as "this dispatch
    carried no frame" is how the pairing would otherwise present a key the
    record could not fit: zero windows and no reason given.
    """
    prefix = _RECORD_PREFIX
    head = "[STRACE] v=1 pid=100 tid=100 inv=1 hid=abc depth=1"
    # The token as the attribute capacity left it: cut mid-value, with the
    # marker the logger appends in place of what it dropped.
    cut_short = containment.format_frame_key((1026041362850311600, 1, 17))[:-2] + "~"
    cut = [
        f"{prefix}{head} name=network1.dispatch ts=10000 dur=200 {_HANDOFF} {cut_short}",
        f"{prefix}{head} name=network1.complete ts=23000 dur=300 {_HANDOFF} outcome=0",
    ]

    assert _windows(cut, _peer_log()) == []
    assert "cut short by the record's attribute capacity" in capsys.readouterr().err


def test_a_peer_window_wider_than_its_caller_window_describes_no_containment():
    """Dropped rather than reported as a negative bound.

    The usual cause is two logs from different runs whose sequence numbers
    collide, which is a fact about that one pairing and not about the method.
    """
    assert _windows(_caller_log(), _peer_log(window=(8_000_000_000, 99_000))) == []


def test_two_frames_from_one_process_pair_to_their_own_peer_windows():
    """The handoff fields repeat across rounds; the frame header does not."""
    first = _caller_log(frame=(1026041362850311600, 1, 17), dispatch=(10_000, 200), complete=(23_000, 300))
    second = _caller_log(frame=(1026041362850311600, 1, 18), dispatch=(30_000, 200), complete=(48_000, 300))
    peers = _peer_log(frame=(1026041362850311600, 1, 17), window=(8_000_000_000, 11_000)) + _peer_log(
        frame=(1026041362850311600, 1, 18), window=(8_000_030_000, 16_000)
    )

    windows = {window.frame: window for window in _windows(first, second, peers)}

    assert sorted(windows) == [(1026041362850311600, 1, 17), (1026041362850311600, 1, 18)]
    assert windows[(1026041362850311600, 1, 17)].slack_ns == pytest.approx(
        2_300 + 13_300 * containment.MAX_RELATIVE_RATE
    )
    assert windows[(1026041362850311600, 1, 18)].caller_duration_ns == 18_300
    assert windows[(1026041362850311600, 1, 18)].slack_ns == pytest.approx(
        2_300 + 18_300 * containment.MAX_RELATIVE_RATE
    )


def test_one_frame_served_twice_refuses_rather_than_choosing():
    """Two peer windows for one header means the pile spans more than one run."""
    doubled = _peer_log(window=(8_000_000_000, 11_000)) + _peer_log(window=(9_000_000_000, 11_000))
    with pytest.raises(containment.ContainmentError, match="more than one run"):
        _windows(_caller_log(), doubled)


def test_a_dispatch_with_no_completion_in_the_log_is_not_paired():
    """Without the closing span the caller's window has no end to measure."""
    prefix = _RECORD_PREFIX
    head = "[STRACE] v=1 pid=100 tid=100 inv=1 hid=abc"
    frame = containment.format_frame_key((41, 1, 17))
    open_only = [f"{prefix}{head} depth=1 name=network1.dispatch ts=10000 dur=200 {_HANDOFF} {frame}"]
    assert _windows(open_only, _peer_log()) == []


def test_the_level_word_does_not_have_to_match_across_the_wire():
    """An L4 dispatches `network1.*` and the L3 that serves it emits `node.*`."""
    (window,) = _windows(_caller_log(level="network1"), _peer_log(level="node"))
    assert window.frame == (1026041362850311600, 1, 17)


def test_metadata_publishes_the_bound_and_both_ends_of_the_pairing():
    (window,) = _windows(_caller_log(), _peer_log())
    metadata = window.metadata()

    assert metadata["method"] == "span_containment_v1"
    assert (metadata["frame_session"], metadata["frame_worker"], metadata["frame_sequence"]) == (
        1026041362850311600,
        1,
        17,
    )
    assert (metadata["caller_pid"], metadata["peer_pid"]) == (100, 200)
    assert metadata["rate_bound_ns"] == 1
    assert metadata["slack_ns"] == 2_301
    assert metadata["place_lo_ns"] == 10_000
    assert metadata["place_hi_ns"] == 12_301


_DAY_NS = 24 * 3_600 * 1_000_000_000


def _three_level_logs(*, l4_base_ns=3 * _DAY_NS, l3_base_ns=900 * _DAY_NS):
    """An L5 dispatching to an L4 that dispatches on to an L3, three clocks.

    Each machine's log is written against its own boot, days apart, which is
    what makes the chain the only way any of them reach one axis.
    """
    prefix = _RECORD_PREFIX

    def head(pid):
        return f"[STRACE] v=1 pid={pid} tid={pid} inv=1 hid=abc"

    l4_frame = containment.format_frame_key((7, 1, 1))
    l3_frame = containment.format_frame_key((8, 1, 1))
    return [
        f"{prefix}{head(3000)} depth=1 name=network2.dispatch ts=1000 dur=100 {_HANDOFF} {l4_frame}",
        f"{prefix}{head(3000)} depth=1 name=network2.complete ts=60000 dur=100 {_HANDOFF} outcome=0",
        f"{prefix}{head(1500)} depth=0 name=network1.remote_task ts={l4_base_ns} dur=50000 {l4_frame}",
        f"{prefix}{head(1500)} depth=1 name=network1.dispatch ts={l4_base_ns + 2000} dur=100 {_HANDOFF} {l3_frame}",
        f"{prefix}{head(1500)} depth=1 name=network1.complete ts={l4_base_ns + 40000} dur=100 {_HANDOFF} outcome=0",
        f"{prefix}{head(2500)} depth=0 name=node.remote_task ts={l3_base_ns} dur=30000 {l3_frame}",
        f"{prefix}{head(2500)} depth=1 name=node.submit ts={l3_base_ns + 1000} dur=400 {_HANDOFF}",
    ]


def test_a_chain_composes_every_hop_and_sums_their_slacks():
    """Three machines, two hops, one axis.

    The L4 block slides inside the L5's window and the L3 block slides inside
    the L4's, so an L3 instant on the L5 axis is free by the sum of the two.
    """
    chain = containment.remote_chain(list(parse_spans(_three_level_logs())), log_pids={3000, 1500, 2500})

    assert len(chain.windows) == 2
    assert chain.peer_pids == frozenset({1500, 2500})
    # The L5 wrote this axis, so its own instants arrive exactly.
    assert chain.place(3000, 1000) == (1000, 0.0)
    # One hop for the L4's dispatch, two for the L3's submit. Each hop carries
    # what its own pair of counters can differ by, and those add too.
    first_rate = 59_100 * containment.MAX_RELATIVE_RATE
    second_rate = 38_100 * containment.MAX_RELATIVE_RATE
    assert chain.place(1500, 3 * _DAY_NS + 2000) == pytest.approx((3_000.0, 9_100.0 + first_rate))
    assert chain.place(2500, 900 * _DAY_NS + 1000) == pytest.approx((4_000.0, 17_200.0 + first_rate + second_rate))


def test_a_chain_is_unmoved_by_where_any_machine_booted():
    """Every hop subtracts two durations, so three boot times all cancel."""
    near = containment.remote_chain(
        list(parse_spans(_three_level_logs(l4_base_ns=_DAY_NS, l3_base_ns=2 * _DAY_NS))), log_pids={3000, 1500, 2500}
    )
    far = containment.remote_chain(
        list(parse_spans(_three_level_logs(l4_base_ns=500 * _DAY_NS, l3_base_ns=9_000 * _DAY_NS))),
        log_pids={3000, 1500, 2500},
    )

    assert near.place(1500, _DAY_NS + 2000) == far.place(1500, 500 * _DAY_NS + 2000)
    assert near.place(2500, 2 * _DAY_NS + 1000) == far.place(2500, 9_000 * _DAY_NS + 1000)


def test_a_peer_instant_belonging_to_no_frame_is_not_placed():
    """The peer's log covers its whole run; only what a frame held is drawn."""
    chain = containment.remote_chain(list(parse_spans(_three_level_logs())), log_pids={3000, 1500, 2500})
    assert chain.place(2500, 900 * _DAY_NS + 999_999) is None


def test_a_process_on_the_axis_needs_no_hop_and_carries_no_bound():
    chain = containment.remote_chain(list(parse_spans(_caller_log() + _peer_log())), log_pids={100, 200})
    assert chain.place(100, 12_345) == (12_345, 0.0)


def test_a_cycle_in_the_windows_refuses_rather_than_looping():
    """Two windows naming each other's process reach no axis at all."""
    first = containment.RemoteWindow(
        frame=(1, 1, 1),
        caller_pid=10,
        caller_start_ns=0,
        caller_duration_ns=100,
        peer_pid=20,
        peer_start_ns=0,
        peer_duration_ns=50,
    )
    second = containment.RemoteWindow(
        frame=(1, 1, 2),
        caller_pid=20,
        caller_start_ns=0,
        caller_duration_ns=100,
        peer_pid=10,
        peer_start_ns=0,
        peer_duration_ns=50,
    )
    chain = containment.RemoteChain((first, second), frozenset())

    with pytest.raises(containment.ContainmentError, match="cycle"):
        chain.place(10, 10)


def test_rebase_moves_a_window_a_peer_recorded_onto_the_axis():
    """A Rank on the far side is bounded twice, and the two bounds add.

    Its own `runner_run` window keeps its width; what the frame adds is where
    that window sits, and how far it can be from where it was drawn.
    """
    spans = list(parse_spans(_three_level_logs()))
    chain = containment.remote_chain(spans, log_pids={3000, 1500, 2500})
    peer_window = containment.HostWindow(
        pid=2500, inv=1, start_ns=900 * _DAY_NS + 2_000, duration_ns=5_000, device_wall_ns=2_000, phases={}
    )
    local_window = containment.HostWindow(
        pid=3000, inv=1, start_ns=1_500, duration_ns=5_000, device_wall_ns=2_000, phases={}
    )

    rebased = {window.pid: window for window in containment.rebase_windows([peer_window, local_window], chain)}

    # On the axis already: untouched, so its own slack is unchanged.
    assert rebased[3000] is local_window
    # Two hops away: the origin moves onto the axis and the width grows by the
    # chain's own bound, so `place()` reports the summed slack.
    chained = 17_200 + (59_100 + 38_100) * containment.MAX_RELATIVE_RATE
    assert rebased[2500].start_ns == 5_000
    assert rebased[2500].duration_ns == round(5_000 + chained)
    assert containment.place(rebased[2500]).slack_ns == pytest.approx(3_000 + chained, abs=1)


def test_a_window_the_chain_cannot_place_is_dropped():
    """A peer window outside every frame has no axis to be drawn on."""
    spans = list(parse_spans(_three_level_logs()))
    chain = containment.remote_chain(spans, log_pids={3000, 1500, 2500})
    stray = containment.HostWindow(
        pid=2500, inv=9, start_ns=900 * _DAY_NS + 900_000, duration_ns=5_000, device_wall_ns=2_000, phases={}
    )
    assert containment.rebase_windows([stray], chain) == []


def test_a_peer_already_on_this_axis_is_left_where_it_was_recorded():
    """A peer whose timestamps already land in the window keeps them.

    Placing it would move it to the window's start and erase the real gap
    between the dispatch and the peer picking the frame up. The bound is
    published either way, because nothing here says whether the two timestamps
    line up from one clock writing them or from two reading alike.
    """
    loopback = _peer_log(window=(15_000, 5_000))
    (window,) = _windows(_caller_log(), loopback)
    assert window.same_axis

    chain = containment.remote_chain(list(parse_spans(_caller_log() + loopback)), log_pids={100, 200})
    placed = chain.place(200, 15_500)

    assert placed is not None
    placed_ns, slack_ns = placed
    assert placed_ns == 15_500
    assert slack_ns == pytest.approx(window.slack_ns)
    assert 200 in chain.observed_pids


def test_a_peer_on_another_clock_is_still_placed():
    """The same test, failing, is what makes a second clock a second clock."""
    (window,) = _windows(_caller_log(), _peer_log())
    assert not window.same_axis


def test_a_process_the_caller_cannot_vouch_for_is_not_placed():
    """A pid nothing names is dropped, not taken for one on the axis.

    A peer's own children write their own logs under their own pids, and no
    frame window names them. Reading such a pid as "already on the axis" would
    draw a second machine's raw timestamps as if they were this one's.
    """
    chain = containment.remote_chain(list(parse_spans(_caller_log() + _peer_log())), log_pids={100, 200})
    assert chain.place(999, 12_345) is None
    assert chain.place(100, 12_345) == (12_345, 0.0)


def test_a_peer_side_process_with_no_window_of_its_own_is_not_on_this_axis():
    """A peer's own children write their own logs, and no frame names them.

    Collecting the peer host's log directory to enable the splice brings them
    along. Their whole interval is days from anything the dispatching process
    wrote, which is what says they are a second machine's - so they are dropped
    rather than drawn at their own clock's raw timestamps.
    """
    prefix = _RECORD_PREFIX
    stray = [
        f"{prefix}[STRACE] v=1 pid=2600 tid=2600 inv=1 hid=abc depth=0 "
        f"name=chip.run ts={8_000_000_000 + 2_000} dur=6000 run_id=1 dispatch_id=1 slot_id=0 generation=0"
    ]
    spans = list(parse_spans(_caller_log() + _peer_log() + stray))
    chain = containment.remote_chain(spans, log_pids={100, 200, 2600})

    assert 2600 not in chain.axis_pids
    assert chain.place(2600, 8_000_000_000 + 2_000) is None
    # The dispatching process it was collected alongside is still on the axis.
    assert chain.place(100, 12_000) == (12_000, 0.0)


def test_a_process_with_no_window_of_its_own_is_left_out_of_a_spliced_merge():
    """A pile that holds two machines vouches for a process or leaves it out.

    Two machines' clocks can read however close to one another, so no rule over
    the timestamps separates a local process from one of the peer's. What the
    merge can vouch for is a process at the top of a chain, and one whose own
    device window the merge is of; the rest are dropped rather than drawn where
    nothing places them.
    """
    prefix = _RECORD_PREFIX
    unvouched = [
        f"{prefix}[STRACE] v=1 pid=101 tid=101 inv=1 hid=abc depth=0 name=node.graph_build ts=9000 dur=400 run_id=1"
    ]
    spans = list(parse_spans(_caller_log() + _peer_log() + unvouched))
    chain = containment.remote_chain(spans, log_pids={100, 101, 200})

    assert chain.axis_pids == frozenset({100})
    assert chain.place(100, 9_000) == (9_000, 0.0)
    assert chain.place(101, 9_000) is None


def test_with_nothing_spliced_every_process_is_on_the_one_axis():
    """A same-host merge holds one clock, so no process has to earn its place."""
    chain = containment.remote_chain(list(parse_spans(_caller_log(frame=None))), log_pids={100, 555})
    assert chain.windows == ()
    assert chain.place(555, 12_345) == (12_345, 0.0)


def test_the_instant_a_frame_ends_at_belongs_to_what_came_after_it():
    """The window is half-open, so its end is the first instant outside it."""
    chain = containment.remote_chain(list(parse_spans(_caller_log() + _peer_log())), log_pids={100, 200})
    (window,) = chain.windows
    last_inside = window.peer_start_ns + window.peer_duration_ns - 1

    assert chain.place(200, last_inside) is not None
    assert chain.place(200, window.peer_start_ns + window.peer_duration_ns) is None


def test_one_frame_dispatched_twice_refuses_rather_than_keeping_the_later():
    """The caller side refuses a repeated frame, as the peer side does.

    Keeping the later span would silently move the window: the earlier
    dispatch's interval is replaced by a different, later one, and the
    completion search then closes the wrong end.
    """
    doubled = _caller_log(dispatch=(10_000, 200), complete=(23_000, 300)) + _caller_log(
        dispatch=(50_000, 200), complete=(70_000, 300)
    )
    with pytest.raises(containment.ContainmentError, match="dispatched by two spans"):
        _windows(doubled, _peer_log())


def test_metadata_separates_where_the_bound_is_from_where_the_block_sits():
    """A peer left where it was recorded is not sitting at the bound's floor."""
    placed = _windows(_caller_log(), _peer_log())[0].metadata()
    observed = _windows(_caller_log(), _peer_log(window=(15_000, 5_000)))[0].metadata()

    assert placed["observed"] is False
    assert placed["drawn_at_ns"] == placed["place_lo_ns"] == 10_000
    assert observed["observed"] is True
    assert observed["place_lo_ns"] == 10_000
    assert observed["drawn_at_ns"] == 15_000


def test_a_peer_with_a_device_window_of_its_own_is_still_only_a_peer():
    """Counted once, down the path that places it, never also as an axis process."""
    prefix = _RECORD_PREFIX
    peer_with_window = _peer_log() + [
        f"{prefix}[STRACE] v=1 pid=200 tid=200 inv=2 hid=abc depth=1 name=chip.run.runner_run ts=8000001000 dur=800 ",
        f"{prefix}[STRACE] v=1 pid=200 tid=200 inv=2 hid=abc depth=2 "
        f"name=chip.run.runner_run.device_wall ts=0 dur=200 clk=dev",
    ]
    chain = containment.remote_chain(list(parse_spans(_caller_log() + peer_with_window)), log_pids={100, 200})

    assert 200 in chain.peer_pids
    assert 200 not in chain.axis_pids
