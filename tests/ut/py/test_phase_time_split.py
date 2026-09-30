# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for the bind-phase CPU-time split reader."""

import pytest

from simpler_setup.tools import phase_time_split


def _span(phase, dur_ns, cpu_ns, rec_cpu_ns, *, faults=None, pid=7, inv=1, ts=1000):
    counters = {"minflt": 0, "tminflt": 0, "nivcsw": 0, "nvcsw": 0, **(faults or {})}
    return (
        f"[STRACE] v=1 pid={pid} tid={pid} inv={inv} hid=abc depth=2 "
        f"name=chip.run.bind.{phase} ts={ts} dur={dur_ns} "
        f"minflt={counters['minflt']} tminflt={counters['tminflt']} "
        f"nivcsw={counters['nivcsw']} nvcsw={counters['nvcsw']} "
        f"cpu_ns={cpu_ns} rec_cpu_ns={rec_cpu_ns}\n"
    )


def test_parse_reads_every_counter(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(
        _span(
            "host_orch", 2_000_000, 1_500_000, 8_000_000, faults={"minflt": 99, "tminflt": 7, "nvcsw": 3, "nivcsw": 1}
        )
    )

    rows = phase_time_split.parse([log])

    assert len(rows) == 1
    assert rows[0] == {
        "phase": "host_orch",
        "pid": 7,
        "inv": 1,
        "ts": 1000,
        "dur_us": 2000.0,
        "minflt": 99,
        "tminflt": 7,
        "nivcsw": 1,
        "nvcsw": 3,
        "cpu_ns": 1_500_000,
        "rec_cpu_ns": 8_000_000,
    }


def test_parse_ignores_records_that_are_not_bind_segments(tmp_path):
    """Only a segment of the stage counts: not another span, and not an
    orchestrator operation one level deeper, which carries none of the counters."""
    log = tmp_path / "run.log"
    log.write_text(
        "some other TIMING line\n"
        + "[STRACE] v=1 pid=7 tid=7 inv=1 hid=abc depth=1 name=chip.run.bind ts=1 dur=9\n"
        + "[STRACE] v=1 pid=7 tid=7 inv=1 hid=abc depth=3 name=chip.run.bind.host_orch.graph_submit ts=2 dur=3\n"
        + _span("graph_pack", 1000, 1000, 0)
    )

    assert [row["phase"] for row in phase_time_split.parse([log])] == ["graph_pack"]


def test_a_truncated_cold_bind_does_not_promote_its_successor(tmp_path, monkeypatch, capsys):
    """Warm-up is decided before any row is dropped.

    A bind whose attribute string was cut is still a bind that ran. Excluding it
    from the statistics first would make the *next* bind the earliest one left,
    and that bind's numbers — genuinely warm — would be reported as the cold row.
    """
    log = tmp_path / "run.log"
    log.write_text(
        # The rank's first bind, truncated: no rec_cpu_ns.
        "[STRACE] v=1 pid=7 tid=7 inv=1 hid=abc depth=2 name=chip.run.bind.host_orch ts=1000 dur=9000000 "
        "minflt=1 tminflt=1 nivcsw=0 nvcsw=0 cpu_ns=9000000 rec_cpu~\n"
        + _span("host_orch", 2_000_000, 1_500_000, 0, pid=7, inv=2, ts=20_000)
        + _span("host_orch", 2_100_000, 1_600_000, 0, pid=7, inv=3, ts=30_000)
    )
    monkeypatch.setattr("sys.argv", ["phase_time_split", str(log)])

    phase_time_split.main()

    rows = [line for line in capsys.readouterr().out.splitlines() if line.startswith("host_orch")]
    # The cold bind is the truncated one, which no row describes, so the two
    # intact binds are both warm rather than one of them standing in for it.
    assert len(rows) == 1
    assert "warm" in rows[0]
    assert "2100.0" not in rows[0]  # median of 2000 and 2100, not one of them alone


def test_a_truncated_attribute_string_is_reported_rather_than_computed_over(tmp_path, monkeypatch, capsys):
    """The logger caps the attribute field and marks a cut with `~`.

    A counter it cut is absent, not zero, so the row is excluded and counted — a
    figure derived from a missing counter would be a guess, and treating the
    absence as 0 would read as a segment that took no CPU.
    """
    log = tmp_path / "run.log"
    log.write_text(
        _span("host_orch", 2_000_000, 1_500_000, 0, pid=7, inv=1)
        + _span("host_orch", 2_000_000, 1_400_000, 0, pid=7, inv=2)
        + "[STRACE] v=1 pid=7 tid=7 inv=3 hid=abc depth=2 name=chip.run.bind.host_orch ts=9000 dur=2000000 "
        "minflt=1 tminflt=1 nivcsw=0 nvcsw=0 cpu_ns=1500000 rec_cpu~\n"
    )
    monkeypatch.setattr("sys.argv", ["phase_time_split", str(log)])

    phase_time_split.main()

    output = capsys.readouterr().out
    assert "1 span(s) had a truncated attribute string" in output
    assert len([line for line in output.splitlines() if "host_orch" in line]) == 2


def test_off_cpu_is_the_wall_the_bind_thread_did_not_run():
    rows = [
        {
            "phase": "p",
            "dur_us": 100.0,
            "cpu_ns": 40_000,
            "rec_cpu_ns": 0,
            **dict.fromkeys(("minflt", "tminflt", "nvcsw", "nivcsw"), 0),
        }
    ]

    stats = phase_time_split.summarise(rows)

    assert stats["cpu"] == pytest.approx(40.0)
    assert stats["offcpu"] == pytest.approx(60.0)


def test_cpu_above_the_wall_is_counted_rather_than_clamped_away():
    """One thread cannot outspend the segment's wall, so the row's mark is another's.

    Clamping off-CPU to zero and printing the row would read as "ran the whole time",
    which is the opposite of what an unusable mark means.
    """
    rows = [
        {
            "phase": "p",
            "dur_us": 10.0,
            "cpu_ns": 11_000,
            "rec_cpu_ns": 0,
            **dict.fromkeys(("minflt", "tminflt", "nvcsw", "nivcsw"), 0),
        }
    ]

    stats = phase_time_split.summarise(rows)

    assert stats["offcpu"] == 0.0
    assert stats["unmarked"] == 1


def test_a_sane_row_is_not_flagged():
    rows = [
        {
            "phase": "p",
            "dur_us": 10.0,
            "cpu_ns": 9_000,
            "rec_cpu_ns": 0,
            **dict.fromkeys(("minflt", "tminflt", "nvcsw", "nivcsw"), 0),
        }
    ]

    assert phase_time_split.summarise(rows)["unmarked"] == 0


def test_recorder_ratio_is_concurrency_not_a_share_of_the_wall():
    """reccpu above dur is the expected reading: eight workers overlap the bind thread."""
    rows = [
        {
            "phase": "p",
            "dur_us": 100.0,
            "cpu_ns": 50_000,
            "rec_cpu_ns": 400_000,
            **dict.fromkeys(("minflt", "tminflt", "nvcsw", "nivcsw"), 0),
        }
    ]

    stats = phase_time_split.summarise(rows)

    assert stats["reccpu"] == pytest.approx(400.0)
    assert stats["rec_ratio"] == pytest.approx(4.0)


def test_medians_are_taken_per_bind_not_across_columns():
    """offcpu is the median of the differences, which a difference of medians can miss."""
    rows = [
        {
            "phase": "p",
            "dur_us": 100.0,
            "cpu_ns": 100_000,
            "rec_cpu_ns": 0,
            **dict.fromkeys(("minflt", "tminflt", "nvcsw", "nivcsw"), 0),
        },
        {
            "phase": "p",
            "dur_us": 300.0,
            "cpu_ns": 100_000,
            "rec_cpu_ns": 0,
            **dict.fromkeys(("minflt", "tminflt", "nvcsw", "nivcsw"), 0),
        },
        {
            "phase": "p",
            "dur_us": 200.0,
            "cpu_ns": 200_000,
            "rec_cpu_ns": 0,
            **dict.fromkeys(("minflt", "tminflt", "nvcsw", "nivcsw"), 0),
        },
    ]

    # Medians: dur 200, cpu 100 -> a difference of medians would say 100.
    assert phase_time_split.summarise(rows)["offcpu"] == pytest.approx(0.0)


def test_a_log_without_the_cpu_counters_is_refused(tmp_path, monkeypatch):
    """Refusing beats printing zeros: a pre-counters log would read as all-on-CPU."""
    log = tmp_path / "run.log"
    log.write_text(
        "[STRACE] v=1 pid=7 tid=7 inv=1 hid=abc depth=2 name=chip.run.bind.host_orch ts=1000 dur=2000 "
        "minflt=1 nivcsw=0 nvcsw=0\n"
    )
    monkeypatch.setattr("sys.argv", ["phase_time_split", str(log)])

    with pytest.raises(SystemExit) as exit_info:
        phase_time_split.main()

    assert "cpu_ns" in str(exit_info.value)


def test_a_log_with_no_markers_at_all_is_refused(tmp_path, monkeypatch):
    log = tmp_path / "run.log"
    log.write_text("nothing here\n")
    monkeypatch.setattr("sys.argv", ["phase_time_split", str(log)])

    with pytest.raises(SystemExit) as exit_info:
        phase_time_split.main()

    assert "SIMPLER_HBG_BIND_BREAKDOWN_ENABLE" in str(exit_info.value)


def test_cold_and_warm_are_reported_separately(tmp_path, monkeypatch, capsys):
    log = tmp_path / "run.log"
    log.write_text(
        # Two ranks; each rank's earliest bind is its warm-up.
        _span("host_orch", 4_000_000, 3_000_000, 0, pid=7, inv=1, ts=1_000)
        + _span("host_orch", 4_000_000, 3_000_000, 0, pid=8, inv=1, ts=1_100)
        + _span("host_orch", 1_000_000, 900_000, 0, pid=7, inv=2, ts=9_000)
        + _span("host_orch", 1_000_000, 900_000, 0, pid=8, inv=2, ts=9_100)
    )
    monkeypatch.setattr("sys.argv", ["phase_time_split", str(log)])

    phase_time_split.main()

    rows = [line for line in capsys.readouterr().out.splitlines() if "host_orch" in line]
    assert len(rows) == 2
    assert "cold" in rows[0] and "4000.0" in rows[0]
    assert "warm" in rows[1] and "1000.0" in rows[1]
