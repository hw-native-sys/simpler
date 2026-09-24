# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Grouping bind-segment spans into binds.

A bind is the set of segments sharing one `(pid, inv)`. The tests below pin that
key, and in particular the case the previous line-based grouping could not
express: two ranks writing one stream, whose segment bursts interleave.
"""

from __future__ import annotations

from simpler_setup.tools.hbg.bind_phases import cold_binds, parse_binds

SEGMENTS = ("args", "host_orch", "graph_upload", "arena_h2d")


def _span(pid: int, inv: int, phase: str, ts: int, dur_ns: int) -> str:
    return f"[STRACE] v=1 pid={pid} tid={pid} inv={inv} hid=abc depth=2 name=chip.run.bind.{phase} ts={ts} dur={dur_ns}"


def _write(path, lines: list[str]):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [path]


def test_two_ranks_writing_one_stream_are_not_confused_for_each_other(tmp_path):
    """The case the old grouping could not see.

    Ranks share the capture fd and their bursts interleave, and the log prefix's
    thread id is identical across them. `inv` is allocated per run per process,
    so `(pid, inv)` separates them with nothing inferred from order.
    """
    lines = []
    for index, phase in enumerate(SEGMENTS):
        # One segment from each rank, alternating, for two binds per rank.
        lines.append(_span(11, 1, phase, 1_000 + index, 1_000_000))
        lines.append(_span(22, 1, phase, 1_500 + index, 2_000_000))
        lines.append(_span(11, 2, phase, 9_000 + index, 3_000_000))
        lines.append(_span(22, 2, phase, 9_500 + index, 4_000_000))

    binds = parse_binds(_write(tmp_path / "run.log", lines))

    assert len(binds) == 4
    assert {(b["pid"], b["inv"]) for b in binds} == {(11, 1), (11, 2), (22, 1), (22, 2)}
    for bind in binds:
        assert sorted(k for k in bind if k in SEGMENTS) == sorted(SEGMENTS)
    by_key = {(b["pid"], b["inv"]): b for b in binds}
    assert by_key[(11, 1)]["host_orch"] == 1.0
    assert by_key[(22, 2)]["host_orch"] == 4.0


def test_each_rank_contributes_exactly_one_cold_bind(tmp_path):
    """Warm-up is per rank and is the earliest bind that rank ran, not a count."""
    lines = []
    for pid, base in ((11, 1_000), (22, 500)):
        for inv, offset in ((1, 0), (2, 10_000), (3, 20_000)):
            for index, phase in enumerate(SEGMENTS):
                lines.append(_span(pid, inv, phase, base + offset + index, 1_000_000))

    binds = parse_binds(_write(tmp_path / "run.log", lines))

    assert cold_binds(binds) == {(11, 1), (22, 1)}


def test_a_group_without_host_orch_is_not_a_bind(tmp_path):
    """`host_orch` is the segment that makes a bind a bind; without it the
    records describe something else that happened to share the key."""
    lines = [
        _span(11, 1, "args", 1_000, 1_000_000),
        _span(11, 1, "arena_h2d", 1_100, 1_000_000),
        _span(11, 2, "args", 2_000, 1_000_000),
        _span(11, 2, "host_orch", 2_100, 1_000_000),
    ]

    binds = parse_binds(_write(tmp_path / "run.log", lines))

    assert [(b["pid"], b["inv"]) for b in binds] == [(11, 2)]


def test_orchestrator_operations_are_not_segments(tmp_path):
    """`chip.run.bind.host_orch.<op>` sits a level deeper and is not a segment of
    the stage, so it must not become a column."""
    lines = [
        _span(11, 1, "host_orch", 1_000, 1_000_000),
        _span(11, 1, "host_orch.graph_submit", 1_100, 50_000),
        _span(11, 1, "host_orch.build_definition", 1_200, 60_000),
    ]

    binds = parse_binds(_write(tmp_path / "run.log", lines))

    assert len(binds) == 1
    assert sorted(k for k in binds[0] if k not in ("pid", "inv", "ts")) == ["host_orch"]
