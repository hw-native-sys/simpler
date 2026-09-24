# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for torch backend autoload state in bind-phase reports."""

from __future__ import annotations

import sys

from simpler_setup.tools.hbg import bind_phases


def _span(pid: int, inv: int, phase: str, ts: int, dur_ns: int) -> str:
    return f"[STRACE] v=1 pid={pid} tid={pid} inv={inv} hid=abc depth=2 name=chip.run.bind.{phase} ts={ts} dur={dur_ns}"


def _write_log(path, autoload_records: tuple[str, ...]) -> None:
    """One rank, two binds — enough that the cold one leaves a warm one behind."""
    lines = ["[stamp] command commit=abc"]
    lines.extend(f"TIMING simpler: {record}" for record in autoload_records)
    lines.extend(
        [
            _span(7, 1, "host_orch", 1_000, 1_000_000),
            _span(7, 1, "arena_h2d", 1_001_000, 1_000_000),
            _span(7, 2, "host_orch", 3_000_000, 500_000),
            _span(7, 2, "arena_h2d", 3_500_000, 500_000),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_report_shows_each_torch_autoload_state(tmp_path, monkeypatch, capsys):
    log = tmp_path / "run.log"
    first = (
        'torch_backend_autoload setting=invalid raw="a b\\"c" raw_truncated=false '
        "effective=disabled torch_imported=true torch_npu_loaded=false"
    )
    second = (
        'torch_backend_autoload setting=1 raw="1" raw_truncated=false '
        "effective=enabled torch_imported=true torch_npu_loaded=true"
    )
    _write_log(log, (first, first, second))
    monkeypatch.setattr(sys, "argv", ["hbg_bind_phases", str(log)])

    assert bind_phases.main() == 0

    output = capsys.readouterr().out
    assert output.count(first) == 1
    assert output.count(second) == 1


def test_report_warns_when_torch_autoload_state_is_missing(tmp_path, monkeypatch, capsys):
    log = tmp_path / "run.log"
    _write_log(log, ())
    monkeypatch.setattr(sys, "argv", ["hbg_bind_phases", str(log)])

    assert bind_phases.main() == 0

    output = capsys.readouterr().out
    assert "no `torch_backend_autoload` record" in output
    assert "must be established before comparing" in output


def test_parser_accepts_record_without_raw_fields(tmp_path):
    log = tmp_path / "run.log"
    record = "torch_backend_autoload setting=0 effective=disabled torch_imported=true torch_npu_loaded=false"
    _write_log(log, (record,))

    assert bind_phases.parse_torch_autoload([log]) == [record]
