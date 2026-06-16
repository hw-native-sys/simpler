# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for simpler_setup.tools.device_log_timing (device-log Total/Orch/Sched parse)."""

import os
import pwd
from pathlib import Path

import pytest

from simpler_setup.tools.device_log_resolver import get_log_root, resolve_device_log_paths
from simpler_setup.tools.device_log_timing import format_device_log_timing, main, parse_device_log_timing

# One round's worth of CANN dlog lines, in the order the runtime emits them.
# orch span = 26959 cyc / orch_cost 539.180us -> 50 cyc/us; the sched lines
# bound the Total span (min start .. max end across orch+sched).
_PREFIX = "[INFO] AICPU(1,aicpu_scheduler):2026-06-09-11:42:20.000.000 [device_log.cpp:89]2 run [V9] "


def _round_block(base, orch_cost, sched_cost, tasks):
    """Render one orch + 3 sched + 3 summary block at cycle offset *base*.

    freq is fixed at 50 cyc/us so cost_us * 50 = cycle span.
    """
    orch_cyc = int(orch_cost * 50)
    sched_cyc = int(sched_cost * 50)
    lines = [
        f'{_PREFIX}"[aicpu_executor.cpp:691] Thread 3: '
        f'orch_start={base + 100} orch_end={base + 100 + orch_cyc} orch_cost={orch_cost:.3f}us"',
        f'{_PREFIX}"[aicpu_executor.cpp:696] PTO2 total submitted tasks = {tasks}, already executed {tasks} tasks"',
    ]
    for tid in (0, 1, 2):
        lines.append(
            f'{_PREFIX}"[scheduler_cold_path.cpp:405] Thread {tid}: '
            f'sched_start={base} sched_end={base + sched_cyc} sched_cost={sched_cost:.3f}us"'
        )
        lines.append(
            f'{_PREFIX}"[scheduler_cold_path.cpp:530] Thread {tid}: '
            f'Scheduler summary: total_time={sched_cost - 5:.3f}us, loops=700, tasks_scheduled=180"'
        )
    return lines


def _write_log(tmp_path, blocks):
    path = tmp_path / "device-1234_20260609114225453.log"
    lines = ["[INFO] CCECPU(1,aicpu_scheduler) some unrelated framework line"]
    cyc = 1_000_000
    for orch_cost, sched_cost, tasks in blocks:
        lines += _round_block(cyc, orch_cost, sched_cost, tasks)
        cyc += 100_000  # next round starts well after the previous
    path.write_text("\n".join(lines) + "\n")
    return path


def test_single_round(tmp_path):
    log = _write_log(tmp_path, [(539.18, 931.78, 355)])
    rounds = parse_device_log_timing(log)
    assert len(rounds) == 1
    r = rounds[0]
    assert r.tasks == 355
    assert abs(r.orch_us - 539.18) < 0.05
    assert abs(r.sched_us - 931.78) < 0.05
    # Total spans min(start) .. max(end): sched_start (base) .. sched_end.
    assert abs(r.total_us - 931.78) < 0.5
    # Total covers the orch span too, so it is never below either component.
    assert r.total_us >= r.orch_us


def test_multi_round_grouping(tmp_path):
    blocks = [(500.0, 900.0, 355), (520.0, 950.0, 355), (480.0, 880.0, 355)]
    log = _write_log(tmp_path, blocks)
    rounds = parse_device_log_timing(log)
    assert len(rounds) == 3
    assert [round(r.orch_us) for r in rounds] == [500, 520, 480]


def test_sched_timeout_variant_parsed(tmp_path):
    # The timeout path prints `sched_end(timeout)=` instead of `sched_end=`.
    line = (
        f'{_PREFIX}"[scheduler_cold_path.cpp:405] Thread 0: '
        f'sched_start=1000000 sched_end(timeout)=1045000 sched_cost=900.000us"'
    )
    orch = f'{_PREFIX}"[aicpu_executor.cpp:691] Thread 3: orch_start=1000100 orch_end=1025100 orch_cost=500.000us"'
    log = tmp_path / "device-1_x.log"
    log.write_text(orch + "\n" + line + "\n")
    rounds = parse_device_log_timing(log)
    assert len(rounds) == 1
    assert abs(rounds[0].sched_us - 900.0) < 0.5


def test_empty_log_reports_no_blocks(tmp_path):
    log = tmp_path / "device-1_empty.log"
    log.write_text("[INFO] CCECPU nothing relevant here\n")
    rounds = parse_device_log_timing(log)
    assert rounds == []
    out = format_device_log_timing(rounds)
    assert "no orch/sched timing blocks" in out


def test_format_includes_per_round_and_average(tmp_path):
    log = _write_log(tmp_path, [(500.0, 900.0, 355), (520.0, 950.0, 355)])
    out = format_device_log_timing(parse_device_log_timing(log), source=log)
    assert "Total (us)" in out and "Orch (us)" in out and "Sched (us)" in out
    assert "Avg" in out
    assert "tasks = 355" in out


def test_parse_accepts_rotated_file_list(tmp_path):
    # CANN rotates at 20 MB: a long run's rounds split across files. Passing the
    # files (in mtime order) must concatenate them into one continuous sequence.
    f1 = tmp_path / "device-1_a.log"
    f2 = tmp_path / "device-1_b.log"
    f1.write_text("\n".join(_round_block(1_000_000, 500.0, 900.0, 355)) + "\n")
    f2.write_text("\n".join(_round_block(2_000_000, 520.0, 950.0, 355)) + "\n")
    rounds = parse_device_log_timing([f1, f2])
    assert len(rounds) == 2
    assert [round(r.orch_us) for r in rounds] == [500, 520]


def test_offsets_skip_pre_run_content(tmp_path):
    # A reused-process log: an earlier case's block, then this run's block. With
    # an offset at the pre-run byte size, only this run's block must be parsed.
    path = tmp_path / "device-9_reused.log"
    stale = "\n".join(_round_block(1_000_000, 999.0, 1500.0, 111)) + "\n"
    path.write_text(stale)
    offset = path.stat().st_size
    with path.open("a") as f:
        f.write("\n".join(_round_block(2_000_000, 500.0, 900.0, 355)) + "\n")

    # Without the offset both blocks are seen.
    assert len(parse_device_log_timing(path)) == 2
    # With the offset only the appended (this-run) block is seen.
    fresh = parse_device_log_timing(path, offsets={str(path): offset})
    assert len(fresh) == 1
    assert round(fresh[0].orch_us) == 500
    assert fresh[0].tasks == 355


def test_parse_skips_missing_file(tmp_path):
    # A path that vanished (rotation) must be skipped, not raise.
    good = tmp_path / "device-1_a.log"
    good.write_text("\n".join(_round_block(1_000_000, 500.0, 900.0, 355)) + "\n")
    missing = tmp_path / "device-1_gone.log"
    rounds = parse_device_log_timing([missing, good])
    assert len(rounds) == 1


def test_resolve_glob_returns_all_rotated_files(tmp_path):
    # An explicit glob must return every match (mtime order) so a rotated run's
    # earlier files aren't dropped; a single explicit file returns one element.
    d = tmp_path / "device-0"
    d.mkdir()
    a, b = d / "device-1_a.log", d / "device-1_b.log"
    a.write_text("x")
    b.write_text("y")
    os.utime(a, (1, 1))
    os.utime(b, (2, 2))  # b newer than a
    logs, strat = resolve_device_log_paths(device_log=str(d / "device-*.log"))
    assert [p.name for p in logs] == ["device-1_a.log", "device-1_b.log"]
    assert "2 files" in strat

    logs1, _ = resolve_device_log_paths(device_log=str(a))
    assert logs1 == [a]


def test_cli_main_reads_log(tmp_path, capsys):
    log = _write_log(tmp_path, [(500.0, 900.0, 355)])
    rc = main(["--device-log", str(log)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Total (us)" in out and "tasks = 355" in out


def test_cli_main_requires_an_arg(capsys):
    with pytest.raises(SystemExit):
        main([])


def test_cli_main_errors_on_stdout_logging(tmp_path, monkeypatch, capsys):
    log = _write_log(tmp_path, [(500.0, 900.0, 355)])
    monkeypatch.setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1")
    rc = main(["--device-log", str(log)])
    assert rc == 1
    assert "ASCEND_SLOG_PRINT_TO_STDOUT=1" in capsys.readouterr().err


def test_get_log_root_env_precedence(tmp_path, monkeypatch):
    # ASCEND_PROCESS_LOG_PATH/debug wins over ASCEND_WORK_PATH/log/debug.
    proc = tmp_path / "proc"
    work = tmp_path / "work"
    (proc / "debug").mkdir(parents=True)
    (work / "log" / "debug").mkdir(parents=True)

    monkeypatch.setenv("ASCEND_WORK_PATH", str(work))
    monkeypatch.delenv("ASCEND_PROCESS_LOG_PATH", raising=False)
    assert get_log_root() == work / "log" / "debug"

    monkeypatch.setenv("ASCEND_PROCESS_LOG_PATH", str(proc))
    assert get_log_root() == proc / "debug"

    # A set-but-nonexistent root falls through to the next candidate.
    monkeypatch.setenv("ASCEND_PROCESS_LOG_PATH", str(tmp_path / "nope"))
    assert get_log_root() == work / "log" / "debug"


def test_get_log_root_default_uses_euid_home_not_home_env(monkeypatch):
    # With no relocation env var, the default root derives from the effective
    # uid's passwd home (where the driver writes device logs), NOT $HOME — sudo /
    # task-submit leave $HOME pointing at the invoking user, so trusting it would
    # look in the wrong directory and the device log would never be found.
    monkeypatch.delenv("ASCEND_PROCESS_LOG_PATH", raising=False)
    monkeypatch.delenv("ASCEND_WORK_PATH", raising=False)
    monkeypatch.setenv("HOME", "/wrong/home/left/by/sudo")

    euid_home = pwd.getpwuid(os.geteuid()).pw_dir
    assert get_log_root() == Path(euid_home) / "ascend" / "log" / "debug"


def test_get_log_root_default_falls_back_when_no_passwd_entry(tmp_path, monkeypatch):
    # An arbitrary container uid (e.g. `docker --user 1234`) may have no passwd
    # entry; getpwuid raises KeyError and we fall back to Path.home() ($HOME)
    # rather than crashing.
    monkeypatch.delenv("ASCEND_PROCESS_LOG_PATH", raising=False)
    monkeypatch.delenv("ASCEND_WORK_PATH", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # Path.home() reads $HOME

    def _raise(_uid):
        raise KeyError("no passwd entry for uid")

    monkeypatch.setattr(pwd, "getpwuid", _raise)
    assert get_log_root() == tmp_path / "ascend" / "log" / "debug"
