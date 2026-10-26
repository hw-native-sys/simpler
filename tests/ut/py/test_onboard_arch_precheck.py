# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU-only CLI contracts for the real onboard silicon precheck."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_CHECK = Path(__file__).resolve().parents[3] / ".claude/skills/onboard-arch-precheck/check.sh"


def _run_check(tmp_path, chip, npu, requested, soc, short, *, board_status=0, privileged_board=None):
    binary = tmp_path / "bin"
    binary.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    cann = tmp_path / "cann"
    config = cann / "x86_64-linux/data/platform_config"
    config.mkdir(parents=True)
    if soc is not None:
        (config / f"{soc}.ini").write_text(f"SoC_version={soc}\nShort_SoC_version={short}\n")
    # Synthetic board layout: CI retained only chip='910B2C', npu='', not its raw output.
    board = f"Chip Name : {chip}\n" + (f"NPU Name : {npu}\n" if npu is not None else "")
    log = tmp_path / "commands.jsonl"
    for name, output, status in (
        ("npu-smi", board if board_status == 0 else "DCMI permission denied", board_status),
        ("task-submit", privileged_board or "unexpected privileged query", 0 if privileged_board else 19),
    ):
        command = binary / name
        command.write_text(
            f"#!{sys.executable}\nimport json,sys\n"
            f"with open({str(log)!r}, 'a') as out: out.write(json.dumps([{name!r}, *sys.argv[1:]])+'\\n')\n"
            f"print({output!r})\nsys.exit({status})\n"
        )
        command.chmod(0o755)
    result = subprocess.run(
        ["bash", str(_CHECK), requested],
        env={**os.environ, "PATH": f"{binary}:/usr/bin:/bin", "TMPDIR": str(cache), "ASCEND_HOME_PATH": str(cann)},
        text=True,
        capture_output=True,
        check=False,
    )
    commands = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, commands, board, cache


@pytest.mark.parametrize(
    "chip,npu,soc,short,actual",
    [
        ("Ascend910", "B3", "Ascend910B3", "Ascend910B", "a2a3"),
        ("Ascend910", "9392", "Ascend910_9392", "Ascend910_93", "a2a3"),
        ("Ascend950", "9586", "Ascend950DT_9586", "Ascend950", "a5"),
        ("Ascend950DT", "9586", "Ascend950DT_9586", "Ascend950", "a5"),
        ("Ascend950PR", "9579", "Ascend950PR_9579", "Ascend950", "a5"),
        ("910B2C", None, "Ascend910B2C", "Ascend910B", "a2a3"),
        ("Ascend910B2C", "", "Ascend910B2C", "Ascend910B", "a2a3"),
        ("Ascend950DT_9586", None, "Ascend950DT_9586", "Ascend950", "a5"),
    ],
)
@pytest.mark.parametrize("requested", ["a2a3", "a5"])
def test_actual_precheck_maps_only_the_installed_soc(tmp_path, chip, npu, soc, short, actual, requested):
    result, commands, _, cache = _run_check(tmp_path, chip, npu, requested, soc, short)
    assert result.returncode == (0 if requested == actual else 2), result.stdout + result.stderr
    assert commands == [["npu-smi", "info", "-t", "board", "-i", "0", "-c", "0"]]
    assert next(cache.glob("*.cache")).read_text() == f"{actual}|{soc}|{short}\n"
    if requested != actual:
        assert "ARCH MISMATCH" in result.stderr
        assert f"SoC={soc}, Short_SoC_version={short}" in result.stderr


@pytest.mark.parametrize(
    "chip,npu,soc,short",
    [
        ("910B2C", None, None, None),
        ("unknown-chip", None, "Ascend910B2C", "Ascend910B"),
        ("", None, "Ascend910B2C", "Ascend910B"),
        ("Ascend910", "B3", None, None),
        ("Ascend910B2C", "unexpected", "Ascend910B2C", "Ascend910B"),
        ("910B2C", None, "Ascend910B2C", "unknown-short"),
        ("910B2C", None, "Ascend910B2C", ""),
        ("../Ascend910B2C", None, "Ascend910B2C", "Ascend910B"),
    ],
)
def test_unusable_successful_board_query_is_refused_and_retained(tmp_path, chip, npu, soc, short):
    result, commands, board, cache = _run_check(tmp_path, chip, npu, "a2a3", soc, short)
    assert result.returncode == 1, result.stdout + result.stderr
    assert board in result.stderr
    assert "npu-smi info -t board -i 0 -c 0" in result.stderr
    assert len(commands) == 1, "exit-zero board output must not trigger the privilege retry"
    assert not list(cache.glob("*.cache"))


def test_nonzero_dcmi_query_retries_through_the_privileged_queue(tmp_path):
    result, commands, _, _ = _run_check(
        tmp_path,
        "",
        None,
        "a2a3",
        "Ascend910B3",
        "Ascend910B",
        board_status=7,
        privileged_board="task-submit: completed\nChip Name : Ascend910\nNPU Name : B3\n",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert commands == [
        ["npu-smi", "info", "-t", "board", "-i", "0", "-c", "0"],
        ["task-submit", "--run", "npu-smi info -t board -i 0 -c 0"],
    ]


def test_all_nonzero_queries_preserve_exit_statuses_and_output(tmp_path):
    result, commands, _, cache = _run_check(tmp_path, "", None, "a2a3", None, None, board_status=7)
    assert result.returncode == 1
    assert len(commands) == 4
    assert "(exit=7)" in result.stderr and "DCMI permission denied" in result.stderr
    assert "(exit=19)" in result.stderr and "unexpected privileged query" in result.stderr
    assert not list(cache.glob("*.cache"))


@pytest.mark.parametrize("requested", ["a2a3sim", "a5sim"])
def test_simulation_does_not_query_hardware(tmp_path, requested):
    result, commands, _, cache = _run_check(tmp_path, "", None, requested, None, None)
    assert result.returncode == 0
    assert commands == []
    assert not list(cache.glob("*.cache"))
