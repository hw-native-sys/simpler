# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional

import pytest

REPO_ROOT = Path(__file__).parents[3]
ACTION = REPO_ROOT / ".github/actions/run-onboard-dfx-smokes/action.yml"


def _run_script() -> str:
    marker = "      run: |\n"
    action = ACTION.read_text()
    assert action.count(marker) == 1
    return textwrap.dedent(action.split(marker, 1)[1])


@pytest.mark.parametrize("device_count", [1, 2, 3, 4])
@pytest.mark.parametrize("platform", ["a2a3", "a5"])
def test_onboard_dfx_smokes_adapt_to_device_count_without_overlap(
    tmp_path: Path, device_count: int, platform: str
) -> None:
    workspace = tmp_path / "workspace"
    bin_dir = tmp_path / "bin"
    state_dir = tmp_path / "state"
    runner_temp = tmp_path / "runner"
    for directory in (workspace / ".venv/bin", bin_dir, state_dir, runner_temp):
        directory.mkdir(parents=True)

    (workspace / ".venv/bin/activate").write_text(":\n")
    cann_env = tmp_path / "set_env.sh"
    cann_env.write_text(":\n")
    task_submit = bin_dir / "task-submit"
    task_submit.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            run=
            while (( $# )); do
              if [[ "$1" == "--run" ]]; then
                shift
                run=$1
              fi
              shift
            done
            DEVICE_NUM=${DEVICE_NUM:-4}
            TASK_DEVICE=$FAKE_TASK_DEVICE bash -c "$run"
            """
        )
    )
    task_submit.chmod(0o755)

    python = bin_dir / "python"
    python.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            if [[ "${1:-}" == "-c" ]]; then
              exec "$REAL_PYTHON" "$@"
            fi
            device=
            previous=
            for argument in "$@"; do
              if [[ "$previous" == "--device" ]]; then
                device=$argument
                break
              fi
              previous=$argument
            done
            lock="$FAKE_STATE/device-$device"
            if ! mkdir "$lock" 2>/dev/null; then
              echo "$device" >> "$FAKE_STATE/overlap"
              exit 91
            fi
            printf '%s\n' "$*" >> "$FAKE_STATE/invocations"
            running="$FAKE_STATE/running-$device"
            touch "$running"
            running_count=0
            for running_job in "$FAKE_STATE"/running-*; do
              if [[ -e "$running_job" ]]; then
                ((running_count += 1))
              fi
            done
            if (( running_count > 1 )); then
              touch "$FAKE_STATE/parallel"
            fi
            sleep 0.05
            rm -f "$running"
            rmdir "$lock"
            """
        )
    )
    python.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "CANN_ENV": str(cann_env),
            "DEVICE_RANGE": f"7-{6 + device_count}",
            "DEVICE_NUM": str(device_count),
            "DFX_PLATFORM": platform,
            "FAKE_STATE": str(state_dir),
            "FAKE_TASK_DEVICE": ",".join(str(device) for device in range(7, 7 + device_count)),
            "GITHUB_WORKSPACE": str(workspace),
            "PATH": f"{bin_dir}:{env['PATH']}",
            "REAL_PYTHON": sys.executable,
            "RUNNER_TEMP": str(runner_temp),
        }
    )
    result = subprocess.run(
        ["bash", "-c", _run_script()],
        cwd=workspace,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not (state_dir / "overlap").exists()
    invocations = (state_dir / "invocations").read_text().splitlines()
    # A5 adds three selection modes for task-time HBG payloads and a Mix arm.
    expected_jobs = 10 if platform == "a5" else 6
    assert len(invocations) == expected_jobs
    assert all("--manual include" in invocation for invocation in invocations)
    counting_window = [invocation for invocation in invocations if "test_pmu_counting_window.py" in invocation]
    assert len(counting_window) == (1 if platform == "a2a3" else 0)
    assert all("--enable-pmu 1 " in invocation for invocation in counting_window)
    if platform == "a5":
        hbg_args = [line for line in invocations if "host_build_graph/dump_args/" in line]
        assert len(hbg_args) == 3
        assert {line.split("--dump-args ", 1)[1].split()[0] for line in hbg_args} == {"full", "hybrid", "partial"}
        assert any("host_build_graph/single_block_mix/" in line and "--dump-args full" in line for line in invocations)
    for device in range(7, 7 + device_count):
        expected_count = expected_jobs // device_count + (device - 7 < expected_jobs % device_count)
        assert sum(f"--device {device}" in invocation for invocation in invocations) == expected_count
    assert (state_dir / "parallel").exists() == (device_count > 1)


@pytest.mark.parametrize("failed_mode", [None, "full", "hybrid"])
def test_hbg_dump_modes_wait_before_starting_successors(tmp_path: Path, failed_mode: Optional[str]) -> None:
    driver = tmp_path / "run-onboard-dfx-smokes-a5.sh"
    driver.write_text(_run_script().split("<<'DFX_DRIVER'\n", 1)[1].split("\nDFX_DRIVER", 1)[0])
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """\
            import json
            import os
            import sys

            args = sys.argv[1:]
            with open(os.environ["DFX_INVOCATIONS"], "a") as output:
                output.write(json.dumps(args) + "\\n")
            if any("host_build_graph/dump_args/" in arg for arg in args):
                mode = args[args.index("--dump-args") + 1]
                if mode == os.environ.get("DFX_FAILED_MODE"):
                    sys.exit(17 if mode == "full" else 23)
            """
        )
    )
    python.chmod(0o755)
    trace = tmp_path / "trace.sh"
    trace.write_text(
        textwrap.dedent(
            r"""            # Observe real driver calls and completed waits, without delaying jobs.
            _dfx_trace() {
              case "$1" in
                'local label=$1')
                  printf 'start\t%s\n' "$2" >> "$DFX_TRACE" ;;
                'job_pids[$index]=""')
                  printf 'waited\t%s\t%s\n' "${job_labels[$index]}" "${job_rcs[$index]}" >> "$DFX_TRACE" ;;
              esac
            }
            set -T
            trap '_dfx_trace "$BASH_COMMAND" "$@"' DEBUG
            """
        )
    )
    events_path = tmp_path / "events"
    invocations_path = tmp_path / "invocations"
    result = subprocess.run(
        ["bash", str(driver), "a5", "7,8,9,10"],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "BASH_ENV": str(trace),
            "DFX_TRACE": str(events_path),
            "DFX_INVOCATIONS": str(invocations_path),
            "DFX_FAILED_MODE": failed_mode or "",
        },
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    (tmp_path / "driver.stdout").write_text(result.stdout)
    (tmp_path / "driver.stderr").write_text(result.stderr)
    assert result.returncode == (1 if failed_mode else 0), result.stdout + result.stderr
    invocations = [json.loads(line) for line in invocations_path.read_text().splitlines()]
    assert len(invocations) == 10
    hbg_args = [args for args in invocations if any("host_build_graph/dump_args/" in arg for arg in args)]
    assert {args[args.index("--dump-args") + 1] for args in hbg_args} == {"full", "hybrid", "partial"}
    assert {args[args.index("--dump-args") + 1]: args[args.index("--device") + 1] for args in hbg_args} == {
        "full": "8",
        "hybrid": "9",
        "partial": "10",
    }
    events = [line.split("\t") for line in events_path.read_text().splitlines()]
    labels = [f"args_dump (host_build_graph, {mode})" for mode in ("full", "hybrid", "partial")]
    for current, following in zip(labels[:-1], labels[1:]):
        waited = next(i for i, event in enumerate(events) if event[:2] == ["waited", current])
        started = events.index(["start", following])
        assert waited < started, events
    for mode, label in zip(("full", "hybrid", "partial"), labels):
        rc = {"full": 17, "hybrid": 23}.get(mode, 0) if mode == failed_mode else 0
        assert ["waited", label, str(rc)] in events
        assert f"::group::DFX {label} (exit={rc})" in result.stdout
