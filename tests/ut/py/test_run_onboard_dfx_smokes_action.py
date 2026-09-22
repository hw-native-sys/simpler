# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
ACTION = REPO_ROOT / ".github/actions/run-onboard-dfx-smokes/action.yml"


def _run_script() -> str:
    marker = "      run: |\n"
    action = ACTION.read_text()
    assert action.count(marker) == 1
    return textwrap.dedent(action.split(marker, 1)[1])


@pytest.mark.parametrize("device_count", [1, 2, 3, 4])
def test_a5_dfx_smokes_adapt_to_device_count_without_overlap(tmp_path: Path, device_count: int) -> None:
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
            "DFX_PLATFORM": "a5",
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
    expected_jobs = 5
    assert len(invocations) == expected_jobs
    for device in range(7, 7 + device_count):
        expected_count = expected_jobs // device_count + (device - 7 < expected_jobs % device_count)
        assert sum(f"--device {device}" in invocation for invocation in invocations) == expected_count
    assert (state_dir / "parallel").exists() == (device_count > 1)
