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
@pytest.mark.parametrize(
    "platform,fail_mix,excerpt_fails",
    [("a2a3", False, False), ("a5", False, False), ("a5", True, False), ("a5", True, True)],
)
def test_dfx_smokes_adapt_to_device_count_without_overlap(
    tmp_path: Path, device_count: int, platform: str, fail_mix: bool, excerpt_fails: bool
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
            printf '%s\n' "$*" >> "$FAKE_STATE/task-submit"
            if [[ "$*" == "--list" ]]; then
              echo '=== synthetic queue ==='
              echo '  task_other secret-command-should-not-be-retained'
              exit 0
            fi
            run=
            while (( $# )); do
              if [[ "$1" == "--run" ]]; then
                shift
                run=$1
              fi
              shift
            done
            DEVICE_NUM=${DEVICE_NUM:-4}
            if [[ "$run" == "npu-smi info" ]]; then
              FAKE_PRIVILEGED=1 bash -c "$run"
            else
              TASK_DEVICE=$FAKE_TASK_DEVICE bash -c "$run"
            fi
            """
        )
    )
    task_submit.chmod(0o755)
    npu_smi = bin_dir / "npu-smi"
    npu_smi.write_text(
        "#!/usr/bin/env bash\n"
        'echo "$*" >> "$FAKE_STATE/npu-queries"\n'
        'echo "synthetic DCMI output"\n'
        'if [[ "$FAKE_FAIL_MIX" == 1 && "${FAKE_PRIVILEGED:-0}" != 1 ]]; then exit 1; fi\n'
    )
    npu_smi.chmod(0o755)

    python = bin_dir / "python"
    python.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            if [[ "${1:-}" == "-c" ]]; then
              if [[ "$2" == *"__build_commit__"* ]]; then
                printf '{"source":"synthetic","native":"synthetic"}\n'
                exit 0
              fi
              if [[ "$2" == *"DFX first-error excerpt"* && "$FAKE_EXCERPT_FAILS" == 1 ]]; then
                exit 93
              fi
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
            echo 'synthetic pytest log'
            if [[ "$FAKE_FAIL_MIX" == 1 && "$*" == *"/single_block_mix/"* ]]; then
              mkdir -p "$ASCEND_PROCESS_LOG_PATH/device-0" "$PWD/outputs/onboard-dfx-ci/failed/synthetic"
              device_log="$ASCEND_PROCESS_LOG_PATH/device-0/device-1.log"
              native_log="$PWD/outputs/onboard-dfx-ci/failed/synthetic/host.1.log"
              printf '[ERROR] synthetic device first-site task=7 site=42\n' > "$device_log"
              printf '[ERROR] synthetic native drain execution_rc=507015\n' > "$native_log"
              printf '%0200000d' 0 >> "$device_log"
              exit 1
            fi
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
            "FAKE_FAIL_MIX": str(int(fail_mix)),
            "FAKE_EXCERPT_FAILS": str(int(excerpt_fails)),
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

    assert result.returncode == int(fail_mix), result.stdout + result.stderr
    assert not (state_dir / "overlap").exists()
    invocations = (state_dir / "invocations").read_text().splitlines()
    expected_jobs = 10 if platform == "a5" else 6
    assert len(invocations) == expected_jobs
    assert all(
        f"--platform {platform}" in command and "--pto-session-timeout 600" in command for command in invocations
    )
    queue = (state_dir / "task-submit").read_text()
    assert "--timeout 1800 --max-time 1800" in queue
    program = [command for command in invocations if "/host_build_graph/dump_args/" in command]
    mix = [command for command in invocations if "/host_build_graph/single_block_mix/" in command]
    if platform == "a5":
        assert len(program) == 3 and len(mix) == 1
        for mode in ("full", "hybrid", "partial"):
            assert sum(f"--dump-args {mode}" in command for command in program) == 1
        assert "--dump-args full" in mix[0]
        assert all("--runtime host_build_graph" in command for command in program + mix)
    else:
        assert program == mix == []
    for device in range(7, 7 + device_count):
        expected_count = expected_jobs // device_count + (device - 7 < expected_jobs % device_count)
        assert sum(f"--device {device}" in invocation for invocation in invocations) == expected_count
    assert (state_dir / "parallel").exists() == (device_count > 1)
    retained = workspace / "outputs/onboard-dfx-ci" / platform
    assert (retained / "identity.json").exists()
    rows = (retained / "status.tsv").read_text().splitlines()
    assert len(rows) == expected_jobs
    assert len(list((retained / "logs").glob("*.log"))) == expected_jobs
    assert all("synthetic pytest log" in path.read_text() for path in (retained / "logs").glob("*.log"))
    assert sum(row.split("\t")[2] == "1" for row in rows) == int(fail_mix)
    assert all("--basetemp" in command for command in invocations)
    assert (retained / "allocated-devices.txt").read_text().strip() == env["FAKE_TASK_DEVICE"]
    assert "synthetic pytest log" in (retained / "task-client.log").read_text()
    for phase in ("before", "after"):
        assert (retained / f"npu-{phase}.txt").read_text() == "synthetic DCMI output\n"
        assert (retained / f"npu-{phase}.exit").read_text().strip() == str(int(fail_mix))
        assert "secret-command" not in (retained / f"queue-{phase}.txt").read_text()
        if fail_mix:
            assert (retained / f"npu-{phase}-privileged.exit").read_text().strip() == "0"
    queries = (state_dir / "npu-queries").read_text().splitlines()
    assert queries == ["info"] * (4 if fail_mix else 2)
    assert queue.count("--run npu-smi info") == (2 if fail_mix else 0)

    if fail_mix and not excerpt_fails:
        assert "synthetic device first-site task=7 site=42" in result.stdout
        assert "synthetic native drain execution_rc=507015" in result.stdout
        assert "DFX first-error excerpt" in result.stdout
        assert len(result.stdout) < 150_000
    if excerpt_fails:
        assert "DFX failure excerpt unavailable" in result.stdout
