# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check that A5 CI retains diagnostics without hiding a failed task."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[3]
_WORKFLOW = _ROOT / ".github/workflows/_st-npu-a5.yml"


@pytest.mark.parametrize("task_status", [0, 19])
def test_a5_task_logs_and_status_survive_tee(tmp_path, task_status):
    workflow = yaml.load(_WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    steps = workflow["jobs"]["run"]["steps"]
    run = next(step for step in steps if step["name"] == "Run pytest scene tests (a5)")
    prepare = next(step for step in steps if step["name"] == "Prepare A5 diagnostic logs")
    upload = next(step for step in steps if step["name"] == "Retain A5 onboard diagnostic logs")
    log_dir = tmp_path / "logs with spaces"
    env = {
        **os.environ,
        "A5_DIAGNOSTIC_DIR": str(log_dir),
        "ASCEND_PROCESS_LOG_PATH": str(log_dir / "ascend"),
        "SESSION_TIMEOUT_S": "1200",
        "MANUAL_MODE": "exclude",
        "DEVICE_NUM": "2",
        "TASK_STATUS": str(task_status),
        "ARGUMENTS": str(tmp_path / "arguments.json"),
    }
    subprocess.run(["bash", "-e", "-c", prepare["run"]], cwd=_ROOT, env=env, check=True)
    assert (log_dir / "source.txt").read_text().strip()
    assert (log_dir / "ascend").is_dir()
    binary = tmp_path / "bin"
    binary.mkdir()
    task = binary / "task-submit"
    task.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "with open(os.environ['ARGUMENTS'],'w') as out: json.dump(sys.argv[1:],out)\n"
        "print('task stdout')\nprint('task stderr',file=sys.stderr)\n"
        "sys.exit(int(os.environ['TASK_STATUS']))\n"
    )
    task.chmod(0o755)
    cann = tmp_path / "set_env.sh"
    cann.write_text(":\n")
    body = run["run"].replace("source /usr/local/Ascend/cann/set_env.sh", f'source "{cann}"')
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", body],
        env={**env, "PATH": f"{binary}:{env['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert run["shell"] == "bash"
    assert result.returncode == task_status, result.stdout + result.stderr
    assert "task stdout" in (log_dir / "pytest-task.log").read_text()
    assert "task stderr" in (log_dir / "pytest-task.log").read_text()
    args = json.loads((tmp_path / "arguments.json").read_text())
    assert args[args.index("--timeout") + 1] == "3300"
    assert args[args.index("--max-time") + 1] == "1800"
    assert args[args.index("--device") + 1] == "auto"
    assert args[args.index("--device-num") + 1] == "2"
    assert any(args[i : i + 2] == ["--env", "ASCEND_PROCESS_LOG_PATH"] for i in range(len(args)))
    assert "--device $TASK_DEVICE" in args[args.index("--run") + 1]
    assert upload["if"] == "always()"
    assert upload["with"]["path"] == "${{ env.A5_DIAGNOSTIC_DIR }}"
