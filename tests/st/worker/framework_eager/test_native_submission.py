# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native submit reaches real framework drain boundaries without requiring GIL."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
_EXAMPLE = ROOT / "examples/workers/l2/framework_eager/native_submission.py"
pytestmark = [
    pytest.mark.requires_hardware,
    pytest.mark.platforms(["a2a3", "a5"]),
    pytest.mark.test_environment("torch_npu"),
]


@pytest.fixture(scope="session")
def native_submit_probe(tmp_path_factory):
    directory = tmp_path_factory.mktemp("native-submit-probe")
    built = subprocess.run(
        [sys.executable, str(ROOT / "tools/framework_queue_gil/probe.py"), "build", "--output", str(directory)],
        env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "MAX_JOBS": "2"},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    path = Path(built.stdout.splitlines()[-1])
    assert path.is_file(), built.stdout
    return path


@pytest.mark.parametrize("runtime", ["host_build_graph", "tensormap_and_ringbuffer"])
@pytest.mark.parametrize("queue", ["0", "1"])
def test_native_submission(runtime, queue, framework_module, native_submit_probe, tmp_path, request):
    platform = request.config.getoption("--platform")
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    logs = tmp_path / "ascend"
    logs.mkdir()
    bridge = next(framework_module.glob("_simpler_framework_queue*.so"))
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(_EXAMPLE),
            "--runtime",
            runtime,
            "--platform",
            platform,
            "--device",
            device,
            "--bridge",
            str(bridge),
            "--probe",
            str(native_submit_probe),
        ],
        env={
            **os.environ,
            "TASK_QUEUE_ENABLE": queue,
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
            "ASCEND_PROCESS_LOG_PATH": str(logs),
        },
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    reports = [
        line.removeprefix("NATIVE_SUBMISSION_PASS ")
        for line in result.stdout.splitlines()
        if line.startswith("NATIVE_SUBMISSION_PASS ")
    ]
    assert len(reports) == 1, result.stdout
    report = json.loads(reports[0])
    assert report == {
        "runtime": runtime,
        "platform": platform,
        "task_queue": queue,
        "calls": 3,
        "expected": [8, 22, 50],
        "cancelled_refused": True,
        "duplicate_refused": True,
    }
