# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Queued per-call DFX selections retain their own exported records."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_EXAMPLE = Path(__file__).resolve().parents[4] / "examples/workers/l2/framework_eager/diagnostic_selection.py"
pytestmark = [pytest.mark.requires_hardware, pytest.mark.platforms(["a2a3"]), pytest.mark.test_environment("torch_npu")]


@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
@pytest.mark.parametrize("task_queue", ["0", "1"])
def test_framework_diagnostic_selection(runtime, task_queue, framework_module, tmp_path, request):
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    logs = tmp_path / "ascend"
    logs.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(_EXAMPLE),
            "--runtime",
            runtime,
            "--device",
            device,
            "--build-dir",
            str(framework_module),
            "--output-dir",
            str(tmp_path / "diagnostics"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        env={
            **os.environ,
            "TASK_QUEUE_ENABLE": task_queue,
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
            "MAX_JOBS": "2",
            "ASCEND_PROCESS_LOG_PATH": str(logs),
        },
    )
    (tmp_path / "consumer.log").write_text(result.stdout + result.stderr)
    print("FRAMEWORK_DIAGNOSTICS_CONSUMER_LOG " + str(tmp_path / "consumer.log"), flush=True)
    print(result.stdout, end="", flush=True)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [
        json.loads(line.removeprefix("FRAMEWORK_DIAGNOSTIC_SELECTION_PASS "))
        for line in result.stdout.splitlines()
        if line.startswith("FRAMEWORK_DIAGNOSTIC_SELECTION_PASS ")
    ]
    assert len(records) == 1, result.stdout + result.stderr
    summary = records[0]
    assert summary["runtime"] == runtime and summary["task_queue"] == task_queue
    assert summary["verified"] and summary["observed"] == [8.0, 12.0, 16.0, 20.0]
    assert summary["calls"] == 4 and summary["selections"] == [[0, 0], [1, 2], [3, 4], [0, 0]]
    assert summary["caller_configs_mutated"]
    assert len(summary["run_epochs"]) == len(set(summary["run_epochs"])) == 2
    assert all(epoch > 0 for epoch in summary["run_epochs"])
    assert len(summary["callback_threads"]) == 4
    assert all((thread != summary["caller_thread"]) == (task_queue == "1") for thread in summary["callback_threads"])
