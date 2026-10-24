# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real framework source/context preflight refusal followed by healthy work."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_EXAMPLE = Path(__file__).resolve().parents[4] / "examples/workers/l2/framework_eager/preflight_recovery.py"
pytestmark = [pytest.mark.requires_hardware, pytest.mark.platforms(["a2a3"]), pytest.mark.test_environment("torch_npu")]


@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
@pytest.mark.parametrize("task_queue", ["0", "1"])
def test_framework_preflight_recovery(runtime, task_queue, framework_module, tmp_path, request):
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
    print("FRAMEWORK_PREFLIGHT_CONSUMER_LOG " + str(tmp_path / "consumer.log"), flush=True)
    print(result.stdout, end="", flush=True)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [
        json.loads(line.removeprefix("FRAMEWORK_PREFLIGHT_RECOVERY_PASS "))
        for line in result.stdout.splitlines()
        if line.startswith("FRAMEWORK_PREFLIGHT_RECOVERY_PASS ")
    ]
    assert len(records) == 1, result.stdout + result.stderr
    summary = records[0]
    assert summary["runtime"] == runtime and summary["task_queue"] == task_queue
    assert summary["verified"] and summary["calls"] == 3 and summary["observed"] == [8.0, 12.0, 16.0]
    assert summary["refusals"] == ["source", "context"] and summary["source_registration_replaced"]
    assert summary["caller_context_preserved"] and summary["alternate_destroyed"]
    assert summary["caller_stream_after_close"] and summary["caller_tensors_retained"]
    assert len(summary["callback_threads"]) == 3
    assert all((thread != summary["caller_thread"]) == (task_queue == "1") for thread in summary["callback_threads"])
