# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Live framework-queue consumer of the ordinary public Worker."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_EXAMPLE = _ROOT / "examples/workers/l2/framework_eager/lifecycle.py"
pytestmark = [
    pytest.mark.requires_hardware,
    pytest.mark.platforms(["a2a3", "a5"]),
    pytest.mark.test_environment("torch_npu"),
]


@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
@pytest.mark.parametrize("task_queue", ["0", "1"])
def test_framework_worker_close_recreate(runtime, task_queue, framework_module, tmp_path, request):
    _run_lifecycle(runtime, runtime, task_queue, framework_module, tmp_path, request)


@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
@pytest.mark.parametrize("task_queue", ["0", "1"])
def test_framework_worker_runtime_transition(runtime, task_queue, framework_module, tmp_path, request):
    successor = "host_build_graph" if runtime == "tensormap_and_ringbuffer" else "tensormap_and_ringbuffer"
    _run_lifecycle(runtime, successor, task_queue, framework_module, tmp_path, request)


def _run_lifecycle(runtime, successor, task_queue, framework_module, tmp_path, request):
    platform = request.config.getoption("--platform")
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    logs = tmp_path / "ascend"
    logs.mkdir()
    command = [
        sys.executable,
        "-u",
        str(_EXAMPLE),
        "--platform",
        platform,
        "--runtime",
        runtime,
        "--device",
        device,
        "--build-dir",
        str(framework_module),
    ]
    if successor != runtime:
        command += ["--successor-runtime", successor]
    result = subprocess.run(
        command,
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
    print("FRAMEWORK_CONSUMER_LOG " + str(tmp_path / "consumer.log"), flush=True)
    print(result.stdout, end="", flush=True)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [
        line.removeprefix("FRAMEWORK_LIFECYCLE_PASS ")
        for line in result.stdout.splitlines()
        if line.startswith("FRAMEWORK_LIFECYCLE_PASS ")
    ]
    assert len(records) == 1, result.stdout + result.stderr
    summary = json.loads(records[0])
    assert summary["verified"] and summary["calls"] == 8 and summary["generations"] == 2
    assert summary["platform"] == platform
    assert summary["runtime"] == runtime and summary["task_queue"] == task_queue
    assert summary["runtimes"] == [runtime, successor]
    assert summary["orchestration_load_counts"] == [
        [2, 0] if backend == "host_build_graph" else [0, 2] for backend in (runtime, successor)
    ]
    assert summary["observed"] == summary["expected"] == [3, 7, 13, 17, 22, 34, 41, 57]
    assert summary["generation_calls"] == [4, 4]
    assert summary["original_handles_preserved"] and summary["guards_verified"]
    assert summary["caller_tensors_retained"] and summary["caller_stream_after_each_close"]
    assert all(context == summary["caller_context"] for context in summary["callback_contexts"])
    assert all((thread != summary["caller_thread"]) == (task_queue == "1") for thread in summary["callback_threads"])
