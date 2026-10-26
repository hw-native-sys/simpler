# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Per-task argument-dump data or explicit unsupported refusal and recovery."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_EXAMPLE = Path(__file__).resolve().parents[4] / "examples/workers/l2/framework_eager/argument_capture.py"
pytestmark = [
    pytest.mark.requires_hardware,
    pytest.mark.platforms(["a2a3", "a5"]),
    pytest.mark.test_environment("torch_npu"),
]


@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
@pytest.mark.parametrize("task_queue", ["0", "1"])
def test_framework_argument_dump_contract(runtime, task_queue, framework_module, tmp_path, request):
    platform = request.config.getoption("--platform")
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    logs = tmp_path / "ascend"
    logs.mkdir()
    result = subprocess.run(
        [
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
            "--output-dir",
            str(tmp_path / "captures"),
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
    print("FRAMEWORK_ARGUMENT_CAPTURE_CONSUMER_LOG " + str(tmp_path / "consumer.log"), flush=True)
    print(result.stdout, end="", flush=True)
    assert result.returncode == 0, result.stdout + result.stderr
    args_dump_supported = not (platform == "a5" and runtime == "host_build_graph")
    marker = (
        "FRAMEWORK_ARGUMENT_CAPTURE_PASS " if args_dump_supported else "FRAMEWORK_ARGUMENT_DUMP_REFUSAL_RECOVERY_PASS "
    )
    records = [
        line
        for line in result.stdout.splitlines()
        if line.startswith(("FRAMEWORK_ARGUMENT_CAPTURE_PASS ", "FRAMEWORK_ARGUMENT_DUMP_REFUSAL_RECOVERY_PASS "))
    ]
    assert len(records) == 1 and records[0].startswith(marker)
    summary = json.loads(records[0].removeprefix(marker))
    assert summary["platform"] == platform
    assert summary["runtime"] == runtime and summary["task_queue"] == task_queue
    assert summary["verified"]
    assert summary["levels"] == [0, 3, 2, 0] and summary["shape"] == [128, 128]
    assert summary["guards_verified"] and summary["config_snapshot_verified"]
    assert summary["args_dump_supported"] is args_dump_supported
    if args_dump_supported:
        assert summary["calls"] == summary["accepted_calls"] == 4 and summary["refusals"] == []
        assert summary["payload_counts"] == [0, 6, 3, 0] and summary["records_per_enabled_call"] == 6
        assert summary["payloads_verified"]
        epochs = summary["run_epochs"]
        assert len(epochs) == 4 and epochs[0] is None and epochs[3] is None
        assert all(type(epoch) is int and epoch > 0 for epoch in epochs[1:3]) and epochs[1] != epochs[2]
    else:
        assert summary["calls"] == summary["accepted_calls"] == 2
        assert summary["payloads_verified"] is False
        assert summary["payload_counts"] == [0, 0, 0, 0] and summary["records_per_enabled_call"] == 0
        assert summary["run_epochs"] == [None] * 4
        assert summary["off_calls_verified"] and summary["refusal_outputs_unchanged"]
        assert summary["caller_tensors_retained"] and summary["caller_stream_after_close"]
        assert [(item["index"], item["level"]) for item in summary["refusals"]] == [(1, 3), (2, 2)]
        assert all(
            re.fullmatch(
                r"prepare_native_run failed with code -1001 "
                r"\(run_id=\d+ slot=\d+ generation=\d+ dispatch_id=\d+ run_epoch=\d+\)",
                item["reason"],
            )
            for item in summary["refusals"]
        )
    assert len(summary["callback_threads"]) == 4
    assert all((thread != summary["caller_thread"]) == (task_queue == "1") for thread in summary["callback_threads"])
