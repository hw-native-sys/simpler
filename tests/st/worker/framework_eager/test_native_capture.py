# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cold recording through the matched original framework graph owner."""

import runpy
import sys
from pathlib import Path

import pytest

_EXAMPLE = Path(__file__).resolve().parents[4] / "examples/workers/l2/framework_eager/native_capture.py"
pytestmark = [
    pytest.mark.runtime_product("kernel"),
    pytest.mark.execution_mode("capture_replay"),
    pytest.mark.requires_hardware,
    pytest.mark.platforms(["a2a3", "a5"]),
    pytest.mark.test_environment("torch_npu"),
]


def _run_capture(runtime, task_queue, framework_module, tmp_path, request, monkeypatch, *extra_args):
    driver = Path(__file__).with_name("native_capture.py")
    bridge = next(framework_module.glob("_simpler_framework_queue*.so"))
    platform = request.config.getoption("--platform")
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    monkeypatch.setenv("TASK_QUEUE_ENABLE", task_queue)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(driver),
            "--runtime",
            runtime,
            "--platform",
            platform,
            "--device",
            device,
            "--bridge",
            str(bridge),
            "--output",
            str(tmp_path / "capture"),
            *extra_args,
        ],
    )
    # The original driver checks the child exit and its exact numerical/cleanup report.
    runpy.run_path(str(driver), run_name="__main__")


@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
@pytest.mark.parametrize("task_queue", ["0", "1"])
def test_native_cold_capture(runtime, task_queue, framework_module, tmp_path, request, monkeypatch):
    _run_capture(runtime, task_queue, framework_module, tmp_path, request, monkeypatch)


@pytest.mark.coverage_runtime("tensormap_and_ringbuffer")
def test_native_cold_capture_destroy_refusal(framework_module, tmp_path, request, monkeypatch):
    _run_capture("tensormap_and_ringbuffer", "0", framework_module, tmp_path, request, monkeypatch, "--destroy-refusal")
