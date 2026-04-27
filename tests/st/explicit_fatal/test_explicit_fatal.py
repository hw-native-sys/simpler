#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Negative ST for explicit orchestration fatal reporting."""

import os

import pytest
from simpler.task_interface import CallConfig, ChipCallable, ChipStorageTaskArgs
from simpler.worker import Worker

from simpler_setup.kernel_compiler import KernelCompiler

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = "tensormap_and_ringbuffer"
ORCH_SRC = os.path.join(HERE, "kernels/orchestration/explicit_fatal_orch.cpp")


def _build_chip_callable(platform: str) -> ChipCallable:
    kc = KernelCompiler(platform=platform)
    orch_bytes = kc.compile_orchestration(runtime_name=RUNTIME, source_path=ORCH_SRC)
    return ChipCallable.build(
        signature=[],
        func_name="aicpu_orchestration_entry",
        binary=orch_bytes,
        children=[],
    )


@pytest.mark.platforms(["a5sim", "a2a3sim"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
def test_explicit_fatal_reports(st_platform, st_device_ids, monkeypatch):
    monkeypatch.setenv("PTO_LOG_LEVEL", "error")

    chip_callable = _build_chip_callable(st_platform)
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    worker.init()
    try:
        config = CallConfig()
        config.block_dim = 24
        config.aicpu_thread_num = 4
        with pytest.raises(RuntimeError, match=r"run_runtime failed with code -9"):
            worker.run(chip_callable, ChipStorageTaskArgs(), config)
    finally:
        worker.close()
