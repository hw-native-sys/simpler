#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise timeout transport and initialization latching through real Workers."""

import os
import re
import time
from pathlib import Path

import pytest
import torch
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    CoreCallable,
    DataType,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

HERE = Path(__file__).parent
RUNTIME = "tensormap_and_ringbuffer"
ENV = "SIMPLER_TENSOR_DATA_TIMEOUT_MS"
PRODUCER_SLEEP_MS = 1000
_TIMEOUT_RE = re.compile(r"failed with code -8\b")


def _build_callable(platform):
    compiler = KernelCompiler(platform=platform)
    binary = compiler.compile_incore(
        source_path=str(HERE / "finite_producer.cpp"),
        core_type="aiv",
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=compiler.get_orchestration_include_dirs(RUNTIME),
    )
    if not platform.endswith("sim"):
        binary = extract_text_section(binary)
    return ChipCallable.build(
        signature=[ArgDirection.OUT],
        func_name="aicpu_orchestration_entry",
        binary=compiler.compile_orchestration(runtime_name=RUNTIME, source_path=str(HERE / "tensor_wait_orch.cpp")),
        children=[(0, CoreCallable.build(signature=[ArgDirection.OUT], binary=binary))],
    )


def _run(worker, handle, expect_timeout, *, device_id=None, budget_ms=None, label=""):
    host = torch.zeros(1, dtype=torch.int32)
    buffer = worker.malloc(host.nbytes)
    try:
        worker.copy_to(buffer, host)
        args = TaskArgs()
        args.add_tensor(buffer.tensor(shapes=(1,), dtype=DataType.INT32), TensorArgType.OUTPUT_EXISTING)
        args.add_scalar(PRODUCER_SLEEP_MS)
        config = CallConfig()
        # Scalar waits require the scheduler to run on a separate thread.
        config.aicpu_thread_num = 2
        t0 = time.perf_counter()
        raised = None
        try:
            worker.run(handle, args, config)
        except BaseException as exc:  # noqa: BLE001 — surface any outcome in diagnostics
            raised = exc
        wall_ms = (time.perf_counter() - t0) * 1000.0
        detail = (
            f"{label} device_id={device_id} budget_ms={budget_ms} "
            f"env[{ENV}]={os.environ.get(ENV)!r} producer_sleep_ms={PRODUCER_SLEEP_MS} "
            f"wall_ms={wall_ms:.1f} raised={raised!r}"
        )
        if expect_timeout:
            if raised is None:
                pytest.fail(f"expected code -8 timeout but run succeeded; {detail}")
            if not isinstance(raised, RuntimeError) or _TIMEOUT_RE.search(str(raised)) is None:
                pytest.fail(f"expected code -8 timeout but got unexpected error; {detail}")
            return
        if raised is not None:
            raise raised
        worker.copy_from(host, buffer)
        assert host.item() == 7, detail
    finally:
        worker.free(buffer)


@pytest.mark.platforms(["a2a3sim", "a5sim", "a2a3", "a5"])
@pytest.mark.runtime(RUNTIME)
@pytest.mark.device_count(1)
@pytest.mark.timeout(90)
@pytest.mark.parametrize("initial", ["250", "3000", "invalid", None])
def test_tensor_timeout_is_latched_at_init(st_platform, st_device_ids, monkeypatch, initial):
    _exercise_latched_timeout(st_platform, st_device_ids, monkeypatch, initial)


def _exercise_latched_timeout(st_platform, st_device_ids, monkeypatch, initial, *, enable_sdma=False):
    if initial is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, initial)
    # Invalid ordering must reject only scheduler/op/stream overrides on NPU.
    monkeypatch.setenv("SIMPLER_SCHEDULER_TIMEOUT_MS", "90000")
    monkeypatch.setenv("SIMPLER_OP_EXECUTE_TIMEOUT_US", "45000000")
    monkeypatch.setenv("SIMPLER_STREAM_SYNC_TIMEOUT_MS", "50000")
    worker = Worker(
        level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]), enable_sdma=enable_sdma
    )
    try:
        handle = worker.register(_build_callable(st_platform))
        worker.init()
        monkeypatch.setenv(ENV, "3000" if initial == "250" else "250")
        device_id = int(st_device_ids[0])
        _run(
            worker,
            handle,
            initial == "250",
            device_id=device_id,
            budget_ms=initial,
            label=f"latched[initial={initial!r}]",
        )
        if initial != "250":
            _run(
                worker,
                handle,
                False,
                device_id=device_id,
                budget_ms=initial,
                label=f"latched-success[initial={initial!r}]",
            )
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim", "a5sim"])
@pytest.mark.runtime(RUNTIME)
@pytest.mark.device_count(2)
@pytest.mark.timeout(90)
def test_tensor_timeout_is_isolated_between_workers(st_platform, st_device_ids, monkeypatch):
    chip_callable = _build_callable(st_platform)
    workers = []
    handles = []
    budgets = ["3000", "250"]
    try:
        for device_id, budget in zip(st_device_ids, budgets, strict=True):
            monkeypatch.setenv(ENV, budget)
            worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(device_id))
            workers.append(worker)
            handles.append(worker.register(chip_callable))
            worker.init()
        _run(
            workers[0],
            handles[0],
            False,
            device_id=int(st_device_ids[0]),
            budget_ms=budgets[0],
            label="isolated workers[0]",
        )
        _run(
            workers[1],
            handles[1],
            True,
            device_id=int(st_device_ids[1]),
            budget_ms=budgets[1],
            label="isolated workers[1]",
        )
    finally:
        for worker in reversed(workers):
            worker.close()


@pytest.mark.sdma
@pytest.mark.platforms(["a2a3"])
@pytest.mark.runtime(RUNTIME)
@pytest.mark.device_count(1)
@pytest.mark.timeout(120)
@pytest.mark.parametrize("initial", ["250", "3000"])
def test_tensor_timeout_with_dma_initialization(st_platform, st_device_ids, monkeypatch, initial):
    _exercise_latched_timeout(st_platform, st_device_ids, monkeypatch, initial, enable_sdma=True)
