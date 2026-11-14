#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Standard L1 collection streams beyond one buffer and survives repeated runs."""

import json
from collections import Counter
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.scene_test import compile_chip_callable_spec
from simpler_setup.tools.swimlane_converter import generate_chrome_trace_json, read_perf_data

pytestmark = [
    pytest.mark.platforms(["a5", "a5sim"]),
    pytest.mark.runtime("percounter"),
    pytest.mark.device_count(1),
    pytest.mark.manual,
]

# Exceeds one 1024-record buffer per AIC even on the widest supported A5.
BLOCKS = 16385
TASKS = 3
EXECUTIONS = BLOCKS * TASKS
STRIDE = 16


def build_callable(platform):
    kernels = Path(__file__).resolve().parents[3] / "host_build_graph/spmd_multiblock_aic/kernels"
    spec = {
        "orchestration": {
            "source": str(Path(__file__).resolve().parent / "kernels/orchestration/streaming_orch.cpp"),
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "source": str(kernels / "aic/kernel_stamp_block.cpp"),
                "core_type": "aic",
                "signature": [D.INOUT],
            },
        ],
    }
    return compile_chip_callable_spec(spec, platform, "percounter", (str(__file__), platform, "percounter"))


def test_l1_rotates_and_reuses_buffers(st_platform, st_device_ids, tmp_path):
    worker = Worker(level=2, platform=st_platform, runtime="percounter", device_id=int(st_device_ids[0]))
    try:
        handle = worker.register(build_callable(st_platform))
        worker.init()
        host = torch.zeros(EXECUTIONS * STRIDE, dtype=torch.float32)
        golden = torch.zeros_like(host)
        golden[::STRIDE] = torch.arange(BLOCKS, dtype=torch.float32).repeat(TASKS)
        golden[1::STRIDE] = BLOCKS
        buffer = worker.malloc(host.nbytes)
        try:
            args = TaskArgs()
            args.add_tensor(buffer.tensor(shapes=(EXECUTIONS * STRIDE,), dtype=DataType.FLOAT32), TensorArgType.INOUT)
            args.add_scalar(BLOCKS)
            epochs = set()
            for run, level in enumerate((1, 0, 1)):
                host.zero_()
                worker.copy_to(buffer, host)
                config = CallConfig()
                config.enable_chip_swimlane = level
                output = tmp_path / f"run{run}"
                output.mkdir()
                config.output_prefix = str(output)
                worker.run(handle, args, config)
                assert worker.scheduler_runtime_mode == 5
                worker.copy_from(host, buffer)
                torch.testing.assert_close(host, golden, rtol=0, atol=0)
                paths = list(output.rglob("chip_swimlane_records.json"))
                if level == 0:
                    assert not paths
                    continue
                assert len(paths) == 1
                capture = json.loads(paths[0].read_text())
                assert capture["metadata"]["runtime"] == "percounter"
                assert capture["chip_swimlane_level"] == 1
                rows = capture["aicore_tasks"]
                assert len(rows) == EXECUTIONS
                assert len({(r[0], r[2], r[6]) for r in rows}) == EXECUTIONS
                assert Counter(r[1] for r in rows) == {0: BLOCKS, 1: BLOCKS, 2: BLOCKS}
                assert all(r[4] >= r[3] > 0 for r in rows)
                assert max(Counter(r[0] for r in rows).values()) > 1024
                current_epochs = {r[6] for r in rows}
                assert len(current_epochs) == 1 and not epochs.intersection(current_epochs)
                epochs.update(current_epochs)
        finally:
            worker.free(buffer)
    finally:
        worker.close()


@pytest.mark.parametrize(
    "levels",
    [
        pytest.param((1, 0, 1), id="1"),
        pytest.param((2, 0, 2), id="2"),
        pytest.param((3, 0, 3), id="3"),
        pytest.param((4, 0, 4), id="4"),
        pytest.param((1, 3, 1, 4, 2, 0, 3), id="mixed"),
    ],
)
def test_profile_levels(st_platform, st_device_ids, tmp_path, levels):
    blocks = 17
    worker = Worker(level=2, platform=st_platform, runtime="percounter", device_id=int(st_device_ids[0]))
    try:
        handle = worker.register(build_callable(st_platform))
        worker.init()
        host = torch.zeros(blocks * TASKS * STRIDE, dtype=torch.float32)
        golden = torch.zeros_like(host)
        golden[::STRIDE] = torch.arange(blocks, dtype=torch.float32).repeat(TASKS)
        golden[1::STRIDE] = blocks
        buffer = worker.malloc(host.nbytes)
        try:
            args = TaskArgs()
            args.add_tensor(buffer.tensor(shapes=(host.numel(),), dtype=DataType.FLOAT32), TensorArgType.INOUT)
            args.add_scalar(blocks)
            for run, level in enumerate(levels):
                worker.copy_to(buffer, torch.zeros_like(host))
                config = CallConfig()
                config.enable_chip_swimlane = level
                output = tmp_path / f"run{run}"
                output.mkdir()
                config.output_prefix = str(output)
                worker.run(handle, args, config)
                assert worker.scheduler_runtime_mode == 5
                worker.copy_from(host, buffer)
                torch.testing.assert_close(host, golden, rtol=0, atol=0)
                paths = list(output.rglob("chip_swimlane_records.json"))
                if level == 0:
                    assert not paths
                    continue
                assert len(paths) == 1
                raw = json.loads(paths[0].read_text())
                assert raw["chip_swimlane_level"] == level
                rows = raw["aicore_tasks"]
                assert len(rows) == blocks * TASKS
                if level >= 2:
                    timing = raw["scheduler_tasks"]
                    assert timing["producer"] == "aicore"
                    assert {(r[0], r[1], r[4]) for r in timing["records"]} == {(r[0], r[2], r[6]) for r in rows}
                else:
                    assert not raw.get("scheduler_tasks", {}).get("records")
                if level >= 3:
                    streams = raw["scheduler_records"]["streams"]
                    assert streams and all(s["capture"]["dropped"] == 0 for s in streams)
                    kinds = {r["kind"] for s in streams for r in s["records"]}
                    assert {"scalar_window", "scalar_trace_finish", "scalar_bypass_load"} <= kinds
                    if level == 4:
                        assert "scalar_percounter_run_prepared" in kinds
                    data = read_perf_data(paths[0])
                    profiles = data["scalar_profiles"]
                    assert all(p["complete"] for p in profiles)
                    assert sum(p["kernel_cycles"] for p in profiles) == sum(r[4] - r[3] for r in rows)
                    assert all(sum(p["exclusive_cycles"].values()) == p["window_cycles"] for p in profiles)
                    assert all((p["function_min_cycles"] == 0) == (level == 4) for p in profiles)
                    trace = output / "swimlane.json"
                    generate_chrome_trace_json(
                        data["tasks"],
                        str(trace),
                        runtime_name="percounter",
                        scheduler_phases=data["scheduler_records"],
                        scheduler_streams=data["scheduler_streams"],
                        scalar_profiles=profiles,
                        scheduler_task_producer=data["scheduler_task_producer"],
                        chip_swimlane_level=level,
                    )
                    events = json.loads(trace.read_text())["traceEvents"]
                    assert any(e.get("pid") == 2 and e.get("name") == "trace_finish" for e in events)
                    assert not any(e.get("ph") == "X" and e.get("pid") == 3 for e in events)
                else:
                    assert not raw.get("scheduler_records", {}).get("streams")
                if level == 4:
                    assert raw.get("host_orchestrator_phases")
        finally:
            worker.free(buffer)
    finally:
        worker.close()
