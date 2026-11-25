#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Full-width AIC/MIX/AIV entry promises, with interval-based co-residency checks."""

import json
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.scene_test import compile_chip_callable_spec

pytestmark = [
    pytest.mark.platforms(["a5", "a5sim"]),
    pytest.mark.runtime("host_build_graph"),
    pytest.mark.device_count(1),
]
STRIDE = 16
MAX_LANES = 320


def build_callable(platform, masks=False):
    kernels = Path(__file__).resolve().parents[2] / "host_build_graph/spmd_cohort_saturation/kernels"
    spec = {
        "orchestration": {
            "source": str(kernels / "orchestration/spmd_cohort_saturation_orch.cpp"),
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.IN, D.IN],
        },
        "incores": [
            {
                "func_id": i,
                "source": str(kernels / name),
                "core_type": "aic" if i == 0 else "aiv",
                "signature": [D.INOUT, D.IN, D.IN, D.IN, D.IN],
            }
            for i, name in enumerate(
                (
                    "aic/kernel_cohort_barrier.cpp",
                    "aiv/kernel_cohort_barrier.cpp",
                    "aiv/kernel_cohort_barrier_lane1.cpp",
                    "aiv/kernel_cohort_barrier_lane2.cpp",
                )
            )
        ],
    }
    if masks:
        spec["orchestration"]["source"] = str(
            Path(__file__).resolve().parent / "kernels/orchestration/mask_cohorts.cpp"
        )
        spec["orchestration"]["signature"] = [D.INOUT, D.INOUT, D.INOUT, D.IN, D.IN, D.IN]
    return compile_chip_callable_spec(
        spec, platform, "host_build_graph", (__file__, platform, "host_build_graph", masks)
    )


@pytest.mark.parametrize("cohorts", [3, 12])
def test_full_width_interleaved(st_platform, st_device_ids, tmp_path, cohorts, assert_cohorts_were_coresident):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=int(st_device_ids[0]))
    try:
        handle = worker.register(build_callable(st_platform))
        worker.init()
        host = torch.full((cohorts * 2 * MAX_LANES * STRIDE,), -1.0)
        buffer = worker.malloc(host.nbytes)
        try:
            args = TaskArgs()
            args.add_tensor(buffer.tensor(shapes=tuple(host.shape), dtype=DataType.FLOAT32), TensorArgType.INOUT)
            args.add_scalar(cohorts)
            args.add_scalar(1000)
            args.add_scalar(0)
            for repeat in range(3):
                host.fill_(-1)
                worker.copy_to(buffer, host)
                output = tmp_path / f"repeat{repeat}"
                output.mkdir()
                config = CallConfig()
                config.enable_chip_swimlane = 1
                config.output_prefix = str(output)
                worker.run(handle, args, config)
                assert worker.scheduler_runtime_mode == 5
                worker.copy_from(host, buffer)
                captures = list(output.rglob("chip_swimlane_records.json"))
                assert len(captures) == 1
                raw = json.loads(captures[0].read_text())
                kinds = raw["metadata"]["core_types"]
                cubes = kinds.count("aic")
                vectors = kinds.count("aiv")
                expected = []
                for index in range(cohorts):
                    blocks = vectors if index % 3 == 2 else cubes
                    lanes = 3 if index % 3 == 1 else 1
                    width = blocks * lanes
                    expected.append(width)
                    start = index * 2 * MAX_LANES
                    arrivals = host[start * STRIDE : (start + width) * STRIDE : STRIDE]
                    results = host[(start + MAX_LANES) * STRIDE : (start + MAX_LANES + width) * STRIDE : STRIDE]
                    assert torch.equal(arrivals, torch.ones_like(arrivals))
                    assert torch.equal(results, torch.arange(blocks, dtype=host.dtype).repeat_interleave(lanes))
                assert_cohorts_were_coresident(output, expected, f"calendar {cohorts} cohorts, repeat {repeat}")
        finally:
            worker.free(buffer)
    finally:
        worker.close()


@pytest.mark.parametrize("mask", [1, 2, 3, 5, 6, 7])
@pytest.mark.parametrize("partial", [False, True], ids=["full", "partial"])
def test_mask_cohorts(st_platform, st_device_ids, tmp_path, mask, partial, assert_cohorts_were_coresident):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=int(st_device_ids[0]))
    buffers = []
    try:
        handle = worker.register(build_callable(st_platform, masks=True))
        worker.init()
        host = torch.full((2 * MAX_LANES * STRIDE,), -1.0)
        args = TaskArgs()
        for _ in range(3):
            buffer = worker.malloc(host.nbytes)
            buffers.append(buffer)
            args.add_tensor(buffer.tensor(shapes=tuple(host.shape), dtype=DataType.FLOAT32), TensorArgType.INOUT)
        args.add_scalar(mask)
        args.add_scalar(int(partial))
        args.add_scalar(1000)
        for repeat in range(3):
            host.fill_(-1)
            for buffer in buffers:
                worker.copy_to(buffer, host)
            output = tmp_path / f"repeat{repeat}"
            output.mkdir()
            config = CallConfig()
            config.enable_chip_swimlane = 1
            config.output_prefix = str(output)
            worker.run(handle, args, config)
            assert worker.scheduler_runtime_mode == 5
            captures = list(output.rglob("chip_swimlane_records.json"))
            assert len(captures) == 1
            raw = json.loads(captures[0].read_text())
            kinds = raw["metadata"]["core_types"]
            blocks = kinds.count("aiv" if mask == 2 else "aic")
            if partial:
                blocks = max(1, blocks // 2)
            lanes = bin(mask).count("1")
            width = blocks * lanes
            for buffer in buffers:
                worker.copy_from(host, buffer)
                arrivals = host[: width * STRIDE : STRIDE]
                results = host[MAX_LANES * STRIDE : (MAX_LANES + width) * STRIDE : STRIDE]
                assert torch.equal(arrivals, torch.ones_like(arrivals))
                assert torch.equal(results, torch.arange(blocks, dtype=host.dtype).repeat_interleave(lanes))
            assert_cohorts_were_coresident(output, [width] * 3, f"mask={mask}, partial={partial}, repeat={repeat}")
    finally:
        try:
            for buffer in buffers:
                worker.free(buffer)
        finally:
            worker.close()


def build_entry_order_callable(platform):
    acquire = Path(__file__).resolve().parent
    hbg = acquire.parent
    spec = {
        "orchestration": {
            "source": str(acquire / "kernels/orchestration/cohort_entry_order.cpp"),
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": str(hbg / "spmd_cohort_with_ordinary/kernels/aic/kernel_busy_mark.cpp"),
                "core_type": "aic",
                "signature": [D.INOUT, D.IN, D.IN],
            },
            {
                "func_id": 1,
                "source": str(hbg / "spmd_cohort_full_width/kernels/aic/kernel_stamp_block_resident.cpp"),
                "core_type": "aic",
                "signature": [D.INOUT],
            },
        ],
    }
    return compile_chip_callable_spec(spec, platform, "host_build_graph", (__file__, platform, "cohort_entry_order"))


def test_cohort_block_can_return_before_its_peer_enters(st_platform, st_device_ids, tmp_path):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=int(st_device_ids[0]))
    try:
        handle = worker.register(build_entry_order_callable(st_platform))
        worker.init()
        slow = torch.full((STRIDE,), -1.0)
        blocks = torch.full((MAX_LANES, STRIDE), -1.0)
        slow_buffer = worker.malloc(slow.nbytes)
        try:
            block_buffer = worker.malloc(blocks.nbytes)
            try:
                worker.copy_to(slow_buffer, slow)
                worker.copy_to(block_buffer, blocks)
                args = TaskArgs()
                args.add_tensor(
                    slow_buffer.tensor(shapes=tuple(slow.shape), dtype=DataType.FLOAT32), TensorArgType.INOUT
                )
                args.add_tensor(
                    block_buffer.tensor(shapes=tuple(blocks.shape), dtype=DataType.FLOAT32), TensorArgType.INOUT
                )
                config = CallConfig()
                config.enable_chip_swimlane = 1
                config.output_prefix = str(tmp_path)
                worker.run(handle, args, config)
                assert worker.scheduler_runtime_mode == 5
                worker.copy_from(slow, slow_buffer)
                worker.copy_from(blocks, block_buffer)
            finally:
                worker.free(block_buffer)
        finally:
            worker.free(slow_buffer)

        (capture,) = tmp_path.rglob("chip_swimlane_records.json")
        raw = json.loads(capture.read_text())
        cubes = raw["metadata"]["core_types"].count("aic")
        assert cubes >= 2
        assert slow[0].item() == 1
        assert torch.equal(blocks[:cubes, 0], torch.arange(cubes, dtype=blocks.dtype))
        assert torch.all(blocks[:cubes, 1] == cubes)
        rows = raw["aicore_tasks"]
        (blocker,) = (row for row in rows if row[1] == 0)
        cohort = [row for row in rows if row[1] == 1]
        assert len(cohort) == len({row[0] for row in cohort}) == cubes
        (delayed,) = (row for row in cohort if row[0] == blocker[0])
        assert min(row[4] for row in cohort) < blocker[4] <= delayed[3]
    finally:
        worker.close()
