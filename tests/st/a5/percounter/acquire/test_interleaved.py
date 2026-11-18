#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent AIC/MIX/AIV sync cohorts with no tensor dependency serialization."""

import json
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.scene_test import compile_chip_callable_spec
from tests.st.a5.percounter.acquire.test_cohorts import MAX_LANES, STRIDE

pytestmark = [pytest.mark.platforms(["a5", "a5sim"]), pytest.mark.runtime("percounter"), pytest.mark.device_count(1)]


def build_callable(platform):
    kernels = Path(__file__).resolve().parents[2] / "host_build_graph/spmd_cohort_saturation/kernels"
    spec = {
        "orchestration": {
            "source": str(Path(__file__).resolve().parent / "kernels/orchestration/independent_interleaved.cpp"),
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT, D.INOUT, D.IN, D.IN],
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
    return compile_chip_callable_spec(spec, platform, "percounter", (__file__, platform, "percounter"))


@pytest.mark.parametrize("partial", [False, True], ids=["full", "partial"])
def test_independent_interleaved(st_platform, st_device_ids, tmp_path, partial, assert_cohorts_were_coresident):
    worker = Worker(level=2, platform=st_platform, runtime="percounter", device_id=int(st_device_ids[0]))
    buffers = []
    try:
        handle = worker.register(build_callable(st_platform))
        worker.init()
        host = torch.full((2 * MAX_LANES * STRIDE,), -1.0)
        args = TaskArgs()
        for _ in range(3):
            buffer = worker.malloc(host.nbytes)
            buffers.append(buffer)
            args.add_tensor(buffer.tensor(shapes=tuple(host.shape), dtype=DataType.FLOAT32), TensorArgType.INOUT)
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
            config.enable_dep_gen = True
            config.output_prefix = str(output)
            worker.run(handle, args, config)
            assert worker.scheduler_runtime_mode == 5
            captures = list(output.rglob("chip_swimlane_records.json"))
            assert len(captures) == 1
            raw = json.loads(captures[0].read_text())
            kinds = raw["metadata"]["core_types"]
            blocks = [kinds.count("aic"), kinds.count("aic"), kinds.count("aiv")]
            if partial:
                blocks = [max(1, count // 2) for count in blocks]
            widths = []
            assert len(buffers) == len(blocks) == 3
            for buffer, count, lanes in zip(buffers, blocks, [1, 3, 1]):
                width = count * lanes
                widths.append(width)
                worker.copy_from(host, buffer)
                expected = torch.full_like(host, -1)
                expected[: width * STRIDE : STRIDE] = 1
                expected[MAX_LANES * STRIDE : (MAX_LANES + width) * STRIDE : STRIDE] = torch.arange(
                    count, dtype=host.dtype
                ).repeat_interleave(lanes)
                assert torch.equal(host, expected)
            rows = raw["aicore_tasks"]
            executing = {int(row[1]) for row in rows}
            assert len(executing) == 3
            for task in executing:
                cores = [row[0] for row in rows if row[1] == task]
                assert len(cores) == len(set(cores))
            deps = json.loads((captures[0].parent / "deps.json").read_text())
            assert not [
                edge for edge in deps["edges"] if int(edge["pred"]) in executing and int(edge["succ"]) in executing
            ]
            assert_cohorts_were_coresident(output, widths, f"independent partial={partial}, repeat={repeat}")
    finally:
        try:
            for buffer in buffers:
                worker.free(buffer)
        finally:
            worker.close()
