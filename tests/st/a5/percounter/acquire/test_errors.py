#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A rejected cohort never launches; the same width remains valid ordinary work."""

from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.scene_test import compile_chip_callable_spec

pytestmark = [pytest.mark.platforms(["a5", "a5sim"]), pytest.mark.runtime("percounter"), pytest.mark.device_count(1)]


def build_callable(platform):
    workloads = Path(__file__).resolve().parents[2] / "host_build_graph"
    spec = {
        "orchestration": {
            "source": str(workloads / "spmd_cohort_too_wide/kernels/orchestration/spmd_cohort_too_wide_orch.cpp"),
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.IN, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "source": str(workloads / "spmd_multiblock_aic/kernels/aic/kernel_stamp_block.cpp"),
                "core_type": "aic",
                "signature": [D.INOUT],
            }
        ],
    }
    return compile_chip_callable_spec(spec, platform, "percounter", (__file__, platform, "percounter"))


def test_overwide_cohort_rejection_leaves_ordinary_run_usable(st_platform, st_device_ids, capfd, drain_host_log):
    worker = Worker(level=2, platform=st_platform, runtime="percounter", device_id=int(st_device_ids[0]))
    buffer = None
    try:
        handle = worker.register(build_callable(st_platform))
        worker.init()
        blocks, stride = 64, 16
        host = torch.full((blocks * stride,), -1.0)
        buffer = worker.malloc(host.nbytes)
        worker.copy_to(buffer, host)
        for sync in [True, False]:
            args = TaskArgs()
            args.add_tensor(buffer.tensor(shapes=tuple(host.shape), dtype=DataType.FLOAT32), TensorArgType.INOUT)
            args.add_scalar(blocks)
            args.add_scalar(int(sync))
            if sync:
                with pytest.raises(RuntimeError):
                    worker.run(handle, args, CallConfig())
                log = drain_host_log(capfd)
                assert "REQUIRE_SYNC_START_INVALID" in log
                assert "require_sync_start block_num=64" in log
                worker.copy_from(host, buffer)
                assert torch.equal(host, torch.full_like(host, -1))
            else:
                worker.run(handle, args, CallConfig())
                assert worker.scheduler_runtime_mode == 5
                worker.copy_from(host, buffer)
                expected = torch.full_like(host, -1)
                expected[::stride] = torch.arange(blocks, dtype=host.dtype)
                expected[1::stride] = blocks
                assert torch.equal(host, expected)
    finally:
        try:
            if buffer is not None:
                worker.free(buffer)
        finally:
            worker.close()
