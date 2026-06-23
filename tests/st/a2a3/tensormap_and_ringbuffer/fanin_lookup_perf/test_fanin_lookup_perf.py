#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""fanin_lookup_perf: validate a 64x64 explicit fanin DAG.

The orchestration submits 64 independent producers and 64 independent
consumers. Each consumer explicitly depends on all 64 producers. The real
kernel case uses disjoint tensor slices so tensormap auto-deps do not add
producer chains or consumer chains.
"""

import ctypes

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestFaninLookupPerf(SceneTestCase):
    """Validate a wide 64-producer/64-consumer explicit fanin DAG."""

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/fanin_lookup_perf_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "WRITE_CONST",
                "source": "kernels/aic/kernel_write_const_visible.cpp",
                "core_type": "aic",
                # Single-AIC task with one INOUT tensor at payload slot 0;
                # arg_index maps it explicitly.
                "signature": [D.INOUT],
                "arg_index": [0],
            },
        ],
    }

    CASES = [
        {
            "name": "LookupOnlyProducers64Consumers64",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {"producer_count": 64, "consumer_count": 64, "use_real_kernels": 0},
        },
        {
            "name": "SwimlaneProducers64Consumers64",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {"producer_count": 64, "consumer_count": 64, "use_real_kernels": 1},
        },
    ]

    def generate_args(self, params):
        slot_elems = 16
        producer_count = int(params["producer_count"])
        consumer_count = int(params["consumer_count"])
        return TaskArgsBuilder(
            Tensor("producer_out", torch.full((producer_count * slot_elems,), -1.0, dtype=torch.float32)),
            Tensor("consumer_out", torch.full((consumer_count * slot_elems,), -1.0, dtype=torch.float32)),
            Scalar("producer_count", ctypes.c_int64(producer_count)),
            Scalar("consumer_count", ctypes.c_int64(consumer_count)),
            Scalar("use_real_kernels", ctypes.c_int64(int(params["use_real_kernels"]))),
        )

    def compute_golden(self, args, params):
        if not params["use_real_kernels"]:
            return
        slot_elems = 16
        for i in range(int(params["producer_count"])):
            args.producer_out[i * slot_elems] = 42.0
        for c in range(int(params["consumer_count"])):
            args.consumer_out[c * slot_elems] = 42.0


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
