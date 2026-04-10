#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3 ChipTask → SubTask dependency via TensorMap.

Worker(level=3) submits a ChipTask then a SubTask that depends on it.
Verifies: TensorMap dependency inference, cross-fork data visibility,
SubWorker reads result produced by ChipWorker.
"""

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import WorkerPayload, WorkerType

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.scene_test import _build_chip_task_args

KERNELS_BASE = "../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"


def verify():
    """SubCallable — dependency target, runs after ChipTask completes."""


def run_dag(w, callables, task_args, config):
    """L3 orchestration: ChipTask → SubTask dependency."""
    chip_args, _ = _build_chip_task_args(task_args, callables.vector_kernel_sig)
    callables.keep(chip_args)  # prevent GC before drain

    chip_p = WorkerPayload()
    chip_p.worker_type = WorkerType.CHIP
    chip_p.callable = callables.vector_kernel.buffer_ptr()
    chip_p.args = chip_args.__ptr__()
    chip_p.block_dim = config.block_dim
    chip_p.aicpu_thread_num = config.aicpu_thread_num
    chip_result = w.submit(WorkerType.CHIP, chip_p, inputs=[], outputs=[task_args.f.numel() * 4])

    sub_p = WorkerPayload()
    sub_p.worker_type = WorkerType.SUB
    sub_p.callable_id = callables.verify
    w.submit(WorkerType.SUB, sub_p, inputs=[chip_result.outputs[0].ptr])


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestL3Dependency(SceneTestCase):
    """L3: ChipTask produces output, SubTask depends on it via TensorMap."""

    CALLABLE = {
        "orchestration": run_dag,
        "callables": [
            {
                "name": "vector_kernel",
                "orchestration": {
                    "source": f"{KERNELS_BASE}/orchestration/example_orchestration.cpp",
                    "function_name": "aicpu_orchestration_entry",
                    "signature": [D.IN, D.IN, D.OUT],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": f"{KERNELS_BASE}/aiv/kernel_add.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.IN, D.OUT],
                    },
                    {
                        "func_id": 1,
                        "source": f"{KERNELS_BASE}/aiv/kernel_add_scalar.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT],
                    },
                    {
                        "func_id": 2,
                        "source": f"{KERNELS_BASE}/aiv/kernel_mul.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.IN, D.OUT],
                    },
                ],
            },
            {"name": "verify", "callable": verify},
        ],
    }

    CASES = [
        {
            "name": "default",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"device_count": 1, "num_sub_workers": 1, "block_dim": 3, "aicpu_thread_num": 4},
            "params": {},
        },
    ]

    def generate_args(self, params):
        SIZE = 128 * 128
        return TaskArgsBuilder(
            Tensor("a", torch.full((SIZE,), 2.0, dtype=torch.float32).share_memory_()),
            Tensor("b", torch.full((SIZE,), 3.0, dtype=torch.float32).share_memory_()),
            Tensor("f", torch.zeros(SIZE, dtype=torch.float32).share_memory_()),
        )

    def compute_golden(self, args, params):
        args.f[:] = (args.a + args.b + 1) * (args.a + args.b + 2) + (args.a + args.b)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
