#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3 launch acceptance is published on every platform.

A ChipTask's sticky acceptance word is written by the platform runner once the
run crosses its launch boundary, through the launch-acceptance target bound per
run and published at the real kernel-launch marker. A completed dispatch must
leave the word set in whichever mailbox frame carried it.
"""

import ctypes

import torch
from simpler.task_interface import (
    MAILBOX_RUN_ENTRY_STRIDE,
    MAILBOX_RUN_ENTRY_TABLE_OFFSET,
    mailbox_run_entry_count,
)
from simpler.task_interface import ArgDirection as D
from simpler.worker import (
    _DEFAULT_PIPELINE_DEPTH_REQUEST,
    _OFF_ENTRY_ACCEPTED,
    _TASK_ACCEPTED,
    _mailbox_load_i32,
)

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _build_l3_task_args

KERNELS_BASE = "../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"
_SIZE = 128 * 128
# This Worker leaves pipeline_depth unset, so its mailbox is created for the default request's
# worth of run-resource sets: one run-state entry per run it may hold, plus the hand-offs in
# flight.
_ENTRY_COUNT = mailbox_run_entry_count(_DEFAULT_PIPELINE_DEPTH_REQUEST)


def run_dag(orch, callables, task_args, config):
    """L3 orchestration: one ChipTask on worker 0."""
    chip_args, _ = _build_l3_task_args(task_args, callables.vector_kernel_sig)
    callables.keep(chip_args)  # prevent GC before drain
    orch.submit_next_level(callables.vector_kernel, chip_args, config, worker=0)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestL3LaunchAcceptance(SceneTestCase):
    """L3: a dispatched ChipTask publishes its sticky launch-acceptance word."""

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
        ],
    }

    CASES = [
        {
            "name": "default",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"device_count": 1, "num_sub_workers": 0},
            "params": {},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("a", torch.full((_SIZE,), 2.0, dtype=torch.float32).share_memory_()),
            TensorArg("b", torch.full((_SIZE,), 3.0, dtype=torch.float32).share_memory_()),
            TensorArg("f", torch.zeros(_SIZE, dtype=torch.float32).share_memory_()),
        )

    def compute_golden(self, args, params):
        args.f[:] = (args.a + args.b + 1) * (args.a + args.b + 2) + (args.a + args.b)

    def _run_and_validate_l3(self, worker, compiled_callables, sub_handles, case, **kwargs):
        super()._run_and_validate_l3(worker, compiled_callables, sub_handles, case, **kwargs)
        self._assert_acceptance_published(worker)

    @staticmethod
    def _assert_acceptance_published(worker):
        """The sticky word survives completion: the parent clears it only when it allocates the
        entry to the next run.

        Read from the run-state entries rather than the request frames. The device writes this
        word after the frame that delivered the payload has already gone back to the parent, so
        the frame is not where it can be.
        """
        shm_buf = worker._chip_shms[0].buf  # noqa: SLF001 -- scene-test white-box validation
        assert shm_buf is not None
        mailbox_addr = ctypes.addressof(ctypes.c_char.from_buffer(shm_buf))
        words = [
            _mailbox_load_i32(
                mailbox_addr + MAILBOX_RUN_ENTRY_TABLE_OFFSET + index * MAILBOX_RUN_ENTRY_STRIDE + _OFF_ENTRY_ACCEPTED
            )
            for index in range(_ENTRY_COUNT)
        ]
        assert _TASK_ACCEPTED in words, f"the chip worker never published launch acceptance: {words}"


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
