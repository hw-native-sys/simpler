#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3 ChipTask -> SubTask dependency via TensorMap.

Worker(level=3) submits a ChipTask then a SubTask that depends on it.
Verifies: TensorMap dependency inference, cross-fork data visibility,
SubWorker reads result produced by ChipWorker.
"""

import threading

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import TaskArgs, TensorArgType

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, make_tensor_arg, scene_test
from simpler_setup.scene_test import CallableNamespace, _build_l3_task_args, _compare_outputs, _RehostedTaskArgs

KERNELS_BASE = "../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"


def verify(args):
    """SubCallable — dependency target, runs after ChipTask completes."""


def run_dag(orch, callables, task_args, config):
    """L3 orchestration: ChipTask → SubTask dependency."""
    # ChipTask: tags inside chip_args drive deps (INPUT → lookup; OUTPUT_EXISTING → insert).
    chip_args, _ = _build_l3_task_args(task_args, callables.vector_kernel_sig)
    callables.keep(chip_args)  # prevent GC before drain

    orch.submit_next_level(callables.vector_kernel, chip_args, config, worker=0)

    # SubTask: tag the chip output as INPUT — Orchestrator wires the dep via TensorMap.
    sub_args = TaskArgs()
    sub_args.add_tensor(make_tensor_arg(task_args.f), TensorArgType.INPUT)
    orch.submit_sub(callables.verify, sub_args)


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
        {
            "name": "ConcurrentTwoRequestPipeline",
            "platforms": ["a2a3"],
            "config": {"device_count": 1, "num_sub_workers": 1, "block_dim": 3, "aicpu_thread_num": 4},
            "manual": True,
            "params": {},
        },
    ]

    def _run_and_validate_l3(self, worker, compiled_callables, sub_handles, case, **kwargs):
        if case["name"] != "ConcurrentTwoRequestPipeline":
            return super()._run_and_validate_l3(worker, compiled_callables, sub_handles, case, **kwargs)

        requests = [self.generate_args(case["params"]) for _ in range(2)]
        goldens = [request.clone() for request in requests]
        for golden in goldens:
            self.compute_golden(golden, case["params"])

        rehosted = []
        try:
            rehosted = [_RehostedTaskArgs(worker, request) for request in requests]
            config = self._build_config(case["config"])
            callables = CallableNamespace({**compiled_callables, **sub_handles})
            ready = threading.Barrier(2, timeout=30.0)

            def request_orch(orch, request, request_id, emitter, cfg):
                chip_args, _ = _build_l3_task_args(request, callables.vector_kernel_sig)
                callables.keep(chip_args)
                ready.wait()
                orch.submit_next_level(callables.vector_kernel, chip_args, cfg, worker=0)

            with worker.open_request_session(request_orch, max_active_runs=2) as session:
                streams = [
                    session.submit(request, config=config, request_id=request_id)
                    for request_id, request in enumerate(requests, start=1)
                ]
                for stream in streams:
                    stream.wait(timeout=30.0)

            for request, golden in zip(requests, goldens):
                _compare_outputs(request, golden, request.tensor_names(), self.RTOL, self.ATOL)
        finally:
            for request_buffers in reversed(rehosted):
                request_buffers.release()

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
