#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""URMA deferred-completion smoke test for onboard a5.

Each rank stages its input inside the communication window. The producer
fetches the peer rank's input into local ``out`` and registers the URMA async
event. The consumer depends on that output and writes ``result = out + 1``.
"""

import ctypes
import os
from contextlib import ExitStack

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _rehosted_buffer_for, _rehosted_ref_for

N = 128 * 128
NRANKS = 2
DTYPE_NBYTES = 4
URMA_DATA_OFFSET_NBYTES = 64 * DTYPE_NBYTES
_URMA_WORKSPACE_ENV = "SIMPLER_ENABLE_PTO_URMA_WORKSPACE"
_WORKSPACE_TRUTHY = {"1", "ON", "TRUE", "YES"}


def _urma_workspace_enabled():
    return os.environ.get(_URMA_WORKSPACE_ENV, "").upper() in _WORKSPACE_TRUTHY


def _require_urma_workspace_enabled():
    if not _urma_workspace_enabled():
        raise RuntimeError(
            "urma_deferred_completion_demo requires host runtime built with "
            f"{_URMA_WORKSPACE_ENV}=ON; set it before rebuilding simpler."
        )


def _urma_domain(orch, name):
    input_nbytes = N * DTYPE_NBYTES
    return orch.allocate_domain(
        name=name,
        workers=list(range(NRANKS)),
        window_size=max(URMA_DATA_OFFSET_NBYTES + input_nbytes, 4 * 1024 * 1024),
        buffers=[
            CommBufferSpec(
                name="urma_reserved",
                dtype="int32",
                count=URMA_DATA_OFFSET_NBYTES // DTYPE_NBYTES,
                nbytes=URMA_DATA_OFFSET_NBYTES,
            ),
            CommBufferSpec(name="input_window", dtype="float32", count=N, nbytes=input_nbytes),
        ],
    )


def _stage_urma_domain(orch, task_args, handle):
    for rank in range(NRANKS):
        source = getattr(task_args, f"in_{rank}")
        orch.copy_to(handle[rank].buffers["input_window"], _rehosted_buffer_for(task_args, source))


def _submit_urma_domain(orch, callables, task_args, config, handle):
    for rank in range(NRANKS):
        domain = handle[rank]
        args = TaskArgs()
        args.add_tensor(domain.buffers["input_window"].tensor((N,), DataType.FLOAT32), TensorArgType.INPUT)
        args.add_tensor(_rehosted_ref_for(task_args, getattr(task_args, f"out_{rank}")), TensorArgType.OUTPUT_EXISTING)
        args.add_tensor(
            _rehosted_ref_for(task_args, getattr(task_args, f"result_{rank}")), TensorArgType.OUTPUT_EXISTING
        )
        args.add_scalar(domain.device_ctx)
        args.add_scalar(task_args.chunks)
        orch.submit_next_level(callables.urma_deferred_completion, args, config, worker=rank)


def urma_deferred_completion_orch_fn(orch, callables, task_args, config):
    _require_urma_workspace_enabled()
    # Direct device control must precede submission in each run. Domain release
    # is deferred by Worker until accepted tasks finish.
    with ExitStack() as domains:
        survivor = None
        if task_args.multi_domain.value:
            survivor = domains.enter_context(_urma_domain(orch, "urma_survivor"))
            _stage_urma_domain(orch, task_args, survivor)
        handles = [domains.enter_context(_urma_domain(orch, f"urma_round_{i}")) for i in range(task_args.rounds.value)]
        for handle in handles:
            _stage_urma_domain(orch, task_args, handle)
        for handle in handles:
            _submit_urma_domain(orch, callables, task_args, config, handle)
            handle.release()
        if survivor is not None:
            _submit_urma_domain(orch, callables, task_args, config, survivor)


@pytest.mark.skipif(
    not _urma_workspace_enabled(),
    reason="URMA workspace overlay not enabled (set SIMPLER_ENABLE_PTO_URMA_WORKSPACE=ON to run). See docs/a5-urma.md.",
)
@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaDeferredCompletionDemo(SceneTestCase):
    CALLABLE = {
        "orchestration": urma_deferred_completion_orch_fn,
        "callables": [
            {
                "name": "urma_deferred_completion",
                "orchestration": {
                    "source": "kernels/orchestration/urma_deferred_completion_orch.cpp",
                    "function_name": "urma_deferred_completion_orchestration",
                    "config_name": "urma_deferred_completion_orchestration_config",
                    "signature": [D.IN, D.OUT, D.OUT, D.IN, D.IN],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": "kernels/aiv/kernel_urma_tget_async.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT, D.IN, D.IN],
                    },
                    {
                        "func_id": 1,
                        "source": "kernels/aiv/kernel_consumer.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT],
                    },
                ],
            }
        ],
    }
    CASES = [
        {
            "name": name,
            "platforms": ["a5"],
            "config": {"device_count": NRANKS, "num_sub_workers": 0},
            "params": {"chunks": chunks, "rounds": rounds, "multi_domain": multi_domain},
        }
        for name, chunks, rounds, multi_domain in [
            ("peer_fetch", 1, 1, 0),
        ]
    ]
    RTOL = 0.0
    ATOL = 1e-3

    def generate_args(self, params):
        specs = []
        for rank in range(NRANKS):
            inp = torch.tensor([float(rank * 1000 + (i % 251)) / 10.0 for i in range(N)], dtype=torch.float32)
            specs.extend(
                [
                    TensorArg(f"in_{rank}", inp),
                    TensorArg(f"out_{rank}", torch.zeros(N, dtype=torch.float32)),
                    TensorArg(f"result_{rank}", torch.zeros(N, dtype=torch.float32)),
                ]
            )
        specs.extend(Scalar(name, ctypes.c_int64(params[name])) for name in ("chunks", "rounds", "multi_domain"))
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        for rank in range(NRANKS):
            expected_out = getattr(args, f"in_{1 - rank}")
            getattr(args, f"out_{rank}").copy_(expected_out)
            getattr(args, f"result_{rank}").copy_(expected_out + 1.0)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaRepeatedPosts(TestUrmaDeferredCompletionDemo):
    CASES = [
        {
            "name": "repeated_posts",
            "platforms": ["a5"],
            "config": {"device_count": NRANKS, "num_sub_workers": 0},
            "params": {"chunks": 4, "rounds": 1, "multi_domain": 0},
        }
    ]


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaDomainLifecycle(TestUrmaDeferredCompletionDemo):
    CASES = [
        {
            "name": "domain_lifecycle",
            "platforms": ["a5"],
            "config": {"device_count": NRANKS, "num_sub_workers": 0},
            "params": {"chunks": 4, "rounds": 3, "multi_domain": 1},
        }
    ]

    def test_run(self, st_platform, st_worker, request):
        # Recreate domains after the previous run has finished and retired them.
        for _ in range(3):
            super().test_run(st_platform, st_worker, request)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
