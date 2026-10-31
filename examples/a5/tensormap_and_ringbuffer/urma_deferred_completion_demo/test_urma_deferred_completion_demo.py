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

import copy
import os

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
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


def urma_deferred_completion_orch_fn(orch, callables, task_args, config, *, workspace_engines=("urma",)):
    _require_urma_workspace_enabled()
    input_nbytes = N * DTYPE_NBYTES
    with orch.allocate_domain(
        workspace_engines=workspace_engines,
        name="urma_deferred_completion",
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
    ) as handle:
        for rank in range(NRANKS):
            source = getattr(task_args, f"in_{rank}")
            orch.copy_to(handle[rank].buffers["input_window"], _rehosted_buffer_for(task_args, source))
        for rank in range(NRANKS):
            domain = handle[rank]
            args = TaskArgs()
            args.add_tensor(domain.buffers["input_window"].tensor((N,), DataType.FLOAT32), TensorArgType.INPUT)
            args.add_tensor(
                _rehosted_ref_for(task_args, getattr(task_args, f"out_{rank}")), TensorArgType.OUTPUT_EXISTING
            )
            args.add_tensor(
                _rehosted_ref_for(task_args, getattr(task_args, f"result_{rank}")), TensorArgType.OUTPUT_EXISTING
            )
            args.add_scalar(domain.device_ctx)
            orch.submit_next_level(callables.urma_deferred_completion, args, config, worker=rank)


@pytest.mark.skipif(
    not _urma_workspace_enabled(),
    reason="URMA workspace overlay not enabled (set SIMPLER_ENABLE_PTO_URMA_WORKSPACE=ON to run). "
    "See docs/a5-sdma-overlay.md (#1315).",
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
                    "signature": [D.IN, D.OUT, D.OUT, D.IN],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": "kernels/aiv/kernel_urma_tget_async.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT, D.IN],
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
            "name": "peer_fetch",
            "platforms": ["a5"],
            "config": {"device_count": NRANKS, "num_sub_workers": 0},
            "params": {},
        }
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
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        for rank in range(NRANKS):
            expected_out = getattr(args, f"in_{1 - rank}")
            getattr(args, f"out_{rank}").copy_(expected_out)
            getattr(args, f"result_{rank}").copy_(expected_out + 1.0)


def mixed_workspace_orch_fn(orch, callables, task_args, config):
    _require_urma_workspace_enabled()
    input_nbytes = N * DTYPE_NBYTES
    with orch.allocate_domain(
        name="sdma_urma",
        workers=list(range(NRANKS)),
        workspace_engines=("sdma", "urma"),
        window_size=max(URMA_DATA_OFFSET_NBYTES + input_nbytes, 4 * 1024 * 1024),
        buffers=[
            CommBufferSpec("reserved", "int32", URMA_DATA_OFFSET_NBYTES // 4, URMA_DATA_OFFSET_NBYTES),
            CommBufferSpec("input_window", "float32", N, input_nbytes),
        ],
    ) as domain:
        for rank in range(NRANKS):
            source = getattr(task_args, f"in_{rank}")
            orch.copy_to(domain[rank].buffers["input_window"], _rehosted_buffer_for(task_args, source))
        # Both callables write result_r, ordering URMA behind SDMA on each rank.
        # sdma_out_r remains an independent numerical witness of the SDMA read.
        for callable_obj, output_prefix in (
            (callables.sdma_async_completion, "sdma_out"),
            (callables.urma_deferred_completion, "out"),
        ):
            for rank in range(NRANKS):
                args = TaskArgs()
                args.add_tensor(
                    domain[rank].buffers["input_window"].tensor((N,), DataType.FLOAT32), TensorArgType.INPUT
                )
                args.add_tensor(
                    _rehosted_ref_for(task_args, getattr(task_args, f"{output_prefix}_{rank}")),
                    TensorArgType.OUTPUT_EXISTING,
                )
                args.add_tensor(
                    _rehosted_ref_for(task_args, getattr(task_args, f"result_{rank}")), TensorArgType.OUTPUT_EXISTING
                )
                args.add_scalar(domain[rank].device_ctx)
                orch.submit_next_level(callable_obj, args, config, worker=rank)


def _mixed_workspace_callable():
    spec = copy.deepcopy(TestUrmaDeferredCompletionDemo.CALLABLE)
    spec["orchestration"] = mixed_workspace_orch_fn
    spec["callables"].append(
        {
            "name": "sdma_async_completion",
            "orchestration": {
                "source": "../sdma_async_completion_demo/kernels/orchestration/sdma_async_completion_orch.cpp",
                "function_name": "sdma_async_completion_orchestration",
                "signature": [D.IN, D.OUT, D.OUT, D.IN],
            },
            "incores": [
                {
                    "func_id": func_id,
                    "source": f"../sdma_async_completion_demo/kernels/aiv/{source}",
                    "core_type": "aiv",
                    "signature": [D.IN, D.OUT, D.OUT, D.IN],
                }
                for func_id, source in enumerate(("kernel_sdma_tget_async.cpp", "kernel_consumer.cpp"))
            ],
        }
    )
    return spec


@pytest.mark.skipif(not _urma_workspace_enabled(), reason="mixed workspace test requires a URMA-enabled build")
@pytest.mark.sdma(worker_workspace=False)
@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestSdmaUrmaWorkspaceMix(SceneTestCase):
    CALLABLE = _mixed_workspace_callable()
    CASES = copy.deepcopy(TestUrmaDeferredCompletionDemo.CASES)
    RTOL = 0.0
    ATOL = 1e-3

    def generate_args(self, params):
        specs = []
        for rank in range(NRANKS):
            inp = torch.tensor([float(rank * 1000 + (i % 251)) / 10.0 for i in range(N)], dtype=torch.float32)
            specs.append(TensorArg(f"in_{rank}", inp))
            specs.extend(
                TensorArg(f"{name}_{rank}", torch.zeros(N, dtype=torch.float32))
                for name in ("sdma_out", "out", "result")
            )
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        for rank in range(NRANKS):
            expected = getattr(args, f"in_{1 - rank}")
            getattr(args, f"sdma_out_{rank}").copy_(expected)
            getattr(args, f"out_{rank}").copy_(expected)
            getattr(args, f"result_{rank}").copy_(expected + 1.0)


def same_task_workspace_orch_fn(orch, callables, task_args, config):
    urma_deferred_completion_orch_fn(orch, callables, task_args, config, workspace_engines=("sdma", "urma"))


def _same_task_workspace_callable():
    spec = copy.deepcopy(TestUrmaDeferredCompletionDemo.CALLABLE)
    spec["orchestration"] = same_task_workspace_orch_fn
    spec["callables"][0]["incores"][0]["source"] = "kernels/aiv/kernel_sdma_urma_tget_async.cpp"
    return spec


@pytest.mark.skipif(not _urma_workspace_enabled(), reason="mixed workspace test requires a URMA-enabled build")
@pytest.mark.sdma(worker_workspace=False)
@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestSdmaUrmaSameTask(SceneTestCase):
    """One producer fills disjoint output halves using two independent engines."""

    CALLABLE = _same_task_workspace_callable()
    CASES = copy.deepcopy(TestUrmaDeferredCompletionDemo.CASES)
    RTOL = 0.0
    ATOL = 1e-3
    generate_args = TestUrmaDeferredCompletionDemo.generate_args
    compute_golden = TestUrmaDeferredCompletionDemo.compute_golden


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
