#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Two-rank URMA transfers whose successors consume deferred outputs."""

import os

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _rehosted_buffer_for, _rehosted_ref_for

NRANKS = 2
DTYPE_NBYTES = 4
CONSUMER_TILE_ELEMS = 64 * 128
SMALL_ELEMS = 128 * 128
CHUNKED_ELEMS = 256 * 1024 * 1024 // DTYPE_NBYTES + 8192
URMA_DATA_OFFSET_NBYTES = 256
_URMA_WORKSPACE_ENV = "SIMPLER_ENABLE_PTO_URMA_WORKSPACE"
_WORKSPACE_TRUTHY = {"1", "ON", "TRUE", "YES"}


def _urma_workspace_enabled():
    return os.environ.get(_URMA_WORKSPACE_ENV, "").upper() in _WORKSPACE_TRUTHY


def urma_deferred_completion_orch_fn(orch, callables, task_args, config):
    if not _urma_workspace_enabled():
        raise RuntimeError("URMA demo requires an explicit URMA=ON host-runtime build")
    elem_count = task_args.in_0.numel()
    input_nbytes = elem_count * DTYPE_NBYTES
    with orch.allocate_domain(
        workspace_engines=("urma",),
        name="urma_deferred_completion",
        workers=list(range(NRANKS)),
        window_size=max(URMA_DATA_OFFSET_NBYTES + 2 * input_nbytes, 4 * 1024 * 1024),
        buffers=[
            CommBufferSpec(name="urma_reserved", dtype="int32", count=64, nbytes=URMA_DATA_OFFSET_NBYTES),
            CommBufferSpec(name="input_window", dtype="float32", count=elem_count, nbytes=input_nbytes),
            CommBufferSpec(name="receive_window", dtype="float32", count=elem_count, nbytes=input_nbytes),
        ],
    ) as handle:
        for rank in range(NRANKS):
            source = getattr(task_args, f"in_{rank}")
            orch.copy_to(handle[rank].buffers["input_window"], _rehosted_buffer_for(task_args, source))
        for rank in range(NRANKS):
            domain = handle[rank]
            args = TaskArgs()
            args.add_tensor(domain.buffers["input_window"].tensor((elem_count,), DataType.FLOAT32), TensorArgType.INPUT)
            args.add_tensor(
                domain.buffers["receive_window"].tensor((elem_count,), DataType.FLOAT32), TensorArgType.INOUT
            )
            for name in ("out", "result"):
                args.add_tensor(
                    _rehosted_ref_for(task_args, getattr(task_args, f"{name}_{rank}")), TensorArgType.OUTPUT_EXISTING
                )
            args.add_scalar(domain.device_ctx)
            args.add_scalar(elem_count)
            args.add_scalar(task_args.mode)
            args.add_scalar(task_args.repeats)
            orch.submit_next_level(callables.urma_deferred_completion, args, config, worker=rank)


@pytest.mark.skipif(not _urma_workspace_enabled(), reason="URMA workspace overlay is OFF")
@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class _UrmaRegression(SceneTestCase):
    CALLABLE = {
        "orchestration": urma_deferred_completion_orch_fn,
        "callables": [
            {
                "name": "urma_deferred_completion",
                "orchestration": {
                    "source": "kernels/orchestration/urma_deferred_completion_orch.cpp",
                    "function_name": "urma_deferred_completion_orchestration",
                    "config_name": "urma_deferred_completion_orchestration_config",
                    "signature": [D.IN, D.INOUT, D.OUT, D.OUT, D.IN, D.IN, D.IN, D.IN],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": "kernels/aiv/kernel_urma_tget_async.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.IN, D.OUT, D.IN, D.IN, D.IN, D.IN, D.IN],
                    },
                    {
                        "func_id": 1,
                        "source": "kernels/aiv/kernel_consumer.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT, D.IN],
                    },
                ],
            }
        ],
    }
    RTOL = 0.0
    ATOL = 1e-3

    def generate_args(self, params):
        from simpler_setup import Scalar  # noqa: PLC0415

        specs = []
        elem_count = params["elems"]
        for rank in range(NRANKS):
            inp = torch.arange(elem_count, dtype=torch.int64).remainder_(251).to(torch.float32)
            inp.mul_(0.1).add_(rank * 100)
            specs.extend(
                [
                    TensorArg(f"in_{rank}", inp),
                    TensorArg(f"out_{rank}", torch.zeros(elem_count, dtype=torch.float32)),
                    TensorArg(f"result_{rank}", torch.zeros(elem_count, dtype=torch.float32)),
                ]
            )
        specs.extend([Scalar("mode", params["mode"]), Scalar("repeats", params["repeats"])])
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        for rank in range(NRANKS):
            source_rank = rank if params["mode"] else 1 - rank
            expected = getattr(args, f"in_{source_rank}")
            getattr(args, f"out_{rank}").copy_(expected)
            getattr(args, f"result_{rank}").copy_(expected + 1.0)


def _case(name, mode=0, repeats=1, elems=SMALL_ELEMS):
    if elems <= 0 or elems % CONSUMER_TILE_ELEMS:
        raise ValueError("URMA regression consumer requires a positive multiple of 8192 elements")
    return [
        {
            "name": name,
            "platforms": ["a5"],
            "config": {"device_count": NRANKS, "num_sub_workers": 0},
            "params": {"mode": mode, "repeats": repeats, "elems": elems},
        }
    ]


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaGet(_UrmaRegression):
    CASES = _case("peer_fetch")


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaPut(_UrmaRegression):
    CASES = _case("peer_put", mode=1)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaRepeatedFetch(_UrmaRegression):
    CASES = _case("repeated_fetch", repeats=16)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaRepeatedPut(_UrmaRegression):
    CASES = _case("repeated_put", mode=1, repeats=16)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaChunkedFetch(_UrmaRegression):
    CASES = _case("chunked_fetch", elems=CHUNKED_ELEMS)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaChunkedPut(_UrmaRegression):
    CASES = _case("chunked_put", mode=1, elems=CHUNKED_ELEMS)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
