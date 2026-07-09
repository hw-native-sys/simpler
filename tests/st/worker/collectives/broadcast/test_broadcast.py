#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scene test for distributed broadcast.

NOTE: Each nranks gets its own SceneTestCase subclass (not multiple CASES per
class) because multiple cases within a single class exhibit a Worker state-leak
that causes golden-mismatch failures on sequential worker.run() calls.
"""

import ctypes

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar as SScalar
from simpler_setup import SceneTestCase, TaskArgsBuilder, scene_test
from simpler_setup import Tensor as STensor

from .._helpers import (
    COUNT_PER_RANK,
    DTYPE_NBYTES,
    SIGNAL_TAIL_NBYTES,
    broadcast_expected_output,
    generic_collective_orch_fn,
)

ROOT_RANK = 0


def broadcast_orch_fn(orch, callables, task_args, config):
    _nranks = int(task_args.nranks.value)
    scratch_nbytes = COUNT_PER_RANK * DTYPE_NBYTES + SIGNAL_TAIL_NBYTES
    window_size = max(scratch_nbytes, 4 * 1024)
    generic_collective_orch_fn(
        orch,
        callables,
        task_args,
        config,
        chip_name="broadcast",
        float_elems=COUNT_PER_RANK,
        scratch_nbytes=scratch_nbytes,
        window_size=window_size,
        extra_scalars=[ROOT_RANK],
    )


_CALLABLE = {
    "orchestration": broadcast_orch_fn,
    "callables": [
        {
            "name": "broadcast",
            "orchestration": {
                "source": "kernels/orchestration/broadcast_orch.cpp",
                "function_name": "broadcast_orchestration",
                "config_name": "broadcast_orchestration_config",
                "signature": [D.IN, D.OUT, D.INOUT],
            },
            "incores": [
                {
                    "func_id": 0,
                    "source": "kernels/aiv/broadcast_kernel.cpp",
                    "core_type": "aiv",
                    "signature": [D.IN, D.OUT, D.INOUT],
                }
            ],
        }
    ],
}


def _make_args(nranks):
    specs = []
    for r in range(nranks):
        inp = torch.tensor(
            [i + r * 100 for i in range(COUNT_PER_RANK)] if r == ROOT_RANK else [0.0] * COUNT_PER_RANK,
            dtype=torch.float32,
        ).share_memory_()
        out = torch.zeros(COUNT_PER_RANK, dtype=torch.float32).share_memory_()
        specs.append(STensor(f"in_{r}", inp))
        specs.append(STensor(f"out_{r}", out))
    specs.append(SScalar("nranks", ctypes.c_int64(nranks)))
    return TaskArgsBuilder(*specs)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestBroadcastP2(SceneTestCase):
    CALLABLE = _CALLABLE
    CASES = [
        {
            "name": "p2",
            "platforms": ["a2a3sim", "a2a3", "a5sim"],
            "config": {"device_count": 2},
            "params": {"nranks": 2},
        }
    ]

    def generate_args(self, params):
        return _make_args(params["nranks"])

    def compute_golden(self, args, params):
        expected = torch.tensor(broadcast_expected_output(ROOT_RANK), dtype=torch.float32)
        for r in range(params["nranks"]):
            getattr(args, f"out_{r}").copy_(expected)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestBroadcastP4(SceneTestCase):
    CALLABLE = _CALLABLE
    CASES = [
        {
            "name": "p4",
            "platforms": ["a2a3sim", "a2a3", "a5sim"],
            "config": {"device_count": 4},
            "params": {"nranks": 4},
        }
    ]

    def generate_args(self, params):
        return _make_args(params["nranks"])

    def compute_golden(self, args, params):
        expected = torch.tensor(broadcast_expected_output(ROOT_RANK), dtype=torch.float32)
        for r in range(params["nranks"]):
            getattr(args, f"out_{r}").copy_(expected)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
