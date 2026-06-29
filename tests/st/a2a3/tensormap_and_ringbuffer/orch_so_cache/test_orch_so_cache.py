#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end coverage for the orchestration SO host-side cache.

The host hashes the orchestration SO's GNU Build-ID, skips re-uploading bytes
that already live on device, and tells AICPU to reuse the cached `dlopen`
handle. The framework reuses one `Worker` (and therefore one `DeviceRunner`)
across cases inside a `SceneTestCase`, so running multiple cases against the
same `CALLABLE` exercises the cache-hit path on every case after the first.

This test deliberately:
  - Reuses the vector_example orchestration & AIV kernels (no new C++ to maintain).
  - Spans three cases with different (a, b) inputs — proves cache hit doesn't
    leak any per-run state across iterations.
  - Uses the same tensor size (128*128) across all cases because the AIV
    kernels have hardcoded tile shapes and do not accept a runtime size.
  - Runs on both sim and hardware (sim DeviceRunner uses the same code path,
    just with `mem_alloc_` returning host memory).

Verification is purely outcome-based: every case must produce the correct
result. A regression in cache logic (stale handle, wrong device buffer,
missing dlopen on first run) shows up as wrong output or a runtime failure.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

_VECTOR_KERNELS = "../../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestOrchSoCache(SceneTestCase):
    """Same callable, three cases — case 0 misses the cache, cases 1-2 hit it."""

    CALLABLE = {
        "orchestration": {
            "source": f"{_VECTOR_KERNELS}/orchestration/example_orchestration.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": f"{_VECTOR_KERNELS}/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": f"{_VECTOR_KERNELS}/aiv/kernel_add_scalar.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "source": f"{_VECTOR_KERNELS}/aiv/kernel_mul.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    # Three cases sharing one callable. The framework iterates them on a
    # single Worker; cases after the first land on cache-hit. Sizes vary so
    # a stale handle would manifest as wrong output, not "happens to pass".
    _COMMON_CONFIG = {"aicpu_thread_num": 4, "block_dim": 3}
    _PLATFORMS = ["a2a3sim", "a2a3"]

    # All cases use the same size (128*128) because the AIV kernels have
    # hardcoded tile shapes (kTRows_=128, kTCols_=128) and do not read a
    # runtime size argument — running with a smaller tensor would cause an
    # out-of-bounds access.  Different (a, b) values are enough to verify
    # that no per-run state leaks across cache-hit iterations.
    CASES = [
        {
            "name": "first_miss",
            "platforms": _PLATFORMS,
            "config": _COMMON_CONFIG,
            "params": {"size": 128 * 128, "a": 2.0, "b": 3.0},
        },
        {
            "name": "second_hit",
            "platforms": _PLATFORMS,
            "config": _COMMON_CONFIG,
            "params": {"size": 128 * 128, "a": 1.0, "b": 4.0},
        },
        {
            "name": "third_hit",
            "platforms": _PLATFORMS,
            "config": _COMMON_CONFIG,
            "params": {"size": 128 * 128, "a": 0.5, "b": 0.5},
        },
    ]

    def generate_args(self, params):
        size = params["size"]
        a = params["a"]
        b = params["b"]
        return TaskArgsBuilder(
            Tensor("a", torch.full((size,), a, dtype=torch.float32)),
            Tensor("b", torch.full((size,), b, dtype=torch.float32)),
            Tensor("f", torch.zeros(size, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        # f = (a+b+1) * (a+b+2) + (a+b) — same formula as vector_example.
        args.f[:] = (args.a + args.b + 1) * (args.a + args.b + 2) + (args.a + args.b)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
