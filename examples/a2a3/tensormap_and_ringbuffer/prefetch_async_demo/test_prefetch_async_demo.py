#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-card TPREFETCH_ASYNC smoke test.

Exercises the runtime-injected SDMA workspace: a Worker created with
``enable_sdma=True`` provisions the workspace once at init and injects its
address into every kernel's GlobalContext, so the kernel obtains it via
``get_dma_workspace(args, DMA_WORKSPACE_SDMA)`` -- no workspace is threaded as a
user arg. A Worker without ``enable_sdma`` creates no SDMA streams and its
kernels read a zero workspace address.
The kernel prefetches ``in`` into L2, waits on the returned event, then copies
``in`` to ``out``.

The prefetch is a pure cache hint that changes no value, so ``out == in``
bit-exactly is the property under test -- together with the event wait actually
completing rather than hanging. Because the kernel returns without writing
``out`` on a null workspace, that equality also proves the address it read was
live.

Onboard a2a3 runs the real engine; the device log (``[SDMA] Created 48 STARS
streams OK``) confirms the real SDMA path ran rather than the skip branch. On
a2a3sim the workspace is inert scratch and ``TPREFETCH_ASYNC`` is a no-op
(pto-isa ``cpu/TPrefetchAsync.hpp``), so what the sim case covers is the
injection path: a live address reaches the kernel.

Unlike the SDMA completion demo this needs no comm domain: the workspace is a
runtime-owned per-device resource, so a single device is enough.
"""

from __future__ import annotations

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

N = 128


@pytest.mark.sdma
@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestPrefetchAsyncDemo(SceneTestCase):
    """Prefetching a GM region then copying it leaves the data bit-exact."""

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/prefetch_async_orch.cpp",
            "function_name": "prefetch_async_orchestration",
            # in (IN), out (OUT) — the SDMA workspace is injected into every
            # kernel's GlobalContext by the enable_sdma Worker, not threaded
            # through as a user arg.
            "signature": [D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aiv/kernel_prefetch_copy.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "prefetch_copy",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {},
            "params": {},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("src", torch.arange(N, dtype=torch.float32) / 8.0),
            TensorArg("out", torch.full((N,), -1.0, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        # The prefetch is a pure cache hint that changes no value, so the copy
        # must reproduce the source bit-for-bit. What the case really proves is
        # that the injected workspace was real and the event wait completed
        # rather than hanging.
        args.out[:] = args.src


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
