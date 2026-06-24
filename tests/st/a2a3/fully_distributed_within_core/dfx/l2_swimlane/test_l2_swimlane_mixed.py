#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L2 swimlane profiling on a chained MIX-task workload.

Companion to ``test_l2_swimlane.py``. vector_example is AIV-only and emits
one perf row per ``task_id`` — the dedup branch in
``compute_dag_stats_from_deps`` (and the matching dedup in the oracle
inside :mod:`_swimlane_validate`) sits idle. ``chained_mix_orch.cpp`` runs
3 MIX tasks where each step's output feeds the next step's input, so
the workload produces *both*:

  - MIX rows: each MIX task_id emits one perf row per subtask/core
  - deps.json edges: 2 unique (pred, succ) pairs from the chain

That combination is what makes the dedup arithmetically observable. Without
``seen_tids`` the oracle would compute fanout = 4 instead of 2 (each MIX
task's fanout being counted once per perf row), and the differential gate
would fire.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

from ._swimlane_validate import validate_perf_artifact

_MATMUL_SIZE = 128
_TILE_ELEMS = _MATMUL_SIZE * _MATMUL_SIZE
# ws_aic / ws_aiv hold two intermediate slots — step 1's output (slot 0)
# is read by step 2, step 2's output (slot 1) is read by step 3.
_WS_SLOTS = 2
_WS_ELEMS = _WS_SLOTS * _TILE_ELEMS


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestL2SwimlaneMixed(SceneTestCase):
    """Chained MIX workload (3 steps, each step is AIC matmul + AIV add).

    Step N reads workspace slot N-1 and writes workspace slot N. Step 3
    writes the user-visible outputs. dep_gen collapses the multi-tensor
    flow between adjacent steps into a single (pred, succ) edge per pair,
    giving 2 unique edges across 3 MIX task_ids.
    """

    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/chained_mix_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.IN, D.IN, D.INOUT, D.INOUT, D.OUT, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "MATMUL",
                "source": "../../mixed_example/kernels/aic/kernel_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "name": "ADD",
                "source": "../../mixed_example/kernels/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "default",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {},
        },
    ]

    def generate_args(self, params):
        torch.manual_seed(42)
        A = torch.randn(_MATMUL_SIZE, _MATMUL_SIZE, dtype=torch.float32) * 0.01
        B = torch.randn(_MATMUL_SIZE, _MATMUL_SIZE, dtype=torch.float32) * 0.01
        D_t = torch.randn(_TILE_ELEMS, dtype=torch.float32) * 0.01
        E = torch.randn(_TILE_ELEMS, dtype=torch.float32) * 0.01

        return TaskArgsBuilder(
            Tensor("A", A.flatten()),
            Tensor("B", B.flatten()),
            Tensor("D", D_t),
            Tensor("E", E),
            Tensor("ws_aic", torch.zeros(_WS_ELEMS, dtype=torch.float32)),
            Tensor("ws_aiv", torch.zeros(_WS_ELEMS, dtype=torch.float32)),
            Tensor("aic_out", torch.zeros(_TILE_ELEMS, dtype=torch.float32)),
            Tensor("aiv_out", torch.zeros(_TILE_ELEMS, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        A_mat = args.A.reshape(_MATMUL_SIZE, _MATMUL_SIZE)
        B_mat = args.B.reshape(_MATMUL_SIZE, _MATMUL_SIZE)
        # AIC chain: B applied three times via matmul.
        s1_aic = torch.matmul(A_mat, B_mat)
        s2_aic = torch.matmul(s1_aic, B_mat)
        s3_aic = torch.matmul(s2_aic, B_mat)
        # AIV chain: E added three times → D + 3E.
        s1_aiv = args.D + args.E
        s2_aiv = s1_aiv + args.E
        s3_aiv = s2_aiv + args.E

        args.aic_out[:] = s3_aic.flatten()
        args.aiv_out[:] = s3_aiv
        # Final workspace state — slot 0 holds step 1's output, slot 1
        # holds step 2's output.
        args.ws_aic[0:_TILE_ELEMS] = s1_aic.flatten()
        args.ws_aic[_TILE_ELEMS:_WS_ELEMS] = s2_aic.flatten()
        args.ws_aiv[0:_TILE_ELEMS] = s1_aiv
        args.ws_aiv[_TILE_ELEMS:_WS_ELEMS] = s2_aiv

    def test_run(self, st_platform, st_worker, request):
        super().test_run(st_platform, st_worker, request)
        if not request.config.getoption("--enable-l2-swimlane", default=False):
            return
        for case in self.CASES:
            if st_platform in case["platforms"]:
                # Rely on the differential gate (Pop / Fanout / Fanin) —
                # the chain produces 3 MIX task_ids × 2 subtask rows = 6
                # perf rows and 2 deps.json edges, so the dedup branch in
                # the oracle has an arithmetically observable effect.
                validate_perf_artifact(f"TestL2SwimlaneMixed_{case['name']}")


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
