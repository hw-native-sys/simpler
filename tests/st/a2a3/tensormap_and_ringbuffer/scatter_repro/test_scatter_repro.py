#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""scatter_repro: PyPTO-generated column-scatter kernel ported into simpler ST.

The PyPTO test
``tests/st/runtime/ops/test_scatter.py::TestScatterIndexForm::test_scatter_fp16[a2a3]``
exhibits non-deterministic data corruption on a2a3 hardware (issue #967): the
``i * cols`` per-row base offset of the flattened scatter index is intermittently
dropped, so writes meant for the high rows (14, 15) either vanish or land in
row 0 with a one-row-stride displacement.

This test bypasses PyPTO entirely: the AICore kernel and orchestration .cpp
files are the exact ptoas-emitted artifacts captured via
``pytest ... --save-kernels --kernels-dir /tmp/scatter_repro``. Running them
through simpler's own ST harness isolates whether the race lives in simpler
(the runtime executing the kernel) or upstream (PyPTO host-side wiring).

Inputs (matching the PyPTO test):
  base[i, j] = -(j + 1 + i)   - negative sentinel, [16, 32] fp16
  idx[i, m]  = (m + i) % 32   - [16, 16] int16
  val[i, m]  = i * 16 + m + 1 - [16, 16] fp16

Golden (Torch ``scatter_`` semantics, last-wins along k):
  out = base.clone(); out[i, idx[i, m]] = val[i, m]
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

ROWS = 16
BASE_COLS = 32
INDEX_COLS = 16


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestScatterRepro(SceneTestCase):
    """PyPTO column-scatter kernel, run under simpler's own test harness."""

    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/scatter_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.IN, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "SCATTER_AIV",
                "source": "kernels/aiv/scatter_aiv.cpp",
                "core_type": "aiv",
            },
        ],
    }

    CASES = [
        {
            "name": "ColumnScatterFP16",
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": 2, "block_dim": 1},
            "params": {},
        },
    ]

    def generate_args(self, params):
        j = torch.arange(BASE_COLS, dtype=torch.float32).view(1, BASE_COLS)
        i = torch.arange(ROWS, dtype=torch.float32).view(ROWS, 1)
        base = (-(j + 1.0 + i)).to(torch.float16)

        m = torch.arange(INDEX_COLS, dtype=torch.int32).view(1, INDEX_COLS)
        ii = torch.arange(ROWS, dtype=torch.int32).view(ROWS, 1)
        idx = ((m + ii) % BASE_COLS).to(torch.int16)

        val = (ii.to(torch.float32) * 16.0 + m.to(torch.float32) + 1.0).to(torch.float16)

        output = base.clone()

        return TaskArgsBuilder(
            Tensor("base", base),
            Tensor("idx", idx),
            Tensor("val", val),
            Tensor("output", output),
        )

    def compute_golden(self, args, params):
        out = args.base.clone()
        for i in range(ROWS):
            for m in range(INDEX_COLS):
                out[i, int(args.idx[i, m].item())] = args.val[i, m]
        args.output[:] = out


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
