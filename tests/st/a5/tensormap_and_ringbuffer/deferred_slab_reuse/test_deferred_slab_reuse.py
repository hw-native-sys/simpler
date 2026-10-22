#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Producer-side header publication telemetry and scheduler reset on slab reuse.

The counters are pre-satisfied. Same-address reuse observes the next task's
initial header; the async_notify scene separately exercises real waits.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

_TASK_COUNT = 512
_PATTERN = (1, 0, 64, 0, 0, 1, 0, 64)
_LINE_WORDS = 16


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestDeferredSlabReuse(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/deferred_slab_reuse.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aiv/kernel_slab_probe.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.INOUT, D.SCALAR, D.SCALAR],
            }
        ],
    }
    # All cases share the fixture's Worker and registered L2 handle. Two AICPU
    # threads give one scheduler owner; single-block tasks reuse its parity slabs.
    CASES = [
        {
            "name": f"invocation_{invocation}",
            "platforms": ["a5", "a5sim"],
            "config": {"aicpu_thread_num": 2},
            "params": {},
        }
        for invocation in range(3)
    ]

    def generate_args(self, params):
        counters = torch.zeros(64 * _LINE_WORDS, dtype=torch.int32)
        counters[::_LINE_WORDS] = torch.arange(1, 65, dtype=torch.int32)
        return TaskArgsBuilder(
            TensorArg("counters", counters),
            TensorArg("records", torch.zeros((_TASK_COUNT, _LINE_WORDS), dtype=torch.int32)),
        )

    def compute_golden(self, args, params):
        checksum = 0
        for sequence in range(_TASK_COUNT):
            count = _PATTERN[sequence % len(_PATTERN)]
            previous_checksum = checksum
            checksum += sequence + 1 + 2 * count
            args.records[sequence] = torch.tensor(
                [sequence, 0, 0, count, count, count, 0, previous_checksum, checksum, 0, 0, 1, 64, 1, 64, 0x534C4142],
                dtype=torch.int32,
            )

    def compare_outputs(self, test_args, golden_args, output_names, params):
        # Address words are self-describing; every other telemetry field is exact.
        columns = [*range(9), *range(11, _LINE_WORDS)]
        assert torch.equal(test_args.records[:, columns], golden_args.records[:, columns])
        last_by_slab = {}
        transitions = set()
        for row in test_args.records.tolist():
            slab = (row[9] & 0xFFFFFFFF) | ((row[10] & 0xFFFFFFFF) << 32)
            assert slab != 0 and slab % 64 == 0
            count = row[3]
            if slab in last_by_slab:
                transitions.add((last_by_slab[slab], count))
            last_by_slab[slab] = count
        assert (64, 0) in transitions, "No observed 64-condition to ordinary same-slab reuse"
        assert (0, 1) in transitions, "No observed ordinary to deferred same-slab reuse"


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
