# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Resident-only native acceptance: block identity, exactly-once execution and real fanin.

Ordinary C/V and GM Mix retain their original block counts, including waves beyond
physical capacity. A numerically correct legacy fallback MUST fail the mode assertion.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

_RECORD_WORDS = 16
_BLOCK_COUNTS = (1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 96, 128, 129, 512)
_MASKS = (1, 2, 3, 5, 6, 7)


def require_fresh_resident(worker, previous_count):
    """Require evidence from this invocation, not an old successful run."""
    count = worker.scheduler_mode_publish_count
    mode = worker.scheduler_runtime_mode
    assert count > previous_count, "no fresh scheduler publication: stale latch or unwired hook"
    assert mode in (1, 2), f"resident required; invocation mode={mode} (3/4 are legacy fallback)"
    return count


@scene_test(level=2, runtime="host_build_graph")
class TestHbgSpmdContract(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": slot,
                "name": f"record_slot_{slot}",
                "source": f"kernels/record_slot_{slot}.cpp",
                "core_type": "aic" if slot == 0 else "aiv",
                "signature": [D.INOUT],
            }
            for slot in range(3)
        ]
        + [
            {
                "func_id": 3,
                "name": "verify_fanin",
                "source": "kernels/verify_fanin.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.INOUT],
            }
        ],
    }

    CASES = [
        {
            "name": f"mask{mask}_n{blocks}",
            "platforms": ["a5", "a5sim"],
            "params": {"mask": mask, "blocks": blocks, "tasks": 2, "tail_delay": 4096},
        }
        for mask in _MASKS
        for blocks in _BLOCK_COUNTS
    ] + [
        {
            "name": f"reuse_mask{mask}",
            "platforms": ["a5", "a5sim"],
            "params": {"mask": mask, "blocks": 3, "tasks": 17, "tail_delay": 8192},
        }
        for mask in _MASKS
    ]

    def generate_args(self, params):
        records_per_task = params["blocks"] * 3 + 1  # Extra record catches an out-of-range context.
        return TaskArgsBuilder(
            TensorArg("records", torch.zeros(params["tasks"] * records_per_task * _RECORD_WORDS, dtype=torch.int64)),
            TensorArg("summary", torch.zeros(_RECORD_WORDS, dtype=torch.int64)),
            Scalar("blocks", params["blocks"]),
            Scalar("mask", params["mask"]),
            Scalar("tasks", params["tasks"]),
            Scalar("tail_delay", params["tail_delay"]),
        )

    def compute_golden(self, args, params):
        records = args.records.view(params["tasks"], params["blocks"] * 3 + 1, _RECORD_WORDS)
        entries = 0
        for task in range(params["tasks"]):
            for block in range(params["blocks"]):
                for slot in range(3):
                    if params["mask"] & (1 << slot):
                        record = records[task, block * 3 + slot]
                        record[0] = 1
                        record[8:12] = torch.tensor([block + 1, params["blocks"], slot + 1, task + 1])
                        entries += 1
        args.summary[0] = entries

    def _run_and_validate(self, worker, *args, **kwargs):
        assert not kwargs.get("skip_golden", False), "contract acceptance cannot skip golden or invocation provenance"
        self._contract_worker = worker
        self._contract_publish_count = worker.scheduler_mode_publish_count
        super()._run_and_validate(worker, *args, **kwargs)

    def compare_outputs(self, test_args, golden_args, output_names, params):
        self._contract_publish_count = require_fresh_resident(self._contract_worker, self._contract_publish_count)
        guards = test_args.records.view(params["tasks"], params["blocks"] * 3 + 1, _RECORD_WORDS)[:, -1, :]
        assert torch.count_nonzero(guards[:, 0]).item() == 0, "SPMD LocalContext/GlobalContext missing pointer"
        assert torch.count_nonzero(guards[:, 1]).item() == 0, (
            "SPMD block_idx/block_num/slot does not match original shape"
        )
        # Integer equality keeps duplicate counts and sentinel fields exact.
        for name in output_names:
            torch.testing.assert_close(getattr(test_args, name), getattr(golden_args, name), rtol=0, atol=0)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
