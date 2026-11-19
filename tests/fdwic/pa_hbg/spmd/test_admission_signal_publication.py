# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Rotating C/V publishers exercise the production signal primitive on private GM.

The atomic lock and independent return notice do not share the tail or packed log
lines. These litmus records do not replace actual scheduler entry/protocol tests.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

if __package__:
    from .test_spmd_contract import TestHbgSpmdContract as _SpmdContract
    from .test_spmd_contract import require_fresh_resident
else:
    from test_spmd_contract import TestHbgSpmdContract as _SpmdContract
    from test_spmd_contract import require_fresh_resident


def _signed(value):
    value &= (1 << 64) - 1
    return value if value < (1 << 63) else value - (1 << 64)


def probe_layout(params):
    blocks, rounds, shared = params["blocks"], params["rounds"], params["shared"]
    participants = blocks * 3 if shared else 3
    entries = participants * rounds
    log_stride = (entries + 15) // 16 * 16
    arena_words = 48 + log_stride + entries * 32 + 16
    return (1 if shared else blocks), participants, entries, log_stride, arena_words


def expected_probe(params):
    groups, participants, entries, log_stride, arena_words = probe_layout(params)
    controls = torch.zeros((groups, arena_words), dtype=torch.int64)
    controls[:, 48 : 48 + log_stride] = -1
    checks = torch.zeros((params["blocks"] * 3 + 1, 16), dtype=torch.int64)
    for group in range(groups):
        row = controls[group]
        row[16] = row[32] = entries
        for index in range(entries):
            row[48 + index] = -index - 7 if index % 2 == 0 else index + 7
            payload = 48 + log_stride + index * 16
            row[payload : payload + 16] = torch.tensor(
                [_signed(params["epoch"] ^ (index << 32) ^ (0x1234000010000100 + word)) for word in range(16)]
            )
            row[48 + log_stride + entries * 16 + index * 16] = 3
    for block in range(params["blocks"]):
        for lane in range(3):
            rank = block * 3 + lane if params["shared"] else lane
            arena = 0 if params["shared"] else block
            record = checks[block * 3 + lane]
            record[0] = 1
            record[9:13] = torch.tensor([params["rounds"], rank + 1, arena + 1, participants])
    return controls.flatten(), checks.flatten()


def validate_probe(controls, checks, params):
    expected_controls, expected_checks = expected_probe(params)
    assert torch.equal(controls, expected_controls), "tail, lock, log, payload, state or guard mismatch"
    assert torch.equal(checks, expected_checks), "missing lane or unpublished prefix observed"


def _case(blocks, rounds, shared, epoch, delayed_rank=-1):
    return {
        "name": f"n{blocks}_r{rounds}_shared{shared}_epoch{epoch}_delay{delayed_rank}",
        "platforms": (["a5sim"] if blocks == 8 else ["a5"]) if shared and blocks in (8, 32) else ["a5", "a5sim"],
        "params": dict(blocks=blocks, rounds=rounds, shared=shared, epoch=epoch, delayed_rank=delayed_rank),
    }


@scene_test(level=2, runtime="host_build_graph")
class TestHbgAdmissionSignalPublication(_SpmdContract):
    RTOL = ATOL = 0
    CALLABLE = {
        "orchestration": {
            "source": "kernels/admission_signal_orchestration.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": lane,
                "name": f"admission_signal_slot_{lane}",
                "source": f"kernels/admission_signal_slot_{lane}.cpp",
                "core_type": "aic" if lane == 0 else "aiv",
                "signature": [D.INOUT, D.INOUT],
            }
            for lane in range(3)
        ],
    }
    CASES = (
        [
            _case(blocks, rounds, shared, epoch)
            for shared in (0, 1)
            for blocks in ((1, 2, 8, 32) if shared else (1, 2, 65))
            for rounds in ((1, 3) if shared else (1, 17))
            for epoch in (0, -1)
        ]
        + [_case(blocks, 3, 1, -1, rank) for blocks in (8, 32) for rank in (0, 1, 2, blocks * 3 - 1)]
        + [_case(65, 17, 0, -1, 2)]
    )

    def generate_args(self, params):
        groups, _, _, log_stride, arena_words = probe_layout(params)
        controls = torch.zeros((groups, arena_words), dtype=torch.int64)
        controls[:, 48 : 48 + log_stride] = -1
        return TaskArgsBuilder(
            TensorArg("controls", controls.flatten()),
            TensorArg("checks", torch.zeros((params["blocks"] * 3 + 1) * 16, dtype=torch.int64)),
            *(Scalar(name, params[name]) for name in ("blocks", "rounds", "shared", "epoch", "delayed_rank")),
        )

    def compute_golden(self, args, params):
        controls, checks = expected_probe(params)
        args.controls.copy_(controls)
        args.checks.copy_(checks)

    def _run_and_validate(self, worker, *args, **kwargs):
        case = kwargs.get("case", args[1] if len(args) > 1 else {})
        assert not case.get("skip_golden", self.SKIP_GOLDEN), "publication acceptance cannot skip validation"
        return super()._run_and_validate(worker, *args, **kwargs)

    def compare_outputs(self, test_args, golden_args, output_names, params):
        self._contract_publish_count = require_fresh_resident(self._contract_worker, self._contract_publish_count)
        validate_probe(test_args.controls, test_args.checks, params)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
