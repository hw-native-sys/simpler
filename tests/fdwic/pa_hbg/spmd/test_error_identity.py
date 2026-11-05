# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device execution of the production error writer on a private diagnostic arena.

The probe does not fault the real scheduler or claim that its synthetic worker
identities are physical placement evidence. Real call sites have separate C++ tests.
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


def expected_probe(params, winners=None):
    blocks, shared, scenario = params["blocks"], params["shared"], params["scenario"]
    count = 1 if shared else blocks
    if winners is None and scenario >= 6:
        winners = [scenario - 6]
    winners = [0] if shared and winners is None else list(range(blocks)) if winners is None else winners
    assert len(winners) == count and all(0 <= winner < blocks for winner in winners)
    controls = torch.zeros((count, 96), dtype=torch.int64)
    checks = torch.zeros((blocks, 16), dtype=torch.int64)
    fields = (63, 52, 51, 48, 55, 63, 63, 63)[scenario]
    for index, winner in enumerate(winners):
        row = controls[index]
        row[32:37] = torch.tensor([1, 0, winner, -1, -1])
        row[48] = 7 if scenario == 1 else 6
        row[41:48] = torch.tensor(
            [
                45 if scenario == 1 else 120,
                -1 if scenario in (1, 3) else params["epoch"],
                -1 if scenario in (1, 3) else winner * 2,
                -1 if scenario in (2, 3) else 0 if scenario == 1 else winner,
                winner + 7 if scenario == 0 or scenario >= 5 else -1,
                winner % 96,
                (fields << 32) | (winner if scenario >= 5 else params["lane"]),
            ]
        )
        if scenario >= 5:
            row[64] = 2
        if scenario >= 6:
            row[65] = 1
    checks[:, 0] = 1
    for block in range(blocks):
        checks[block, 9] = winners[0] if shared else block
    return controls.flatten(), checks.flatten()


def validate_probe(controls, checks, params):
    count = 1 if params["shared"] else params["blocks"]
    assert controls.shape == (count * 96,) and checks.shape == (params["blocks"] * 16,)
    winners = controls.view(count, 96)[:, 34].tolist()
    assert all(0 <= winner < params["blocks"] for winner in winners), "error writer identity out of range"
    if params["scenario"] >= 6:
        assert winners == [params["scenario"] - 6], "wrong directed publisher"
    if not params["shared"]:
        assert winners == list(range(params["blocks"])), "private error arenas were crossed"
    expected_controls, expected_checks = expected_probe(params, winners)
    assert torch.equal(controls, expected_controls), "incoherent or overwritten first-error identity"
    assert torch.equal(checks, expected_checks), "missing block or publication observed before full identity"


def _case(lane, blocks, shared, epoch, scenario=0):
    return {
        "name": f"lane{lane}_n{blocks}_shared{shared}_epoch{epoch}_scenario{scenario}",
        "platforms": ["a5", "a5sim"],
        "params": {"lane": lane, "blocks": blocks, "shared": shared, "epoch": epoch, "scenario": scenario},
    }


@scene_test(level=2, runtime="host_build_graph")
class TestHbgErrorIdentity(_SpmdContract):
    RTOL = ATOL = 0
    CALLABLE = {
        "orchestration": {
            "source": "kernels/error_identity_orchestration.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": lane,
                "name": f"error_identity_{kind}",
                "source": f"kernels/error_identity_{kind}.cpp",
                "core_type": kind,
                "signature": [D.INOUT, D.INOUT],
            }
            for lane, kind in enumerate(("aic", "aiv"))
        ],
    }
    CASES = (
        [
            _case(lane, blocks, shared, epoch)
            for lane in (0, 1)
            for blocks in (1, 2, 65)
            for shared in (0, 1)
            for epoch in (0, 61, -1)
        ]
        + [_case(lane, 1 if scenario == 1 else 65, 0, 61, scenario) for lane in (0, 1) for scenario in (1, 2, 3, 4)]
        + [_case(2, 2, 1, epoch, 5) for epoch in (0, -1)]
        + [_case(2, 2, 1, epoch, scenario) for scenario in (6, 7) for epoch in (0, -1)]
    )

    def generate_args(self, params):
        count = 1 if params["shared"] else params["blocks"]
        return TaskArgsBuilder(
            TensorArg("controls", torch.zeros(count * 96, dtype=torch.int64)),
            TensorArg("checks", torch.zeros(params["blocks"] * 16, dtype=torch.int64)),
            *(Scalar(name, params[name]) for name in ("blocks", "lane", "shared", "epoch", "scenario")),
            Scalar("expected_fields", (63, 52, 51, 48, 55, 63, 63, 63)[params["scenario"]]),
        )

    def compute_golden(self, args, params):
        controls, checks = expected_probe(params)
        args.controls.copy_(controls)
        args.checks.copy_(checks)

    def _run_and_validate(self, worker, *args, **kwargs):
        assert not kwargs.get("skip_golden", False), "error identity acceptance cannot skip validation"
        case = kwargs.get("case", args[1] if len(args) > 1 else {})
        assert not case.get("skip_golden", self.SKIP_GOLDEN), "error identity acceptance cannot skip validation"
        self._error_worker = worker
        self._error_publish_count = worker.scheduler_mode_publish_count
        return super()._run_and_validate(worker, *args, **kwargs)

    def compare_outputs(self, test_args, golden_args, output_names, params):
        self._error_publish_count = require_fresh_resident(self._error_worker, self._error_publish_count)
        validate_probe(test_args.controls, test_args.checks, params)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
