# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pure C/V native identity, completion visibility, overlap and retained-run contracts.

Entry/return ordinals are diagnostic atomic events, not performance measurements.
Every invocation requires fresh resident-mode evidence, including all reuse steps.
"""

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import TensorTransfer

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

if __package__:
    from .test_spmd_contract import TestHbgSpmdContract as _SpmdContract
    from .test_spmd_contract import require_fresh_resident
else:
    from test_spmd_contract import TestHbgSpmdContract as _SpmdContract
    from test_spmd_contract import require_fresh_resident

_RECORD_WORDS = 16
_MAX_DEPS = 12


def make_ordered_plan(params):
    scenario = params["scenario"]
    mask = params["mask"]
    blocks = params.get("blocks", 65)
    delay = params.get("tail_delay", 8192)
    if scenario == "alternate":
        shapes = [(n, mask) for n in (1, 512, 1, 65, 1, 2, 1)]
    elif scenario in ("overlap", "edge"):
        shapes = [(blocks, mask), (blocks, params["other_mask"])]
    elif scenario == "fanin":
        shapes = [(blocks, mask), (1, 2 if mask == 1 else 1)]
    else:
        assert scenario == "shape", f"unknown ordered probe scenario: {scenario}"
        shapes = [(blocks, mask), (blocks, mask)]
    rows, base = [], 0
    for task, (n, active_mask) in enumerate(shapes):
        deps = [task - 1] if task and scenario in ("alternate", "edge", "fanin") else []
        delayed = {"head": 0, "middle": n // 2, "tail": n - 1}[params.get("delay_block", "tail")]
        rows.append(
            [n, active_mask, base, task + 1, delay, len(deps), len(deps), delayed]
            + deps
            + [-1] * (_MAX_DEPS - len(deps))
            + deps
            + [-1] * (_MAX_DEPS - len(deps))
        )
        base += n * 3 + 1
    return torch.tensor(rows, dtype=torch.int64), base


def _case(name, scenario, mask, **params):
    return {
        "name": name,
        "platforms": ["a5", "a5sim"],
        "params": {"scenario": scenario, "mask": mask, **params},
    }


@scene_test(level=2, runtime="host_build_graph")
class TestHbgOrderedExecution(_SpmdContract):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/cla_orchestration.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.IN, D.IN],
        },
        "incores": [
            {
                "func_id": slot,
                "name": f"ordered_record_slot_{slot}",
                "source": f"kernels/ordered_record_slot_{slot}.cpp",
                "core_type": "aic" if slot == 0 else "aiv",
                "signature": [D.INOUT, D.IN, D.IN],
            }
            for slot in (0, 1)
        ],
    }
    CASES = (
        [
            _case(f"pure_{kind}_n{n}", "shape", mask, blocks=n)
            for kind, mask, counts in (
                ("c", 1, (1, 2, 31, 32, 33, 65, 512)),
                ("v", 2, (1, 2, 63, 64, 65, 129, 512)),
            )
            for n in counts
        ]
        + [_case(f"alternate_n1_spmd_mask{mask}", "alternate", mask) for mask in (1, 2)]
        + [
            _case(
                f"fanin_mask{mask}_slow_{position}", "fanin", mask, blocks=65, delay_block=position, tail_delay=262144
            )
            for mask in (1, 2)
            for position in ("head", "middle", "tail")
        ]
        + [
            _case(
                f"{scenario}_mask{mask}_to{other}",
                scenario,
                mask,
                blocks=65,
                other_mask=other,
                tail_delay=8000000 if scenario == "overlap" else 262144,
            )
            for scenario in ("overlap", "edge")
            for mask, other in ((1, 1), (2, 2), (1, 2), (2, 1))
        ]
        + [
            _case(f"retained_large_small_mask{mask}", "shape", mask, invocation_blocks=(512, 1, 65, 1, 512))
            for mask in (1, 2)
        ]
    )

    def generate_args(self, params):
        plan, records = make_ordered_plan(params)
        return TaskArgsBuilder(
            TensorArg("records", torch.zeros(records * _RECORD_WORDS, dtype=torch.int64)),
            TensorArg("plan", plan),
            TensorArg("host_plan", plan.clone(), transfer=TensorTransfer.NONE),
        )

    def compute_golden(self, args, params):
        rows = args.records.view(-1, _RECORD_WORDS)
        total = 0
        for n, mask, base, tag, *_ in args.plan.tolist():
            for block in range(n):
                slot = 0 if mask == 1 else 1
                row = rows[base + block * 3 + slot]
                row[0] = 1
                row[8:12] = torch.tensor([block + 1, n, slot + 1, tag])
                total += 1
        rows[args.plan[0, 2].item() + args.plan[0, 0].item() * 3, 7] = 2 * total

    def _run_and_validate(self, worker, callable_obj, case, *args, **kwargs):
        sequence = case.get("params", {}).get("invocation_blocks")
        if sequence is None:
            return super()._run_and_validate(worker, callable_obj, case, *args, **kwargs)
        for invocation, blocks in enumerate(sequence):
            invocation_case = {
                **case,
                "name": f"{case['name']}_run{invocation}_n{blocks}",
                "params": {**case["params"], "blocks": blocks},
            }
            super()._run_and_validate(worker, callable_obj, invocation_case, *args, **kwargs)

    def compare_outputs(self, test_args, golden_args, output_names, params):
        # The inherited run hook captures a fresh mode counter before every
        # invocation. Do not substitute record correctness for resident provenance.
        self._contract_publish_count = require_fresh_resident(self._contract_worker, self._contract_publish_count)
        actual = test_args.records.view(-1, _RECORD_WORDS)
        active_rows, intervals, events = [], [], []
        for n, mask, base, tag, _, dependencies, _, _, *rest in golden_args.plan.tolist():
            guard = actual[base + n * 3]
            assert guard[0].item() == 0, f"task {tag}: missing execution context"
            assert guard[1].item() == 0, f"task {tag}: original N or block index was corrupted"
            task_rows = [base + block * 3 + (0 if mask == 1 else 1) for block in range(n)]
            active_rows.extend(task_rows)
            entered = actual[task_rows, 13]
            returned = actual[task_rows, 14]
            assert torch.all(entered > 0).item() and torch.all(returned > entered).item(), "invalid execution interval"
            intervals.append((entered.min().item(), returned.max().item()))
            events.extend(entered.tolist() + returned.tolist())
            for predecessor in rest[:dependencies]:
                assert intervals[predecessor][1] < intervals[-1][0], (
                    "consumer entered before all producer blocks returned"
                )
        assert sorted(events) == list(range(1, len(events) + 1)), "duplicate or missing entry/return publication"
        assert torch.count_nonzero(actual[:, 12]).item() == 0, "consumer observed incomplete predecessor output"
        inactive = torch.ones(actual.shape[0], dtype=torch.bool)
        inactive[active_rows] = False
        assert torch.count_nonzero(actual[inactive, 13:15]).item() == 0, "inactive slot or guard emitted an execution"
        if params["scenario"] == "overlap":
            assert intervals[0][0] < intervals[1][1] and intervals[1][0] < intervals[0][1], (
                "independent tasks did not overlap: a remote tail must not impose a global TaskDone barrier"
            )
        comparable = test_args.records.clone().view(-1, _RECORD_WORDS)
        comparable[active_rows, 13:15] = 0
        torch.testing.assert_close(comparable, golden_args.records.view(-1, _RECORD_WORDS), rtol=0, atol=0)
        for name in output_names:
            if name != "records":
                torch.testing.assert_close(getattr(test_args, name), getattr(golden_args, name), rtol=0, atol=0)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
