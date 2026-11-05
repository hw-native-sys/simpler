# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Native TDD gates for the CLA graph shape; probe kernels are not the original CLA math."""

import json
from pathlib import Path

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import TensorTransfer

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

_PLAN_WIDTH = 32
_MAX_DEPS = 12
_RECORD_WORDS = 16
_SOURCE_SHA = "a33e497915b499f2ee13b9ae8182f4cb358d62106f8d358bbae9bc67fb30d747"


def make_plan(params):
    source = json.loads((Path(__file__).parent / "fixtures/cla_workload.json").read_text())
    assert source["source_sha256"] == _SOURCE_SHA
    nodes = source["nodes"]
    labels = params.get("labels")
    chosen = [n for n in nodes if n["kind"] != "alloc" and (labels is None or n["label"] in labels)]
    assert chosen and (labels is None or set(labels) == {n["label"] for n in chosen})
    refs = {n["submit_index"]: i for i, n in enumerate(chosen)}

    def contracted(ref):
        if ref in refs:
            return {refs[ref]}
        return set().union(*(contracted(p) for p in nodes[ref]["deps"]))

    dependencies = [sorted(set().union(*(contracted(p) for p in n["deps"]))) for n in chosen]

    def kernel_predecessors(ref):
        if chosen[ref]["kind"] == "kernel":
            return {ref}
        return set().union(*(kernel_predecessors(p) for p in dependencies[ref]))

    rows, base = [], 0
    for i, node in enumerate(chosen):
        deps = dependencies[i]
        checks = sorted(set().union(*(kernel_predecessors(p) for p in deps)))
        assert len(deps) <= _MAX_DEPS and len(checks) <= _MAX_DEPS
        assert all(p < i for p in deps + checks)
        mask = node["canonical_mask"]
        if mask == 3 and params.get("aiv1", False):
            mask = 5
        n = node["blocks"]
        slow = 0 if params.get("tail_first", False) else n - 1
        delay = 32768 if node["label"] in ("r2t17", "r2t20", "r3t10", "r3t40") else 1024
        rows.append(
            [n, mask, base, i + 1, delay, len(deps), len(checks), slow]
            + deps
            + [-1] * (_MAX_DEPS - len(deps))
            + checks
            + [-1] * (_MAX_DEPS - len(checks))
        )
        base += n * 3 + 1
    return torch.tensor(rows, dtype=torch.int64), base, chosen


@scene_test(level=2, runtime="host_build_graph")
class TestClaHbgContract(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/cla_orchestration.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.IN, D.IN],
        },
        "incores": [
            {
                "func_id": slot,
                "name": f"cla_record_{slot}",
                "source": f"kernels/cla_slot_{slot}.cpp",
                "core_type": "aic" if slot == 0 else "aiv",
                "signature": [D.INOUT, D.IN, D.IN],
            }
            for slot in range(3)
        ],
    }
    CASES = [
        {"name": "cla_full", "platforms": ["a5", "a5sim"], "params": {}},
        {"name": "cla_tail_first", "platforms": ["a5", "a5sim"], "params": {"tail_first": True}},
        {"name": "cla_full_aiv1", "platforms": ["a5", "a5sim"], "params": {"aiv1": True}},
        {
            "name": "cla_two_gathers",
            "platforms": ["a5", "a5sim"],
            "params": {
                "labels": ["r1t33", "r2t15", "r2t10", "r2t20", "r1t34", "r3t12", "r3t14", "r3t13", "r3t15", "r2t22"]
            },
        },
        {
            "name": "cla_eight_way_join",
            "platforms": ["a5", "a5sim"],
            "params": {
                "labels": ["r3t13", "r3t15", "r2t22", "r3t40"]
                + [f"r3t{i}" for i in (17, 18, 20, 21, 23, 24, 26, 27, 29, 30, 32, 33, 35, 36, 38, 39)]
            },
        },
        {
            "name": "cla_cross_ring",
            "platforms": ["a5", "a5sim"],
            "params": {"labels": ["r2t0", "r2t3", "r2t4", "r1t13", "r1t14"]},
        },
        {
            "name": "cla_irregular_shapes",
            "platforms": ["a5", "a5sim"],
            "params": {"labels": ["r2t1", "r2t8", "r2t3", "r3t0", "r2t2", "r2t6", "r3t2"]},
        },
    ]

    def generate_args(self, params):
        plan, records, _ = make_plan(params)
        return TaskArgsBuilder(
            TensorArg("records", torch.zeros(records * _RECORD_WORDS, dtype=torch.int64)),
            TensorArg("plan", plan),
            TensorArg("host_plan", plan.clone(), transfer=TensorTransfer.NONE),
        )

    def compute_golden(self, args, params):
        for n, mask, base, tag, *_ in args.plan.tolist():
            for block in range(n):
                for slot in range(3):
                    if mask & (1 << slot):
                        row = args.records.view(-1, _RECORD_WORDS)[base + block * 3 + slot]
                        row[0] = 1
                        row[8:12] = torch.tensor([block + 1, n, slot + 1, tag])

    def _run_and_validate(self, worker, *args, **kwargs):
        assert not kwargs.get("skip_golden", False), "CLA contract cannot skip validation"
        self._cla_worker = worker
        self._cla_count = worker.scheduler_mode_publish_count
        super()._run_and_validate(worker, *args, **kwargs)

    def compare_outputs(self, test_args, golden_args, output_names, params):
        rows = test_args.records.view(-1, _RECORD_WORDS)
        for n, _, base, tag, *_ in golden_args.plan.tolist():
            guard = rows[base + n * 3]
            assert guard[0].item() == 0, f"CLA task {tag}: LocalContext/GlobalContext missing pointer"
            assert guard[1].item() == 0, f"CLA task {tag}: block_idx/block_num/slot does not match the plan"
        assert torch.count_nonzero(rows[:, 12]).item() == 0, (
            "predecessors are incomplete or outputs not visible at entry"
        )
        for name in output_names:
            torch.testing.assert_close(getattr(test_args, name), getattr(golden_args, name), rtol=0, atol=0)
        assert self._cla_worker.scheduler_mode_publish_count > self._cla_count, "no fresh scheduler publication"
        self._cla_count = self._cla_worker.scheduler_mode_publish_count
        mode = self._cla_worker.scheduler_runtime_mode
        assert mode in (1, 2), f"CLA records and dependency sentinels passed, but resident required; mode={mode}"


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
