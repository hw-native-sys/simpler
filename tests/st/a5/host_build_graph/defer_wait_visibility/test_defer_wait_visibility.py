# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Async payload aliases and predicate-only inputs across ordinary and Graph tasks."""

import json
from importlib import import_module

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

OUT_WORDS = 4224


def _check_tail_cache_ranges(raw):
    protocol = raw["aicore_protocol"]
    # Scalar records carry the public task ID. Graph expansion adds ENTER/EXIT
    # and child nodes, so even a top-level tail can have a different node ref.
    public_ids = {node["node_ref"]: node["public_task_id"] for node in protocol.get("node_map", [])}
    tails = {
        public_ids[row["task_ref"]] if "node_map" in protocol else row["task_ref"]
        for row in protocol["tasks"]
        if row["kernel_ids"] == [65535, 3, 65535]
    }
    assert len(tails) == 2, "expected one AIV tail per instance"
    observed = []
    for stream in raw["scheduler_records"]["streams"]:
        records = stream["records"]
        for region in records:
            if region["kind"] != "scalar_cache_observe" or region["task_id"] not in tails:
                continue
            assert stream["capture"]["dropped"] == 0 and not stream["capture"]["truncated"]
            operations = [
                event["kind"]
                for event in records
                if event["kind"].startswith("scalar_dcci_")
                and region["start_cycles"] <= event["start_cycles"] <= event["end_cycles"] <= region["end_cycles"]
            ]
            # The reads at payload offsets 0 and 32 share a cache line. The
            # stores at output offsets 68 and 72 touch a second, distinct line.
            assert operations == ["scalar_dcci_single", "scalar_dcci_single"], operations
            observed.append(region["task_id"])
    assert sorted(observed) == sorted(tails), "both tails must execute targeted cache observation"


@scene_test(level=2, runtime="host_build_graph")
class TestDeferWaitVisibility(SceneTestCase):
    RTOL = 0
    ATOL = 0
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/visibility.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT] * 4 + [D.SCALAR] * 3,
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aic/body.cpp",
                "core_type": "aic",
                "signature": [D.INOUT] * 3 + [D.SCALAR] * 2,
            },
            {
                "func_id": 1,
                "source": "kernels/aiv/body.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT] * 3 + [D.SCALAR] * 2,
            },
            {"func_id": 2, "source": "kernels/aiv/waiter.cpp", "core_type": "aiv", "signature": [D.IN]},
            {
                "func_id": 3,
                "source": "kernels/aiv/tail.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }
    CASES = [
        {
            "name": f"{('aic', 'aiv')[core]}_{'graph' if graph else 'ordinary'}_{'take' if target else 'skip'}",
            "platforms": ["a5", "a5sim"],
            "params": {"core": core, "graph": graph, "target": target},
        }
        for core in (0, 1)
        for graph in (False, True)
        for target in (0, 1)
    ]

    def test_run(self, st_platform, st_worker, request, monkeypatch):
        if st_platform != "a5" or self._effective_enable_chip_swimlane(request) != 4:
            return super().test_run(st_platform, st_worker, request)
        scene = import_module("simpler_setup.scene_test")
        build_prefix = scene.build_output_prefix
        prefixes = []

        def remember_prefix(label):
            prefix = build_prefix(label)
            prefixes.append(prefix)
            return prefix

        monkeypatch.setattr(scene, "build_output_prefix", remember_prefix)
        super().test_run(st_platform, st_worker, request)
        assert prefixes
        for prefix in prefixes:
            captures = list(prefix.rglob("chip_swimlane_records.json"))
            assert len(captures) == 1, captures
            _check_tail_cache_ranges(json.loads(captures[0].read_text()))

    def generate_args(self, params):
        return TaskArgsBuilder(
            *(TensorArg(name, torch.zeros(64, dtype=torch.int32)) for name in ("signal", "predicate", "payload")),
            TensorArg("out", torch.zeros(2 * OUT_WORDS, dtype=torch.int32)),
            *(Scalar(name, int(params[name])) for name in ("core", "graph", "target")),
        )

    def compute_golden(self, args, params):
        for instance in range(2):
            row = args.out[instance * OUT_WORDS : (instance + 1) * OUT_WORDS]
            if params["target"]:
                row[:16] = torch.arange(16, dtype=torch.int32) * 2 + instance * 200 + 8
                row[16] = 1
            row[17] = 1
            row[18] = instance * 200 + 8

    def compare_outputs(self, actual, expected, output_names, params):
        for instance in range(2):
            row = actual.out[instance * OUT_WORDS : (instance + 1) * OUT_WORDS]
            golden = expected.out[instance * OUT_WORDS : (instance + 1) * OUT_WORDS].clone()
            blocks = int(row[64])
            assert 0 < blocks <= 128, "missing full-pool cache warmup"
            golden[64] = blocks
            golden[128 : 128 + blocks * 32 : 32] = -17 * 32 + 1 - params["target"]
            torch.testing.assert_close(row, golden, rtol=0, atol=0)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
