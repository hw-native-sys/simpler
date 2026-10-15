#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CSA host-built graph; L4 records identify the selected scheduler path."""

import pytest
import torch
from simpler.task_interface import TensorTransfer

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

if __package__:
    from . import golden, workload
else:
    import golden
    import workload


@pytest.mark.skipif(
    not workload.fixture.DATA_DIR.is_dir(), reason="CSA frozen data is not installed; run this case's fixture.py first"
)
@scene_test(level=2, runtime="host_build_graph")
class TestDeepSeekV4ProCSAHBG(SceneTestCase):
    CALLABLE = workload.CALLABLE
    CASES = [{"name": "B4S2Seed0", "platforms": ["a5"], "manual": True, "params": {}}]

    @classmethod
    def compile_chip_callable(cls, platform):
        workload.checkout_identity()
        return super().compile_chip_callable(platform)

    def generate_args(self, params):
        workload.fixture.verify_snapshot()
        specs = []
        for arg in workload.fixture.manifest()["arguments"]:
            if arg["direction"] == "OUT":
                value = torch.zeros(arg["shape"], dtype=getattr(torch, arg["dtype"].removeprefix("torch.")))
            else:
                value = workload.fixture.load_tensor(arg["name"], "in")
            spec = TensorArg(arg["name"], value)
            if arg["name"] == "hc_attn_scale":
                spec = spec._replace(transfer=TensorTransfer.NONE)
            specs.append(spec)
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        for arg in workload.fixture.manifest()["arguments"]:
            if arg["direction"] != "IN":
                getattr(args, arg["name"]).copy_(workload.fixture.load_tensor(arg["name"], "out"))

    def compare_outputs(self, test_args, golden_args, output_names, params):
        expected_names = {arg["name"] for arg in workload.fixture.manifest()["arguments"] if arg["direction"] != "IN"}
        if set(output_names) != expected_names:
            raise AssertionError(f"CSA output contract changed: {output_names}")
        actual = {name: getattr(test_args, name) for name in output_names}
        expected = {name: getattr(golden_args, name) for name in output_names}
        inputs = {arg["name"]: getattr(golden_args, arg["name"]) for arg in workload.fixture.manifest()["arguments"]}
        golden.validate_outputs(actual, expected, inputs)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
