# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Real CSA kernels and seven original goldens, with explicit resident-only RED gates."""

from examples.a5.host_build_graph.deepseek_v4_pro_csa import business_cases as variants
from examples.a5.host_build_graph.deepseek_v4_pro_csa import fixture
from examples.a5.host_build_graph.deepseek_v4_pro_csa import test_deepseek_v4_pro_csa as original
from simpler_setup import SceneTestCase, TaskArgsBuilder, scene_test


@scene_test(level=2, runtime="host_build_graph")
class TestCSABusiness(original.TestDeepSeekV4ProCSAHBG):
    CASES = [
        {"name": name, "platforms": ["a5"], "manual": True, "params": params} for name, params in variants.VARIANTS
    ]

    def generate_args(self, params):
        args = super().generate_args({})
        values = variants.transform_inputs({spec.name: spec.value for spec in args.specs}, params)
        for spec in args.specs:
            assert values[spec.name].shape == spec.value.shape, f"{spec.name}: CSA shape changed"
            assert values[spec.name].dtype == spec.value.dtype, f"{spec.name}: CSA dtype changed"
        return TaskArgsBuilder(*(spec._replace(value=values[spec.name]) for spec in args.specs))

    def compute_golden(self, args, params):
        outputs = {
            arg["name"]: fixture.load_tensor(arg["name"], "out")
            for arg in fixture.manifest()["arguments"]
            if arg["direction"] != "IN"
        }
        for name, value in variants.transform_outputs(outputs, params).items():
            getattr(args, name).copy_(value)

    def _run_and_validate(self, worker, *args, **kwargs):
        assert not kwargs.get("skip_golden", False), "CSA TDD cannot skip the seven-output oracle"
        case = kwargs.get("case", args[1] if len(args) > 1 else {})
        assert not case.get("skip_golden", self.SKIP_GOLDEN), "CSA TDD cannot skip the seven-output oracle"
        self._csa_worker = worker
        self._csa_publish_count = worker.scheduler_mode_publish_count
        return super()._run_and_validate(worker, *args, **kwargs)

    def compare_outputs(self, test_args, golden_args, output_names, params):
        super().compare_outputs(test_args, golden_args, output_names, params)
        count = self._csa_worker.scheduler_mode_publish_count
        assert count > self._csa_publish_count, "missing fresh scheduler publication for this CSA invocation"
        self._csa_publish_count = count
        mode = self._csa_worker.scheduler_runtime_mode
        print(f"[CSA TDD] seven outputs passed; scheduler_mode={mode}; params={params}", flush=True)
        if params.get("resident", False):
            assert mode in (1, 2), f"CSA numerical checks passed, but resident required; mode={mode}"


@scene_test(level=2, runtime="host_build_graph")
class TestCSAResident(TestCSABusiness):
    CASES = [
        {
            "name": f"resident_{name}",
            "platforms": ["a5"],
            "manual": True,
            "params": dict(params, resident=True),
        }
        for name, params in variants.VARIANTS
        if name in ("baseline", "all_scramble", "all_pages_scale_poison")
    ]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
