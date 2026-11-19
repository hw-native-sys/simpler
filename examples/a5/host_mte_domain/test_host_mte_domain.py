# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Peer Host MTE domains on each A5 runtime, including PCIe PR and UB DT hosts."""

from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, scene_test

if __package__:
    from . import run as lifecycle
else:
    import run as lifecycle


class _HostMteDomain(SceneTestCase):
    CALLABLE = {
        "callables": [
            {
                "name": "peer_read",
                "orchestration": {
                    "source": "kernels/orchestration/peer_read.cpp",
                    "function_name": "host_mte_domain",
                    "config_name": "host_mte_domain_config",
                    "signature": [D.IN, D.OUT, D.OUT, D.IN, D.IN],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": "kernels/aiv/peer_read.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT, D.IN, D.IN],
                    },
                    {
                        "func_id": 1,
                        "source": "kernels/aiv/consume.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT],
                    },
                ],
            }
        ]
    }
    CASES = [
        {
            "name": "peer_mte_lifecycle",
            "platforms": ["a5"],
            "config": {"device_count": 2, "num_sub_workers": 0},
        }
    ]

    @classmethod
    def _compile_l3_callables(cls, platform):
        compiled = super()._compile_l3_callables(platform)
        lifecycle.prepare_inputs(cls._st_runtime, lifecycle.ARTIFACTS)
        return compiled

    def build_callable(self, platform):
        # Execution consumes data saved by the queue-external class compilation pass.
        lifecycle.load_inputs(self._st_runtime, lifecycle.ARTIFACTS)
        return super()._compile_l3_callables(platform)

    def _run_and_validate_l3(self, worker, compiled_callables, sub_handles, case, **kwargs):
        inputs = lifecycle.load_inputs(self._st_runtime, lifecycle.ARTIFACTS)
        config = self._build_config(
            case.get("config", {}),
            **{key: value for key, value in kwargs.items() if key.startswith("enable_") or key == "output_prefix"},
        )
        for _ in range(kwargs.get("rounds", 1)):
            lifecycle.run_lifecycle(worker, compiled_callables["peer_read"], self._st_runtime, inputs, config)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestHostMteTmr(_HostMteDomain):
    pass


@scene_test(level=3, runtime="host_build_graph")
class TestHostMteHbg(_HostMteDomain):
    pass


@scene_test(level=3, runtime="percounter")
class TestHostMtePercounter(_HostMteDomain):
    pass


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
