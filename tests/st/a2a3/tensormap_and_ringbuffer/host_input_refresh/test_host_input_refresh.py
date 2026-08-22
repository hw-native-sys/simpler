#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reused L3 workers reload fork-shared host inputs after in-place updates."""

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType

from simpler_setup import SceneTestCase, scene_test

_RANKS = 2
_ID_ELEMENTS = 24
_ACTIVATION_ELEMENTS = 4096
_HOST_IDS = [torch.zeros(_ID_ELEMENTS, dtype=torch.int64).share_memory_() for _ in range(_RANKS)]
_HOST_ACTIVATIONS = [torch.zeros(_ACTIVATION_ELEMENTS, dtype=torch.float32).share_memory_() for _ in range(_RANKS)]
_OUTPUT_IDS = [torch.zeros(1, dtype=torch.int64).share_memory_() for _ in range(_RANKS)]
_OUTPUT_ACTIVATIONS = [torch.zeros(1, dtype=torch.float32).share_memory_() for _ in range(_RANKS)]
_ROUND_VALUES = (101, 202, 101) * 8


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestHostInputRefresh(SceneTestCase):
    """Small and large host inputs must be staged again on every dispatch."""

    CALLABLE = {
        "callables": [
            {
                "name": "copy_first",
                "orchestration": {
                    "source": "kernels/orchestration/copy_first_orch.cpp",
                    "function_name": "aicpu_orchestration_entry",
                    "signature": [D.IN, D.IN, D.OUT, D.OUT],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": "kernels/aic/kernel_copy_first.cpp",
                        "core_type": "aic",
                        "signature": [D.IN, D.IN, D.OUT, D.OUT],
                    },
                ],
            },
        ],
    }

    CASES = [
        {
            "name": "in_place_update",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"device_count": _RANKS, "num_sub_workers": 0, "aicpu_thread_num": 2},
            "params": {},
        },
    ]

    def _run_and_validate_l3(  # noqa: PLR0913 -- scene-test runner hook
        self,
        worker,
        compiled_callables,
        sub_handles,
        case,
        rounds=1,
        skip_golden=False,
        enable_chip_swimlane=0,
        enable_dump_args=False,
        enable_pmu=0,
        enable_dep_gen=False,
        enable_scope_stats=False,
        output_prefix="",
    ):
        del (
            sub_handles,
            rounds,
            skip_golden,
            enable_chip_swimlane,
            enable_dump_args,
            enable_pmu,
            enable_dep_gen,
            enable_scope_stats,
            output_prefix,
        )
        type(self)._st_chip_handles = compiled_callables
        platform = str(worker._config["platform"])  # noqa: SLF001 -- scene-test white-box validation
        assert platform in case["platforms"]
        self.test_run(platform, worker)

    def test_run(self, st_platform, st_worker):
        del st_platform
        chip_handle = type(self)._st_chip_handles["copy_first"]
        config = self._build_config(self.CASES[0]["config"])
        keepalive = []

        id_refs = [
            st_worker.make_tensor_arg(ids, shapes=tuple(ids.shape), dtype=DataType.INT64.value) for ids in _HOST_IDS
        ]
        activation_refs = [
            st_worker.make_tensor_arg(
                activation,
                shapes=tuple(activation.shape),
                dtype=DataType.FLOAT32.value,
            )
            for activation in _HOST_ACTIVATIONS
        ]
        output_id_refs = [
            st_worker.make_tensor_arg(output, shapes=tuple(output.shape), dtype=DataType.INT64.value)
            for output in _OUTPUT_IDS
        ]
        output_activation_refs = [
            st_worker.make_tensor_arg(
                output,
                shapes=tuple(output.shape),
                dtype=DataType.FLOAT32.value,
            )
            for output in _OUTPUT_ACTIVATIONS
        ]

        for round_index, value in enumerate(_ROUND_VALUES, start=1):
            for rank in range(_RANKS):
                _HOST_IDS[rank].fill_(value + rank)
                _HOST_ACTIVATIONS[rank].fill_(-float(value + rank))
                _OUTPUT_IDS[rank].fill_(-1)
                _OUTPUT_ACTIVATIONS[rank].fill_(float("nan"))

            def dispatch(orch, _args, _config):
                for rank in range(_RANKS):
                    task_args = TaskArgs()
                    task_args.add_tensor(id_refs[rank], TensorArgType.INPUT)
                    task_args.add_tensor(activation_refs[rank], TensorArgType.INPUT)
                    task_args.add_tensor(output_id_refs[rank], TensorArgType.OUTPUT_EXISTING)
                    task_args.add_tensor(output_activation_refs[rank], TensorArgType.OUTPUT_EXISTING)
                    keepalive.append(task_args)
                    orch.submit_next_level(chip_handle, task_args, config, worker=rank)

            st_worker.run(dispatch, args=None, config=CallConfig())
            for rank in range(_RANKS):
                expected_id = value + rank
                expected_activation = -float(expected_id)
                assert _OUTPUT_IDS[rank][0].item() == expected_id, (
                    f"round {round_index}, rank {rank}: expected refreshed INT64 value {expected_id}, "
                    f"got {_OUTPUT_IDS[rank][0].item()}"
                )
                assert _OUTPUT_ACTIVATIONS[rank][0].item() == expected_activation, (
                    f"round {round_index}, rank {rank}: expected refreshed FP32 value {expected_activation}, "
                    f"got {_OUTPUT_ACTIVATIONS[rank][0].item()}"
                )
            keepalive.clear()


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
