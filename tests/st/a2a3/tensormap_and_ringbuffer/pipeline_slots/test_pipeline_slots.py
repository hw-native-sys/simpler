#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""tensormap_and_ringbuffer's depth-two slot topology differs from host_build_graph's.

TMR classifies its GM heap, GM shared memory, and runtime image as
`DEVICE_SCRATCH`, so every slot shares arena bank 0 — only its task args are
`HOST_PER_RUN`. What still separates per slot is the host `Runtime` staging
buffer and the retained temporary buffer TMR stages device args through.
host_build_graph exercises the opposite split, so neither runtime's coverage
substitutes for the other's.
"""

import itertools

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _build_chip_task_args, _build_l2_ref_args, _compare_outputs

_VECTOR_KERNELS = "../../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestPipelineSlotsTmr(SceneTestCase):
    """A leased run on slot 1 must not disturb slot 0's per-run state."""

    RTOL = 1e-5
    ATOL = 1e-5

    CALLABLE = {
        "orchestration": {
            "source": f"{_VECTOR_KERNELS}/orchestration/example_orchestration.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": f"{_VECTOR_KERNELS}/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": f"{_VECTOR_KERNELS}/aiv/kernel_add_scalar.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "source": f"{_VECTOR_KERNELS}/aiv/kernel_mul.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "slot_topology",
            "platforms": ["a2a3", "a2a3sim"],
            "params": {},
        },
    ]

    # Shared st_worker means a shared per-slot high-water mark; draw from one
    # increasing counter so tests do not depend on pytest's ordering.
    _generation_counter = itertools.count(1)

    def generate_args(self, params):
        size = 128 * 128
        return TaskArgsBuilder(
            TensorArg("a", torch.full((size,), 2.0, dtype=torch.float32)),
            TensorArg("b", torch.full((size,), 3.0, dtype=torch.float32)),
            TensorArg("f", torch.zeros(size, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        a, b = args.a, args.b
        args.f[:] = (a + b + 1) * (a + b + 2) + (a + b)

    def _run(self, worker, handle, *, slot_id=None):
        """One run, on the ordinary path or through a lease, result checked.

        The two entry points take different argument types: ``Worker.run`` takes ``TensorArg`` args and
        materializes them in-process, while the lease is the direct chip API and takes the
        runtime.so-ABI POD.
        """
        test_args = self.generate_args({})
        golden = test_args.clone()
        self.compute_golden(golden, {})
        signature = self.CALLABLE["orchestration"]["signature"]
        config = self._build_config(self.CASES[0].get("config", {}))
        if slot_id is None:
            args, output_names = _build_l2_ref_args(test_args, signature, worker)
            worker.run(handle, args, config=config)
        else:
            chip_args, output_names = _build_chip_task_args(test_args, signature)
            state = worker._resolve_handle(handle)
            worker._chip_worker._run_with_pipeline_lease(
                state.slot_id, chip_args, slot_id=slot_id, generation=next(self._generation_counter), config=config
            )
        _compare_outputs(test_args, golden, output_names, self.RTOL, self.ATOL)

    def test_slot_one_keeps_tmr_per_run_state_separate(self, st_platform, st_worker):
        chip_worker = st_worker._chip_worker
        assert chip_worker.pipeline_depth == 2
        assert chip_worker.runtime_slot_count == 2

        addrs = chip_worker.runtime_buffer_addrs
        assert len(addrs) == 2, addrs
        assert addrs[0] != addrs[1], f"both slots stage into one host Runtime: {addrs}"

        handle = st_worker.register(self.build_callable(st_platform))
        try:
            # Interleave the slots: if slot 1 shared slot 0's host Runtime or
            # its retained temporary buffer, the run after the switch back
            # would read arguments the other slot staged.
            self._run(st_worker, handle)
            self._run(st_worker, handle, slot_id=1)
            self._run(st_worker, handle)
            self._run(st_worker, handle, slot_id=1)

            # Sequential runs re-stage their arguments every round, so correct
            # output alone would survive both slots sharing one staging buffer.
            # Read the addresses instead.
            temp0 = chip_worker.retained_temp_addr(0)
            temp1 = chip_worker.retained_temp_addr(1)
            assert temp0 != 0 and temp1 != 0, f"a slot that ran staged nothing: slot0={temp0:#x}, slot1={temp1:#x}"
            assert temp0 != temp1, f"both slots stage through one retained buffer: {temp0:#x}"

            # TMR declares its three pooled regions DEVICE_SCRATCH, so both
            # slots must still resolve to the one committed bank.
            bank0 = chip_worker.arena_bank_gm_heap_base(0)
            bank1 = chip_worker.arena_bank_gm_heap_base(1)
            assert bank0 != 0, "the shared device-scratch bank was never committed"
            assert bank1 == 0, f"slot 1 committed a second device-scratch bank: {bank1:#x}"
        finally:
            st_worker.unregister(handle)
