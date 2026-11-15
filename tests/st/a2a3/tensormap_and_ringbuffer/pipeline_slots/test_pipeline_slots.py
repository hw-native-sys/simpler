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
`SCRATCH`, so every slot shares arena bank 0 — only its task args are
`DELIVERED`. Each live call owns its native storage and the retained temporary buffer
TMR stages device args through.
host_build_graph exercises the opposite split, so neither runtime's coverage
substitutes for the other's.
"""

import itertools
import time

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _build_chip_task_args, _build_l2_ref_args, _compare_outputs

_VECTOR_KERNELS = "../../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"


_SLOT_ORACLE_FAILURE_KEEPERS = []


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
        """Four live calls preserve their outputs while both execution slots are reused."""
        chip_worker = st_worker._chip_worker
        assert chip_worker.pipeline_depth == 2
        assert chip_worker.runtime_slot_count == 2
        records = []
        handle = st_worker.register(self.build_callable(st_platform))
        try:
            state = st_worker._resolve_handle(handle)
            signature = self.CALLABLE["orchestration"]["signature"]
            config = self._build_config(self.CASES[0].get("config", {}))
            for index, slot_id in enumerate((0, 1, 0, 1)):
                test_args = self.generate_args({})
                test_args.a.add_(index)
                test_args.b.add_(2 * index)
                sentinel = -1000.0 - index
                test_args.f.fill_(sentinel)
                golden = test_args.clone()
                self.compute_golden(golden, {})
                chip_args, output_names = _build_chip_task_args(test_args, signature)
                record = {
                    "args": test_args,
                    "chip_args": chip_args,
                    "golden": golden,
                    "outputs": output_names,
                    "sentinel": sentinel,
                    "run": None,
                }
                records.append(record)
                native_run = chip_worker._prepare_native_run_with_pipeline_lease(
                    state.slot_id, chip_args, slot_id=slot_id, generation=next(self._generation_counter), config=config
                )
                record["run"] = native_run
                # Slot aliases name this new owner only until physical retirement.
                storage = chip_worker.runtime_buffer_addrs[slot_id]
                staging = chip_worker.retained_temp_addr(slot_id)
                assert storage != 0 and staging != 0
                assert all(storage != old["storage"] and staging != old["staging"] for old in records[:-1])
                record.update(storage=storage, staging=staging)
                chip_worker._launch_native_run(native_run)
                deadline = time.monotonic() + 60.0
                while not chip_worker._poll_native_run(native_run):
                    assert time.monotonic() < deadline, "native execution did not physically retire"
                    time.sleep(0)
                # Physical progress releases capacity, not these calls' HOST outputs.
                for owner in records:
                    assert torch.all(owner["args"].f == owner["sentinel"]), (
                        "physical poll delivered an unresolved output"
                    )
            bank0 = chip_worker.arena_bank_gm_heap_base(0)
            bank1 = chip_worker.arena_bank_gm_heap_base(1)
            assert bank0 != 0, "the shared device-scratch bank was never committed"
            assert bank1 == 0, f"slot 1 committed a second device-scratch bank: {bank1:#x}"
            # Both same-slot successors are admitted before any strong observation.
            while records:
                record = records[-1]
                chip_worker._wait_native_run(record["run"])
                chip_worker._finalize_native_run(record["run"])
                _compare_outputs(record["args"], record["golden"], record["outputs"], self.RTOL, self.ATOL)
                records.pop()
            st_worker.unregister(handle)
        except BaseException:
            # Keep unknown last-use owners and their argument storage through teardown.
            _SLOT_ORACLE_FAILURE_KEEPERS.append((st_worker, handle, records))
            raise
