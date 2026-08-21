#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2A3 gives every run a freshly created AICore stream.

A2A3 submits each run's AICore and AICPU kernels on the stream set of the
selected pipeline slot rather than on the persistent bootstrap pair. The AICPU
stream carries no instruction-cache state and belongs to the slot for the
worker's lifetime. The AICore stream belongs to one run: the platform offers no
instruction-cache invalidation for code replaced at a reused GM address, and
creating the stream is the only operation known to leave a core free of the
previous image's instructions — selecting an already-existing stream is not, so
reuse cannot be made safe by tracking which image a stream last ran.

`Worker.run_stream_set_create_count` counts those creations, so it advances once
per run.
"""

import itertools

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.worker import Worker

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _build_chip_task_args, _build_l2_ref_args, _compare_outputs

_VECTOR_KERNELS = "../vector_example/kernels"
_REPEATED_RUNS = 4


@scene_test(level=2, runtime="host_build_graph")
class _SubtractCallable(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": f"{_VECTOR_KERNELS}/orchestration/example_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aiv/kernel_sub.cpp",
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


@scene_test(level=2, runtime="host_build_graph")
class TestRunStreamReuseHbg(SceneTestCase):
    """Each run gets its own AICore stream; its pipeline slot owns the rest."""

    RTOL = 1e-5
    ATOL = 1e-5

    CALLABLE = {
        "orchestration": {
            "source": f"{_VECTOR_KERNELS}/orchestration/example_orch.cpp",
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
            "name": "repeated_runs",
            "platforms": ["a2a3", "a2a3sim"],
            "config": {"block_dim": 3},
            "params": {},
        },
    ]

    def generate_args(self, params):
        size = 128 * 128
        return TaskArgsBuilder(
            TensorArg("a", torch.full((size,), 2.0, dtype=torch.float32)),
            TensorArg("b", torch.full((size,), 3.0, dtype=torch.float32)),
            TensorArg("f", torch.zeros(size, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        a, b = args.a, args.b
        args.f[:] = (a + b + 1) * (a + b + 2)

    def test_every_run_creates_its_own_aicore_stream(self, st_platform, st_worker):
        """N runs create N AICore streams, and every result is right."""
        if st_platform != "a2a3":
            pytest.skip("run stream sets are an a2a3 onboard resource")

        callable_obj = self.build_callable(st_platform)
        self._run_and_validate_l2(st_worker, callable_obj, self.CASES[0], rounds=1)
        after_first = st_worker.run_stream_set_create_count
        rounds = _REPEATED_RUNS - 1
        self._run_and_validate_l2(st_worker, callable_obj, self.CASES[0], rounds=rounds)

        assert st_worker.run_stream_set_create_count == after_first + rounds, (
            f"{rounds} runs did not each create an AICore stream: "
            f"{after_first} -> {st_worker.run_stream_set_create_count}"
        )

    def _run_registered(self, worker, handle, *, subtract):
        params = self.CASES[0]["params"]
        test_args = self.generate_args(params)
        # Worker.run takes TensorArg args and materializes them in-process; the runtime.so-ABI POD is
        # the direct chip API's shape, used by the lease path below.
        args, output_names = _build_l2_ref_args(test_args, self.CALLABLE["orchestration"]["signature"], worker)
        golden_args = test_args.clone()
        a, b = golden_args.a, golden_args.b
        base = a - b if subtract else a + b
        golden_args.f[:] = (base + 1) * (base + 2)
        worker.run(handle, args, config=self._build_config(self.CASES[0]["config"]))
        _compare_outputs(test_args, golden_args, output_names, self.RTOL, self.ATOL)

    # The st_worker fixture is shared by every test in this class, and so is the
    # generation each slot has last admitted. Tests draw from one increasing
    # counter so they stay independent of the order pytest runs them in.
    _generation_counter = itertools.count(1)

    @classmethod
    def _next_generation(cls):
        return next(cls._generation_counter)

    def _run_registered_with_lease(self, worker, handle, *, slot_id, generation):
        params = self.CASES[0]["params"]
        test_args = self.generate_args(params)
        golden_args = test_args.clone()
        self.compute_golden(golden_args, params)
        chip_args, output_names = _build_chip_task_args(test_args, self.CALLABLE["orchestration"]["signature"])
        state = worker._resolve_handle(handle)
        worker._chip_worker._run_slot_with_pipeline_lease(
            state.slot_id,
            chip_args,
            slot_id,
            generation,
            config=self._build_config(self.CASES[0]["config"]),
        )
        _compare_outputs(test_args, golden_args, output_names, self.RTOL, self.ATOL)

    def test_alternating_code_images_never_execute_stale_instructions(self, st_platform, st_worker):
        """A→B→A→B at one reused GM code address always runs the selected image.

        Each result is compared against the golden for the image that run asked
        for, so an AICore that executed the previous image's instructions shows
        up as wrong data rather than as a stream-count mismatch.
        """
        if st_platform != "a2a3":
            pytest.skip("AICore code images are an a2a3 onboard resource")

        add_handle = st_worker.register(self.build_callable(st_platform))
        sub_handle = st_worker.register(_SubtractCallable.compile_chip_callable(st_platform))
        try:
            for handle, subtract in (
                (add_handle, False),
                (sub_handle, True),
                (add_handle, False),
                (sub_handle, True),
            ):
                before = st_worker.run_stream_set_create_count
                self._run_registered(st_worker, handle, subtract=subtract)
                assert st_worker.run_stream_set_create_count == before + 1
        finally:
            st_worker.unregister(sub_handle)
            st_worker.unregister(add_handle)

    def test_depth_two_slots_own_separate_resources(self, st_platform, st_worker):
        """Each slot owns its own host Runtime buffer and its own arena bank.

        Runs on simulation too: sim implements the same depth, so HBG — whose
        GM heap is HOST_PER_RUN — must commit a distinct bank per slot there as
        well. This is the only case that exercises two *committed* banks; the
        TMR suite cannot, because its arenas are DEVICE_SCRATCH and both slots
        correctly resolve to bank 0.
        """
        if not st_platform.startswith("a2a3"):
            pytest.skip("pipeline slot banks are an a2a3 resource")

        chip_worker = st_worker._chip_worker
        assert chip_worker.pipeline_depth == 2
        assert chip_worker.runtime_slot_count == 2

        addrs = chip_worker.runtime_buffer_addrs
        assert len(addrs) == 2, addrs
        assert addrs[0] != addrs[1], f"both pipeline slots stage into one Runtime buffer: {addrs}"

        add_handle = st_worker.register(self.build_callable(st_platform))
        try:
            self._run_registered(st_worker, add_handle, subtract=False)
            self._run_registered_with_lease(st_worker, add_handle, slot_id=1, generation=self._next_generation())

            # hbg declares its GM heap HOST_PER_RUN, so a run on slot 1 must
            # commit a second device allocation rather than reuse slot 0's.
            bank0 = chip_worker.arena_bank_gm_heap_base(0)
            bank1 = chip_worker.arena_bank_gm_heap_base(1)
            assert bank0 != 0 and bank1 != 0, f"a served bank is uncommitted: {bank0:#x}, {bank1:#x}"
            assert bank0 != bank1, f"both arena banks resolve to one GM heap: {bank0:#x}"

            # HBG retains argument staging per pipeline slot. Both slots must
            # be populated and independent, just like their arena banks.
            temp0 = chip_worker.retained_temp_addr(0)
            temp1 = chip_worker.retained_temp_addr(1)
            assert temp0 != 0 and temp1 != 0, f"a served slot has no retained staging: {temp0:#x}, {temp1:#x}"
            assert temp0 != temp1, f"both pipeline slots resolve to one retained staging buffer: {temp0:#x}"
        finally:
            st_worker.unregister(add_handle)

    def test_unleased_runs_do_not_consume_lease_generations(self, st_platform, st_worker):
        """A slot pool's first lease is admitted however many runs precede it.

        `PipelineSlotPool` is the only mint, and it starts every slot at
        generation 1. A consumer that minted generations for its own unleased
        runs would have advanced past that, and the first real lease it was
        ever handed would be rejected as stale. This needs a worker whose
        generations no other test has touched, hence a second one.
        """
        if st_platform != "a2a3":
            pytest.skip("pipeline slot leases are an a2a3 onboard resource")

        other = Worker(level=2, **st_worker._config)
        other.init()
        try:
            handle = other.register(self.build_callable(st_platform))
            try:
                for _ in range(3):
                    self._run_registered(other, handle, subtract=False)
                self._run_registered_with_lease(other, handle, slot_id=0, generation=1)
            finally:
                other.unregister(handle)
        finally:
            other.close()

    def test_depth_two_slot_is_generation_safe(self, st_platform, st_worker):
        if st_platform != "a2a3":
            pytest.skip("pipeline slot banks are an a2a3 onboard resource")

        add_handle = st_worker.register(self.build_callable(st_platform))
        superseded = self._next_generation()
        generation = self._next_generation()
        try:
            self._run_registered_with_lease(st_worker, add_handle, slot_id=1, generation=generation)

            # One run may dispatch more than once under its lease.
            self._run_registered_with_lease(st_worker, add_handle, slot_id=1, generation=generation)

            # A rejected lease must not reach launch, so it creates no stream.
            stream_sets = st_worker.run_stream_set_create_count
            with pytest.raises(RuntimeError, match="generation is stale"):
                self._run_registered_with_lease(st_worker, add_handle, slot_id=1, generation=superseded)
            assert st_worker.run_stream_set_create_count == stream_sets
        finally:
            st_worker.unregister(add_handle)


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestRunStreamFreshTmr(SceneTestCase):
    """TMR preparation keeps the same fresh run-owned AICore stream rule."""

    CALLABLE = TestRunStreamReuseHbg.CALLABLE
    CASES = TestRunStreamReuseHbg.CASES

    generate_args = TestRunStreamReuseHbg.generate_args
    compute_golden = TestRunStreamReuseHbg.compute_golden

    def test_every_run_creates_its_own_aicore_stream(self, st_platform, st_worker):
        if st_platform != "a2a3":
            pytest.skip("run stream sets are an a2a3 onboard resource")

        callable_obj = self.build_callable(st_platform)
        self._run_and_validate_l2(st_worker, callable_obj, self.CASES[0], rounds=1)
        after_first = st_worker.run_stream_set_create_count
        rounds = _REPEATED_RUNS - 1
        self._run_and_validate_l2(st_worker, callable_obj, self.CASES[0], rounds=rounds)
        assert st_worker.run_stream_set_create_count == after_first + rounds
