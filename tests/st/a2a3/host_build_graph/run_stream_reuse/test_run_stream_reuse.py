#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2A3 reuses one AICore stream between code publications.

A2A3 submits every run's AICore and AICPU kernels on its own run stream pair
rather than on the persistent bootstrap pair. One pair carries all runs: the
execution claim is exclusive, so runs reach the device one at a time and the
stream orders them. A pipeline slot indexes the resources preparation mutates,
and preparing a run writes nothing to a stream.

The AICPU stream carries no instruction-cache state. The AICore stream stays
reusable until a new AICore code buffer is uploaded, which marks it stale; the
next launch then replaces it.

`Worker.run_stream_set_create_count` counts AICore stream creations, so runs
plateau after the first launch until another code publication.
"""

import copy
import itertools
import time

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _build_chip_task_args, _build_l2_ref_args, _compare_outputs

_REPEATED_RUNS = 4
_TMR_KERNELS = "../../tensormap_and_ringbuffer/run_stream_reuse/kernels"


def _rebase_callable(callable_spec: dict, kernels_dir: str) -> dict:
    """Copy a CALLABLE with every `kernels/...` source rebased onto `kernels_dir`."""
    out = copy.deepcopy(callable_spec)
    out["orchestration"]["source"] = out["orchestration"]["source"].replace("kernels", kernels_dir, 1)
    for incore in out["incores"]:
        incore["source"] = incore["source"].replace("kernels", kernels_dir, 1)
    return out


_SLOT_ORACLE_FAILURE_KEEPERS = []


@scene_test(level=2, runtime="host_build_graph")
class _SubtractCallable(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/example_orch.cpp",
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
                "source": "kernels/aiv/kernel_add_scalar.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "source": "kernels/aiv/kernel_mul.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }


@scene_test(level=2, runtime="host_build_graph")
class TestRunStreamReuseHbg(SceneTestCase):
    """One run stream pair serves every slot until a code publication."""

    RTOL = 1e-5
    ATOL = 1e-5

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/example_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": "kernels/aiv/kernel_add_scalar.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "source": "kernels/aiv/kernel_mul.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "repeated_runs",
            "platforms": ["a2a3", "a2a3sim"],
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

    def compute_golden(self, args, params, *, subtract=False):
        a, b = args.a, args.b
        base = a - b if subtract else a + b
        args.f[:] = (base + 1) * (base + 2)

    def test_repeated_runs_without_publication_reuse_aicore_stream(self, st_platform, st_worker):
        """Repeated runs preserve a warm stream while no code is published."""
        if st_platform != "a2a3":
            pytest.skip("run stream sets are an a2a3 onboard resource")

        handle = st_worker.register(self.build_callable(st_platform))
        try:
            self._run_registered(st_worker, handle, subtract=False)
            after_first = st_worker.run_stream_set_create_count
            for _ in range(_REPEATED_RUNS - 1):
                self._run_registered(st_worker, handle, subtract=False)
            assert st_worker.run_stream_set_create_count == after_first, (
                f"runs without code publication recreated their AICore stream: "
                f"{after_first} -> {st_worker.run_stream_set_create_count}"
            )
        finally:
            st_worker.unregister(handle)

    def test_deduplicated_registration_does_not_invalidate_stream(self, st_platform, st_worker):
        """A content-hash dedup hit does not publish code or stale a stream."""
        if st_platform != "a2a3":
            pytest.skip("AICore code publication is an a2a3 onboard resource")

        chip_callable = self.build_callable(st_platform)
        first_handle = st_worker.register(chip_callable)
        duplicate_handle = None
        try:
            self._run_registered_with_lease(
                st_worker,
                first_handle,
                slot_id=0,
                generation=self._next_generation(),
            )
            warmed = st_worker.run_stream_set_create_count

            duplicate_handle = st_worker.register(chip_callable)
            self._run_registered_with_lease(
                st_worker,
                duplicate_handle,
                slot_id=0,
                generation=self._next_generation(),
            )
            assert st_worker.run_stream_set_create_count == warmed
        finally:
            if duplicate_handle is not None:
                st_worker.unregister(duplicate_handle)
            st_worker.unregister(first_handle)

    def _run_registered(self, worker, handle, *, subtract):
        params = self.CASES[0]["params"]
        test_args = self.generate_args(params)
        # Worker.run takes TensorArg args and materializes them in-process; the runtime.so-ABI POD is
        # the direct chip API's shape, used by the lease path below.
        args, output_names = _build_l2_ref_args(test_args, self.CALLABLE["orchestration"]["signature"], worker)
        golden_args = test_args.clone()
        self.compute_golden(golden_args, params, subtract=subtract)
        worker.run(handle, args, config=self._build_config(self.CASES[0].get("config", {})))
        _compare_outputs(test_args, golden_args, output_names, self.RTOL, self.ATOL)

    # The st_worker fixture is shared by every test in this class, and so is the
    # generation each slot has last admitted. Tests draw from one increasing
    # counter so they stay independent of the order pytest runs them in.
    _generation_counter = itertools.count(1)

    @classmethod
    def _next_generation(cls):
        return next(cls._generation_counter)

    def _run_registered_with_lease(self, worker, handle, *, slot_id, generation, subtract=False):
        params = self.CASES[0]["params"]
        test_args = self.generate_args(params)
        golden_args = test_args.clone()
        self.compute_golden(golden_args, params, subtract=subtract)
        chip_args, output_names = _build_chip_task_args(test_args, self.CALLABLE["orchestration"]["signature"])
        state = worker._resolve_handle(handle)
        worker._chip_worker._run_with_pipeline_lease(
            state.slot_id,
            chip_args,
            slot_id=slot_id,
            generation=generation,
            config=self._build_config(self.CASES[0].get("config", {})),
        )
        _compare_outputs(test_args, golden_args, output_names, self.RTOL, self.ATOL)

    def test_resident_code_images_reuse_one_stream(self, st_platform, st_worker):
        """A->B->A->B reuses one slot stream when no code is published.

        Each result is compared against the golden for the image that run asked
        for, so an AICore that executed the previous image's instructions shows
        up as wrong data rather than as a stream-count mismatch.
        """
        if st_platform != "a2a3":
            pytest.skip("AICore code images are an a2a3 onboard resource")

        add_handle = st_worker.register(self.build_callable(st_platform))
        sub_handle = st_worker.register(_SubtractCallable.compile_chip_callable(st_platform))
        try:
            self._run_registered_with_lease(
                st_worker,
                add_handle,
                slot_id=0,
                generation=self._next_generation(),
            )
            warmed = st_worker.run_stream_set_create_count
            for handle, subtract in (
                (sub_handle, True),
                (add_handle, False),
                (sub_handle, True),
            ):
                self._run_registered_with_lease(
                    st_worker,
                    handle,
                    slot_id=0,
                    generation=self._next_generation(),
                    subtract=subtract,
                )
                assert st_worker.run_stream_set_create_count == warmed
        finally:
            st_worker.unregister(sub_handle)
            st_worker.unregister(add_handle)

    def test_depth_two_slots_reuse_one_pair_across_resident_images(self, st_platform, st_worker):
        """Both slots submit on the one pair while resident images alternate."""
        if st_platform != "a2a3":
            pytest.skip("AICore code images are an a2a3 onboard resource")

        add_handle = st_worker.register(self.build_callable(st_platform))
        sub_handle = st_worker.register(_SubtractCallable.compile_chip_callable(st_platform))
        try:
            self._run_registered_with_lease(
                st_worker,
                add_handle,
                slot_id=0,
                generation=self._next_generation(),
            )
            # The pair is warm after the first launch. A slot is an index into
            # the resources preparation mutates, not into the stream set.
            warmed = st_worker.run_stream_set_create_count
            for slot_id, handle, subtract in (
                (1, sub_handle, True),
                (0, sub_handle, True),
                (1, add_handle, False),
                (0, add_handle, False),
            ):
                self._run_registered_with_lease(
                    st_worker,
                    handle,
                    slot_id=slot_id,
                    generation=self._next_generation(),
                    subtract=subtract,
                )
                assert st_worker.run_stream_set_create_count == warmed
        finally:
            st_worker.unregister(sub_handle)
            st_worker.unregister(add_handle)

    def test_code_publication_invalidates_the_pair_across_reregistration(self, st_platform, st_worker):
        """A->B->A uploads invalidate the pair a run on either slot warmed."""
        if st_platform != "a2a3":
            pytest.skip("AICore code publication is an a2a3 onboard resource")

        add_callable = self.build_callable(st_platform)
        add_handle = st_worker.register(add_callable)
        try:
            self._run_registered_with_lease(
                st_worker,
                add_handle,
                slot_id=0,
                generation=self._next_generation(),
            )
        finally:
            st_worker.unregister(add_handle)

        sub_handle = st_worker.register(_SubtractCallable.compile_chip_callable(st_platform))
        try:
            before_sub = st_worker.run_stream_set_create_count
            self._run_registered_with_lease(
                st_worker,
                sub_handle,
                slot_id=1,
                generation=self._next_generation(),
                subtract=True,
            )
            assert st_worker.run_stream_set_create_count == before_sub + 1
        finally:
            st_worker.unregister(sub_handle)

        add_handle = st_worker.register(add_callable)
        try:
            before_add = st_worker.run_stream_set_create_count
            self._run_registered_with_lease(
                st_worker,
                add_handle,
                slot_id=0,
                generation=self._next_generation(),
            )
            assert st_worker.run_stream_set_create_count == before_add + 1
        finally:
            st_worker.unregister(add_handle)

    def test_depth_two_slots_own_separate_resources(self, st_platform, st_worker):
        """Four live calls preserve their outputs while both execution slots are reused."""
        if not st_platform.startswith("a2a3"):
            pytest.skip("pipeline slot banks are an a2a3 resource")

        chip_worker = st_worker._chip_worker
        assert chip_worker.pipeline_depth == 2
        assert chip_worker.runtime_slot_count == 2
        records = []
        handle = st_worker.register(self.build_callable(st_platform))
        try:
            state = st_worker._resolve_handle(handle)
            signature = self.CALLABLE["orchestration"]["signature"]
            config = self._build_config(self.CASES[0].get("config", {}))
            stream_sets = st_worker.run_stream_set_create_count
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
                    state.slot_id, chip_args, slot_id=slot_id, generation=self._next_generation(), config=config
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
                expected_increment = 1 if st_platform == "a2a3" else 0
                assert st_worker.run_stream_set_create_count == stream_sets + expected_increment
            # host_build_graph serves every bank's graph heap from the Worker's
            # shared ScratchPool, so the per-slot runtime and staging addresses
            # above are the ownership evidence; a per-bank heap base is not.
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

    def test_unleased_runs_do_not_consume_lease_generations(
        self, st_platform, device_pool, _l2_worker_pool, _l2_poisoned
    ):
        """A fresh slot pool admits generation 1 after three unleased runs.

        Other methods can have advanced a pooled Worker's leases. Retire that
        owner and keep this fresh lifetime inside this method, so no class-owned
        Worker overlaps a later pooled method even when test order changes.
        """
        if st_platform != "a2a3":
            pytest.skip("pipeline slot leases are an a2a3 onboard resource")

        with _l2_worker_pool.isolated(st_platform, self._st_runtime, device_pool, _l2_poisoned) as worker:
            handle = worker.register(self.build_callable(st_platform))
            try:
                for _ in range(3):
                    self._run_registered(worker, handle, subtract=False)
                self._run_registered_with_lease(worker, handle, slot_id=0, generation=1)
            finally:
                worker.unregister(handle)

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
class TestRunStreamReuseTmr(SceneTestCase):
    """TMR preparation keeps the same publication-aware stream rule."""

    # Its own copies under the tensormap_and_ringbuffer tree: a source names one
    # runtime's Tensor, so it cannot be compiled under both.
    CALLABLE = _rebase_callable(TestRunStreamReuseHbg.CALLABLE, _TMR_KERNELS)
    CASES = TestRunStreamReuseHbg.CASES

    generate_args = TestRunStreamReuseHbg.generate_args
    compute_golden = TestRunStreamReuseHbg.compute_golden
    _run_registered = TestRunStreamReuseHbg._run_registered

    def test_repeated_runs_without_publication_reuse_aicore_stream(self, st_platform, st_worker):
        if st_platform != "a2a3":
            pytest.skip("run stream sets are an a2a3 onboard resource")

        handle = st_worker.register(self.build_callable(st_platform))
        try:
            self._run_registered(st_worker, handle, subtract=False)
            after_first = st_worker.run_stream_set_create_count
            for _ in range(_REPEATED_RUNS - 1):
                self._run_registered(st_worker, handle, subtract=False)
            assert st_worker.run_stream_set_create_count == after_first
        finally:
            st_worker.unregister(handle)
