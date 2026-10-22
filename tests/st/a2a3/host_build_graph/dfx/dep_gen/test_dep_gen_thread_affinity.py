#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A host_build_graph run emits its own graph even when another thread drains it.

host_build_graph captures the dependency graph while the orchestrator runs on the
host, inside ``submit`` (bind), and the graph lives in state private to the thread
that ran it. Emitting it at the end of that same bind is what keeps the write on
the capturing thread; emitting at drain instead would not, because the run lane
serializes with a mutex — mutual exclusion, not thread affinity.

This drives the shape that separates the two: ``submit`` at depth one blocks the
second submission and drains its predecessor before admitting itself, so a second
thread's submit is what reaps the first thread's run. With the write at drain the
first run's ``deps.json`` was never produced — emit reported "no capture on this
thread" (-3) and returned.

Both submissions are ordinary public API on one Worker: ``submit`` returns without
waiting and admits concurrent callers through an operation lease, so nothing here
reaches past the supported surface.
"""

import json
import threading
from pathlib import Path

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _build_l2_ref_args

KERNELS_BASE = "kernels"

EXPECTED_TASKS = 5
EXPECTED_EDGES = 6


@scene_test(level=2, runtime="host_build_graph")
class TestDepGenHostGraphThreadAffinity(SceneTestCase):
    """Submit from one thread, let another thread's submit drain it."""

    CALLABLE = {
        "orchestration": {
            "source": f"{KERNELS_BASE}/orchestration/example_orchestration.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": f"{KERNELS_BASE}/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": f"{KERNELS_BASE}/aiv/kernel_add_scalar.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "source": f"{KERNELS_BASE}/aiv/kernel_mul.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "drained_by_another_thread",
            "platforms": ["a2a3sim", "a2a3"],
            "manual": ["a2a3sim"],
            "params": {},
        },
    ]

    def generate_args(self, params):
        SIZE = 128 * 128
        return TaskArgsBuilder(
            TensorArg("a", torch.full((SIZE,), 2.0, dtype=torch.float32)),
            TensorArg("b", torch.full((SIZE,), 3.0, dtype=torch.float32)),
            TensorArg("f", torch.zeros(SIZE, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        args.f[:] = (args.a + args.b + 1) * (args.a + args.b + 2) + (args.a + args.b)

    def test_run(self, st_platform, st_worker, request):
        # Deliberately does NOT call super().test_run(): the standard loop submits
        # and waits on one thread, which is the shape that already works.
        cases = self._matching_cases(st_platform, request)
        if not cases:
            return
        case = cases[0]

        callable_obj = self.build_callable(st_platform)
        handle = st_worker.register(callable_obj)
        orch_sig = self.CALLABLE.get("orchestration", {}).get("signature", [])

        def make_config():
            config = CallConfig()
            config.enable_dep_gen = True
            return config

        # The resolved descriptors in chip_args carry addresses into the torch
        # tensors' storage, so the builder has to outlive the run. Dropping it
        # frees that storage under the still-pending bind, which is a
        # heap-use-after-free in the test rather than a finding about the runtime.
        keepalive = []

        def build_args():
            args = self.generate_args(case.get("params", {}))
            chip_args, _names = _build_l2_ref_args(args, orch_sig, st_worker)
            keepalive.append((args, chip_args))
            return chip_args

        # This thread orchestrates the first run, so its graph lands in this
        # thread's capture state. It stays alive for the whole test: a thread that
        # exited would destroy that state for a different reason.
        first = st_worker.submit(handle, build_args(), make_config())
        first_prefix = Path(st_worker.capture_output_prefix)

        # The second submit blocks at depth one and drains the first run — on this
        # thread, not the one that orchestrated it.
        drain_error: list[BaseException] = []
        second_prefixes: list[Path] = []

        def submit_from_other_thread():
            try:
                second = st_worker.submit(handle, build_args(), make_config())
                second_prefixes.append(Path(st_worker.capture_output_prefix))
                second.wait()
            except BaseException as exc:  # noqa: BLE001 -- reported on the main thread
                drain_error.append(exc)

        other = threading.Thread(target=submit_from_other_thread, name="dep-gen-drainer")
        other.start()
        other.join(timeout=300)
        assert not other.is_alive(), "the second submit did not return within 300s"
        if drain_error:
            raise drain_error[0]
        first.wait()
        [second_prefix] = second_prefixes

        for label, prefix in (("first", first_prefix), ("second", second_prefix)):
            deps_path = prefix / "deps.json"
            assert deps_path.exists(), (
                f"{label} run: deps.json missing under {prefix}. host_build_graph holds the "
                f"captured graph in thread-local state between orchestration and emit, so a "
                f"drain on a different thread finds no capture and writes nothing."
            )
            with deps_path.open() as f:
                deps = json.load(f)
            assert len(deps.get("tasks", [])) == EXPECTED_TASKS, (
                f"{label} run: expected {EXPECTED_TASKS} tasks, got {len(deps.get('tasks', []))}"
            )
            assert len(deps.get("edges", [])) == EXPECTED_EDGES, (
                f"{label} run: expected {EXPECTED_EDGES} edges, got {len(deps.get('edges', []))}"
            )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
