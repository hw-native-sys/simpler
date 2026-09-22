#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Several diagnostic runs on one worker, each with its own output prefix.

The collectors are resident: their device resources and threads are created on
the first run that needs them and released at ``finalize_device``. Everything
that describes a single run therefore has to be reset between runs, and nothing
in the framework exercises that — ``--rounds > 1`` disables every diagnostic
channel (``effective_diagnostic_options``), so "one worker, several runs, DFX
on" does not otherwise occur in the test suite.

Two distinct failures are covered, both silent — they produce *wrong artifacts*
rather than an error:

* **Leaked per-run state.** A record vector or device counter carried over makes
  a run's artifact describe an earlier one. For ``deps.json`` a leaked device
  counter makes ``reconcile_counters`` compare one run's collected count against
  a device total accumulated over both, which suppresses the export entirely.
  For the swimlane the observed shape is subtler than duplication: the merge of
  the collector shards is latched by a "already merged" flag, so a second run
  re-exports the *first run's records verbatim* — same count, same everything.
  A count or non-emptiness check therefore cannot see it, and neither can
  comparing the two runs' payloads, which are identical by construction. What
  separates them is time: the runs are sequential, so their record windows
  cannot overlap, and a re-export of stale records lands in the earlier window.

* **A latched pool shape.** Collector pools are seeded for the core and
  AICPU-thread counts of the run that built them, and a core's recycled lane is
  assigned modulo the thread count. A run with different counts must rebuild
  them, so the last run here asks for a different ``aicpu_thread_num`` and must
  still produce a complete artifact.

Scope, stated plainly: this catches dep_gen and the swimlane export. The other
three channels (pmu, args_dump, scope_stats) reconcile with a WARN and export
anyway, so a leaked counter there does not change their artifacts and this test
would not see it — their resets rest on being structurally identical to the two
that are covered. Asserting on the reconcile warnings is not an option: they are
emitted by the C++ HostLogger, which does not route through Python's logging, so
a ``caplog`` assertion silently matches nothing (measured, not assumed).

The collectors are shared code, so one arch covers them. It is a2a3 because that
is the arch with a host-map path: a resident collector that leaks a
halHostRegister mapping across runs fails the next one at rc=8, and only the
onboard a2a3 lane can observe it.
"""

import json

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _build_l2_ref_args, build_output_prefix

KERNELS_BASE = "../../../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"

# Runs 0 and 1 share a shape, so residency keeps their pools. Run 2 changes it,
# which forces the collectors to rebuild.
RUN_THREAD_COUNTS = (2, 2, 3)

EXPECTED_TASKS = 5
EXPECTED_EDGES = 6

# An aicore_tasks record is
# [core_id, task_token_raw, reg_task_id, start_cycles, end_cycles,
#  receive_to_start_cycles, run_epoch].
RECORD_START = 3
RECORD_END = 4
RECORD_EPOCH = 6


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestCollectorResidency(SceneTestCase):
    """Drive two runs through one worker with dep_gen and swimlane on."""

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
            "name": "repeated_runs",
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

    @staticmethod
    def _aicore_records(prefix):
        path = prefix / "chip_swimlane_records.json"
        assert path.exists(), f"chip_swimlane_records.json missing under {prefix}"
        with path.open() as f:
            return json.load(f).get("aicore_tasks", [])

    @staticmethod
    def _deps(prefix):
        path = prefix / "deps.json"
        assert path.exists(), (
            f"deps.json missing under {prefix}. A resident collector whose device counters were "
            f"not reset fails reconcile_counters() and skips the export."
        )
        with path.open() as f:
            return json.load(f)

    def test_run(self, st_platform, st_worker, request):
        # Deliberately does NOT call super().test_run(): the standard loop routes
        # repeats through --rounds, which turns the diagnostics off. This drives
        # Worker.run directly so every run keeps dep_gen and swimlane enabled.
        cases = self._matching_cases(st_platform, request)
        if not cases:
            return
        case = cases[0]

        callable_obj = self.build_callable(st_platform)
        handle = st_worker.register(callable_obj)
        orch_sig = self.CALLABLE.get("orchestration", {}).get("signature", [])

        artifacts = []
        for run_index, thread_count in enumerate(RUN_THREAD_COUNTS):
            prefix = build_output_prefix(f"{type(self).__name__}_{case['name']}_run{run_index}")
            args = self.generate_args(case.get("params", {}))
            chip_args, _output_names = _build_l2_ref_args(args, orch_sig, st_worker)

            config = CallConfig()
            # dep_gen is a bool; chip_swimlane is a level.
            config.enable_dep_gen = True
            config.enable_chip_swimlane = 1
            config.aicpu_thread_num = thread_count
            config.output_prefix = str(prefix)
            st_worker.run(handle, chip_args, config)
            artifacts.append(prefix)

        assert len(set(artifacts)) == len(artifacts), "the runs shared an output prefix; the test proves nothing"

        # Every run's set must be complete on its own, including the one that
        # rebuilt the pools at a new shape.
        for run_index, prefix in enumerate(artifacts):
            deps = self._deps(prefix)
            tasks, edges = deps.get("tasks", []), deps.get("edges", [])
            assert len(tasks) == EXPECTED_TASKS, (
                f"run {run_index}: expected {EXPECTED_TASKS} tasks, got {len(tasks)}. More than that "
                f"means an earlier run's records were not cleared from the in-memory set."
            )
            assert len(edges) == EXPECTED_EDGES, f"run {run_index}: expected {EXPECTED_EDGES} edges, got {len(edges)}"
            assert self._aicore_records(prefix), (
                f"run {run_index}: swimlane captured no records. An unreset collected counter makes "
                f"reconcile drop the whole run."
            )

        # The runs are sequential, so each one's records must lie entirely after
        # the previous one's. Re-exported stale records fail this while matching
        # on every other observable.
        previous_end = None
        previous_epoch = None
        for run_index, prefix in enumerate(artifacts):
            records = self._aicore_records(prefix)
            first_start = min(record[RECORD_START] for record in records)
            last_end = max(record[RECORD_END] for record in records)
            # The time window is a proxy; the stamped identity is the direct
            # discriminator. A re-exported stale record carries the epoch of the
            # run that produced it, so it cannot pass as this run's.
            epochs = {record[RECORD_EPOCH] for record in records}
            assert len(epochs) == 1, f"run {run_index}'s export mixes epochs {sorted(epochs)}"
            epoch = epochs.pop()
            assert epoch != 0, f"run {run_index}'s records carry no run identity"
            if previous_epoch is not None:
                assert epoch > previous_epoch, (
                    f"run {run_index} exported epoch {epoch}, which does not follow run "
                    f"{run_index - 1}'s {previous_epoch}. A resident collector that re-exports the "
                    f"previous run's merged records shows up here as a repeated epoch."
                )
            previous_epoch = epoch
            if previous_end is not None:
                assert first_start > previous_end, (
                    f"run {run_index}'s records start at {first_start}, which is not after run "
                    f"{run_index - 1}'s last record end {previous_end}. A resident collector that keeps "
                    f"the previous run's merged records re-exports them under this run's prefix."
                )
            previous_end = last_end


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
