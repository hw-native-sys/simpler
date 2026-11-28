#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""dep_gen overflow chain regression — submits with >64 explicit deps.

A submit with explicit_dep_count > DEP_GEN_MAX_EXPLICIT_DEPS (=64) spills the
extra deps into one or more DepGenOverflowRecord slots that overlay the same
buffer ring. Before the chain wire format, dep_gen would silently truncate
the tail in deps.json; this test verifies every explicit dep edge survives
the round-trip writer → host collector → replay → deps.json.

Test shape (chain_barrier_orch.cpp): N producers each INOUT X, then a dummy
barrier `set_dependencies_with_kinds({all N producer ids})`, then a consumer
`add_dep_wait(barrier_id)` reading X and writing Y. With N spanning
the {64, 65, 588, 589} boundaries we exercise:

  - n=64: base only (no chain) — sanity baseline
  - n=65: base + 1 overflow record (1 dep in overflow)
  - n=200: base + 1 overflow (136 deps in overflow)
  - n=589: base + 2 overflow (524 + 1 deps across two overflows)

Validation: the barrier task in deps.json must have exactly N predecessors,
all of which are the producer ids. The consumer must have one explicit
predecessor — the barrier. A final dummy pair verifies that an explicit WAIT
and creator retention for the same producer merge into one WAIT|RETAIN edge.
"""

import json
import time

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename

# Path is relative to this file's directory (the SceneTestCase build helper
# resolves CALLABLE sources from there). dummy_task already ships the two
# kernels we need (write_const + copy_first), so we reuse those instead of
# duplicating the source.
DUMMY_KERNELS = "../../dummy_task/kernels"


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestDepGenChain(SceneTestCase):
    """dep_gen overflow chain: many-to-one barrier with >64 explicit deps."""

    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/chain_barrier_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT],  # X, Y; N goes as scalar
        },
        "incores": [
            {
                "func_id": 0,
                "name": "WRITE_CONST",
                "source": f"{DUMMY_KERNELS}/aic/kernel_write_const.cpp",
                "core_type": "aic",
            },
            {
                "func_id": 1,
                "name": "COPY_FIRST",
                "source": f"{DUMMY_KERNELS}/aic/kernel_copy_first.cpp",
                "core_type": "aic",
            },
        ],
    }

    # Sentinel must match kernel_write_const (writes 42.0f).
    SENTINEL = 42.0
    INIT_VAL = -1.0

    # One scheduler owns every core, keeping the capture shard topology fixed
    # while the explicit-dependency count crosses overflow boundaries.
    CASES = [
        {
            "name": "n_64_no_chain",
            "platforms": ["a5sim", "a5"],
            "manual": ["a5sim"],
            "config": {"aicpu_thread_num": 2},
            "params": {"n": 64},
        },
        # Keep one representative boundary in the default Sim sweep; the
        # dedicated DFX step reruns it with dep-gen capture/replay enabled.
        {
            "name": "n_65_single_overflow",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 2},
            "params": {"n": 65},
        },
        {
            "name": "n_200_single_overflow",
            "platforms": ["a5sim", "a5"],
            "manual": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 2},
            "params": {"n": 200},
        },
        {
            "name": "n_589_two_overflow",
            "platforms": ["a5sim", "a5"],
            "manual": ["a5sim"],
            "config": {"aicpu_thread_num": 2},
            "params": {"n": 589},
        },
    ]

    def generate_args(self, params):
        # Single-element tensors are enough — kernel_write_const writes index 0
        # and kernel_copy_first reads index 0.
        x = torch.full((16,), self.INIT_VAL, dtype=torch.float32)
        y = torch.full((16,), self.INIT_VAL, dtype=torch.float32)
        return TaskArgsBuilder(
            TensorArg("x", x),
            TensorArg("y", y),
            Scalar("n", int(params["n"])),
        )

    def compute_golden(self, args, params):
        # Producers each write SENTINEL to X[0]; consumer copies X[0] -> Y[0].
        # Tensormap dependencies also order these accesses, so this value check
        # is an execution sanity gate; deps.json validates the explicit barrier.
        args.x[0] = self.SENTINEL
        args.y[0] = self.SENTINEL

    def test_run(self, st_platform, st_worker, request):
        # Marker taken before the run so _post_validate binds to this
        # invocation's output dir rather than a stale same-label leftover.
        run_marker = int(time.time())  # floor to whole seconds: safe if outputs/ ever lands on a coarse-mtime fs
        super().test_run(st_platform, st_worker, request)
        if not self._effective_enable_dep_gen(request):
            return
        for case in self._matching_cases(st_platform, request):
            self._post_validate(case, run_marker)

    def _post_validate(self, case, run_marker):
        """Verify every explicit dep edge survived the writer → replay round-trip.

        With dep_gen on, deps.json must contain N edges from the producers to
        the barrier task (one per `set_dependencies` entry the orchestration
        emitted), plus the consumer's one explicit edge back from the barrier
        and the final cross-source dedup probe. Pre-chain code would truncate
        the producer→barrier edge set to 16/64.
        """
        case_name = case["name"]
        n = int(case["params"]["n"])
        safe_label = _sanitize_for_filename(f"TestDepGenChain_{case_name}")
        outputs = _outputs_dir()
        matches = [p for p in outputs.glob(f"{safe_label}_*") if p.stat().st_mtime >= run_marker]
        assert matches, (
            f"no output dir for case {case_name!r} created this run — scene didn't run / capture regression?"
        )
        out_dir = max(matches, key=lambda p: p.stat().st_mtime)
        deps_path = out_dir / "deps.json"
        # _post_validate is only invoked when dep_gen was effectively enabled;
        # absence of deps.json means the host runner declined to emit it (most
        # likely reconcile_counters failed). Surface that as a hard failure
        # rather than silently passing — the whole point of this test is to
        # catch chain-side reconciliation regressions.
        assert deps_path.exists(), (
            f"dep_gen was enabled but {deps_path} is missing. Likely cause: "
            f"reconcile_counters() detected a count mismatch and suppressed deps.json emission. "
            f"Check the run log for 'dep_gen reconcile' warnings."
        )

        with deps_path.open() as f:
            deps = json.load(f)

        tasks = deps.get("tasks", [])
        raw_edges = deps.get("edges", [])
        # Project annotated edges → (pred, succ) — we only care about graph
        # structure here; the annot-vs-oracle agreement gate already ran
        # inside the replay before deps.json was written.
        edges = set()
        explicit_edges = set()
        for e in raw_edges:
            if not isinstance(e, dict):
                continue
            pred, succ = e.get("pred"), e.get("succ")
            if pred is None or succ is None:
                continue
            pair = (int(pred), int(succ))
            edges.add(pair)
            if e.get("source") == "explicit":
                explicit_edges.add(pair)

        # Identify the barrier task: it's the task with exactly n explicit-source
        # incoming edges. (Producers have 0; consumer has 1 — the one to barrier.)
        explicit_by_succ = {}
        for pred, succ in explicit_edges:
            explicit_by_succ.setdefault(succ, set()).add(pred)
        barrier_candidates = [tid for tid, preds in explicit_by_succ.items() if len(preds) == n]
        assert len(barrier_candidates) == 1, (
            f"expected exactly one task with {n} explicit predecessors "
            f"(the barrier), got {len(barrier_candidates)}: "
            f"{[(tid, len(preds)) for tid, preds in explicit_by_succ.items()]}"
        )
        barrier_id = barrier_candidates[0]
        barrier_preds = explicit_by_succ[barrier_id]

        # All N producer→barrier edges must be present. This is the chain
        # round-trip assertion: pre-chain code drops anything past index 63.
        assert len(barrier_preds) == n, f"barrier has {len(barrier_preds)} preds, expected {n}"

        # Check the raw list before indexing it so duplicate emitted edges
        # cannot be hidden by the producer-keyed map.
        barrier_explicit_edge_list = [
            e
            for e in raw_edges
            if isinstance(e, dict) and e.get("source") == "explicit" and int(e.get("succ", -1)) == barrier_id
        ]
        assert len(barrier_explicit_edge_list) == n, (
            f"barrier emitted {len(barrier_explicit_edge_list)} explicit edges, expected exactly {n}"
        )
        barrier_explicit_edges = {int(e["pred"]): e for e in barrier_explicit_edge_list}
        assert len(barrier_explicit_edges) == n

        # Derive producer order from tasks[] rather than TaskId numeric order.
        # The barrier alternates WAIT and WAIT|RETAIN kinds, validating the
        # parallel kind array across the base and every overflow record.
        barrier_task_idx = next((idx for idx, task in enumerate(tasks) if int(task["task_id"]) == barrier_id), None)
        assert barrier_task_idx is not None, f"barrier task {barrier_id} is missing from tasks[]"
        producer_ids = [
            int(task["task_id"]) for task in tasks[:barrier_task_idx] if int(task["task_id"]) in barrier_preds
        ]
        assert len(producer_ids) == n, f"found {len(producer_ids)} ordered barrier producers, expected {n}"
        for idx, producer_id in enumerate(producer_ids):
            edge = barrier_explicit_edges[producer_id]
            expected_flags = ["wait"] if idx % 2 == 0 else ["wait", "retain"]
            assert edge.get("flags") == expected_flags, (
                f"barrier dep {idx} flags are {edge.get('flags')}, expected {expected_flags}"
            )

        # Consumer must have one ordering-only explicit edge from the barrier.
        outgoing_explicit_from_barrier = [
            e for e in raw_edges if e.get("source") == "explicit" and int(e.get("pred", -1)) == barrier_id
        ]
        assert len(outgoing_explicit_from_barrier) == 1, (
            f"barrier {barrier_id} has {len(outgoing_explicit_from_barrier)} outgoing explicit edges, "
            f"expected 1 (the consumer)"
        )
        assert outgoing_explicit_from_barrier[0].get("flags") == ["wait"], (
            f"barrier ordering edge flags are {outgoing_explicit_from_barrier[0].get('flags')}, expected ['wait']"
        )

        # The final two dummy tasks exercise cross-source dedup: the consumer
        # explicitly waits on the producer and reads its runtime-owned output.
        assert len(tasks) >= 2
        combined_pred = int(tasks[-2]["task_id"])
        combined_succ = int(tasks[-1]["task_id"])
        combined_edges = [
            e for e in raw_edges if int(e.get("pred", -1)) == combined_pred and int(e.get("succ", -1)) == combined_succ
        ]
        assert len(combined_edges) == 1, (
            f"combined dependency {combined_pred}->{combined_succ} has {len(combined_edges)} edges, expected 1"
        )
        assert combined_edges[0].get("source") == "explicit"
        assert combined_edges[0].get("flags") == ["wait", "retain"], (
            f"combined dependency flags are {combined_edges[0].get('flags')}, expected ['wait', 'retain']"
        )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
