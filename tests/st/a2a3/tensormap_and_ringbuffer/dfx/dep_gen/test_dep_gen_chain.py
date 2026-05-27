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
barrier `set_dependencies({all N producer ids})`, then a consumer
`set_dependencies({barrier_id})` reading X and writing Y. With N spanning
the {64, 65, 390, 391} boundaries we exercise:

  - n=64: base only (no chain) — sanity baseline
  - n=65: base + 1 overflow record (1 dep in overflow)
  - n=200: base + 1 overflow (136 deps in overflow)
  - n=391: base + 2 overflow (326 + 1 deps across two overflows)

Validation: the barrier task in deps.json must have exactly N predecessors,
all of which are the producer ids. The consumer must have one explicit
predecessor — the barrier.
"""

import json

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test
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

    CASES = [
        {
            "name": "n_64_no_chain",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 2, "block_dim": 1},
            "params": {"n": 64},
        },
        {
            "name": "n_65_single_overflow",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 2, "block_dim": 1},
            "params": {"n": 65},
        },
        {
            "name": "n_200_single_overflow",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 2, "block_dim": 1},
            "params": {"n": 200},
        },
        {
            "name": "n_391_two_overflow",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 2, "block_dim": 1},
            "params": {"n": 391},
        },
    ]

    def generate_args(self, params):
        # Single-element tensors are enough — kernel_write_const writes index 0
        # and kernel_copy_first reads index 0.
        x = torch.full((16,), self.INIT_VAL, dtype=torch.float32)
        y = torch.full((16,), self.INIT_VAL, dtype=torch.float32)
        return TaskArgsBuilder(
            Tensor("x", x),
            Tensor("y", y),
            Scalar("n", int(params["n"])),
        )

    def compute_golden(self, args, params):
        # Producers each write SENTINEL to X[0]; consumer copies X[0] -> Y[0].
        # If the barrier didn't actually wait for all producers, the consumer
        # could race ahead and copy INIT_VAL instead — making the host check
        # a defacto sanity gate even before we look at deps.json.
        args.x[0] = self.SENTINEL
        args.y[0] = self.SENTINEL

    def test_run(self, st_platform, st_worker, request):
        super().test_run(st_platform, st_worker, request)
        if not self._effective_enable_dep_gen(request):
            return
        for case in self.CASES:
            if st_platform in case.get("platforms", []):
                self._post_validate(case)

    def _post_validate(self, case):
        """Verify every explicit dep edge survived the writer → replay round-trip.

        With dep_gen on, deps.json must contain N edges from the producers to
        the barrier task (one per `set_dependencies` entry the orchestration
        emitted), plus the consumer's one explicit edge back from the barrier.
        Pre-chain code would truncate the producer→barrier edge set to 16/64.
        """
        case_name = case["name"]
        n = int(case["params"]["n"])
        safe_label = _sanitize_for_filename(f"TestDepGenChain_{case_name}")
        outputs = _outputs_dir()
        matches = sorted(outputs.glob(f"{safe_label}_*"), key=lambda p: p.stat().st_mtime)
        assert matches, f"no output dir for case {case_name!r} — scene didn't run on this platform?"
        out_dir = matches[-1]
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
        assert deps.get("version") == 3, f"deps.json version {deps.get('version')} != 3"

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

        # Consumer must explicit-depend on the barrier — exactly one outgoing
        # explicit edge from the barrier.
        outgoing_explicit_from_barrier = {succ for pred, succ in explicit_edges if pred == barrier_id}
        assert len(outgoing_explicit_from_barrier) == 1, (
            f"barrier {barrier_id} has {len(outgoing_explicit_from_barrier)} outgoing explicit edges, "
            f"expected 1 (the consumer)"
        )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
