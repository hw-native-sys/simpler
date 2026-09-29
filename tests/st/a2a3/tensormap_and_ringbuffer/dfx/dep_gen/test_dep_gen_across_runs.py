#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""dep_gen background output on the real Worker path.

Two ordinary runs through the public ``collect_across_runs`` option, each
submitting a **different device orchestration**, then one
``flush_diagnostics()``. What this covers is the public wiring end to end: the
option reaches the chip child's collector, the flush is the barrier that says
the files exist, and each run's graph is published under the output identity
that run was given.

The two DAGs are genuinely different device orchestrations, not the same one
submitted twice — repeating a callable changes how many chip invocations there
are, not the graph inside one:

* ``vector`` — the vector example's five submissions ``t0..t4`` with the six
  dependencies its own header documents.
* ``barrier`` — a 65-producer many-to-one barrier, a consumer, and a tail pair
  that reaches one producer through an explicit dependency and through
  ownership of the tensor it created, so ``69`` tasks and ``65`` explicit
  edges into the barrier. 65 is past ``DEP_GEN_MAX_EXPLICIT_DEPS`` (64), so
  this run also drives the overflow chain wire format through the retained
  carrier and the background replay.

Task ids are **run-local** — ``TaskId`` is ``(ring << RING_SHIFT) | local_id``
and ``local_id`` restarts every run — so the two runs repeat low ids. That is
expected and deliberately not asserted against; each artifact is checked
against its own topology, and the runs are separated by the directory each was
given rather than by identity.

Output identity is this invocation's, not the newest thing on disk: each case
records a marker before it runs and requires exactly one output directory
created at or after it, so a leftover from an earlier run cannot stand in for a
missing artifact.
"""

import json
import time

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _build_l3_task_args, _outputs_dir, _sanitize_for_filename

VECTOR_KERNELS = "../../../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"
DUMMY_KERNELS = "../../dummy_task/kernels"

# The vector example's own header documents t0..t4 and
# t0->t1, t0->t2, t1->t3, t2->t3, t0->t4, t3->t4.
_VECTOR_TASKS = 5
_VECTOR_EDGES = 6

# `chain_barrier_orch.cpp` submits, in order: N producers, the barrier whose
# explicit dependency list names every one of them, a consumer ordering on the
# barrier, then a pair that exists for the fold this file checks — a task
# creating a runtime-owned output, and one reaching it through both an explicit
# WAIT and that ownership. The oracle is the whole submission list, not the part
# this file's headline is about. 65 > DEP_GEN_MAX_EXPLICIT_DEPS, so the
# barrier's dependency list spills into an overflow record.
_BARRIER_PRODUCERS = 65
_BARRIER_TAIL = ("barrier", "consumer", "output creator", "combined consumer")
_BARRIER_TASKS = _BARRIER_PRODUCERS + len(_BARRIER_TAIL)

_SENTINEL = 42.0
_INIT_VAL = -1.0


def run_graph(orch, callables, task_args, config):
    """L3 orchestration: submit the one device graph this run is for.

    Exactly one ``submit_next_level`` per run, so a run is one chip invocation
    and therefore one ``deps.json``. Which graph is decided by the args this
    case built: the barrier orchestration takes a scalar producer count, the
    vector one does not.
    """
    if hasattr(task_args, "n"):
        chip_args, _ = _build_l3_task_args(task_args, callables.barrier_chain_sig)
        callables.keep(chip_args)
        orch.submit_next_level(callables.barrier_chain, chip_args, config, worker=0)
        return
    chip_args, _ = _build_l3_task_args(task_args, callables.vector_kernel_sig)
    callables.keep(chip_args)
    orch.submit_next_level(callables.vector_kernel, chip_args, config, worker=0)


@scene_test(level=3, runtime="tensormap_and_ringbuffer", collect_across_runs=True)
class TestDepGenAcrossRuns(SceneTestCase):
    """Two runs, two device DAGs, one flush, two independently owned graphs."""

    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": run_graph,
        "callables": [
            {
                "name": "vector_kernel",
                "orchestration": {
                    "source": f"{VECTOR_KERNELS}/orchestration/example_orchestration.cpp",
                    "function_name": "aicpu_orchestration_entry",
                    "signature": [D.IN, D.IN, D.OUT],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": f"{VECTOR_KERNELS}/aiv/kernel_add.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.IN, D.OUT],
                    },
                    {
                        "func_id": 1,
                        "source": f"{VECTOR_KERNELS}/aiv/kernel_add_scalar.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT],
                    },
                    {
                        "func_id": 2,
                        "source": f"{VECTOR_KERNELS}/aiv/kernel_mul.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.IN, D.OUT],
                    },
                ],
            },
            {
                "name": "barrier_chain",
                "orchestration": {
                    "source": "kernels/orchestration/chain_barrier_orch.cpp",
                    "function_name": "aicpu_orchestration_entry",
                    "signature": [D.INOUT, D.INOUT],  # X, Y; the producer count goes as a scalar
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
            },
        ],
    }

    CASES = [
        {
            "name": "vector",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"device_count": 1, "num_sub_workers": 0},
            "params": {"graph": "vector"},
        },
        {
            "name": "barrier",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"device_count": 1, "num_sub_workers": 0, "aicpu_thread_num": 2},
            "params": {"graph": "barrier"},
        },
    ]

    def generate_args(self, params):
        if params["graph"] == "barrier":
            # Single-element reads/writes are enough: write_const writes index
            # 0 and copy_first reads index 0.
            return TaskArgsBuilder(
                TensorArg("x", torch.full((16,), _INIT_VAL, dtype=torch.float32).share_memory_()),
                TensorArg("y", torch.full((16,), _INIT_VAL, dtype=torch.float32).share_memory_()),
                Scalar("n", _BARRIER_PRODUCERS),
            )
        size = 128 * 128
        return TaskArgsBuilder(
            TensorArg("a", torch.full((size,), 2.0, dtype=torch.float32).share_memory_()),
            TensorArg("b", torch.full((size,), 3.0, dtype=torch.float32).share_memory_()),
            TensorArg("f", torch.zeros(size, dtype=torch.float32).share_memory_()),
        )

    def compute_golden(self, args, params):
        if params["graph"] == "barrier":
            args.x[0] = _SENTINEL
            args.y[0] = _SENTINEL
            return
        args.f[:] = (args.a + args.b + 1) * (args.a + args.b + 2) + (args.a + args.b)

    def test_run(self, st_platform, st_worker, request):
        # The scene loop runs each matching case, so the two cases are the two
        # ordinary runs — not `--rounds 2`, which zeroes every diagnostic.
        # Marker first, so each case's output directory is bound to this
        # invocation rather than to whatever is newest.
        marker = int(time.time())
        super().test_run(st_platform, st_worker, request)
        if not self._effective_enable_dep_gen(request):
            return
        cases = list(self._matching_cases(st_platform, request))
        by_name = {c["name"]: c for c in cases}
        if "vector" not in by_name or "barrier" not in by_name:
            return  # a filter selected one case; the cross-run property needs both

        # With retention on, a run returning does not mean its file exists.
        # This is the barrier, and it is the public one.
        st_worker.flush_diagnostics()

        vector = self._read_case_graph(by_name["vector"], marker)
        barrier = self._read_case_graph(by_name["barrier"], marker)

        for name, graph in (("vector", vector), ("barrier", barrier)):
            assert graph["runtime"] == "tensormap_and_ringbuffer", name

        # Each graph against its own topology. Repeated run-local task ids
        # across the two runs are expected and not asserted against.
        assert len(vector["tasks"]) == _VECTOR_TASKS, f"vector: {len(vector['tasks'])} tasks"
        assert len(self._edge_pairs(vector)) == _VECTOR_EDGES, f"vector: {len(self._edge_pairs(vector))} edges"

        assert len(barrier["tasks"]) == _BARRIER_TASKS, f"barrier: {len(barrier['tasks'])} tasks"
        # The barrier is the one task whose explicit predecessors are every
        # producer — the property the overflow chain has to survive.
        explicit_by_succ: dict[str, set] = {}
        for edge in barrier["edges"]:
            if edge.get("source") != "explicit":
                continue
            explicit_by_succ.setdefault(edge["succ"], set()).add(edge["pred"])
        barriers = [succ for succ, preds in explicit_by_succ.items() if len(preds) == _BARRIER_PRODUCERS]
        assert len(barriers) == 1, (
            f"barrier: expected one task with {_BARRIER_PRODUCERS} explicit predecessors, "
            f"got {[(s, len(p)) for s, p in explicit_by_succ.items()]}"
        )

        # The tail pair reaches one task for two reasons at once — an explicit
        # WAIT on it, and ownership of the tensor it created — and the runtime
        # folds both into a single edge whose flags carry both, rather than an
        # explicit edge plus a creator one. `tasks[]` is in submission order, so
        # the pair is the last two entries; the creator is independently the one
        # task holding a runtime-allocated OUTPUT arg, which is what makes
        # reading it positionally safe. This is the same property
        # `test_dep_gen_chain.py` pins on the default path, asserted here
        # through the background writer instead.
        tasks = barrier["tasks"]
        creators = [t for t in tasks if any(a["type"] == "OUTPUT" for a in t["args"])]
        assert [t["task_id"] for t in creators] == [tasks[-2]["task_id"]], (
            f"barrier: expected the second-to-last task to be the one runtime-output creator, got "
            f"{[t['task_id'] for t in creators]} against {tasks[-2]['task_id']}"
        )
        creator_id, combined_id = int(tasks[-2]["task_id"]), int(tasks[-1]["task_id"])
        combined_edges = [e for e in barrier["edges"] if int(e["pred"]) == creator_id and int(e["succ"]) == combined_id]
        assert len(combined_edges) == 1, (
            f"barrier: {creator_id}->{combined_id} should be one folded edge, got "
            f"{[(e['source'], e.get('flags')) for e in combined_edges]}"
        )
        assert combined_edges[0]["source"] == "explicit", combined_edges[0]
        assert combined_edges[0]["flags"] == ["wait", "retain"], combined_edges[0]

        # And the two runs did not publish the same graph.
        assert len(vector["tasks"]) != len(barrier["tasks"])

    @staticmethod
    def _edge_pairs(graph):
        return {(e["pred"], e["succ"]) for e in graph.get("edges", [])}

    def _read_case_graph(self, case, marker):
        """This case's own deps.json, from the directory this invocation made."""
        safe_label = _sanitize_for_filename(f"TestDepGenAcrossRuns_{case['name']}")
        # Only directories this invocation created: a leftover from an earlier
        # run must not stand in for a missing artifact, and two of them would
        # make the choice arbitrary.
        matches = [p for p in _outputs_dir().glob(f"{safe_label}_*") if p.stat().st_mtime >= marker]
        assert len(matches) == 1, (
            f"case {case['name']!r} should own exactly one output directory from this run, found "
            f"{[p.name for p in matches]}"
        )
        out_dir = matches[0]
        # One chip invocation per run, so one per-run directory holding one
        # graph. The name is fixed inside it; the directory is the identity.
        deps_paths = sorted(out_dir.glob("rank*/d*/deps.json"))
        assert len(deps_paths) == 1, (
            f"flush_diagnostics() returned but {case['name']!r} has {len(deps_paths)} graphs under "
            f"{out_dir} — expected exactly this run's one"
        )
        deps_path = deps_paths[0]
        assert not deps_path.with_suffix(".json.tmp").exists(), "a successful publication left its temporary behind"
        with deps_path.open(encoding="utf-8") as handle:
            return json.load(handle)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
