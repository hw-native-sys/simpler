#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""host_build_graph's dependency graph, published off the submit path.

``host_build_graph`` builds the whole graph on the host, inside ``submit``, and
until now serialized and wrote ``deps.json`` there too. With
``collect_across_runs`` the finished graph is handed to a background writer
instead: the hand-off still happens on the building thread — which is the whole
reason it happens where it does — and the file appears by ``flush_diagnostics()``.

Two ordinary ``Worker.submit`` calls, each building a **different** device DAG
from the same orchestration source. Different graphs, not one graph twice: the
orchestration takes a shape selector and builds an unrelated topology for each,
so each run's artifact can only be checked against its own.

* ``chain`` — 64 tasks in a line, so 63 explicit edges.
* ``diamond`` — a root plus eight split/join layers: 25 tasks and 32 edges.

Both shapes declare their dependencies explicitly and take their tensor with no
dependency, so the graph holds exactly the edges the orchestration asked for.

Task ids are run-local, so the two runs repeat low ids. That is expected and not
asserted against; each graph is checked against its own topology, and the runs
are told apart by the directory each was given rather than by identity. That
directory is identified by set difference against a pre-run snapshot, the way
``test_dep_gen.py`` does it: an mtime comparison floors to whole seconds, so a
leftover from the same second would also match.
"""

import ctypes
import json

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _build_l3_task_args, _outputs_dir, _sanitize_for_filename

DAG_KERNELS = "../../single_core_dag/kernels"

# The two shapes `single_core_dag_orch.cpp` builds, and what each one submits.
# Derived from the orchestration, not copied from another suite: `build_chain`
# submits one task per slot and depends each on the one before it, and
# `build_diamonds` submits a root then eight layers of (left, right, join).
_CHAIN_CASE = 0
_CHAIN_TASKS = 64
_CHAIN_EDGES = _CHAIN_TASKS - 1

_DIAMOND_CASE = 1
_DIAMOND_LAYERS = 8
_DIAMOND_TASKS = 1 + _DIAMOND_LAYERS * 3
_DIAMOND_EDGES = _DIAMOND_LAYERS * 4  # left<-prior, right<-prior, join<-{left,right}

_TASK_SLOTS = 64
_CORE_TYPE_AIC = 0


def run_graph(orch, callables, task_args, config):
    """L3 orchestration: one ``submit_next_level`` per run, so one graph per run."""
    chip_args, _ = _build_l3_task_args(task_args, callables.dag_sig)
    callables.keep(chip_args)
    orch.submit_next_level(callables.dag, chip_args, config, worker=0)


@scene_test(level=3, runtime="host_build_graph", collect_across_runs=True)
class TestHbgDepGenAcrossRuns(SceneTestCase):
    """Two runs, two host-built DAGs, one flush, two independently owned graphs."""

    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": run_graph,
        "callables": [
            {
                "name": "dag",
                "orchestration": {
                    "source": f"{DAG_KERNELS}/orchestration/single_core_dag_orch.cpp",
                    "function_name": "aicpu_orchestration_entry",
                    "signature": [D.INOUT],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": f"{DAG_KERNELS}/check_dag.cpp",
                        "core_type": "aic",
                        "signature": [D.INOUT],
                    },
                    {
                        "func_id": 1,
                        "source": f"{DAG_KERNELS}/check_dag.cpp",
                        "core_type": "aiv",
                        "signature": [D.INOUT],
                    },
                ],
            },
        ],
    }

    CASES = [
        {
            "name": "chain",
            "platforms": ["a5sim", "a5"],
            "config": {"device_count": 1, "num_sub_workers": 0},
            "params": {"graph_case": _CHAIN_CASE, "task_count": _CHAIN_TASKS},
        },
        {
            "name": "diamond",
            "platforms": ["a5sim", "a5"],
            "config": {"device_count": 1, "num_sub_workers": 0},
            "params": {"graph_case": _DIAMOND_CASE, "task_count": _DIAMOND_TASKS},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("task_state", torch.zeros(_TASK_SLOTS * 8, dtype=torch.int64).share_memory_()),
            Scalar("graph_case", ctypes.c_int64(params["graph_case"])),
            Scalar("core_type", ctypes.c_int64(_CORE_TYPE_AIC)),
            Scalar("kernel_repeats", ctypes.c_int64(1)),
        )

    def compute_golden(self, args, params):
        count = params["task_count"]
        args.task_state[: count * 8 : 8] = torch.arange(1, count + 1, dtype=torch.int64)

    def test_run(self, st_platform, st_worker, request):
        # The scene loop runs each matching case, so the two cases are the two
        # ordinary runs. The snapshot comes first, so each case's output
        # directory is bound to this invocation rather than to whatever is there.
        before = {case["name"]: self._output_dirs(case["name"]) for case in self._matching_cases(st_platform, request)}
        super().test_run(st_platform, st_worker, request)
        if not self._effective_enable_dep_gen(request):
            return
        by_name = {c["name"]: c for c in self._matching_cases(st_platform, request)}
        if "chain" not in by_name or "diamond" not in by_name:
            return  # a filter selected one case; the cross-run property needs both

        # With retention on, a run returning does not mean its file exists.
        # This is the barrier, and it is the public one.
        st_worker.flush_diagnostics()

        chain = self._read_case_graph("chain", before["chain"])
        diamond = self._read_case_graph("diamond", before["diamond"])

        for name, graph, tasks, edges in (
            ("chain", chain, _CHAIN_TASKS, _CHAIN_EDGES),
            ("diamond", diamond, _DIAMOND_TASKS, _DIAMOND_EDGES),
        ):
            assert graph["runtime"] == "host_build_graph", name
            assert len(graph["tasks"]) == tasks, f"{name}: {len(graph['tasks'])} tasks, expected {tasks}"
            pairs = {(e["pred"], e["succ"]) for e in graph["edges"]}
            assert len(pairs) == edges, f"{name}: {len(pairs)} edges, expected {edges}"
            # Every edge lands inside this graph: an artifact that mixed two
            # runs' records would show an endpoint no task here declares.
            ids = {t["task_id"] for t in graph["tasks"]}
            unknown = {p for pair in pairs for p in pair} - ids
            assert not unknown, f"{name}: edges name task ids outside this graph: {sorted(unknown)}"
            assert all(e["source"] == "explicit" for e in graph["edges"]), (
                f"{name}: the shape declares its dependencies, so every edge is explicit"
            )

        # The chain is a line and the diamond is not, so the two runs did not
        # publish the same graph under two names.
        assert self._max_successors(chain) == 1, "the chain's tasks have one successor each"
        assert self._max_successors(diamond) == 2, "a diamond split has two"

    @staticmethod
    def _max_successors(graph):
        counts = {}
        for edge in graph["edges"]:
            counts[edge["pred"]] = counts.get(edge["pred"], 0) + 1
        return max(counts.values()) if counts else 0

    @staticmethod
    def _output_dirs(case_name):
        safe_label = _sanitize_for_filename(f"TestHbgDepGenAcrossRuns_{case_name}")
        return set(_outputs_dir().glob(f"{safe_label}_*"))

    def _read_case_graph(self, case_name, dirs_before_run):
        """This case's own deps.json, from the directory this invocation made."""
        # Only directories this invocation created: a leftover from an earlier
        # run must not stand in for a missing artifact, and two of them would
        # make the choice arbitrary.
        fresh = self._output_dirs(case_name) - dirs_before_run
        assert len(fresh) == 1, (
            f"case {case_name!r} should own exactly one new output directory from this run, found "
            f"{sorted(p.name for p in fresh)}"
        )
        out_dir = next(iter(fresh))
        # One chip invocation per run, so exactly one graph under it. The
        # directory is the identity; the file name is fixed inside it.
        deps_paths = sorted(out_dir.rglob("deps.json"))
        assert len(deps_paths) == 1, (
            f"flush_diagnostics() returned but {case_name!r} has {len(deps_paths)} graphs under "
            f"{out_dir} — expected exactly this run's one"
        )
        assert not deps_paths[0].with_suffix(".json.tmp").exists(), "a successful publication left its temporary behind"
        with deps_paths[0].open(encoding="utf-8") as handle:
            return json.load(handle)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
