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

Two ordinary ``Worker.submit`` calls, each submitting a **different** device
orchestration. The two are the pair ``test_dep_gen.py`` already validates on the
default path, so each run's topology is known independently of this file:

* ``vector`` — the vector example's five submissions, six edges, all ``creator``.
* ``predicated`` — ``predicated_dispatch``'s four tasks and three edges, one
  ``explicit`` and two ``tensormap``.

Between them the two runs carry all three edge kinds and the producer-side
geometry only a tensormap edge has, so the background hand-off is checked
against every field the schema can hold, not just the simplest one.

Task ids are run-local, so the two runs repeat low ids. That is expected and not
asserted against; each graph is checked against its own topology. The directory
each run owns is identified by set difference against a pre-run snapshot, the way
``test_dep_gen.py`` does it: an mtime comparison floors to whole seconds, so a
leftover from the same second would also match.
"""

import json

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _build_l3_task_args, _outputs_dir, _sanitize_for_filename

VECTOR_KERNELS = "kernels"
PREDICATED_KERNELS = "../../predicated_dispatch/kernels"

# The vector example's own header documents t0..t4 and t0->t1, t0->t2, t1->t3,
# t2->t3, t0->t4, t3->t4.
_VECTOR_TASKS = 5
_VECTOR_EDGES = 6

# predicated_dispatch_orch.cpp, in submit order: t0 WRITE_GATE(gate),
# t1 WRITE_CONST(X), t2 CLOBBER(X, set_dependencies={t0}), t3 COPY_FIRST(X -> Y).
# t2 declares t0 and reads X after t1 wrote it; t3 reads X after t2.
_PREDICATED_TASKS = 4
_PREDICATED_EDGES = {(0, 2, "explicit"), (1, 2, "tensormap"), (2, 3, "tensormap")}


def run_graph(orch, callables, task_args, config):
    """L3 orchestration: one ``submit_next_level`` per run, so one graph per run.

    Which graph is decided by the args this case built — the predicated
    orchestration takes a case scalar, the vector one does not.
    """
    if hasattr(task_args, "case"):
        chip_args, _ = _build_l3_task_args(task_args, callables.predicated_sig)
        callables.keep(chip_args)
        orch.submit_next_level(callables.predicated, chip_args, config, worker=0)
        return
    chip_args, _ = _build_l3_task_args(task_args, callables.vector_sig)
    callables.keep(chip_args)
    orch.submit_next_level(callables.vector, chip_args, config, worker=0)


@scene_test(level=3, runtime="host_build_graph", collect_across_runs=True)
class TestHbgDepGenAcrossRuns(SceneTestCase):
    """Two runs, two host-built DAGs, one flush, two independently owned graphs."""

    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": run_graph,
        "callables": [
            {
                "name": "vector",
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
                "name": "predicated",
                "orchestration": {
                    "source": f"{PREDICATED_KERNELS}/orchestration/predicated_dispatch_orch.cpp",
                    "function_name": "aicpu_orchestration_entry",
                    "signature": [D.INOUT, D.INOUT, D.INOUT],  # X, Y, gate
                },
                "incores": [
                    {
                        "func_id": 0,
                        "name": "WRITE_CONST",
                        "source": f"{PREDICATED_KERNELS}/aic/kernel_write_const.cpp",
                        "core_type": "aic",
                        "signature": [D.INOUT],
                    },
                    {
                        "func_id": 1,
                        "name": "COPY_FIRST",
                        "source": f"{PREDICATED_KERNELS}/aic/kernel_copy_first.cpp",
                        "core_type": "aic",
                        "signature": [D.IN, D.INOUT],
                    },
                    {
                        "func_id": 2,
                        "name": "CLOBBER",
                        "source": f"{PREDICATED_KERNELS}/aic/kernel_clobber.cpp",
                        "core_type": "aic",
                        "signature": [D.INOUT],
                    },
                    {
                        "func_id": 3,
                        "name": "WRITE_GATE",
                        "source": f"{PREDICATED_KERNELS}/aic/kernel_write_gate.cpp",
                        "core_type": "aic",
                        "signature": [D.INOUT],
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
            "name": "predicated",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"device_count": 1, "num_sub_workers": 0, "aicpu_thread_num": 2},
            "params": {"graph": "predicated"},
        },
    ]

    def generate_args(self, params):
        if params["graph"] == "predicated":
            return TaskArgsBuilder(
                TensorArg("x", torch.full((16,), -1.0, dtype=torch.float32).share_memory_()),
                TensorArg("y", torch.full((16,), -1.0, dtype=torch.float32).share_memory_()),
                TensorArg("gate", torch.full((16,), -1, dtype=torch.int32).share_memory_()),
                Scalar("case", 2),
            )
        size = 128 * 128
        return TaskArgsBuilder(
            TensorArg("a", torch.full((size,), 2.0, dtype=torch.float32).share_memory_()),
            TensorArg("b", torch.full((size,), 3.0, dtype=torch.float32).share_memory_()),
            TensorArg("f", torch.zeros(size, dtype=torch.float32).share_memory_()),
        )

    def compute_golden(self, args, params):
        if params["graph"] == "predicated":
            # case=2 opens the gate, so the clobber dispatches and X/Y end at 999.0.
            args.gate[0] = 1
            args.x[0] = 999.0
            args.y[0] = 999.0
            return
        args.f[:] = (args.a + args.b + 1) * (args.a + args.b + 2) + (args.a + args.b)

    def test_run(self, st_platform, st_worker, request):
        # The scene loop runs each matching case, so the two cases are the two
        # ordinary runs. The snapshot comes first, so each case's output
        # directory is bound to this invocation rather than to whatever is there.
        before = {case["name"]: self._output_dirs(case["name"]) for case in self._matching_cases(st_platform, request)}
        super().test_run(st_platform, st_worker, request)
        if not self._effective_enable_dep_gen(request):
            return
        names = {c["name"] for c in self._matching_cases(st_platform, request)}
        if "vector" not in names or "predicated" not in names:
            return  # a filter selected one case; the cross-run property needs both

        # With retention on, a run returning does not mean its file exists.
        # This is the barrier, and it is the public one.
        st_worker.flush_diagnostics()

        vector = self._read_case_graph("vector", before["vector"])
        predicated = self._read_case_graph("predicated", before["predicated"])

        for name, graph in (("vector", vector), ("predicated", predicated)):
            assert graph["runtime"] == "host_build_graph", name
            ids = {t["task_id"] for t in graph["tasks"]}
            unknown = {p for e in graph["edges"] for p in (e["pred"], e["succ"])} - ids
            assert not unknown, f"{name}: edges name task ids outside this graph: {sorted(unknown)}"

        assert len(vector["tasks"]) == _VECTOR_TASKS, f"vector: {len(vector['tasks'])} tasks"
        assert len({(e["pred"], e["succ"]) for e in vector["edges"]}) == _VECTOR_EDGES, (
            f"vector: {len({(e['pred'], e['succ']) for e in vector['edges']})} edges"
        )

        assert len(predicated["tasks"]) == _PREDICATED_TASKS, f"predicated: {len(predicated['tasks'])} tasks"
        position = {t["task_id"]: i for i, t in enumerate(predicated["tasks"])}
        got = {(position[e["pred"]], position[e["succ"]], e["source"]) for e in predicated["edges"]}
        assert got == _PREDICATED_EDGES, (
            f"predicated: captured graph differs from the orchestration's dependencies: "
            f"missing={_PREDICATED_EDGES - got}, extra={got - _PREDICATED_EDGES}"
        )

        # A tensormap edge is the only kind carrying the producer's slice, and it
        # is read off the live entry during capture. An empty producer block here
        # would mean the hand-off dropped fields the synchronous path keeps.
        for edge in predicated["edges"]:
            if edge["source"] != "tensormap":
                continue
            assert edge.get("overlap"), f"tensormap edge {edge['pred']}->{edge['succ']} lost its overlap status"
            assert edge.get("producer_shape"), f"tensormap edge {edge['pred']}->{edge['succ']} lost producer_shape"
            assert "producer_start_offset" in edge and "producer_strides" in edge, (
                f"tensormap edge {edge['pred']}->{edge['succ']} lost its producer geometry"
            )

        # And the two runs did not publish the same graph under two names.
        assert len(vector["tasks"]) != len(predicated["tasks"])

    @staticmethod
    def _output_dirs(case_name):
        safe_label = _sanitize_for_filename(f"TestHbgDepGenAcrossRuns_{case_name}")
        return set(_outputs_dir().glob(f"{safe_label}_*"))

    def _read_case_graph(self, case_name, dirs_before_run):
        """This case's own deps.json, from the directory this invocation made."""
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
