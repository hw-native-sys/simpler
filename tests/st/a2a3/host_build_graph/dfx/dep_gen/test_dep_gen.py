#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""dep_gen host-direct capture test for host_build_graph.

Runs the same ``vector_example`` orchestration the tensormap_and_ringbuffer
dep_gen test uses, with ``--enable-dep-gen``, and asserts the same 6 edges come
out. host_build_graph orchestrates on the host, so its graph is captured from
the orchestrator's own dependency path rather than replayed from a device ring —
sharing the orchestration with the device-orch test is what makes the two
answers comparable, and this test is where a divergence would surface.

Compute correctness is delegated to the upstream ``vector_example`` tests; this
case re-uses the orchestration purely to keep coverage on the capture pipeline.
"""

import json

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import TensorTransfer

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename

# graph carries all three edge sources, so it covers the tensormap (Step B) and
# explicit (STEP 1) capture hooks the vector_example case above cannot reach.
PREDICATED_KERNELS = "../../predicated_dispatch/kernels"
# graph_execution replays config-keyed Graph Definitions, so its deps.json names
# Graph bodies, one of them replayed by two Graph tasks.
GRAPH_EXECUTION = "../../graph_execution"
VECTOR_KERNELS = "../../vector_example/kernels/aiv"
# Graph bodies whose tasks carry a launch attribute: a dispatch predicate, and a
# sync-start cohort.
GRAPH_PREDICATED = "../../graph_predicated_dispatch"
GRAPH_SYNC_START = "../../graph_sync_start_early_dispatch"
TMR_SYNC_START_KERNELS = "../../../tensormap_and_ringbuffer/spmd_sync_start_early_dispatch/kernels"


def _output_dirs(test_cls_name, case_name):
    """Every output dir this case has ever produced, newest run included."""
    safe_label = _sanitize_for_filename(f"{test_cls_name}_{case_name}")
    return set(_outputs_dir().glob(f"{safe_label}_*"))


def _load_deps(test_cls_name, case_name, dirs_before_run):
    """Parse the deps.json under the output dir this invocation created.

    Identified by set difference against a pre-run snapshot rather than by
    timestamp: an mtime comparison floors to whole seconds, so a leftover dir
    from an earlier run in the same second also matches, and a run that emitted
    nothing would silently validate that stale deps.json.
    """
    fresh = _output_dirs(test_cls_name, case_name) - dirs_before_run
    assert fresh, (
        f"--enable-dep-gen is on and case {case_name!r} ran, but no output dir "
        f"was created this run — capture pipeline regression"
    )
    out_dir = max(fresh, key=lambda p: p.stat().st_mtime)
    deps_path = out_dir / "deps.json"
    assert deps_path.exists(), (
        f"--enable-dep-gen is on and {out_dir} exists, but deps.json was not produced — host-direct capture regression"
    )
    with deps_path.open() as f:
        return json.load(f)


def _edges_by_position(deps):
    """Project edges onto (submit_index(pred), submit_index(succ), source).

    tasks[] is in submit order, so position identifies a task independently of
    which ring the runtime placed it on.
    """
    position = {int(t["task_id"]): i for i, t in enumerate(deps["tasks"])}
    unknown = {
        (e["pred"], e["succ"])
        for e in deps["edges"]
        if int(e["pred"]) not in position or int(e["succ"]) not in position
    }
    assert not unknown, f"deps.json contains edges referencing unknown task ids: {unknown}"
    return {(position[int(e["pred"])], position[int(e["succ"])], e["source"]) for e in deps["edges"]}


def _assert_annotations(deps):
    """Every non-explicit edge names a registered tensor and its consumer slice."""
    tensor_ids = {int(t["tensor_id"]) for t in deps.get("tensors", []) if "tensor_id" in t}
    for e in deps["edges"]:
        assert isinstance(e, dict), f"deps.json edge must be an object, got {type(e).__name__}: {e!r}"
        if e.get("source") == "explicit":
            continue
        tid = e.get("tensor_id")
        assert tid is not None and int(tid) in tensor_ids, (
            f"edge {e.get('pred')}->{e.get('succ')} (source={e.get('source')}) "
            f"references tensor_id {tid} absent from tensors[]"
        )
        assert "consumer_shape" in e and "consumer_start_offset" in e and "consumer_strides" in e, (
            f"edge {e.get('pred')}->{e.get('succ')} (source={e.get('source')}) "
            f"missing consumer_shape/start_offset/strides"
        )


@scene_test(level=2, runtime="host_build_graph")
class TestDepGenHostBuildGraph(SceneTestCase):
    """Vector example on host_build_graph, run with dep_gen enabled."""

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/example_orchestration.cpp",
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
            "name": "default",
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
        # Run the standard scene-test loop, then assert the captured graph for
        # the cases that ran on this platform. Without the override the pytest
        # path would pass while capture produced nothing. The snapshot is taken
        # before the run so _post_validate binds to a directory this invocation
        # created rather than a leftover from an earlier one.
        dirs_before_run = {
            case["name"]: _output_dirs("TestDepGenHostBuildGraph", case["name"])
            for case in self._matching_cases(st_platform, request)
        }
        super().test_run(st_platform, st_worker, request)
        if not self._effective_enable_dep_gen(request):
            return
        for case in self._matching_cases(st_platform, request):
            self._post_validate(case, dirs_before_run[case["name"]])

    def _post_validate(self, case, dirs_before_run):
        """Assert deps.json holds the 6 edges of example_orchestration.cpp."""
        deps = _load_deps("TestDepGenHostBuildGraph", case["name"], dirs_before_run)

        tasks = deps.get("tasks", [])
        assert len(tasks) == 5, f"expected 5 submitted tasks, got {len(tasks)}: {[t.get('task_id') for t in tasks]}"

        # example_orchestration.cpp, in submit order:
        #   t0: c = a + b   t1: d = c + 1   t2: e = c + 2   t3: g = d * e   t4: f = g + c
        # Every edge is born in creator retention: the intermediates are
        # runtime-allocated OUTPUT tensors, which register_task_outputs does not
        # put in the tensormap, so Step B has nothing to match here.
        got = _edges_by_position(deps)
        expected = {
            (0, 1, "creator"),
            (0, 2, "creator"),
            (1, 3, "creator"),
            (2, 3, "creator"),
            (0, 4, "creator"),
            (3, 4, "creator"),
        }
        assert got == expected, (
            f"captured graph differs from the orchestration's dependencies: "
            f"missing={expected - got}, extra={got - expected}"
        )

        _assert_annotations(deps)


@scene_test(level=2, runtime="host_build_graph")
class TestDepGenHostBuildGraphEdgeSources(SceneTestCase):
    """predicated_dispatch on host_build_graph: covers the tensormap + explicit hooks.

    The vector_example case above only produces creator edges, so on its own it
    leaves two of the three capture hooks unasserted. This orchestration declares
    a dependency (`set_dependencies` → explicit) and passes INOUT tensors through
    the tensormap (→ tensormap edges with producer-side geometry).
    """

    RTOL = 0
    ATOL = 0

    CALLABLE = {
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
    }

    # One scheduler owns every core, so the captured graph exercises the
    # predicate and explicit-dependency hooks without cross-shard merging.
    CASES = [
        {
            "name": "gate_open",
            "platforms": ["a2a3sim", "a2a3"],
            "manual": ["a2a3sim"],
            "config": {"aicpu_thread_num": 2},
            "params": {"case": 2},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("x", torch.full((16,), -1.0, dtype=torch.float32)),
            TensorArg("y", torch.full((16,), -1.0, dtype=torch.float32)),
            TensorArg("gate", torch.full((16,), -1, dtype=torch.int32)),
            Scalar("case", int(params["case"])),
        )

    def compute_golden(self, args, params):
        # case=2 opens the gate, so the clobber dispatches and X/Y end at 999.0.
        args.gate[0] = 1
        args.x[0] = 999.0
        args.y[0] = 999.0

    def test_run(self, st_platform, st_worker, request):
        dirs_before_run = {
            case["name"]: _output_dirs("TestDepGenHostBuildGraphEdgeSources", case["name"])
            for case in self._matching_cases(st_platform, request)
        }
        super().test_run(st_platform, st_worker, request)
        if not self._effective_enable_dep_gen(request):
            return
        for case in self._matching_cases(st_platform, request):
            self._post_validate(case, dirs_before_run[case["name"]])

    def _post_validate(self, case, dirs_before_run):
        deps = _load_deps("TestDepGenHostBuildGraphEdgeSources", case["name"], dirs_before_run)

        tasks = deps.get("tasks", [])
        assert len(tasks) == 4, f"expected 4 submitted tasks, got {len(tasks)}: {[t.get('task_id') for t in tasks]}"

        # predicated_dispatch_orch.cpp, in submit order:
        #   t0 WRITE_GATE(gate)  t1 WRITE_CONST(X)
        #   t2 CLOBBER(X, set_dependencies={t0})  t3 COPY_FIRST(X -> Y)
        # t2 declares t0 (explicit) and reads X after t1 wrote it (tensormap);
        # t3 reads X after t2 (tensormap).
        got = _edges_by_position(deps)
        expected = {(0, 2, "explicit"), (1, 2, "tensormap"), (2, 3, "tensormap")}
        assert got == expected, (
            f"captured graph differs from the orchestration's dependencies: "
            f"missing={expected - got}, extra={got - expected}"
        )

        _assert_annotations(deps)

        # A tensormap edge is the only one that carries the producer's slice, and
        # it is read off the live ChipTensorMapEntry before an INOUT+COVERED
        # lookup removes it — an empty producer block means that ordering broke.
        for e in deps["edges"]:
            if e["source"] != "tensormap":
                continue
            assert e.get("overlap"), f"tensormap edge {e['pred']}->{e['succ']} missing overlap status"
            assert e.get("producer_shape"), f"tensormap edge {e['pred']}->{e['succ']} missing producer_shape"
            assert "producer_start_offset" in e and "producer_strides" in e, (
                f"tensormap edge {e['pred']}->{e['succ']} missing producer_start_offset/producer_strides"
            )

        # An explicit edge is not tied to a tensor arg and carries no slice.
        for e in deps["edges"]:
            if e["source"] != "explicit":
                continue
            assert e["arg"] == -1, f"explicit edge {e['pred']}->{e['succ']} should have arg=-1, got {e['arg']}"
            assert "tensor_id" not in e, f"explicit edge {e['pred']}->{e['succ']} should carry no tensor_id"

        # Each task carries its launch attributes; only the clobber has a predicate.
        attrs = [(t["sync_start"], t["predicated"]) for t in tasks]
        assert attrs == [(False, False), (False, False), (False, True), (False, False)], attrs


@scene_test(level=2, runtime="host_build_graph")
class TestDepGenHostBuildGraphGraphBodies(SceneTestCase):
    """graph_execution on host_build_graph: the bodies its Graph tasks replay.

    A Graph task is one kernel-less entry in tasks[]. graph_definitions[] holds
    each Definition once, its tasks by index and its internal edges, and
    graph_shells[] names the one each Graph task replays.
    """

    RTOL = 1e-5
    ATOL = 1e-5

    CALLABLE = {
        "orchestration": {
            "source": f"{GRAPH_EXECUTION}/kernels/orchestration/graph_execution_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT, D.OUT, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": f"{VECTOR_KERNELS}/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": f"{VECTOR_KERNELS}/kernel_add_scalar.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "source": f"{VECTOR_KERNELS}/kernel_mul.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "graph_bodies",
            "platforms": ["a2a3sim", "a2a3"],
            "manual": ["a2a3sim"],
            "params": {"shape": (128 * 128,)},
        },
    ]

    def generate_args(self, params):
        shape = params["shape"]
        return TaskArgsBuilder(
            TensorArg("a", torch.full(shape, 2.0, dtype=torch.float32)),
            TensorArg("b", torch.full(shape, 3.0, dtype=torch.float32)),
            TensorArg("output_1", torch.zeros(shape, dtype=torch.float32)),
            TensorArg("output_3", torch.zeros(shape, dtype=torch.float32)),
            TensorArg("output_5", torch.zeros(shape, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        base = args.a + args.b
        args.output_1[:] = (base + 1.0) * (base + 2.0)
        args.output_3[:] = (base + 100.0) * (base + 4.0)
        args.output_5[:] = (base + 5.0) * (base + 6.0)

    def test_run(self, st_platform, st_worker, request):
        dirs_before_run = {
            case["name"]: _output_dirs("TestDepGenHostBuildGraphGraphBodies", case["name"])
            for case in self._matching_cases(st_platform, request)
        }
        super().test_run(st_platform, st_worker, request)
        if not self._effective_enable_dep_gen(request):
            return
        for case in self._matching_cases(st_platform, request):
            self._post_validate(case, dirs_before_run[case["name"]])

    def _post_validate(self, case, dirs_before_run):
        deps = _load_deps("TestDepGenHostBuildGraphGraphBodies", case["name"], dirs_before_run)

        # graph_execution_orch.cpp: a seed task, then three Graph tasks; the first
        # and the third share a config, and so one Definition.
        graph_tasks = [int(t["task_id"]) for t in deps["tasks"] if t["kernel_ids"] == [-1, -1, -1]]
        assert graph_tasks == [1, 2, 3], f"Graph tasks in tasks[]: {graph_tasks}"
        shells = {int(s["task_id"]): s["definition"] for s in deps.get("graph_shells", [])}
        assert shells == {1: 0, 2: 1, 3: 0}, f"graph_shells names the wrong bodies: {shells}"

        # Each body is the orchestration's sub-DAG: add, two add_scalar and mul,
        # with two kernel-less tasks.
        definitions = deps.get("graph_definitions", [])
        assert [d["definition"] for d in definitions] == [0, 1], definitions
        for d in definitions:
            kernels = [t["kernel_ids"] for t in d["tasks"]]
            assert [t["index"] for t in d["tasks"]] == list(range(6)), d["tasks"]
            assert kernels == [[-1, -1, -1], [-1, 0, -1], [-1, -1, -1], [-1, 1, -1], [-1, 1, -1], [-1, 2, -1]], kernels
            assert all(t["block_num"] == 1 and not t["sync_start"] and not t["predicated"] for t in d["tasks"])
            edges = sorted(map(tuple, d["edges"]))
            assert edges == [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5), (4, 5)], edges


class _GraphBodyAttrsCase(SceneTestCase):
    """Runs a Graph scene, then asserts the launch attributes of its body tasks.

    Compute correctness is the scene's own test's to check; this one checks only
    what deps.json records.
    """

    # (sync_start, predicated) per body task, by index.
    BODY_ATTRS: list = []

    def compute_golden(self, args, params):
        pass

    def compare_outputs(self, test_args, golden_args, output_names, params):
        pass

    def test_run(self, st_platform, st_worker, request):
        name = type(self).__name__
        dirs_before_run = {
            case["name"]: _output_dirs(name, case["name"]) for case in self._matching_cases(st_platform, request)
        }
        super().test_run(st_platform, st_worker, request)
        if not self._effective_enable_dep_gen(request):
            return
        for case in self._matching_cases(st_platform, request):
            definitions = _load_deps(name, case["name"], dirs_before_run[case["name"]]).get("graph_definitions", [])
            assert definitions, "deps.json records no Graph body"
            for d in definitions:
                got = [(t["sync_start"], t["predicated"]) for t in d["tasks"]]
                assert got == self.BODY_ATTRS, f"definition {d['definition']}: got {got}, expected {self.BODY_ATTRS}"


@scene_test(level=2, runtime="host_build_graph")
class TestDepGenHostBuildGraphGraphPredicated(_GraphBodyAttrsCase):
    """graph_predicated_dispatch: three of the body's five tasks run under a predicate."""

    CALLABLE = {
        "orchestration": {
            "source": f"{GRAPH_PREDICATED}/kernels/orchestration/graph_predicated_dispatch_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT] * 8 + [D.IN] * 4,  # x0..x3, y0..y3, gate0..gate3
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
                "signature": [D.OUT],
            },
            {
                "func_id": 4,
                "name": "CLOBBER_ALT",
                "source": f"{GRAPH_PREDICATED}/kernels/aic/kernel_clobber_alt.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "predicated_body",
            "platforms": ["a2a3sim", "a2a3"],
            "manual": ["a2a3sim"],
            "params": {},
        },
    ]

    # WRITE_GATE, WRITE_CONST, then CLOBBER, CLOBBER_ALT and COPY_FIRST under a predicate.
    BODY_ATTRS = [(False, False), (False, False), (False, True), (False, True), (False, True)]

    def generate_args(self, params):
        # Every gate closed: a predicate's value decides a dispatch, not the
        # attribute deps.json records.
        xy = [TensorArg(f"{n}{i}", torch.full((16,), -1.0, dtype=torch.float32)) for n in "xy" for i in range(4)]
        gates = [TensorArg(f"gate{i}", torch.zeros((16,), dtype=torch.int32)) for i in range(4)]
        return TaskArgsBuilder(*xy, *gates, *(Scalar(f"internal_gate{i}", 0) for i in range(4)))


@scene_test(level=2, runtime="host_build_graph")
class TestDepGenHostBuildGraphGraphSyncStart(_GraphBodyAttrsCase):
    """graph_sync_start_early_dispatch: the body's first task is a sync-start cohort."""

    CALLABLE = {
        "orchestration": {
            "source": f"{GRAPH_SYNC_START}/kernels/orchestration/graph_sync_start_early_dispatch_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": f"{TMR_SYNC_START_KERNELS}/aiv/kernel_spmd_write_slow.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "source": f"{GRAPH_EXECUTION}/kernels/aic/kernel_spmd_mix.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 2,
                "source": f"{GRAPH_EXECUTION}/kernels/aiv/kernel_spmd_mix.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 3,
                "source": f"{GRAPH_EXECUTION}/kernels/aiv/kernel_spmd_mix.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "sync_start_body",
            "platforms": ["a2a3sim", "a2a3"],
            "manual": ["a2a3sim"],
            "params": {},
        },
    ]

    # The sync-start MIX cohort, then the ordinary tail it feeds.
    BODY_ATTRS = [(True, False), (False, False)]

    def generate_args(self, params):
        # The orchestration's layout at its widest, 24 clusters: 82 cache lines.
        return TaskArgsBuilder(
            TensorArg("output", torch.full((82 * 16,), -1.0, dtype=torch.float32)),
            TensorArg("layout", torch.zeros(1, dtype=torch.int32), transfer=TensorTransfer.NONE),
        )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
