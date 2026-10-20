#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Which scene-test classes may share one Worker in the standalone `python test_x.py` path.

Two Worker properties are granted once, before the first run, and cannot be changed afterwards:
`pipeline_depth` and cross-run diagnostic retention. Classes that disagree on either therefore
cannot share a Worker, so both belong in the standalone dispatcher's grouping key alongside
runtime and level. Retention carries a second reason: with it on, a completed `run()` has not
yet written its diagnostic artifact, so a class that never asked for it must not be moved onto
that path by a neighbour that did. This is the rule for that grouping; the pytest path builds one
Worker per class and enforces the same per-class coherence in `conftest.py`.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from simpler_setup.scene_test import (
    _class_pipeline_depth,
    _create_standalone_worker,
    _scene_admission_kwargs,
    _standalone_worker_groups,
    scene_test,
)


def _case(name, **config):
    return {"name": name, "platforms": ["a2a3"], "config": config}


def _cls(name, runtime="host_build_graph", level=3, collect_across_runs=False):
    made = type(name, (), {})
    made._st_runtime = runtime
    made._st_level = level
    made._st_collect_across_runs = collect_across_runs
    return made


# ---------------------------------------------------------------------------
# One class, one capacity
# ---------------------------------------------------------------------------


def test_a_class_that_asks_for_nothing_reports_zero():
    # Zero is "unset", which the Worker turns into the standing default request.
    assert _class_pipeline_depth(_cls("Plain"), [_case("a"), _case("b")]) == 0


def test_a_class_with_no_selected_cases_reports_zero():
    assert _class_pipeline_depth(_cls("Empty"), []) == 0


def test_a_class_that_asks_uniformly_reports_that_capacity():
    cases = [_case("a", pipeline_depth=3), _case("b", pipeline_depth=3)]
    assert _class_pipeline_depth(_cls("Three"), cases) == 3


def test_a_class_whose_cases_disagree_is_refused_by_name():
    cases = [_case("a", pipeline_depth=3), _case("b")]
    with pytest.raises(SystemExit, match="Mixed mixes pipeline_depth values \\[0, 3\\]"):
        _class_pipeline_depth(_cls("Mixed"), cases)


# ---------------------------------------------------------------------------
# Grouping, which is what the standalone route actually does
# ---------------------------------------------------------------------------


def test_classes_asking_for_different_capacities_do_not_share_a_worker():
    """The mixed-module shape: one class asks for three, its neighbours ask for nothing.

    Runtime and level alone put these in one group, and one Worker cannot serve both — so the
    capacity belongs in the key. Every class still runs; none is handed a capacity it did not ask
    for, and the one that asked is not handed the smallest.
    """
    deep = _cls("TestThreeRunCapacity")
    shallow = _cls("TestEarlyEnqueueDepthTwo")
    control = _cls("TestEarlyEnqueueDepthOneControl")
    selected = {
        deep: [_case("three_run_capacity", pipeline_depth=3, launch_depth=3)],
        shallow: [_case("early_enqueue", launch_depth=2)],
        control: [_case("control")],
    }

    groups = _standalone_worker_groups(selected)

    assert groups == {
        ("host_build_graph", 3, 3, False, None): [deep],
        ("host_build_graph", 3, 0, False, None): [shallow, control],
    }


def test_classes_asking_for_the_same_capacity_still_share_one_worker():
    first = _cls("FirstThree")
    second = _cls("SecondThree")
    selected = {
        first: [_case("a", pipeline_depth=3)],
        second: [_case("b", pipeline_depth=3)],
    }

    assert _standalone_worker_groups(selected) == {("host_build_graph", 3, 3, False, None): [first, second]}


def test_runtime_and_level_still_separate_groups():
    hbg = _cls("Hbg", runtime="host_build_graph")
    tmr = _cls("Tmr", runtime="tensormap_and_ringbuffer")
    level_two = _cls("LevelTwo", level=2)
    selected = {hbg: [_case("a")], tmr: [_case("b")], level_two: [_case("c")]}

    assert _standalone_worker_groups(selected) == {
        ("host_build_graph", 3, 0, False, None): [hbg],
        ("tensormap_and_ringbuffer", 3, 0, False, None): [tmr],
        ("host_build_graph", 2, 0, False, None): [level_two],
    }


def test_classes_disagreeing_on_cross_run_retention_do_not_share_a_worker():
    """Otherwise identical classes, one retaining diagnostics across runs and one not.

    Everything else about them matches, so runtime, level and capacity alone put them in one
    group — and the setting decides whether a returned `run()` has already written the class's
    diagnostic artifact. Neither answer is safe for the class that chose the other.
    """
    retaining = _cls("TestDepGenAcrossRuns", collect_across_runs=True)
    plain = _cls("TestDepGenSingleRun")
    selected = {retaining: [_case("across")], plain: [_case("single")]}

    assert _standalone_worker_groups(selected) == {
        ("host_build_graph", 3, 0, True, None): [retaining],
        ("host_build_graph", 3, 0, False, None): [plain],
    }


def test_a_class_that_never_declared_retention_groups_as_not_retaining():
    # Every class the decorator built carries the attribute; one assembled by hand does not, and
    # the absent case is the conservative one.
    bare = type("Bare", (), {})
    bare._st_runtime = "host_build_graph"
    bare._st_level = 3

    assert _standalone_worker_groups({bare: [_case("a")]}) == {("host_build_graph", 3, 0, False, None): [bare]}


# ---------------------------------------------------------------------------
# The Worker builder refuses a mixed group rather than reconciling it
# ---------------------------------------------------------------------------


def test_building_a_worker_for_a_mixed_group_is_refused_before_any_device_is_touched():
    # Defence in depth for the partition above: a caller that groups by runtime and level alone
    # is told, rather than handed a Worker at the smaller capacity. `args` is never read — the
    # refusal comes first, which is what keeps it off the device.
    deep = _cls("Deep")
    shallow = _cls("Shallow")
    selected = {deep: [_case("a", pipeline_depth=3)], shallow: [_case("b")]}

    with pytest.raises(SystemExit, match="one Worker cannot serve pipeline_depth values \\[0, 3\\]"):
        _create_standalone_worker([deep, shallow], 3, None, selected)


def test_building_a_worker_for_a_retention_mixed_group_is_refused_the_same_way():
    retaining = _cls("Retaining", collect_across_runs=True)
    plain = _cls("Plain")
    selected = {retaining: [_case("a")], plain: [_case("b")]}

    with pytest.raises(SystemExit, match="one Worker cannot serve both cross-run retention settings"):
        _create_standalone_worker([retaining, plain], 3, None, selected)


@pytest.mark.parametrize("control", ["pipeline_depth", "launch_depth", "pending_run_depth", "workspace_budget_bytes"])
@pytest.mark.parametrize("value", [None, 0, 1])
def test_automatic_metadata_rejects_even_zero_or_none_explicit_controls(control, value):
    cls = _cls("Automatic")
    cls.CASES = [_case("contradiction", **{control: value})]
    with pytest.raises(ValueError, match="automatic_admission contradicts"):
        scene_test(level=3, runtime="host_build_graph", automatic_admission=True)(cls)


def test_automatic_metadata_requires_a_root_capable_level():
    cls = _cls("LevelTwo", level=2)
    cls.CASES = [_case("a")]
    with pytest.raises(ValueError, match="requires level 3"):
        scene_test(level=2, runtime="host_build_graph", automatic_admission=True)(cls)


def test_automatic_and_explicit_classes_never_share_a_standalone_worker():
    automatic = _cls("Automatic")
    automatic._st_automatic_admission = True
    explicit = _cls("Explicit")
    selected = {automatic: [_case("a")], explicit: [_case("b")]}
    assert list(_standalone_worker_groups(selected).values()) == [[automatic], [explicit]]
    with pytest.raises(SystemExit, match="both automatic and explicit"):
        _create_standalone_worker([automatic, explicit], 3, None, selected)


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("factory", ["pytest", "standalone"])
def test_both_worker_factories_preserve_exact_admission_kwargs(monkeypatch, automatic, factory):
    import simpler.worker as worker_mod  # noqa: PLC0415

    cls = _cls("FactoryScene")
    cls._st_automatic_admission = automatic
    cls.CALLABLE = {}
    cls.CASES = [_case("a")]
    captured = []

    class FakeWorker:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        def init(self, prewarm_config=None):
            pass

        def close(self):
            pass

    monkeypatch.setattr(worker_mod, "Worker", FakeWorker)
    if factory == "standalone":
        _create_standalone_worker(
            [cls],
            3,
            SimpleNamespace(
                platform="a2a3",
                device=0,
                device_ids=[0],
                enable_chip_swimlane=0,
                dump_args=0,
                enable_pmu=0,
                enable_dep_gen=False,
                enable_scope_stats=False,
                enable_swimlane_overhead=False,
            ),
            {cls: cls.CASES},
        )
    else:
        path = Path(__file__).resolve().parents[3] / "conftest.py"
        spec = importlib.util.spec_from_file_location("_automatic_scene_conftest", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pool = SimpleNamespace(allocate=lambda count: [0], release=lambda ids: None)
        request = SimpleNamespace(
            node=SimpleNamespace(cls=cls), config=SimpleNamespace(getoption=lambda name, default=None: default)
        )
        fixture = module.st_worker.__wrapped__(request, "a2a3", pool, {}, set())
        next(fixture)
        with pytest.raises(StopIteration):
            next(fixture)
    controls = {"pipeline_depth", "launch_depth", "pending_run_depth", "workspace_budget_bytes"}
    assert len(captured) == 1
    actual = {key: value for key, value in captured[0].items() if key in controls}
    assert actual == ({} if automatic else {"launch_depth": 1})


def test_shared_policy_preserves_explicit_capacity_and_depth():
    assert _scene_admission_kwargs(_cls("Configured"), [_case("a", launch_depth=2, pipeline_depth=5)]) == {
        "pipeline_depth": 5,
        "launch_depth": 2,
    }


def test_each_automatic_class_owns_its_cold_start():
    first, second = _cls("First"), _cls("Second")
    first._st_automatic_admission = second._st_automatic_admission = True
    selected = {first: [_case("one")], second: [_case("two")]}
    assert list(_standalone_worker_groups(selected).values()) == [[first], [second]]
