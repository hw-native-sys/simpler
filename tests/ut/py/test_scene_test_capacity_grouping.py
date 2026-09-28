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

A Worker's `pipeline_depth` is granted once, before its first run, and cannot be changed
afterwards. Classes that ask for different capacities therefore cannot share a Worker, and the
standalone dispatcher groups by runtime and level — which says nothing about capacity. This is the
rule for that grouping; the pytest path builds one Worker per class and enforces the same
per-class coherence in `conftest.py`.
"""

import pytest

from simpler_setup.scene_test import (
    _class_pipeline_depth,
    _create_standalone_worker,
    _standalone_worker_groups,
)


def _case(name, **config):
    return {"name": name, "platforms": ["a2a3"], "config": config}


def _cls(name, runtime="host_build_graph", level=3):
    made = type(name, (), {})
    made._st_runtime = runtime
    made._st_level = level
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
        ("host_build_graph", 3, 3): [deep],
        ("host_build_graph", 3, 0): [shallow, control],
    }


def test_classes_asking_for_the_same_capacity_still_share_one_worker():
    first = _cls("FirstThree")
    second = _cls("SecondThree")
    selected = {
        first: [_case("a", pipeline_depth=3)],
        second: [_case("b", pipeline_depth=3)],
    }

    assert _standalone_worker_groups(selected) == {("host_build_graph", 3, 3): [first, second]}


def test_runtime_and_level_still_separate_groups():
    hbg = _cls("Hbg", runtime="host_build_graph")
    tmr = _cls("Tmr", runtime="tensormap_and_ringbuffer")
    level_two = _cls("LevelTwo", level=2)
    selected = {hbg: [_case("a")], tmr: [_case("b")], level_two: [_case("c")]}

    assert _standalone_worker_groups(selected) == {
        ("host_build_graph", 3, 0): [hbg],
        ("tensormap_and_ringbuffer", 3, 0): [tmr],
        ("host_build_graph", 2, 0): [level_two],
    }


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
