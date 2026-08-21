# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from _pytest.outcomes import Failed

from simpler_setup import SceneTestCase, SceneTestLevel, scene_level, scene_test
from simpler_setup.parallel_scheduler import JobResult
from simpler_setup.scene_test import _discover_module_test_classes

_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location("_root_conftest_for_scene_level_tests", _ROOT / "conftest.py")
assert _SPEC is not None and _SPEC.loader is not None
root_conftest = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(root_conftest)


class _FakeMarker:
    def __init__(self, name, *args):
        self.name = name
        self.args = args


class _FakeItem:
    def __init__(self, nodeid, *, cls=None, function=None, markers=()):
        self.nodeid = nodeid
        self.cls = cls
        self.function = function
        self._markers = list(markers)

    def iter_markers(self, name=None):
        return (marker for marker in self._markers if name is None or marker.name == name)

    def get_closest_marker(self, name):
        for marker in self._markers:
            if marker.name == name:
                return marker
        return None

    def add_marker(self, marker):
        self._markers.append(marker)


class _FakeHook:
    def __init__(self):
        self.deselected = []

    def pytest_deselected(self, items):
        self.deselected.extend(items)


class _FakeConfig:
    def __init__(self, **options):
        self.options = options
        self.hook = _FakeHook()

    def getoption(self, name, default=None):
        return self.options.get(name, self.options.get(name.lstrip("-"), default))


def test_scene_level_normalizes_int_and_enum():
    @scene_level(4)
    def network1_fn():
        return None

    @scene_level(SceneTestLevel.CHIP)
    def chip_fn():
        return None

    assert network1_fn._st_level is SceneTestLevel.NETWORK1
    assert chip_fn._st_level is SceneTestLevel.CHIP


def test_scene_test_accepts_int_levels():
    @scene_test(level=2, runtime="tensormap_and_ringbuffer")
    class L2:
        CALLABLE = {}
        CASES = []

    assert L2._st_level is SceneTestLevel.CHIP
    assert L2._st_runtime == "tensormap_and_ringbuffer"


def test_standalone_discovery_excludes_imported_scene_test_classes():
    imported_module = ModuleType("tmr_case")
    current_module = ModuleType("hbg_case")

    imported_cls = type("TestTmr", (SceneTestCase,), {"__module__": imported_module.__name__, "CASES": []})
    local_cls = type("TestHbg", (SceneTestCase,), {"__module__": current_module.__name__, "CASES": []})
    current_module._ImportedTmr = imported_cls
    current_module.TestHbg = local_cls

    assert _discover_module_test_classes(current_module) == [local_cls]


def test_invalid_scene_level_rejected():
    with pytest.raises(ValueError):
        scene_level(5)

    with pytest.raises(ValueError):
        scene_test(level=5, runtime="tensormap_and_ringbuffer")


def test_level4_filter_keeps_only_explicit_level4_functions():
    def plain_fn():
        return None

    @scene_level(SceneTestLevel.NETWORK1)
    def network1_fn():
        return None

    items = [
        _FakeItem("tests::plain", function=plain_fn),
        _FakeItem("tests::network1", function=network1_fn),
    ]
    config = _FakeConfig(
        platform="a2a3",
        level=4,
        **{"exclude-level": None, "runtime": None, "enable-chip-swimlane": 0},
    )

    root_conftest.pytest_collection_modifyitems(None, config, items)

    assert [item.nodeid for item in items] == ["tests::network1"]


def test_exclude_level4_keeps_unlevelled_functions():
    def plain_fn():
        return None

    @scene_level(SceneTestLevel.NETWORK1)
    def network1_fn():
        return None

    items = [
        _FakeItem("tests::plain", function=plain_fn),
        _FakeItem("tests::network1", function=network1_fn),
    ]
    config = _FakeConfig(
        platform="a2a3",
        **{"level": None, "exclude-level": 4, "runtime": None, "enable-chip-swimlane": 0},
    )

    root_conftest.pytest_collection_modifyitems(None, config, items)

    assert [item.nodeid for item in items] == ["tests::plain"]


def test_sorting_uses_function_level_metadata():
    @scene_level(SceneTestLevel.NODE)
    def host_fn():
        return None

    class L2:
        _st_level = SceneTestLevel.CHIP
        _st_runtime = "tensormap_and_ringbuffer"
        CASES = [{"platforms": ["a2a3"]}]

    items = [
        _FakeItem("tests::l2", cls=L2),
        _FakeItem("tests::host", function=host_fn),
    ]
    config = _FakeConfig(
        platform="a2a3",
        level=None,
        **{"exclude-level": None, "runtime": None, "enable-chip-swimlane": 0},
    )

    root_conftest.pytest_collection_modifyitems(None, config, items)

    assert [item.nodeid for item in items] == ["tests::host", "tests::l2"]


def test_resource_jobs_record_resource_last():
    class NormalL3:
        _st_level = SceneTestLevel.NODE
        _st_runtime = "host_build_graph"
        CASES = [{"platforms": ["a2a3"]}]

    class LastL3:
        _st_level = SceneTestLevel.NODE
        _st_runtime = "host_build_graph"
        CASES = [{"platforms": ["a2a3"]}]

    items = [
        _FakeItem("tests::last", cls=LastL3, markers=[_FakeMarker("resource_last")]),
        _FakeItem("tests::normal", cls=NormalL3),
    ]

    jobs = root_conftest._collect_resource_jobs(items, "a2a3")

    assert [(job.nodeid, job.run_last) for job in jobs] == [
        ("tests::last", True),
        ("tests::normal", False),
    ]


def test_resource_last_jobs_run_after_l2(monkeypatch, tmp_path):
    class L2:
        _st_level = SceneTestLevel.CHIP
        _st_runtime = "host_build_graph"

    config = _FakeConfig(
        device="0",
        exitfirst=False,
        platform="a2a3",
        manual="exclude",
    )
    config.invocation_params = SimpleNamespace(dir=tmp_path)
    session = SimpleNamespace(config=config, items=[_FakeItem("tests::l2", cls=L2)], testsfailed=0)
    specs = [
        root_conftest._ResourceJob("l3", "tests::normal", "Normal", "host_build_graph", 1, 0),
        root_conftest._ResourceJob("l3", "tests::last", "Last", "host_build_graph", 1, 1),
    ]
    events = []

    def fake_run_jobs(jobs, *_args, **_kwargs):
        events.append(("resource", tuple(job.nodeid for job in jobs)))
        return [JobResult(label=job.label, returncode=0, device_ids=[0], nodeid=job.nodeid) for job in jobs]

    def fake_subprocess_run(*_args, **_kwargs):
        events.append(("l2", "host_build_graph"))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(root_conftest, "_base_pytest_argv", lambda *_args, **_kwargs: ["pytest"])
    monkeypatch.setattr(root_conftest, "_resolve_max_parallel", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(root_conftest._ps, "run_jobs", fake_run_jobs)
    monkeypatch.setattr(root_conftest.subprocess, "run", fake_subprocess_run)

    assert root_conftest._dispatch_test_phases(session, specs)
    assert events == [
        ("resource", ("tests::normal",)),
        ("l2", "host_build_graph"),
        ("resource", ("tests::last",)),
    ]


def test_multi_round_chip_swimlane_does_not_reject_l3_items():
    @scene_level(SceneTestLevel.NODE)
    def host_fn():
        return None

    items = [_FakeItem("tests::host", function=host_fn)]
    config = _FakeConfig(
        platform="a2a3",
        level=None,
        **{
            "exclude-level": None,
            "runtime": None,
            "rounds": 5,
            "enable-chip-swimlane": 4,
        },
    )

    root_conftest.pytest_collection_modifyitems(None, config, items)

    assert [item.nodeid for item in items] == ["tests::host"]


def test_single_round_chip_swimlane_rejects_l3_items():
    @scene_level(SceneTestLevel.NODE)
    def host_fn():
        return None

    items = [_FakeItem("tests::host", function=host_fn)]
    config = _FakeConfig(
        platform="a2a3",
        level=None,
        **{
            "exclude-level": None,
            "runtime": None,
            "rounds": 1,
            "enable-chip-swimlane": 4,
        },
    )

    with pytest.raises(pytest.UsageError, match="not supported for L3 tests"):
        root_conftest.pytest_collection_modifyitems(None, config, items)


def test_level_filters_are_mutually_exclusive():
    config = _FakeConfig(level=4, **{"exclude-level": 4})

    with pytest.raises(pytest.UsageError, match="cannot be used together"):
        root_conftest._validate_level_filters(config)


def test_network1_logs_requires_network1_level(monkeypatch):
    @scene_level(SceneTestLevel.CHIP)
    def chip_fn():
        return None

    request = SimpleNamespace(node=_FakeItem("tests::chip", function=chip_fn))

    with pytest.raises(Failed, match="SceneTestLevel\\.NETWORK1"):
        root_conftest.st_network1_logs.__wrapped__(request, monkeypatch)
