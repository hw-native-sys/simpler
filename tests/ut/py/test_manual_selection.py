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
from types import SimpleNamespace

import pytest

from simpler_setup.scene_test import SceneTestCase, is_manual_for_platform, manual_marker_applies

_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location("_root_conftest_for_manual_selection_tests", _ROOT / "conftest.py")
assert _SPEC is not None and _SPEC.loader is not None
root_conftest = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(root_conftest)


class _FakeMarker:
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs


class _FakeItem:
    def __init__(self, nodeid, *, markers=()):
        self.nodeid = nodeid
        self.cls = None
        self.function = None
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


class _SelectionScene(SceneTestCase):
    CASES = [
        {"name": "ordinary", "platforms": ["a2a3sim", "a5sim"]},
        {"name": "sim_manual", "platforms": ["a2a3sim", "a5sim"], "manual": ["a2a3sim"]},
        {"name": "other_platform", "platforms": ["a5sim"]},
    ]


@pytest.mark.parametrize(
    ("manual", "platform", "expected"),
    [
        (True, "a2a3sim", True),
        (False, "a2a3sim", False),
        (None, "a2a3sim", False),
        ("a2a3sim", "a2a3sim", True),
        ("a5sim", "a2a3sim", False),
        (["a2a3sim", "a5sim"], "a5sim", True),
        (("a2a3sim", "a5sim"), "a2a3", False),
        ({"a2a3sim", "a5sim"}, "a2a3sim", True),
    ],
)
def test_is_manual_for_platform(manual, platform, expected):
    assert is_manual_for_platform(manual, platform) is expected


@pytest.mark.parametrize(
    ("is_manual", "manual_mode", "expected"),
    [
        (False, "exclude", True),
        (True, "exclude", False),
        (False, "include", True),
        (True, "include", True),
        (False, "only", False),
        (True, "only", True),
    ],
)
def test_manual_mode_matches(is_manual, manual_mode, expected):
    assert root_conftest._manual_mode_matches(is_manual, manual_mode) is expected


@pytest.mark.parametrize(
    ("marker", "platform", "expected"),
    [
        (None, "a2a3sim", False),
        (_FakeMarker("manual"), "a2a3sim", True),
        (_FakeMarker("manual", ["a2a3sim", "a5sim"]), "a5sim", True),
        (_FakeMarker("manual", ["a2a3sim", "a5sim"]), "a2a3", False),
        (_FakeMarker("manual", platforms=["a2a3sim", "a5sim"]), "a2a3sim", True),
        (_FakeMarker("manual", platforms=["a2a3sim", "a5sim"]), "a5", False),
    ],
)
def test_manual_marker_applies(marker, platform, expected):
    assert manual_marker_applies(marker, platform) is expected


def test_manual_marker_rejects_positional_and_keyword_platforms():
    marker = _FakeMarker("manual", ["a2a3sim"], platforms=["a5sim"])

    with pytest.raises(pytest.UsageError, match="either positionally or by keyword"):
        manual_marker_applies(marker, "a2a3sim")


def test_manual_marker_rejects_unknown_keyword_arguments():
    marker = _FakeMarker("manual", platforms=["a2a3sim"], reason="slow")

    with pytest.raises(pytest.UsageError, match=r"unsupported keyword argument\(s\): reason"):
        manual_marker_applies(marker, "a2a3sim")


@pytest.mark.parametrize(
    ("platform", "manual_mode", "expected"),
    [
        ("a2a3sim", "exclude", ["tests::ordinary"]),
        ("a2a3sim", "include", ["tests::global_manual", "tests::ordinary", "tests::sim_manual"]),
        ("a2a3sim", "only", ["tests::global_manual", "tests::sim_manual"]),
        ("a2a3", "exclude", ["tests::ordinary", "tests::sim_manual"]),
        ("a2a3", "include", ["tests::global_manual", "tests::ordinary", "tests::sim_manual"]),
        ("a2a3", "only", ["tests::global_manual"]),
    ],
)
def test_collection_filters_platform_scoped_manual_tests(platform, manual_mode, expected):
    items = [
        _FakeItem("tests::ordinary"),
        _FakeItem("tests::global_manual", markers=[_FakeMarker("manual")]),
        _FakeItem("tests::sim_manual", markers=[_FakeMarker("manual", platforms=["a2a3sim"])]),
    ]
    config = _FakeConfig(
        platform=platform,
        manual=manual_mode,
        runtime=None,
        level=None,
        **{"exclude-level": None, "enable-chip-swimlane": 0},
    )

    root_conftest.pytest_collection_modifyitems(None, config, items)

    assert [item.nodeid for item in items] == expected


@pytest.mark.parametrize(
    ("manual_mode", "selectors", "expected"),
    [
        ("exclude", [], ["ordinary"]),
        ("include", [], ["ordinary", "sim_manual"]),
        ("only", [], ["sim_manual"]),
        ("include", ["_SelectionScene::sim_manual"], ["sim_manual"]),
    ],
)
def test_scene_case_selection_matches_run_filters(manual_mode, selectors, expected):
    request = SimpleNamespace(node=_FakeItem("scene::test_run"), config=_FakeConfig(manual=manual_mode, case=selectors))

    matched = _SelectionScene()._matching_cases("a2a3sim", request)

    assert [case["name"] for case in matched] == expected


@pytest.mark.parametrize(("mode", "selected"), [("exclude", False), ("include", True), ("only", True)])
def test_manual_scene_method_and_parameter_combine_platforms(mode, selected):
    item = _FakeItem(
        "scene::test_method[param]",
        markers=[_FakeMarker("manual", ["a5sim"]), _FakeMarker("manual", ["a5"])],
    )

    class Scene(SceneTestCase):
        _st_level = 3
        _st_runtime = "host_build_graph"
        CASES = [{"name": "ordinary", "platforms": ["a5"], "config": {"device_count": 2}}]

    item.cls = Scene
    config = _FakeConfig(platform="a5", manual=mode)
    items = [item]
    root_conftest.pytest_collection_modifyitems(None, config, items)
    jobs = root_conftest._collect_resource_jobs(items, "a5", mode)
    cases = Scene()._matching_cases("a5", SimpleNamespace(node=item, config=config))
    assert items == ([item] if selected else [])
    assert [job.device_count for job in jobs] == ([2] if selected else [])
    assert cases == (Scene.CASES if selected else [])
