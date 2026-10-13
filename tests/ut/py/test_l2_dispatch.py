# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for L2 dispatcher child-command construction."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[3]


def _load_root_conftest():
    spec = importlib.util.spec_from_file_location("_root_conftest_l2_dispatch", _ROOT / "conftest.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Config:
    def __init__(
        self,
        *,
        xdist_active: bool,
        xdist_blocked: bool,
        disable_arg: bool = False,
        xdist_numprocesses=None,
        xdist_dist="no",
        usepdb=False,
        extra_args=(),
    ):
        args = ["first.py", "second.py"]
        if disable_arg:
            args.extend(["-p", "no:xdist"])
        args.extend(extra_args)
        args.extend(["--max-parallel", "2"])
        self.invocation_params = SimpleNamespace(
            args=tuple(args),
            dir=_ROOT,
        )
        self.pluginmanager = SimpleNamespace(
            hasplugin=lambda name: name == "xdist" and xdist_active,
            is_blocked=lambda name: name == "xdist" and xdist_blocked,
        )
        self._xdist_numprocesses = xdist_numprocesses
        self._xdist_dist = xdist_dist
        self._usepdb = usepdb

    def getoption(self, name, default=None):
        return {
            "--device": "0-1",
            "--exitfirst": False,
            "--platform": "a2a3sim",
            "--manual": "include",
            "--max-parallel": "2",
            "numprocesses": self._xdist_numprocesses,
            "dist": self._xdist_dist,
            "usepdb": self._usepdb,
        }.get(name, default)


def test_l2_dispatch_does_not_add_xdist_options_when_plugin_is_disabled(monkeypatch):
    cf = _load_root_conftest()
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cf.subprocess, "run", fake_run)
    config = _Config(xdist_active=False, xdist_blocked=True, disable_arg=True)
    items = [
        SimpleNamespace(cls=SimpleNamespace(_st_runtime="host_build_graph", _st_level=2)),
        SimpleNamespace(cls=SimpleNamespace(_st_runtime="tensormap_and_ringbuffer", _st_level=2)),
    ]
    session = SimpleNamespace(config=config, items=items, testsfailed=0, testscollected=0)

    assert cf._dispatch_test_phases(session, []) is True
    assert len(commands) == 2
    for command in commands:
        assert "no:xdist" in command
        assert "-n" not in command
        assert "--dist" not in command


def test_l2_dispatch_adds_xdist_options_when_plugin_is_active(monkeypatch):
    cf = _load_root_conftest()
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cf.subprocess, "run", fake_run)
    config = _Config(xdist_active=True, xdist_blocked=False)
    items = [SimpleNamespace(cls=SimpleNamespace(_st_runtime="host_build_graph", _st_level=2))]
    session = SimpleNamespace(config=config, items=items, testsfailed=0, testscollected=0)

    assert cf._dispatch_test_phases(session, []) is True
    assert len(commands) == 1
    assert commands[0][-8:] == [
        "--runtime",
        "host_build_graph",
        "--level",
        "2",
        "-n",
        "2",
        "--dist",
        "loadfile",
    ]


def test_l2_dispatch_warns_and_stays_serial_when_xdist_is_not_loaded(monkeypatch, capsys):
    cf = _load_root_conftest()
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cf.subprocess, "run", fake_run)
    config = _Config(xdist_active=False, xdist_blocked=False)
    items = [SimpleNamespace(cls=SimpleNamespace(_st_runtime="host_build_graph", _st_level=2))]
    session = SimpleNamespace(config=config, items=items, testsfailed=0, testscollected=0)

    assert cf._dispatch_test_phases(session, []) is True
    assert "pytest-xdist plugin is not active" in capsys.readouterr().out
    assert "-n" not in commands[0]
    assert "--dist" not in commands[0]


def test_l2_dispatch_honors_xdist_disabled_with_numprocesses_zero(monkeypatch):
    cf = _load_root_conftest()
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cf.subprocess, "run", fake_run)
    config = _Config(
        xdist_active=True,
        xdist_blocked=False,
        xdist_numprocesses=0,
        extra_args=("-n", "0"),
    )
    items = [SimpleNamespace(cls=SimpleNamespace(_st_runtime="host_build_graph", _st_level=2))]
    session = SimpleNamespace(config=config, items=items, testsfailed=0, testscollected=0)

    assert cf._dispatch_test_phases(session, []) is True
    assert commands[0].count("-n") == 1
    assert commands[0][commands[0].index("-n") + 1] == "0"
    assert "--dist" not in commands[0]


def test_l2_dispatch_preserves_explicit_xdist_worker_count_and_distribution(monkeypatch):
    # A positive `-n` reaches the dispatcher only from a direct caller such as
    # this test: under pytest it puts xdist in distribution mode, whose
    # pytest_runtestloop claims the session first.
    cf = _load_root_conftest()
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cf.subprocess, "run", fake_run)
    config = _Config(
        xdist_active=True,
        xdist_blocked=False,
        xdist_numprocesses=1,
        xdist_dist="loadscope",
        extra_args=("-n", "1", "--dist", "loadscope"),
    )
    items = [SimpleNamespace(cls=SimpleNamespace(_st_runtime="host_build_graph", _st_level=2))]
    session = SimpleNamespace(config=config, items=items, testsfailed=0, testscollected=0)

    assert cf._dispatch_test_phases(session, []) is True
    assert commands[0].count("-n") == 1
    assert commands[0].count("--dist") == 1
    assert commands[0][commands[0].index("-n") + 1] == "1"
    assert commands[0][commands[0].index("--dist") + 1] == "loadscope"


def test_l2_dispatch_adds_distribution_default_when_only_worker_count_is_explicit(monkeypatch):
    cf = _load_root_conftest()
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cf.subprocess, "run", fake_run)
    config = _Config(
        xdist_active=True,
        xdist_blocked=False,
        xdist_numprocesses=1,
        xdist_dist="load",
        extra_args=("-n", "1"),
    )
    items = [SimpleNamespace(cls=SimpleNamespace(_st_runtime="host_build_graph", _st_level=2))]
    session = SimpleNamespace(config=config, items=items, testsfailed=0, testscollected=0)

    assert cf._dispatch_test_phases(session, []) is True
    assert commands[0].count("-n") == 1
    assert commands[0][-2:] == ["--dist", "loadfile"]


def test_l2_dispatch_treats_the_dist_shortcut_as_an_explicit_distribution(monkeypatch):
    cf = _load_root_conftest()
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cf.subprocess, "run", fake_run)
    config = _Config(xdist_active=True, xdist_blocked=False, xdist_dist="load", extra_args=("-d",))
    items = [SimpleNamespace(cls=SimpleNamespace(_st_runtime="host_build_graph", _st_level=2))]
    session = SimpleNamespace(config=config, items=items, testsfailed=0, testscollected=0)

    assert cf._dispatch_test_phases(session, []) is True
    assert commands[0][-2:] == ["-n", "2"]
    assert "--dist" not in commands[0]


def test_l2_dispatch_stays_serial_under_pdb_without_an_explicit_worker_count(monkeypatch):
    # xdist zeroes `numprocesses` only for `-n auto` / `-n logical`, so a bare
    # `--pdb` arrives with it unset. The child inherits `--pdb`, and xdist
    # rejects that combination once `-n` puts the child in distribution mode.
    cf = _load_root_conftest()
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cf.subprocess, "run", fake_run)
    config = _Config(xdist_active=True, xdist_blocked=False, usepdb=True, extra_args=("--pdb",))
    items = [SimpleNamespace(cls=SimpleNamespace(_st_runtime="host_build_graph", _st_level=2))]
    session = SimpleNamespace(config=config, items=items, testsfailed=0, testscollected=0)

    assert cf._dispatch_test_phases(session, []) is True
    assert "-n" not in commands[0]
    assert "--dist" not in commands[0]


class _Marker:
    def __init__(self, name, *args):
        self.name = name
        self.args = args


class _Item:
    """A collected item carrying only what the phase collectors read."""

    def __init__(self, nodeid, *, markers=(), cls=None):
        self.nodeid = nodeid
        self.name = nodeid.rsplit("::", 1)[-1]
        self.cls = cls
        self.function = None
        self._markers = {m.name: m for m in markers}

    def iter_markers(self):
        return iter(self._markers.values())

    def get_closest_marker(self, name):
        return self._markers.get(name)


def test_a_standalone_resource_function_needs_no_runtime_marker():
    cf = _load_root_conftest()
    items = [_Item("t.py::test_two_rank", markers=[_Marker("device_count", 2)])]

    jobs = cf._collect_resource_jobs(items, "a2a3")

    assert [(j.kind, j.nodeid, j.runtime, j.device_count) for j in jobs] == [
        ("standalone", "t.py::test_two_rank", None, 2)
    ]


def test_a_runtime_less_resource_child_is_narrowed_by_nodeid_alone():
    # --runtime is both the child's filter and the signal that puts it in child
    # mode; a runtime-agnostic job has no value to pass, so it must not appear.
    cf = _load_root_conftest()
    spec = cf._ResourceJob(
        kind="standalone", nodeid="t.py::test_two_rank", label="test_two_rank", runtime=None, device_count=2
    )
    config = _Config(xdist_active=False, xdist_blocked=False)

    command = cf._resource_child_command(spec, [4, 5], "a2a3", "exclude", config)

    assert "--runtime" not in command
    assert "t.py::test_two_rank" in command


def test_an_item_no_phase_runs_is_reported_as_unclaimed():
    # The Resource phase selects by marker and L2 by SceneTestCase runtime, so
    # a plain function belongs to neither and would otherwise be dropped.
    cf = _load_root_conftest()
    resource = _Item("t.py::test_two_rank", markers=[_Marker("device_count", 2)])
    l2 = _Item(
        "s.py::TestScene::test_run",
        cls=SimpleNamespace(_st_runtime="host_build_graph", _st_level=2, CASES=[]),
    )
    plain = _Item("u.py::test_plain")
    jobs = cf._collect_resource_jobs([resource], "a2a3")

    unclaimed = cf._collect_unclaimed_items([resource, l2, plain], jobs, ["host_build_graph"])

    assert [item.nodeid for item in unclaimed] == ["u.py::test_plain"]
