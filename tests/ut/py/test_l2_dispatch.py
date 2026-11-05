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

import pytest

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
    assert "--resource-child" in commands[0]
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


def test_l2_dispatch_replaces_parent_selection_in_child_command(monkeypatch):
    cf = _load_root_conftest()
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cf.subprocess, "run", fake_run)
    config = _Config(
        xdist_active=False,
        xdist_blocked=True,
        extra_args=("--runtime=host_build_graph", "--level", "2"),
    )
    items = [SimpleNamespace(cls=SimpleNamespace(_st_runtime="host_build_graph", _st_level=2))]
    session = SimpleNamespace(config=config, items=items, testsfailed=0, testscollected=0)

    assert cf._dispatch_test_phases(session, []) is True
    assert len(commands) == 1
    assert commands[0].count("--runtime") == 1
    assert "--runtime=host_build_graph" not in commands[0]
    assert commands[0].count("--level") == 1
    assert commands[0].count("--resource-child") == 1


def test_l2_poison_retry_marks_its_fresh_process_as_a_child(monkeypatch):
    cf = _load_root_conftest()
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cf.subprocess, "run", fake_run)

    assert cf._l2_poison_retry(["python", "-m", "pytest"], "host_build_graph", ["TestScene::"], _ROOT) == 0
    assert "--resource-child" in commands[0]
    assert commands[0][commands[0].index("--runtime") + 1] == "host_build_graph"


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
    # The child marker is separate from runtime selection, so a runtime-agnostic
    # job needs only its nodeid and the marker.
    cf = _load_root_conftest()
    spec = cf._ResourceJob(
        kind="standalone", nodeid="t.py::test_two_rank", label="test_two_rank", runtime=None, device_count=2
    )
    config = _Config(xdist_active=False, xdist_blocked=False)

    command = cf._resource_child_command(spec, [4, 5], "a2a3", "exclude", config)

    assert "--runtime" not in command
    assert "--resource-child" in command
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


class _Hook:
    """Records protocol calls and lets a test act as the item that ran."""

    def __init__(self, session, on_run=None):
        self._session = session
        self._on_run = on_run or (lambda item: None)
        self.ran = []

    def pytest_runtest_protocol(self, item, nextitem):
        self.ran.append((item.nodeid, nextitem.nodeid if nextitem else None))
        self._on_run(item)


class _Failed(Exception):
    pass


class _Interrupted(Exception):
    pass


def _leftover_session(items, on_run=None):
    config = _Config(xdist_active=False, xdist_blocked=False)
    session = SimpleNamespace(
        config=config,
        items=items,
        testsfailed=0,
        testscollected=0,
        shouldfail=False,
        shouldstop=False,
        Failed=_Failed,
        Interrupted=_Interrupted,
    )
    hook = _Hook(session, on_run)
    for item in items:
        item.config = SimpleNamespace(hook=hook)
    return session, hook


def test_the_leftover_phase_runs_each_unclaimed_item_once_in_order():
    cf = _load_root_conftest()
    items = [_Item("u.py::test_one"), _Item("u.py::test_two")]
    session, hook = _leftover_session(items)

    assert cf._dispatch_test_phases(session, []) is True

    assert hook.ran == [("u.py::test_one", "u.py::test_two"), ("u.py::test_two", None)]
    assert session.testsfailed == 0


def test_a_failing_leftover_item_makes_the_dispatcher_report_failure():
    cf = _load_root_conftest()
    items = [_Item("u.py::test_one")]
    session, hook = _leftover_session(items)
    hook._on_run = lambda item: setattr(session, "testsfailed", session.testsfailed + 1)

    assert cf._dispatch_test_phases(session, []) is True

    assert session.testsfailed == 1


def test_a_stopped_leftover_phase_interrupts_rather_than_reporting_success():
    # A stop leaves the remaining items unrun with nothing failed, so only the
    # exception pytest's own loop raises can keep the run from passing.
    cf = _load_root_conftest()
    items = [_Item("u.py::test_one"), _Item("u.py::test_two")]
    session, hook = _leftover_session(items)
    hook._on_run = lambda item: setattr(session, "shouldstop", "out of time")

    with pytest.raises(_Interrupted):
        cf._dispatch_test_phases(session, [])

    assert hook.ran == [("u.py::test_one", "u.py::test_two")]


def test_a_maxfail_stop_in_the_leftover_phase_raises_failed():
    cf = _load_root_conftest()
    items = [_Item("u.py::test_one"), _Item("u.py::test_two")]
    session, hook = _leftover_session(items)

    def _fail(item):
        session.testsfailed += 1
        session.shouldfail = "stopping after 1 failure"

    hook._on_run = _fail

    with pytest.raises(_Failed):
        cf._dispatch_test_phases(session, [])

    assert hook.ran == [("u.py::test_one", "u.py::test_two")]


def test_a_resource_child_is_told_it_is_one():
    # The parent already scheduled this job. A runtime-less job carries no
    # other option saying so, and the child must not schedule itself again.
    cf = _load_root_conftest()
    spec = cf._ResourceJob(
        kind="standalone", nodeid="t.py::test_two_rank", label="test_two_rank", runtime=None, device_count=2
    )
    config = _Config(xdist_active=False, xdist_blocked=False)

    command = cf._resource_child_command(spec, [4, 5], "a2a3", "exclude", config)

    assert "--resource-child" in command


def test_a_resource_child_does_not_dispatch_again(monkeypatch):
    # The item is a resource job, so without the flag this would schedule
    # itself again — the test is vacuous against an item that no phase claims.
    cf = _load_root_conftest()
    jobs = []
    monkeypatch.setattr(cf._ps, "run_jobs", lambda j, *a, **k: jobs.extend(j) or [])
    config = _Config(xdist_active=False, xdist_blocked=False)
    base = config.getoption
    config.getoption = lambda name, default=None: True if name == "--resource-child" else base(name, default)
    items = [_Item("t.py::test_two_rank", markers=[_Marker("device_count", 2)])]
    session = SimpleNamespace(
        config=config, items=items, testsfailed=0, testscollected=0, shouldfail=False, shouldstop=False
    )

    assert cf.pytest_runtestloop(session) is None
    assert jobs == []


@pytest.mark.parametrize("selection,value", [("--runtime", "host_build_graph"), ("--level", "3")])
def test_selection_does_not_skip_resource_dispatch(monkeypatch, selection, value):
    cf = _load_root_conftest()
    jobs = []
    monkeypatch.setattr(cf._ps, "run_jobs", lambda j, *a, **k: jobs.extend(j) or [])
    config = _Config(xdist_active=False, xdist_blocked=False)
    base = config.getoption
    config.getoption = lambda name, default=None: value if name == selection else base(name, default)
    items = [_Item("t.py::test_one_rank", markers=[_Marker("device_count", 1), _Marker("runtime", "host_build_graph")])]
    session = SimpleNamespace(
        config=config, items=items, testsfailed=0, testscollected=0, shouldfail=False, shouldstop=False
    )

    assert cf.pytest_runtestloop(session) is True
    assert [job.nodeid for job in jobs] == ["t.py::test_one_rank"]


def test_a_parent_narrowed_to_one_resource_item_still_schedules_its_job(monkeypatch):
    # The shape a parent and its child share: one selected item, no runtime.
    # Only the flag separates them, so without it this ran the test in-process
    # and the scheduler enforced neither device_count nor isolation.
    cf = _load_root_conftest()
    jobs = []
    monkeypatch.setattr(cf._ps, "run_jobs", lambda j, *a, **k: jobs.extend(j) or [])
    config = _Config(xdist_active=False, xdist_blocked=False)
    items = [_Item("t.py::test_two_rank", markers=[_Marker("device_count", 2)])]
    session = SimpleNamespace(
        config=config, items=items, testsfailed=0, testscollected=0, shouldfail=False, shouldstop=False
    )

    assert cf.pytest_runtestloop(session) is True
    assert [(job.nodeid, job.device_count) for job in jobs] == [("t.py::test_two_rank", 2)]
