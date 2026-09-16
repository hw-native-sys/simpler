#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CLI contracts shared by pytest and standalone SceneTest execution."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from simpler_setup import parallel_scheduler
from simpler_setup.scene_test import (
    SceneTestCase,
    _dispatch_test_phases_standalone,
    effective_diagnostic_options,
    run_class_cases,
    standalone_pytest_options,
)


def _recording_converter(calls):
    """Stand in for the converter subprocess, reporting the success it models.

    The return value is the contract `_convert_rank_swimlanes` reads to decide
    whether a merge needs its per-Rank fallback, so a stub that dropped it
    would exercise the fallback in every test.
    """

    def run(**kwargs):
        calls.append(kwargs)
        return True

    return run


def test_l3_swimlane_postprocess_merges_dispatches_present_on_every_rank(tmp_path, monkeypatch) -> None:
    scene_test_module = importlib.import_module("simpler_setup.scene_test")
    for rank in (0, 1):
        for dispatch in ("d0", "d1"):
            records = tmp_path / f"rank{rank}" / dispatch / "chip_swimlane_records.json"
            records.parent.mkdir(parents=True)
            records.write_text("{}")

    calls = []
    monkeypatch.setattr(scene_test_module, "_run_swimlane_converter", _recording_converter(calls))

    scene_test_module._convert_case_swimlane("case", tmp_path)

    assert [call["dispatch"] for call in calls] == ["d0", "d1"]
    assert [call["output_path"] for call in calls] == [
        Path(tmp_path) / "l3_swimlane_d0.json",
        Path(tmp_path) / "l3_swimlane_d1.json",
    ]


def test_single_file_postprocess_passes_the_processes_actual_bound_host_log(tmp_path, monkeypatch) -> None:
    """A later case must not assume the process-wide log moved beside it."""
    scene_test_module = importlib.import_module("simpler_setup.scene_test")
    task_interface = importlib.import_module("_task_interface")
    bound = tmp_path / "first-case"
    current = tmp_path / "later-case"
    bound.mkdir()
    current.mkdir()
    host_log = bound / "host.42.log"
    host_log.write_text("")
    (current / "chip_swimlane_records.json").write_text("{}")
    monkeypatch.setattr(task_interface, "_host_log_directory", lambda: str(bound))
    calls = []
    monkeypatch.setattr(scene_test_module, "_run_swimlane_converter", _recording_converter(calls))

    scene_test_module._convert_case_swimlane("case", current)

    assert calls[0]["host_log_paths"] == [host_log]


def test_l3_swimlane_postprocess_merges_below_level_four(tmp_path, monkeypatch, caplog) -> None:
    """Placement comes from the Host log, so no capture level gates the merge.

    A level-3 capture has fewer device streams to draw and a looser join, both of
    which the converter reports; neither is a reason to fall back to one trace
    per Rank, which is what a reader of a collective cannot use.
    """
    scene_test_module = importlib.import_module("simpler_setup.scene_test")
    for rank in (0, 1):
        records = tmp_path / f"rank{rank}" / "d0" / "chip_swimlane_records.json"
        records.parent.mkdir(parents=True)
        records.write_text(json.dumps({"chip_swimlane_level": 3}))

    calls = []
    monkeypatch.setattr(scene_test_module, "_run_swimlane_converter", _recording_converter(calls))

    scene_test_module._convert_case_swimlane("case", tmp_path)

    assert [call["input_path"] for call in calls] == [tmp_path]
    assert calls[0]["dispatch"] == "d0"


def test_l3_swimlane_postprocess_refuses_asymmetric_local_capture_indexes(tmp_path, monkeypatch, caplog) -> None:
    scene_test_module = importlib.import_module("simpler_setup.scene_test")
    for rank, dispatches in ((0, ("d0", "d1")), (1, ("d0",))):
        for dispatch in dispatches:
            records = tmp_path / f"rank{rank}" / dispatch / "chip_swimlane_records.json"
            records.parent.mkdir(parents=True)
            records.write_text("{}")

    calls = []
    monkeypatch.setattr(scene_test_module, "_run_swimlane_converter", _recording_converter(calls))

    scene_test_module._convert_case_swimlane("case", tmp_path)

    assert calls == []
    assert "refusing to pair asymmetric local capture indexes" in caplog.text


def test_l3_swimlane_postprocess_uses_parent_identity_when_rank_d_paths_are_reordered(tmp_path, monkeypatch) -> None:
    scene_test_module = importlib.import_module("simpler_setup.scene_test")
    captures = {
        (0, "d0"): (5, 0),
        (0, "d1"): (6, 0),
        (1, "d0"): (6, 1),
        (1, "d1"): (5, 1),
    }
    for (rank, dispatch), (task_slot, group_index) in captures.items():
        capture = tmp_path / f"rank{rank}" / dispatch
        capture.mkdir(parents=True)
        (capture / "chip_swimlane_records.json").write_text("{}")
        (capture / "dispatch_identity.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": 17,
                    "task_slot": task_slot,
                    "group_index": group_index,
                    "group_size": 2,
                    "chip_rank": rank,
                    "local_capture_index": int(dispatch.removeprefix("d")),
                    "endpoint_dispatch_id": int(dispatch.removeprefix("d")) + 1,
                    "pipeline_slot": 0,
                    "pipeline_generation": 1,
                    "callable_digest": "ab" * 32,
                }
            )
        )

    calls = []
    monkeypatch.setattr(scene_test_module, "_run_swimlane_converter", _recording_converter(calls))

    scene_test_module._convert_case_swimlane("case", tmp_path)

    assert [(call["dispatch"], call["dispatch_id"]) for call in calls] == [(None, "17:5"), (None, "17:6")]
    assert [call["output_path"].name for call in calls] == [
        "l3_swimlane_run17_task5.json",
        "l3_swimlane_run17_task6.json",
    ]


def test_rank_local_dep_and_scope_postprocessors_follow_swimlane_output_prefix(tmp_path, monkeypatch) -> None:
    scene_test_module = importlib.import_module("simpler_setup.scene_test")
    captures = [tmp_path / f"rank{rank}" / "d0" for rank in (0, 1)]
    for capture in captures:
        (capture / "scope_stats").mkdir(parents=True)
        (capture / "deps.json").write_text("{}")
        (capture / "scope_stats" / "scope_stats.jsonl").write_text("")

    dep_calls = []
    scope_calls = []
    monkeypatch.setattr(
        scene_test_module, "_graph_case_dep_gen", lambda _label, path, **_kwargs: dep_calls.append(path)
    )
    monkeypatch.setattr(scene_test_module, "_plot_case_scope_stats", lambda _label, path: scope_calls.append(path))

    scene_test_module.finalize_diagnostic_outputs("case", tmp_path, dep_gen=True, scope_stats=True)

    assert dep_calls == captures
    assert scope_calls == captures


def test_multi_rounds_disable_every_diagnostic() -> None:
    options = effective_diagnostic_options(
        2,
        chip_swimlane=4,
        dump_args=3,
        pmu=5,
        dep_gen=True,
        scope_stats=True,
        swimlane_overhead=True,
    )

    assert options == (0, 0, 0, False, False, False)


def test_swimlane_overhead_requires_chip_swimlane() -> None:
    with pytest.raises(ValueError, match="requires --enable-chip-swimlane"):
        effective_diagnostic_options(
            1,
            chip_swimlane=0,
            dump_args=0,
            pmu=0,
            dep_gen=False,
            scope_stats=False,
            swimlane_overhead=True,
        )


def test_pytest_front_end_reports_a_usage_error_not_a_test_failure() -> None:
    conftest = importlib.import_module("conftest")
    options = {"--enable-swimlane-overhead": True, "--enable-chip-swimlane": 0}
    config = SimpleNamespace(getoption=lambda name, default=None: options.get(name, default))

    with pytest.raises(pytest.UsageError, match="requires --enable-chip-swimlane"):
        conftest._validate_diagnostic_flags(config)


def test_dep_gen_extension_hook_uses_the_shared_multi_round_gate() -> None:
    options = {"--rounds": 2, "--enable-dep-gen": True}
    request = SimpleNamespace(config=SimpleNamespace(getoption=lambda name, default=None: options.get(name, default)))

    assert not SceneTestCase._effective_enable_dep_gen(request)


def test_thin_pytest_wrapper_forwards_the_shared_cli_contract() -> None:
    options = {
        "--rounds": 7,
        "--skip-golden": True,
        "--enable-chip-swimlane": 3,
        "--dump-args": 2,
        "--enable-pmu": 4,
        "--enable-dep-gen": True,
        "--enable-scope-stats": True,
        "--enable-swimlane-overhead": True,
    }
    request = SimpleNamespace(config=SimpleNamespace(getoption=lambda name, default=None: options.get(name, default)))

    assert standalone_pytest_options(request) == {
        "rounds": 7,
        "skip_golden": True,
        "enable_chip_swimlane": 3,
        "dump_args": 2,
        "enable_pmu": 4,
        "enable_dep_gen": True,
        "enable_scope_stats": True,
        "enable_swimlane_overhead": True,
    }


@pytest.mark.parametrize(("rounds", "expected"), [(1, 3), (2, 0)])
def test_chip_swimlane_extension_hook_uses_the_shared_multi_round_gate(rounds, expected) -> None:
    options = {"--rounds": rounds, "--enable-chip-swimlane": 3}
    request = SimpleNamespace(config=SimpleNamespace(getoption=lambda name, default=None: options.get(name, default)))

    assert SceneTestCase._effective_enable_chip_swimlane(request) == expected


def test_swimlane_overhead_allocates_a_diagnostic_output_prefix(monkeypatch) -> None:
    scene_test_module = importlib.import_module("simpler_setup.scene_test")
    output_prefix = scene_test_module.Path("diagnostic-output")
    captured = {}

    class FakeScene:
        def _run_and_validate(self, *_args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(scene_test_module, "build_output_prefix", lambda _case_label: output_prefix)

    run_class_cases(
        object(),
        FakeScene(),
        [{"name": "overhead"}],
        callable_obj=object(),
        sub_handles={},
        rounds=1,
        skip_golden=False,
        enable_chip_swimlane=0,
        enable_dump_args=0,
        enable_pmu=0,
        enable_dep_gen=False,
        enable_scope_stats=False,
        enable_swimlane_overhead=True,
    )

    assert captured["output_prefix"] == str(output_prefix)


def test_run_class_cases_reports_the_failing_case_name() -> None:
    class FailingScene:
        def _run_and_validate(self, *_args, **_kwargs):
            raise ValueError("device run failed")

    case = {"name": "large_bf16", "params": {"batch": 64, "dtype": "bfloat16"}}

    # The scene's own exception type reaches the caller: a negative scene test
    # asserts on it, so a fixed-type wrapper would make such a test unable to pass.
    with pytest.raises(
        ValueError,
        match=r"SceneTest case failed: FailingScene::large_bf16: device run failed$",
    ) as failure:
        run_class_cases(
            object(),
            FailingScene(),
            [case],
            callable_obj=object(),
            sub_handles={},
            rounds=1,
            skip_golden=False,
            enable_chip_swimlane=0,
            enable_dump_args=0,
            enable_pmu=0,
            enable_dep_gen=False,
            enable_scope_stats=False,
        )

    # Annotated in place, so there is no wrapper to unwrap and the traceback still
    # points at the scene that raised.
    assert failure.value.__cause__ is None
    assert failure.value.args[0] == "SceneTest case failed: FailingScene::large_bf16: device run failed"


def test_run_class_cases_names_the_case_without_args() -> None:
    """An exception carrying no message still gets the case name, not `'…: '`."""

    class FailingScene:
        def _run_and_validate(self, *_args, **_kwargs):
            raise ValueError

    with pytest.raises(ValueError) as failure:
        run_class_cases(
            object(),
            FailingScene(),
            [{"name": "empty_args"}],
            callable_obj=object(),
            sub_handles={},
            rounds=1,
            skip_golden=False,
            enable_chip_swimlane=0,
            enable_dump_args=0,
            enable_pmu=0,
            enable_dep_gen=False,
            enable_scope_stats=False,
        )

    assert str(failure.value) == "SceneTest case failed: FailingScene::empty_args"


def test_run_class_cases_keeps_device_error_visible_to_poison_classifier() -> None:
    conftest = importlib.import_module("conftest")

    class FailingScene:
        def _run_and_validate(self, *_args, **_kwargs):
            raise RuntimeError("prepare_native_run failed with code 507018")

    with pytest.raises(RuntimeError) as failure:
        run_class_cases(
            object(),
            FailingScene(),
            [{"name": "device_error"}],
            callable_obj=object(),
            sub_handles={},
            rounds=1,
            skip_golden=False,
            enable_chip_swimlane=0,
            enable_dump_args=0,
            enable_pmu=0,
            enable_dep_gen=False,
            enable_scope_stats=False,
        )

    excinfo = SimpleNamespace(type=type(failure.value), value=failure.value)
    assert conftest._requires_l2_worker_retirement(excinfo)


def test_standalone_dispatch_forwards_every_diagnostic_flag(monkeypatch) -> None:
    scene_class = type("StandaloneDiagnosticScene", (), {"_st_level": 2, "_st_runtime": "tensormap_and_ringbuffer"})
    args = SimpleNamespace(
        platform="a2a3sim",
        manual="exclude",
        log_level="timing",
        sanitizer="none",
        rounds=1,
        skip_golden=False,
        enable_chip_swimlane=4,
        dump_args=3,
        enable_pmu=5,
        enable_dep_gen=True,
        enable_scope_stats=True,
        enable_swimlane_overhead=True,
        device_ids=[0, 1],
        max_parallel=1,
        exitfirst=False,
        case=None,
    )
    commands = []

    def fake_run_jobs(jobs, *_args, **_kwargs):
        commands.extend(job.build_cmd([0]) for job in jobs)
        return [SimpleNamespace(returncode=0) for _ in jobs]

    monkeypatch.setattr(parallel_scheduler, "run_jobs", fake_run_jobs)

    assert _dispatch_test_phases_standalone(__name__, {scene_class: [{}]}, args)
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--enable-chip-swimlane") + 1] == "4"
    assert command[command.index("--dump-args") + 1] == "3"
    assert command[command.index("--enable-pmu") + 1] == "5"
    assert "--enable-dep-gen" in command
    assert "--enable-scope-stats" in command
    assert "--enable-swimlane-overhead" in command
    assert command[0] == sys.executable
