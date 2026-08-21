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
import sys
from types import SimpleNamespace

import pytest

from simpler_setup import parallel_scheduler
from simpler_setup.scene_test import (
    SceneTestCase,
    _dispatch_test_phases_standalone,
    _effective_diagnostic_options,
    run_class_cases,
)


def test_multi_rounds_disable_every_diagnostic() -> None:
    options = _effective_diagnostic_options(
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
        _effective_diagnostic_options(
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


def test_swimlane_overhead_allocates_a_diagnostic_output_prefix(monkeypatch) -> None:
    scene_test_module = importlib.import_module("simpler_setup.scene_test")
    output_prefix = scene_test_module.Path("diagnostic-output")
    captured = {}

    class FakeScene:
        def _run_and_validate(self, *_args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(scene_test_module, "_build_output_prefix", lambda _case_label: output_prefix)

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

    with pytest.raises(
        RuntimeError,
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

    assert isinstance(failure.value.__cause__, ValueError)
    assert str(failure.value.__cause__) == "device run failed"


def test_run_class_cases_keeps_device_error_visible_to_poison_classifier() -> None:
    conftest = importlib.import_module("conftest")

    class FailingScene:
        def _run_and_validate(self, *_args, **_kwargs):
            raise RuntimeError("run_prepared failed with code 507018")

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
    assert conftest._is_device_runtime_error(excinfo)


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
