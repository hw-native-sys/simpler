# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for the SceneTest torch backend autoload timing record."""

from __future__ import annotations

import importlib
import sys

import pytest

from simpler_setup import SceneTestCase, TaskArgsBuilder
from simpler_setup.log_config import TIMING

scene_test = importlib.import_module("simpler_setup.scene_test")

_MESSAGE_PREFIX = "torch_backend_autoload "


class _MinimalCase(SceneTestCase):
    CALLABLE = {"orchestration": lambda *_args: None}

    def generate_args(self, params):
        return TaskArgsBuilder()


@pytest.fixture(autouse=True)
def _reset_record():
    scene_test._log_torch_backend_autoload_once.cache_clear()


def _set_modules(monkeypatch, *, torch_loaded: bool, torch_npu_loaded: bool) -> None:
    if torch_loaded:
        monkeypatch.setitem(sys.modules, "torch", object())
    else:
        monkeypatch.delitem(sys.modules, "torch", raising=False)
    if torch_npu_loaded:
        monkeypatch.setitem(sys.modules, "torch_npu", object())
    else:
        monkeypatch.delitem(sys.modules, "torch_npu", raising=False)


def _messages(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.getMessage().startswith(_MESSAGE_PREFIX)]


@pytest.mark.parametrize(
    "env_value, setting, raw, effective",
    [
        (None, "unset", "null", "enabled"),
        ("0", "0", '"0"', "disabled"),
        ("1", "1", '"1"', "enabled"),
        ("unexpected", "invalid", '"unexpected"', "disabled"),
    ],
)
def test_record_reports_effective_setting(monkeypatch, caplog, env_value, setting, raw, effective):
    if env_value is None:
        monkeypatch.delenv("TORCH_DEVICE_BACKEND_AUTOLOAD", raising=False)
    else:
        monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", env_value)
    _set_modules(monkeypatch, torch_loaded=True, torch_npu_loaded=False)
    caplog.set_level(TIMING, logger="simpler")

    scene_test._log_torch_backend_autoload_once()

    assert _messages(caplog) == [
        f"torch_backend_autoload setting={setting} raw={raw} raw_truncated=false "
        f"effective={effective} torch_imported=true torch_npu_loaded=false"
    ]


@pytest.mark.parametrize(
    "env_value, encoded",
    [
        ("a b", '"a b"'),
        ('a"b', '"a\\"b"'),
        ("a\\b", '"a\\\\b"'),
        ("a\nb", '"a\\nb"'),
    ],
)
def test_record_escapes_invalid_raw_setting(monkeypatch, caplog, env_value, encoded):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", env_value)
    _set_modules(monkeypatch, torch_loaded=True, torch_npu_loaded=False)
    caplog.set_level(TIMING, logger="simpler")

    scene_test._log_torch_backend_autoload_once()

    assert _messages(caplog) == [
        f"torch_backend_autoload setting=invalid raw={encoded} raw_truncated=false "
        "effective=disabled torch_imported=true torch_npu_loaded=false"
    ]


def test_record_truncates_invalid_raw_setting(monkeypatch, caplog):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "x" * 65)
    _set_modules(monkeypatch, torch_loaded=True, torch_npu_loaded=False)
    caplog.set_level(TIMING, logger="simpler")

    scene_test._log_torch_backend_autoload_once()

    assert _messages(caplog) == [
        f'torch_backend_autoload setting=invalid raw="{"x" * 64}" raw_truncated=true '
        "effective=disabled torch_imported=true torch_npu_loaded=false"
    ]


def test_record_reports_observed_modules(monkeypatch, caplog):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
    _set_modules(monkeypatch, torch_loaded=True, torch_npu_loaded=True)
    caplog.set_level(TIMING, logger="simpler")

    scene_test._log_torch_backend_autoload_once()

    assert _messages(caplog) == [
        'torch_backend_autoload setting=1 raw="1" raw_truncated=false '
        "effective=enabled torch_imported=true torch_npu_loaded=true"
    ]


def test_record_does_not_import_torch(monkeypatch, caplog):
    monkeypatch.delenv("TORCH_DEVICE_BACKEND_AUTOLOAD", raising=False)
    _set_modules(monkeypatch, torch_loaded=False, torch_npu_loaded=False)
    caplog.set_level(TIMING, logger="simpler")

    scene_test._log_torch_backend_autoload_once()

    assert _messages(caplog) == [
        "torch_backend_autoload setting=unset raw=null raw_truncated=false "
        "effective=enabled torch_imported=false torch_npu_loaded=false"
    ]


def test_record_is_emitted_once_per_interpreter(monkeypatch, caplog):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    _set_modules(monkeypatch, torch_loaded=True, torch_npu_loaded=False)
    caplog.set_level(TIMING, logger="simpler")

    scene_test._log_torch_backend_autoload_once()
    scene_test._log_torch_backend_autoload_once()

    assert len(_messages(caplog)) == 1


def test_l2_records_after_argument_build_before_run(monkeypatch):
    events = []

    class _Worker:
        def register(self, _callable):
            return object()

        def run(self, _handle, _args, *, config):
            events.append("run")

    monkeypatch.setattr(_MinimalCase, "CALLABLE", {"orchestration": {"signature": []}})
    monkeypatch.delattr(_MinimalCase, "_st_l2_handle", raising=False)
    monkeypatch.setattr(
        scene_test,
        "_build_l2_ref_args",
        lambda *_args, **_kwargs: (events.append("args") or object(), []),
    )
    monkeypatch.setattr(
        _MinimalCase,
        "compute_golden",
        lambda _self, _args, _params: events.append("golden"),
    )
    monkeypatch.setattr(scene_test, "_log_torch_backend_autoload_once", lambda: events.append("record"))
    monkeypatch.setattr(_MinimalCase, "_build_config", lambda *_args, **_kwargs: object())

    _MinimalCase()._run_and_validate_l2(
        _Worker(),
        object(),
        {"name": "case"},
    )

    assert events == ["args", "golden", "record", "run"]


def test_l3_records_after_rehost_before_run(monkeypatch):
    events = []

    class _Worker:
        # `config` is what carries `output_prefix`, and at L3 that is what binds
        # this process's own `[STRACE]` log directory, so the double takes it
        # for the same reason `Worker.run` does.
        def run(self, _task, *, config):
            events.append("run")

    class _Rehosted:
        def __init__(self, _worker, _args):
            events.append("rehost")

        def release(self):
            events.append("release")

    monkeypatch.setattr(scene_test, "_RehostedTaskArgs", _Rehosted)
    monkeypatch.setattr(
        _MinimalCase,
        "compute_golden",
        lambda _self, _args, _params: events.append("golden"),
    )
    monkeypatch.setattr(scene_test, "_log_torch_backend_autoload_once", lambda: events.append("record"))
    monkeypatch.setattr(_MinimalCase, "_build_config", lambda *_args, **_kwargs: object())

    _MinimalCase()._run_and_validate_l3(
        _Worker(),
        {},
        {},
        {"name": "case"},
    )

    assert events == ["golden", "rehost", "record", "run", "release"]
