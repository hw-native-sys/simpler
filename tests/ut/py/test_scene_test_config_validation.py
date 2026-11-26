#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validation contracts for framework-owned scene-test configuration."""

import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from simpler_setup import scene_test
from simpler_setup.scene_test import _temporary_env


def _decorate(cases):
    class ConfigScene:
        CASES = cases

    return scene_test(level=2, runtime="host_build_graph")(ConfigScene)


def test_scene_test_accepts_all_config_keys_and_custom_case_keys() -> None:
    decorated = _decorate(
        [
            {
                "name": "valid",
                "platforms": ["a2a3sim"],
                "config": {
                    "aicpu_thread_num": 2,
                    "runtime_env": {
                        "ring_task_window": 64,
                        "ring_heap": 1024,
                        "ring_dep_pool": 32,
                    },
                    "device_count": 2,
                    "num_sub_workers": 1,
                },
                "required_sched_phases": ("release",),
            },
            {"name": "no_config", "platforms": ["a2a3sim"]},
        ]
    )

    assert decorated.CASES[0]["required_sched_phases"] == ("release",)


def test_scene_test_rejects_unknown_config_key() -> None:
    with pytest.raises(ValueError, match=r"ConfigScene\.CASES\[0\].*block_dim.*allowed keys"):
        _decorate([{"name": "bad", "config": {"block_dim": 24}}])


def test_scene_test_rejects_unknown_runtime_env_key() -> None:
    with pytest.raises(ValueError, match=r"ConfigScene\.CASES\[0\].*runtime_env.*ring_heep.*allowed keys"):
        _decorate([{"name": "bad", "config": {"runtime_env": {"ring_heep": 1024}}}])


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (None, "config must be a mapping"),
        ({"runtime_env": None}, "config.runtime_env must be a mapping"),
    ],
)
def test_scene_test_rejects_non_mapping_config_layers(config, match) -> None:
    with pytest.raises(TypeError, match=match):
        _decorate([{"name": "bad", "config": config}])


@pytest.mark.parametrize("first_override,second_override", [(True, True), (True, False), (False, True), (False, False)])
def test_overlapping_scene_environment_scopes(monkeypatch, first_override, second_override):
    key = "SIMPLER_TEST_SCENE_COLOR"
    monkeypatch.setenv(key, "baseline")

    def enter_second():
        with _temporary_env({key: "blue"} if second_override else {}):
            return os.environ[key]

    with ThreadPoolExecutor(1) as pool:
        with _temporary_env({key: "red"} if first_override else {}):
            future = pool.submit(enter_second)
            if first_override or second_override:
                with pytest.raises(RuntimeError, match="exclusive"):
                    future.result(timeout=5)
            else:
                assert future.result(timeout=5) == "baseline"
            assert os.environ[key] == ("red" if first_override else "baseline")
    with _temporary_env({key: "green"}):
        assert os.environ[key] == "green"
    assert os.environ[key] == "baseline"
