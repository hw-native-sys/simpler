# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Focused UT for the dep-gen post-case hook in scene_test.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from simpler_setup.scene_test import _graph_case_dep_gen, _run_deps_viewer

_SCENE_TEST_MOD = sys.modules["simpler_setup.scene_test"]


def _write_minimal_deps_json(path: Path) -> Path:
    data = {"edges": [], "tensors": [], "tasks": []}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def _patch_run_deps_viewer():
    return patch.object(_SCENE_TEST_MOD, "_run_deps_viewer")


def test_graph_case_dep_gen_calls_tool_when_deps_json_exists(tmp_path):
    _write_minimal_deps_json(tmp_path / "deps.json")

    with _patch_run_deps_viewer() as mock_run:
        _graph_case_dep_gen("my_case", tmp_path, callable_spec=None)

    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args
    assert str(tmp_path / "deps.json") in str(
        call_kwargs[1].get("input_path", call_kwargs[0][0] if call_kwargs[0] else "")
    )


def test_graph_case_dep_gen_skips_when_deps_json_missing(tmp_path):
    with _patch_run_deps_viewer() as mock_run:
        _graph_case_dep_gen("my_case", tmp_path, callable_spec=None)

    mock_run.assert_not_called()


def test_graph_case_dep_gen_passes_func_names_from_callable_spec(tmp_path):
    _write_minimal_deps_json(tmp_path / "deps.json")
    callable_spec = {
        "incores": [{"name": "kernel_add", "func_id": 0}, {"name": "kernel_mul", "func_id": 1}],
        "orchestration": {"name": "MyOrch"},
    }

    with _patch_run_deps_viewer() as mock_run:
        _graph_case_dep_gen("my_case", tmp_path, callable_spec=callable_spec)

    func_names_path = mock_run.call_args[1].get("func_names_path")
    assert func_names_path is not None
    assert func_names_path.exists()
    mapping = json.loads(func_names_path.read_text())
    assert mapping["callable_id_to_name"]["0"] == "kernel_add"
    assert mapping["callable_id_to_name"]["1"] == "kernel_mul"


def test_graph_case_dep_gen_reuses_existing_name_map(tmp_path):
    _write_minimal_deps_json(tmp_path / "deps.json")
    name_map_path = tmp_path / "name_map_my_case.json"
    name_map_path.write_text(json.dumps({"callable_id_to_name": {"0": "existing_kernel"}}))

    callable_spec = {
        "incores": [{"name": "kernel_add", "func_id": 0}],
        "orchestration": {"name": "MyOrch"},
    }

    with _patch_run_deps_viewer() as mock_run:
        _graph_case_dep_gen("my_case", tmp_path, callable_spec=callable_spec)

    func_names_path = mock_run.call_args[1].get("func_names_path")
    assert func_names_path == name_map_path


def test_graph_case_dep_gen_no_func_names_without_callable_spec(tmp_path):
    _write_minimal_deps_json(tmp_path / "deps.json")

    with _patch_run_deps_viewer() as mock_run:
        _graph_case_dep_gen("my_case", tmp_path, callable_spec=None)

    func_names_path = mock_run.call_args[1].get("func_names_path")
    assert func_names_path is None


def test_run_deps_viewer_invokes_subprocess_correctly(tmp_path):
    deps_file = _write_minimal_deps_json(tmp_path / "deps.json")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="", stderr="")
        _run_deps_viewer(input_path=deps_file)

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "simpler_setup.tools.deps_viewer" in cmd
    assert str(deps_file) in cmd
    assert "--format" in cmd
    assert "text" in cmd


def test_run_deps_viewer_passes_func_names(tmp_path):
    deps_file = _write_minimal_deps_json(tmp_path / "deps.json")
    name_map = tmp_path / "name_map.json"
    name_map.write_text("{}")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="", stderr="")
        _run_deps_viewer(input_path=deps_file, func_names_path=name_map)

    cmd = mock_run.call_args[0][0]
    assert "--func-names" in cmd
    assert str(name_map) in cmd


def test_run_deps_viewer_warning_on_failure(tmp_path):
    deps_file = _write_minimal_deps_json(tmp_path / "deps.json")

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, cmd=[], stderr="boom")
        with patch("logging.getLogger") as _:
            _run_deps_viewer(input_path=deps_file)
