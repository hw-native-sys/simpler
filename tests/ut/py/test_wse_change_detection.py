# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Replay the canonical change detector, with independent sibling architectures."""

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[3]


def _detect(tmp_path, files):
    workflow = yaml.safe_load((_ROOT / ".github/workflows/_detect-changes.yml").read_text())
    script = next(step["run"] for step in workflow["jobs"]["detect"]["steps"] if step.get("id") == "check")
    # Everything after attribution is executed verbatim; avoid fetching refs.
    script = script[script.index('if [ -z "$FILES" ]; then') :]
    output = tmp_path / "outputs"
    env = dict(os.environ, FILES=files, GITHUB_OUTPUT=str(output))
    result = subprocess.run(
        ["bash", "-e", "-c", script], env=env, check=True, capture_output=True, text=True, timeout=10
    )
    flags = dict(line.split("=", 1) for line in output.read_text().splitlines())
    return flags, result.stdout


@pytest.mark.parametrize(
    ("files", "arches", "categories"),
    [
        ("docs/a.md", (False, False, False), (False, False)),
        ("tests/fdwic/dsv41_scheduler/production/export_cases.py", (False, False, False), (False, False)),
        ("src/wse/docs/sram.md", (False, False, False), (False, False)),
        (".gitignore", (False, False, False), (False, False)),
        ("src/wse/platform/sim/host/device_runner.cpp", (False, False, True), (True, True)),
        ("tests/st/wse/host_build_graph/test_equivalence.py", (False, False, True), (True, False)),
        ("src/a5/platform/sim/host/device_runner.cpp", (False, True, False), (True, True)),
        ("tests/st/a5/host_build_graph/vector_example/test_vector_example.py", (False, True, False), (True, False)),
        ("src/a2a3/platform/sim/host/device_runner.cpp", (True, False, False), (True, True)),
        ("simpler_setup/kernel_compiler.py", (True, True, True), (True, True)),
        (".github/workflows/_st-sim-wse.yml", (True, True, True), (True, True)),
        ("unclassified/new_file.cpp", (True, True, True), (True, True)),
        ("", (True, True, True), (True, True)),
        ("tests/ut/cpp/wse/test_runtime.cpp", (False, False, True), (False, True)),
        ("src/a2a3/runtime.cpp\nsrc/a5/runtime.cpp", (True, True, False), (True, True)),
        ("src/a2a3/runtime.cpp\nsrc/wse/runtime.cpp", (True, False, True), (True, True)),
        ("src/a5/runtime.cpp\nsrc/wse/runtime.cpp", (False, True, True), (True, True)),
        ("tests/st/wse/test_scene.py\ntests/ut/cpp/a5/test_runtime.cpp", (False, True, True), (True, True)),
        ("src/wse/docs/sram.md\nsrc/a5/runtime.cpp", (False, True, False), (True, True)),
    ],
)
def test_architecture_flags_from_actual_detector(tmp_path, files, arches, categories):
    flags, stdout = _detect(tmp_path, files)
    assert tuple(flags[arch + "_changed"] == "true" for arch in ("a2a3", "a5", "wse")) == arches
    assert tuple(flags[category + "_affected"] == "true" for category in ("st", "ut")) == categories
    if files:
        for arch, affected in zip(("a2a3", "a5", "wse"), arches, strict=True):
            if affected:
                assert f"Files affecting {arch}:\n" in stdout
                assert f"skipping {arch}" not in stdout
            else:
                assert f"All changes are confined to other architectures or non-code; skipping {arch}" in stdout
                assert f"Files affecting {arch}:" not in stdout
    if not files:
        assert flags["st_affected"] == flags["ut_affected"] == "true"


def test_workflow_consumers_use_declared_outputs():
    text = (_ROOT / ".github/workflows/_detect-changes.yml").read_text()
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    outputs = set(workflow["on"]["workflow_call"]["outputs"])
    for path in (_ROOT / ".github/workflows").glob("*.yml"):
        consumer = path.read_text()
        assert set(re.findall(r"needs\.detect-changes\.outputs\.(\w+)", consumer)) <= outputs


@pytest.mark.parametrize(
    "files",
    [
        "tests/fdwic/dsv41_scheduler/production/record_tool.cpp",
        "tests/fdwic/dsv41_scheduler/tests/test_dsv41_half_sync.py",
        "tests/fdwic/dsv41_scheduler/production/requirements-bound.txt",
        "tests/fdwic/dsv41_scheduler/production/sync_bound.py\ndocs/benchmark.md",
    ],
)
def test_manual_only_keeps_precommit_but_no_automatic_test_jobs(tmp_path, files):
    flags, _ = _detect(tmp_path, files)
    assert flags["non_code_only"] == "false"
    assert flags["tests_only"] == "true"
    assert _selected_jobs(flags) == {"detect-changes", "pre-commit"}


def _selected_jobs(flags):
    workflow = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text())
    selected = set()
    for name, job in workflow["jobs"].items():
        expression = job.get("if", "True")
        expression = (
            re.sub(r"needs\.detect-changes\.outputs\.(\w+)", lambda match: repr(flags[match[1]]), expression)
            .replace("&&", " and ")
            .replace("||", " or ")
        )
        if eval(expression, {"__builtins__": {}}):
            selected.add(name)
    return selected


@pytest.mark.parametrize(
    ("other", "arches", "categories"),
    [
        ("src/a5/runtime/host_build_graph/runtime.h", (False, True, False), (True, True)),
        ("tests/st/a2a3/vector/test_vector.py", (True, False, False), (True, False)),
        ("tests/ut/py/test_buffer.py", (True, True, True), (False, True)),
        ("simpler_setup/kernel_compiler.py", (True, True, True), (True, True)),
        (".github/workflows/ci.yml", (True, True, True), (True, True)),
        ("unclassified/new.cpp", (True, True, True), (True, True)),
        ("tests/fdwic/dsv41_scheduler_other/new.py", (True, True, True), (True, True)),
    ],
)
def test_manual_exclusion_does_not_hide_other_changes(tmp_path, other, arches, categories):
    files = "tests/fdwic/dsv41_scheduler/production/export_cases.py\n" + other
    flags, _ = _detect(tmp_path, files)
    assert tuple(flags[arch + "_changed"] == "true" for arch in ("a2a3", "a5", "wse")) == arches
    assert tuple(flags[category + "_affected"] == "true" for category in ("st", "ut")) == categories
    if other == ".github/workflows/ci.yml":
        workflow = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text())
        assert _selected_jobs(flags) == set(workflow["jobs"])


def test_unknown_diff_still_selects_every_automatic_job(tmp_path):
    flags, _ = _detect(tmp_path, "")
    workflow = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text())
    assert _selected_jobs(flags) == set(workflow["jobs"])
