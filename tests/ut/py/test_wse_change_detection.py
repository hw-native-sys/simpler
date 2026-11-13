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


@pytest.mark.parametrize(
    ("files", "arches", "categories"),
    [
        ("docs/a.md", (False, False, False), (False, False)),
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
    assert tuple(flags[arch + "_changed"] == "true" for arch in ("a2a3", "a5", "wse")) == arches
    assert tuple(flags[category + "_affected"] == "true" for category in ("st", "ut")) == categories
    if files:
        for arch, affected in zip(("a2a3", "a5", "wse"), arches, strict=True):
            if affected:
                assert f"Files affecting {arch}:\n" in result.stdout
                assert f"skipping {arch}" not in result.stdout
            else:
                assert f"All changes are confined to other architectures or non-code; skipping {arch}" in result.stdout
                assert f"Files affecting {arch}:" not in result.stdout
    if not files:
        assert flags["st_affected"] == flags["ut_affected"] == "true"


def test_workflow_consumers_use_declared_outputs():
    text = (_ROOT / ".github/workflows/_detect-changes.yml").read_text()
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    outputs = set(workflow["on"]["workflow_call"]["outputs"])
    for path in (_ROOT / ".github/workflows").glob("*.yml"):
        consumer = path.read_text()
        assert set(re.findall(r"needs\.detect-changes\.outputs\.(\w+)", consumer)) <= outputs
