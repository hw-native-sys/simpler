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
_WSE_PY_UT = "tests/ut/py/wse/test_wse_tile_operations.py"


def _detect(tmp_path, files=None, *, base_sha=None):
    workflow = yaml.safe_load((_ROOT / ".github/workflows/_detect-changes.yml").read_text())
    script = next(step["run"] for step in workflow["jobs"]["detect"]["steps"] if step.get("id") == "check")
    output = tmp_path / "outputs"
    env = dict(os.environ, GITHUB_OUTPUT=str(output))
    if files is not None:
        # The routing matrix starts with the attributed paths.
        script = script[script.index('if [ -z "$FILES" ]; then') :]
        env["FILES"] = files
    else:
        # Exercise the full detector against the test's local Git history.
        assert base_sha
        env.update(BASE_SHA=base_sha, HEAD_SHA="HEAD", BASE_REF="main", BASE_REPOSITORY="")
    result = subprocess.run(
        ["bash", "-e", "-c", script], cwd=tmp_path, env=env, check=True, capture_output=True, text=True, timeout=10
    )
    flags = dict(line.split("=", 1) for line in output.read_text().splitlines())
    return flags, result.stdout


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
        ("tests/support/a5/host_build_graph/spmd/kernels/record_slot_0.cpp", (False, True, False), (True, True)),
        ("src/a2a3/platform/sim/host/device_runner.cpp", (True, False, False), (True, True)),
        ("simpler_setup/kernel_compiler.py", (True, True, True), (True, True)),
        ("tools/foo.py", (True, True, True), (False, True)),
        (".github/workflows/_st-sim-wse.yml", (True, True, True), (True, True)),
        ("unclassified/new_file.cpp", (True, True, True), (True, True)),
        ("", (True, True, True), (True, True)),
        ("tests/ut/cpp/wse/test_runtime.cpp", (False, False, True), (False, True)),
        (_WSE_PY_UT, (False, False, True), (False, True)),
        (f"tests/st/wse/test_scene.py\n{_WSE_PY_UT}", (False, False, True), (True, True)),
        (f"tests/ut/cpp/a5/test_runtime.cpp\n{_WSE_PY_UT}", (False, True, True), (False, True)),
        (f"src/wse/platform/sim/host/device_runner.cpp\n{_WSE_PY_UT}", (False, False, True), (True, True)),
        (f"tests/ut/py/test_buffer.py\n{_WSE_PY_UT}", (True, True, True), (False, True)),
        (f".github/workflows/_detect-changes.yml\n{_WSE_PY_UT}", (True, True, True), (True, True)),
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
    selected = _selected_jobs(flags)
    _, a5, wse = arches
    st, ut = categories
    assert ("ut" in selected) == ut
    assert ("st-onboard-a5" in selected) == (a5 and st)
    assert ("st-framework-a5" in selected) == (a5 and st)
    assert ("st-sim-a5" in selected) == (a5 and st)
    assert ("st-sim-wse" in selected) == (wse and st)
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
    ("files", "cpu_arches", "hardware", "sdk"),
    [
        ("docs/ci.md", "", "", ""),
        ("tests/st/a5/host_build_graph/test_scene.py", "", "", ""),
        ("tests/st/a5/test_scene.py\n" + _WSE_PY_UT, "wse", "", ""),
        ("tests/st/wse/test_scene.py\ntests/ut/py/a2a3/test_runtime.py", "a2a3", "", ""),
        ("tests/ut/py/wse/kernel_fixture.cpp\ntests/ut/cpp/a5/runtime/test_calendar.cpp", "a5,wse", "", ""),
        ("src/a2a3/runtime/runtime.cpp", "a2a3", "a2a3", "a2a3"),
        ("src/a5/runtime/runtime.cpp\n" + _WSE_PY_UT, "a5,wse", "a5", "a5"),
        ("tests/ut/py/a5/hardware/test_runtime.py", "a5", "a5", ""),
        ("tests/ut/cpp/a2a3/hardware/test_comm_lifecycle.cpp", "a2a3", "a2a3", ""),
        ("tests/ut/cpp/a5/hardware/test_device.cpp", "a5", "a5", ""),
        ("tests/ut/cpp/a5/toolkit/test_a5_domain_release.cpp", "a5", "", "a5"),
        ("tests/ut/cpp/a2a3/toolkit/test_builder.cpp", "a2a3", "", "a2a3"),
        ("tests/ut/py/a2a3/toolkit/test_builder.py\ntests/ut/py/a5/test_input_cache.py", "a2a3,a5", "", "a2a3"),
        ("tests/st/a2a3/test_scene.py\ntests/ut/py/a5/toolkit/test_builder.py", "a5", "", "a5"),
        ("tests/ut/py/a5/toolkit/test_builder.py\ntests/ut/py/a2a3/hardware/test_device.py", "a2a3,a5", "a2a3", "a5"),
        ("tests/ut/py/test_worker/test_memory.py", "a2a3,a5,wse", "", ""),
        ("tests/ut/cpp/common/worker/test_memory.cpp", "a2a3,a5,wse", "", ""),
        ("tests/ut/py/test_worker/hardware/test_memory.py", "a2a3,a5,wse", "a2a3,a5", ""),
        ("tests/ut/py/toolkit/test_runtime_builder_integration.py", "a2a3,a5,wse", "", "a2a3,a5"),
        ("tests/ut/py/a5/conftest.py", "a5", "a5", "a5"),
        ("tests/ut/py/a5/kernel_fixture.py", "a5", "a5", "a5"),
        ("tests/ut/py/a2a3/support/test_helper.py", "a2a3", "a2a3", "a2a3"),
        ("tests/ut/py/a5/nohardware/kernel_fixture.cpp", "a5", "", ""),
        ("tests/ut/cpp/a2a3/nohardware/shared_state.h", "a2a3", "", ""),
        ("tests/ut/py/nohardware/fixture.py", "a2a3,a5,wse", "", ""),
        ("tests/ut/cpp/a2a3/runtime/scheduler_fixture.h", "a2a3", "a2a3", "a2a3"),
        ("tests/ut/py/a2a3/toolkit/a2a3_domain_lifecycle_host.cpp", "a2a3", "", "a2a3"),
        ("tests/ut/py/a5/hardware/worker_fixture.py", "a5", "a5", ""),
        ("tests/ut/py/support/fixture.py", "a2a3,a5,wse", "a2a3,a5", "a2a3,a5"),
        ("src/common/worker.cpp", "a2a3,a5,wse", "a2a3,a5", "a2a3,a5"),
        ("unclassified/new.cpp", "a2a3,a5,wse", "a2a3,a5", "a2a3,a5"),
        (".github/workflows/_detect-changes.yml", "a2a3,a5,wse", "a2a3,a5", "a2a3,a5"),
        ("", "a2a3,a5,wse", "a2a3,a5", "a2a3,a5"),
    ],
)
def test_unit_test_jobs_follow_architecture_and_execution_environment(tmp_path, files, cpu_arches, hardware, sdk):
    flags, _ = _detect(tmp_path, files)
    jobs = _selected_jobs(flags)
    assert flags["ut_arches"] == cpu_arches
    assert ("ut" in jobs) == bool(cpu_arches)
    for arch in ("a2a3", "a5"):
        runs_hardware = arch in hardware.split(",")
        runs_sdk = arch in sdk.split(",")
        assert (flags[f"ut_hardware_{arch}"] == "true") == runs_hardware
        assert (flags[f"ut_sdk_{arch}"] == "true") == runs_sdk
        assert (f"ut-{arch}" in jobs) == (runs_hardware or runs_sdk)


def test_reusable_unit_test_jobs_receive_detector_inputs():
    jobs = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    assert jobs["ut"]["with"]["ut_arches"] == "${{ needs.detect-changes.outputs.ut_arches }}"
    cpu_workflow = yaml.load((_ROOT / jobs["ut"]["uses"]).read_text(), Loader=yaml.BaseLoader)
    assert cpu_workflow["on"]["workflow_call"]["inputs"]["ut_arches"]["default"] == "a2a3,a5,wse"
    for arch in ("a2a3", "a5"):
        job = jobs[f"ut-{arch}"]
        workflow = yaml.load((_ROOT / job["uses"]).read_text(), Loader=yaml.BaseLoader)
        declared = workflow["on"]["workflow_call"]["inputs"]
        for capability in ("hardware", "sdk"):
            flag = f"needs.detect-changes.outputs.ut_{capability}_{arch}"
            name = f"run_{capability}"
            assert job["with"][name] == f"${{{{ {flag} == 'true' }}}}"
            assert declared[name]["type"] == "boolean"
            assert declared[name]["default"] == "true"


@pytest.mark.parametrize("arch", ["a2a3", "a5"])
def test_selected_suites_run_after_sibling_failure_when_setup_succeeded(arch):
    workflow = yaml.safe_load((_ROOT / f".github/workflows/_ut-npu-{arch}.yml").read_text())
    steps = {step["id"]: step for step in workflow["jobs"]["run"]["steps"] if "id" in step}
    assert steps["setup"]["uses"] == "./.github/actions/setup-venv"
    # An explicit status function replaces Actions' implicit success() after an earlier step fails.
    ready = {"!cancelled()", "steps.setup.outcome == 'success'"}
    sdk = ready | {"inputs.run_sdk"}
    hardware = ready | {"inputs.run_hardware"}
    conditions = {
        "sdk_py": sdk,
        "sdk_cpp": sdk | {f"hashFiles('tests/ut/cpp/{arch}/toolkit/**/test_*.cpp') != ''"},
        "hardware_cpp": hardware | {f"hashFiles('tests/ut/cpp/{arch}/hardware/**/test_*.cpp') != ''"},
        "hardware": hardware,
        "smoke": hardware | {"inputs.include_cann_examples"},
    }
    for step_id, required in conditions.items():
        step = steps[step_id]
        expression = step["if"].removeprefix("${{").removesuffix("}}")
        assert {term.strip() for term in expression.split("&&")} == required, step_id
        assert not step.get("continue-on-error"), step_id

    assert f"requires_toolkit --ut-arches {arch}" in steps["sdk_py"]["run"]
    for step_id in ("sdk_py", "sdk_cpp", "hardware_cpp"):
        assert "task-submit" not in steps[step_id]["run"], step_id
    assert "task-submit" in steps["hardware"]["run"]
    assert f"tools/run_hardware_ut.sh {arch} " in steps["hardware"]["run"]
    device_tests = (_ROOT / "tools/run_hardware_ut.sh").read_text()
    assert "requires_hardware" in device_tests
    assert '--platform "$arch"' in device_tests
    assert "task-submit" in steps["smoke"]["run"]
    assert "tools/build_cann_examples.sh" in steps["smoke"]["run"]


def test_rename_from_hardware_to_cpu_preserves_both_architectures(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "--quiet")
    git("config", "user.email", "ci@example.invalid")
    git("config", "user.name", "CI")
    old = tmp_path / "tests/ut/py/a5/hardware/test_case.py"
    old.parent.mkdir(parents=True)
    old.write_text("def test_case(): pass\n")
    git("add", "tests")
    git("-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "hardware test")
    base_sha = git("rev-parse", "HEAD")
    new = tmp_path / "tests/ut/py/wse/test_case.py"
    new.parent.mkdir(parents=True)
    old.rename(new)
    git("add", "--all")
    git("-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "move to simulator")
    flags, _ = _detect(tmp_path, base_sha=base_sha)
    assert flags["ut_arches"] == "a5,wse"
    assert flags["ut_hardware_a5"] == "true"
    assert flags["ut_hardware_a2a3"] == "false"


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ("docs/ci.md", False),
        ("src/a2a3/runtime/host_build_graph/runtime.h", False),
        ("src/wse/platform/sim/host/device_runner.cpp", False),
        ("src/a5/runtime/host_build_graph/runtime.h", True),
        ("simpler_setup/test_environments.py", True),
        ("tests/ut/cpp/a5/test_runtime.cpp", False),
        ("tests/st/a5/host_build_graph/test_scene.py", True),
    ],
)
def test_automatic_a5_framework_uses_actual_architecture_and_scene_flags(tmp_path, files, expected):
    flags, _ = _detect(tmp_path, files)
    assert ("st-framework-a5" in _selected_jobs(flags)) == expected


def test_unknown_diff_still_selects_every_automatic_job(tmp_path):
    flags, _ = _detect(tmp_path, "")
    workflow = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text())
    assert _selected_jobs(flags) == set(workflow["jobs"])
