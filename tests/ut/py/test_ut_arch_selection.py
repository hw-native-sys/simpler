# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise CPU architecture selection and resource declarations through pytest collection."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_test_environments import _pytest_subprocess_env

_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def project(tmp_path):
    (tmp_path / "tests/ut/py").mkdir(parents=True)
    for conftest in ("conftest.py", "tests/ut/py/conftest.py"):
        shutil.copyfile(_ROOT / conftest, tmp_path / conftest)
    (tmp_path / "pytest.ini").write_text("[pytest]\naddopts = --import-mode=importlib\n")
    return tmp_path


def write_case(project, path, declarations=""):
    target = project / path
    target.parent.mkdir(parents=True, exist_ok=True)
    args = "platform" if "parametrize" in declarations else ""
    target.write_text(f"import pytest\n{declarations}\ndef test_case({args}): pass\n")


def run_pytest(project, *options):
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *options, "tests"],
        check=False,
        cwd=project,
        env=_pytest_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=45,
    )


@pytest.mark.parametrize("selection", ["a2a3", "a5", "a2a3,wse", "a2a3,a5,wse"])
def test_selects_shared_and_matching_arches_without_filtering_scene_tests(project, selection):
    ut = "tests/ut/py/"
    shared, paired, params = (ut + f"test_{name}.py" for name in ("shared", "paired", "params"))
    scene = "tests/st/a5/test_example.py"
    write_case(project, shared)
    write_case(project, paired, '@pytest.mark.ut_arches("a5", "wse")')
    write_case(project, params, '@pytest.mark.parametrize("platform", ["a2a3sim", "a5sim", "wsesim"])')
    for arch in ("a2a3", "a5", "wse"):
        write_case(project, ut + arch + "/test_private.py")
    write_case(project, scene)
    result = run_pytest(project, "--collect-only", "--ut-arches", selection)
    assert result.returncode == 0, result.stdout + result.stderr
    arches = set(selection.split(","))
    expected = {shared + "::test_case", scene + "::test_case"}
    expected.update(ut + f"{arch}/test_private.py::test_case" for arch in arches)
    expected.update(params + f"::test_case[{arch}sim]" for arch in arches)
    if arches & {"a5", "wse"}:
        expected.add(paired + "::test_case")
    assert {line for line in result.stdout.splitlines() if "::" in line} == expected


@pytest.mark.parametrize(
    ("path", "resource", "declared_platform", "message"),
    [
        ("a2a3/test_wrong.py", "hardware", "a2a3", "hardware/ directory"),
        ("a2a3/nohardware/hardware/test_wrong.py", "hardware", "a2a3", "hardware/ directory"),
        ("a5/hardware/test_wrong.py", "hardware", "a2a3", "disagree with its a5 directory"),
        ("a5/hardware/test_wrong.py", "toolkit", "a5", "toolkit/ directory"),
        ("toolkit/test_wrong.py", "toolkit", "a5sim", "onboard platforms"),
    ],
)
def test_invalid_resource_is_reported_before_deselection(project, path, resource, declared_platform, message):
    write_case(
        project,
        "tests/ut/py/" + path,
        f'@pytest.mark.requires_{resource}\n@pytest.mark.platforms(["{declared_platform}"])',
    )
    result = run_pytest(
        project, "--collect-only", "--ut-arches", "wse", "-m", "not requires_hardware and not requires_toolkit"
    )
    assert result.returncode != 0
    assert message in result.stdout + result.stderr


def test_rejects_unknown_arch(project):
    result = run_pytest(project, "--collect-only", "--ut-arches", "a5,wsesim")
    assert result.returncode != 0
    assert "expects a comma-separated subset" in result.stderr


def test_sdk_runs_for_explicit_arch_without_a_device_platform(project):
    for resource in ("hardware", "toolkit"):
        for arch in ("a2a3", "a5"):
            write_case(
                project,
                f"tests/ut/py/{arch}/{resource}/test_owned.py",
                f'@pytest.mark.requires_{resource}\n@pytest.mark.platforms(["{arch}"])',
            )
    targeted = run_pytest(project, "--ut-arches", "a5")
    assert targeted.returncode == 0, targeted.stdout + targeted.stderr
    assert "1 passed, 1 skipped, 2 deselected" in targeted.stdout
    default = run_pytest(project)
    assert default.returncode == 0, default.stdout + default.stderr
    assert "4 skipped" in default.stdout


def test_onboard_parameter_matches_its_resource_mark(project):
    write_case(
        project,
        "tests/ut/py/hardware/test_wrong.py",
        '@pytest.mark.requires_hardware\n@pytest.mark.platforms(["a2a3"])\n'
        '@pytest.mark.parametrize("platform", ["a5"])',
    )
    result = run_pytest(project, "--collect-only", "-m", "not requires_hardware")
    assert result.returncode != 0
    assert "parameter platform=a5 disagrees with its onboard platforms" in result.stdout + result.stderr
