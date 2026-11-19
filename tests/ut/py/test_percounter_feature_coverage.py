# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Percounter collection and platform-scoped Program coverage without device execution."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import test_dispatch_receipts_prototype as receipt_tests
from test_feature_coverage_prototype import FRAMEWORK, SOURCE
from test_test_environments import _run_pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def execution():
    yield from receipt_tests.execution.__wrapped__()


@pytest.mark.parametrize("declaration", ["scene", "runtime", "coverage_runtime", "parameter"])
def test_percounter_declarations_emit_verified_program_plan(execution, declaration):
    source = SOURCE.replace("def test_program_percounter():", "def helper_program_percounter():")
    if declaration == "scene":
        source += """
class TestPercounter:
    _st_level = 2
    _st_runtime = "percounter"
    CASES = [{"name": "cpu", "platforms": ["a5"]}]
    def test_run(self):
        pass
"""
    elif declaration == "parameter":
        source += """
@pytest.mark.parametrize("runtime", ["percounter"])
def test_percounter(runtime):
    pass
"""
    else:
        source += f"""
@pytest.mark.{declaration}("percounter")
def test_percounter():
    pass
"""
    result, plan, _ = execution(source, "--platform", "a5", "--verify-feature-coverage", require_capture=False)
    assert result.returncode == 0, result.stdout + result.stderr
    content = json.loads(plan.read_text())
    required = content["coverage"]["required_groups"]
    assert "program/percounter" in required
    assert not any(group.startswith("kernel/percounter/") for group in required)
    members = [case for case in content["cases"] if case["feature"]["runtime"] == "percounter"]
    assert len(members) == 1
    assert members[0]["nodeid"] in content["coverage"]["required_nodeids"]
    report = json.loads(plan.with_suffix(".coverage.json").read_text())
    assert report["groups"]["program/percounter"]["passed"] == 1


@pytest.mark.parametrize("selection", ["absent", "deselected"])
def test_a5_requires_percounter_program_members(execution, selection):
    source = SOURCE
    options = []
    if selection == "absent":
        source = source.replace("def test_program_percounter():", "def helper_program_percounter():")
    else:
        options += ["-k", "not program_percounter"]
    result, _, _ = execution(source, "--platform", "a5", "--verify-feature-coverage", *options, require_capture=False)
    assert result.returncode == 4
    expected = "Required feature groups absent" if selection == "absent" else "Required feature cases deselected"
    assert expected in result.stderr


def test_a2a3_does_not_require_percounter(execution):
    result, plan, _ = execution(SOURCE, "--verify-feature-coverage")
    assert result.returncode == 0, result.stdout + result.stderr
    content = json.loads(plan.read_text())
    assert not any("percounter" in group for group in content["coverage"]["required_groups"])
    assert not any(case["feature"]["runtime"] == "percounter" for case in content["cases"])


@pytest.mark.parametrize("platform,product", [("a2a3", "program"), ("a5", "kernel")])
@pytest.mark.parametrize("declaration", ["scene", "runtime"])
def test_percounter_does_not_qualify_unsupported_features(execution, platform, product, declaration):
    if declaration == "scene":
        source = f"""
import pytest
@pytest.mark.runtime_product("{product}")
class TestUnsupported:
    _st_runtime = "percounter"
    _st_execution_mode = "eager"
    def test_run(self):
        raise AssertionError("must not execute")
"""
    else:
        mode = '@pytest.mark.execution_mode("eager")' if product == "kernel" else ""
        source = f"""
import pytest
@pytest.mark.runtime_product("{product}")
@pytest.mark.runtime("percounter")
{mode}
def test_unsupported():
    raise AssertionError("must not execute")
"""
    result, plan, _ = execution(source, "--platform", platform, "--verify-feature-coverage", require_capture=False)
    assert result.returncode == 4
    assert "percounter" in result.stderr or "feature classification" in result.stderr
    assert not plan.exists()
    assert "must not execute" not in result.stdout


@pytest.mark.parametrize("selection", ["runtime", "manual"])
def test_percounter_only_scope_does_not_demand_other_runtimes(execution, selection):
    source = SOURCE
    if selection == "runtime":
        options = ["--runtime", "percounter"]
    else:
        source = source.replace("def test_program_percounter():", "@pytest.mark.manual\ndef test_program_percounter():")
        options = ["--manual", "only"]
    result, plan, _ = execution(
        source, "--platform", "a5", "--verify-feature-coverage", *options, require_capture=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    content = json.loads(plan.read_text())
    assert content["coverage"]["required_groups"] == ["program/percounter"]


def test_a5_framework_retains_hbg_and_tmr_requirements(tmp_path):
    result, plan, _ = _run_pytest(
        tmp_path, FRAMEWORK, "--platform", "a5", "--environment-receipts", "--verify-feature-coverage"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(plan.read_text())["coverage"]["required_groups"] == [
        "kernel/host_build_graph/eager",
        "kernel/tensormap_and_ringbuffer/eager",
    ]


@pytest.mark.parametrize(
    "path,manual",
    [
        ("examples/a5/host_mte_domain/test_host_mte_domain.py", "exclude"),
        ("tests/st/a5/percounter", "exclude"),
        ("tests/st/a5/percounter", "include"),
    ],
)
def test_actual_percounter_modules_collect_without_device_execution(tmp_path, path, manual):
    inventory = tmp_path / "inventory.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            path,
            "--collect-only",
            "--platform",
            "a5",
            "--runtime",
            "percounter",
            "--exclude-level",
            "4",
            "--manual",
            manual,
            "--environment-receipts",
            "--verify-feature-coverage",
            "--coverage-partition",
            "all",
            "-p",
            "simpler_setup.tools.collection_inventory",
            "--collection-inventory",
            str(inventory),
        ],
        cwd=ROOT,
        env={
            **{key: value for key, value in os.environ.items() if not key.startswith("PYTEST_XDIST_")},
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
            "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "python"))),
        },
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    content = json.loads(inventory.read_text())
    assert content["cases"]
    assert not content["classification_errors"]
    assert not content["collection_errors"]
    assert not content["duplicates"]
    assert content["policy"]["required_groups"] == ["program/percounter"]
    assert set(content["policy"]["required_nodeids"]) == {case["nodeid"] for case in content["cases"]}
