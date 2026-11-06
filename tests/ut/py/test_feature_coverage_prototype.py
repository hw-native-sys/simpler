# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Actual collection/dispatch checks for feature obligations and scoped skips."""

import json
import xml.etree.ElementTree as ET

import pytest
import test_dispatch_receipts_prototype as receipt_tests
from test_test_environments import _run_pytest

from simpler_setup.test_environments import verify_result_files


@pytest.fixture
def execution():
    yield from receipt_tests.execution.__wrapped__()


SOURCE = """
import pytest

@pytest.mark.runtime_product("program")
@pytest.mark.runtime("host_build_graph")
@pytest.mark.execution_mode("eager")
def test_program_hbg():
    pass

@pytest.mark.runtime_product("program")
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.execution_mode("eager")
def test_program_tmr():
    pass

@pytest.mark.runtime_product("kernel")
@pytest.mark.runtime("host_build_graph")
@pytest.mark.execution_mode("eager")
@pytest.mark.device_count(1)
@pytest.mark.parametrize("index", [0, 1])
def test_native(index):
    pass

@pytest.mark.runtime_product("kernel")
@pytest.mark.execution_mode("eager")
class TestL2:
    _st_level = 2
    _st_runtime = "tensormap_and_ringbuffer"
    CASES = [{"name": "cpu", "platforms": ["a2a3", "a5"]}]
    def test_run(self):
        pass

@pytest.mark.runtime_product("kernel")
@pytest.mark.runtime("host_build_graph")
@pytest.mark.execution_mode("capture_replay")
@pytest.mark.platforms(["a2a3"])
def test_native_capture():
    pass
"""

OPTIONAL = """
@pytest.mark.runtime_product("kernel")
@pytest.mark.runtime("host_build_graph")
@pytest.mark.execution_mode("eager")
@pytest.mark.coverage_skip(platforms=["a2a3"], reason="optional probe unavailable")
def test_optional_probe():
    pytest.skip("optional probe unavailable")
"""


def test_complete_feature_matrix_and_parallel_dispatch(execution):
    result, plan, _ = execution(
        SOURCE, "--verify-feature-coverage", "-p", "xdist", "--max-parallel", "2", "--device", "0-1"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    content = json.loads(plan.read_text())
    assert len(content["coverage"]["required_groups"]) == 5
    assert len(content["coverage"]["required_nodeids"]) == 6
    report = json.loads(plan.with_suffix(".coverage.json").read_text())
    assert report["groups"]["kernel/host_build_graph/eager"]["passed"] == 2
    assert report["not_delivered"] == ["kernel/tensormap_and_ringbuffer/capture_replay"]


def test_inherited_scene_declarations_and_explicit_method_override(execution):
    source = SOURCE.replace(
        '@pytest.mark.runtime_product("kernel")\n@pytest.mark.execution_mode("eager")\nclass TestL2:',
        'class SceneBase:\n    _st_runtime_product = "kernel"\n    _st_execution_mode = "eager"\n\n'
        "class TestL2(SceneBase):",
    )
    source = source.replace(
        "    def test_run(self):\n        pass",
        """    def test_run(self):
        pass

    @pytest.mark.runtime_product("auxiliary")
    @pytest.mark.coverage_skip(platforms=["a2a3"], reason="placeholder")
    def test_placeholder(self):
        pytest.skip("placeholder")
""",
    )
    result, plan, _ = execution(source, "--verify-feature-coverage")
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(plan.with_suffix(".coverage.json").read_text())
    assert report["groups"]["kernel/tensormap_and_ringbuffer/eager"]["passed"] == 1
    assert report["groups"]["auxiliary/tensormap_and_ringbuffer/eager"]["skipped"] == 1


def test_coverage_runtime_does_not_change_runtime_filter(execution):
    source = (
        SOURCE
        + """
@pytest.mark.runtime_product("auxiliary")
@pytest.mark.execution_mode("eager")
@pytest.mark.coverage_runtime("common")
def test_runtime_independent_probe():
    pass
"""
    )
    result, plan, _ = execution(source, "--verify-feature-coverage", "--runtime", "host_build_graph")
    assert result.returncode == 0, result.stdout + result.stderr
    content = json.loads(plan.read_text())
    assert any(case["nodeid"].endswith("test_runtime_independent_probe") for case in content["cases"])


@pytest.mark.parametrize("expression", ["not test_native_capture", "not test_native[1]", "not TestL2"])
def test_required_group_and_member_cannot_be_filtered_out(execution, expression):
    result, _, _ = execution(SOURCE, "--verify-feature-coverage", "-k", expression)
    assert result.returncode != 0
    assert "Required feature cases deselected" in result.stderr


def test_missing_group_is_detected_independently_of_selection(execution):
    source = SOURCE.replace("def test_program_hbg():", "def helper_program_hbg():")
    result, _, _ = execution(source, "--verify-feature-coverage")
    assert result.returncode != 0
    assert "Required feature groups absent" in result.stderr


def test_missing_classification_cannot_hide_behind_k(execution):
    source = SOURCE.replace('@pytest.mark.runtime_product("program")', "", 1)
    result, _, _ = execution(source, "--verify-feature-coverage", "-k", "not program_hbg")
    assert result.returncode != 0
    assert "feature classification" in result.stderr


def test_allowed_skip_is_not_a_pass(execution):
    result, plan, _ = execution(SOURCE + OPTIONAL, "--verify-feature-coverage")
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(plan.with_suffix(".coverage.json").read_text())
    assert report["groups"]["kernel/host_build_graph/eager"] == {
        "selected": 3,
        "executed": 3,
        "passed": 2,
        "skipped": 1,
    }


def test_setup_skip_is_not_counted_as_executed(execution):
    source = OPTIONAL.replace(
        "def test_optional_probe():",
        '@pytest.mark.skip(reason="optional probe unavailable")\ndef test_optional_probe():',
    )
    result, plan, _ = execution(SOURCE + source, "--verify-feature-coverage")
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(plan.with_suffix(".coverage.json").read_text())
    assert report["groups"]["kernel/host_build_graph/eager"] == {
        "selected": 3,
        "executed": 2,
        "passed": 2,
        "skipped": 1,
    }


@pytest.mark.parametrize("platform", ["a2a3", "a5"])
@pytest.mark.parametrize("declared", [True, False])
def test_quarantined_scene_skip_is_visible_but_not_execution(execution, platform, declared):
    quarantine = """
@pytest.mark.skip(reason="known device timeout")
@pytest.mark.coverage_skip(platforms=["a2a3", "a5"], reason="known device timeout")
class TestQuarantined:
    _st_level = 3
    _st_runtime = "tensormap_and_ringbuffer"
    _st_runtime_product = "program"
    _st_execution_mode = "eager"
    CASES = [{"name": "quarantined", "platforms": ["a5"]}]
    def test_run(self):
        raise AssertionError("quarantined test must not run")
"""
    if not declared:
        quarantine = quarantine.replace(
            '@pytest.mark.coverage_skip(platforms=["a2a3", "a5"], reason="known device timeout")', ""
        )
    result, plan, _ = execution(
        SOURCE + quarantine, "--verify-feature-coverage", "--platform", platform, require_capture=False
    )
    content = json.loads(plan.read_text())
    target = next(case for case in content["cases"] if "TestQuarantined" in case["nodeid"])
    assert (target["nodeid"] in content["coverage"]["required_nodeids"]) != declared
    if not declared:
        assert result.returncode != 0
        assert "Selected test environment reports a failure, error or skip" in result.stderr
        assert not plan.with_suffix(".coverage.json").exists()
        return
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(plan.with_suffix(".coverage.json").read_text())
    assert report["groups"]["program/tensormap_and_ringbuffer/eager"] == {
        "selected": 2,
        "executed": 1,
        "passed": 1,
        "skipped": 1,
    }


def test_passing_without_call_evidence_is_rejected(execution):
    result, plan, paths = execution(SOURCE, "--verify-feature-coverage")
    assert result.returncode == 0, result.stdout + result.stderr
    tree = ET.parse(paths[0])
    prop = next(p for p in tree.getroot().iter("property") if p.get("name") == "simpler_executed")
    prop.set("value", "false")
    tree.write(paths[0])
    with pytest.raises(ValueError, match="without call execution"):
        verify_result_files(plan, paths, dispatcher=True)


@pytest.mark.parametrize("change", ["reason", "platform", "xfail", "mandatory"])
def test_skip_exception_is_exact_and_scoped(execution, change):
    optional = OPTIONAL
    if change == "reason":
        optional = optional.replace('pytest.skip("optional probe unavailable")', 'pytest.skip("unexpected failure")')
    elif change == "platform":
        optional = optional.replace('platforms=["a2a3"]', 'platforms=["a5"]')
    elif change == "xfail":
        optional = optional.replace("pytest.skip(", "pytest.xfail(")
    else:
        optional = optional.replace(
            '@pytest.mark.coverage_skip(platforms=["a2a3"], reason="optional probe unavailable")', ""
        )
    result, _, _ = execution(SOURCE + optional, "--verify-feature-coverage")
    assert result.returncode != 0
    assert "VERIFICATION_FAILED" in result.stderr


def test_skip_exemption_cannot_replace_required_group(execution):
    source = SOURCE.replace(
        "@pytest.mark.device_count(1)", '@pytest.mark.coverage_skip(platforms=["a2a3"], reason="unavailable")'
    )
    result, _, _ = execution(source, "--verify-feature-coverage")
    assert result.returncode != 0
    assert "Required feature groups absent" in result.stderr


def test_a5_does_not_claim_native_capture(execution):
    result, plan, _ = execution(SOURCE, "--verify-feature-coverage", "--platform", "a5", require_capture=False)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(plan.with_suffix(".coverage.json").read_text())
    assert "kernel/host_build_graph/capture_replay" in report["not_delivered"]
    assert all("capture_replay" not in group for group in report["groups"])


def test_manual_only_does_not_demand_nonmanual_groups(execution):
    source = SOURCE.replace("def test_program_hbg():", "@pytest.mark.manual\ndef test_program_hbg():")
    result, plan, _ = execution(source, "--verify-feature-coverage", "--manual", "only", require_capture=False)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(plan.with_suffix(".coverage.json").read_text())
    assert list(report["groups"]) == ["program/host_build_graph/eager"]
    assert "kernel/host_build_graph/capture_replay" in report["not_in_scope"]


def test_sdma_partition_does_not_demand_main_capture(execution):
    source = SOURCE.replace("def test_program_hbg():", "@pytest.mark.sdma\ndef test_program_hbg():")
    result, plan, _ = execution(
        source, "--verify-feature-coverage", "-m", "sdma and not sdma_fault", require_capture=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(plan.with_suffix(".coverage.json").read_text())
    assert list(report["groups"]) == ["program/host_build_graph/eager"]
    assert report["scope"]["partition"] == "sdma"


FRAMEWORK = """
import pytest
pytestmark = [pytest.mark.test_environment("torch_npu"),
              pytest.mark.runtime_product("kernel"), pytest.mark.execution_mode("eager")]
@pytest.mark.parametrize("runtime", ["host_build_graph", "tensormap_and_ringbuffer"])
def test_argument_capture(runtime):
    pass
"""


def test_framework_argument_capture_stays_eager(tmp_path):
    result, plan, _ = _run_pytest(tmp_path, FRAMEWORK, "--environment-receipts", "--verify-feature-coverage")
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(plan.with_suffix(".coverage.json").read_text())
    assert len(report["groups"]) == 2
    assert all(group.endswith("/eager") for group in report["groups"])
    assert len(report["not_delivered"]) == 2


def test_framework_skip_rules_are_not_weakened(tmp_path):
    source = FRAMEWORK.replace("    pass", '    pytest.skip("optional probe unavailable")')
    result, _, _ = _run_pytest(tmp_path, source, "--environment-receipts", "--verify-feature-coverage")
    assert result.returncode != 0
    assert "VERIFICATION_FAILED" in result.stderr


def test_framework_cannot_declare_skip_exception(tmp_path):
    source = FRAMEWORK.replace(
        "def test_argument_capture(runtime):",
        '@pytest.mark.coverage_skip(platforms=["a2a3"], reason="unavailable")\ndef test_argument_capture(runtime):',
    )
    result, _, _ = _run_pytest(tmp_path, source, "--environment-receipts", "--verify-feature-coverage")
    assert result.returncode != 0
    assert "framework skip exceptions are forbidden" in result.stderr


@pytest.mark.parametrize("mutation", ["program_capture", "runtime_conflict", "unsupported_capture"])
def test_invalid_feature_combinations_fail_before_execution(execution, mutation):
    source = SOURCE
    if mutation == "program_capture":
        source = source.replace(
            '@pytest.mark.execution_mode("eager")', '@pytest.mark.execution_mode("capture_replay")', 1
        )
    elif mutation == "runtime_conflict":
        source = source.replace("class TestL2:", '@pytest.mark.runtime("host_build_graph")\nclass TestL2:')
    else:
        source += """
@pytest.mark.runtime_product("kernel")
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.execution_mode("capture_replay")
def test_undelivered():
    raise AssertionError("must not execute")
"""
    result, _, _ = execution(source, "--verify-feature-coverage")
    assert result.returncode != 0
    assert "must not execute" not in result.stdout


def test_daily_include_keeps_manual_members_required(execution):
    source = SOURCE.replace("def test_program_hbg():", "@pytest.mark.manual\ndef test_program_hbg():")
    result, _, _ = execution(source, "--verify-feature-coverage", "--manual", "include", "-k", "not program_hbg")
    assert result.returncode != 0
    assert "Required feature cases deselected" in result.stderr


def test_optional_setup_skip_and_poison_retry_remain_distinct(execution):
    poison = receipt_tests.POISON[len(receipt_tests.CASES) :].replace(
        "class TestPoison:", '@pytest.mark.runtime_product("kernel")\nclass TestPoison:'
    )
    optional = OPTIONAL.replace(
        "def test_optional_probe():",
        '@pytest.mark.skip(reason="optional probe unavailable")\ndef test_optional_probe():',
    )
    result, plan, _ = execution(
        SOURCE + poison + optional,
        "--verify-feature-coverage",
        "-p",
        "xdist",
        "--max-parallel",
        "2",
        "--device",
        "0-1",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(plan.with_suffix(".coverage.json").read_text())
    assert report["recovered"] == 1
    assert report["groups"]["kernel/host_build_graph/eager"] == {
        "selected": 3,
        "executed": 2,
        "passed": 2,
        "skipped": 1,
    }
    assert report["groups"]["kernel/tensormap_and_ringbuffer/eager"] == {
        "selected": 2,
        "executed": 2,
        "passed": 2,
        "skipped": 0,
    }
