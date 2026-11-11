# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Workflow command and report contracts using CPU-only execution receipts."""

import ast
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import test_dispatch_receipts_prototype as receipt_tests
import yaml
from test_feature_coverage_prototype import FRAMEWORK, SOURCE
from test_test_environments import _run_pytest

from simpler_setup.tools.feature_coverage_report import report

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def execution():
    yield from receipt_tests.execution.__wrapped__()


def test_inventory_keeps_policy_failure_json(execution, tmp_path):
    inventory = tmp_path / "inventory.json"
    result, _, _ = execution(
        SOURCE.replace("def test_native_capture():", "def helper_native_capture():"),
        "--collect-only",
        "--verify-feature-coverage",
        "-p",
        "simpler_setup.tools.collection_inventory",
        "--collection-inventory",
        str(inventory),
    )
    assert result.returncode != 0
    assert "INTERNALERROR" not in result.stdout + result.stderr
    content = json.loads(inventory.read_text())
    assert "Required feature groups absent" in content["policy_error"]
    assert content["cases"] and "policy" not in content


def test_daily_pmu_skip_uses_real_declaration_and_reason(execution):
    path = ROOT / "tests/st/a2a3/host_build_graph/dfx/pmu/test_pmu_counting_window.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TestPmuCountingWindow")
    marker = next(
        node
        for node in cls.decorator_list
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "pytest.mark.coverage_skip"
    )
    skip = next(
        node for node in ast.walk(cls) if isinstance(node, ast.Call) and ast.unparse(node.func) == "pytest.skip"
    )
    source = (
        SOURCE
        + "\n"
        + "@"
        + ast.unparse(marker)
        + "\n"
        + """
@pytest.mark.manual
@pytest.mark.runtime("host_build_graph")
def test_pmu_without_events():
    SKIP
""".replace("SKIP", ast.unparse(skip))
    )
    result, plan, _ = execution(source, "--verify-feature-coverage", "--manual", "include")
    assert result.returncode == 0, result.stdout + result.stderr
    counts = json.loads(plan.with_suffix(".coverage.json").read_text())["groups"]["program/host_build_graph"]
    assert counts["passed"] == 1 and counts["skipped"] == 1


def test_auxiliary_probe_is_not_a_feature_claim(execution):
    source = (
        SOURCE
        + """
@pytest.mark.runtime_product("auxiliary")
@pytest.mark.runtime("tensormap_and_ringbuffer")
def test_auxiliary_probe():
    pass
"""
    )
    result, plan, _ = execution(source, "--verify-feature-coverage")
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(plan.with_suffix(".coverage.json").read_text())
    assert "kernel/tensormap_and_ringbuffer/capture_replay" in summary["not_delivered"]
    assert "auxiliary/tensormap_and_ringbuffer" in summary["groups"]


@pytest.mark.parametrize(
    "path,environment,platform",
    [
        ("tests/st/worker/framework_eager/test_native_submission.py", "torch_npu", "a2a3"),
        ("tests/st/worker/framework_eager/test_native_submission.py", "torch_npu", "a5"),
        ("examples/a5/tensormap_and_ringbuffer/host_comm_domain/test_host_comm_domain.py", "default", "a5"),
    ],
)
def test_native_consumers_have_collectable_feature_declarations(tmp_path, path, environment, platform):
    inventory = tmp_path / "inventory.json"
    result, _, _ = _run_pytest(
        tmp_path,
        (ROOT / path).read_text(),
        "--collect-only",
        "--platform",
        platform,
        "--manual",
        "include",
        "-p",
        "simpler_setup.tools.collection_inventory",
        "--collection-inventory",
        str(inventory),
        profile=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    content = json.loads(inventory.read_text())
    assert not content["classification_errors"]
    assert not content["duplicates"]
    assert content["cases"]
    expected_product = "kernel" if environment == "torch_npu" else "program"
    expected_mode = "eager" if environment == "torch_npu" else None
    assert {case["feature"]["mode"] for case in content["cases"]} == {expected_mode}
    assert {case["feature"]["product"] for case in content["cases"]} == {expected_product}


@pytest.mark.parametrize("phase", ["setup", "call", "teardown", "call_and_teardown"])
def test_failed_framework_cases_have_feature_counts(tmp_path, phase):
    fixture = """
@pytest.fixture(autouse=True)
def fail_phase():
    BEFORE
    yield
    AFTER
""".replace("BEFORE", 'raise RuntimeError("setup broke")' if phase == "setup" else "pass").replace(
        "AFTER", 'raise RuntimeError("teardown broke")' if phase in ("teardown", "call_and_teardown") else "pass"
    )
    source = FRAMEWORK + fixture
    if phase in ("call", "call_and_teardown"):
        source = source.replace("    pass", '    raise RuntimeError("call broke")', 1)
    result, _, _ = _run_pytest(tmp_path, source, "--environment-receipts", "--verify-feature-coverage")
    assert result.returncode != 0
    assert report(tmp_path, "torch_npu", "a2a3", "exclude") == 1
    summary = json.loads((tmp_path / "coverage-summary.json").read_text())
    partition = summary["partitions"]["results"]
    assert partition["verification_status"] == "failed"
    for counts in partition["groups"].values():
        assert counts == {
            "selected": 1,
            "executed": int(phase != "setup"),
            "passed": 0,
            "failed": 1,
            "skipped": 0,
            "missing": 0,
        }
    assert "| Failed |" in (tmp_path / "coverage-summary.md").read_text()


@pytest.mark.parametrize("mutation", ["missing", "foreign", "duplicate", "unexpected_skip"])
def test_partial_framework_report_never_qualifies_bad_evidence(tmp_path, mutation):
    result, _, xml = _run_pytest(tmp_path, FRAMEWORK, "--environment-receipts", "--verify-feature-coverage")
    assert result.returncode == 0, result.stdout + result.stderr
    tree = ET.parse(xml)
    suite = next(tree.getroot().iter("testsuite"))
    case = suite.find("testcase")
    assert case is not None
    if mutation == "missing":
        suite.remove(case)
    elif mutation == "foreign":
        source_property = case.find("properties/property[@name='simpler_source']")
        assert source_property is not None
        source_property.set("value", "foreign")
    elif mutation == "duplicate":
        suite.append(ET.fromstring(ET.tostring(case)))
    else:
        ET.SubElement(case, "skipped", {"type": "pytest.skip", "message": "unexpected"})
        suite.set("skipped", "1")
    tree.write(xml)
    assert report(tmp_path, "torch_npu", "a2a3", "exclude") == 1
    summary = json.loads((tmp_path / "coverage-summary.json").read_text())["partitions"]["results"]
    assert summary["verification_status"] != "verified"
    totals = {key: sum(group[key] for group in summary["groups"].values()) for key in ("passed", "missing", "skipped")}
    assert totals == {
        "passed": 1,
        "missing": int(mutation != "unexpected_skip"),
        "skipped": int(mutation == "unexpected_skip"),
    }


def test_missing_dispatcher_shard_retains_other_feature_counts(execution, tmp_path):
    result, plan, _ = execution(
        SOURCE, "--verify-feature-coverage", "--platform", "a5", "--exclude-level", "4", require_capture=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    shutil.copyfile(plan, tmp_path / "main.plan.json")
    manifest = json.loads(plan.with_suffix(".results.json").read_text())
    Path(manifest["files"][1]).unlink()
    (tmp_path / "main.plan.results.json").write_text(json.dumps(manifest))
    assert report(tmp_path, "default", "a5", "exclude") == 1
    partition = json.loads((tmp_path / "coverage-summary.json").read_text())["partitions"]["main"]
    assert partition["verification_status"] == "incomplete"
    assert sum(group["passed"] for group in partition["groups"].values()) > 0
    assert sum(group["missing"] for group in partition["groups"].values()) > 0


def test_inventory_preserves_collection_error(execution, tmp_path):
    inventory = tmp_path / "inventory.json"
    result, _, _ = execution(
        'raise ImportError("binding source mismatch")',
        "--collect-only",
        "--verify-feature-coverage",
        "-p",
        "simpler_setup.tools.collection_inventory",
        "--collection-inventory",
        str(inventory),
    )
    assert result.returncode == 2
    assert "binding source mismatch" in result.stdout
    assert "INTERNALERROR" not in result.stdout + result.stderr
    content = json.loads(inventory.read_text())
    assert content["collection_errors"] == 1
    assert "policy" not in content


@pytest.mark.parametrize("mutation", [None, "missing_shard", "wrong_scope", "wrong_source"])
def test_report_rechecks_real_dispatcher_results(execution, tmp_path, monkeypatch, mutation):
    result, plan, _ = execution(
        SOURCE, "--verify-feature-coverage", "--platform", "a5", "--exclude-level", "4", require_capture=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    shutil.copyfile(plan, tmp_path / "main.plan.json")
    shutil.copyfile(plan.with_suffix(".results.json"), tmp_path / "main.plan.results.json")
    manifest = json.loads((tmp_path / "main.plan.results.json").read_text())
    if mutation == "missing_shard":
        Path(manifest["files"][0]).unlink()
    elif mutation in ("wrong_scope", "wrong_source"):
        content = json.loads((tmp_path / "main.plan.json").read_text())
        if mutation == "wrong_scope":
            content["coverage"]["scope"]["manual"] = "include"
        else:
            content["source"] = "stale-source"
        (tmp_path / "main.plan.json").write_text(json.dumps(content))
    stale = tmp_path / "main.plan.coverage.json"
    stale.write_text('{"untrusted_cached_pass": true}')
    step_summary = tmp_path / "step-summary"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step_summary))
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.example")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/project")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    assert report(tmp_path, "default", "a5", "exclude") == int(mutation is not None)
    rendered = (tmp_path / "coverage-summary.md").read_text()
    assert step_summary.read_text() == rendered
    if mutation:
        assert "Not verified" in rendered
        assert ("| Selected |" in rendered) == (mutation == "missing_shard")
        assert not stale.exists()
    else:
        assert "Verified receipts" in rendered
        assert "kernel/tensormap_and_ringbuffer/eager | 1 | 1 | 1 | 0" in rendered
        assert "Not delivered" in rendered
        document = json.loads((tmp_path / "coverage-summary.json").read_text())
        assert document["run_url"] == "https://github.example/owner/project/actions/runs/123"
        assert document["partitions"]["main"]["identity"] == json.loads(plan.read_text())["identity"]


def test_environment_callers_keep_trigger_and_device_contracts():
    def workflow(name):
        return yaml.load((ROOT / f".github/workflows/{name}.yml").read_text(), Loader=yaml.BaseLoader)

    ci, daily, comment = (workflow(name) for name in ("ci", "daily", "ci-a5-comment"))
    assert ci["on"] == {"pull_request": {"branches": ["main"]}}
    assert "st-onboard-a5" not in ci["jobs"] and "st-framework-a5" not in ci["jobs"]
    for family in ("onboard", "framework"):
        job = ci["jobs"][f"st-{family}-a2a3"]
        assert job["needs"] == ["detect-changes", "pre-commit"]
        assert job["if"] == (
            "needs.detect-changes.outputs.a2a3_changed == 'true' && needs.detect-changes.outputs.st_affected == 'true'"
        )
        assert job["with"].get("manual_mode", "exclude") == "exclude"
        for platform in ("a2a3", "a5"):
            assert daily["jobs"][f"st-{family}-{platform}"]["with"]["manual_mode"] == "include"
        assert comment["jobs"][f"st-{family}-a5"]["needs"] == ["gate-and-resolve"]
    for name in ("ci", "daily", "ci-a5-comment"):
        callers = workflow(name)["jobs"]
        suites = [job for job in callers.values() if job.get("uses") == "./.github/workflows/_st-framework.yml"]
        assert len({job["with"]["platform"] for job in suites}) == len(suites)
        assert all("capture" not in key for key in callers)


def test_report_requires_both_a2a3_sdma_receipts(execution, tmp_path):
    result, plan, _ = execution(
        SOURCE,
        "--verify-feature-coverage",
        "--exclude-level",
        "4",
        "-m",
        "not sdma and not sdma_fault",
        "--coverage-partition",
        "main",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    shutil.copyfile(plan, tmp_path / "main.plan.json")
    shutil.copyfile(plan.with_suffix(".results.json"), tmp_path / "main.plan.results.json")
    assert report(tmp_path, "default", "a2a3", "exclude") == 1
    rendered = (tmp_path / "coverage-summary.md").read_text()
    assert rendered.count("Not verified") == 2
    assert "### sdma\n" in rendered and "### sdma_fault\n" in rendered


def test_manual_only_report_does_not_require_sdma(execution, tmp_path):
    source = SOURCE.replace("def test_program_hbg():", "@pytest.mark.manual\ndef test_program_hbg():")
    result, plan, _ = execution(
        source,
        "--verify-feature-coverage",
        "--exclude-level",
        "4",
        "--manual",
        "only",
        "-m",
        "not sdma and not sdma_fault",
        "--coverage-partition",
        "main",
        require_capture=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    shutil.copyfile(plan, tmp_path / "main.plan.json")
    shutil.copyfile(plan.with_suffix(".results.json"), tmp_path / "main.plan.results.json")
    assert report(tmp_path, "default", "a2a3", "only") == 0
    assert "### sdma" not in (tmp_path / "coverage-summary.md").read_text()


@pytest.mark.parametrize("empty", [False, True])
def test_framework_report_is_strict_and_empty_is_not_verified(tmp_path, empty):
    args = ["--environment-receipts", "--verify-feature-coverage"]
    if empty:
        args += ["--manual", "only"]
    result, _, _ = _run_pytest(tmp_path, FRAMEWORK, *args)
    assert (result.returncode != 0) == empty
    assert report(tmp_path, "torch_npu", "a2a3", "only" if empty else "exclude") == int(empty)
    rendered = (tmp_path / "coverage-summary.md").read_text()
    assert ("Not verified" in rendered) == empty


@pytest.mark.parametrize(
    "platform,step_name,filename,partition,coverage_partition",
    [
        ("a2a3", "Run pytest scene tests (a2a3)", "main", "not sdma and not sdma_fault", "main"),
        ("a2a3", "SDMA pytest (a2a3)", "sdma", "sdma and not sdma_fault", "sdma"),
        ("a2a3", "SDMA fault recovery pytest (a2a3)", "sdma_fault", "sdma_fault", "sdma_fault"),
        ("a5", "Run pytest scene tests (a5)", "main", None, "all"),
    ],
)
@pytest.mark.parametrize("manual", ["exclude", "include"])
def test_default_workflow_commands_preserve_partition_and_receipt_path(
    tmp_path, platform, step_name, filename, partition, coverage_partition, manual
):
    workflow = yaml.load((ROOT / f".github/workflows/_st-npu-{platform}.yml").read_text(), Loader=yaml.BaseLoader)
    step = next(s for s in workflow["jobs"]["run"]["steps"] if s["name"] == step_name)
    binary = tmp_path / "bin"
    binary.mkdir()
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        f"#!{sys.executable}\nimport json,os,sys\nopen(os.environ['COMMAND_LOG'],'w').write(json.dumps(sys.argv[1:]))\n"
    )
    python.chmod(0o755)
    task = binary / "task-submit"
    task.write_text(
        f"#!{sys.executable}\nimport os,subprocess,sys\n"
        "subprocess.run(['bash','-e','-c',sys.argv[sys.argv.index('--run')+1]], "
        "env={**os.environ,'TASK_DEVICE':'3,5'},check=True)\n"
    )
    task.chmod(0o755)
    log = tmp_path / "commands.json"
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"].replace("source /usr/local/Ascend/cann/set_env.sh", ":")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PATH": f"{binary}:{os.environ['PATH']}",
            "MANUAL_MODE": manual,
            "DEVICE_NUM": "2",
            "SESSION_TIMEOUT_S": "1200",
            "COMMAND_LOG": str(log),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads(log.read_text())
    assert args[:4] == ["-m", "pytest", "examples", "tests/st"]
    assert args[args.index("--platform") + 1] == platform
    assert args[args.index("--device") + 1] == "3,5"
    assert args[args.index("--junitxml") + 1] == f"outputs/default-ci/{filename}.xml"
    assert "--environment-receipts" in args and "--verify-feature-coverage" in args
    assert args[args.index("--coverage-partition") + 1] == coverage_partition
    if partition:
        assert args[args.index("-m", 2) + 1] == partition
    if filename == "main":
        assert args[args.index("--manual") + 1] == manual
        assert args[args.index("--exclude-level") + 1] == "4"
    else:
        assert "--manual" not in args and "--exclude-level" not in args
