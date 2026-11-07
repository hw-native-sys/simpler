# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Workflow command and report contracts using CPU-only execution receipts."""

import json
import os
import shutil
import subprocess
import sys
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
    assert report(tmp_path, "default", "a5", "exclude") == int(mutation is not None)
    rendered = (tmp_path / "coverage-summary.md").read_text()
    assert step_summary.read_text() == rendered
    if mutation:
        assert "Not verified" in rendered
        assert "| Selected |" not in rendered
        assert not stale.exists()
    else:
        assert "Verified receipts" in rendered
        assert "kernel/tensormap_and_ringbuffer/eager | 1 | 1 | 1 | 0" in rendered
        assert "Not delivered" in rendered


def test_report_requires_both_a2a3_sdma_receipts(execution, tmp_path):
    result, plan, _ = execution(
        SOURCE, "--verify-feature-coverage", "--exclude-level", "4", "-m", "not sdma and not sdma_fault"
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
    "platform,step_name,filename,partition",
    [
        ("a2a3", "Run pytest scene tests (a2a3)", "main", "not sdma and not sdma_fault"),
        ("a2a3", "SDMA pytest (a2a3)", "sdma", "sdma and not sdma_fault"),
        ("a2a3", "SDMA fault recovery pytest (a2a3)", "sdma_fault", "sdma_fault"),
        ("a5", "Run pytest scene tests (a5)", "main", None),
    ],
)
@pytest.mark.parametrize("manual", ["exclude", "include"])
def test_default_workflow_commands_preserve_partition_and_receipt_path(
    tmp_path, platform, step_name, filename, partition, manual
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
    if partition:
        assert args[args.index("-m", 2) + 1] == partition
    if filename == "main":
        assert args[args.index("--manual") + 1] == manual
        assert args[args.index("--exclude-level") + 1] == "4"
    else:
        assert "--manual" not in args and "--exclude-level" not in args
