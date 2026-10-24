# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU checks of the selected hardware test's dependency and JUnit contract."""

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from simpler_setup.test_environments import verify_results

_ROOT = Path(__file__).resolve().parents[3]
_ST_RELATIVE = Path("tests/st/a2a3/framework_eager/test_framework_eager.py")
_ST = _ROOT / _ST_RELATIVE
_CASES = {
    f"test_framework_queue_orders_public_worker[{queue}-{runtime}]"
    for queue in ("0", "1")
    for runtime in ("tensormap_and_ringbuffer", "host_build_graph")
}


def _run_selected_contract(tmp_path, *, missing):
    # Only the external compiler/device subprocess is replaced. Pytest executes
    # the actual fixture, four parameterizations, and consumer JSON assertions.
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    requires_hardware\n    platforms(value)\n    test_environment(value)\n"
    )
    test_path = tmp_path / _ST_RELATIVE
    test_path.parent.mkdir(parents=True)
    test_path.write_text(
        f"""
import builtins
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

real_import = builtins.__import__
def import_without_framework(name, *args, **kwargs):
    if name.split(".")[0] in ("torch", "torch_npu"):
        raise AssertionError("the CPU test contract must not import a framework backend")
    return real_import(name, *args, **kwargs)
builtins.__import__ = import_without_framework

original = runpy.run_path({str(_ST)!r})
framework_module = original["framework_module"]
test_framework_queue_orders_public_worker = original["test_framework_queue_orders_public_worker"]
pytestmark = original["pytestmark"]

@pytest.fixture(scope="session", autouse=True)
def cpu_transport():
    patch = pytest.MonkeyPatch()
    namespace = framework_module.__wrapped__.__globals__
    missing = {missing!r}
    def find_spec(name):
        assert name in ("torch", "torch_npu")
        return None if name == missing else SimpleNamespace(origin="/cpu-contract/" + name)
    def run(command, **kwargs):
        assert kwargs["env"]["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
        if "--compile-only" in command:
            module = Path(command[command.index("--build-dir") + 1]) / "cpu-contract.so"
            module.write_bytes(b"CPU test transport; no framework code")
            output = "FRAMEWORK_QUEUE_MODULE " + str(module) + "\\n"
        else:
            runtime = command[command.index("--runtime") + 1]
            queue = kwargs["env"]["TASK_QUEUE_ENABLE"]
            summary = dict(verified=True, calls=3, runtime=runtime, task_queue=queue,
                caller_tensors_retained=True, caller_stream_after_close=True,
                callback_error_observed=True, zero_stream_preflight_refusals=1,
                caller_thread=1, callback_threads=[1 if queue == "0" else 2] * 3)
            output = "FRAMEWORK_EAGER_PASS " + json.dumps(summary) + "\\n"
        return SimpleNamespace(returncode=0, stdout=output, stderr="")
    patch.setitem(namespace, "importlib", SimpleNamespace(
        util=SimpleNamespace(find_spec=find_spec),
        metadata=SimpleNamespace(version=lambda name: "CPU-contract")))
    patch.setitem(namespace, "subprocess", SimpleNamespace(run=run))
    yield
    patch.undo()
"""
    )
    (tmp_path / "conftest.py").write_text(
        f"""
import subprocess
import sys
import pytest
sys.path.insert(0, {str(_ROOT)!r})
from simpler_setup import test_environments as environments

def pytest_addoption(parser):
    parser.addoption("--device", default="0")
    parser.addoption("--test-environment", default="torch_npu")

def pytest_collection_finish(session):
    # Device/compiler transport is synthetic in this CPU fixture contract.
    environments.preflight = lambda name: {{"environment": name, "cpu_transport": True}}
    environments._source = lambda: subprocess.check_output(
        ["git", "-C", {str(_ROOT)!r}, "rev-parse", "HEAD"], text=True).strip()
    environments.prepare_session(session)

@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_sessionfinish(session):
    yield
    environments.finish_session(session)
"""
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            "pytest.ini",
            "--rootdir=.",
            "--import-mode=importlib",
            "--confcutdir=.",
            "--junitxml=results.xml",
            "-q",
            str(_ST_RELATIVE),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
    )
    return result, ET.parse(tmp_path / "results.xml").getroot().findall(".//testcase")


def test_selected_framework_matrix_has_four_passed_junit_cases(tmp_path):
    result, cases = _run_selected_contract(tmp_path, missing=None)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(cases) == 4
    assert {case.attrib["name"] for case in cases} == _CASES
    assert all(not any(child.tag in {"failure", "error", "skipped"} for child in case) for case in cases)
    verify_results(tmp_path / "results.plan.json", tmp_path / "results.xml")


def test_framework_ci_verifier_rejects_changed_actual_junit_classname(tmp_path):
    result, _ = _run_selected_contract(tmp_path, missing=None)
    assert result.returncode == 0, result.stdout + result.stderr
    path = tmp_path / "results.xml"
    verify_results(tmp_path / "results.plan.json", path)
    tree = ET.parse(path)
    case = tree.getroot().find(".//testcase")
    assert case is not None
    case.set("classname", "wrong.framework_module")
    tree.write(path)
    with pytest.raises(ValueError, match="wrong JUnit identity"):
        verify_results(tmp_path / "results.plan.json", path)


@pytest.mark.parametrize("missing", ["torch", "torch_npu"])
def test_selected_framework_matrix_fails_missing_dependency_instead_of_skipping(tmp_path, missing):
    result, cases = _run_selected_contract(tmp_path, missing=missing)
    assert result.returncode == 1, result.stdout + result.stderr
    assert len(cases) == 4
    assert {case.attrib["name"] for case in cases} == _CASES
    assert all(case.find("error") is not None and case.find("skipped") is None for case in cases)
    assert "requires matching torch and torch_npu" in result.stdout


def test_framework_setup_preserves_the_callers_installed_toolchain(tmp_path):
    action = yaml.safe_load((_ROOT / ".github/actions/setup-framework-eager/action.yml").read_text())
    bootstrap = action["runs"]["steps"][0]["run"]
    home = tmp_path / "caller toolkit"
    home.mkdir()
    (home / "ascend_toolkit_install.info").write_text("version=9.0.0\n")
    script = home / "set_env.sh"
    script.write_text(f'export ASCEND_HOME_PATH="{home}"\n')
    default = tmp_path / "runner-default.sh"
    default.write_text('export ASCEND_HOME_PATH="/not-the-callers-toolkit"\n')
    # The runner's default script is a filesystem boundary; the action body
    # and actual selector otherwise execute unchanged, without loading CANN.
    bootstrap = bootstrap.replace("/usr/local/Ascend/cann/set_env.sh", str(default))
    output = tmp_path / "action-output"
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", bootstrap],
        cwd=_ROOT,
        env={**os.environ, "ASCEND_HOME_PATH": str(home), "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_text() == f"cann_env={script}\n"


@pytest.mark.parametrize("mode", ["exclude", "include", "only", "invalid"])
def test_framework_workflow_propagates_manual_selection(tmp_path, mode):
    workflow = yaml.load((_ROOT / ".github/workflows/_st-framework-eager-a2a3.yml").read_text(), Loader=yaml.BaseLoader)
    assert workflow["on"]["workflow_call"]["inputs"]["manual_mode"]["default"] == "exclude"
    daily = yaml.load((_ROOT / ".github/workflows/daily.yml").read_text(), Loader=yaml.BaseLoader)
    assert daily["jobs"]["st-framework-eager-a2a3"]["with"]["manual_mode"] == "include"
    step = next(step for step in workflow["jobs"]["run"]["steps"] if "run" in step)
    assert step["env"]["MANUAL_MODE"] == "${{ inputs.manual_mode }}"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = tmp_path / ".venv-framework/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "with open(os.environ['ARGV_LOG'], 'a') as stream: stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    python.chmod(0o755)
    task = bin_dir / "task-submit"
    task.write_text(
        f"#!{sys.executable}\nimport json, os, subprocess, sys\n"
        "args = sys.argv[1:]\n"
        "assert ['--env', 'MANUAL_MODE'] == args[2:4], args\n"
        "subprocess.run(['bash', '-e', '-c', args[args.index('--run') + 1]], "
        "env={**os.environ, 'TASK_DEVICE': '0'}, check=True)\n"
    )
    task.chmod(0o755)
    git = bin_dir / "git"
    git.write_text("#!/bin/sh\nprintf 'CPU-workflow-contract\\n'\n")
    git.chmod(0o755)
    cann = tmp_path / "toolkit env.sh"
    cann.write_text(":\n")
    log = tmp_path / "argv.jsonl"
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "CANN_ENV": str(cann),
            "MANUAL_MODE": mode,
            "ARGV_LOG": str(log),
        },
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if mode == "invalid":
        assert result.returncode != 0
        assert "manual_mode must be" in result.stdout
        assert not log.exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        commands = [json.loads(line) for line in log.read_text().splitlines()]
        command = next(args for args in commands if args[:2] == ["-m", "pytest"])
        assert command[command.index("--manual") + 1] == mode
