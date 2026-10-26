# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU contracts for platform-consistent framework CI installation and execution."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[3]
_ACTION = _ROOT / ".github/actions/setup-framework-eager"
_WORKFLOW = _ROOT / ".github/workflows/_st-framework-eager.yml"


def _yaml(path):
    return yaml.load(path.read_text(), Loader=yaml.BaseLoader)


def _executable(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    path.chmod(0o755)


@pytest.mark.parametrize("platform_name", ["a2a3", "a5", "a5sim", "../a5", ""])
def test_setup_selects_only_the_validated_build_target(tmp_path, platform_name):
    action = _yaml(_ACTION / "action.yml")
    step = next(step for step in action["runs"]["steps"] if step.get("id") == "platform")
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        env={**os.environ, "PLATFORM": platform_name, "GITHUB_OUTPUT": str(output)},
        text=True,
        capture_output=True,
        check=False,
    )
    if platform_name in ("a2a3", "a5"):
        assert result.returncode == 0, result.stderr
        assert output.read_text() == f"build_target=build_package_{platform_name}\n"
        install = next(step for step in action["runs"]["steps"] if "Install isolated" in step["name"])
        assert install["env"]["BUILD_TARGET"] == "${{ steps.platform.outputs.build_target }}"
        assert '--config-settings="build.targets=$BUILD_TARGET"' in install["run"]
    else:
        assert result.returncode != 0
        assert "platform must be a2a3 or a5" in result.stdout
        assert not output.exists()


@pytest.mark.parametrize("platform_name", ["a2a3", "a5"])
@pytest.mark.parametrize("shell_builtins", ["native", "bash3"])
def test_install_body_builds_the_selected_sdk(tmp_path, platform_name, shell_builtins):
    action = _yaml(_ACTION / "action.yml")
    install = next(step for step in action["runs"]["steps"] if "Install isolated" in step["name"])
    binary = tmp_path / "bin"
    command_log = tmp_path / "install.jsonl"
    requirements = [
        line for line in (_ACTION / "constraints.txt").read_text().splitlines() if line and not line.startswith("#")
    ]
    _executable(
        binary / "python",
        f"#!{sys.executable}\nimport json,os,sys\n"
        "with open(os.environ['COMMAND_LOG'],'a') as out: out.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "if sys.argv[1:4] == ['simpler_setup/test_environments.py', 'requirements', 'torch_npu']:\n"
        f"    selected = {requirements!r}\n"
        "    if len(sys.argv) == 5: selected = [item for item in selected if item.split('==')[0] == sys.argv[4]]\n"
        "    print('\\n'.join(selected))\n",
    )
    _executable(binary / "python3", "#!/bin/sh\nexit 0\n")
    activate = tmp_path / ".venv-framework/bin/activate"
    activate.parent.mkdir(parents=True)
    activate.write_text(":\n")
    cann = tmp_path / "set_env.sh"
    cann.write_text(":\n")
    # Apple Bash lacks the array-reading builtins introduced in Bash 4.
    shell_setup = "enable -n mapfile readarray 2>/dev/null || :\n" if shell_builtins == "bash3" else ""
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", shell_setup + install["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{binary}:{os.environ['PATH']}",
            "BUILD_TARGET": f"build_package_{platform_name}",
            "CANN_ENV": str(cann),
            "COMMAND_LOG": str(command_log),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [json.loads(line) for line in command_log.read_text().splitlines()]
    wheel_installs = [
        args for args in commands if args[:5] == ["-m", "pip", "install", "--ignore-installed", "--no-deps"]
    ]
    assert wheel_installs == [
        [
            "-m",
            "pip",
            "install",
            "--ignore-installed",
            "--no-deps",
            requirements[0],
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ],
        ["-m", "pip", "install", "--ignore-installed", "--no-deps", requirements[1]],
    ]
    dependency_install = next(args for args in commands if "scikit-build-core>=0.10" in args)
    assert dependency_install == [
        "-m",
        "pip",
        "install",
        "-c",
        ".github/actions/setup-framework-eager/constraints.txt",
        *requirements,
        "scikit-build-core>=0.10",
        "nanobind>=2,<3",
        "cmake>=3.20",
        "ninja",
        "setuptools",
    ]
    sdk_install = next(args for args in commands if any(arg.startswith("--config-settings=") for arg in args))
    assert sdk_install[:3] == ["-m", "pip", "install"]
    assert f"--config-settings=build.targets=build_package_{platform_name}" in sdk_install


def _identity_module():
    spec = importlib.util.spec_from_file_location("framework_ci_identity", _ACTION / "identity.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("platform_name", ["a2a3", "a5"])
def test_identity_records_the_selected_platform_runtime_and_environment(tmp_path, monkeypatch, platform_name):
    identity = _identity_module()
    for candidate in ("a2a3", "a5"):
        for runtime in ("host_build_graph", "tensormap_and_ringbuffer"):
            directory = tmp_path / "build/lib" / candidate / "onboard" / runtime
            directory.mkdir(parents=True)
            for name in ("libhost_runtime.so", "libaicpu_kernel.so", "aicore_kernel.o"):
                (directory / name).write_text(f"{candidate}/{runtime}/{name}")
    module_path = tmp_path / "module.so"
    module_path.write_bytes(b"CPU identity transport")
    for name in ("torch", "torch_npu", "_task_interface"):
        monkeypatch.setitem(
            sys.modules,
            name,
            SimpleNamespace(__file__=str(module_path), __version__="CPU-transport", __build_commit__="fixed-source"),
        )
    monkeypatch.setattr(identity.subprocess, "check_output", lambda *args, **kwargs: "fixed-source\n")
    observed_environment = {"cann": {"version": "9.2.0-beta.2"}, "cpu_transport": True}
    monkeypatch.setitem(sys.modules, "simpler_setup.environment", SimpleNamespace(PROJECT_ROOT=tmp_path))
    monkeypatch.setitem(
        sys.modules, "simpler_setup.test_environments", SimpleNamespace(preflight=lambda name: observed_environment)
    )
    monkeypatch.setattr(sys, "argv", ["identity.py", "--platform", platform_name])
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs/framework-eager-ci").mkdir(parents=True)
    identity.main()
    record = json.loads((tmp_path / "outputs/framework-eager-ci/identity.json").read_text())
    assert record["platform"] == platform_name
    assert record["environment"] == observed_environment
    for runtime in record["runtimes"].values():
        for file in runtime.values():
            assert Path(file["path"]).is_relative_to(tmp_path / "build/lib" / platform_name / "onboard")


def test_identity_refuses_bad_platform_before_optional_backend_import(tmp_path):
    script = """
import builtins, runpy, sys
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name in ('torch', 'torch_npu', '_task_interface'):
        raise AssertionError('optional backend imported before platform validation')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ['identity.py', '--platform', 'a5sim']
runpy.run_path(sys.argv_path, run_name='__main__')
""".replace("sys.argv_path", repr(str(_ACTION / "identity.py")))
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(_ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "invalid choice" in result.stderr
    assert "optional backend imported" not in result.stderr


@pytest.mark.parametrize("platform_name,actual_platform", [("a2a3", "a2a3"), ("a5", "a5"), ("a5", "a2a3")])
def test_real_workflow_body_preserves_platform_through_the_exclusive_task(tmp_path, platform_name, actual_platform):
    workflow = _yaml(_WORKFLOW)
    steps = workflow["jobs"]["run"]["steps"]
    setup = next(step for step in steps if "uses" in step and step["uses"].endswith("setup-framework-eager"))
    assert setup["with"]["platform"] == "${{ inputs.platform }}"
    step = next(step for step in steps if "run" in step)
    assert step["env"]["PLATFORM"] == "${{ inputs.platform }}"
    log = tmp_path / "commands.jsonl"
    _executable(
        tmp_path / ".venv-framework/bin/python",
        f"#!{sys.executable}\nimport json,os,sys\n"
        "with open(os.environ['COMMAND_LOG'], 'a') as out: out.write(json.dumps(sys.argv[1:])+'\\n')\n",
    )
    binary = tmp_path / "bin"
    _executable(
        binary / "task-submit",
        f"#!{sys.executable}\nimport os,subprocess,sys\n"
        "args=sys.argv[1:]\n"
        "if args==['--list']: sys.exit(0)\n"
        "assert args[args.index('--device-num')+1]=='1'\n"
        "assert args[args.index('--timeout')+1]=='900'\n"
        "assert args[args.index('--max-time')+1]=='900'\n"
        "assert ['--env','PLATFORM']==args[4:6]\n"
        "subprocess.run(['bash','-e','-c',args[args.index('--run')+1]], "
        "env={**os.environ,'TASK_DEVICE':'0'},check=True)\n",
    )
    _executable(binary / "git", "#!/bin/sh\necho CPU-source\n")
    _executable(binary / "npu-smi", "#!/bin/sh\necho CPU-inventory\n")
    _executable(
        tmp_path / ".claude/skills/onboard-arch-precheck/check.sh", '#!/bin/sh\n[ "$1" = "$ACTUAL_PLATFORM" ]\n'
    )
    cann = tmp_path / "set_env.sh"
    cann.write_text(":\n")
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{binary}:{os.environ['PATH']}",
            "CANN_ENV": str(cann),
            "PLATFORM": platform_name,
            "MANUAL_MODE": "exclude",
            "ACTUAL_PLATFORM": actual_platform,
            "COMMAND_LOG": str(log),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    if platform_name != actual_platform:
        assert result.returncode != 0
        assert not log.exists(), "wrong-silicon execution must stop before backend/pytest"
        return
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    assert commands[0] == [".github/actions/setup-framework-eager/identity.py", "--platform", platform_name]
    pytest_command = commands[1]
    assert pytest_command[:4] == ["-m", "pytest", "examples", "tests/st"]
    assert pytest_command[pytest_command.index("--platform") + 1] == platform_name
    assert pytest_command[pytest_command.index("--max-parallel") + 1] == "1"
    assert pytest_command[pytest_command.index("--test-environment") + 1] == "torch_npu"
    assert commands[2][:3] == ["-m", "simpler_setup.test_environments", "verify"]


@pytest.mark.parametrize("caller", ["ci", "ci-self-cpu", "daily"])
def test_each_caller_routes_both_platforms_to_the_same_reusable_workflow(caller):
    jobs = _yaml(_ROOT / f".github/workflows/{caller}.yml")["jobs"]
    for platform_name in ("a2a3", "a5"):
        job = jobs[f"st-framework-eager-{platform_name}"]
        assert job["uses"] == "./.github/workflows/_st-framework-eager.yml"
        assert job["with"]["platform"] == platform_name
        assert json.loads(job["with"]["runs_on"]) == ["self-hosted", platform_name]
        if caller == "daily":
            assert job["with"]["manual_mode"] == "include"
        else:
            assert f"outputs.{platform_name}_changed == 'true'" in job["if"]
            assert "outputs.st_affected == 'true'" in job["if"]
            assert job["needs"] == ["detect-changes", "pre-commit"]
    workflow = _yaml(_WORKFLOW)
    assert workflow["jobs"]["run"]["timeout-minutes"] == "45"
    assert workflow["jobs"]["run"]["steps"][-1]["with"]["name"] == (
        "framework-eager-${{ inputs.platform }}-${{ github.run_id }}-${{ github.run_attempt }}"
    )
