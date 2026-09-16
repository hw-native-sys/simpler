# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).parents[3]
ACTION_PATH = REPO_ROOT / ".github/actions/setup-gcc-15/action.yml"
APT_ACTION_PATH = REPO_ROOT / ".github/actions/apt-install/action.yml"
PPA_KEY_PATH = REPO_ROOT / ".github/actions/setup-gcc-15/ubuntu-toolchain-r-test.asc"
SETUP_ACTION = "./.github/actions/setup-gcc-15"
WORKFLOW_PATHS = (
    REPO_ROOT / ".github/workflows/_pre-commit.yml",
    REPO_ROOT / ".github/workflows/_profiling-flags-smoke.yml",
    REPO_ROOT / ".github/workflows/_st-npu-a2a3.yml",
    REPO_ROOT / ".github/workflows/_st-sim-a2a3.yml",
    REPO_ROOT / ".github/workflows/_st-sim-a5.yml",
    REPO_ROOT / ".github/workflows/ci-self-cpu.yml",
    REPO_ROOT / ".github/workflows/ci.yml",
    REPO_ROOT / ".github/workflows/daily.yml",
    REPO_ROOT / ".github/workflows/sanitizers.yml",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text())
    assert isinstance(document, dict)
    return document


def _setup_step(path: Path) -> dict[str, Any]:
    workflow = _load_yaml(path)
    matches = [step for job in workflow["jobs"].values() for step in job["steps"] if step.get("uses") == SETUP_ACTION]
    assert len(matches) == 1
    return matches[0]


def _named_step(path: Path, name: str) -> dict[str, Any]:
    workflow = _load_yaml(path)
    matches = [step for job in workflow["jobs"].values() for step in job["steps"] if step.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def _action_script(condition: str) -> str:
    action = _load_yaml(ACTION_PATH)
    return next(step["run"] for step in action["runs"]["steps"] if step.get("if") == condition)


def _apt_action_script() -> str:
    action = _load_yaml(APT_ACTION_PATH)
    return action["runs"]["steps"][0]["run"]


def _write_command(path: Path, body: str = "exit 0") -> None:
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)


def _run_ppa_key_validation(tmp_path: Path, key_metadata: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    script = _action_script("runner.os == 'Linux'")
    validation_start = script.index('key_metadata="$(')
    validation_end = script.index("sudo install -d", validation_start)
    validation_script = "set -euo pipefail\n" + script[validation_start:validation_end]

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    metadata_file = tmp_path / "key-metadata.txt"
    metadata_file.write_text(key_metadata)
    dearmor_marker = tmp_path / "dearmor-called"
    _write_command(
        bin_dir / "gpg",
        """case " $* " in
  *" --show-keys "*) cat "$GPG_METADATA_FILE" ;;
  *" --dearmor "*) : > "$GPG_DEARMOR_MARKER" ;;
  *) exit 2 ;;
esac""",
    )

    linux = next(step for step in _load_yaml(ACTION_PATH)["runs"]["steps"] if step.get("if") == "runner.os == 'Linux'")
    env = os.environ.copy()
    env.update(
        {
            "EXPECTED_PPA_FINGERPRINT": linux["env"]["EXPECTED_PPA_FINGERPRINT"],
            "GPG_DEARMOR_MARKER": str(dearmor_marker),
            "GPG_METADATA_FILE": str(metadata_file),
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "key_file": str(tmp_path / "key.asc"),
            "keyring_file": str(tmp_path / "key.gpg"),
        }
    )
    result = subprocess.run(
        ["bash", "-c", validation_script],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, dearmor_marker


def test_action_has_linux_and_macos_branches() -> None:
    action = _load_yaml(ACTION_PATH)
    assert "Linux or macOS" in action["description"]
    assert "Ubuntu or macOS" in action["description"]

    conditions = {step.get("if") for step in action["runs"]["steps"]}
    assert "runner.os == 'Linux'" in conditions
    assert "runner.os == 'macOS'" in conditions


def test_linux_branch_installs_only_missing_tools_on_ubuntu_and_pins_the_ppa_key() -> None:
    action = _load_yaml(ACTION_PATH)
    linux = next(step for step in action["runs"]["steps"] if step.get("if") == "runner.os == 'Linux'")
    script = linux["run"]

    install_guard = script.index("if (( ${#packages[@]} )); then")
    assert install_guard < script.index("source /etc/os-release")
    assert script.index("source /etc/os-release") < script.index('[[ "${ID:-}" != ubuntu ]]')
    assert "automatically install missing Linux tools only on Ubuntu" in script
    assert "command -v gcc" in script
    assert "command -v g++" in script
    fingerprint = linux["env"]["EXPECTED_PPA_FINGERPRINT"]
    assert re.fullmatch(r"[0-9A-F]{40}", fingerprint)
    assert "keyserver.ubuntu.com" not in script
    assert "${{ github.action_path }}/ubuntu-toolchain-r-test.asc" in script
    assert ': "${VERSION_CODENAME:?Ubuntu /etc/os-release must define VERSION_CODENAME}"' in script
    assert "Signed-By: /etc/apt/keyrings/ubuntu-toolchain-r-test.gpg $EXPECTED_PPA_FINGERPRINT" in script


def test_apt_install_preserves_system_indexes_for_ppa_only_refresh_and_recovers_interrupted_installs() -> None:
    action = _load_yaml(APT_ACTION_PATH)
    script = action["runs"]["steps"][0]["run"]

    assert action["inputs"]["update-mode"]["default"] == "none"
    assert "Dir::Etc::SourceList=/dev/null" in script
    assert "Dir::Etc::SourceParts=$sourceparts_dir" in script
    assert "APT::Get::List-Cleanup=0" in script
    assert "timeout --kill-after=15s" in script
    assert "dpkg --configure -a" in script
    assert "Acquire::Retries=0" in script
    assert "sudo DEBIAN_FRONTEND=noninteractive timeout" in script
    assert "apt install failed on cached indexes; refreshing system indexes once" in script
    assert "run_update system 1" in script
    assert "${{ inputs." not in script


def test_apt_install_refreshes_the_system_index_after_a_cached_install_failure(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "apt.log"
    install_marker = tmp_path / "install-attempted"
    _write_command(
        bin_dir / "sudo",
        '''case "$1" in
  DEBIAN_FRONTEND=*) export "$1"; shift ;;
esac
case "$1" in
  rm|dpkg) exit 0 ;;
esac
exec "$@"''',
    )
    _write_command(bin_dir / "timeout", 'shift\nshift\nexec "$@"')
    _write_command(
        bin_dir / "apt-get",
        """echo "$*" >> "$APT_LOG"
if [ ! -e "$APT_INSTALL_MARKER" ]; then
  : > "$APT_INSTALL_MARKER"
  exit 1
fi""",
    )

    env = os.environ.copy()
    env.update(
        {
            "APT_INSTALL_MARKER": str(install_marker),
            "APT_LOG": str(log_path),
            "INSTALL_ATTEMPTS": "1",
            "INSTALL_TIMEOUT_SECONDS": "240",
            "OPTIONAL_PACKAGES": "",
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "REQUIRED_PACKAGES": "libgtest-dev",
            "RUNNER_TEMP": str(tmp_path),
            "SOURCE_FILE": "",
            "UPDATE_ATTEMPTS": "1",
            "UPDATE_MODE": "none",
            "UPDATE_TIMEOUT_SECONDS": "120",
        }
    )
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", _apt_action_script()],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    commands = log_path.read_text().splitlines()
    assert "install" in commands[0]
    assert "update" in commands[1]
    assert "install" in commands[2]


def test_apt_install_retries_a_required_install_before_refreshing_indexes(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "apt.log"
    install_marker = tmp_path / "install-attempted"
    _write_command(
        bin_dir / "sudo",
        '''case "$1" in
  DEBIAN_FRONTEND=*) export "$1"; shift ;;
esac
case "$1" in
  rm|dpkg) exit 0 ;;
esac
exec "$@"''',
    )
    _write_command(bin_dir / "timeout", 'shift\nshift\nexec "$@"')
    _write_command(
        bin_dir / "apt-get",
        """echo "$*" >> "$APT_LOG"
if [ ! -e "$APT_INSTALL_MARKER" ]; then
  : > "$APT_INSTALL_MARKER"
  exit 1
fi""",
    )

    env = os.environ.copy()
    env.update(
        {
            "APT_INSTALL_MARKER": str(install_marker),
            "APT_LOG": str(log_path),
            "INSTALL_ATTEMPTS": "2",
            "INSTALL_TIMEOUT_SECONDS": "240",
            "OPTIONAL_PACKAGES": "",
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "REQUIRED_PACKAGES": "g++-15",
            "RUNNER_TEMP": str(tmp_path),
            "SOURCE_FILE": "",
            "UPDATE_ATTEMPTS": "1",
            "UPDATE_MODE": "none",
            "UPDATE_TIMEOUT_SECONDS": "120",
        }
    )
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", _apt_action_script()],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    commands = log_path.read_text().splitlines()
    assert len(commands) == 2
    assert all("install" in command for command in commands)


def test_apt_install_rejects_invalid_input_before_running_apt(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    apt_called = tmp_path / "apt-called"
    _write_command(bin_dir / "apt-get", f': > "{apt_called}"')

    env = os.environ.copy()
    env.update(
        {
            "INSTALL_ATTEMPTS": "1",
            "INSTALL_TIMEOUT_SECONDS": "240",
            "OPTIONAL_PACKAGES": "",
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "REQUIRED_PACKAGES": "libgtest-dev",
            "RUNNER_TEMP": str(tmp_path),
            "SOURCE_FILE": "",
            "UPDATE_ATTEMPTS": "zero",
            "UPDATE_MODE": "none",
            "UPDATE_TIMEOUT_SECONDS": "120",
        }
    )
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", _apt_action_script()],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "must be positive integers" in result.stdout + result.stderr
    assert not apt_called.exists()


def test_apt_install_source_file_update_uses_only_the_requested_source(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "apt.log"
    source_file = tmp_path / "toolchain.sources"
    source_file.write_text("Types: deb\nURIs: https://example.invalid/apt\nSuites: noble\nComponents: main\n")
    _write_command(
        bin_dir / "sudo",
        '''case "$1" in
  DEBIAN_FRONTEND=*) export "$1"; shift ;;
esac
case "$1" in
  rm|dpkg) exit 0 ;;
esac
exec "$@"''',
    )
    _write_command(bin_dir / "timeout", 'shift\nshift\nexec "$@"')
    _write_command(bin_dir / "apt-get", 'echo "$*" >> "$APT_LOG"')

    env = os.environ.copy()
    env.update(
        {
            "APT_LOG": str(log_path),
            "INSTALL_ATTEMPTS": "1",
            "INSTALL_TIMEOUT_SECONDS": "240",
            "OPTIONAL_PACKAGES": "",
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "REQUIRED_PACKAGES": "",
            "RUNNER_TEMP": str(tmp_path),
            "SOURCE_FILE": str(source_file),
            "UPDATE_ATTEMPTS": "1",
            "UPDATE_MODE": "source-file",
            "UPDATE_TIMEOUT_SECONDS": "60",
        }
    )
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", _apt_action_script()],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    command = log_path.read_text()
    assert "Dir::Etc::SourceList=/dev/null" in command
    assert "Dir::Etc::SourceParts=" in command
    assert "APT::Get::List-Cleanup=0" in command
    assert command.rstrip().endswith("update")


@pytest.mark.parametrize(
    "workflow_name, step_name",
    [
        ("_packaging.yml", "Install Linux C++ compiler tools"),
        ("_pre-commit.yml", "Install build and lint tools"),
        ("_ut-no-hardware.yml", "Install GoogleTest"),
    ],
)
def test_workflow_apt_sites_use_the_shared_action(workflow_name: str, step_name: str) -> None:
    step = _named_step(REPO_ROOT / ".github/workflows" / workflow_name, step_name)
    assert step["uses"] == "./.github/actions/apt-install"


def test_vendored_ppa_key_has_exactly_one_expected_primary_key() -> None:
    linux = next(step for step in _load_yaml(ACTION_PATH)["runs"]["steps"] if step.get("if") == "runner.os == 'Linux'")
    result = subprocess.run(
        ["gpg", "--batch", "--show-keys", "--with-colons", str(PPA_KEY_PATH)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    primary_keys = [line for line in result.stdout.splitlines() if line.startswith("pub:")]
    fingerprints = [line.split(":")[9] for line in result.stdout.splitlines() if line.startswith("fpr:")]
    assert len(primary_keys) == 1
    assert fingerprints[0] == linux["env"]["EXPECTED_PPA_FINGERPRINT"]


def test_install_build_essential_input_documents_its_linux_only_contract() -> None:
    description = _load_yaml(ACTION_PATH)["inputs"]["install-build-essential"]["description"]
    assert "Linux only" in description
    assert "_ensure_host_compilers" in description
    assert "Simulator builds still use gcc-15/g++-15" in description
    assert "Ignored on macOS" in description


def test_ppa_key_validation_accepts_expected_primary_key_with_subkey(tmp_path: Path) -> None:
    fingerprint = _load_yaml(ACTION_PATH)["runs"]["steps"][0]["env"]["EXPECTED_PPA_FINGERPRINT"]
    metadata = f"pub:::::::::\nfpr:::::::::{fingerprint}:\nsub:::::::::\nfpr:::::::::{'1' * 40}:\n"

    result, dearmor_marker = _run_ppa_key_validation(tmp_path, metadata)

    assert result.returncode == 0, result.stdout + result.stderr
    assert dearmor_marker.exists()


def test_ppa_key_validation_rejects_unexpected_primary_key(tmp_path: Path) -> None:
    metadata = f"pub:::::::::\nfpr:::::::::{'1' * 40}:\n"

    result, dearmor_marker = _run_ppa_key_validation(tmp_path, metadata)

    assert result.returncode != 0
    assert "unexpected PPA signing key" in result.stdout + result.stderr
    assert not dearmor_marker.exists()


def test_ppa_key_validation_rejects_an_appended_primary_key(tmp_path: Path) -> None:
    fingerprint = _load_yaml(ACTION_PATH)["runs"]["steps"][0]["env"]["EXPECTED_PPA_FINGERPRINT"]
    metadata = f"pub:::::::::\nfpr:::::::::{fingerprint}:\npub:::::::::\nfpr:::::::::{'1' * 40}:\n"

    result, dearmor_marker = _run_ppa_key_validation(tmp_path, metadata)

    assert result.returncode != 0
    assert "expected exactly one PPA primary key, found 2" in result.stdout + result.stderr
    assert not dearmor_marker.exists()


def test_linux_branch_accepts_a_preinstalled_toolchain_without_ubuntu(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    compiler_body = 'if [ "${1:-}" = -dumpversion ]; then echo 15.2.0; fi'
    for command in ("gcc-15", "g++-15"):
        _write_command(bin_dir / command, compiler_body)
    for command in ("gcc", "g++", "ninja", "dot"):
        _write_command(bin_dir / command)

    script = _action_script("runner.os == 'Linux'")
    script = script.replace("${{ inputs.install-graphviz }}", "true")
    script = script.replace("${{ inputs.install-build-essential }}", "true")
    env = os.environ.copy()
    env.update({"PATH": f"{bin_dir}:/usr/bin:/bin", "RUNNER_TEMP": str(tmp_path)})
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Ubuntu" not in result.stdout + result.stderr


def test_each_install_branch_verifies_gcc_15_and_build_tools() -> None:
    action = _load_yaml(ACTION_PATH)
    branches = {
        step["if"]: step["run"]
        for step in action["runs"]["steps"]
        if step.get("if") in {"runner.os == 'Linux'", "runner.os == 'macOS'"}
    }

    for script in branches.values():
        assert script.rindex("command -v gcc-15") < script.rindex("is_version_15 gcc-15")
        assert script.rindex("command -v g++-15") < script.rindex("is_version_15 g++-15")
        assert "command -v ninja" in script
        assert "command -v dot" in script


def test_sanitizer_requests_graphviz_and_build_essential() -> None:
    step = _setup_step(REPO_ROOT / ".github/workflows/sanitizers.yml")
    assert step["with"] == {
        "install-graphviz": "true",
        "install-build-essential": "true",
    }


def test_a2a3_scene_workflow_has_no_legacy_sdma_mode() -> None:
    workflow_path = REPO_ROOT / ".github/workflows/_st-npu-a2a3.yml"
    workflow = _load_yaml(workflow_path)
    steps = workflow["jobs"]["run"]["steps"]
    names = [step["name"] for step in steps]

    assert "a2a3_sdma_mode" not in workflow_path.read_text()
    assert names.count("Run pytest scene tests (a2a3)") == 1
    assert names.count("SDMA pytest (a2a3)") == 1
    assert not any("legacy SDMA paths" in name for name in names)
    for caller_name in ("ci.yml", "ci-self-cpu.yml", "daily.yml"):
        assert "a2a3_sdma_mode" not in (REPO_ROOT / ".github/workflows" / caller_name).read_text()


@pytest.mark.parametrize("workflow_name", ["_st-sim-a2a3.yml", "_st-sim-a5.yml"])
def test_scene_tests_request_graphviz(workflow_name: str) -> None:
    step = _setup_step(REPO_ROOT / ".github/workflows" / workflow_name)
    assert step["with"]["install-graphviz"] == "true"


@pytest.mark.parametrize("workflow_name", ["_st-sim-a2a3.yml", "_st-sim-a5.yml"])
@pytest.mark.parametrize("has_gcc15", [False, True])
def test_self_cpu_scene_tests_preserve_the_preprovisioned_compiler_contract(
    tmp_path: Path, workflow_name: str, has_gcc15: bool
) -> None:
    bin_dir = tmp_path / "preinstalled-bin"
    bin_dir.mkdir()
    for command in ("bash", "mkdir", "ln"):
        command_path = shutil.which(command)
        assert command_path is not None
        (bin_dir / command).symlink_to(command_path)
    for command in ("ninja", "dot"):
        _write_command(bin_dir / command)
    _write_command(bin_dir / "g++", 'if [ "${1:-}" = -dumpversion ]; then echo 12.3.0; fi')
    if has_gcc15:
        _write_command(bin_dir / "g++-15", 'if [ "${1:-}" = -dumpversion ]; then echo 15.1.0; fi')

    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    github_path = tmp_path / "github-path"
    script = _named_step(REPO_ROOT / ".github/workflows" / workflow_name, "Verify provisioned toolchain")["run"]
    env = os.environ.copy()
    env.update(
        {
            "GITHUB_PATH": str(github_path),
            "PATH": str(bin_dir),
            "RUNNER_TEMP": str(runner_temp),
        }
    )

    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    if has_gcc15:
        assert not (runner_temp / "bin/g++-15").exists()
        assert not github_path.exists()
    else:
        assert (runner_temp / "bin/g++-15").resolve() == bin_dir / "g++"
        assert github_path.read_text().splitlines() == [str(runner_temp / "bin")]


@pytest.mark.parametrize("path", (ACTION_PATH, APT_ACTION_PATH, *WORKFLOW_PATHS), ids=lambda path: path.name)
def test_changed_action_and_workflows_are_valid_yaml(path: Path) -> None:
    _load_yaml(path)
