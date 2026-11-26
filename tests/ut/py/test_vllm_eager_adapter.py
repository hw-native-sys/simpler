# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU contracts for the shared vLLM adapter and NNAL/ATB helpers."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from simpler_setup import vllm_eager


def _write(path, text="runner\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_adapter_runner_resolves_a_checkout_or_the_case_directory(tmp_path, monkeypatch):
    runner = _write(tmp_path / vllm_eager.ADAPTER_CASE / vllm_eager.RUNNER)
    monkeypatch.delenv("VLLM_ADAPTER_ROOT", raising=False)
    assert vllm_eager.adapter_runner(tmp_path) == runner
    assert vllm_eager.adapter_runner(tmp_path / vllm_eager.ADAPTER_CASE) == runner
    with pytest.raises(ValueError, match="VLLM_ADAPTER_ROOT"):
        vllm_eager.adapter_runner(tmp_path / "missing")


def test_adapter_root_environment_wins_over_the_given_checkout(tmp_path, monkeypatch):
    configured = _write(tmp_path / "configured" / vllm_eager.RUNNER)
    _write(tmp_path / "repo" / vllm_eager.ADAPTER_CASE / vllm_eager.RUNNER)
    monkeypatch.setenv("VLLM_ADAPTER_ROOT", str(tmp_path / "configured"))
    assert vllm_eager.adapter_runner(tmp_path / "repo") == configured


def test_adapter_identity_records_the_committed_revision(tmp_path, monkeypatch):
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(name, raising=False)
    repository = tmp_path / "adapter"
    runner = _write(repository / vllm_eager.RUNNER)
    git = ["git", "-C", str(repository), "-c", "user.email=ci@example.invalid", "-c", "user.name=ci"]
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    subprocess.run([*git, "add", str(runner)], check=True)
    subprocess.run([*git, "commit", "--quiet", "-m", "adapter"], check=True)
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()

    record = vllm_eager.adapter_identity(runner)
    digest = hashlib.sha256(runner.read_bytes()).hexdigest()
    assert record["runner"] == {"path": str(runner.resolve()), "sha256": digest}
    assert record["head"] == head
    assert record["modified"] is False

    runner.write_text("changed\n")
    assert vllm_eager.adapter_identity(runner)["modified"] is True


def test_adapter_identity_without_a_checkout_records_only_the_file(tmp_path):
    runner = _write(tmp_path / "plain" / vllm_eager.RUNNER)
    assert set(vllm_eager.adapter_identity(runner)) == {"runner"}


def test_atb_home_walks_up_to_the_installation_that_owns_version_info(tmp_path):
    install = tmp_path / "atb" / "9.0.0"
    _write(install / "version.info", "Ascend-cann-atb : 9.0.0\n")
    assert vllm_eager.atb_home([install / "aarch64-linux"]) == install


def test_atb_home_honours_the_environment_pointer(tmp_path, monkeypatch):
    install = tmp_path / "nnal" / "atb" / "9.0.0"
    _write(install / "version.info", "Ascend-cann-atb : 9.0.0\n")
    monkeypatch.setenv("ATB_ENV", str(install / "set_env.sh"))
    monkeypatch.delenv("ATB_HOME_PATH", raising=False)
    monkeypatch.delenv("ATB_HOME", raising=False)
    assert vllm_eager.atb_home() == install


def test_ci_install_exports_wheel_paths_to_the_device_task(tmp_path):
    root = Path(__file__).resolve().parents[3]
    action = yaml.load((root / ".github/actions/setup-vllm-eager/action.yml").read_text(), Loader=yaml.BaseLoader)
    install = next(step for step in action["runs"]["steps"] if step["name"].startswith("Install Simpler"))
    binary = tmp_path / ".venv-vllm/bin"
    binary.mkdir(parents=True)
    (binary / "activate").write_text(":\n")
    python = binary / "python"
    python.write_text(
        f'#!{sys.executable}\nimport sys\nif sys.argv[1] == "-c": print({str(tmp_path / "wheel-site")!r})\n'
    )
    python.chmod(0o755)
    toolchain = _write(tmp_path / "set_env.sh", ":\n")
    output = tmp_path / "github-env"
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", install["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{binary}:{os.environ['PATH']}",
            "CANN_ENV": str(toolchain),
            "ATB_ENV": str(toolchain),
            "GITHUB_ENV": str(output),
            "PYTHONPATH": "/existing-python-path",
            "PYTEST_ADDOPTS": "-q",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_text().splitlines() == [
        f"PYTHONPATH={tmp_path}/wheel-site:/existing-python-path",
        f"PYTEST_ADDOPTS=-q -o pythonpath={tmp_path}/wheel-site",
    ]
    workflow = yaml.load((root / ".github/workflows/_st-vllm-eager-a2a3.yml").read_text(), Loader=yaml.BaseLoader)
    run = next(step["run"] for step in workflow["jobs"]["run"]["steps"] if step["name"].startswith("Run the declared"))
    assert "--env PYTHONPATH" in run
    assert "--env PYTEST_ADDOPTS" in run
