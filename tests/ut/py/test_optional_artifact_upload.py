# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep optional artifact dependencies out of job initialization."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[3]
_ACTION = "./.ci-support/.github/actions/upload-diagnostics"


def _yaml(path):
    return yaml.load(path.read_text(), Loader=yaml.BaseLoader)


def test_optional_uploads_load_only_after_required_steps():
    for workflow in ("_st-npu-a2a3.yml", "_st-npu-a5.yml", "_st-network1.yml", "_st-framework.yml"):
        steps = next(iter(_yaml(_ROOT / ".github/workflows" / workflow)["jobs"].values()))["steps"]
        assert not any(step.get("uses", "").startswith("actions/upload-artifact@") for step in steps), workflow
        prepare = next(step for step in steps if step.get("id") == "diagnostics")
        required = [index for index, step in enumerate(steps) if step.get("continue-on-error") != "true"]
        assert steps.index(prepare) > max(required), workflow
        uploads = [step for step in steps if step.get("uses") == _ACTION]
        assert uploads and prepare["continue-on-error"] == "true", workflow
        for upload in uploads:
            assert upload["continue-on-error"] == "true", workflow
            assert "steps.diagnostics.outcome == 'success'" in upload["if"], workflow


@pytest.mark.parametrize(
    ("workflow", "action_name", "step_id"),
    [
        ("_st-npu-a5.yml", "upload-diagnostics", "diagnostics"),
        ("_pre-commit.yml", "setup-python", None),
    ],
)
def test_ci_helper_checkout_works_with_an_older_pr_head(tmp_path, workflow, action_name, step_id):
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "CI",
        "GIT_AUTHOR_EMAIL": "ci@example.invalid",
        "GIT_COMMITTER_NAME": "CI",
        "GIT_COMMITTER_EMAIL": "ci@example.invalid",
    }

    def git(cwd, *args):
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()

    remote = tmp_path / "repository"
    git(tmp_path, "init", "-b", "main", str(remote))
    (remote / "tested.txt").write_text("old PR payload\n")
    git(remote, "add", ".")
    git(remote, "commit", "-m", "old head")
    old_head = git(remote, "rev-parse", "HEAD")
    relative = Path(".github/actions") / action_name / "action.yml"
    manifest = remote / relative
    manifest.parent.mkdir(parents=True)
    expected = (_ROOT / relative).read_text()
    manifest.write_text(expected)
    git(remote, "add", ".")
    git(remote, "commit", "-m", "workflow helpers")
    workflow_sha = git(remote, "rev-parse", "HEAD")
    manifest.write_text("name: unrelated later revision\n")
    git(remote, "add", ".")
    git(remote, "commit", "-m", "main advances")

    workspace = tmp_path / "workspace"
    git(tmp_path, "clone", remote.as_uri(), str(workspace))
    git(workspace, "checkout", "--detach", old_head)
    assert not (workspace / relative).exists()
    doc = _yaml(_ROOT / ".github/workflows" / workflow)
    steps = next(iter(doc["jobs"].values()))["steps"]
    prepare = next(
        step
        for step in steps
        if (step_id and step.get("id") == step_id) or step.get("name") == "Checkout Python setup action"
    )
    assert prepare["with"]["ref"] == "${{ github.workflow_sha }}"
    support = workspace / prepare["with"]["path"]
    support.mkdir()
    git(support, "init")
    git(support, "remote", "add", "origin", remote.as_uri())
    git(support, "sparse-checkout", "init", "--cone")
    git(support, "sparse-checkout", "set", prepare["with"]["sparse-checkout"])
    git(support, "fetch", "--no-tags", "--depth", prepare["with"]["fetch-depth"], "origin", workflow_sha)
    git(support, "checkout", "--detach", "FETCH_HEAD")
    assert (support / relative).read_text() == expected
    assert git(workspace, "rev-parse", "HEAD") == old_head
    assert (workspace / "tested.txt").read_text() == "old PR payload\n"
    assert not (workspace / relative).exists()
