# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise CI base resolution against local forks and shallow Git histories."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[3]


def _git(repo, *args, env=None):
    return subprocess.run(["git", *args], cwd=repo, env=env, text=True, capture_output=True, check=True).stdout.strip()


@pytest.fixture
def fork_history(tmp_path):
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "CI",
        "GIT_AUTHOR_EMAIL": "ci@example.invalid",
        "GIT_COMMITTER_NAME": "CI",
        "GIT_COMMITTER_EMAIL": "ci@example.invalid",
    }
    source = tmp_path / "source"
    _git(tmp_path, "init", "-b", "main", str(source), env=env)
    (source / "README.md").write_text("base\n")
    _git(source, "add", ".", env=env)
    _git(source, "commit", "-m", "base", env=env)
    common = _git(source, "rev-parse", "HEAD", env=env)
    _git(source, "branch", "feature", env=env)
    (source / "main_only.cpp").write_text("int main_only;\n")
    _git(source, "add", ".", env=env)
    _git(source, "commit", "-m", "main advances", env=env)
    base = _git(source, "rev-parse", "HEAD", env=env)
    canonical = tmp_path / "canonical.git"
    _git(tmp_path, "clone", "--bare", str(source), str(canonical), env=env)
    _git(source, "checkout", "feature", env=env)
    (source / "change.md").write_text("docs\n")
    _git(source, "add", ".", env=env)
    _git(source, "commit", "-m", "docs", env=env)
    fork = tmp_path / "fork.git"
    _git(tmp_path, "clone", "--bare", str(source), str(fork), env=env)
    _git(
        tmp_path,
        "config",
        "--global",
        f"url.{canonical.as_uri()}.insteadOf",
        "https://github.com/test/base.git",
        env=env,
    )
    return source, fork, common, base, env


def _checkout(tmp_path, history, commits):
    source, fork, common, base, env = history
    head = _git(source, "rev-parse", "HEAD", env=env)
    tree = _git(source, "rev-parse", "HEAD^{tree}", env=env)
    for index in range(commits - 1):
        head = _git(source, "commit-tree", tree, "-p", head, "-m", f"feature {index}", env=env)
    _git(source, "update-ref", "refs/heads/feature", head, env=env)
    _git(source, "push", str(fork), "feature", env=env)
    checkout = tmp_path / "checkout"
    _git(
        tmp_path,
        "clone",
        "--depth",
        "100",
        "--single-branch",
        "--branch",
        "feature",
        fork.as_uri(),
        str(checkout),
        env=env,
    )
    return checkout, head


def _run_step(tmp_path, history, checkout, head, workflow, base):
    env = history[-1]
    doc = yaml.load((_ROOT / ".github/workflows" / workflow).read_text(), Loader=yaml.BaseLoader)
    steps = next(iter(doc["jobs"].values()))["steps"]
    step_id = "base" if workflow == "_pre-commit.yml" else "check"
    script = next(step["run"] for step in steps if step.get("id") == step_id)
    output = tmp_path / "output"
    trace = tmp_path / "trace.jsonl"
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        cwd=checkout,
        env={
            **env,
            "BASE_REPOSITORY": "test/base",
            "BASE_REF": "main",
            "BASE_SHA": base,
            "HEAD_SHA": head,
            "GITHUB_OUTPUT": str(output),
            "GIT_TRACE2_EVENT": str(trace),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines()) if output.exists() else {}
    commands = [json.loads(line).get("argv", []) for line in trace.read_text().splitlines()]
    fetches = [args for args in commands if len(args) > 1 and args[1] == "fetch"]
    return result, outputs, fetches


@pytest.mark.parametrize("workflow", ["_pre-commit.yml", "_detect-changes.yml"])
def test_local_base_needs_no_network(tmp_path, fork_history, workflow):
    checkout, head = _checkout(tmp_path, fork_history, 1)
    common = fork_history[2]
    result, outputs, fetches = _run_step(tmp_path, fork_history, checkout, head, workflow, common)
    assert result.returncode == 0, result.stderr
    assert fetches == []
    if workflow == "_pre-commit.yml":
        assert outputs == {"base_sha": common}
    else:
        assert outputs["non_code_only"] == "true"
        assert outputs["a2a3_changed"] == outputs["a5_changed"] == "false"


@pytest.mark.parametrize("workflow", ["_pre-commit.yml", "_detect-changes.yml"])
def test_fork_with_diverged_main_preserves_the_pr_diff(tmp_path, fork_history, workflow):
    checkout, head = _checkout(tmp_path, fork_history, 320)
    result, outputs, fetches = _run_step(tmp_path, fork_history, checkout, head, workflow, fork_history[3])
    assert result.returncode == 0, result.stderr
    if workflow == "_pre-commit.yml":
        assert outputs == {"base_sha": fork_history[2]}
    else:
        assert outputs["non_code_only"] == "true"
        assert outputs["a2a3_changed"] == outputs["a5_changed"] == "false"
    assert fetches
    assert all("--no-tags" in args for args in fetches)
    assert all("--unshallow" not in args and not any("refs/heads/*" in arg for arg in args) for args in fetches)


def test_omitted_head_sha_deepens_the_checked_out_commit(tmp_path, fork_history):
    checkout, head = _checkout(tmp_path, fork_history, 120)
    _git(fork_history[1], "symbolic-ref", "HEAD", "refs/heads/main", env=fork_history[-1])
    result, outputs, fetches = _run_step(tmp_path, fork_history, checkout, "", "_detect-changes.yml", fork_history[3])
    assert result.returncode == 0, result.stderr
    assert outputs["non_code_only"] == "true"
    origin_fetches = [args for args in fetches if "origin" in args]
    assert origin_fetches and all(args[-1] == head for args in origin_fetches)


@pytest.mark.parametrize("workflow", ["_pre-commit.yml", "_detect-changes.yml"])
def test_missing_remote_does_not_silently_skip_checks(tmp_path, fork_history, workflow):
    checkout, head = _checkout(tmp_path, fork_history, 120)
    _git(checkout, "remote", "set-url", "origin", str(tmp_path / "missing.git"), env=fork_history[-1])
    result, outputs, _ = _run_step(tmp_path, fork_history, checkout, head, workflow, fork_history[3])
    if workflow == "_pre-commit.yml":
        assert result.returncode != 0
        assert "base_sha" not in outputs
    else:
        assert result.returncode == 0, result.stderr
        assert outputs == {
            "non_code_only": "false",
            "a2a3_changed": "true",
            "a5_changed": "true",
            "wse_changed": "true",
            "st_affected": "true",
            "ut_affected": "true",
            "examples_only": "false",
            "tests_only": "false",
        }
