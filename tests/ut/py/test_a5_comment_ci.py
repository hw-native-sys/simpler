# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 adapters for the PyPTO command/status/progress workflow protocol."""

import json
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[3]


def _workflow(name="ci-a5-comment"):
    return yaml.load((_ROOT / f".github/workflows/{name}.yml").read_text(), Loader=yaml.BaseLoader)


def test_a5_ut_and_framework_run_automatically_only_for_affected_changes():
    ci = _workflow("ci")["jobs"]
    job = ci["ut-a5"]
    assert job["needs"] == ["detect-changes", "pre-commit"]
    assert job["if"] == (
        "needs.detect-changes.outputs.ut_hardware_a5 == 'true' || needs.detect-changes.outputs.ut_sdk_a5 == 'true'"
    )
    assert job["uses"] == "./.github/workflows/_ut-npu-a5.yml"
    assert json.loads(job["with"]["runs_on"]) == ["self-hosted", "a5"]
    assert job["with"]["include_cann_examples"] == "true"
    assert "st-onboard-a5" not in ci
    framework = ci["st-framework-a5"]
    assert framework["needs"] == ["detect-changes", "pre-commit"]
    assert framework["if"] == (
        "needs.detect-changes.outputs.a5_changed == 'true' && needs.detect-changes.outputs.st_affected == 'true'"
    )
    assert framework["uses"] == "./.github/workflows/_st-framework.yml"
    assert framework["with"]["platform"] == "a5"
    for name in ("ut", "packaging-matrix", "st-sim-a5", "profiling-flags-smoke", "ut-a2a3", "st-onboard-a2a3"):
        assert name in ci
    daily = _workflow("daily")["jobs"]
    assert "ut-a5" not in daily
    assert "st-onboard-a5" in daily and "st-framework-a5" in daily


def test_a5_test_scope_and_pinned_checkout_are_preserved():
    workflow = _workflow()
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["on"]) == {"workflow_dispatch"}
    jobs = workflow["jobs"]
    assert set(jobs) == {"prepare", "ut-a5", "st-onboard-a5", "a5-result"}
    steps = jobs["prepare"]["steps"]
    auth = next(i for i, step in enumerate(steps) if step.get("name") == "Authorize A5 dispatch")
    assert steps[auth]["run"].endswith("ci_control.py authorize-dispatch")
    assert steps[auth + 1]["run"].endswith("ci_control.py prepare")
    assert "if" not in steps[auth + 1] and "continue-on-error" not in steps[auth]
    for name, reusable in {"ut-a5": "_ut-npu-a5.yml", "st-onboard-a5": "_st-npu-a5.yml"}.items():
        suite = jobs[name]
        assert suite["needs"] == "prepare"
        assert suite["uses"] == f"./.github/workflows/{reusable}"
        assert suite["with"]["ref"] == "${{ needs.prepare.outputs.checkout_sha }}"
        assert suite["with"]["repository"] == "${{ github.repository }}"
        assert json.loads(suite["with"]["runs_on"]) == ["self-hosted", "a5"]
        assert suite["concurrency"]["cancel-in-progress"] == "false"
        assert suite["concurrency"]["queue"] == "max"
    assert jobs["ut-a5"]["with"]["include_cann_examples"] == "true"
    assert jobs["st-onboard-a5"]["with"]["include_dfx_smokes"] == "true"
    assert jobs["a5-result"]["needs"] == ["prepare", "ut-a5", "st-onboard-a5"]
    assert jobs["a5-result"]["if"] == "always()"
    scripts = "\n".join(step.get("run", "") for step in jobs["a5-result"]["steps"])
    assert "ci_control.py aggregate" in scripts
    assert "ci-command-status.yml" in scripts and "wait_for_completion=true" in scripts
    assert "ci-command-progress.yml" in scripts


def test_command_and_aggregate_publishers_share_the_short_writer_queue():
    command = _workflow("ci-command")
    status = _workflow("ci-command-status")
    assert command["on"] == {"issue_comment": {"types": ["created"]}}
    assert set(command["jobs"]) == {"route", "dispatch", "feedback"}
    assert command["jobs"]["dispatch"]["name"] == (
        "Dispatch ${{ needs.route.outputs.command }} PR #${{ github.event.issue.number }} "
        "@${{ needs.route.outputs.head }}"
    )
    assert "contains(github.event.comment.body, '/run-a5')" in command["jobs"]["route"]["if"]
    assert "contains(github.event.comment.body, '/run a5')" in command["jobs"]["route"]["if"]
    dispatch_env = next(step["env"] for step in command["jobs"]["dispatch"]["steps"] if step.get("id") == "dispatch")
    for field in ("head", "base", "merge"):
        assert command["jobs"]["route"]["outputs"][field] == "${{ steps.route.outputs." + field + " }}"
        assert dispatch_env[f"CI_ROUTED_{field.upper()}"] == "${{ needs.route.outputs." + field + " }}"
    lock = command["jobs"]["dispatch"]["concurrency"]
    assert lock == status["jobs"]["publish"]["concurrency"]
    assert lock["queue"] == "max" and lock["cancel-in-progress"] == "false"
    assert command["jobs"]["feedback"]["permissions"] == {"contents": "read", "pull-requests": "write"}
    assert status["jobs"]["publish"]["permissions"]["statuses"] == "write"
    progress_lock = _workflow("ci-command-progress")["jobs"]["progress"]["concurrency"]
    assert progress_lock["group"] != lock["group"]


def test_control_jobs_execute_only_trusted_main_files():
    for name in ("ci-command", "ci-a5-comment", "ci-a3-network1", "ci-command-status", "ci-command-progress"):
        for job in _workflow(name)["jobs"].values():
            if "uses" in job:
                continue
            for step in job["steps"]:
                if step.get("uses", "").startswith("actions/checkout@"):
                    assert step["with"]["ref"] == "${{ github.workflow_sha }}"
                    assert step["with"]["persist-credentials"] == "false"
                    assert ".github/ci-commands.json" in step["with"]["sparse-checkout"]
    assert not (_ROOT / ".github/scripts/pr_commands.cjs").exists()
    assert not (_ROOT / ".github/workflows/ci-command-feedback.yml").exists()
    assert not (_ROOT / ".github/workflows/ci-pod-comment.yml").exists()


def test_registered_suites_match_the_real_reusable_job_names():
    config = json.loads((_ROOT / ".github/ci-commands.json").read_text())
    assert set(config) == {"a5", "a3-network1"}
    for label, api_name in config["a5"]["jobs"].items():
        reusable = "_ut-npu-a5" if label == "ut-a5" else "_st-npu-a5"
        inner_jobs = _workflow(reusable)["jobs"]
        assert api_name == f"{label} / {inner_jobs['run']['name']}"


def test_a3_network1_reuses_daily_with_the_admitted_merge_and_independent_feedback():
    config = json.loads((_ROOT / ".github/ci-commands.json").read_text())["a3-network1"]
    workflow = _workflow(Path(config["workflow"]).stem)
    jobs = workflow["jobs"]
    (suite,) = config["jobs"]
    daily = _workflow("daily")["jobs"][suite]
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["env"]["CI_COMMAND"] == "a3-network1"
    router = _workflow("ci-command")["jobs"]["route"]
    assert all(f"contains(github.event.comment.body, '{alias}')" in router["if"] for alias in config["aliases"])
    assert jobs[suite]["needs"] == "prepare"
    assert jobs[suite]["concurrency"]["cancel-in-progress"] == "false"
    assert jobs[suite]["concurrency"]["queue"] == "max"
    assert jobs[suite]["uses"] == daily["uses"]
    assert jobs[suite]["with"] == {
        **daily["with"],
        "repository": "${{ github.repository }}",
        "ref": "${{ needs.prepare.outputs.checkout_sha }}",
    }
    inner = _workflow("_st-network1")["jobs"]["run"]["name"].replace("${{ inputs.platform }}", "a2a3")
    assert config["jobs"][suite] == f"{suite} / {inner}"
    assert jobs[config["result_job"]]["needs"] == ["prepare", suite]
    assert set(jobs) == {"prepare", suite, config["result_job"]}
    assert config["status_context"] == "ci/run/a3-network1"
    for key in ("status_workflow", "progress_workflow"):
        publisher = _workflow(Path(config[key]).stem)
        assert set(publisher["on"]["workflow_run"]["workflows"]) == {workflow["name"], _workflow()["name"]}
        assert "a3-network1" in publisher["on"]["workflow_dispatch"]["inputs"]["command"]["options"]
