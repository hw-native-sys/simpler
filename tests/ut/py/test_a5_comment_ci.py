# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the actual comment workflow scripts against CPU-only GitHub API substitutes."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[3]
_SUITES = {
    "ut-a5": "_ut-npu-a5.yml",
    "st-onboard-a5": "_st-npu-a5.yml",
    "st-framework-a5": "_st-framework.yml",
}
_SHA = "1234567890abcdef" * 2 + "12345678"
_BASE = "b" * 40
_MERGE = "c" * 40
_NODE = r"""
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const state = {outputs: {}, failures: [], calls: [], comments: []};
const realSetTimeout = globalThis.setTimeout;
globalThis.setTimeout = (fn, _ms) => realSetTimeout(fn, 0);
const core = {
  setOutput: (name, value) => { state.outputs[name] = value; },
  setFailed: message => state.failures.push(message),
  info: () => {},
  summary: {addRaw: text => {state.summary = text;}, write: async () => {}},
};
let pullIndex = 0;
let commitIndex = 0;
const github = {rest: {
  repos: {getCollaboratorPermissionLevel: async args => {
    state.calls.push({api: 'permission', args});
    if (input.permissionError) throw new Error('permission unavailable');
    return {data: {permission: input.permission, role_name: input.roleName}};
  }, getCommit: async args => {
    state.calls.push({api: 'commit', args});
    const sequence = input.commitSequence || [input.commit];
    const commit = sequence[Math.min(commitIndex, sequence.length - 1)];
    commitIndex += 1;
    return {data: commit};
  }},
  pulls: {get: async args => {
    state.calls.push({api: 'pull', args});
    if (input.pullError) throw new Error('PR unavailable');
    const sequence = input.prSequence || [input.pr];
    const pr = sequence[Math.min(pullIndex, sequence.length - 1)];
    pullIndex += 1;
    return {data: pr};
  }},
  issues: {createComment: async args => { state.comments.push(args); }},
}};
process.env.NEEDS_JSON = JSON.stringify(input.needs || {});
process.env.GITHUB_RUN_ATTEMPT = input.runAttempt || '1';
const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
(async () => {
  try { await new AsyncFunction('github', 'context', 'core', input.script)(github, input.context, core); }
  catch (error) { state.error = error.message; }
  process.stdout.write(JSON.stringify(state));
})();
"""


def _workflow(name="ci-a5-comment"):
    return yaml.load((_ROOT / f".github/workflows/{name}.yml").read_text(), Loader=yaml.BaseLoader)


def _fixture():
    return {
        "permission": "admin",
        "context": {
            "repo": {"owner": "upstream", "repo": "project"},
            "serverUrl": "https://github.example",
            "runId": 321,
            "sha": "f" * 40,
            "payload": {
                "issue": {"number": 42, "pull_request": {}},
                "comment": {"body": "/run-a5", "user": {"login": "maintainer"}},
            },
        },
        "pr": {
            "number": 42,
            "state": "open",
            "base": {"ref": "main", "sha": _BASE, "repo": {"full_name": "upstream/project"}},
            "head": {"sha": _SHA, "repo": {"full_name": "contributor/fork"}},
            "mergeable": True,
            "merge_commit_sha": _MERGE,
        },
        "commit": {"sha": _MERGE, "parents": [{"sha": _BASE}, {"sha": _SHA}]},
    }


def _execute(job, fixture, workflow="ci-a5-comment", step_index=0):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the workflow's github-script controls")
    step = _workflow(workflow)["jobs"][job]["steps"][step_index]
    result = subprocess.run(
        [node, "-e", _NODE],
        input=json.dumps({**fixture, "script": step["with"]["script"]}),
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    return json.loads(result.stdout)


def test_only_three_hardware_suites_move_from_automatic_ci():
    ci = _workflow("ci")["jobs"]
    assert all(name not in ci for name in _SUITES)
    for name in ("ut", "packaging-matrix", "st-sim-a5", "profiling-flags-smoke", "ut-a2a3", "st-onboard-a2a3"):
        assert name in ci
    daily = _workflow("daily")["jobs"]
    assert "ut-a5" not in daily
    assert "st-onboard-a5" in daily and "st-framework-a5" in daily


def test_authorized_topology_pins_all_suites_and_keeps_reporting_permission_separate():
    workflow = _workflow()
    assert workflow["on"] == {"issue_comment": {"types": ["created"]}}
    assert workflow["permissions"] == {"contents": "read"}
    assert "concurrency" not in workflow
    jobs = workflow["jobs"]
    assert set(jobs) == {"gate-and-resolve", "report", *_SUITES}
    gate = jobs["gate-and-resolve"]
    assert "github.event.issue.pull_request" in gate["if"]
    assert gate["runs-on"] == jobs["report"]["runs-on"] == "ubuntu-24.04"
    assert gate["permissions"] == {"contents": "read", "pull-requests": "read"}
    for name in ("ready", "repository", "ref", "pr_number"):
        assert gate["outputs"][name] == f"${{{{ steps.resolve.outputs.{name} }}}}"
    groups = []
    for name, reusable in _SUITES.items():
        job = jobs[name]
        assert job["needs"] == ["gate-and-resolve"]
        assert job["if"] == "needs.gate-and-resolve.outputs.ready == 'true'"
        assert job["uses"] == f"./.github/workflows/{reusable}"
        assert "steps" not in job
        assert job.get("permissions", workflow["permissions"]) == {"contents": "read"}
        assert job["with"]["repository"] == "${{ needs.gate-and-resolve.outputs.repository }}"
        assert job["with"]["ref"] == "${{ needs.gate-and-resolve.outputs.ref }}"
        assert json.loads(job["with"]["runs_on"]) == ["self-hosted", "a5"]
        assert job["concurrency"]["cancel-in-progress"] == "true"
        group = job["concurrency"]["group"]
        assert "${{ needs.gate-and-resolve.outputs.pr_number }}" in group
        groups.append(group)
    assert len(set(groups)) == len(_SUITES)
    for reusable in _SUITES.values():
        jobs_of_suite = _workflow(reusable.removesuffix(".yml"))["jobs"].values()
        steps = [step for job in jobs_of_suite for step in job.get("steps", [])]
        checkouts = [step for step in steps if step.get("uses", "").startswith("actions/checkout@")]
        source_checkouts = [step for step in checkouts if "path" not in step["with"]]
        assert len(source_checkouts) == 1, reusable
        for step in checkouts:
            if step in source_checkouts:
                assert step["with"]["repository"] == "${{ inputs.repository || github.repository }}"
                assert step["with"]["ref"] == "${{ inputs.ref || github.ref }}"
            else:
                assert step["with"]["path"] == ".ci-support"
                assert step["with"]["repository"] == "${{ github.repository }}"
                assert step["with"]["ref"] == "${{ github.workflow_sha }}"
                assert step["with"]["sparse-checkout"] == ".github/actions/upload-diagnostics"
                assert step["continue-on-error"] == "true"
    assert jobs["ut-a5"]["with"]["include_cann_examples"] == "true"
    assert jobs["st-onboard-a5"]["with"]["include_dfx_smokes"] == "true"
    assert jobs["st-framework-a5"]["with"]["platform"] == "a5"
    report = jobs["report"]
    assert set(report["needs"]) == {"gate-and-resolve", *_SUITES}
    assert report["if"].startswith("always() &&")
    assert "needs.gate-and-resolve.result == 'success'" in report["if"]
    assert "needs.gate-and-resolve.outputs.ready == 'true'" in report["if"]
    assert report["permissions"] == {"contents": "read", "pull-requests": "write"}
    assert report["steps"][0]["env"]["NEEDS_JSON"] == "${{ toJSON(needs) }}"


@pytest.mark.parametrize("body", ["/run-a5", " \t/run-a5\n"])
def test_admin_command_resolves_fork_and_full_head_sha(body):
    fixture = _fixture()
    fixture["context"]["payload"]["comment"]["body"] = body
    result = _execute("gate-and-resolve", fixture)
    assert result["failures"] == []
    assert "error" not in result
    assert result["outputs"] == {
        "ready": "true",
        "repository": "upstream/project",
        "ref": _MERGE,
        "head_sha": _SHA,
        "base_sha": _BASE,
        "pr_number": "42",
    }
    assert result["calls"] == [
        {"api": "permission", "args": {"owner": "upstream", "repo": "project", "username": "maintainer"}},
        {"api": "pull", "args": {"owner": "upstream", "repo": "project", "pull_number": 42}},
        {"api": "commit", "args": {"owner": "upstream", "repo": "project", "ref": _MERGE}},
    ]


def test_maintainer_role_can_admit_a5_with_write_permission():
    fixture = _fixture()
    fixture.update(permission="write", roleName="maintain")
    result = _execute("gate-and-resolve", fixture)
    assert result["failures"] == []
    assert result["outputs"]["ready"] == "true"
    assert result["outputs"]["ref"] == _MERGE


@pytest.mark.parametrize("mutation", [None, "conflict", "pending", "stale_head", "stale_base", "missing"])
def test_comment_lanes_pin_matching_merge_parents(mutation):
    fixture = _fixture()
    if mutation == "conflict":
        fixture["pr"]["mergeable"] = False
    elif mutation == "pending":
        fixture["pr"]["mergeable"] = None
    elif mutation == "missing":
        fixture["pr"]["merge_commit_sha"] = None
    elif mutation in ("stale_head", "stale_base"):
        fixture["commit"]["parents"][int(mutation == "stale_head")]["sha"] = "d" * 40
    result = _execute("gate-and-resolve", fixture)
    assert "error" not in result
    if mutation:
        assert result["failures"]
        assert "ref" not in result["outputs"]
        assert result["outputs"]["ready"] == "false"
    else:
        assert result["outputs"]["ref"] == _MERGE
        assert result["outputs"]["repository"] == "upstream/project"
        assert _SHA in result["summary"] and _BASE in result["summary"] and _MERGE in result["summary"]
        assert set(result["outputs"]) == set(_workflow()["jobs"]["gate-and-resolve"]["outputs"])


def test_pending_mergeability_is_retried_before_admission():
    fixture = _fixture()
    fixture["prSequence"] = [
        dict(fixture["pr"], mergeable=None),
        dict(fixture["pr"], mergeable=True),
    ]
    result = _execute("gate-and-resolve", fixture)
    assert result["failures"] == []
    assert result["outputs"]["ready"] == "true"
    assert result["outputs"]["ref"] == _MERGE
    pulls = [call for call in result["calls"] if call["api"] == "pull"]
    assert len(pulls) == 2
    assert any(call["api"] == "commit" for call in result["calls"])


def test_pending_mergeability_then_conflict_is_rejected():
    fixture = _fixture()
    fixture["prSequence"] = [
        dict(fixture["pr"], mergeable=None),
        dict(fixture["pr"], mergeable=False),
    ]
    result = _execute("gate-and-resolve", fixture)
    assert result["failures"]
    assert "unavailable or conflicting" in result["failures"][0]
    assert result["outputs"] == {"ready": "false"}


def test_pending_mergeability_then_stale_snapshot_is_rejected():
    fixture = _fixture()
    fixture["prSequence"] = [
        dict(fixture["pr"], mergeable=None),
        dict(fixture["pr"], mergeable=True),
    ]
    fixture["commit"] = {"sha": _MERGE, "parents": [{"sha": _BASE}, {"sha": "d" * 40}]}
    result = _execute("gate-and-resolve", fixture)
    assert result["failures"]
    assert "stale" in result["failures"][0]
    assert result["outputs"] == {"ready": "false"}
    assert len([call for call in result["calls"] if call["api"] == "pull"]) == 13


def test_stale_merge_snapshot_refreshes_before_admission():
    fixture = _fixture()
    previous = "a" * 40
    fixture["prSequence"] = [dict(fixture["pr"], merge_commit_sha=previous), fixture["pr"]]
    fixture["commitSequence"] = [
        {"sha": previous, "parents": [{"sha": "d" * 40}, {"sha": _SHA}]},
        fixture["commit"],
    ]
    result = _execute("gate-and-resolve", fixture)
    assert result["failures"] == []
    assert result["outputs"]["ready"] == "true"
    assert result["outputs"]["ref"] == _MERGE
    assert len([call for call in result["calls"] if call["api"] == "pull"]) == 2
    assert [call["args"]["ref"] for call in result["calls"] if call["api"] == "commit"] == [previous, _MERGE]


@pytest.mark.parametrize("attempt", ["0", "2", "3"])
def test_rerun_gate_cannot_admit_the_current_head(attempt):
    fixture = _fixture()
    fixture["runAttempt"] = attempt
    result = _execute("gate-and-resolve", fixture)
    assert result["calls"] == result["comments"] == []
    assert result["outputs"] == {"ready": "false"}
    assert len(result["failures"]) == 1


@pytest.mark.parametrize(
    "body", ["/run-a5 extra", "/run-a5evil", "please /run-a5", "/run-self", "", "/run-a5\n/run-a5"]
)
def test_unmatched_comment_does_not_query_permission_or_resolve(body):
    fixture = _fixture()
    fixture["context"]["payload"]["comment"]["body"] = body
    result = _execute("gate-and-resolve", fixture)
    assert result["calls"] == result["failures"] == result["comments"] == []
    assert result["outputs"] == {"ready": "false"}


def test_non_pr_comment_is_ignored():
    fixture = _fixture()
    del fixture["context"]["payload"]["issue"]["pull_request"]
    result = _execute("gate-and-resolve", fixture)
    assert result["calls"] == result["comments"] == []
    assert result["outputs"] == {"ready": "false"}


@pytest.mark.parametrize("permission", ["write", "maintain", "read", "none", None, "api-error"])
def test_unprivileged_or_unverified_access_never_resolves_a_head(permission):
    fixture = _fixture()
    fixture["permission"] = permission
    fixture["permissionError"] = permission == "api-error"
    result = _execute("gate-and-resolve", fixture)
    assert [call["api"] for call in result["calls"]] == ["permission"]
    assert result["outputs"] == {"ready": "false"}
    assert len(result["failures"]) == 1
    assert result["comments"] == []


@pytest.mark.parametrize(
    "invalid", ["closed", "branch", "base-repo", "deleted-fork", "short-sha", "wrong-number", "api-error"]
)
def test_invalid_pr_or_lookup_failure_cannot_admit_suites(invalid):
    fixture = _fixture()
    pr = fixture["pr"]
    if invalid == "closed":
        pr["state"] = "closed"
    elif invalid == "branch":
        pr["base"]["ref"] = "release"
    elif invalid == "base-repo":
        pr["base"]["repo"]["full_name"] = "another/project"
    elif invalid == "deleted-fork":
        pr["head"]["repo"] = None
    elif invalid == "short-sha":
        pr["head"]["sha"] = "1234567"
    elif invalid == "wrong-number":
        pr["number"] = 43
    else:
        fixture["pullError"] = True
    result = _execute("gate-and-resolve", fixture)
    assert result["outputs"] == {"ready": "false"}
    assert result["failures"] or result.get("error")
    assert result["comments"] == []


def _report_fixture():
    fixture = _fixture()
    needs: dict[str, Any] = {name: {"result": "success"} for name in _SUITES}
    needs["gate-and-resolve"] = {
        "result": "success",
        "outputs": {
            "ready": "true",
            "repository": "upstream/project",
            "ref": _MERGE,
            "head_sha": _SHA,
            "base_sha": _BASE,
            "pr_number": "42",
        },
    }
    fixture["needs"] = needs
    return fixture


@pytest.mark.parametrize("status", ["success", "failure", "cancelled", "skipped", "unknown"])
def test_report_names_exact_tested_head_run_and_all_suite_results(status):
    fixture = _report_fixture()
    fixture["needs"]["st-onboard-a5"]["result"] = status
    result = _execute("report", fixture)
    assert "error" not in result
    assert len(result["comments"]) == 1
    comment = result["comments"][0]
    assert comment["issue_number"] == 42
    body = comment["body"]
    assert f"upstream/project@{_MERGE}" in body
    assert f"PR head: `{_SHA}`" in body and f"Base: `{_BASE}`" in body
    assert fixture["context"]["sha"] not in body
    assert "https://github.example/upstream/project/actions/runs/321" in body
    for name in _SUITES:
        expected = status if name == "st-onboard-a5" else "success"
        assert f"{name}: **{expected}**" in body
    assert ("**/run-a5 passed**" in body) == (status == "success")


@pytest.mark.parametrize("admitted", [False, "failed-gate"])
def test_report_ignores_unadmitted_requests(admitted):
    fixture = _report_fixture()
    gate = fixture["needs"]["gate-and-resolve"]
    if admitted is False:
        gate["outputs"]["ready"] = "false"
    else:
        gate["result"] = "failure"
    assert _execute("report", fixture)["comments"] == []
