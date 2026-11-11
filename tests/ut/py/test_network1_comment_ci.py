# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise Network1 comment admission and reporting without scheduling devices."""

import json

import pytest
from test_a5_comment_ci import _SHA, _execute, _fixture, _workflow

_WORKFLOW = "ci-network1-comment"
_SUITE = "st-network1-onboard-a2a3"


def _request():
    fixture = _fixture()
    fixture["context"]["payload"]["comment"]["body"] = "/run-network"
    return fixture


def test_network1_requires_admission_and_retains_daily_coverage():
    workflow = _workflow(_WORKFLOW)
    assert workflow["on"] == {"issue_comment": {"types": ["created"]}}
    assert workflow["permissions"] == {"contents": "read"}
    assert "concurrency" not in workflow
    jobs = workflow["jobs"]
    assert set(jobs) == {"gate-and-resolve", _SUITE, "report"}
    gate = jobs["gate-and-resolve"]
    assert gate["runs-on"] == jobs["report"]["runs-on"] == "ubuntu-24.04"
    assert gate["permissions"] == {"contents": "read", "pull-requests": "read"}
    suite = jobs[_SUITE]
    assert suite["needs"] == ["gate-and-resolve"]
    assert suite["if"] == "needs.gate-and-resolve.outputs.ready == 'true'"
    assert suite["with"]["repository"] == "${{ needs.gate-and-resolve.outputs.repository }}"
    assert suite["with"]["ref"] == "${{ needs.gate-and-resolve.outputs.ref }}"
    assert suite["concurrency"]["cancel-in-progress"] == "true"
    assert "${{ needs.gate-and-resolve.outputs.pr_number }}" in suite["concurrency"]["group"]
    daily = _workflow("daily")["jobs"][_SUITE]
    assert suite["uses"] == daily["uses"] == "./.github/workflows/_st-network1.yml"
    assert suite["with"]["platform"] == daily["with"]["platform"] == "a2a3"
    assert json.loads(suite["with"]["runs_on"]) == json.loads(daily["with"]["runs_on"])
    assert _SUITE not in _workflow("ci")["jobs"]
    report = jobs["report"]
    assert set(report["needs"]) == {"gate-and-resolve", _SUITE}
    assert report["if"].startswith("always() &&")
    assert "needs.gate-and-resolve.outputs.ready == 'true'" in report["if"]
    assert report["permissions"] == {"contents": "read", "pull-requests": "write"}


@pytest.mark.parametrize("body", ["/run-network", " \t/run-network\n"])
def test_admin_request_pins_the_fork_head(body):
    fixture = _request()
    fixture["context"]["payload"]["comment"]["body"] = body
    result = _execute("gate-and-resolve", fixture, _WORKFLOW)
    assert result["failures"] == []
    assert "error" not in result
    assert result["outputs"] == {"ready": "true", "repository": "contributor/fork", "ref": _SHA, "pr_number": "42"}
    assert [call["api"] for call in result["calls"]] == ["permission", "pull"]


@pytest.mark.parametrize("permission", ["write", "maintain", "read", "none", None, "api-error"])
def test_non_admin_or_unverified_permission_cannot_admit_network1(permission):
    fixture = _request()
    fixture["permission"] = permission
    fixture["permissionError"] = permission == "api-error"
    result = _execute("gate-and-resolve", fixture, _WORKFLOW)
    assert result["outputs"] == {"ready": "false"}
    assert [call["api"] for call in result["calls"]] == ["permission"]
    assert len(result["failures"]) == 1
    assert result["comments"] == []


@pytest.mark.parametrize(
    "body", ["/run-network extra", "/run-networkevil", "please /run-network", "/run-a5", "/run-network1", ""]
)
def test_other_comments_cannot_resolve_or_cancel_a_request(body):
    fixture = _request()
    fixture["context"]["payload"]["comment"]["body"] = body
    result = _execute("gate-and-resolve", fixture, _WORKFLOW)
    assert result["outputs"] == {"ready": "false"}
    assert result["calls"] == result["failures"] == result["comments"] == []


def test_non_pr_comment_is_ignored():
    fixture = _request()
    del fixture["context"]["payload"]["issue"]["pull_request"]
    result = _execute("gate-and-resolve", fixture, _WORKFLOW)
    assert result["outputs"] == {"ready": "false"}
    assert result["calls"] == result["comments"] == []


@pytest.mark.parametrize("invalid", ["closed", "base", "fork", "sha", "number", "api-error"])
def test_invalid_pr_cannot_admit_network1(invalid):
    fixture = _request()
    pr = fixture["pr"]
    if invalid == "closed":
        pr["state"] = "closed"
    elif invalid == "base":
        pr["base"]["ref"] = "release"
    elif invalid == "fork":
        pr["head"]["repo"] = None
    elif invalid == "sha":
        pr["head"]["sha"] = "1234567"
    elif invalid == "number":
        pr["number"] = 43
    else:
        fixture["pullError"] = True
    result = _execute("gate-and-resolve", fixture, _WORKFLOW)
    assert result["outputs"] == {"ready": "false"}
    assert result["failures"] or result.get("error")


@pytest.mark.parametrize("attempt", ["0", "2", "3"])
def test_rerun_cannot_resolve_a_different_head(attempt):
    fixture = _request()
    fixture["runAttempt"] = attempt
    result = _execute("gate-and-resolve", fixture, _WORKFLOW)
    assert result["outputs"] == {"ready": "false"}
    assert result["calls"] == result["comments"] == []
    assert len(result["failures"]) == 1


@pytest.mark.parametrize("status", ["success", "failure", "cancelled", "skipped", "unknown", None])
def test_report_preserves_the_tested_head_and_actual_network1_result(status):
    fixture = _request()
    fixture["needs"] = {
        "gate-and-resolve": {
            "result": "success",
            "outputs": {"ready": "true", "repository": "contributor/fork", "ref": _SHA, "pr_number": "42"},
        },
        _SUITE: {"result": status} if status else {},
    }
    result = _execute("report", fixture, _WORKFLOW)
    assert "error" not in result
    assert len(result["comments"]) == 1
    comment = result["comments"][0]
    assert comment["issue_number"] == 42
    body = comment["body"]
    assert f"contributor/fork@{_SHA}" in body
    assert fixture["context"]["sha"] not in body
    assert "https://github.example/upstream/project/actions/runs/321" in body
    assert f"{_SUITE}: **{status or 'unknown'}**" in body
    assert ("**/run-network passed**" in body) == (status == "success")


@pytest.mark.parametrize("ready,result", [("false", "success"), ("true", "failure")])
def test_unadmitted_request_does_not_report(ready, result):
    fixture = _request()
    fixture["needs"] = {"gate-and-resolve": {"result": result, "outputs": {"ready": ready}}}
    assert _execute("report", fixture, _WORKFLOW)["comments"] == []
