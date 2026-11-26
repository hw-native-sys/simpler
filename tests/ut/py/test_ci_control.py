# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise CI routing and merge-gate failures without GitHub writes or devices."""

import copy
import importlib.util
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[3]
REPO = "owner/repo"
HEAD, BASE, MERGE = "a" * 40, "b" * 40, "c" * 40


def test_a5_command_requires_maintain_or_admin(controller, monkeypatch, event, pr):
    event["comment"]["body"] = "/run a5"

    def api(path, payload=None):
        if path.endswith("/permission"):
            return {"permission": "write", "role_name": "write"}
        if path.endswith("/pulls/123"):
            return pr
        raise AssertionError(path)

    monkeypatch.setattr(controller, "api", api)
    with pytest.raises(ValueError, match="maintain or admin"):
        controller.validated_request(REPO, event)


@pytest.fixture
def a5_api(controller, monkeypatch, pr, tmp_path):
    access = {"permission": "admin", "role_name": "admin"}
    parent = {
        "id": 900,
        "run_attempt": 1,
        "workflow_id": 88,
        "path": ".github/workflows/ci-command.yml",
        "event": "issue_comment",
        "head_branch": "main",
        "repository": {"id": 1, "full_name": REPO},
        "head_repository": {"id": 1},
        "actor": {"login": "maintainer"},
    }
    jobs = [
        {"name": "route", "conclusion": "success"},
        {"name": f"Dispatch a5 PR #123 @{HEAD}", "conclusion": None},
    ]
    calls, comments = [], []

    def api(path, payload=None, **kwargs):
        calls.append((path, copy.deepcopy(payload)))
        if "/permission" in path:
            return access
        if path.endswith("/pulls/123"):
            return pr
        if path.endswith("/git/ref/heads/main"):
            return {"ref": "refs/heads/main", "object": {"type": "commit", "sha": BASE}}
        if path.endswith(f"/git/commits/{MERGE}"):
            return {"parents": [{"sha": BASE}, {"sha": HEAD}]}
        if path.endswith("/runs/900/attempts/1"):
            return parent
        if path.endswith("/workflows/ci-command.yml"):
            return {"id": 88}
        if "/runs/900/attempts/1/jobs?" in path:
            return {"jobs": jobs}
        if "/issues/123/comments" in path:
            if payload:
                comments.append({"user": {"login": "github-actions[bot]"}, **payload})
                return comments[-1]
            return comments
        if "event=workflow_dispatch" in path:
            return {"workflow_runs": []}
        if f"/commits/{HEAD}/status?" in path:
            return {"statuses": []}
        if payload is not None:
            if path.endswith(f"/{controller.CONFIG['workflow']}/dispatches"):
                return {"workflow_run_id": 50}
            return payload
        raise AssertionError(path)

    monkeypatch.setattr(controller, "api", api)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    target = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    return access, parent, jobs, calls, comments, target


@pytest.mark.parametrize("alias", ["/run-a5", "/run a5", " \t/run a5\n"])
@pytest.mark.parametrize("role", ["admin", "maintain"])
def test_a5_dispatch_binds_pending_before_tests_and_progress(controller, event, a5_api, alias, role):
    access, _, _, calls, _, _ = a5_api
    access.update(permission="admin" if role == "admin" else "write", role_name=role)
    event["comment"]["body"] = alias
    controller.dispatch(REPO, event)
    writes = [(path, payload) for path, payload in calls if payload is not None]
    assert [path.split("/")[-1] for path, _ in writes] == [HEAD, "dispatches", HEAD, "dispatches"]
    assert writes[0][1]["state"] == "pending"
    assert "request 900-1" in writes[0][1]["description"]
    assert writes[1][0].endswith("/ci-a5-comment.yml/dispatches")
    assert writes[1][1] == {
        "ref": "main",
        "return_run_details": True,
        "inputs": {"pr_number": "123", "head_sha": HEAD, "base_sha": BASE, "merge_sha": MERGE, "request_id": "900-1"},
    }
    assert writes[2][1]["state"] == "pending"
    assert writes[2][1]["target_url"].endswith("/runs/50/attempts/1")
    assert writes[3][0].endswith("/ci-command-progress.yml/dispatches")


@pytest.mark.parametrize("controller", ["a3-network1"], indirect=True)
def test_network1_routes_only_its_command_and_reports_the_tested_merge(controller, event, a5_api):
    access, _, _, calls, _, _ = a5_api
    for comment in ("/run a5", "please /run a3-network1", "/run a3-network1 extra"):
        event["comment"]["body"] = comment
        controller.dispatch(REPO, event)
    assert not calls
    event["comment"]["body"] = "/run a3-network1"
    access.update(permission="write", role_name="write")
    with pytest.raises(ValueError, match="maintain or admin"):
        controller.dispatch(REPO, event)
    assert not any(payload for _, payload in calls)
    access.update(role_name="maintain")
    controller.dispatch(REPO, event)
    pending, hardware, bound, progress = [(path, payload) for path, payload in calls if payload]
    assert pending[1]["context"] == bound[1]["context"] == "ci/run/a3-network1"
    assert hardware[0].endswith("/ci-a3-network1.yml/dispatches")
    assert hardware[1]["inputs"]["merge_sha"] == MERGE
    assert progress[1]["inputs"] == {"run_id": "50", "command": "a3-network1"}
    needs = {"prepare": {"result": "success"}, "st-network1-onboard-a2a3": {"result": "success"}}
    controller.aggregate(needs)
    needs["st-network1-onboard-a2a3"]["result"] = "skipped"
    with pytest.raises(ValueError, match="st-network1-onboard-a2a3"):
        controller.aggregate(needs)


@pytest.mark.parametrize("changed", ["base", "merge", "head"])
def test_dispatch_keeps_the_snapshot_admitted_before_queueing(controller, monkeypatch, event, pr, a5_api, changed):
    original = controller.api
    controller.command_route(REPO, event)
    routed = dict(line.split("=", 1) for line in a5_api[-1].read_text().splitlines())
    for key, expected in {"head": HEAD, "base": BASE, "merge": MERGE}.items():
        assert routed[key] == expected
        monkeypatch.setenv(f"CI_ROUTED_{key.upper()}", routed[key])

    new_base = "d" * 40 if changed == "base" else BASE
    new_head = "e" * 40 if changed == "head" else HEAD
    pr["head"]["sha"] = new_head
    pr["merge_commit_sha"] = "f" * 40

    def api(path, payload=None):
        if path.endswith("/git/ref/heads/main"):
            return {"ref": "refs/heads/main", "object": {"type": "commit", "sha": new_base}}
        if path.endswith("/git/commits/" + pr["merge_commit_sha"]):
            return {"parents": [{"sha": new_base}, {"sha": new_head}]}
        return original(path, payload)

    monkeypatch.setattr(controller, "api", api)
    with pytest.raises(ValueError, match="snapshot changed"):
        controller.dispatch(REPO, event)
    assert not any(payload for _, payload in a5_api[3])


@pytest.mark.parametrize("still_owned", [True, False])
def test_dispatch_rejection_closes_only_its_own_marker(controller, monkeypatch, event, a5_api, still_owned):
    original = controller.api
    statuses = []

    def api(path, payload=None):
        if path.endswith(f"/statuses/{HEAD}"):
            statuses.append(payload)
            return payload
        if "/status?" in path and statuses:
            owner = statuses[-1] if still_owned else {**statuses[-1], "description": "another request"}
            return {"statuses": [owner]}
        if path.endswith("/ci-a5-comment.yml/dispatches"):
            raise subprocess.CalledProcessError(1, ["gh", "api"], output="HTTP/2.0 422 Unprocessable Entity\n")
        return original(path, payload)

    monkeypatch.setattr(controller, "api", api)
    with pytest.raises(subprocess.CalledProcessError):
        controller.dispatch(REPO, event)
    assert [status["state"] for status in statuses] == (["pending", "failure"] if still_owned else ["pending"])
    if still_owned:
        assert statuses[-1]["target_url"].endswith("/runs/900/attempts/1")
        assert "900-1" in statuses[-1]["description"]


@pytest.mark.parametrize("created,protected", [("01", True), ("03", False)])
def test_rejected_request_protects_status_from_late_completions(controller, monkeypatch, created, protected):
    marker = {
        "context": controller.STATUS_CONTEXT,
        "state": "failure",
        "created_at": "02",
        "description": "A5 CI dispatch rejected (request 900-1)",
        "target_url": f"https://github.com/{REPO}/actions/runs/900/attempts/1",
    }
    monkeypatch.setattr(controller, "pages", lambda *args: [marker])
    assert controller.newer_status_exists(REPO, 123, HEAD, {"created_at": created}) is protected


def test_dispatch_rest_response_includes_http_status(controller, monkeypatch):
    def gh(command, **kwargs):
        assert "--include" in command
        return SimpleNamespace(stdout='HTTP/2.0 200 OK\nContent-Type: application/json\n\n{"workflow_run_id":50}')

    monkeypatch.setattr(controller.subprocess, "run", gh)
    assert controller.api("repos/owner/repo/actions/workflows/ci.yml/dispatches", {"ref": "main"}) == {
        "workflow_run_id": 50
    }


def test_lost_dispatch_response_recovers_indexed_admitted_run(controller, monkeypatch, event, a5_api):
    original = controller.api
    posts = []

    def api(path, payload=None):
        if path.endswith("/ci-a5-comment.yml/dispatches"):
            posts.append(payload)
            raise subprocess.CalledProcessError(1, ["gh", "api"], output="HTTP/2.0 502 Bad Gateway\n")
        if "/ci-a5-comment.yml/runs?" in path:
            return {
                "workflow_runs": [
                    {
                        "id": 77,
                        "workflow_id": 33,
                        "run_attempt": 1,
                        "path": controller.WORKFLOW,
                        "display_title": f"CI a5 PR #123 @{HEAD} base:{BASE} request:900-1",
                        "actor": {"login": "github-actions[bot]"},
                    }
                ]
            }
        if path.endswith("/workflows/ci-a5-comment.yml"):
            return {"id": 33}
        if "/runs/77/attempts/1/jobs?" in path:
            return {
                "jobs": [{"name": "prepare", "steps": [{"name": "Authorize A5 dispatch", "conclusion": "success"}]}]
            }
        return original(path, payload)

    monkeypatch.setattr(controller, "api", api)
    controller.dispatch(REPO, event)
    assert len(posts) == 1
    assert a5_api[3][-1][1]["inputs"]["run_id"] == "77"


def test_unidentified_marker_does_not_claim_a_status(controller, monkeypatch):
    monkeypatch.setattr(
        controller,
        "pages",
        lambda *args: [
            {
                "context": controller.STATUS_CONTEXT,
                "state": "pending",
                "target_url": f"https://github.com/{REPO}/pull/123",
                "description": "A5 CI requested; waiting for dispatch",
            }
        ],
    )
    assert not controller.newer_status_exists(REPO, 123, HEAD, {})


def test_pending_write_failure_does_not_launch_a5(controller, monkeypatch, event, a5_api):
    calls = a5_api[3]
    original_api = controller.api

    def api(path, payload=None):
        if payload is not None:
            raise subprocess.CalledProcessError(1, ["gh", "api"])
        return original_api(path, payload)

    monkeypatch.setattr(controller, "api", api)
    with pytest.raises(subprocess.CalledProcessError):
        controller.dispatch(REPO, event)
    assert not any(path.endswith("/dispatches") for path, _ in calls)


def test_rerunning_comment_admission_cannot_authorize_another_head(controller, monkeypatch, event, a5_api):
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    with pytest.raises(ValueError, match="new command"):
        controller.dispatch(REPO, event)
    assert not any(payload is not None for _, payload in a5_api[3])


def test_a5_prepare_accepts_only_the_admitted_pr_head(controller, a5_api):
    event = {
        "sender": {"login": "github-actions[bot]"},
        "inputs": {
            "pr_number": "123",
            "head_sha": HEAD,
            "base_sha": BASE,
            "merge_sha": MERGE,
            "request_id": "900-1",
        },
    }
    controller.authorize_dispatch(REPO, event)
    controller.prepare(REPO, event)
    assert a5_api[-1].read_text() == f"checkout_sha={MERGE}\n"


def test_bot_dispatch_without_a_command_request_cannot_run_a5(controller, a5_api):
    event = {
        "sender": {"login": "github-actions[bot]"},
        "inputs": {"pr_number": "123", "head_sha": HEAD, "base_sha": BASE, "merge_sha": MERGE},
    }
    with pytest.raises(ValueError, match="command request"):
        controller.authorize_dispatch(REPO, event)
    assert not a5_api[-1].exists()


@pytest.mark.parametrize("parents", [[BASE, "d" * 40], ["d" * 40, HEAD], [BASE, HEAD, MERGE]])
def test_a5_prepare_rejects_a_merge_with_different_parents(controller, monkeypatch, a5_api, parents):
    original_api = controller.api

    def api(path, payload=None):
        if path.endswith(f"/git/commits/{MERGE}"):
            return {"parents": [{"sha": value} for value in parents]}
        return original_api(path, payload)

    monkeypatch.setattr(controller, "api", api)
    event = {
        "sender": {"login": "github-actions[bot]"},
        "inputs": {
            "pr_number": "123",
            "head_sha": HEAD,
            "base_sha": BASE,
            "merge_sha": MERGE,
            "request_id": "900-1",
        },
    }
    with pytest.raises(ValueError, match="parents"):
        controller.prepare(REPO, event)
    assert not a5_api[-1].exists()


@pytest.mark.parametrize("transition", ["refresh", "changed-head", "timeout"])
def test_a5_admission_wait_keeps_the_original_revision(controller, monkeypatch, event, pr, a5_api, transition):
    pr["mergeable"] = None
    clock = [0]

    def sleep(seconds):
        clock[0] += seconds
        if transition == "refresh":
            pr["mergeable"] = True
        elif transition == "changed-head":
            pr["head"]["sha"] = "d" * 40

    monkeypatch.setattr(controller, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))
    if transition == "refresh":
        controller.dispatch(REPO, event)
        assert clock[0] == 5
        assert any(path.endswith("/dispatches") for path, _ in a5_api[3])
    else:
        with pytest.raises(ValueError, match="changed" if transition == "changed-head" else "60 seconds"):
            controller.dispatch(REPO, event)
        assert not any(payload is not None for _, payload in a5_api[3])
        assert clock[0] <= 60


@pytest.mark.parametrize("invalid", ["foreign-repo", "workflow", "branch", "event", "unadmitted", "head", "permission"])
def test_untrusted_dispatch_never_produces_a_test_checkout(controller, a5_api, invalid):
    access, parent, jobs, _, _, target = a5_api
    event = {
        "sender": {"login": "github-actions[bot]"},
        "inputs": {
            "pr_number": "123",
            "head_sha": HEAD,
            "base_sha": BASE,
            "merge_sha": MERGE,
            "request_id": "900-1",
        },
    }
    if invalid == "foreign-repo":
        parent["head_repository"]["id"] = 2
    elif invalid == "workflow":
        parent["workflow_id"] = 89
    elif invalid == "branch":
        parent["head_branch"] = "feature"
    elif invalid == "event":
        parent["event"] = "pull_request"
    elif invalid == "unadmitted":
        jobs[-1]["conclusion"] = "skipped"
    elif invalid == "head":
        event["inputs"]["head_sha"] = "d" * 40
    else:
        access.update(permission="write", role_name="write")
    with pytest.raises(ValueError):
        controller.authorize_dispatch(REPO, event)
    assert not target.exists()


def test_denial_reason_reaches_the_pr_comment(controller, monkeypatch, event, a5_api, tmp_path):
    access, _, _, _, comments, output = a5_api
    access.update(permission="write", role_name="write")
    event_file = tmp_path / "event.json"
    event_file.write_text(json.dumps(event))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_file))
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    monkeypatch.setattr(controller.sys, "argv", ["ci_control.py", "command-route"])
    with pytest.raises(SystemExit):
        controller.main()
    reason = output.read_text().strip().removeprefix("command_error=")
    monkeypatch.setenv("CI_COMMAND_ERROR", reason)
    controller.command_feedback(REPO, event)
    assert len(comments) == 1 and "maintain or admin" in comments[0]["body"]


@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped"])
def test_a5_aggregate_requires_every_existing_suite(controller, result):
    needs = {name: {"result": "success"} for name in controller.REQUIRED}
    controller.aggregate(needs)
    needs["st-onboard-a5"]["result"] = result
    with pytest.raises(ValueError, match="st-onboard-a5"):
        controller.aggregate(needs)


@pytest.fixture
def controller(monkeypatch, request):
    monkeypatch.setenv("CI_COMMAND", getattr(request, "param", "a5"))
    spec = importlib.util.spec_from_file_location("ci_control", ROOT / ".github/scripts/ci_control.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv("CI_ROUTED_EVENT", raising=False)
    monkeypatch.setenv("CI_ROUTED_HEAD", HEAD)
    monkeypatch.setenv("CI_ROUTED_BASE", BASE)
    monkeypatch.setenv("CI_ROUTED_MERGE", MERGE)
    monkeypatch.setenv("GITHUB_RUN_ID", "900")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected subprocess"))
    return module


@pytest.fixture
def admission_check(controller):
    return controller.run_admitted


@pytest.fixture
def control(controller, monkeypatch, admission_check):
    # Most cases isolate other API contracts; live-ref tests use controller.
    monkeypatch.setattr(controller, "run_admitted", lambda repo, run: True)
    monkeypatch.setattr(controller, "current_base", lambda repo: BASE)
    return controller


@pytest.fixture
def pr():
    return {
        "state": "open",
        "number": 123,
        "created_at": "2026-01-01T00:00:00Z",
        "mergeable": True,
        "merge_commit_sha": MERGE,
        "head": {"sha": HEAD, "repo": {"id": 2}},
        "base": {"sha": BASE, "ref": "main", "repo": {"id": 1, "full_name": REPO}},
    }


@pytest.fixture
def event():
    return {
        "action": "created",
        "issue": {"number": 123, "pull_request": {}},
        "comment": {"body": "/run a5", "user": {"login": "external-user"}, "author_association": "NONE"},
    }


@pytest.fixture
def merge_clock(control, monkeypatch):
    elapsed = [0.0]
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        elapsed[0] += seconds

    monkeypatch.setattr(control, "time", SimpleNamespace(monotonic=lambda: elapsed[0], sleep=sleep))
    return waits


@pytest.fixture
def publication(control, monkeypatch, pr):
    run = {
        "id": 50,
        "run_attempt": 2,
        "run_number": 8,
        "display_title": f"CI {control.COMMAND} PR #123 @{HEAD} base:{BASE}",
        "created_at": "2026-11-05T01:00:00Z",
        "event": "workflow_dispatch",
        "path": control.WORKFLOW,
        "workflow_id": 99,
        "head_branch": "main",
        "repository": {"full_name": REPO, "id": 1},
        "head_repository": {"id": 1},
        "status": "completed",
        "conclusion": "success",
    }
    jobs = [{"name": control.CONFIG["result_job"], "conclusion": "success"}]
    candidates = [run]
    writes = []

    def api(path, payload=None):
        if payload is not None:
            writes.append((path, payload))
        elif path.endswith("/actions/runs/50"):
            return run
        elif path.endswith(f"/workflows/{control.CONFIG['workflow']}"):
            return {"id": 99, "path": control.WORKFLOW}
        elif path.endswith("/pulls/123"):
            return pr
        elif "event=workflow_dispatch" in path:
            assert "created=>=2026-11-05T01:00:00Z" in path
            return {"workflow_runs": candidates}
        else:
            raise AssertionError(path)

    def pages(path, key):
        if key == "statuses":
            assert path == f"repos/{REPO}/commits/{HEAD}/status"
            return [
                {
                    "context": control.STATUS_CONTEXT,
                    "state": "success",
                    "target_url": f"https://github.com/{REPO}/actions/runs/50/attempts/1",
                }
            ]
        if path.endswith("/attempts/2/jobs"):
            return jobs
        assert "event=workflow_dispatch" in path
        return candidates

    monkeypatch.setattr(control, "api", api)
    monkeypatch.setattr(control, "pages", pages)
    return run, jobs, candidates, writes


@pytest.mark.parametrize("controller", ["a5", "a3-network1"], indirect=True)
def test_native_rerun_updates_original_head(control, publication):
    _, _, _, writes = publication
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert writes[0][0] == f"repos/{REPO}/statuses/{HEAD}"
    assert writes[0][1]["state"] == "success"
    assert writes[0][1]["context"] == f"ci/run/{control.COMMAND}"
    assert writes[0][1]["target_url"].endswith("/runs/50/attempts/2")


@pytest.mark.parametrize("conclusion", ["skipped", "failure", "cancelled", None])
def test_non_success_finalizer_never_marks_green(control, publication, conclusion):
    _, jobs, _, writes = publication
    jobs[0]["conclusion"] = conclusion
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert writes[0][1]["state"] == "failure"


def test_missing_finalizer_in_current_attempt_blocks_merge(control, publication):
    _, jobs, _, writes = publication
    jobs.clear()
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert writes[0][1]["state"] == "failure"


def test_delayed_completion_cannot_green_running_rerun(control, publication):
    run, _, _, writes = publication
    run["status"] = "in_progress"
    control.publish(REPO, {"workflow_run": {"id": 50, "status": "completed"}})
    assert writes[0][1]["state"] == "pending"


@pytest.mark.parametrize("change", ["head", "closed", "base_ref", "base_repo"])
@pytest.mark.parametrize(
    "run_status,conclusion", [("in_progress", None), ("completed", "failure"), ("completed", "success")]
)
def test_obsolete_pr_invalidates_original_head_before_it_can_return(
    control, publication, pr, change, run_status, conclusion
):
    original_pr = copy.deepcopy(pr)
    run, _, _, writes = publication
    run.update(status=run_status, conclusion=conclusion)
    if change == "head":
        pr["head"]["sha"] = "d" * 40
    elif change == "closed":
        pr["state"] = "closed"
    elif change == "base_ref":
        pr["base"]["ref"] = "release"
    else:
        pr["base"]["repo"]["full_name"] = "different/repo"
    control.publish(REPO, {"workflow_run": {"id": 50}})
    # Resetting/reopening the PR after publication must not revive the old green.
    pr.update(original_pr)
    assert len(writes) == 1
    assert writes[0][0] == f"repos/{REPO}/statuses/{HEAD}"
    assert writes[0][1]["state"] == "failure"
    assert writes[0][1]["target_url"].endswith("/runs/50/attempts/2")


@pytest.mark.parametrize("run_status", ["in_progress", "completed"])
@pytest.mark.parametrize("newer_lookup_fails", [False, True])
def test_moved_head_preserves_only_verified_newer_status(
    control, monkeypatch, publication, pr, run_status, newer_lookup_fails
):
    run, _, _, writes = publication
    run.update(status=run_status, conclusion="failure")
    pr["head"]["sha"] = "d" * 40
    candidate = copy.deepcopy(run)
    candidate.update(id=51, run_number=9, status="completed", conclusion="success")
    original_api = control.api

    def request(path, payload=None):
        if path.endswith("/actions/runs/51"):
            if newer_lookup_fails:
                raise subprocess.CalledProcessError(1, ["gh", "api"])
            return candidate
        return original_api(path, payload)

    monkeypatch.setattr(control, "api", request)
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [
            {
                "context": "ci/run/a5",
                "state": "success",
                "target_url": f"https://github.com/{REPO}/actions/runs/51/attempts/1",
            }
        ],
    )
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert not writes


def test_old_request_cannot_overwrite_newer_request(control, publication):
    run, _, candidates, writes = publication
    newer = copy.deepcopy(run)
    newer.update(id=51, run_number=9, status="queued")
    candidates.append(newer)
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert not writes


@pytest.mark.parametrize("branch", ["feature", "release-tag"])
@pytest.mark.parametrize("phase", ["in_progress", "completed"])
def test_feature_dispatch_cannot_suppress_genuine_result(control, publication, branch, phase):
    run, jobs, candidates, writes = publication
    run["status"] = phase
    run["conclusion"] = "failure"
    jobs[0]["conclusion"] = "failure"
    newer = copy.deepcopy(run)
    newer.update(id=51, run_number=9, head_branch=branch)
    candidates.append(newer)
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert len(writes) == 1
    assert writes[0][1]["state"] == ("pending" if phase == "in_progress" else "failure")


@pytest.mark.parametrize("branch", ["feature", "release-tag"])
def test_failed_metadata_lookup_preserves_other_workflow_binding(control, monkeypatch, publication, branch):
    run, _, _, writes = publication
    newer = copy.deepcopy(run)
    newer.update(id=51, run_number=9, head_branch=branch)
    original_api = control.api

    def request(path, payload=None):
        if path.endswith("/pulls/123"):
            raise ValueError("PR metadata unavailable")
        if path.endswith("/actions/runs/51"):
            return newer
        return original_api(path, payload)

    monkeypatch.setattr(control, "api", request)
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [{"context": "ci/run/a5", "target_url": f"https://github.com/{REPO}/actions/runs/51/attempts/1"}],
    )
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert not writes


@pytest.mark.parametrize(
    "field,value", [("head_branch", "feature"), ("workflow_id", 100), ("path", ".github/workflows/fake.yml")]
)
def test_title_alone_cannot_authorize_status(control, publication, field, value):
    publication[0][field] = value
    with pytest.raises(ValueError, match="trusted workflow"):
        control.publish(REPO, {"workflow_run": {"id": 50}})
    assert not publication[3]


def test_cancelled_run_cannot_reuse_successful_finalizer(control, publication):
    publication[0]["conclusion"] = "cancelled"
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert publication[3][0][1]["state"] == "failure"


def test_pagination_reads_every_job(control, monkeypatch):
    calls = []

    def request(path):
        calls.append(path)
        return {
            "jobs": [{"name": "matrix leg"}] * 100
            if path.endswith("&page=1")
            else [{"name": "a5-result", "conclusion": "failure"}]
        }

    monkeypatch.setattr(control, "api", request)
    result = control.pages("repos/owner/repo/actions/runs/50/jobs?filter=latest", "jobs")
    assert len(result) == 101
    assert result[-1]["name"] == "a5-result"
    assert len(calls) == 2


def test_forged_merge_with_correct_parents_is_rejected(control, pr):
    with pytest.raises(ValueError, match="current test merge"):
        control.verify_merge(REPO, pr, HEAD, BASE, "d" * 40)


def test_new_request_on_another_base_supersedes_old_run(control, publication):
    run, _, candidates, writes = publication
    newer = copy.deepcopy(run)
    newer.update(id=51, run_number=9, display_title=f"CI a5 PR #123 @{HEAD} base:{'d' * 40}")
    candidates.append(newer)
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert not writes


def test_publication_history_is_bounded_by_creation_time(control, monkeypatch):
    calls = []

    def request(path):
        calls.append(path)
        assert "&branch=main" in path
        assert "&created=>=2026-11-05T01:00:00Z&per_page=100" in path
        if path.endswith("page=1"):
            # Unrelated runs can fill pages. Do not mistake them for requests
            # for this PR, or assume a short history that fits in one page.
            return {"workflow_runs": [{"display_title": "Manual main", "run_number": 90}] * 100}
        assert path.endswith("page=2")
        return {
            "workflow_runs": [
                {
                    "display_title": f"CI a5 PR #123 @{HEAD} base:{BASE}",
                    "run_number": 10,
                    "workflow_id": 99,
                    "path": control.WORKFLOW,
                    "head_branch": "main",
                }
            ]
        }

    monkeypatch.setattr(control, "api", request)
    assert control.newer_request_exists(
        REPO,
        123,
        HEAD,
        {"created_at": "2026-11-05T01:00:00Z", "run_number": 8, "workflow_id": 99, "path": control.WORKFLOW},
    )
    assert len(calls) == 2


@pytest.mark.parametrize("publication_scan", [False, True])
def test_filtered_run_search_cap_cannot_silently_truncate(control, monkeypatch, publication_scan):
    calls = []

    def request(path):
        calls.append(path)
        assert len(calls) <= 10, "The API cannot return more than 1,000 filtered runs"
        return {"workflow_runs": [{"display_title": "Manual main", "run_number": 90}] * 100}

    monkeypatch.setattr(control, "api", request)
    with pytest.raises(ValueError, match="pagination limit"):
        if publication_scan:
            control.newer_request_exists(
                REPO,
                123,
                HEAD,
                {
                    "created_at": "2026-11-05T01:00:00Z",
                    "run_number": 8,
                    "workflow_id": 99,
                    "path": control.WORKFLOW,
                },
            )
        else:
            control.pages("repos/owner/repo/actions/workflows/ci-a5-comment.yml/runs?status=queued", "workflow_runs")
    assert len(calls) == 10


@pytest.mark.parametrize("run_status", ["in_progress", "completed"])
def test_capped_scan_preserves_existing_status(control, monkeypatch, publication, run_status):
    run, _, _, writes = publication
    run.update(status=run_status, conclusion="failure")

    def capped(*args):
        raise ValueError("Recent CI requests exceeded the pagination limit")

    def latest_status(path, key):
        assert path == f"repos/{REPO}/commits/{HEAD}/status"
        assert key == "statuses"
        return [
            {
                "context": "ci/run/a5",
                "state": "success",
                "target_url": f"https://github.com/{REPO}/actions/runs/50/attempts/1",
            }
        ]

    monkeypatch.setattr(control, "newer_request_exists", capped)
    monkeypatch.setattr(control, "pages", latest_status)
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert not writes


@pytest.mark.parametrize("candidate_number", [7, 9])
def test_capped_scan_cannot_overwrite_another_binding(control, monkeypatch, publication, candidate_number):
    run, _, _, writes = publication
    candidate = copy.deepcopy(run)
    candidate.update(id=51, run_number=candidate_number)
    original_api = control.api

    def capped(*args):
        raise ValueError("Recent CI requests exceeded the pagination limit")

    def request(path, payload=None):
        if path.endswith("/actions/runs/51"):
            return candidate
        return original_api(path, payload)

    monkeypatch.setattr(control, "newer_request_exists", capped)
    monkeypatch.setattr(control, "api", request)
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [{"context": "ci/run/a5", "target_url": f"https://github.com/{REPO}/actions/runs/51/attempts/1"}],
    )
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert not writes


@pytest.mark.parametrize("status_lookup_fails", [False, True])
def test_metadata_failure_without_a_readable_binding_preserves_status(
    control, monkeypatch, publication, status_lookup_fails
):
    def request(path, key):
        if key == "statuses" and not status_lookup_fails:
            return []
        raise subprocess.CalledProcessError(1, ["gh", "api"], stderr="API unavailable")

    monkeypatch.setattr(control, "pages", request)
    if status_lookup_fails:
        with pytest.raises(subprocess.CalledProcessError):
            control.publish(REPO, {"workflow_run": {"id": 50}})
    else:
        control.publish(REPO, {"workflow_run": {"id": 50}})
    assert not publication[3]


@pytest.mark.parametrize("lookup_path", ["/pulls/123", "/workflows/ci-a5-comment.yml"])
@pytest.mark.parametrize("run_status", ["in_progress", "completed"])
@pytest.mark.parametrize("lookup_error", ["http", "json"])
def test_gate_lookup_failure_invalidates_previous_success(
    control, monkeypatch, publication, run_status, lookup_error, lookup_path
):
    run, _, _, writes = publication
    run.update(status=run_status, conclusion="failure")
    original_api = control.api

    def request(path, payload=None):
        if path.endswith(lookup_path):
            if lookup_error == "http":
                raise subprocess.CalledProcessError(1, ["gh", "api"], stderr="Metadata lookup unavailable")
            raise ValueError("Invalid metadata JSON")
        return original_api(path, payload)

    def latest_status(path, key):
        assert path == f"repos/{REPO}/commits/{HEAD}/status"
        assert key == "statuses"
        return [
            {
                "context": "ci/run/a5",
                "state": "success",
                "target_url": f"https://github.com/{REPO}/actions/runs/50/attempts/1",
            }
        ]

    monkeypatch.setattr(control, "api", request)
    monkeypatch.setattr(control, "pages", latest_status)
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert len(writes) == 1
    assert writes[0][0] == f"repos/{REPO}/statuses/{HEAD}"
    assert writes[0][1]["state"] == "failure"


@pytest.mark.parametrize("lookup_path", ["/pulls/123", "/workflows/ci-a5-comment.yml"])
def test_gate_lookup_failure_preserves_verified_newer_status(control, monkeypatch, publication, lookup_path):
    run, _, _, writes = publication
    candidate = copy.deepcopy(run)
    candidate.update(id=51, run_number=9)
    original_api = control.api

    def request(path, payload=None):
        if path.endswith(lookup_path):
            raise subprocess.CalledProcessError(1, ["gh", "api"], stderr="Metadata lookup unavailable")
        if path.endswith("/actions/runs/51"):
            return candidate
        return original_api(path, payload)

    monkeypatch.setattr(control, "api", request)
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [{"context": "ci/run/a5", "target_url": f"https://github.com/{REPO}/actions/runs/51/attempts/1"}],
    )
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert not writes


@pytest.mark.parametrize("cli_error", [False, True])
def test_expected_errors_reach_summary_and_exit_nonzero(control, monkeypatch, tmp_path, cli_error):
    event_file, summary = tmp_path / "event.json", tmp_path / "summary"
    event_file.write_text(json.dumps({}))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_file))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    monkeypatch.setenv("GH_TOKEN", "secret-token")
    output_file = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setattr(control.sys, "argv", ["ci_control.py", "dispatch"])

    def fail(*args):
        if cli_error:
            raise subprocess.CalledProcessError(1, ["gh", "api"], stderr="Forbidden <detail> secret-token")
        raise ValueError("PR changed; post /run a5 again for the new revision")

    monkeypatch.setattr(control, "dispatch", fail)
    with pytest.raises(SystemExit) as error:
        control.main()
    assert error.value.code == 1
    text = summary.read_text()
    public_error = json.loads(output_file.read_text().strip().split("=", 1)[1])
    assert "secret-token" not in public_error
    assert "dispatch failed:" in text
    if cli_error:
        assert "Forbidden &lt;detail&gt; [REDACTED]" in text
        assert "secret-token" not in text
        assert "GitHub API validation or dispatch failed" in public_error
    else:
        assert "PR changed" in text
        assert "PR changed" in public_error


@pytest.mark.parametrize("reason", [None, "PR has merge conflicts; update the branch", "<unsafe>\nsecond line"])
@pytest.mark.parametrize("previous_author", [None, "github-actions[bot]", "external-user"])
def test_failed_command_feedback_links_logs_and_deduplicates_its_own_attempt(
    control, monkeypatch, event, reason, previous_author
):
    monkeypatch.setenv("GITHUB_RUN_ID", "50")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setenv("CI_COMMAND_ERROR", json.dumps(reason) if reason is not None else "")
    marker = "<!-- ci-command-feedback:50:2 -->"
    existing = [] if previous_author is None else [{"user": {"login": previous_author}, "body": marker}]
    monkeypatch.setattr(control, "pages", lambda path: existing)
    writes = []
    monkeypatch.setattr(control, "api", lambda path, payload: writes.append((path, payload)))
    control.command_feedback(REPO, event)
    if previous_author == "github-actions[bot]":
        assert not writes
    else:
        assert len(writes) == 1 and writes[0][0] == f"repos/{REPO}/issues/123/comments"
        body = writes[0][1]["body"]
        assert marker in body and f"https://github.com/{REPO}/actions/runs/50/attempts/2" in body
        assert "could not be completed" in body
        assert "<unsafe>" not in body
        if reason == "<unsafe>\nsecond line":
            assert "&lt;unsafe&gt;\nsecond line" in body


@pytest.mark.parametrize("invalid", ["edited", "issue", "other-command"])
def test_feedback_ignores_non_commands(control, monkeypatch, event, invalid):
    if invalid == "edited":
        event["action"] = "edited"
    elif invalid == "issue":
        del event["issue"]["pull_request"]
    else:
        event["comment"]["body"] = "/run a5 extra"
    monkeypatch.setattr(control, "pages", lambda *args: pytest.fail("No comment reads"))
    control.command_feedback(REPO, event)


@pytest.fixture
def publication_event(publication):
    run = copy.deepcopy(publication[0])
    # Even a delayed successful completion must only invalidate on lookup failure.
    run.update(run_attempt=1, status="completed", conclusion="success")
    return {
        "action": "completed",
        "repository": copy.deepcopy(run["repository"]),
        "workflow": {"id": run["workflow_id"], "path": run["path"]},
        "workflow_run": run,
    }


@pytest.mark.parametrize("marker", [" (request 900-1)", " (request 900-2)"])
@pytest.mark.parametrize("lookup_fails", [False, True])
def test_unbound_request_marker_survives_old_publisher(
    control, monkeypatch, publication, publication_event, lookup_fails, marker
):
    original_api = control.api

    def request(path, payload=None):
        if lookup_fails and path.endswith("/actions/runs/50"):
            raise subprocess.CalledProcessError(1, ["gh", "api"])
        return original_api(path, payload)

    monkeypatch.setattr(control, "api", request)
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [
            {
                "context": "ci/run/a5",
                "state": "pending",
                "description": f"A5 CI requested; waiting for dispatch{marker}",
                "target_url": f"https://github.com/{REPO}/pull/123",
            }
        ],
    )
    # No replacement run exists yet (including a failed/lost dispatch).
    control.publish(REPO, publication_event)
    assert not publication[3]


@pytest.mark.parametrize("failure", ["bind", "ack", "lost_response"])
@pytest.mark.parametrize("result", ["success", "failure"])
def test_accepted_full_recovers_unbound_request(control, monkeypatch, pr, event, publication, failure, result):
    run, jobs, _, writes = publication
    run["conclusion"] = jobs[0]["conclusion"] = result
    gate = {}
    original_api, original_pages = control.api, control.pages
    monkeypatch.setattr(control, "validated_request", lambda *args: (pr, BASE))
    monkeypatch.setattr(control, "already_running_or_passed", lambda *args: False)
    monkeypatch.setattr(control, "unfinished_request_exists", lambda *args: False)
    monkeypatch.setattr(control, "verify_merge", lambda *args: BASE)

    def api(path, payload=None):
        if payload is not None and path.endswith(f"/statuses/{HEAD}"):
            if payload["description"] == "A5 CI dispatched; waiting for results":
                assert failure == "bind"
                raise subprocess.CalledProcessError(1, ["gh", "api"])
            gate.update(payload)
        if payload is not None and path.endswith("ci-a5-comment.yml/dispatches"):
            run["display_title"] += f" request:{payload['inputs']['request_id']}"
            if failure == "lost_response":
                raise subprocess.CalledProcessError(1, ["gh", "api"])
            return None if failure == "ack" else {"workflow_run_id": 50}
        return original_api(path, payload)

    monkeypatch.setattr(control, "api", api)
    monkeypatch.setattr(control, "pages", lambda path, key: [gate] if key == "statuses" else original_pages(path, key))
    with pytest.raises(subprocess.CalledProcessError if failure == "bind" else ValueError):
        control.dispatch(REPO, event)
    assert gate["state"] == "pending" and "request 900-1" in gate["description"]
    # Completion and explicit reconciliation use the same publication path.
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert gate["state"] == result
    assert gate["target_url"].endswith("/runs/50/attempts/2")
    # A new command attempt must still exclude this now-completed source run.
    gate.update(
        state="pending",
        description="A5 CI requested; waiting for dispatch (request 900-2)",
        target_url=f"https://github.com/{REPO}/pull/123",
    )
    before = len(writes)
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert len(writes) == before and gate["state"] == "pending"


def test_bound_replacement_survives_stale_run_history(control, monkeypatch, publication):
    run, _, _, writes = publication
    newer = copy.deepcopy(run)
    newer.update(id=51, run_number=9, run_attempt=1)
    original_api = control.api

    def request(path, payload=None):
        if path.endswith("/actions/runs/51"):
            return newer
        return original_api(path, payload)

    monkeypatch.setattr(control, "api", request)
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [
            {
                "context": "ci/run/a5",
                "state": "pending",
                "target_url": f"https://github.com/{REPO}/actions/runs/51/attempts/1",
            }
        ],
    )
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert not writes


def test_shared_head_preserves_newer_owner_from_another_pr(control, monkeypatch, publication):
    run = publication[0]
    newer = {
        **copy.deepcopy(run),
        "id": 51,
        "run_number": 9,
        "run_attempt": 1,
        "display_title": f"CI a5 PR #124 @{HEAD} base:{BASE}",
    }
    original_api = control.api

    def api(path, payload=None):
        if path.endswith("/runs/51"):
            return newer
        return original_api(path, payload)

    monkeypatch.setattr(control, "api", api)
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [
            {
                "context": "ci/run/a5",
                "state": "pending",
                "target_url": f"https://github.com/{REPO}/actions/runs/51/attempts/1",
            }
        ],
    )
    assert control.newer_status_exists(REPO, 123, HEAD, run)
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [
            {
                "context": "ci/run/a5",
                "state": "pending",
                "description": "A5 CI requested; waiting for dispatch (request 901-1)",
                "target_url": f"https://github.com/{REPO}/pull/124",
            }
        ],
    )
    assert control.newer_status_exists(REPO, 123, HEAD, run)


@pytest.mark.parametrize("action", ["requested", "in_progress", "completed"])
def test_verified_event_routes_all_attempts_and_bases_to_same_head(
    control, monkeypatch, publication_event, tmp_path, action
):
    output_file = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    def lookup(path):
        assert path == (f"repos/{REPO}/actions/runs/50/attempts/{publication_event['workflow_run']['run_attempt']}")
        return copy.deepcopy(publication_event["workflow_run"])

    monkeypatch.setattr(control, "api", lookup)
    publication_event["action"] = action
    control.route(REPO, publication_event)
    publication_event["workflow_run"].update(run_attempt=2, display_title=f"CI a5 PR #123 @{HEAD} base:{'d' * 40}")
    control.route(REPO, publication_event)
    lines = output_file.read_text().splitlines()
    assert lines[::2] == [f"key=a5-{HEAD}"] * 2
    assert all(json.loads(line.removeprefix("event="))["workflow_run"]["id"] == 50 for line in lines[1::2])


@pytest.mark.parametrize("action", ["requested", "in_progress", "completed"])
@pytest.mark.parametrize("lookup_error", ["http", "json"])
def test_live_lookup_failure_invalidates_event_success(
    control, monkeypatch, publication, publication_event, action, lookup_error
):
    publication_event["action"] = action
    original_api = control.api

    def request(path, payload=None):
        if path.endswith("/actions/runs/50"):
            if lookup_error == "http":
                raise subprocess.CalledProcessError(1, ["gh", "api"], stderr="Live run unavailable")
            raise ValueError("Invalid run JSON")
        return original_api(path, payload)

    monkeypatch.setattr(control, "api", request)
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [{"context": "ci/run/a5", "target_url": f"https://github.com/{REPO}/actions/runs/50/attempts/1"}],
    )
    control.publish(REPO, publication_event)
    writes = publication[3]
    assert len(writes) == 1
    assert writes[0][0] == f"repos/{REPO}/statuses/{HEAD}"
    assert writes[0][1]["state"] == "failure"


@pytest.mark.parametrize(
    "path,value",
    [
        (("action",), "unrelated"),
        (("repository", "full_name"), "unrelated/repo"),
        (("repository", "id"), 3),
        (("workflow", "path"), ".github/workflows/fake.yml"),
        (("workflow", "id"), 100),
        (("workflow_run", "path"), ".github/workflows/fake.yml"),
        (("workflow_run", "head_branch"), "feature"),
        (("workflow_run", "event"), "pull_request"),
        (("workflow_run", "repository", "full_name"), "unrelated/repo"),
        (("workflow_run", "head_repository", "id"), 2),
        (("workflow_run", "run_attempt"), 0),
        (("workflow_run", "run_number"), "8"),
        (("workflow_run", "workflow_id"), None),
        (("workflow",), None),
    ],
)
@pytest.mark.parametrize("operation", ["publish", "route"])
def test_untrusted_event_cannot_publish_or_join_status_queue(
    control, monkeypatch, publication, publication_event, path, value, operation, tmp_path
):
    output_file = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    target = publication_event
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    def unavailable(path, payload=None):
        assert payload is None
        assert path.endswith("/actions/runs/50")
        raise subprocess.CalledProcessError(1, ["gh", "api"])

    monkeypatch.setattr(control, "api", unavailable)
    with pytest.raises(control.UntrustedRunError, match="event provenance"):
        getattr(control, operation)(REPO, publication_event)
    assert not publication[3]
    assert not output_file.exists()


@pytest.mark.parametrize("newer_lookup_fails", [False, True])
def test_live_lookup_failure_preserves_only_verified_newer_writer(
    control, monkeypatch, publication, publication_event, newer_lookup_fails
):
    candidate = copy.deepcopy(publication[0])
    candidate.update(id=51, run_number=9)
    original_api = control.api

    def request(path, payload=None):
        if path.endswith("/actions/runs/50") or (newer_lookup_fails and path.endswith("/actions/runs/51")):
            raise subprocess.CalledProcessError(1, ["gh", "api"])
        if path.endswith("/actions/runs/51"):
            return candidate
        return original_api(path, payload)

    monkeypatch.setattr(control, "api", request)
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [{"context": "ci/run/a5", "target_url": f"https://github.com/{REPO}/actions/runs/51/attempts/1"}],
    )
    control.publish(REPO, publication_event)
    assert not publication[3]


@pytest.mark.parametrize("action", ["requested", "in_progress", "completed"])
@pytest.mark.parametrize("state", ["pending", "success", "failure"])
def test_delayed_event_preserves_verified_later_attempt(
    control, monkeypatch, publication, publication_event, action, state
):
    publication_event["action"] = action
    candidate = copy.deepcopy(publication[0])
    candidate.update(status="in_progress" if state == "pending" else "completed", conclusion=state)
    calls = []
    original_api = control.api

    def request(path, payload=None):
        calls.append(path)
        if path.endswith("/actions/runs/50"):
            raise subprocess.CalledProcessError(1, ["gh", "api"])
        if path.endswith("/actions/runs/50/attempts/2"):
            return candidate
        return original_api(path, payload)

    monkeypatch.setattr(control, "api", request)
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [
            {
                "context": "ci/run/a5",
                "state": state,
                "target_url": f"https://github.com/{REPO}/actions/runs/50/attempts/2",
            }
        ],
    )
    control.publish(REPO, publication_event)
    assert f"repos/{REPO}/actions/runs/50/attempts/2" not in calls
    assert not publication[3]


@pytest.mark.parametrize("suffix", ["", "/attempts/1", "/attempts/2", "/attempts/invalid"])
def test_same_or_older_attempt_url_cannot_preserve_success(control, monkeypatch, publication, suffix):
    run = publication[0]  # The live attempt is 2.
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [{"context": "ci/run/a5", "target_url": f"https://github.com/{REPO}/actions/runs/50{suffix}"}],
    )
    monkeypatch.setattr(control, "api", lambda *args: pytest.fail("No newer attempt to verify"))
    assert not control.newer_status_exists(REPO, 123, HEAD, run)


@pytest.mark.parametrize("unfinished", [False, True])
def test_duplicate_scan_splits_large_history_without_losing_old_reruns(control, monkeypatch, unfinished):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(seconds=1100)
    title = f"CI a5 PR #123 @{HEAD} base:{BASE}"
    calls = []
    runs = [
        {
            "display_title": title,
            "head_branch": "main",
            "event": "workflow_dispatch",
            "path": control.WORKFLOW,
            "status": "requested" if unfinished and second == 0 else "completed",
            "created_at": f"{start + timedelta(seconds=second):%Y-%m-%dT%H:%M:%SZ}",
        }
        for second in range(1101)
    ]

    def request(path):
        query = parse_qs(urlsplit(path).query)
        assert "status" not in query
        lower, upper = (
            datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            for v in query["created"][0].split("..")
        )
        page = int(query["page"][0])
        assert 1 <= page <= 10
        calls.append((lower, upper, page))
        # A rerun created at the oldest boundary can change state between pages.
        # Its membership must not depend on which status was queried first.
        if unfinished and len(calls) > 1:
            runs[0]["status"] = "queued"
        matches = [
            run
            for run in reversed(runs)
            if f"{lower:%Y-%m-%dT%H:%M:%SZ}" <= run["created_at"] <= f"{upper:%Y-%m-%dT%H:%M:%SZ}"
        ]
        return {"workflow_runs": matches[(page - 1) * 100 : page * 100]}

    monkeypatch.setattr(control, "api", request)
    assert control.unfinished_request_exists(REPO, title, start, end) is unfinished
    windows = {(lower, upper) for lower, upper, _ in calls}
    middle = start + timedelta(seconds=550)
    assert windows == {(start, end), (start, middle), (middle + timedelta(seconds=1), end)}


def test_duplicate_scan_rejects_unsplittable_cap(control, monkeypatch):
    point = datetime(2026, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(control, "api", lambda *args: {"workflow_runs": [{"display_title": "Other"}] * 100})
    with pytest.raises(ValueError, match="cannot verify duplicates"):
        control.unfinished_request_exists(REPO, "Full request", point, point)


def test_duplicate_scan_propagates_api_failure(control, monkeypatch):
    point = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def unavailable(*args):
        raise subprocess.CalledProcessError(1, ["gh", "api"])

    monkeypatch.setattr(control, "api", unavailable)
    with pytest.raises(subprocess.CalledProcessError):
        control.unfinished_request_exists(REPO, "Full request", point, point)


def test_effective_jobs_preserves_successful_dependencies(control, monkeypatch):
    attempts = {
        1: [
            {"name": "pre-commit", "id": 1, "conclusion": "success"},
            {"name": "clang-tidy", "id": 2, "conclusion": "failure"},
        ],
        2: [{"name": "clang-tidy", "id": 3, "conclusion": "success"}],
    }

    def pages(path, key):
        assert key == "jobs"
        number = int(path.split("/attempts/")[1].split("/")[0])
        return attempts[number]

    monkeypatch.setattr(control, "pages", pages)
    jobs = control.effective_jobs(REPO, {"id": 40, "run_attempt": 2})
    assert {job["name"]: job["id"] for job in jobs} == {"pre-commit": 1, "clang-tidy": 3}


@pytest.mark.parametrize("title", [None, "", "CI", f"CI a5 PR #456 @{'d' * 40} base:{BASE}"])
def test_router_resolves_webhook_title_from_exact_api_attempt(
    control, monkeypatch, publication, publication_event, tmp_path, title
):
    target = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    source = publication_event["workflow_run"]
    actual = copy.deepcopy(source)
    if title is None:
        source.pop("display_title")
    else:
        source["display_title"] = title

    def api(path):
        assert path == f"repos/{REPO}/actions/runs/50/attempts/1"
        return actual

    monkeypatch.setattr(control, "api", api)
    control.route(REPO, publication_event)
    outputs = dict(line.split("=", 1) for line in target.read_text().splitlines())
    assert outputs["key"] == f"a5-{HEAD}"
    routed = json.loads(outputs["event"])
    assert routed["workflow_run"]["display_title"] == actual["display_title"]
    assert control.verified_event_run(REPO, routed)["run_attempt"] == 1


@pytest.mark.parametrize("field", ["id", "workflow_id", "run_number", "run_attempt"])
def test_router_rejects_api_identity_mismatch(control, monkeypatch, publication_event, tmp_path, field):
    target = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    actual = copy.deepcopy(publication_event["workflow_run"])
    actual[field] += 1
    monkeypatch.setattr(control, "api", lambda path: actual)
    with pytest.raises(control.UntrustedRunError):
        control.route(REPO, publication_event)
    assert not target.exists()


def test_router_ignores_manual_main_ci_without_pr_inputs(control, monkeypatch, publication_event, tmp_path):
    target = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    actual = {**publication_event["workflow_run"], "display_title": "CI workflow_dispatch main"}
    monkeypatch.setattr(control, "api", lambda path: actual)
    control.route(REPO, publication_event)
    assert not target.exists()


def test_reconciliation_publishes_existing_result_without_dispatching(control, monkeypatch, publication, tmp_path):
    target = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    event = {"inputs": {"run_id": "50"}}
    control.route(REPO, event)
    outputs = dict(line.split("=", 1) for line in target.read_text().splitlines())
    assert outputs["key"] == f"a5-{HEAD}"
    monkeypatch.setenv("CI_ROUTED_EVENT", outputs["event"])
    control.publish(REPO, event)
    assert len(publication[3]) == 1
    path, payload = publication[3][0]
    assert path == f"repos/{REPO}/statuses/{HEAD}"
    assert payload["state"] == "success"
    assert payload["target_url"].endswith("/runs/50/attempts/2")


@pytest.mark.parametrize("ref", ["refs/heads/feature", "refs/tags/main", ""])
def test_reconciliation_requires_main(control, monkeypatch, ref):
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", ref)
    with pytest.raises(control.UntrustedRunError, match="must run from main"):
        control.route(REPO, {"inputs": {"run_id": "50"}})


@pytest.mark.parametrize("run_id", ["", "0", "-1", "1/attempts/2", "1\n", 50])
def test_reconciliation_rejects_invalid_run_id(control, monkeypatch, run_id):
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    with pytest.raises(ValueError, match="positive workflow run ID"):
        control.route(REPO, {"inputs": {"run_id": run_id}})


def test_routed_identity_invalidates_result_on_api_failure(control, monkeypatch, publication, tmp_path):
    target = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    event = {"inputs": {"run_id": "50"}}
    control.route(REPO, event)
    outputs = dict(line.split("=", 1) for line in target.read_text().splitlines())
    monkeypatch.setenv("CI_ROUTED_EVENT", outputs["event"])

    def unavailable(path, payload=None):
        if payload is not None:
            publication[3].append((path, payload))
            return None
        assert path.endswith("/actions/runs/50")
        raise subprocess.CalledProcessError(1, ["gh", "api"])

    monkeypatch.setattr(control, "api", unavailable)
    control.publish(REPO, event)
    assert publication[3][0][1]["state"] == "failure"


@pytest.mark.parametrize("action", ["in_progress", "completed"])
def test_failed_route_reaches_failure_only_publisher(
    control, monkeypatch, publication, publication_event, tmp_path, action
):
    target = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    publication_event["action"] = action
    publication_event["workflow_run"]["run_attempt"] = 2
    reads = []

    def api(path, payload=None):
        if payload is not None:
            publication[3].append((path, payload))
            return None
        reads.append(path)
        assert path == f"repos/{REPO}/actions/runs/50/attempts/2"
        raise subprocess.CalledProcessError(1, ["gh", "api"])

    monkeypatch.setattr(control, "api", api)
    control.route(REPO, publication_event)
    outputs = dict(line.split("=", 1) for line in target.read_text().splitlines())
    assert outputs["key"] == f"a5-{HEAD}"
    assert json.loads(outputs["event"])["routing_failed"] is True
    monkeypatch.setenv("CI_ROUTED_EVENT", outputs["event"])
    control.publish(REPO, publication_event)
    assert len(reads) == 1, "A failed route cannot be upgraded to success by a later lookup"
    assert len(publication[3]) == 1
    path, payload = publication[3][0]
    assert path == f"repos/{REPO}/statuses/{HEAD}"
    assert payload["state"] == "failure"
    assert payload["target_url"].endswith("/runs/50/attempts/2")


@pytest.mark.parametrize("newer", ["run", "attempt", "unbound"])
def test_failed_route_preserves_newer_status_owner(
    control, monkeypatch, publication, publication_event, tmp_path, newer
):
    target = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    candidate = copy.deepcopy(publication[0])
    if newer == "run":
        candidate.update(id=51, run_number=9)
    status = {
        "context": "ci/run/a5",
        "state": "success",
        "target_url": f"https://github.com/{REPO}/actions/runs/{candidate['id']}/attempts/2",
    }
    if newer == "unbound":
        status.update(
            state="pending",
            description="A5 CI requested; waiting for dispatch (request 900-2)",
            target_url=f"https://github.com/{REPO}/pull/123",
        )

    def api(path, payload=None):
        assert payload is None, "A delayed failure-only route must preserve the newer writer"
        if path == f"repos/{REPO}/actions/runs/50/attempts/1":
            raise subprocess.CalledProcessError(1, ["gh", "api"])
        expected = "51" if newer == "run" else "50/attempts/2"
        assert path == f"repos/{REPO}/actions/runs/{expected}"
        return candidate

    monkeypatch.setattr(control, "api", api)
    monkeypatch.setattr(control, "pages", lambda *args: [status])
    control.route(REPO, publication_event)
    outputs = dict(line.split("=", 1) for line in target.read_text().splitlines())
    monkeypatch.setenv("CI_ROUTED_EVENT", outputs["event"])
    control.publish(REPO, publication_event)


@pytest.mark.parametrize("title", [None, "", "CI"])
def test_failed_route_cannot_invent_pr_identity(control, monkeypatch, publication_event, tmp_path, title):
    target = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    if title is None:
        publication_event["workflow_run"].pop("display_title")
    else:
        publication_event["workflow_run"]["display_title"] = title

    def unavailable(path):
        raise subprocess.CalledProcessError(1, ["gh", "api"])

    monkeypatch.setattr(control, "api", unavailable)
    with pytest.raises(control.UntrustedRunError):
        control.route(REPO, publication_event)
    assert not target.exists()


def test_progress_resolution_keeps_api_failure_for_retry(control, monkeypatch, publication_event):
    def unavailable(path):
        raise subprocess.CalledProcessError(1, ["gh", "api"])

    monkeypatch.setattr(control, "api", unavailable)
    with pytest.raises(subprocess.CalledProcessError):
        control.resolved_event(REPO, publication_event)


@pytest.mark.parametrize("same_base", [True, False])
@pytest.mark.parametrize("phase", ["queued", "in_progress"])
def test_running_request_reused_only_for_admitted_base(control, pr, publication, same_base, phase):
    run = publication[0]
    run["status"] = phase
    assert control.already_running_or_passed(REPO, pr, BASE if same_base else "d" * 40) is same_base


def test_denied_manual_dispatch_cannot_supersede_or_publish(control, monkeypatch, publication, admission_check):
    accepted, jobs, candidates, writes = publication
    denied = {**accepted, "id": 51, "run_number": 9, "conclusion": "failure"}
    candidates.append(denied)
    original_pages, original_api = control.pages, control.api

    def pages(path, key):
        if key == "jobs":
            # Attempt two only reran the finalizer. Admission is from attempt one.
            if path.endswith("/attempts/1/jobs"):
                result = "failure" if "/runs/51/" in path else "success"
                return [{"name": "prepare", "steps": [{"name": "Authorize A5 dispatch", "conclusion": result}]}]
            return jobs
        return original_pages(path, key)

    def api(path, payload=None):
        if path.endswith("/runs/51"):
            return denied
        return original_api(path, payload)

    monkeypatch.setattr(control, "api", api)
    monkeypatch.setattr(control, "pages", pages)
    monkeypatch.setattr(control, "run_admitted", admission_check)
    control.publish(REPO, {"workflow_run": {"id": 51}})
    assert not writes
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert [payload["state"] for _, payload in writes] == ["success"]
