# Review, publication and continuation

## Before publishing

Review the integrated diff and acceptance for the complete feature. Apply the
repository testing and commit workflows; do not create redundant tests for plain
wording edits or claim hardware validation from a host-only check. Keep one final
commit as required by `github-pr`; child commits can be integrated locally before
that final publication. Inspect exact current base and overlapping open/draft PRs
again. Preserve other contributors' work and report any unresolved integration.

Use the existing `github-pr` skill for an authorized PR. Prepare a concrete title,
body, checks and limitations. Design approval is not blanket permission to post
messages, close Issues or merge. Reuse explicit publication authority when given;
otherwise finish the reviewable local result before requesting that final action.
Reread mutable remote bodies before updating them, preserve others' newer edits,
and verify the published result. If a push/create request has an ambiguous response,
look up remote refs/PRs before retrying. Do not create duplicate PRs or empty commits.

## Feedback, CI and Ready

Use `fix-pr` to reconcile all three feedback surfaces: inline threads, review
summary bodies and conversation comments. Record which feedback is fixed,
explained or still pending, and explicitly assign authorized reply/resolve work.
A patch alone is not a published reply or resolved thread. Reread the remote state.

Read failed CI logs and the relevant workflow/job/attempt/step, including cleanup.
Distinguish a test failure, provisioning failure and cancellation. A later green
retry does not establish the original root cause. Follow the current task's retry
limits and keep counts across resumes; never invent fresh retry permission by
changing a head or starting another agent. Diagnose repeated failures instead of
blind reruns. Use existing `testing`/`fix-pr` rules to select focused checks.

Ready means the complete agreed feature has been reviewed, feedback handled and
required checks passed at the **current exact PR head**. A new head withdraws old
Ready until the changed implementation and relevant checks have been reconciled.
Keep pending decisions separate from Ready PRs. The local HTML is a snapshot;
verify GitHub before telling the user it is ready to merge. A consistent local
record does not establish the truth of its external evidence.

## Merge and handoff

The user controls PR merge unless they explicitly delegate it. After a merge
notification, verify GitHub's target branch, merged head/commit and any required
baseline acceptance. `merged` and `baseline_verified` are separate facts. Continue
already-authorized successors after their actual dependencies pass; do not ask for
an already-approved scope again or call the whole parent Issue complete because
one PR landed. Follow-up code still uses the repository PR and CI workflow.

A cross-repository handoff includes the usable commit/version, API/call examples,
validation commands and results, unsupported scopes, remaining dependencies and
responsible owner. Business progress belongs in the Issue body; substantive design
discussion belongs in comments when authorized. Do not publish private execution
logs or maintain a competing completion ledger.

## Resume checkpoint

Keep a concise checkpoint beside the plan with:

- Current agreement, accepted design and explicit user limits.
- Base/PR head and last observed GitHub state.
- Active authors, attempts, commands and uncertain launch/publication outcomes.
- Candidate artifacts awaiting main-agent review and remaining feedback.
- Evidence for tests/review, pending decisions and merge/baseline waits.
- Exact next action and whether automatic wake is verified or manual resume is needed.

On resume, read that checkpoint plus new user input, recheck enabled personal
coordination, inspect live/unknown operations, consume results, then continue.
Never blindly repeat a spawn, CI rerun, push or PR creation after losing context.
