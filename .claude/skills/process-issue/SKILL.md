---
name: process-issue
description: Process a GitHub Issue as a maintainer, from triage and ownership routing through dependencies and evidence-based closure. Use to manage an Issue's next step; use fix-issue for implementation.
---

# Process an Issue as maintainer

Accept an Issue number or URL, or an explicitly scoped set of Issues. Resume
from current GitHub state instead of repeating triage on every invocation.
GitHub is the business record; do not register a duplicate task in another queue.

## Read the current situation

Use [GitHub setup](../../lib/github/setup.md) to identify repository and account.
Write access is not authorization to change ownership, publish messages or close
Issues: stay within the requested maintenance scope and reuse authorization already
given. For an analysis-only request, recommend actions without writing to GitHub.

Read the body, comments, assignees, labels, parent/sub-Issues, explicit blocking
relations and linked PRs, including drafts. Paginate connections when necessary.
Check current main and related open or recently merged PRs before assigning new
implementation. Compare scope and evidence, not just matching titles.

Use current repository ownership guidance, CODEOWNERS if present, and the relevant
track's latest body. The simpler
[work overview](https://github.com/hw-native-sys/simpler/issues/2670) is a starting
point for module routing; follow any replacement it names. Verify current GitHub
logins and assignability rather than copying a historical roster into this skill.

## Choose the next action

| Current situation | Maintainer action |
| ----------------- | ----------------- |
| New, sufficiently clear | Establish scope and completion criteria, route to an owner and link the track |
| Missing essential information | Ask for the specific missing reproduction, expected behavior or decision; identify unaffected work |
| Duplicate or already implemented | Link the canonical Issue or merged implementation, explain matching scope; close only when justified and authorized |
| Assigned and active | Preserve ownership; inspect progress and blockers rather than starting another implementation |
| Delivery PR exists | Follow current review/CI state using [process-pr](../process-pr/SKILL.md) |
| Candidate completion | Check all acceptance conditions and agreed integration or baseline validation before closure |
| Closed | Explain the resolution; reopen or start follow-up work only within the requested scope |

An Issue needs a useful goal, supported scope and observable completion criteria,
not a complete technical design before assignment. Put confirmed design and progress
in the body; keep design discussion in comments. Use existing labels and conventions.

## Route responsibility and relationships

- Keep one accountable assignee. Route an unassigned Issue first to the module
  owner, who may transfer it to the actual implementer. Preserve existing ownership.
  Multiple existing assignees require clarifying accountability, not arbitrarily
  removing people. If the module or owner is unclear, use the overview's fallback.
- If assignment fails, report routing as pending; do not claim success or substitute
  an unrelated account.
- Split sub-Issues when independently deliverable parts need different owners.
  One owner's internal checklist or agent DAG normally remains inside that Issue.
  Use [create-issue](../create-issue/SKILL.md) for authorized child creation after
  checking existing children and overlapping work.
- Parent/sub-Issue links mean decomposition. Blocking links mean a real prerequisite.
  A shared track does not establish execution order. Prefer native GitHub relations;
  if unavailable, record explicit links and their meaning without claiming native
  relations were created.
- Hand implementation to the assignee's [fix-issue](../fix-issue/SKILL.md) workflow.
  Do not acquire execution rights or launch a coding agent on their behalf merely
  because you triaged the Issue.

Different users coordinate through visible ownership and discussion, without a
mandatory global execution lock. Explicitly requested independent investigations
may overlap; record scopes and the integration owner. Optional WudiTask coordination
concerns one user's machines, not team ownership.

## Update and close carefully

Before writing, reread the latest body and relevant comments; preserve others'
changes and edit only agreed fields or sections. Use a body file or structured API
arguments for multiline text. Verify the result. On an ambiguous API outcome, read
back before retrying; do not duplicate comments, children or assignments.

Distinguish ready PRs, merged PRs and accepted Issue completion. One merged child
does not finish its parent track. Keep remaining deliverables, dependencies and
responsibility visible. If closing a duplicate, completed or declined Issue, record
the reason and evidence within the authorized maintenance scope.

When process-pr hands back merge evidence, reconcile acceptance here rather than
calling it again for the same PR state. Report actual changes, remaining blockers,
responsible person and next action. Avoid no-change status comments.
