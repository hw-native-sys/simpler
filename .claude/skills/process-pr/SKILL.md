---
name: process-pr
description: Process a GitHub PR as a maintainer, routing review, following revisions, checking merge readiness and reconciling linked Issues. Use for PR lifecycle management; use review-pr for technical review and fix-pr for author-side repairs.
---

# Process a PR as maintainer

Accept a PR number or URL, or an explicitly scoped set of PRs. Take the next
justified action from its current state. The PR author owns delivery; reviewers
review it. Do not require a PR assignee or use the committer as its owner.

When called by [maintainer](../maintainer/SKILL.md), preserve its selected write
scope, authority and observation context. Linked items outside that scope remain
read-only; return the result to the caller without starting another maintenance loop.
Developers may use this workflow for their own scope without acquiring maintainer
permissions or taking over other contributors' work.

## Establish scope and authority

Use [GitHub setup](../../lib/github/setup.md). Read the PR body, author, base and
head SHA, draft/state, requested reviews, checks and mergeability. Read linked
Issues and agreed design, plus all three feedback surfaces via
[fetch-comments](../../lib/github/fetch-comments.md): inline threads, review
summaries and conversation comments. Paginate when needed.

Check current main and related open/draft PRs for duplicates, dependencies and
integration conflicts. A small self-contained PR may carry its own goal without
a separate Issue. A linked PR is delivery under its Issue, not a duplicate task.

Respect actual authority and scope. An analysis/review request does not authorize
publication or merging. A request to manage the PR may authorize routine routing
and updates, but merging, closing without merge, dismissing reviews and overriding
checks require explicit authorization for those actions. Reuse authorization
already given; never infer it from repository permissions.

## Route the next step

| Current situation | Maintainer action |
| ----------------- | ----------------- |
| Draft or unclear scope | Identify missing implementation, explanation or evidence; keep incomplete work out of Ready |
| Ready for technical review | Request module reviewers within scope, or use [review-pr](../review-pr/SKILL.md) when asked to review |
| Feedback, failed CI or conflict | Route findings to the author for [fix-pr](../fix-pr/SKILL.md); keep unresolved work visible |
| Author pushed changes | Review changed implementation and reconcile prior findings and checks at the new head |
| Acceptance gates satisfied | Report Ready at the exact head; merge only if explicitly authorized |
| Merged | Verify target branch and merge commit, reconcile linked Issues and agreed follow-up validation |
| Closed without merge | Record the reason and successor if any; do not mark linked Issues implemented |

Use current ownership guidance and the relevant track to choose reviewers. Multiple
module reviewers are reasonable; the author integrates their feedback. Requesting
a reviewer does not make them the PR author.

Do not silently repair the author's branch. An explicitly authorized maintainer
repair uses `fix-pr` after agreeing the writing scope and checking the latest remote
head and active work. Push access alone does not establish a safe handoff. Independent
read-only reviews do not need a writer lease.

## Establish merge readiness

Ready is a conclusion about a particular PR head, not a permanent label:

- The agreed useful feature and acceptance are satisfied, with limitations stated.
- Technical review covers current changes; prior approvals and resolved threads
  are reconciled after updates. An agent's favorable analysis is not a GitHub
  approval from a required reviewer.
- Required checks completed successfully according to repository rules. Inspect
  failed job logs; pending, cancelled, missing or unexplained skipped checks are not
  evidence of success. Do not bypass checks to make a PR Ready.
- Blocking findings across all three feedback surfaces are addressed. Disputed
  findings remain pending until the reviewer or authorized maintainer accepts their
  disposition. Resolved threads alone do not clear a changes-requested review.
- Dependencies are met and the PR can merge into its actual target branch.

If the head changes, reconcile the new diff and relevant checks before declaring
Ready again. Recheck head, reviews, checks and mergeability immediately before an
authorized merge; use the hosting tool's expected-head guard where available.
Do not merge a different head, silently enable auto-merge, dismiss reviews or override
protection. Read GitHub before retrying an ambiguous merge response.

## Reconcile business progress

Verify actual merged state, target and merge commit. Update linked Issues through
[process-issue](../process-issue/SKILL.md) only within authorized scope, passing merge
evidence rather than restarting PR review. Keep merged and baseline-validated
separate when post-merge acceptance was agreed. Partial delivery updates parent
progress; it does not close the whole track.

Use closing links only for Issues fully resolved by this PR; use ordinary related
links for partial delivery. If automatic closure conflicts with outstanding agreed
acceptance, reconcile explicitly rather than treating closure as proof. Reread
remote bodies before updates, preserve others' changes and verify the result.
Publish substantive changes, not repeated no-change status comments.

Finish with current head/state, readiness evidence or blockers, and the next actor.
Awaiting author, reviewer or human merge is a valid handoff, not a reason to take
over their task or claim autonomous monitoring.
