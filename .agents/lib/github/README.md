# GitHub Shared Procedures

Reusable procedures for repository GitHub workflows.

## Procedures

- [`setup.md`](setup.md): authenticate, detect canonical repo, remotes, base ref, push target, and state.
- [`lookup-pr.md`](lookup-pr.md): find a PR by number, branch, upstream tracking, or user selection.
- [`detect-permission.md`](detect-permission.md): determine whether the current user can push to a PR branch.
- [`checkout-fork-branch.md`](checkout-fork-branch.md): create a local work branch for cross-fork PR edits.
- [`commit-and-push.md`](commit-and-push.md): rebase, squash to one commit when required, and push.
- [`fetch-comments.md`](fetch-comments.md): fetch unresolved review threads.
- [`reply-and-resolve.md`](reply-and-resolve.md): reply to review comments and resolve threads.
- [`branch-naming.md`](branch-naming.md): derive branch names from commit subjects.
- [`common-issues.md`](common-issues.md): common `gh` and shell pitfalls.

## Standard Variables

The GitHub workflow skills use these shell variables after setup:

- `DEFAULT_BRANCH`: canonical default branch, usually `main`.
- `BASE_REF`: fetched base ref, usually `origin/main` for owners or `upstream/main` for forks.
- `PUSH_REMOTE`: remote to push the current branch to.
- `PR_REPO_OWNER` / `PR_REPO_NAME`: repository where PRs and issues are created.
- `PR_HEAD_PREFIX`: empty for owner branches, `user:` for fork PR heads.
- `BRANCH_NAME`: current local branch or push refspec.
- `COMMITS_AHEAD`: commits ahead of `BASE_REF`.
- `UNCOMMITTED`: porcelain git status.
