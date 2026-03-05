# GitHub Shared Procedures

Reusable procedures for GitHub PR workflows.

## Available Procedures

| Procedure | Description | Used By |
| --------- | ----------- | ------- |
| [setup](setup.md) | Authenticate and detect repository context | All |
| [lookup-pr](lookup-pr.md) | Find PR by number, branch, or list all | All |
| [detect-permission](detect-permission.md) | Check push access to PR | address-pr-comments |
| [commit-and-push](commit-and-push.md) | Squash commits, rebase, and push | github-pr, address-pr-comments |
| [fetch-comments](fetch-comments.md) | Get unresolved PR review comments | address-pr-comments |
| [reply-and-resolve](reply-and-resolve.md) | Reply to and resolve review threads | address-pr-comments |
| [branch-naming](branch-naming.md) | Generate branch name from commit | github-pr |
| [common-issues](common-issues.md) | Troubleshooting reference | All |

## Standard Variables

After running `detect-context`, these variables are available:

| Variable | Description | Example (owner) | Example (fork) |
| -------- | ----------- | --------------- | -------------- |
| `REPO_OWNER` | Origin repo owner | `ChaoWao` | `contributor` |
| `REPO_NAME` | Origin repo name | `simpler` | `simpler` |
| `PR_REPO_OWNER` | PR target owner | `ChaoWao` | `ChaoWao` |
| `PR_REPO_NAME` | PR target name | `simpler` | `simpler` |
| `DEFAULT_BRANCH` | Base branch name | `main` | `main` |
| `BASE_REF` | Full base ref | `origin/main` | `upstream/main` |
| `PUSH_REMOTE` | Where to push | `origin` | `origin` |
| `PR_HEAD_PREFIX` | PR head prefix | *(empty)* | `contributor:` |
| `ROLE` | Repository role | `owner` | `fork` |
| `BRANCH_NAME` | Current branch | `feat/new` | `feat/new` |
| `COMMITS_AHEAD` | Commits ahead of base | `1` | `1` |
| `UNCOMMITTED` | Uncommitted changes | *(empty or files)* | *(empty or files)* |
