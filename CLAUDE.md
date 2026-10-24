# Developer Guidelines

See [docs/developer-guide.md](docs/developer-guide.md) for full directory structure, compilation pipeline, and conventions.

## Directory Ownership

| Role | Working directory |
| ---- | ----------------- |
| Platform Developer | `src/{arch}/platform/` |
| Runtime Developer | `src/{arch}/runtime/` |
| Codegen Developer | `examples/` |

## Common Commands

See [docs/testing.md](docs/testing.md) for the full testing guide (st, pyut, cpput) and [docs/ci.md](docs/ci.md) for CI pipeline details.

### Python environment

This repo requires `pip install .` to run. **Always use a project-local venv created with `--system-site-packages`** — never install into the user/global site. Applies equally to git worktrees (create `.venv` inside the worktree). See [.claude/rules/venv-isolation.md](.claude/rules/venv-isolation.md).

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install .
```

### Format C++ code

```bash
clang-format -i <file>
```

## Important Rules

1. **Consult `.claude/rules/` for conventions and working rules** ([ascend](.claude/rules/ascend.md), [ci-change-detection](.claude/rules/ci-change-detection.md), [codestyle](.claude/rules/codestyle.md), [comments](.claude/rules/comments.md), [discipline](.claude/rules/discipline.md), [doc-consistency](.claude/rules/doc-consistency.md), [env-macro-gating](.claude/rules/env-macro-gating.md), [known-issues-tracking](.claude/rules/known-issues-tracking.md), [project-layout](.claude/rules/project-layout.md), [running-onboard](.claude/rules/running-onboard.md), [venv-isolation](.claude/rules/venv-isolation.md)) — these are always-loaded guidelines. **Consult `.claude/skills/` for task-specific workflows** (e.g., `git-commit/` when committing, `testing/` when running tests)
2. **Do not modify directories outside your assigned area** unless the user explicitly requests it
3. Create new subdirectories under your assigned directory as needed
4. When in doubt, ask the user before making changes to other areas
5. **Avoid including private information in documentation or code** such as usernames, absolute paths with usernames, or other personally identifiable information. Use relative paths or generic placeholders instead

## Substantial task delivery

Use [.claude/skills/develop-task/SKILL.md](.claude/skills/develop-task/SKILL.md)
for goal/design alignment, internal task dependencies, parallel agents and
reviewed delivery. Small changes can use the focused skills directly.

## GitHub work by role

Use [maintainer](.claude/skills/maintainer/SKILL.md) to maintain a selected repository,
Issue, track or personal work scope. Developers and committers can use it too;
scope does not grant extra authority. It discovers changes and routes to the
focused workflows below. Recurring wakeups require separately configured scheduling.

- Maintainer: [process-issue](.claude/skills/process-issue/SKILL.md) for Issue
  triage, ownership and closure; [process-pr](.claude/skills/process-pr/SKILL.md)
  for review routing, merge readiness and authorized merge/Issue follow-through.
- Assignee: [fix-issue](.claude/skills/fix-issue/SKILL.md), which selects direct
  repair or `develop-task` according to the work's complexity.
- PR author: [github-pr](.claude/skills/github-pr/SKILL.md) to publish and
  [fix-pr](.claude/skills/fix-pr/SKILL.md) for feedback and CI repairs.
- Reviewer: [review-pr](.claude/skills/review-pr/SKILL.md) for technical review.

Issues carry an accountable assignee; PRs carry their author and requested reviewers.
These workflows share GitHub records, not a second task queue.
