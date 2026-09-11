# AGENTS Guide

**EVERY AI AGENT MUST FOLLOW THIS GUIDE BEFORE ANY WORK.**

## Required startup sequence

1. Read `CLAUDE.md` before running commands, analyzing code, or editing files.
2. Treat `CLAUDE.md` as the source of truth for role boundaries, architecture context, and repository workflow.
3. Load always-on conventions from `.claude/rules/` (for example: architecture, codestyle, device constraints).
4. Load only task-relevant workflows from `.claude/skills/`.

## Additional rules

- If `CLAUDE.md` changes, read it again before continuing.
- If relevant files under `.claude/rules/` or `.claude/skills/` change, refresh your context before proceeding.
- If user instructions conflict with repository conventions, prioritize user intent for that task.
- Higher-priority system/developer/user instructions override this guide.

## HBG kernel-mode development

- Use the user's supplied `kernel-mode-design.md` and its [v9 final decisions](https://icc.gt.tc/vllm-pto#v9-design) for HBG kernel-mode architecture. Final decisions in §0 override historical alternatives, except where the user explicitly supersedes them. Preserve the current K1 ABI unless an explicit ABI change is requested.
- The development baseline is [PR #2064](https://github.com/hw-native-sys/simpler/pull/2064/).
- The user supersedes the v9 two-stream decision: kernel mode uses three distinct streams (borrowed caller/vLLM Ascend, dedicated non-hidden AICPU, hidden AICore), ordered by event record/wait. Keep both execution stream resource kinds; neither may alias caller.
- Keep each HBG milestone (H1, 2B, H2 and subsequent items) in its own independently buildable commit, including its tests and documentation. Commit shared context resource lifecycle changes separately so later PRs can follow dependency order.
