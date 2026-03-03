# AGENTS Guide

**EVERY AI AGENT MUST FOLLOW THIS GUIDE BEFORE ANY WORK.**

## Required startup sequence

1. Read `CLAUDE.md` before running commands, analyzing code, or editing files.
2. Treat `CLAUDE.md` as the source of truth for role boundaries, architecture context, and repository workflow.
3. Load always-on conventions from `.claude/rules/` (for example: architecture, codestyle, device constraints).
4. Load only task-relevant workflows from `.claude/skills/` and `.claude/commands/`.

## Additional rules

- If `CLAUDE.md` changes, read it again before continuing.
- If relevant files under `.claude/rules/`, `.claude/skills/`, or `.claude/commands/` change, refresh your context before proceeding.
- If user instructions conflict with repository conventions, prioritize user intent for that task.
- Higher-priority system/developer/user instructions override this guide.
