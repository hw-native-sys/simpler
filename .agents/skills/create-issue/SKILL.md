---
name: create-issue
description: Create a GitHub issue with duplicate checking and repository-appropriate labels/body. Use when filing bugs, features, documentation tasks, or code-health issues.
---

# Create GitHub Issue

Create an issue in the current repository using `gh`.

## Workflow

1. Run `gh auth status`; stop and ask the user to authenticate if it fails.
2. Determine the target repo with `gh repo view --json owner,name`.
3. Check for duplicates:
   - Search open issues with relevant keywords.
   - Deep-read up to three likely candidates.
   - If exact duplicate exists, report it and stop.
   - If related issues exist, reference them in the new body.
4. Classify the issue as bug, feature, performance, documentation, or code health.
5. If `.github/ISSUE_TEMPLATE/` exists, follow the closest template and fill every required field.
6. If no templates exist, create a concise Markdown issue body with:
   - Summary
   - Context
   - Expected behavior or desired outcome
   - Actual behavior or current gap, when applicable
   - Reproduction or validation notes, when applicable
   - Related issues, when applicable
7. Create the issue with `gh issue create --title ... --body ...` and labels when available.
8. Report the created issue URL.

## Repository Notes

This repository currently may not have issue templates. Do not invent unavailable template fields;
use the generic Markdown body when templates are absent.

## Constraints

- Do not create an issue when an exact duplicate exists.
- Ask the user for missing high-impact facts instead of fabricating them.
- Use concrete file paths, commands, and observed behavior when available.
