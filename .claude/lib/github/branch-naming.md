# Branch Naming

Generate branch name from commit message.

## Rules

1. Determine the commit type prefix:

| Commit type | Branch prefix |
| ----------- | ------------- |
| Add | `feat/` |
| Fix | `fix/` |
| Update | `feat/` |
| Refactor | `refactor/` |
| Support | `support/` |
| Sim | `sim/` |
| CI | `support/` |

2. Take the commit subject description (after `Type: `), lowercase it, replace spaces and special characters with hyphens, strip trailing hyphens.

3. Truncate to 50 characters.

## Examples

**Short commit (no truncation):**
`Refactor: inline ring buffer hot paths` → `refactor/inline-ring-buffer-hot-paths` (37 chars)

**Long commit (with truncation):**
`Support: add complete-phase poll hit-rate logging with detailed statistics` → `support/add-complete-phase-poll-hit-rate-logging-w` (50 chars)
