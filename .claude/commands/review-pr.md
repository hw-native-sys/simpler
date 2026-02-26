Review the current pull request or staged changes.

1. If a PR number is provided via $ARGUMENTS, use `gh pr diff $ARGUMENTS` to get the diff
2. Otherwise, use `git diff main...HEAD` to review all changes on the current branch
3. Read `.ai-instructions/coding/codestyle.md` for code style rules
4. For each changed file, check:
   - Adherence to code style (enum class, volatile, offsetof conventions)
   - No plan-specific comments (Phase 1, Step 1, Gap #3, etc.)
   - No hardcoded absolute paths or private information
   - Kernel sources match the expected directory layout
   - golden.py functions follow the required signature
5. Summarize findings as:
   - **Issues** (must fix before merge)
   - **Suggestions** (optional improvements)
   - **Looks good** (files with no issues)
