# Independent review

Use only when the invocation opts in with `codex`, `claude`, or `subagents`.
The main reviewer owns the complete review and verifies all findings.
Record unavailable or skipped reviewers and the reason. Ignore a CLI token
that names the current harness and record it as skipped.

## Brief and boundary

Give each reviewer the pinned base/head, exact diff range, requested path
scope, and requirement sources. Keep the main reviewer's analysis out of the
brief. Pass the range and let the reviewer read the diff from Git. Use
`git show <head>:<path>` for context that differs from the checkout.

Require a read-only execution boundary. Work includes reading the diff and
necessary callers. Builds, tests, hardware, checkout, edits, publication, and
additional reviewers require their own authorization.

Ask for up to ten findings with severity, `file:line`, evidence level, and
a concrete consequence. Each Must-fix needs a trigger, violated contract, or
budget breach. Request an explicit no-findings result if applicable.

## External CLIs

Check binary availability and use a task-specific scratch directory with
separate output files. Pass the same brief to each opted-in reviewer and
run independent reviewers in parallel where supported. Read captured output
directly or check its size during execution. Diagnose a stalled process and
stop that process if necessary. Record errors or incomplete output.

For Codex CLI, use its read-only sandbox:

```bash
codex exec -s read-only "$(cat "$PROMPT_FILE")" >"$CODEX_OUT" 2>&1
```

For Claude CLI, use the read-only tool allowlist and suppress user and project
settings. Ensure the host enforces the read-only boundary for the process.

```bash
claude -p \
  --setting-sources "" \
  --tools "Read,Grep,Glob,Bash" \
  --allowedTools "Read,Grep,Glob,Bash(git diff:*),Bash(git show:*),Bash(git log:*),Bash(git grep:*)" \
  --permission-mode dontAsk \
  "$(cat "$PROMPT_FILE")" >"$CLAUDE_OUT" 2>&1
```

## Native subagents

With the `subagents` option, run two independent read-only reviews in parallel
if the host supports that boundary:

- **Standards:** Check the changed lines against the repository rules,
  developer guide, and testing guide. Report the rule and the impact.
- **Spec:** Check the user goal, linked Issue or design, and the PR claims
  against the diff. Report missing, partial, extra, and incorrect behavior.

## Synthesize the results

Verify each finding at the reviewed revisions. Identify the changed behavior
that introduces or worsens it. For a missing bugfix test, check that the bugfix
belongs to the range. For a race, identify the concurrent paths and specific
interleaving. For a leak, trace allocation and the missing release. For design
or performance, verify the criterion and impact. Recheck disagreements in the
source. Agreement from the same model family provides limited independence.

Add verified findings to the main severity-ranked report. In Appendix C, name
each reviewer, note availability and relevant model overlap, and list findings
that were confirmed or dropped with a reason. Keep Spec and Standards coverage
independently visible.
