# Working Discipline

Repo-agnostic behavioral guidelines that complement the built-in simplicity
and surgical-edit rules already in the system prompt. These target gaps the
system prompt does not cover.

## 1. State assumptions on ambiguous requests

If a task has multiple reasonable interpretations, name them and pick one
explicitly, or ask. Do not silently choose one and implement it — when the
silent choice turns out wrong, the rework costs more than the clarification
would have.

- "Add validation" → ask which inputs and what failure mode (raise? return
  an error? log?) before writing any code.
- "Refactor X" → ask what the success condition is (same tests pass?
  benchmark unchanged? API preserved?).
- "Make it faster" → ask what "fast enough" means and how to measure.

## 2. Match local style when editing

Follow the conventions of the file you are touching, even if you would write
it differently elsewhere. Do not reformat, rename, or refactor adjacent code
that is not part of the request.

- If surrounding code uses snake_case and yours uses camelCase, match the
  file.
- If unrelated dead code is next to your edit, mention it — do not delete
  it as a drive-by.
- Every changed line should trace directly back to the user's request.

## 3. Bug fixes start with a failing repro

For any defect that can be reproduced in code, write a test that fails
against the current behavior first, confirm it fails, then make it pass.
This guarantees the fix actually addresses the reported bug and leaves a
regression barrier behind.

Exceptions: infrastructure/build breakage where a test is impractical, or
one-character typo fixes where the diff itself is the verification.

## 4. Check `docs/investigations/` before proposing optimizations or refactors

The repo records considered-and-dropped proposals in
`docs/investigations/`. Before suggesting a non-trivial optimization,
refactor, or design change, grep that folder for the subsystem and the
mechanism you're about to propose. If an entry already shut the idea
down, either:

- Adopt that verdict (and tell the user the proposal was previously
  rejected, with the link), or
- If new information changes the calculus, update the existing entry
  with the new measurement instead of opening a parallel discussion.

When an investigation you ran ends with "we didn't do it" — measured
no signal, found a blocking constraint, decided the cost outweighs the
benefit — write a new entry there **and add it to the index in
`docs/investigations/README.md`** before closing the session. An
unlinked entry is invisible to the next person; the index is the only
discovery surface. Future-you will re-derive the same conclusion
otherwise.

## 5. Every CI failure on your PR is yours to triage

A red check on your PR is not "done" until you have read the actual
failure log and named the real cause. CI health is a shared
responsibility of every PR and every developer — a failing pipeline is
everyone's problem, not just the author of the change that happens to
have caused it.

- **Read the log, don't guess.** Pull the failing job's output
  (`gh run view --job "$JOB_ID" --log-failed`), find the actual failing test
  or error line, and state the root cause. The job name and a hunch are
  not a diagnosis.
- **"Unrelated to my change" is a conclusion, not a default.** Earn it:
  show your diff doesn't touch the failing path, and ideally that the
  same failure reproduces on `main` or a sibling job. Only then call it
  pre-existing.
- **Unrelated and flaky still aren't "ignore."** Re-run it; if it
  persists, flag it to the maintainers, file/append an issue, and link
  the run. Never leave a red PR with a silent "it's flaky" — that is how
  a real regression hides behind assumed flakiness.
- **A PR is not green-and-done while any required check is red,**
  regardless of whose change caused it.

This is the converse of the `running-onboard.md` anti-pattern (don't read
"ci passed" as proof a fix worked): equally, don't wave away "ci failed"
as someone else's problem.
