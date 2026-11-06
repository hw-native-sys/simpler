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

## 6. Never force-take a branch another worktree holds

This repo is worked through many `git worktree` checkouts at once, several of
them driven by concurrent agent sessions; the count changes by the hour, so
check rather than assume. A branch ref is **shared by all of them**; `HEAD`, the
index, and the working tree are per worktree. So `git checkout -B <name>` does
not just move "your" branch: if any other worktree has `<name>` checked out, you
have taken it out from under that worktree — and because its `HEAD` is a symref
to that shared ref, it follows the move while its index and files do not.

**Only ever create a new branch, or reset a branch you exclusively own.**

```bash
git fetch upstream
git checkout --detach upstream/main            # a clean base that owns no branch
git checkout -b <task-branch> upstream/main    # new branch — -b fails loudly on a name clash
```

Between tasks, park the worktree on a **detached HEAD**, not on a shared
branch. Reading merged content needs no checkout at all:

```bash
git show upstream/main:path/to/file
git log upstream/main --oneline -1
git diff upstream/main -- path
```

Keep `upstream/main` and `main` distinct in your head. The first is a
remote-tracking ref that `fetch` maintains and that costs nothing to read; the
second is a local branch some worktree owns.

### Why `-B` is the trap

Git already refuses the obvious forms — both are rejected when another worktree
holds the branch:

```text
git checkout main            → fatal: 'main' is already checked out at '...'
git branch -f main <ref>     → fatal: cannot force update the branch 'main' checked out at '...'
```

**`git checkout -B main <ref>` is not refused** (observed on git 2.39.1). It
creates the dual-checkout state those two guards exist to prevent, and from
then on every further force-move is legal, because moving a branch *your own*
worktree has checked out is an ordinary reset. The guard never fires again.

### What it does to the other worktree

That worktree's HEAD is a symref to the branch, so it silently follows the ref
to the new commit — while its **index and files stay at the old one**. Since
"staged" is by definition the HEAD-to-index diff, `git status` there reports the
**net tree difference between the two commits, in reverse**, as staged
modifications: a file you added shows up as a staged *deletion*. Changes that
later commits undid cancel out and never appear, so the list is the net delta,
not one entry per intervening commit.

It is not cosmetic. While that index still matches the old tree, a `git commit`
there would land a commit reverting the whole net difference, and it would look
entirely routine to whoever ran it.

### Self-check

```bash
git worktree list        # each branch name must appear at most once
```

Any branch listed against two worktrees is already the broken state; move your
own worktree off it with `git checkout --detach HEAD` before doing anything
else, and leave the other worktree's copy alone.

## 7. CI correctness tests use explicit events

Every CI correctness test — unit, simulation and onboard hardware — must
establish the state or event ordering it asserts. Use explicit handshakes,
barriers, acknowledgements or observable state transitions to reach the
scenario and verify its functional result.

Do not use sleeps, calibrated spin counts, elapsed observation windows or
relative execution speed to infer concurrency, ordering, blocking or successful
fault coverage. For example, "the successor prepared before a one-second
predecessor finished" depends on machine speed; hold the predecessor at an
observable gate, verify the successor's state, then release the gate. A fault
test must confirm the injected event occurred at the required state, rather
than waiting long enough for background interference to become likely.

Timeouts bound waits and fail a stuck test; their expiry or non-expiry must
not stand in for an unrelated correctness property. When time is itself the
contract under test (a timeout, deadline or configured budget), assert that
contract with tolerance for CI variance and check its observable outcome.
Explicit performance benchmarks are separate from functional correctness tests.

Probabilistic duration-based stress experiments belong in explicitly requested
diagnostics, not required correctness CI. Increasing a delay, timeout or retry
count does not replace deterministic synchronization.

See [the testing assertion principle](../../docs/testing.md#assertion-principle-no-wall-clock-proxies).
