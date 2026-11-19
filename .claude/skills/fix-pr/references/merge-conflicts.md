# Resolve a merge or rebase conflict

Use this when `git rebase "$BASE_REF"` or a merge stops with conflicts while
updating a PR. The goal is to preserve both sides' intent, not to make the
conflict markers disappear.

## 1. See the current state

```bash
git status                              # operation in progress, unmerged files
git diff --name-only --diff-filter=U    # only the conflicted files
git diff -- <file>                      # the conflict hunks
```

To see the commits behind the two sides, use the form that matches the
operation. A stopped rebase has no `MERGE_HEAD`, so `git log --merge` fails
there with `fatal: --merge without MERGE_HEAD?`:

```bash
# rebase: REBASE_HEAD is the commit being replayed
git show --oneline --stat REBASE_HEAD
# commits already replayed onto the base; empty at the first stop
git log --oneline "$BASE_REF"..HEAD

# merge only
git log --merge --oneline
```

Identify which side is the base and which is the PR's commit before editing.
In a rebase the roles are inverted from what the markers suggest: `ours` is the
base you are replaying onto, and `theirs` is your own commit.

## 2. Find the primary sources

For each conflict, read the commit messages, PR description and comments, and
the linked Issue that produced each side. Use `git log -p -- <file>`,
`git blame`, and `git show <sha> -- <file>`. State the intent of each side
before choosing a resolution.

## 3. Resolve each hunk

- Preserve both intents where possible.
- If they are incompatible, stop and ask the maintainer which intent wins.
  Dropping the base's side discards a change that is already merged — often a
  fix another PR depends on — and recording a trade-off does not preserve it.
  Present both intents and your recommendation; resolve on your own only when
  one side is plainly the other's superset.
- Do not invent new behavior in a conflict resolution.
- Do not blindly use `--ours` or `--theirs`; either can silently drop a fix.
- Generated trees (`build/`, `build/lib/`, `build/cache/`, `outputs/`) are
  regenerated, not hand-merged. Resolve the producer, not its output.
- If `pto_isa.pin` conflicts, choose one pin deliberately and rebuild the
  affected onboard runtime; do not keep both values.

## 4. Finish the operation

Format any touched C++ before staging, then continue. A rebase stops once per
conflicting commit, so repeat steps 1–4 until it reports success:

```bash
git add <files>
git rebase --continue        # or: git merge --continue
```

## 5. Verify the result

Only now does `HEAD` contain the resolution — during a stopped rebase it does
not, so a diff taken before step 4 is empty or misleading. Read the **full**
combined diff, not `--stat`: a reverted hunk inside an otherwise-changed file
does not show up in per-file counts.

```bash
git diff "$BASE_REF"...HEAD          # every line that lands on the base
git diff "$BASE_REF"...HEAD --stat   # file list, as a cross-check only
```

Confirm no unrelated file was reverted, then run the scope-appropriate checks
from [testing](../../testing/SKILL.md), including the runtime rebuild decision
when C++ changed, and the repository's pre-commit hooks.

Keep the PR at one commit. Fold any repair into the existing commit per
[commit-and-push](../../../lib/github/commit-and-push.md) §2, then re-run the
PR's checks and confirm the diff still matches the stated goal.

Do not `git rebase --abort` merely because the conflict is hard. Abort only
when the rebase target itself is wrong or the branch state is corrupted, and
record why in the PR or Issue.
