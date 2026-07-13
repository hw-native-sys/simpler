---
name: weekly-changelog
description: Summarize user-facing changes merged in the current Friday-anchored week (most recent Friday up through yesterday) in the simpler repo into a markdown changelog with before/after code examples. Also emits a full all-PR inventory (WEEKLY_ALL_PRS) and Chinese (_zh) translations of both docs. Use when the user asks for a weekly changelog, weekly summary, or weekly external changes report.
---

# Weekly Changelog

Generate a focused, example-driven changelog of **user-facing** changes merged
in the **current Friday-anchored window**: from the most recent Friday strictly
before today, up through **yesterday**. The team's release cadence runs
Friday→Thursday, so on Friday morning this yields the just-ended Fri→Thu week;
mid-week it yields a partial-week running report.

This skill produces **four files** in the project root, all keyed to the
**Friday** that started the window (`$START` from §1):

| File | What | Built in |
| ---- | ---- | -------- |
| `WEEKLY_CHANGES_<Friday>.md` | Curated, user-facing changelog (filtered) | §4 |
| `WEEKLY_ALL_PRS_<Friday>.md` | Full inventory of **every** PR in the window | §6 |
| `WEEKLY_CHANGES_<Friday>_zh.md` | Chinese translation of the changelog | §7c |
| `WEEKLY_ALL_PRS_<Friday>_zh.md` | Chinese translation of the inventory | §7c |

**Output language.** The two canonical docs are authored in **English** (repo
convention / English-only lint). The Chinese `_zh.md` files are a **post-hoc
translation pass** over the finished English docs (§7c) — write English first,
then translate; translation drift is acceptable. In every file, PR titles,
identifiers, file paths, code/diff blocks, and shell commands stay in their
original English — only prose is translated.

## 1. Compute the week range

The window **ends yesterday** (`today - 1 day`) and **starts on the most
recent Friday strictly before today** — i.e. the Friday that anchors the
current Fri→Thu release cycle. Do **not** use "last 7 days" — the window is
anchored to weekdays so the same report can be regenerated deterministically.

```bash
# End = yesterday
END=$(date -d "yesterday" +%Y-%m-%d)
# Start = most recent Friday strictly before today.
# On Friday, the new cycle has just started — go back 7 days to the
# Friday that anchored the just-ended cycle.
if [ "$(date +%u)" = "5" ]; then
    START=$(date -d "7 days ago" +%Y-%m-%d)
else
    START=$(date -d "last friday" +%Y-%m-%d)
fi
echo "$START .. $END"
```

The variable names `START` / `END` replace the old `FRI` / `THU` throughout
section 2 — `END` is no longer guaranteed to be a Thursday on mid-week runs.

## 2. Collect commits and triage in two passes

**Pass A — list and pre-filter (one batched call).** Get the whole commit
list first, then pull just titles for every PR in a single `gh` call so the
network round-trips don't dominate the run:

```bash
git log --since="$START 00:00" --until="$END 23:59" \
        --pretty=format:"%h %ad %s" --date=short

# Extract PR numbers and batch-fetch titles in ONE call via GraphQL
# aliases — `gh pr view` accepts only one PR per call, and
# `gh pr list --search "number:X number:Y"` returns empty (the `number:`
# qualifier does not OR across multiple values). Aliases give a precise,
# single round-trip:
gh api graphql -F owner=<owner> -F name=<repo> -f query='
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    pr<N1>: pullRequest(number: <N1>) { number title }
    pr<N2>: pullRequest(number: <N2>) { number title }
    # ... one alias per PR in the window
  }
}'
```

Apply the keep/skip rules in section 3 against titles alone. Most weeks
roughly half the PRs are pure internals and drop out here without a body
fetch.

**Pass B — deepen on kept PRs only.** For each PR that survived pass A,
fetch body + diff:

```bash
gh pr view <num> --json title,body -q '.title + "\n" + .body'
gh pr diff <num>
```

**Pass C — data for the full inventory (§6).** The curated changelog needs
only kept-PR bodies. The full inventory additionally needs, for **every** PR
in the window: a 1-3 line description (so fetch the *skipped* PRs' bodies too —
trivial ones can be summarized from the title) and a size stat. Batch the
sizes in one GraphQL call (one alias per PR, same shape as Pass A):

```bash
gh api graphql -F owner=<owner> -F name=<repo> -f query='
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    pr<N>: pullRequest(number: <N>) { number additions deletions changedFiles }
    # ... one alias per PR in the window
  }
}'
# render each as `+<additions>/-<deletions>, <changedFiles>f`
```

## 3. Keep / skip rules (user-facing test)

A change is **user-facing** if a user writing `examples/...` or
`tests/st/...` code (Python API or orchestration C++) would have to change
something they wrote, or could observe a new behavior at runtime. Anything
else is internal.

**Skip** (do **not** include in the changelog):

- Pure refactors with no API/behavior change visible to users — even if the
  diff is large
- C++ / C-API internals only platform-backend developers touch
  (e.g. `HostApi::*`, `DeviceRunner::*`, `pto_runtime_c_api.h` symbols)
- Compile-time macros with no external override path
- CI / build infra fixes invisible at runtime
- Scene-test runner / fixture-only fixes
- Anything whose explanation would only mean something to a
  platform-backend or runtime developer

**Include** when it changes any of:

- Python API exposed to users (`Worker`, `ChipWorker`, `Orchestrator`,
  `Arg`, submit helpers, examples)
- C++ orchestration API users call from `kernels/orchestration/*.cpp`
  (`rt_submit_*`, `Arg::*`, `ArgWithDeps`, etc.)
- Runtime behavior users will observe (new timeouts, validation errors,
  diagnostics, env vars, CLI flags)
- New examples, new test entry points, new tools
- **DFX / diagnostic tooling the user runs** — an args-dump level, a
  profiling flag, a new channel/field in a report (`scope_stats`), a new
  track/arrow in a trace (`swimlane`), or corrected report output is
  user-facing *when it adds or changes something the user observes while
  running the tool* (a new `--flag`, a new level, a new resource channel,
  a value that was previously wrong/unreadable). A pure internal refactor of
  the same tool with **no observable change** (e.g. gating a counter that was
  already free, renaming an internal field) stays internal.

For each kept PR, decide which bucket it lands in:

| Bucket | Definition | Section heading |
| ------ | ---------- | --------------- |
| Interface changes | signature/contract changes existing user code must migrate to | `## I. Interface changes` |
| New features | new capabilities users can opt into | `## II. New features` |
| User-visible bug fixes | fixes whose absence users would have hit | `## III. User-visible bug fixes` |

### Group stacked / multi-PR features into ONE entry

If multiple PRs in the window implement one logical feature (a capture
subsystem split across capture + replay + viewer + follow-up fix is the
common shape), emit a **single** entry whose title links every PR number.

- Title shape: `### N. <subsystem> — <short description> — [#NNN](...) [#NNN](...) ...`

A separate entry per PR for the same feature inflates the report and
hides the story. Cross-check before writing: scan kept PRs for shared
subsystem keywords in titles, shared file paths in their diffs, or
explicit "stacked on #NNN" / "follow-up to #NNN" language in the body.

## 4. Document structure

Every run assumes the project has no prior weekly report — render the full
structure below, do not "match prior reports." The skeleton is shown using
four-backtick fences so that the inner triple-backtick example blocks
render literally. Use the headings literally in the English doc — translation,
if any, happens only in the `_zh.md` pass (§7c).

````markdown
# simpler weekly external changes (YYYY-MM-DD ~ YYYY-MM-DD)

This document presents core interface and feature changes via example
comparisons; see each PR for details.

---

## I. Interface changes

### N. <short title> — [#NNN](https://github.com/.../pull/NNN)

**Why:** <the problem this change solves — what broke, was missing, or forced a
migration without it. Give enough context to feel the pain: what the user hit,
under what condition, and why it mattered. When there is more than one distinct
problem, enumerate them `1.` / `2.` so the How can answer each by number.>

**How:** <how this change resolves the Why — in the SAME order and numbering:
point 1 here answers problem 1 above. Name the concrete mechanism (new
argument, changed default, added validation, new contract), not just "it is
fixed". One clause per Why point.>

```diff
- <old code>
+ <new code>
```

<necessary constraints / caveats, 1-3 lines>

---

## II. New features

### N. <feature name> — [#NNN](https://github.com/.../pull/NNN)

**Why:** <the problem this feature solves — what was impossible, broken, or
costly before it. Give enough context to feel the gap: what the user could not
do, or paid for, without it. Enumerate `1.` / `2.` when several drivers
motivate the feature.>

**How:** <how the feature closes the Why — in the SAME order and numbering: the
surface the user calls plus what it does underneath, mapped back to the problem
each part removes. One clause per Why point.>

```<lang>
<minimum callable example: Python API / orch C++ / shell command>
```

<necessary supplement / constraints, 1-3 lines>

---

## III. User-visible bug fixes

| PR          | Fix description                                                |
| ----------- | -------------------------------------------------------------- |
| [#NNN](...) | <one line: user-observable symptom, not the internal cause>    |
````

## 5. Writing style

- **Examples over prose.** Every interface change has a before/after diff
  block from a real example file in the repo (`examples/...` or
  `tests/...`). Pull the snippet directly from the PR diff — do not
  paraphrase.
- **Motivation is mandatory for features and interface changes.** Every entry
  in §I and §II opens with a `**Why:**` that names the concrete problem the
  change solves — a dropped capability, an error code, a perf cost, a missing
  surface. Give enough context that a reader feels the problem: what was hit,
  under what condition, and why it mattered — not a single terse clause. 2-5
  lines, grounded in the PR body; do not skip it and do not pad it to filler.
  If you cannot name a concrete problem, re-check whether the PR is actually
  user-facing (it may belong in the excluded list).
- **Why and How pair one-to-one.** §I and §II entries follow `**Why:**` with a
  `**How:**` that explains how the change resolves it, lining up point-for-point
  with the Why. If Why enumerates `1.` / `2.`, How answers `1.` / `2.` in the
  same order; if Why is a single problem, How is a single matching answer. How
  names the concrete mechanism — the new argument, the changed default, the
  validation added, the contract introduced — so the reader can trace each
  problem to its fix. Do not bury the solution in the caveat line or leave it
  implicit in the diff: the diff shows *what* the code is now, the How says
  *why that resolves the Why*.
- **Concise elsewhere.** Outside `**Why:**` / `**How:**`, keep each supplement
  to 1-3 lines.
- **Bug-fix rows are user-symptom only.** Each §III row is one line, takes no
  `**Why:**`, and describes what a user would have *observed* before the fix —
  a log line, an error code, a wrong output, a hang — **not** the internal
  cause ("missing finalize call", "wrong header order"). If you cannot phrase
  the user symptom in one line, the PR is internal and belongs in the excluded
  list, not §III.

## 6. Full-PR inventory document

Alongside the curated changelog, produce a **full inventory** —
`WEEKLY_ALL_PRS_<Friday>.md` — listing **every** PR merged in the window,
internal and user-facing alike. This is the "what did simpler change this
week, in total" view; the curated changelog is the filtered subset.

Group PRs by **subsystem / theme** (e.g. Platform/AICore, Device recovery,
Scheduler/Runtime, Performance, DFX, Remote-L3, Examples, Build·CI, Docs —
derive the actual themes from the window's PRs; do not hardcode this list).
Lead with a scannable overview table, then one detail entry per PR. Mark a PR
`✓` in the `User-visible` column iff it also appears in the curated changelog,
so the two docs stay cross-referenced (and update the `✓` if the user later
moves a PR in or out of the curated set).

````markdown
# simpler — all merged PRs (YYYY-MM-DD ~ YYYY-MM-DD)

Full inventory of every PR merged in the window (internal + user-facing).
`User-visible` ✓ = also in the curated `WEEKLY_CHANGES_<Friday>.md`.
Size = `+added/-deleted, N files`.

## Overview

| PR | Title | Category | User-visible |
| --- | --- | --- | :---: |
| [#NNN](url) | <short title> | <theme> | ✓ |
| [#NNN](url) | <short title> | <theme> | — |

Total N PRs; M user-visible (#NNN #NNN ...).

## <Theme>

### [#NNN](url) <title> · `+A/-D, Nf` · ✓

<1-3 lines: what changed + why / observable effect; note a2a3/a5 scope if it differs>
````

## 7. Output & Chinese translation

### 7a. Curated changelog (English)

Write `<repo_root>/WEEKLY_CHANGES_<Friday>.md` where `<Friday>` is `$START`
from §1 (NOT the end date). The Friday-anchored filename is reused all cycle,
so mid-week re-runs overwrite in place and the file grows into the full weekly
report by cycle close. Contains **only** the three §4 sections — no excluded
list, no window range, no meta commentary (those go in the chat reply, §7d).

### 7b. Full inventory (English)

Write `<repo_root>/WEEKLY_ALL_PRS_<Friday>.md` per §6 — every PR, overview
table + per-theme detail. Same Friday-anchored, overwrite-in-place rule.

### 7c. Chinese translations

After **both** English docs are final, translate each into a `_zh.md`
companion in the same directory:

- `WEEKLY_CHANGES_<Friday>.md`  → `WEEKLY_CHANGES_<Friday>_zh.md`
- `WEEKLY_ALL_PRS_<Friday>.md`  → `WEEKLY_ALL_PRS_<Friday>_zh.md`

Translate **prose only**, per the Output-language rule in the intro
(identifiers, paths, code/diff, commands, table keys stay English; drift is
fine; author *from* the English, never from scratch). The `_zh.md` files are
local-only and need not pass the repo's English-only lint.

### 7d. Chat reply

Report back in the chat reply (not in any md):

- the four file paths
- the window range (`$START` ~ `$END`, noting partial-week if `$END` is not
  yet a Thursday)
- counts: N interface changes / N new features / N bug fixes (curated), and
  total PRs / user-visible count (inventory)
- the excluded PR numbers, **aggregated by category** so the user can scan
  them at a glance instead of reading a flat list:

  ```text
  Excluded as internal:
  - Pure refactors: #NNN #NNN ...
  - Test / CI infra: #NNN #NNN ...
  - Example-only fixes / docs: #NNN #NNN ...
  - Skill / tooling: #NNN
  ```
