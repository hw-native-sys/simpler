---
name: weekly-changelog
description: Summarize user-facing changes merged in the current Friday-anchored week (most recent Friday up through yesterday) in the simpler repo into a markdown changelog with before/after code examples. Use when the user asks for a weekly changelog, weekly summary, or weekly external changes report.
---

# Weekly Changelog

Generate a focused, example-driven changelog of **user-facing** changes merged
in the **current Friday-anchored window**: from the most recent Friday strictly
before today, up through **yesterday**. The team's release cadence runs
Friday→Thursday, so on Friday morning this yields the just-ended Fri→Thu week;
mid-week it yields a partial-week running report. Write the output to
`WEEKLY_CHANGES_<YYYY-MM-DD>.md` in the project root, where the date is the
**Friday** that started the window.

**Output language.** The rendered report is **English** — same convention
as the rest of the repo. PR titles, identifiers, file paths, and code
snippets stay in their original form (they already are English).

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
render literally. Use the headings literally — no translation step.

````markdown
# simpler weekly external changes (YYYY-MM-DD ~ YYYY-MM-DD)

This document presents core interface and feature changes via example
comparisons; see each PR for details.

---

## I. Interface changes

### N. <short title> — [#NNN](https://github.com/.../pull/NNN)

<one-line motivation>

```diff
- <old code>
+ <new code>
```

<necessary constraints / caveats, 1-3 lines>

---

## II. New features

### N. <feature name> — [#NNN](https://github.com/.../pull/NNN)

```<lang>
<minimum callable example: Python API / orch C++ / shell command>
```

<necessary supplement, 1-3 lines>

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
- **Concise prose.** One sentence of "why" is the cap.
- **Bug-fix rows are user-symptom only.** Each row in section III must
  describe what a user would have *observed* before the fix — a log line,
  an error code, a wrong output, a hang. **Do not** describe the internal
  cause ("missing finalize call", "wrong header order"). If you cannot
  phrase the user symptom in one line, the PR is internal and belongs in
  the excluded list, not in section III.

## 6. Output

### 6a. The md file

Write the file to `<repo_root>/WEEKLY_CHANGES_<Friday>.md` where `<Friday>`
is the Friday that started the window — i.e. `$START` from section 1, NOT
the end date. The same Friday-anchored filename is reused throughout the
Fri→Thu cycle so mid-week re-runs overwrite in place and the file grows
into a complete weekly report by the time the cycle closes. Overwrite if
it already exists.

The md file contains **only** the three sections from §4 (Interface
changes / New features / User-visible bug fixes). Do **not** write the
excluded-PR list, the window range, or any meta commentary into the md —
those belong in the chat reply, not the file.

### 6b. The chat reply

Separately, report back to the user in the chat reply (not in the md):

- the file path
- the window range (`$START` ~ `$END`, noting partial-week if `$END` is
  not yet a Thursday)
- counts: N interface changes / N new features / N bug fixes
- the excluded PR numbers, **aggregated by category** so the user can scan
  them at a glance instead of reading a flat list:

  ```text
  Excluded as internal:
  - Pure refactors: #NNN #NNN ...
  - Test / CI infra: #NNN #NNN ...
  - Example-only fixes / docs: #NNN #NNN ...
  - Skill / tooling: #NNN
  ```
