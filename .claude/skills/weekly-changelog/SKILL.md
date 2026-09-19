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
| `WEEKLY_CHANGES_<Friday>.md` | Curated, user-facing changelog (filtered) | §4-5 → [writing-rules.md](writing-rules.md) |
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

The team, the release cadence, and every git commit timestamp are in **China
time (UTC+8)**. The box this skill runs on may be in a **different** timezone
(the dev/CI box has run PDT), so plain `date -d` computes the window in the
box's local time and silently shifts it by the offset — that misfiles the
Fri→Thu boundary and drops or double-counts a day's PRs. **First surface the
offset, then anchor all date math to `Asia/Shanghai`:**

```bash
# Understand the box's timezone vs China's before trusting any date.
echo "box tz: $(date +%Z%z)   china tz: $(TZ='Asia/Shanghai' date +%Z%z)"
export TZ='Asia/Shanghai'   # anchor ALL later date/git-log timestamps to China time
```

With `TZ` exported, the window **ends yesterday** (`today - 1 day`) and
**starts on the most recent Friday strictly before today** — the Friday that
anchors the current Fri→Thu release cycle, all in China time. Do **not** use
"last 7 days" — the window is weekday-anchored so the same report regenerates
deterministically.

```bash
END=$(date -d "yesterday" +%Y-%m-%d)
# Start = most recent Friday strictly before today.
# On Friday, the new cycle has just started — go back 7 days to the
# Friday that anchored the just-ended cycle.
if [ "$(date +%u)" = "5" ]; then
    START=$(date -d "7 days ago" +%Y-%m-%d)
else
    START=$(date -d "last friday" +%Y-%m-%d)
fi
echo "$START .. $END (China time)"
```

The variable names `START` / `END` replace the old `FRI` / `THU` throughout
section 2 — `END` is no longer guaranteed to be a Thursday on mid-week runs.
`$START` (the Friday) still names the output files.

## 2. Collect commits and triage in two passes

**Pass A — list and pre-filter (one batched call).** Get the whole commit
list first, then pull just titles for every PR in a single `gh` call so the
network round-trips don't dominate the run:

```bash
# List every commit whose China-local COMMIT date falls in [START, END].
# Two traps this avoids:
#  - Filter on commit date (%cd with --date=iso-local), NOT author date (%ad):
#    a rebased PR keeps an older author date and would be misfiled by a day/week.
#  - Do NOT use `git log --since/--until`: on non-linear history it prunes the
#    walk at the first out-of-range commit and silently drops in-range PRs.
#    Range-filter the full list in awk instead. (TZ is already Asia/Shanghai.)
#  - Tab-delimit `%cd`/`%s` (not `|`): a subject containing a pipe would else
#    split across awk fields and truncate the title. No commit cap, so an older
#    window past the newest N commits is never silently dropped.
git log --date=iso-local --pretty=format:"%h %cd%x09%s" \
  | awk -F'\t' -v s="$START" -v e="$END" \
      '{split($1,a," "); d=a[2]} d>=s && d<=e {print a[1], d, $2}'

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

**A PR body is a claim, not a source.** It is the author's narrative, written
before review, and a later PR in the same window can invalidate it. Every
statement you write about **what the code looked like before** must be read out
of the tree, not transcribed from the body:

```bash
git show <sha>^:path/to/file.h | grep -n -A15 '<symbol>'   # the before state
git show <sha> --format='' -- path/to/file.cpp             # what actually changed
git show <sha> --format='' --name-only                      # full paths (see the scope line)
```

Four kinds of error this catches, all of them observed:

- **A "pre-existing problem" that the series itself created.** A body describes
  the state its own predecessor left behind. Check the *predecessor's* parent:
  if `git show <earlier-sha>^:<file>` shows the API in its original form, the
  problem belongs to the series, not to the world before it — say so, or describe
  only the state that predates the whole series.
- **Attributing a shared facility to one component.** Grep for where a helper is
  actually defined (`git grep -n 'inline .*<name>'`) before writing "X has it and
  Y does not" — it may live in a header both include.
- **A mechanism described more loosely than it is.** "Matched by address range"
  vs. the code's exact-equality search are different claims, and only the second
  explains why the fix had to change the *criterion* rather than tighten a check.
- **An example the public API does not accept.** Before writing a call-site
  snippet, confirm the entry point's signature (`grep -n 'using .*Function\|static
  inline .* <fn>('`). A capturing lambda against a function-pointer parameter
  does not compile.

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

**When `gh` flakes, fall back to local git — no network.** The `gh api
graphql` / `gh pr view` calls intermittently fail with `Post "...": EOF` or a
timeout. Every datum they return is also recoverable from the local clone,
because merged PRs are squash-commits whose message body **is** the PR
description. Map each PR's merge SHA from the §2 commit list (`%h`), then:

```bash
git show -s --format='%b' <sha>            # PR body (squash-merge description)
git show --stat --format='' <sha> | tail -1  # "N files changed, +A insertions, -D deletions"
```

Retry `gh` once or twice first (the EOF is usually transient); if it keeps
failing, switch to the git commands above rather than blocking the run.

## 3. Keep / skip rules (user-facing test)

A change is **user-facing** if a user writing `examples/...` or
`tests/st/...` code (Python API or orchestration C++) would have to change
something they wrote, or could observe a new behavior at runtime. Anything
else is internal.

**Skip** (do **not** include in the changelog):

- Pure refactors with no API/behavior change visible to users — even if the
  diff is large
- C++ / C-API internals only platform-backend developers touch
  (e.g. `HostApi::*`, `DeviceRunner::*`, `runtime_c_api.h` symbols)
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

**The test is "same reader", not "same feature".** Two PRs that sound like one
story still belong in separate entries when they land in different layers,
because the layer decides who has to act on them:

Match **longest prefix first**, so the rows are read top to bottom and the first
hit wins — `simpler_setup/tools/` is a subpath of `simpler_setup/`, and without
an order a change to a tool would satisfy two rows at once:

| Layer | Path signature (first match wins) | Who must read it |
| ----- | --------------------------------- | ---------------- |
| Offline tooling | `simpler_setup/tools/` | people reading a capture |
| Scene test | `simpler_setup/` (excluding `tools/`), `examples/`, `tests/st/` — **no `src/`** | people writing cases in this repo |
| Runtime / platform | `src/{arch}/runtime/`, `src/common/platform/` | every caller, including other repos |

Merging across that boundary is worse than splitting: a runtime change filed
under a scene-test entry reads as "only applies to scene tests", which is the
opposite of true. When two such PRs genuinely interlock, keep two entries and
give each a scope line ([writing-rules.md](writing-rules.md) §4) plus one
sentence pointing at the other.

The inverse also holds: **two PRs that change the same observable contract
belong together even when their stated goals differ.** A scheduler optimization
that alters which profiling phases get emitted changes the same schema the
profiling PR just defined; filing them apart leaves the reader with two
half-descriptions of one contract. Detect this by diffing the doc: if PR B's
diff of a `docs/` file starts from the blob PR A produced, they are a relay.

```bash
# The `index` line is the SECOND line of a file's diff — `head -1` returns the
# `diff --git` line, which is identical for both PRs and would never match.
git show <shaA> --format='' -- docs/path.md | sed -n '/^index /{p;q}'  # index <old>..<X>
git show <shaB> --format='' -- docs/path.md | sed -n '/^index /{p;q}'  # index <X>..<new>  ← relay
```

## 4-5. Document structure and writing style

**[writing-rules.md](writing-rules.md) holds both, and you must read it before
writing the first entry.** It carries the §I/§II/§III skeleton, the scope line,
the retired/added/wire shape for structural changes, the example and
problem-statement rules, the ~25 line soft cap, and the five-question
readability check to run before shipping.

It lives in its own file for two reasons: this one is loaded in full on every
invocation, and the rules only matter once triage (§1-§3) has already decided
what goes in the report.

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
report by cycle close. Contains **only** the three sections of
[writing-rules.md](writing-rules.md) §4 — no excluded
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

## 8. Publish to the wiki (only when asked)

The default output is the four local files (§7). **When the user asks to
publish**, the running log lives in the GitHub wiki, a **separate git repo**
`hw-native-sys/simpler.wiki.git`. Only the two `_zh.md` docs are published there
(the wiki convention is Chinese-only); the English files stay local.

Clone the wiki into the scratchpad, copy the two `_zh.md` in under their
existing names (`WEEKLY_CHANGES_<Friday>_zh.md` /
`WEEKLY_ALL_PRS_<Friday>_zh.md`), and prepend this week's links to the **top**
of each section in the `simpler-Wiki.md` index (it is newest-first):

```bash
- [WEEKLY_CHANGES_<Friday>_zh (<START> ~ <END>)](WEEKLY_CHANGES_<Friday>_zh)
- [WEEKLY_ALL_PRS_<Friday>_zh (<START> ~ <END>)](WEEKLY_ALL_PRS_<Friday>_zh)
```

**The wiki's Chinese wording is fixed, and it is not what translating the English
produces.** Several headings there are established terms that a fresh
translation of the same English heading will render differently — close enough
to look right, different enough to break the run of pages. So do not re-derive
any of them: read the previous week's two pages and copy each string verbatim,
substituting only the dates.

Six strings drift every run if you skip this (this file cannot quote them —
`tests/lint` enforces English-only source, and the pages are the authority
anyway):

```bash
PREV=<wiki>/WEEKLY_CHANGES_<prev-Friday>_zh.md
PREVALL=<wiki>/WEEKLY_ALL_PRS_<prev-Friday>_zh.md
head -3  "$PREV"      # 1. CHANGES H1 (note its full-width parentheses)  2. intro line
grep -n '^## ' "$PREV"    # 3. the three section headings
head -5  "$PREVALL"       # 4. ALL_PRS H1 (an em-dash *pair*)  5. its three intro lines
grep -n '^| PR |' "$PREVALL"   # 6a. overview table header
# 6b. the count line: it sits INSIDE the first section, after the overview
# table, so anything anchored on the first `## ` returns the intro lines
# instead. Take the first section's non-table prose.
awk '/^## /{n++} n==1 && !/^\||^## |^$/' "$PREVALL"
```

Two content rules for the count line: it states total PRs and user-visible PRs,
and when PRs were merged into one curated entry it also reconciles that number
against the curated entry count — otherwise the two documents appear to
disagree.

Apply every correction to the **local** `_zh.md` files first, then copy those
in, so the next run starts from the wiki's wording instead of repeating the same
edits. Two gotchas:

- **A fresh wiki clone has no git identity** — `commit` fails with an
  unknown-author error. Set it first (reuse the main repo's): `git -C <wiki>
  config user.name "$(git -C <repo> config user.name)"` and the same for
  `user.email`.
- Publishing to the wiki is **team-visible** — it is an outward push, so do it
  only on an explicit request, not as part of the default run.

```bash
WIKI=<scratchpad>/simpler.wiki
git clone git@github.com:hw-native-sys/simpler.wiki.git "$WIKI"
cp WEEKLY_CHANGES_<Friday>_zh.md WEEKLY_ALL_PRS_<Friday>_zh.md "$WIKI/"
# edit "$WIKI/simpler-Wiki.md": prepend the two links above to their sections
git -C "$WIKI" config user.name  "$(git config user.name)"
git -C "$WIKI" config user.email "$(git config user.email)"
git -C "$WIKI" add -A
git -C "$WIKI" commit -m "Add weekly docs for <START> ~ <END> (zh)"
git -C "$WIKI" push origin HEAD
```
