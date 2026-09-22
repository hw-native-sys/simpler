# Writing a curated entry

The structure and prose rules for `WEEKLY_CHANGES_<Friday>.md`, split out of
[SKILL.md](SKILL.md) so the flow stays readable. **Read this before writing the
first entry** — the rules here are what the readability check at the end gates
on.

Section numbering continues SKILL.md: §4 is the document structure, §5 the
writing style, and both are referenced by those numbers from there.

## 4. Document structure

Every run assumes the project has no prior weekly report — render the full
structure below, do not "match prior reports." The skeleton is shown using
four-backtick fences so that the inner triple-backtick example blocks
render literally. Use the headings literally in the English doc — translation,
if any, happens only in the `_zh.md` pass ([SKILL.md](SKILL.md) §7c).

````markdown
# simpler weekly external changes (YYYY-MM-DD ~ YYYY-MM-DD)

This document presents core interface and feature changes via example
comparisons; see each PR for details.

---

## I. Interface changes

### N. <short title> — [#NNN](https://github.com/.../pull/NNN)

*<scope line — only when the change is confined: arch (a2a3 / a5), runtime
(hbg / tmr), or layer. Derive it from `--name-only` paths, do not guess.>*

**Why:** <ONE problem → one paragraph, no list. What broke, was missing, or
forced a migration; enough context to feel it, no business detail the reader
does not need.>

**How:** <the mechanism that resolves it — the new argument, changed default,
added validation, new contract. Not "it is fixed".>

<!-- TWO OR MORE distinct problems → use the numbered form instead, and drop
     the single-paragraph form above: -->

**Why:**

1. **<one-sentence verdict in bold.>** <the problem.>
2. **<second distinct problem.>** <...>

**How:**

1. **<the mechanism, in bold.>** <how it resolves problem 1.>
2. **<answers problem 2.>** <...>

<example for problem 1 — see §5 below for the ordering and same-shape rules>

```diff
- <old code>
+ <new code>
```

<example for problem 2>

<necessary constraints / caveats, 1-3 lines>

---

## II. New features

### N. <feature name> — [#NNN](https://github.com/.../pull/NNN)

*<scope line, same rule>*

**Why:** <one problem → one paragraph. Several drivers → the numbered-list shape
above. What was impossible, broken, or costly before it.>

**How:** <the surface the user calls plus what it does underneath, mapped back to
the problem each part removes, in the same order.>

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

### The scope line

A reader's first question is "does this reach me?", and the answer is mechanical
— it is the set of path prefixes in the diff. Emit the line whenever the change
is confined to part of the matrix, and omit it when it genuinely spans
everything:

```bash
# --name-only, not --stat: --stat's trailing "N files changed" summary would
# contribute a bare "N" to the list, and it renders a rename as `a/{x => y}/f`.
git show <sha> --format='' --name-only | sed 's|/[^/]*$||' | sort -u
```

- All runtime paths under one arch → *a5 only*; under one runtime → *hbg only*.
- No `src/` at all → say so explicitly (*scene-test layer: `simpler_setup/` and
  `examples/` only — no `src/` change*). This is the single most misread case:
  without the line, a scene-test change reads as a runtime change.
- `src/common/platform/` or `src/common/<runtime>/` → applies to every caller,
  including other repos. Say that, because the opposite is assumed by default.

### Structural changes: retired / added / wire

When a PR reshapes a schema, a device struct, or a record channel, prose buries
what the reader needs. Use three labelled blocks instead:

- **Retired** — name the fields or channels and say why they went (usually
  "recorded, consumed by nothing").
- **Added** — name what replaces them, and if an index changed (per-core →
  per-thread) say so, because that changes the meaning of every row.
- **Wire contract** — did the device-side layout keep its size? A removed field
  backfilled by `reserved[N]` is not an ABI change; say which it is.

## 5. Writing style

- **Examples over prose.** Every interface change has a before/after diff
  block from a real example file in the repo (`examples/...` or
  `tests/...`). Pull the snippet directly from the PR diff — do not
  paraphrase.
- **Examples follow the Why's order, and each one names which point it serves.**
  With two numbered problems, the example for problem 1 comes first, labelled
  (`**For 1** — ...`). An entry whose only diff illustrates the *second* point
  leaves the first one unevidenced, which is how a Why survives being wrong.
- **The main example shows the headline change, not its side effects.** If the
  PR makes a hardcoded operation selectable, the diff to show is the operation
  becoming a `switch` — not the argument-count bump that follows from it. Pick
  the snippet by asking which line the reader would have to write differently.
- **A before/after comparison must be same-shape.** When contrasting two call
  sites, hold everything constant except the difference: same variable name,
  same type, same surrounding context. Contrasting
  `static_cast<DataType>(ctx.scalar(10))` with
  `from_u64<int32_t>(orch_args.scalar(0))` buries a one-token difference under
  three irrelevant ones. Prefer the same symbol read from two places (the same
  workload's two runtime ports is the ideal case); if no such pair exists in the
  diff, normalize the names and say the snippet is illustrative.
- **Motivation is mandatory for features and interface changes.** Every entry
  in §I and §II opens with a `**Why:**` that names the concrete problem the
  change solves — a dropped capability, an error code, a perf cost, a missing
  surface. Give enough context that a reader feels the problem: what was hit,
  under what condition, and why it mattered — not a single terse clause. 2-5
  lines; do not skip it and do not pad it to filler. If you cannot name a
  concrete problem, re-check whether the PR is actually user-facing (it may
  belong in the excluded list).
- **State the problem, do not narrate a use case.** The Why explains the
  mechanism that was missing or wrong. It does not need the domain detail of
  whichever caller happened to hit it — "an orchestration may read an input's
  bytes to decide the graph shape" carries the problem; what `paged_attention`
  computes from those bytes does not. Name a caller only when the reader cannot
  otherwise tell whether the case applies to them.
- **Describe only the state that predates the whole change.** If the entry
  covers a stack of PRs, the Why is the world before the first of them. A
  transitional state one PR opened and the next closed (a deprecation that lived
  for three days, an API shape that existed only between two merges) is not a
  problem the reader ever faced — leave it out of Why and How, and let the
  per-PR inventory ([SKILL.md](SKILL.md) §6) record what each PR did on its own.
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
- **Soft cap: ~25 lines per entry.** Past that, what usually has to go is
  second-order numbers (a measurement supporting a measurement), a platform
  table restated in prose next to the table, and any example illustrating a
  point the reader has already accepted. Keep whatever the reader needs to make
  a decision — which platform behaves how, what the new spelling is, what is
  still not allowed — and drop the rest to the PR link.
- **Bug-fix rows are user-symptom only.** Each §III row is one line, takes no
  `**Why:**`, and describes what a user would have *observed* before the fix —
  a log line, an error code, a wrong output, a hang — **not** the internal
  cause ("missing finalize call", "wrong header order"). If you cannot phrase
  the user symptom in one line, the PR is internal and belongs in the excluded
  list, not §III.

### Before you ship: read each entry as someone who did not open the PR

For every §I / §II entry, answer these without looking at anything else. Any
"no" is a rewrite, not a footnote.

1. **Can I restate the problem in one sentence?** If the Why only makes sense
   once you know the PR's internals, it is not a problem statement yet.
2. **Does the How obviously resolve that problem?** The mechanism and the
   problem must be visibly the same subject — "conversions were written at every
   call site, so some were wrong" → "the conversion moves into the accessor" is
   traceable; "both sides gain a template parameter" is not.
3. **Does the example show the Why's first point?** (See the ordering rule
   above.)
4. **Does it say whether this reaches me?** Arch, runtime, layer — the scope
   line.
5. **Is anything here a mechanism I did not verify in the tree?**
   ([SKILL.md](SKILL.md) §2 Pass B.)

This check exists because its absence is expensive in a specific way: the report
looks finished, and the cost lands later as a reader asking "I don't understand
what this PR is doing" — once per unclear entry.
