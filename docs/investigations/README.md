# Investigations

Write-ups of work that was done to **answer a question** and where the
answer wasn't a code change that lives in the repo. Specifically:

- Optimizations that were considered, measured, and dropped (no signal /
  not worth the complexity).
- Designs that were prototyped and rejected.
- "I think we should do X" proposals where the analysis ruled out X
  (or scoped X to "later, when Y is true").

The point is to save the **next** person's time. If you find yourself
reaching for the same idea, the entry tells you:

1. It was already considered.
2. What measurement / argument shut it down.
3. Under what conditions it might become worth re-opening.

If you re-investigate anyway, update the entry with the new data —
don't open a parallel doc.

## What does NOT go here

- **Active bugs / unresolved problems** → `KNOWN_ISSUES.md` (local) or a
  GitHub issue.
- **Problems with a known fix or workaround** →
  `docs/troubleshooting/`.
- **Designs that were prototyped and shipped** → the design doc lives
  with the subsystem (e.g. `docs/dfx/<feature>.md`).
- **Architectural decisions that constrain future code** → if/when we
  adopt ADRs, those would live elsewhere; this folder is for things we
  *didn't* do.

## File naming

`YYYY-MM-<short-slug>.md` — e.g. `2026-06-l2-swimlane-defer-wmb.md`.
The date is the month the investigation was done so entries sort
chronologically and stale ones are easy to spot.

## Template

```markdown
# <Title — what was proposed, in one line>

**Date**: YYYY-MM-DD
**Verdict**: dropped / deferred-pending-X / superseded-by-Y

## Question

Brief statement of the proposal. Why it might be a good idea — the
intuition that would make a future contributor reach for the same
change.

## What was tried

Concrete actions. Commands, files touched, measurement setup. Enough
that someone can reproduce the measurement, not enough to retell the
whole codebase.

## Result

The numbers, the diff size, the bug found — whatever the actual output
of the investigation was.

## Why not (now)

The decision. Tie it to a specific signal in the result, not just
preference.

## When to reconsider

The condition under which this becomes worth re-opening. "If workload X
shows >Y µs in profile" / "after Z lands" / "if hardware changes such
that ...".

## References

- PRs, commits, issue links.
- Related rules (`.claude/rules/...`) or docs that informed the
  decision.
```

## Index

Newest first.

- [2026-06 — Replacing COND with GM+dcci for AICore→AICPU notification](2026-06-cond-vs-gm-notification.md)
- [2026-06 — Letting AICore directly read or write the SPR MMIO window](2026-06-aicore-mmio-to-spr.md)
- [2026-06 — PA-unroll 207001: an op-timeout-window issue fixed by #1035, not a launch-order bug](2026-06-pa-unroll-207001-optimeout-window.md)
- [2026-06 — Cross-task batched publish: hoist wmb across distinct tasks in one pop](2026-06-cross-task-batched-publish.md) — also carries the root cause + fix for the `spmd_sync_start_stress` 507018 drain-barrier hang
- [2026-06 — AICore first-task cold-start: pre-warm dispatch path](2026-06-aicore-cold-start-warmup.md)
- [2026-06 — a5 AICore op-timeout poisons the shared L2 worker (cascade)](2026-06-a5-aicore-op-timeout-cascade.md)
- [2026-06 — a5 AICPU filter gate: Scenario B fail-fast guard not added](2026-06-a5-aicpu-filter-gate-scenario-b-validation.md)
- [2026-06 — Sanitizer rollout scope: macOS, TSAN gating, LSan](2026-06-sanitizer-scope.md)
- [2026-06 — L2 swimlane: defer per-task wmb to rotation](2026-06-l2-swimlane-defer-wmb.md)
