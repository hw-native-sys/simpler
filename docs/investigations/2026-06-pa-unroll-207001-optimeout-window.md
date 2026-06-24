# PagedAttentionUnrollManualScope intermittent 207001: an op-timeout-window issue fixed by #1035; a5 launch reorder kept as latency + defense-in-depth

**Date**: 2026-06-16
**Verdict**: #1019 itself is fixed by #1035 (op-execute timeout 1 s → 3 s,
landed for an unrelated tensor-dump feature) — no code change is *needed* to
close #1019. Separately, the a5 launch reorder (AICore before the AICPU Run
task) **is kept** as a first-launch latency optimization (~1.4 s → ~0.4 ms,
measured) and op-timeout-family defense-in-depth — it removes the slow launch
that trips the timeout, rather than only widening the window. a2a3 mirrors the
same reorder to keep the two arches symmetric; it is **unverified on a2a3
silicon** (a5-only dev box), relying on CI.

## Question

Issue #1019: `TestPagedAttentionUnrollManualScope` on a5 intermittently
(~30 % on a poisoned device) fails with `207001`, whose CANN string is
"ACL_ERROR_RT_MEMORY_ALLOCATION / host memory has been exhausted" — which
reads like HBM OOM. It is filed as "op-timeout family, #1016". The intuition
that sends you down the wrong path: the error string says memory, and the
test does paged-attention with sizeable KV buffers, so it looks like a
sizing/leak/OOM bug.

## What was tried

- **Confirmed not OOM**: an HBM probe showed 122 GB free at the failing
  launch. `207001` = `RT_ERROR_MODULE_NEW` aliased to
  `ACL_ERROR_RT_MEMORY_ALLOCATION` with a hardcoded misleading string
  (`errcode_manage.cc:243`) — a *null module object*, not host OOM.
- **Same-device A/B on launch order** (host-side submit timing instrumented
  around `rtKernelLaunchWithHandleV2`): with the AICPU Run task launched
  before the AICore (the original order), the first AICore submit blocked
  0.95–11.5 s; launched after, ≤424 µs. On a poisoned device: OLD order
  5/14 pytests failed, NEW order 0/80 launches. This *looked* like proof the
  order is the root cause.
- **CANN runtime source trace** (gitcode `cann/runtime`): chased several
  candidate mechanisms for the multi-second block — per-device task-id pool
  back-pressure (`task.cc` `TryAgainAlloc`, 1000×10 ms), lazy device-side
  module load (`Module::Load` = HBM `DevMemAlloc` + H2D `MemCopySync`). None
  was confirmed; each failed a sanity check (e.g. a single spinning AICPU
  task holds one task-id, nowhere near pool exhaustion).
- **Full st-onboard-a5 suite, 6× on 2 cards** with the OLD-order build:
  0 reproductions. Single-example loops the same day: ~60–80 launches,
  0 reproductions. The bug would not reproduce on a healthy device at all.
- **Bisect** (the step that actually answered it): the branch's merge-base
  was afb5c5a9 (#1016, op-timeout **1 s**); current upstream is post-#1035
  (op-timeout **3 s**). On the *current* code, reverting only the #1035
  timeout values to 1 s/2 s (keeping the OLD launch order) reproduced the
  op-timeout wedge **on the same clean device** the 3 s build never failed
  on.

## Result

- OLD launch order makes the **first** AICore launch slow — measured ~1.2–1.6 s
  median, up to 11.5 s worst case on an already-poisoned device. It is a
  *latency* effect of submitting the AICore right after the AICPU Run task
  starts spinning in `handshake_all_cores()`. The slow path is the lazy
  device-side binary load inside `rtKernelLaunchWithHandleV2`
  (`Module::Load` = HBM `DevMemAlloc` + H2D `MemCopySync`).
- **Spin poll-frequency is NOT the cause** (throttle experiment, inconclusive):
  the handshake spin on `aicore_regs_ready` is a no-op `SPIN_WAIT_HINT()` =
  a tight GM-hammer loop. Inserting a busy delay between polls gave a
  non-monotonic, within-noise result — median first-launch submit 1.46 s (no
  throttle) → 1.10 s (200k-iter delay) → 1.40 s (4M-iter delay), n=8 each with
  heavily overlapping spreads. If the spin's GM traffic were the cause, more
  throttle would help monotonically; it didn't. So reducing the poll rate does
  not reliably change the launch time — the cause is **not** the spin's polling
  frequency. The actual internal cause of the ~1.4 s OLD-order slow launch
  remains unpinned (it is not isolable from outside CANN).
- **What IS robust**: NEW order (AICore launched before the AICPU Run task, so
  no AICPU is running during the binary load) submits in **~0.4 ms** vs OLD
  order's **~1.4 s** median — a ~1.4 s first-launch difference, far outside the
  noise. Whatever the internal cause, having the AICPU Run task already
  launched is what makes the first AICore launch slow, and the reorder removes
  it entirely.
- With op-timeout **1 s** (pre-#1035): that ~1.9 s launch exceeds the
  threshold → STARS reaps the op → op-timeout cascade. Surfaces as `207001`
  (null module, when the reap lands during module creation) or `507046`/
  `507000` (stream-sync timeout) — same family, different reap point.
  Reproduced **1/15** on a clean device.
- With op-timeout **3 s** (post-#1035): the same ~1.9 s launch fits inside
  the window and completes → no cascade. **0/~40** on the same device.
- #1035 ("tensor dump on the abnormal path", closes #1034) raised
  `PLATFORM_OP_EXECUTE_TIMEOUT_US` 1 s → 3 s and
  `PLATFORM_STREAM_SYNC_TIMEOUT_MS` 2 s → 4 s as a side effect of ordering
  the timeout chain (SCHEDULER 2 s < OP_EXEC 3 s < STREAM_SYNC 4 s). Nobody
  connected it to #1019.
- A launch-order reorder (AICore before AICPU Run) removes the slow launch
  entirely (NEW order submits in ~0.4 ms). It is **kept on a5** — not as the
  #1019 fix, but as a latency + defense-in-depth measure (below).

## Disposition: #1035 fixes #1019; the a5 reorder is kept on its own merits

**#1019 itself needs no code change** — #1035 already prevents the failure on
current `main` by widening the op-timeout window past the slow launch. The bug
is "fixed by #1035; verify and close". Labelling a reorder as *the* #1019 fix
would overclaim a root cause we never pinned (the ~1.4 s contention) and
duplicate a fix that already landed; the cascade is additionally *recovered*
in-process by #1016's force-reset and *contained* by #1005.

**The a5 reorder is nonetheless kept**, justified independently of #1019:

- *Latency*: it removes a robust, large first-launch cost (~1.4 s → ~0.4 ms,
  measured, far outside noise) on every fresh worker's first AICore launch.
  Zero functional risk — the handshake is launch-order-independent (NEW order
  passes the full suite).
- *Defense-in-depth*: the slow launch is what trips the op-execute timeout.
  #1035 widened the window; the reorder removes the slow launch itself, so the
  wedge cannot return if the timeout is later tightened or a slower device
  pushes the launch past it.

**a2a3 mirrors the same reorder** to keep the two arches symmetric (one launch
order, not split per-arch). a2a3 has the same Init→`core_type`-publish→Run→AICore
structure as a5, and the a5 reorder keeps AICore after Init and before Run
(verified), so the mirror is the same logical change. It is **unverified on
a2a3 silicon** (a5-only dev box) — relying on CI; the ~1.4 s slow-launch was
only measured on a5.

## When to reconsider

- If `PLATFORM_OP_EXECUTE_TIMEOUT_US` is ever lowered back toward the slow
  first-launch latency (~2 s), #1019 returns — the reorder (AICore-first) is
  then the correct fix, and the ~1.9 s OLD-order first-launch latency is the
  number to design against.
- If first-run latency on a5 becomes a measured concern, the AICore-first
  reorder is a free ~1.9 s win on the first launch (handshake is
  launch-order-independent), independent of #1019.
- The still-open part: *why* the OLD-order first launch takes ~1.4 s (it is not
  the spin poll frequency — throttling didn't help). CANN's per-stage
  timestamps (`DumpTimeStampPart1`, gated by the `TEMP_PERFORMANCE` build macro
  — needs a CANN rebuild) on an OLD-order run would show whether it is in
  `MemCopySync`/`DevMemAlloc` (the binary load) or the task-submit path.

## References

- Issue #1019 (open), #1034 / PR #1035 (the timeout retune), #1016
  (force-reset recovery), #1005 (cascade containment).
- `src/a5/platform/include/common/platform_config.h` —
  `PLATFORM_OP_EXECUTE_TIMEOUT_US`, `PLATFORM_STREAM_SYNC_TIMEOUT_MS`.
- CANN runtime (gitcode `cann/runtime`): `common/errcode_manage.cc:243`
  (207001 alias), `launch/aix_stars.cc`, `kernel/module.cc`.
- Related: [`2026-06-a5-aicore-op-timeout-cascade.md`](2026-06-a5-aicore-op-timeout-cascade.md)
  (same op-timeout family, recovery side).
- Rule: [`.claude/rules/discipline.md`](../../.claude/rules/discipline.md)
  §"bug fixes start with a failing repro" and the rebase-before-testing
  guidance — testing on the pre-#1035 base is exactly what made this look
  like an unfixed bug.
