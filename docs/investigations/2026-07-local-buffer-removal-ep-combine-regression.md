# 2026-07 — Removing PTO2LocalReadyBuffer exposed a missing dcci in EP dispatch

## Status

**RESOLVED** in PR #1245. Root cause was a latent kernel bug (dispatch never
flushed `recv_count_out` to HBM), unmasked by the dispatch-timing change from
removing `PTO2LocalReadyBuffer`. Fixed by a one-line `dcci` in the example
kernel; the local-buffer removal itself is correct. Onboard a2a3 3/3 pass after
the fix. See "Fix" below.

## Symptom

`examples/workers/l3/ep_dispatch_combine` (a2a3 onboard, `device_count=2`)
fails its golden check **deterministically** after the local-buffer removal:

- baseline (local buffer present): onboard **3/3 PASS**
- current (local buffer removed): onboard **3/3 FAIL**, sim **PASS**

No runtime-error signatures at all (zero 507018 / HandleTaskTimeout / deadlock /
stall in the device log). It is a **wrong result**, not a hang.

## Isolation (what the diagnostics prove)

The example verifies two stages; only the second fails:

- **dispatch stage** (`_verify_recv_outputs`: recv_x / recv_w / recv_idx /
  recv_count) — **passes** (prints nothing; it only prints on mismatch).
- **combine stage** (`_verify_routed_y`) — **fails**, and the chip's `routed_y`
  is **all zero**: `got min=0 median=0 max=0`, `bad=524287/524288`, `rel=1.0`,
  bad-d span = full [0,4095], bad rows = 128/128 on both chips.

So dispatch (which does the same class of cross-rank TPUT/TNOTIFY) is fine; the
combine kernel produces an entirely empty result.

## Hypothesis history

### REJECTED: dispatch→combine pub_counts visibility gap

Original theory: combine's push loop no-ops because `pub_counts` (written by
dispatch into window scratch, read by combine) reads as zero under the new
dispatch timing, leaving `routed_y_buf` untouched → all-zero `routed_y`.

**Disproved by direct measurement.** A temporary probe summed the whole
`pub_counts` table combine reads and broadcast it into `routed_y[0]` (host-
visible); on current-onboard it read **non-zero** (chip0 sum surfaced as
`got max=20476.7`). So the dispatch→combine scratch dataflow IS visible — combine
sees correct counts. This edge is NOT the bug.

### OPEN: root cause is in combine's push→reduce section

With pub_counts confirmed visible, the all-zero `routed_y` must come from
downstream of that read, inside combine.cpp:

1. **push phase** — the cross-rank `TPUT` of `recv_y` rows into the peer's
   `routed_y_buf` never lands, or
2. **combine_done barrier** (TNOTIFY/TWAIT) — mis-synchronizes, or
3. **reduce phase** — reads `routed_y_buf` as zero.

A second probe (sum `routed_y_buf` after the barrier, before reduce) would
disambiguate push-landed vs reduce-read, but the ad-hoc GM-scalar-loop probe
**failed to compile** on AICore (bisheng frontend error 70). Next approach:
runtime-side timing comparison (baseline vs current [STRACE] / PTO2_SCHED_PROFILING
of the combine task) rather than more in-kernel probes.

### NARROWED: the empty result originates at local_expert, not combine

Python-side probes (main.py, host-visible OUTPUT_EXISTING tensors, no kernel
recompile) on current-onboard show:

| tensor | producer | state after run |
| ------ | -------- | --------------- |
| recv_x_out | dispatch | **correct** (nz≈3.2M, absmax 5472) |
| recv_w_out | dispatch | **correct** (nz=782) |
| recv_count_out | dispatch | **correct** (sum 782 / 754) |
| recv_y | local_expert | **ALL ZERO** (nz=0) |
| routed_y | combine | all zero (combine pushes recv_y's zeros) |

So combine is a red herring — it correctly propagates an already-empty input.
The regression is at the **dispatch → local_expert** task edge: local_expert
had all three inputs correct in the *final* state, yet produced nothing.

local_expert (`kernels/aiv/local_expert.cpp`) bounds its work by
`n_rows = recv_count[e]` and skips the row loop entirely when it reads 0.
**Leading hypothesis:** under the new dispatch timing, local_expert's AICore
reads `recv_count` (dispatch's HBM OUTPUT_EXISTING output, reused as
local_expert INPUT) as stale/zero → every expert does 0 rows → recv_y stays
zero. The host sees recv_count=782 afterwards because dispatch's write *did*
land — just not before local_expert consumed it.

Two overlapping-sched-window facts, established via device_wall.sched markers:
the two ranks' combine tasks DO run concurrently (rules out "one rank exits
before the other's push"), consistent with the fault being upstream in
local_expert, not the cross-rank combine.

A kernel sentinel probe to confirm "did local_expert read count=0" crashed the
AICore (507015 — bad UB slot addresses in the ad-hoc probe) and was reverted.

### CONFIRMED ROOT CAUSE: dispatch never flushes recv_count_out to HBM

A compile-safe probe settled it: forcing local_expert to process **all R rows**
(`n_rows = R`, ignoring `recv_count[e]`) makes the whole test **PASS** on
current-onboard. So recv_x / recv_w are visible and correct — the *only* wrong
input is `recv_count`, which local_expert reads as **0**.

Why: in `dispatch.cpp`, `recv_count_out[e] = sum;` (line ~304) is a **raw scalar
GM store** from the AICore scalar unit, followed only by `pipe_barrier(PIPE_ALL)`.
`pipe_barrier` orders on-core pipelines but does **NOT** flush the cache line to
HBM — that needs a `dcci(..., CACHELINE_OUT)`. recv_x/recv_w/recv_idx are safe
because they go out through `TSTORE` (a real vector→GM write). `recv_count_out`
sits in cache; whether the downstream `local_expert` task's AICore sees it in
HBM depended on incidental timing.

Removing `PTO2LocalReadyBuffer` changed AICPU dispatch timing enough that
local_expert now reads `recv_count_out` before dispatch's cached scalar store
lands in HBM → every expert runs 0 rows → recv_y all-zero → combine faithfully
pushes zeros → routed_y all-zero. Baseline timing happened to let the store
land first; sim has no cache so it never reproduced.

This is a **latent kernel bug**, not a runtime regression: the local-buffer
removal is correct; it merely unmasked a missing `dcci` in the example kernel.

## Fix

In `examples/workers/l3/ep_dispatch_combine/kernels/aiv/dispatch.cpp`, after the
`recv_count_out[e] = sum;` loop, flush the written range to HBM with `dcci`
(pattern already used in `qwen3_14b_decode/fa_work_build.cpp` and
`deferred_notify_demo/kernel_producer.cpp`: `dcci(ptr, ENTIRE_DATA_CACHE,
CACHELINE_OUT)` / `dcci(ptr, SINGLE_CACHE_LINE, CACHELINE_OUT)`). The runtime's
completion-before-dispatch invariant then carries the flushed value to
local_expert correctly. Codegen-owned area (`examples/`); no runtime change.

Robustness note: the same raw-scalar-store-without-dcci pattern should be
audited elsewhere in dispatch.cpp (any non-TSTORE GM write consumed by a later
task), since all of them were relying on the same incidental timing.

## Superseded ambiguity (kept for history)

- If the dispatch→combine **window-scratch visibility** is a guarantee the
  runtime is supposed to provide across a task dependency edge, the real fix is
  in the **runtime** (`src/{arch}/runtime/tensormap_and_ringbuffer/`) — restore
  the cross-task visibility the local buffer was incidentally providing — and
  the local-buffer removal is fine once that guarantee is explicit. This is the
  Runtime-owned area.
- If combine is simply **missing a synchronization** it always needed (and only
  passed before by timing luck), the fix is in the **kernel**
  (`examples/.../combine.cpp`, Codegen-owned) — e.g. a barrier / acquire on
  `pub_counts` before the push loop.

Deciding requires the pub_counts dump above. Do not edit either area before the
dump confirms which edge actually fails.

## Do NOT re-derive

- sim passes; the bug is hardware-timing-only. Reproducing needs a2a3 onboard,
  `device_count=2`, via `task-submit`.
- dispatch is not the culprit — it verifies clean. Focus on the dispatch→combine
  window edge (`pub_counts`) and the combine push loop's `n == 0` early-out.
