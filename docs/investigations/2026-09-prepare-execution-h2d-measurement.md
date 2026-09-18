# Measuring what the per-run Runtime H2D reductions bought in `prepare_execution`

**Date**: 2026-09-18
**Verdict**: inconclusive — no stable effect identified, and the experiment does
not bound the real one

## Question

Issue #2254 argues the per-run `Runtime` device image carries content that has no
business travelling, and names instrumenting `prepare_execution` as a
prerequisite before anything claims a speedup. Two of its steps then landed —
**#2297** removed AICore's device `KernelArgs` copy along with one host-to-device
transfer per run (two when any DFX channel is on), and **#2309** cut
`host_build_graph`'s per-run image from 51,584 B to 17,536 B on a2a3 (56,192 →
22,144 on a5) by ending it before the host-only tail.

Both PRs stated no latency claim, because there was nothing to measure with.
**#2307** then added that instrumentation: a `chip.run.prepare_execution` STRACE
span at depth 1, sibling of `chip.run.bind`. So the obvious next question is what
the two reductions are actually worth in wall time.

The intuition that makes this worth checking: 34,048 bytes is 66% of the old
image, and a synchronous `rtMemcpy` blocks the host inside the span. At a naive
3 GB/s that is ~11 µs per run, on a path that runs once per invocation.

## What was tried

A/B on one variable, at a fixed commit so nothing else moved.

- Base: `8726d7e13` (the #2309 merge). Both arms are that tree; the **only**
  difference is `runtime_device_copy_size()` in
  `src/common/host_build_graph/shared/runtime.cpp` returning
  `Runtime::device_image_bytes()` (17,536 B) versus `sizeof(Runtime)` (51,584 B).
  Same struct layout, same call count, same code paths — only the transfer
  length differs. Changing the line rather than reverting the commit also keeps
  `HEAD` fixed, which matters because the `_task_interface` build-stamp guard
  trips on any `HEAD` move and forces a reinstall between arms.
- Workload: `examples/a2a3/host_build_graph/benchmark_bgemm`, a2a3 **onboard**,
  `--rounds 400`, `-s` (pytest captures stderr and discards it on pass, so
  without this the markers never reach the log). One `prepare_execution` span per
  invocation.
- Order: **ABBA** — B1, A1, A2, B2 — **two batches per arm**. Each batch recorded
  `task-submit --list` and `pgrep -af task-submit` at launch.
- Analysis: parse `[STRACE] ... name=chip.run.prepare_execution ... dur=` from the
  host log, drop `inv<=2` as cold.

Only the transfer **length** was varied. **#2297 was not an arm of this
experiment** and is not measured anywhere below.

Simulation is not a valid arm here: it performs no `Runtime` H2D at all, handing
the device the host object directly.

## Result

Per batch, warm invocations only (µs):

| batch | length | n | median | lag-1 autocorr |
| ----- | ------ | - | ------ | -------------- |
| B1 | 17,536 B | 398 | 77.1 | +0.815 |
| A1 | 51,584 B | 398 | 75.7 | +0.225 |
| A2 | 51,584 B | 398 | 59.1 | +0.252 |
| B2 | 17,536 B | 398 | 65.0 | +0.192 |

Aggregated per batch, which is the coarsest unit this design offers:

```text
batch medians:       A(old) 75.7, 59.1 us   B(new) 77.1, 65.0 us
mean of batch medians: A 67.4 us            B 71.0 us   (difference -3.6 us)
within-arm spread:   A 16.6 us              B 12.1 us
```

**No stable effect is identified.** The difference between arms is smaller than
the spread between two batches of the same arm, in both arms.

That is a description of what was observed, not a bound on the real effect. With
two batches per arm and the batches run under different conditions, this
experiment supports no interval on the true benefit: a pooled-variance t
calculation on these four numbers returns ±44 µs, but it assumes the batches are
independent draws with a common variance, and neither assumption is established
by this design — aggregating to one number per batch makes the batch a sensible
*reporting* unit without making it an independent *replicate*. Widening a
standard error also does nothing about bias that is correlated with the arm,
which this design cannot rule out. Quote that ±44 µs only as an illustration of
how little two batches per arm constrain anything, never as a 95% interval on the
benefit.

Two features of the data are why no conclusion is available:

- **Batch drift of 12–17 µs.** Two batches of the *identical* arm differ by
  16.6 µs (old: 75.7 → 59.1) and 12.1 µs (new: 77.1 → 65.0) — larger than the
  effect being hunted.
- **The arms ran under different recorded conditions**, so environmental
  confounding cannot be excluded. The snapshots show A1/A2 with 6/16 devices
  occupied — four of them by a concurrent session of this same user — B1 with
  2/16, B2 with 7/16. Separately, B1's lag-1 autocorrelation of **+0.815**,
  against ~0.2 in the other three, shows its samples carry strong temporal
  dependence. **The source of that dependence was not identified.** Contention
  from other tenants is one candidate and the snapshots make it plausible, but
  thermal or frequency drift over a batch, and the workload's own state evolving
  across 400 rounds, would produce the same signature; a single occupancy reading
  taken at launch establishes no causal correspondence with the latencies.

### The analysis error this entry exists to record

The first version of this write-up pooled all 796 invocations per arm and
bootstrapped the difference of medians, reporting **[−4.5, +3.6] µs** — and drew
two conclusions from it: that an 11 µs effect was *excluded*, and therefore that
fixed per-call `rtMemcpy` cost dominates the byte cost at these sizes.

**Both conclusions are withdrawn, because the interval they rested on assumed an
independence the data contradict.** Pooling invocations treats 796 samples as
independent draws — classic pseudo-replication. Any influence that differs
between batches is by construction common to the invocations inside one, and the
measured lag-1 autocorrelations indicate the samples are not independent *within*
a batch either. So that interval does not mean what it was read to mean.

Note the asymmetry with the batch-level figure above, which is why the two are
handled differently here: the pooled bootstrap's independence assumption is
contradicted by a measurement in hand, whereas the batch-level t's assumptions
are merely untested — two batches per arm cannot check either of them.

How far it understated the uncertainty is **not quantified here, and cannot be by
comparing it to the batch-level number**: the two are different statistics — a
difference of medians over pooled invocations against a difference of means of
batch medians — and the second is not a calibrated reference to measure the first
against. What can be said is that its independence assumption is contradicted, so
it
supports no exclusion.

A corollary: the claim that **#2297** (which removed whole calls) is a better
latency bet than #2309 (which removed bytes) rested entirely on that exclusion,
and is withdrawn — it was never an arm here and remains unmeasured.

## Why not (now)

The arms ran under different recorded conditions, so environmental confounding
cannot be excluded — this is not merely a power problem. Batch drift within an
arm exceeds the difference between arms, so more invocations inside these same
four batches would not help: they would sharpen a number whose centre may already
be displaced by whatever differed between the arms.

What a conclusive version needs is **experimental control**, not finer spans:
several batches per arm rather than two, arms interleaved so any drift is shared
rather than assigned to one arm, an idle box or a dedicated runner, and the
analysis done at batch level. A controlled A/B on the **total** span is a valid
way to measure whether the change is worth anything overall.

Sub-spans are a separate tool for a separate question. `prepare_execution` covers
at least eight operations — `ensure_device_wall_buffer`,
`ensure_device_run_result_region`, `ensure_aicore_reg_table` twice (Ctrl and Pmu),
`resolve_task_binary_addrs`, `prepare_orch_so`, `init_runtime_args_with_metadata`
(the H2D in question) and `kernel_args_init_ffts_base_addr` — so a sub-span around
the transfer would say *where* a measured effect came from. It would not fix
unequal load, and it is not a substitute for control.

This does not retract anything either PR claimed. Both reported removed
allocations and removed transfers — which are counted, not timed — and explicitly
declined a latency claim. That claim remains unmade, now with a measurement that
also fails to support it either way.

## Unmeasured, and stated as such

The first invocation costs **~601 µs against a ~69 µs warm median, ~8.7×**. That
is a cold/warm difference and nothing more: the decomposition into allocation,
initialization, first upload of a cold buffer, page faults and driver setup was
**not measured**. Reading the code, the prepare path does perform lazy
per-slot allocation on first use, which is a plausible contributor — but this
experiment does not show that it is the whole gap, or most of it. Splitting that
601 µs is its own piece of work and would need the same phase subdivision named
above.

## When to reconsider

- **When a controlled re-run is possible** — idle box or dedicated runner, arms
  interleaved rather than run in two blocks, several batches per arm, analysis at
  batch level. Six batches per arm is a starting budget rather than a threshold:
  how many are actually needed depends on the target resolution, the batch
  variance that environment turns out to produce, and the design, none of which
  this run establishes. The point of the re-run is to remove the arm-correlated
  environmental difference; sample count alone does not.
- **If a sub-span is added around the Runtime H2D** (a child of
  `chip.run.prepare_execution`, the way `chip.run.bind.*` subdivides bind for
  `hbg_bind_phases`). Useful for attributing a measured effect and for splitting
  the cold cost; not a substitute for the control above.
- **If #2254 step 2 lands**, which removes the handshake region and
  `func_id_to_addr_` from the image — another ~17.5 KB, and possibly the copy's
  reason to exist rather than its size.
- **If a workload makes prepare hot** — high invocation count with short device
  work, where ~69 µs of host prepare stops being amortised.

## References

- PRs: #2297 (AICore launch arguments), #2309 (hbg host-only tail), #2307
  (the `prepare_execution` span), #2295 (moved the lease ledger, rebased through).
- Issues: #2254 (parent), #2308 (the hbg subtask), #2257 (the AICore subtask).
- Measurement base `8726d7e13`; harness and raw logs are not committed.
- **#2315 supersedes the mechanism, not the numbers.** It re-expresses the same
  boundary as a named `DeviceRuntimeLaunchDesc` first member instead of an offset
  into the host-only tail, and reaches byte-for-byte the same image sizes. After
  it lands the reproduction recipe changes — the A/B line becomes
  `sizeof(runtime->dev)` against `sizeof(Runtime)` rather than
  `Runtime::device_image_bytes()` against `sizeof(Runtime)`.
- [`.claude/rules/running-onboard.md`](../../.claude/rules/running-onboard.md) —
  every arm ran under `task-submit`; the per-arm load snapshots exist because
  that rule's anti-pattern list calls out unlocked and unrecorded comparisons.
  They are also what showed, after the fact, that the arms had not run under
  equal conditions — which is what a comparison needs recorded whether or not it
  turns out to explain anything.
- [`.claude/rules/discipline.md`](../../.claude/rules/discipline.md) §4 — the rule
  that says a measured-no-signal investigation gets written down rather than
  re-derived.
