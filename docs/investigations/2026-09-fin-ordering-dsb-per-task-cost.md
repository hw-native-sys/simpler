# Narrowing the AICore FIN-ordering `dsb` to tasks that have outputs

**Date**: 2026-09-18
**Verdict**: dropped — the barrier costs ~10 ns per task, an order of magnitude
too little to pay for a branch on the same path

## Question

\#2324 made `OUT_OF_ORDER_STORE_BARRIER()` a real `dsb(DSB_DDR)` on a2a3 and a5
onboard, which puts a barrier on two per-task AICore paths: after the kernel
returns in `execute_task` and after the early-dispatch args fill. The barrier is
required for correctness — without it a kernel's output write-back can still be
in flight when its task reports FIN, and #2233 is the resulting intermittent
wrong result.

Required is not the same as free. A barrier on a per-task path invites the
obvious narrowing: only tasks that actually wrote an output need to wait for a
flush, so gate it on the task having outputs and skip it otherwise. That is
worth doing if the barrier costs enough to notice, and the prior said it might —
`2026-07-aicore-swimlane-switch-overhead-and-ack-gate.md` attributes a ~0.5 µs
tail to a record write-back `dsb`, and at 0.5 µs against a 1001-task critical
path the barrier would be 3% of `sliding_window_deps`.

## What was tried

Two arms differing only in `src/a2a3/platform/onboard/aicore/inner_kernel.h`,
plus a repeat of the first arm run after the second as a drift control:

| arm | macro | `aicore_kernel.o` md5 |
| --- | ----- | --------------------- |
| A, A2 | `dsb(DSB_DDR)` | `b9ba5d90` |
| B | `((void)0)` | `ca0e4eae` |

```bash
# per arm, from a worktree parked on a fixed commit
rm -rf build/cache/a2a3/onboard          # see "the two traps" below
.venv/bin/pip install --no-build-isolation -e .
md5sum build/lib/a2a3/onboard/tensormap_and_ringbuffer/aicore_kernel.o
task-submit --device auto --device-num 1 \
  --run "./tools/benchmark_rounds.sh -p a2a3 -d \$TASK_DEVICE -n 100 -r tensormap_and_ringbuffer"
```

Comparison is a Welch interval on the difference of per-round `Device` means,
100 rounds per arm per case, over the harness's whole a2a3 TMR corpus.

**The two traps, both of which produced a wrong answer first.** `build_runtimes`
does not track this header (#2331), so without the cache wipe both arms run the
same binary and report no difference — that is how an earlier attempt measured a
fake 18/20-vs-17/20 null on #2233 itself. The md5 column is the positive
control, and A2 reproducing A's byte-for-byte also shows the build is
deterministic. Second, the box is shared: occupancy was 13/16 devices during A
and B and 9/16 during A2, which is why the same-arm repeat is load-bearing
rather than decorative.

## Result

`Device` wall, mean of 100 rounds, µs:

| case | A | A2 | B | A−A2 *(same binary)* | mean(A)−B |
| ---- | - | -- | - | -------------------- | --------- |
| alternating_matmul_add C1 | 814.9 | 812.4 | 814.2 | +2.5 ±7.7 | −0.5 ±6.6 |
| benchmark_bgemm C0 | 764.1 | 762.6 | 762.6 | +1.4 ±7.5 | +0.7 ±6.3 |
| paged_attention_unroll C1 | 1127.0 | 1123.3 | 1126.9 | +3.8 ±8.7 | −1.7 ±7.6 |
| paged_attention_unroll C2 | 608.4 | 604.6 | 608.8 | +3.8 ±6.9 | −2.3 ±6.4 |
| pa_unroll_manual_scope C1 | 1118.3 | 1119.6 | 1117.7 | −1.2 ±8.4 | +1.3 ±7.2 |
| pa_unroll_manual_scope C2 | 598.1 | 592.5 | 590.4 | +5.6 ±6.2 | +4.9 ±5.5 |
| batch_paged_attention C1 | 3562.0 | 3561.5 | 3570.2 | +0.5 ±19.2 | −8.5 ±18.0 |
| sliding_window_deps Dense16 | 16803.6 | 16804.1 | 16789.9 | −0.4 ±16.0 | **+14.0 ±12.3** |
| qwen3_14b_decode Stress | 34810.6 | 34841.0 | 34791.9 | **−30.4 ±11.2** | +33.9 ±10.5 |

**The drift control is the whole result.** On A vs B alone two cases look real:
qwen3 at +33.9 ±10.5 (t = 3.11) and `pa_unroll_manual_scope` C2 at +4.9 ±5.5.
Both are accounted for by the same-binary repeat — qwen3's two runs of an
identical `b9ba5d90` differ by −30.4 ±11.2, the same magnitude in the opposite
direction, and C2's same-binary delta (+5.6) exceeds its cross-arm one (+4.9).
Nine cases with one comparison each also expect about one false positive at
α = 0.05.

`sliding_window_deps Dense16` is the single case that survives the control, and
it is the case built to expose exactly this: `sliding_window_orch.cpp:76-103`
submits one AIV task per step in a strict `i → i-1` chain, so `steps=1000` puts
1001 tasks on the critical path. That gives **~14 ns per task** (95% CI ~2–26
ns), and since the hot path takes up to two barriers per task, a single
`dsb(DSB_DDR)` here is order **7–14 ns**.

The ~0.5 µs prior was off by roughly 50×, which is why nothing appeared in the
other eight cases: the effect was always below this harness's resolution. A
`dsb` costs what it has to drain, and the swimlane figure was measured where a
record write-back was outstanding.

## Why not (now)

At 7–14 ns the barrier is cheaper than the branch that would skip it, on the
same path, for every task including the ones that keep it. There is no version
of the narrowing that wins.

The broader reading matters more than this one decision: an end-to-end A/B on
this corpus cannot resolve a per-task cost below roughly 10 ns × critical-path
length, and only `sliding_window_deps` has a long enough chain to reach even
that. A future per-task micro-cost question should go straight to a long-chain
case with a same-binary repeat, and should not expect an answer from the
production workloads at all.

## When to reconsider

- If a workload appears whose critical path is much longer than 1001 tasks and
  whose per-task work is small, the barrier's share grows linearly while the
  measurement noise does not. At ~10 ns/task it takes a ~10⁵-task chain before
  the barrier is a millisecond.
- If a5's cost differs. This was measured on a2a3 only — the box's
  `onboard-arch-precheck` does not pass a5 — and the identical a5 change is
  unmeasured.
- If the barrier is ever moved somewhere with more outstanding write-back than
  a returning kernel has, the 7–14 ns does not transfer; re-measure rather than
  quoting this entry, for the same reason the swimlane figure did not transfer
  here.

## References

- \#2324 — the change measured (`OUT_OF_ORDER_STORE_BARRIER()` → `dsb(DSB_DDR)`),
  and the comment carrying this measurement
- \#2233 — the wrong result the barrier fixes; why removing it is not an option
- \#2331 — `build_runtimes` does not track platform AICore headers; the reason
  every arm here wipes `build/cache/a2a3/onboard` and checks an md5
- \#2332 — filed claiming `benchmark_rounds.sh` had a stale corpus and retracted;
  the harness was fine, a stale editable install failed all nine cases
- \#2336 — the harness now prints a failed run's output instead of only its exit
  status, which is what made #2332's misdiagnosis possible
- `tools/cann-examples/aicore-fin-ordering` (#2317, #2322) — the standalone
  producer/consumer matrix that established the ordering requirement
- [2026-07-aicore-swimlane-switch-overhead-and-ack-gate.md](2026-07-aicore-swimlane-switch-overhead-and-ack-gate.md)
  — source of the ~0.5 µs `dsb` prior this entry corrects for this site
- [`.claude/rules/running-onboard.md`](../../.claude/rules/running-onboard.md) —
  device locking and why occupancy is recorded per arm
