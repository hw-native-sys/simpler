# DFX Buffer Capacity Audit (a2a3)

> Cross-subsystem buffer-capacity audit of the five a2a3 DFX profiling
> collectors (l2_swimlane / dep_gen / pmu / scope_stats / args_dump), with a
> full **data → theory → measurement → conclusion** argument per verdict.
> The right-sizing it recommends is landed in this PR; a2a3 onboard-validated
> zero-drop. Responds to
> [simpler#977](https://github.com/hw-native-sys/simpler/issues/977).

## 1. TL;DR

Across 6 tensormap_and_ringbuffer examples, **only the bare no-dependency flood
micro-benchmark drops; every real workload is zero-drop.** Right-sizing the
over-provisioned pools frees **~66 MB, zero ABI**.

| Subsystem | Verdict | Action | Memory |
| --------- | ------- | ------ | -----: |
| **l2_swimlane** | phase pools 2–16× over (task pools at the edge, leave) | split `BUFFERS_PER_THREAD` → `SCHED 16→6` / `ORCH 16→8` | phase 72 → **28 MB** |
| **pmu** | over, free margin 75%, L≈1 | `BUFFERS_PER_CORE` **4→2** | 9.2 → **4.6 MB** |
| **scope_stats** | over, free margin 87.5%, L≈1 | `BUFFERS_PER_INSTANCE` **8→4** | 0.21 → **0.11 MB** |
| **dep_gen** | drops are **rate-limited, not capacity** | `RECORDS_PER_BUFFER` → **1024** (corrects a 2048 overshoot) | frees ~17.5 MB |
| **args_dump** | adequate (0 overwrite across 6 cases) | **unchanged** | — |

**Two counterintuitive findings** (argued in §6): ① drops are set by the
**dispatch pattern (burst intensity)**, not submit count; ② dep_gen drops are
**unfixable by buffer sizing** (rate-limited).

## 2. Criterion & Method

The 5 subsystems share an **SPSC + ProfilerBase** contract: the device writes
records into a buffer, rotates when full and pops an empty buffer from the
free_queue; the host drain/replenish mgmt path concurrently returns
collector-done buffers to recycled lanes and refills free queues from those
lanes. Modules can also ask the replenish thread to keep recycled lanes above
host-side watermarks by batched allocation; only the drain path writes device
free queues. **When a buffer is full and the free_queue is empty, the record
is dropped** (no back-pressure on the dispatch hot path — a deliberate
invariant).

- **Criterion**: each pool's `configured / actually-needed` ratio should be
  roughly consistent across subsystems — over-provisioning wastes memory,
  under-provisioning drops. Goal: **minimize memory subject to zero drop.**
- **Loads**: 6 examples covering flood / manual scoping / GEMM / real decoder.
  Worst case = bare flood `paged_attention` Case1 (~65536 submits).
- **Two watermarks** (printed at reconcile; pure host diagnostic, no device ABI):
  - **`free margin = min_free_depth / SLOT_COUNT`** — how much of the
    host→device empty-buffer supply queue remains; hitting 0 drops.
  - **`ready margin = 1 − max_ready_depth / READYQUEUE_SIZE`** — how much of the
    device→host full-buffer recycle queue remains; hitting 0 also drops.
  - **Weakest link**: whichever hits 0 first drops, so the **verdict reads the
    tighter dimension.**

## 3. Theory: Little's Law

### 3.1 Why `L = λ×W` is independent of the configured count

Each collector is an SPSC producer-consumer (device borrows buffers to write,
host drains and returns them). The steady-state in-flight (borrowed-not-yet-
returned) buffer count is given by Little's Law:

```text
L = λ × W            needed BUFFERS ≈ L + burst margin
```

- **L** = buffers borrowed by the device but not yet returned by the host =
  **the capacity actually needed.**
- **λ** = buffer production rate = record rate ÷ `RECORDS_PER_BUFFER`.
- **W** = host recycle latency for one buffer (drain bytes ÷ bandwidth + mgmt).

**Argument**: at steady state "borrow rate = return rate", and a buffer stays
out for an average of W, so the instantaneous in-flight count is λ×W. This
quantity is set purely by "how fast records are produced × how fast the host
drains" — **independent of how many buffers you preallocate.** Hence
`configured ÷ L = headroom`; headroom ≫ 1 means over-provisioned and cuttable.

### 3.2 Applied per subsystem (theory; §4–5 validate against measurement)

| Subsystem | λ | W | **theory L** | config | headroom |
| --------- | - | - | -----------: | -----: | -------: |
| pmu | low (sampling, decoupled from submit) | small (64B, instant drain) | **≈1** | 4 | 4× |
| scope_stats | low (per scope enter/exit) | small (52B) | **≈1** | 8 | 8× |
| l2 task | high (one per task, flood is huge) | small (32B) | **≈4** | 4/8 | 1–2× |
| l2 phase | high | small W **but buffer is huge (16384)** | T_fill ≫ W → **≪16** | 16 | several× |
| dep_gen | high (per submit, AICPU at full speed) | **large** (4672B record) | **explodes** under flood | 4 | far short |

Reasoning per row:

- **pmu / scope**: λ low (production is **decoupled** from submit — pmu samples
  at a fixed rate, scope fires on enter/exit, neither tracks the flood) × W small
  (small record drains instantly) → L≈1. Configured 4 / 8 is a 4–8× over-provision;
  production is steady (non-bursty) so peak ≈ steady L ≈ 1 → can cut to 2 / 4 and
  still keep 2–4× safety.
- **l2 task**: λ high (flood, one record per task) × W small → L≈4 = config →
  **at the edge, leave.**
- **l2 phase**: viewed differently — the per-buffer capacity 16384 is enormous, so
  filling one takes `T_fill = 16384/λ`, far larger than W (recycle latency). So the
  number of phase buffers simultaneously in flight is ≪ 16. Configured 16 is over.
- **dep_gen**: λ high × W **large** (4672B record drains slowly) → under flood L
  exceeds any realistic config → **sizing can't fix it** (proven via the dynamic
  form `drop ∝ (λ−µ)T` in §5.2).

## 4. Measurement

### 4.1 Cross-example matrix + margin stats

Format `lowest-free / peak-ready-backlog` (raw counts); `dropN`; `ovf` = tensor
overwrite count.

| Example | dep_gen | l2_swim | pmu | scope | dump |
| ------- | ------- | ------- | --- | ----- | ---- |
| paged_attention (**bare flood**) | 3/4 **drop31744** | 0/41 | 3/20 | 7/1 | 3/1 ovf0 |
| paged_attention_manual_scope | 3/1 drop0 | 0/45 | 3/21 | 7/1 | 3/5 ovf0 |
| paged_attention_unroll_manual_scope | 3/1 drop0 | 0/49 | 3/21 | 7/2 | 3/1 ovf0 |
| benchmark_bgemm | 3/1 drop0 | 0/33 | 3/17 | 7/1 | 3/0 ovf0 |
| paged_attention_ringbuffer | 3/1 drop0 | 0/25 | 3/12 | 7/1 | 3/1 ovf0 |
| qwen3_14b_decode | 3/1 drop0 | 0/23 | 3/11 | 7/1 | 3/1 ovf0 |

Converted to worst-case margins (`free = lowest/cap`, `ready = 1 − peak/cap`,
verdict reads the weaker dimension):

| Subsystem | free margin | ready margin | verdict |
| --------- | ----------- | ------------ | ------- |
| pmu | 3/4 → **75%** | 93% | **over** |
| scope_stats | 7/8 → **87.5%** | 87% | **over** |
| l2_swimlane | **0/4 → 0%** | 95% | **at edge** (free bottomed, but zero-drop) → §4.2 |
| dep_gen | 3/4 → 75% | 50% | **rate-limited**: both have room yet it still drops (§5.2) |
| args_dump | 3/4 → 75% | 92% | **adequate** (no overwrite) |

### 4.2 l2_swimlane: the four pools in detail

l2_swimlane is not one pool — 4 **heterogeneous** buffer pools, all reusing the
same `ActiveHead(64B) + FreeQueue(128B)` three-party pipeline (Host refills empty
buffers → device writes & rotates → Host recycles & drains); both `free_queue`
and `ready_queue` are SPSC rings of depth `SLOT_COUNT=4`. **A single buffer does
not wrap; the pool does** (rotate + pool → arbitrarily long sessions never wrap).

| Pool | unit | writer | produce trigger | capacity (REC×BUF) | record | memory (before→after) |
| ---- | ---- | ------ | --------------- | ------------------ | ------ | --------------------- |
| **AicpuTask** | per-core | AICPU | per task completion | 1000 × 8 | 32B | ~18 MB (unchanged) |
| **AicoreTask** | per-core | **AICore** writes, AICPU rotates | per task dispatch | 1024 × 4 | 32B | ~9 MB (unchanged) |
| **Sched phase** | per-thread×4 | AICPU | per scheduler-loop phase | 16384 × 16 | 64B | 64 → **24 MB** |
| **Orch phase** | per-thread×1 | AICPU | per submit | 16384 × 16 | 32B | 8 → **4 MB** |

Structural differences that drive each pool's behavior:

- **Different writers**: AicpuTask is written by AICPU on the completion path;
  AicoreTask is written by **AICore itself** and AICPU **rotates it on the
  dispatch path** (the only coupling to slot dispatch, hooked just before
  `write_reg(DATA_MAIN_BASE)`, lock-free thanks to the completion-before-dispatch
  invariant). Their record counts are identical (1 dispatch ≈ 1 execution), so
  their peaks match.
- **Full → rotate, can't rotate → drop + reconcile accounting** (never blocks the
  kernel). But the drop differs: AicpuTask on empty free_queue overwrites the
  current buffer and bumps `dropped`; AicoreTask on empty free_queue does **not**
  rotate and the slot guard **silently drops** (does not bump `dropped` — the lost
  count is computed by the host at reconcile, avoiding double-counting with the
  flush retry).
- **Phase buffers are huge**: `PHASE_RECORDS_PER_THREAD=16384` → sched 1MB/buf,
  orch 512KB/buf, 16–32× the task pools (32KB). This is why phase dominates memory
  (72%) and is the mechanism behind the free=0 below.
- **Sched allocates 4 threads** (`MAX_AICPU_THREADS`): under the 1:3 topology only
  3 produce; the 4th is the orch→sched conversion reserve (measured empty). Orch is
  a **single instance.**

### 4.3 Per-pool watermarks (the global watermark misleads)

A global `min_free_depth=0` only says "some pool bottomed out", with no
attribution. **Split per pool (consistent across 4 runs):**

| Pool | min_free_depth | throughput | T_fill | verdict |
| ---- | -------------- | ---------- | ------ | ------- |
| AicpuTask | **1** | — | 9.5 ms | did not bottom out, leave |
| AicoreTask | **1** | — | 9.7 ms | **did not bottom out**, leave |
| Sched phase | **0** | 1/16 (whole session < 1 buffer) | 56.8 ms | bottomed but zero-drop → **over** |
| Orch phase | **0** | 5/16 (~5 full rotations) | 12.5 ms | bottomed but zero-drop → **over** |

> Measured λ (flood, 50.4 ms window): task 105k, sched 288k, orch 1.31M rec/s.
> `T_fill = capacity / λ` = time to fill one buffer.

**The reversal, argued**: the old verdict recorded l2 as "at the edge, leave"
(because the *global* free=0). Per-pool measurement proves **free=0 belongs to the
phase pools, not the task pools.** Task pools hold a steady 1 slot; only phase
bottoms out. And the phase bottom-out is not a capacity shortage: phase buffers
are huge (1MB/512KB), the host copy is slow → **large recycle latency W** →
free_queue drains; but phase `T_fill` (12–57 ms) far exceeds the inter-arrival
gap, so recycling always beats the next arrival → **zero drop.** So phase is a
benign "big-buffer-slows-recycle" bottom-out, and its **buffer count is actually
over-provisioned.**

## 5. Per-subsystem argument

### 5.1 pmu / scope: over-provisioned, cut

- **Theory**: §3.2 derives L≈1.
- **Measurement**: `free` is a steady pmu 3/4, scope 7/8 across all 6 examples
  (peak borrow only 1, confirming L≈1); production is steady, non-bursty.
- **Landed + validated**: cut pmu `BUFFERS_PER_CORE 4→2`, scope
  `BUFFERS_PER_INSTANCE 8→4`; reran the 6 examples onboard, **zero-drop**
  (min_free=2 / 4, keeping 2–4× margin). The theoretical L≈1 went from
  extrapolation to measured.

### 5.2 dep_gen: rate-limited drops, sizing can't fix

dep_gen records are produced in `submit_task`, so the rate = the **dispatch
rate** (independent of whether AICore finished computing). Drops are set by single
burst intensity, dynamic form:

```text
drop ≈ (P − R) × T − N × S      P=produce rate, R=host drain rate, N×S=total buffer capacity
```

- **Guaranteed capacity** (no host drain at all, still no drop) = `BUFFERS ×
  RECORDS` = 4×1024 = 4096 submits.
- **Doubling any of three knobs leaves drops unchanged** (proving `N×S` is
  negligible vs the flood deficit):

  | config | total in-flight | total memory | drop |
  | ------ | --------------: | -----------: | ---: |
  | 4×2048 SLOT4 | 8192 | 36.5 MB | 24576 |
  | 8×2048 SLOT4 | 16384 | 73 MB | 26624 |
  | 8×2048 SLOT8 | 16384 | 73 MB | 24576 |
  | batch=16 (4096 submits/burst) | — | — | **0** |

- **The only variable is R** (host drain rate), pinned by record size (4672B) +
  bandwidth.
- **Corollary**: a real layer-serial decoder (qwen3: 40 residual layers, layer
  N+1 waits on N) has `P<R` and is already zero-drop → **dep_gen needs no
  enlargement.** Surviving a 65536 flood would need ≥146 MB in flight, paid for a
  workload that doesn't exist. So it stays at 1024 (cohort-aligned), correcting the
  2048 overshoot.

### 5.3 l2_swimlane: phase over-provisioned → cut count

- **Verdict** (§4.3): phase pools over-provisioned (throughput 1/16, 5/16), task
  pools at the edge, leave.
- **Over-provisioning is in count, not size**: sched's 16384-rec buffer is 89%
  full over the whole session and orch fully rotates ~5 times, so the per-buffer
  **size is in use**; and a larger buffer means fewer rotations (lower rotation
  overhead) → **size is untouched, only count is cut.**
- **Split the constant**: sched/orch throughput is asymmetric (1 vs 5 buffers), so
  a single shared value over-provisions the lighter one → split into `SCHED 16→6`
  / `ORCH 16→8` (different values are what makes the split worthwhile).
- **Zero ABI**: `READYQUEUE_SIZE` sizes the device-visible shm header, so it is
  decoupled and pinned at the original 16; only the host preallocation count moves.
- **Validation**: onboard flood `dropped=0`; per-pool `min_free_depth` identical to
  before the cut (free=0 is recycle-latency-bound, not capacity, so cutting count
  does not worsen it). phase 72→28 MB.

### 5.4 args_dump: adequate, leave

All 6 examples (including the bare flood) show `ovf=0` (no arena overwrite) and a
steady free of 3/4. The old "arena overwrite under-provisioned" claim is refuted by
the new data. Re-evaluate if a genuinely large-dump workload appears.

## 6. Cross-cutting mechanisms

### 6.1 Drops are set by dispatch pattern, not submit count

dep_gen drops 31744 only on the single "bare flood" `paged_attention`;
`manual_scope` / `unroll` with the **same batch=256** are zero-drop — manual
scoping breaks the flood into small bursts, erasing the `(P−R)×T` deficit. **So
burst intensity sets the drops, not the total.**

### 6.2 Multiple collectors at once: the bottleneck is host concurrency

| run mode | dep_gen | l2 | pmu | scope |
| -------- | ------: | -: | --: | ----: |
| any one / any two together | 0 | 0 | 0 | 0 |
| all four together | 32–51k | 14–30k | 1.5–7.7k | 12–20k |

1–2 collectors: the host keeps up, zero-drop. All four together: each collector's
mgmt+poll threads split host CPU/bandwidth → recycling slows → free_queue drains
repeatedly → drops everywhere, with large run-to-run variance (write-out race).
**No device buffer size helps** — the bottleneck is host concurrency, needing a
separate fix (fewer threads / merged mgmt / scheduling priority).

### 6.3 Metric pitfall + W is not one-size-fits-all

- `min_free_depth` only measures the free_queue drop path and its ~10µs tick
  misses transient bottom-outs → **trustworthy only when `dropped=0`.** Subsystems
  that do drop (dep_gen) must read `max_ready_depth` + the throughput model.
- **W is not one-size-fits-all**: in l2, plain theory with a uniform conservative
  W=10ms mis-judged AicoreTask as at-the-edge (needed 5 > configured 4); **per-pool
  measurement refuted it** — AicoreTask's buffer is small, recycle is fast, real
  W≪10ms, and it holds a steady 1 slot. **W must be measured per pool**, else
  small-buffer pools are overestimated and big-buffer pools underestimated.

## 7. Open items

- **l2 sched can be cut further**: at 1/6 it still has slack, theoretically down to
  ~4 (another ~8 MB), but 4 = SLOT_COUNT is the floor and needs a "peak in-flight"
  measurement + a dependency-heavy example to confirm the rotation case first. This
  round stops at 6.
- **Multi-collector concurrent drops (§6.2)**: a host-side problem, not addressed
  here; needs a separate fix.
- **a5**: a separate copy from a2a3; this audit is a2a3-only, a5 mirrors the same
  structure with a5-silicon validation pending.

## Appendix: config constant reference

| Subsystem | size constant | num constant | record | unit |
| --------- | ------------- | ------------ | -----: | ---- |
| l2 AicpuTask | `PROF_BUFFER_SIZE` (1000) | `PROF_BUFFERS_PER_CORE` (8) | 32B | per-core |
| l2 AicoreTask | `AICORE_BUFFER_SIZE` (1024) | `AICORE_BUFFERS_PER_CORE` (4) | 32B | per-core |
| l2 phase | `PHASE_RECORDS_PER_THREAD` (16384) | `PROF_{SCHED,ORCH}_BUFFERS_PER_THREAD` (6/8) | 64/32B | per-thread |
| pmu | `PMU_RECORDS_PER_BUFFER` (512) | `PMU_BUFFERS_PER_CORE` (2) | 64B | per-core |
| dep_gen | `DEP_GEN_RECORDS_PER_BUFFER` (1024) | `DEP_GEN_BUFFERS_PER_INSTANCE` (4) | 4672B | per-instance |
| scope_stats | `SCOPE_STATS_RECORDS_PER_BUFFER` (512) | `SCOPE_STATS_BUFFERS_PER_INSTANCE` (4) | 52B | per-instance |
| args_dump | `PLATFORM_DUMP_RECORDS_PER_BUFFER` (256) + arena | `PLATFORM_DUMP_BUFFERS_PER_THREAD` (8) | 128B+tensor | per-thread |

All prefixed with `PLATFORM_`. a2a3 and a5 are independent copies; this report
covers a2a3 only.
