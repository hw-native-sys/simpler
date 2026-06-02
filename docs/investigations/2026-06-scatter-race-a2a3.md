# a2a3 column-scatter race (issue #967): PIPE_S→PIPE_V handshake missing in PyPTO codegen

**Date**: 2026-06-02
**Verdict**: simpler-side cannot fix — race lives in the PyPTO-generated
AICore kernel. Filed as a regression-guard ST in this repo;
codegen fix belongs upstream in PyPTO.

## Question

[Issue #967](https://github.com/hw-native-sys/simpler/issues/967) reports
non-deterministic data corruption in a per-row column-scatter kernel on
a2a3, bisected to PR #878. The reporter explicitly framed #878 as a
*trigger, not a root cause* (its diff is gated behind L2-swimlane
profiling, which is OFF in the failing scenario; only the recompiled
binary layout differs) and suggested looking at "the synchronization
between AICore writing scatter results and AICPU collecting task
completion". A reply from luohuan19 narrowed it to "the corruption is
contained to the AICore↔AICPU scatter-task completion path."

The symptom is highly specific: `val[14, *]` and `val[15, *]` either
vanish (output keeps the negative `base` sentinel) or land at row 0
with displacement exactly one row stride (`14 * 32 = 448` or
`15 * 32 = 480`), i.e. the per-row base offset `i * cols` in the
flattened scatter index is intermittently dropped.

The interesting question: where exactly does the race live, and what
shape of fix does it need?

## What was tried

### Step 1 — reproduce against the runtime under test

Set up a clean PyPTO checkout at
`/data/wcwxyai/workspace/pypto-issue-967/pypto`, repointed its
`runtime/` submodule at this branch, pinned pto-isa at HEAD. The exact
pytest invocation from the issue reproduces 100% on a2a3 (devices
4/6/7 via `task-submit --device auto`):

```bash
pytest -q tests/st/runtime/ops/test_scatter.py::TestScatterIndexForm::test_scatter_fp16[a2a3]
```

The first run's mismatches matched the issue's Run 1 verbatim
(`[495] actual=-31, expected=241`).

### Step 2 — empirical falsification of simpler-side hypotheses

With a deterministic 10/10-fail repro, four runtime-side hypotheses
were tested by patching `aicore_executor.cpp`, rebuilding via
`pip install --no-build-isolation -e ./runtime/`, and re-running 10
iters under task-submit. All four left the failure rate at 100%:

| Hypothesis | Patch | Result |
| ---------- | ----- | ------ |
| Kernel output writes not committed to DDR before FIN | `dsb(DSB_DDR)` before `write_reg(COND, FIN(task_id))`, gated by `__DAV_C220__` | 10/10 fail |
| Above, but the guard macro silently didn't match | `dsb((mem_dsb_t)0)` unconditional (matches `l2_swimlane_collector_aicore.h` pattern) | 10/10 fail |
| Prior task's pipe / vector-mask state leaking into the kernel prologue | `pipe_barrier(PIPE_ALL)` before `execute_task` | 10/10 fail |
| `dcci(payload, ENTIRE_DATA_CACHE)` dropping in-flight dirty output lines from the prior task | Per-cache-line `dcci(addr, SINGLE_CACHE_LINE)` looped over the 512-byte payload | 10/10 fail |

These rule out: writes lost to DDR before FIN, dirty-line eviction by
the per-task dcci, and inter-task pipe-state leakage. The race is
elsewhere.

### Step 3 — port the kernel into simpler ST, narrow to the kernel itself

Captured the saved orchestration + AIV kernel via
`pytest ... --save-kernels --kernels-dir /tmp/scatter_repro` and
dropped them verbatim into
`tests/st/a2a3/tensormap_and_ringbuffer/scatter_repro/`. Built the same
@scene_test harness used elsewhere, generating the same
base/index/val/output tensors as the PyPTO test. The simpler ST also
reproduces 10/10 fail with `max_diff = 302.0` (== `val[15, 15] = 256`
vs. `base[15, 30] = -46`), confirming the race lives entirely below
PyPTO host-side wiring and within the simpler-built AICore .o.

### Step 4 — pinpoint the race in the kernel

The saved AIV kernel has this prologue around the row-offset col
vector setup (kernels/aiv/scatter_aiv.cpp lines 92–104):

```cpp
TLOAD(v33, v37);                                      // MTE2, val tile
set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
... v38 setup ...
TCI<...>(v38, v11=0);                                 // PIPE_S writes iota [0..15] into the UB buffer @ 4640
... v40, v42 alias the same buffer ...
TASSIGN(v40, v41=4640);
TASSIGN(v42, v43=4640);
pipe_barrier(PIPE_V);                                 // NO-OP — PIPE_V has nothing in flight here
TMULS(v42, v40, v10=32);                              // PIPE_V reads v40, writes v42, scaling the iota by cols
```

`TCI` runs on **PIPE_S**, and the only barrier before `TMULS`
(PIPE_V) is `pipe_barrier(PIPE_V)`, which drains PIPE_V — nothing on
PIPE_V is pending at that program point, so the barrier is effectively
a no-op. There is no PIPE_S→PIPE_V handshake, so TMULS can fire while
TCI is still writing the high lanes of the iota. Lanes that TMULS
sees as stale read whatever was in the UB buffer before TCI started
(typically zero, either BSS-initial or a prior task's leftover),
producing `v42 = [0, 32, 64, …, 416, 0, 0]` instead of
`[0, 32, …, 480]`. `v44` is a 16×1 ColMajor view of the same buffer;
the next-step `TROWEXPANDADD(v46, v28, v44)` broadcasts the bad
col vector across rows, so rows 14 and 15 of the scatter index lose
their `i * cols` row stride. `TSCATTER` then writes
`val[14, *]` / `val[15, *]` at row 0 with column offsets matching
exactly the issue's fingerprint.

### Step 5 — confirm the fix

Added the missing handshake to `kernels/aiv/scatter_aiv.cpp`
immediately after the `TCI` call:

```cpp
TCI<...>(v38, v11);
set_flag(PIPE_S, PIPE_V, EVENT_ID1);
wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
... v40, v42 setup ...
pipe_barrier(PIPE_V);
TMULS(v42, v40, v10);
```

30 consecutive iterations under task-submit on a2a3: 30/30 pass.

## Result

- **Race located**: PyPTO-generated AICore scatter kernel,
  `TCI(PIPE_S)` → `TMULS(PIPE_V)` on the row-offset col vector buffer,
  guarded only by `pipe_barrier(PIPE_V)` instead of an S→V `set_flag`
  / `wait_flag` pair.
- **#878 is a timing trigger**: its `s_aicore_buffer_states[…]` BSS
  array and `PLATFORM_PROF_READYQUEUE_SIZE` 640→928 shift recompile
  into every AICore .o and shift code/data placement enough to move
  TMULS into TCI's hazard window. Live behavior was always equivalent
  to the parent for L2-swimlane-off runs — exactly as reported.
- **simpler cannot patch this from outside the kernel**: the
  hazard is between two instructions in the *same* kernel issued on
  two pipes. The four runtime-side patches in step 2 confirm no
  AICore-execution-loop change covers the internal-pipe-pair gap.
- **Regression guard checked in**:
  `tests/st/a2a3/tensormap_and_ringbuffer/scatter_repro/` carries
  the patched kernel so simpler's runtime is exercised on a
  column-scatter AIV-only workload. If a future runtime change
  re-introduces the race for the *patched* kernel, the test fires;
  if PyPTO's codegen ships an equivalent fix, the snapshot test
  stays representative.

## Why not (in simpler)

The bisect to #878 is genuine but misleading: it identifies the commit
whose binary-layout change pushed timing across the hazard threshold,
not the commit that introduced the race. Adding a runtime-side
workaround that broadly slows the dispatch path would (a) be brittle
to further layout shifts and (b) tax every AICore task to mask one
specific codegen bug. The clean fix is to emit the S→V handshake from
the codegen that generates `TCI` followed by a same-buffer PIPE_V
consumer.

## When to reconsider

- If PyPTO codegen is patched upstream and the
  `tests/st/runtime/ops/test_scatter.py` cases pass, mark the issue
  resolved and leave `scatter_repro/` as a snapshot regression for
  the simpler runtime dispatch path.
- If a different kernel surfaces the same shape of cross-pipe
  same-buffer hazard, this entry generalises to "audit PyPTO codegen
  for any `pipe_barrier(PIPE_X)` that immediately precedes a PIPE_X
  consumer of a buffer last written by another pipe."

## References

- Issue: [#967](https://github.com/hw-native-sys/simpler/issues/967)
- Trigger commit: `1d424912` ("Refactor: AICore-as-producer for L2
  swimlane (skip per-task staging read)", PR #878)
- Last-good parent: `1469d791`
- Regression ST:
  `tests/st/a2a3/tensormap_and_ringbuffer/scatter_repro/`
- Falsified runtime-side patches: lived only on the local fix branch
  during this investigation; reverted.
