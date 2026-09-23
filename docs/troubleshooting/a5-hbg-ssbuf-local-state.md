# A5 HBG SSBUF completion-generation mismatch

## Failure signature and scope

The initial SSBUF mailbox implementation failed on A5 at
`SchedulerErrorSite::COMPLETION_GENERATION_MISMATCH` (74), before completing
`aic_chain_64`, `aiv_chain_64`, or paged-attention-unroll `Case1`.
The host surfaced `507018` and scheduler status `INVALID_ARGS`.

The investigation used CANN 9.2.0 on Ascend950DT, with CCEC reporting
`clang version 15.0.5`, build `2026-09-08T23:37:56+08:00`.
The feature base was `72a4cc035089cae1043ac475638425f13cc8fad5`.

Cold error-path instrumentation distinguished scheduler-local completion
tokens from SSBUF tokens. One failure had a local completion token of 1,
an SSBUF token of 0, and a local slot in `EMPTY` with generation 0.
That is inconsistent local scheduler state, not evidence that a remote
Executor published a bad SSBUF token.

## Controlled experiments

The following AIC-chain experiments kept the SSBUF region, mailbox protocol,
and payload location unchanged. Each header edit forced recompilation of
the executor translation unit.

| Trace payload alignment | Dispatch-loop boundary | Result |
| ----------------------- | ---------------------- | ------ |
| 64 bytes | Original 10 arguments | Generation mismatch |
| Natural, 8 bytes | Original 10 arguments | AICore fault followed by scheduler timeout |
| 64 bytes | Forced inline | Generation mismatch |
| Natural, 8 bytes | Forced inline | Passed |
| Natural, 8 bytes | 8 arguments, base/lane derived inside | Passed |

A separate CCEC compilation probe compared direct `__ssbuf__` scalar-pointer
stores with equivalent structure-member stores. Both generated the same
address translation and store operations. A general prohibition on SSBUF
structure pointers is therefore not supported by this evidence.

The experiments identify a local-layout/call-boundary-sensitive failure in
this compiled implementation and a working source-level workaround. They do
not identify a specific faulty compiler pass, prove that every over-aligned
local object is broken, or establish an eight-argument hardware limit.

## Implementation contract

- `SchedulerExecutorTaskTrace` is a naturally aligned 48-byte payload. Its
  local copies do not impose cache-line alignment on `SchedulerLocalState`.
- `SchedulerSsbufLane` supplies shared-storage alignment: trace slots remain
  at lane offsets `0x40` and `0x80`, both 64-byte aligned.
- `run_ready_dispatch_loop` has seven arguments, including one local-state
  pointer carrying configuration and mutable state. The SSBUF base is derived
  inside the function, and the cluster lane comes from the local configuration.
- The HBG reservation remains the final contiguous KiB, `[2048, 3072)`, of
  the cluster SSBUF. Its header, three lanes, and reserve remain
  64 + 3 × 256 + 192 bytes. `DispatchPayload` remains in GM.
- SPSC token accesses retain their volatile and DSB ordering. No atomic
  read-modify-write operation is introduced in SSBUF.
- Each lane's completion generations are packed into one aligned
  64-bit publication word. Its Executor keeps a full-word shadow; each Scheduler
  scan loads one snapshot per remote lane. Dispatch publication packs generation
  and signed timing slot into another 64-bit word; task ID remains separate.
- Polling misses have no DSB. A completion hit acquires before reading trace
  data. Dispatch writeback and SSBUF control share one release DSB before ready;
  Executor payload invalidation and ready acquire share one DSB before consumption.
- Metadata and callable addresses are immutable after startup publication.
  After READY acquire, local configuration caches three worker IDs and a common
  payload-region offset, immutable region offsets, scheduler index/count,
  completion-ID stride, and self/scheduler lanes. All discovered cluster lanes
  participate; core-type checks read the shared worker context.
  Graph metadata reuses the existing graph view; startup trace timestamps remain
  lazily observed after Executor publication. Mutable wake-list links still require invalidation and a DSB.
- Resolved counts accumulate locally across scheduling passes, including
  deferred AIV completion. Publication occurs only when a pass has no progress,
  no local executable task, and no deferred reservation, or in the common run
  epilogue. Remote tasks may still be running. Trace/error publication precedes
  the count, and each flush clears its local accumulator. The AICPU no-progress
  watchdog remains 20 seconds by default: continuously busy work without any
  count publication for that interval can time out. No periodic forced flush
  interrupts the busy path; the device op timeout remains 45 seconds.
- `scheduler_refresh_ready_inbox` publishes pending owner work and updates the
  directory's stealable-work bits. Locally empty types require no refresh.
  A type's directory query is skipped when no unreserved FREE slot can
  accept work, respecting the direct-refill skip mask and deferred AIV capacity.
- Profiling loop IDs are core-local. Bootstrap records and the first main-loop
  records retain loop ID zero. The shared context has no `profiling_loop_iter`
  or redundant `inbox_index`; scheduler index identifies its ready inbox.
  Explicit cache-line group alignment preserves the shared context's layout
  boundaries without replacement reserved fields.
- Idle polling backs off from 8 to at most 32 loop iterations, resetting on
  progress. The profiling-off serial AIC handoff has five protocol DSBs:
  completion release, completion acquire, mutable next-waiter invalidation,
  payload/ready release, and ready/payload acquire. Startup, profiling, and
  kernel-internal barriers are outside that count.
- Executor trace fields are copied to SSBUF only when chip-swimlane or sampled
  task timing is enabled.
- Dispatch trace GM writes follow ready publication.
  The ready timestamp occupies the existing dispatch-end trace field; the Host
  computes dispatch-to-kernel latency without an Executor-side GM timestamp read.
  The JSON schema and the 1 KiB reservation are unchanged.

## Regression coverage

The final implementation passed ten hardware rounds with golden checking
for each of the three target cases. The single-core-DAG scene suite also
passed with chip-swimlane level 3, exercising shared trace publication and
the scheduler's trace consumption. All 152 non-hardware C++ test targets
passed, including checks for naturally aligned local state and aligned
SSBUF trace slots.

The full A5 simulation sweep passed with GCC 15.2.0 and its matching
`libstdc++`. On hardware, the HBG sweep passed 27 tests with one skipped;
the other runtime passed 48 tests. Five resource-phase cases initially
failed to load CANN libraries and all passed when retried after sourcing
CANN's `set_env.sh`. `pypto-setup --export` alone did not populate the
required dynamic-library search path in this shell.

The onboard cases are important: host-only layout tests cannot detect a
CCEC code-generation-sensitive failure. To reproduce the targeted checks:

```bash
python tests/st/a5/host_build_graph/single_core_dag/test_single_core_dag.py \
  --platform a5 --device "$TASK_DEVICE" --rounds 10 --case aic_chain_64
python tests/st/a5/host_build_graph/single_core_dag/test_single_core_dag.py \
  --platform a5 --device "$TASK_DEVICE" --rounds 10 --case aiv_chain_64
python tests/st/a5/host_build_graph/paged_attention_unroll/test_paged_attention_unroll.py \
  --platform a5 --device "$TASK_DEVICE" --rounds 10 --case Case1 --manual include
python -m pytest tests/st/a5/host_build_graph/single_core_dag \
  --platform a5 --device "$TASK_DEVICE" --enable-chip-swimlane 3 --manual include
```

Run the architecture precheck before taking the `task-submit` allocation;
execute these commands inside that allocation with the worktree's venv
activated. See [Testing](../testing.md) for rebuild requirements.

## Performance cross-check

Both orders ran on device 3, inside exclusive `task-submit` allocations:
100 rounds per case and revision, chip-swimlane disabled, golden checking
disabled after the correctness runs above. The base was `72a4cc03`;
both used PTO-ISA `3b4faf67aebb3e0d41be7952c56908b3adba7a8f`.
The fixed side includes the complete SSBUF feature, not just the local-state
workaround.

The table reports the built-in `strace_timing --rounds-table` trimmed
averages: 10 low and 10 high samples discarded, 80 samples retained.

| Order | Case | Host base → fixed (µs) | Device base → fixed (µs) | Device change |
| ----- | ---- | ---------------------- | ------------------------ | ------------- |
| Base then fixed | AIC chain | 856.8 → 4060.5 | 520.3 → 512.0 | −1.60% |
| Base then fixed | AIV chain | 902.0 → 4095.8 | 547.7 → 507.3 | −7.38% |
| Base then fixed | PA Case1 | 33230.7 → 33392.8 | 995.9 → 985.1 | −1.08% |
| Fixed then base | AIC chain | 847.4 → 989.9 | 515.2 → 482.9 | −6.27% |
| Fixed then base | AIV chain | 4126.2 → 1232.4 | 528.8 → 513.1 | −2.97% |
| Fixed then base | PA Case1 | 31931.0 → 32471.8 | 994.0 → 982.0 | −1.21% |

The chain Host regressions in the first batch exceed 5% and must not be
hidden by the lower Device numbers. On reversal, current AIC Host fell
from 4060.5 to 989.9 µs with no code change, while baseline AIV Host rose
from 902.0 to 4126.2 µs. Host timing is therefore unstable in this session;
these measurements do not establish an end-to-end improvement or a
source-level cause for the Host differences.

AIV Device time improved in both orders, by 3.0–7.4%. AIC Device time also
fell, but its magnitude varied from 1.6% to 6.3%. PA's 1.1–1.2% change is
within the benchmark's ±2% noise band. Do not claim a stable percentage
gain for all three cases from these two batches alone.

Raw logs and per-round summaries are retained locally in
`tmp/ssbuf-fix-benchmark/` and `tmp/ssbuf-fix-benchmark-reverse/`.

## Mailbox protocol optimization validation

The layout-v2 synchronization changes were compared against `0b80c0f0`
on device 0 in one exclusive allocation, baseline first, with the same
PTO-ISA pin above. Each case ran 100 rounds with chip-swimlane disabled.
These are trimmed averages (10 low and 10 high samples removed).

| Case | Host baseline → optimized (µs) | Device baseline → optimized (µs) | Device change |
| ---- | ------------------------------ | -------------------------------- | ------------- |
| AIC chain | 17572.0 → 3929.5 | 522.4 → 439.5 | −15.9% |
| AIV chain | 2241.6 → 848.2 | 510.8 → 454.0 | −11.1% |
| PA Case1 | 33847.3 → 32475.3 | 979.4 → 928.8 | −5.2% |

Host timing varied substantially on this shared host; these measurements do
not establish a repeatable Host speedup. Device results cover one ordered
comparison, not a cross-device or order-reversed study.

Separate level-3 captures keep the Executor-to-Executor handoff definition:
the previous task's kernel-end timestamp to the dependent task's kernel-start
timestamp, including every intervening operation.

| Profiled metric (µs) | AIC baseline → optimized | AIV baseline → optimized |
| -------------------- | ------------------------ | ------------------------ |
| Executor end → next start, mean over 63 edges | 7.723 → 7.199 | 7.514 → 7.321 |
| Ready publication → kernel start, median | 0.885 → 0.219 | 0.935 → 0.249 |
| Kernel end → Scheduler completion observation, mean | 2.584 → 3.615 | 2.467 → 3.851 |

Profiling remains intrusive: GM trace output after ready publication overlaps
the next kernel and still occupies the Scheduler. The larger completion
observation gap in these captures must not be presented as an improvement,
or used to infer profiling-off SSBUF latency. The baseline kernel-start
timestamp also precedes its online GM timestamp read; the optimized Executor
does not perform that read. Raw JSON columns and schema remain compatible.

Validation passed: 152 non-hardware C++ targets, 73 swimlane Python tests,
22 HBG simulation tests, the single-core DAG simulation suite with level-3
profiling, and 29 HBG onboard tests with one skip. Each target performance
case also passed ten golden-checked hardware rounds and a level-3 capture.
All changed files passed pre-commit checks.

CCEC disassembly shows 64-bit SSBUF token loads/stores and eight payload
DCCI instructions followed by one DSB before the SSBUF task-ID load.
Miss branches do not execute that DSB. Logs, timings, baseline captures,
and disassembly are retained in `tmp/ssbuf-opt-evaluation/`; current captures
are under `outputs/`.
