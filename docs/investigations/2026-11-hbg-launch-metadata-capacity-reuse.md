# HBG launch metadata capacity reuse

Date: 2026-11-13. Decision: snapshot-capacity candidate reverted; no net performance benefit established. The separate result-pool experiment remains unaccepted.

## Scope and mechanism

Issue #2883 classifies Delivered, Scratch and Reported resources. Performance is an additional acceptance requirement for this experiment. The published stack is `01f97b0e5`; its merge-base is `b01e5bdd8`, not latest main.

`RuntimeLaunchMetadata` captures host metadata and copies the captured bytes into slot-owned pinned DMA storage. A single 362240-byte image allocates a region vector and a byte vector on every iteration. The candidate used contiguous owned snapshot bytes plus a spare buffer, keeping prepare failure atomic and publication sources independent of caller mutation. Both full-size copies remained.

Capacity must return to the slot after publication: preparation moves metadata into `KernelArgsHelper`. Caching only inside the class fails to reuse capacity at the next slot prepare. The final candidate explicitly returned detached capacity after publication without changing pinned DMA source lifetime. A test exercising slot-to-launch-to-slot movement confirmed allocation-free warm iterations, including appended metadata.

## Validation

Snapshot unit tests: 6 passed, including allocation failure, caller mutation, insufficient pinned capacity, partial publication failure, reentrant preparation, move semantics and capacity return. Rebuilt HBG admission targets: A3 165 passed, A5 169 passed; architecture-inapplicable cases skipped. Rebuilt TMR temporary-buffer targets: 42 passed per architecture. Onboard runtime rebuilt and callable precompiled before device allocation.

## Hardware result

The final candidate also includes the separately tested bounded ReportedPool cache. Therefore these results assess the combination against the merge-base, not snapshot reuse in isolation. Same-card alternating baseline/candidate processes, six pairs per depth, 16 warmup plus 128 measured batches. Primary metric: host submit plus wait per completed run, including Python and IPC. Workload: 512-task vector chain with 16384 elements; no model throughput claim. Depth one used device 7, depth four device 6. Every final-round process passed golden checks.

| Depth | Paired median latency change | Individual pair changes (%) |
| ----- | ---------------------------- | --------------------------- |
| 1 | +0.83% | -1.22, +2.86, +1.59, +3.48, +0.06, -2.20 |
| 4 | -0.23% | +11.91, +3.74, -2.26, -3.34, -2.84, +1.81 |

Negative means faster. Neither depth establishes a repeatable net improvement. The +11.91% pair is retained, not discarded. Fewer allocations alone do not justify added retained host memory and lifecycle machinery. The snapshot candidate was reverted; do not reintroduce it based solely on allocation counts.

## Failed baseline run and evidence

An earlier depth-four sequence failed on the unmodified merge-base at run 306: `RunPreparationRefused: prepare_native_run requires an active predecessor before staging a successor`. Cleanup then emitted live-host-export errors and exited 139. Root cause remains unresolved; the final retry-free comparison does not fix or explain it. The failed log and initial pairs are preserved separately; a complete new sequence was run under new tags.

Local artifacts under `reports/issue-2883/performance-20261113/`: `metadata-v2-candidate.patch`, `meta-v2-summary.json`, `meta-baseline-summary.json`, raw `meta-v2-*` samples/logs, and `meta-baseline-base-d4-r3.log`. The patch includes the separate pool experiment; isolate the metadata files when reviewing this candidate. No PR was changed by this experiment.

Revisit with evidence that eliminating an entire image copy or synchronization changes the critical path, followed by a representative model benchmark. Buffer ownership and asynchronous DMA lifetime remain requirements.

## Follow-up: Qwen model validation of the pool-only stack

After reverting metadata reuse, the onboard runtime was rebuilt. Compared `b01e5bdd8` against `01f97b0e5` plus the local bounded ReportedPool candidate using in-repo HBG Qwen3-14B, 40 layers, batch 16, sequence 3500. This comparison does not contain the rejected metadata optimization. Kernel compilation, fixed input generation and CPU golden computation preceded device allocation. All six processes used device 0; 64 rounds each, first 16 excluded. Outputs and both KV caches matched the precomputed golden after each process's measured rounds.

| Pair | Baseline worker.run median (ms) | Candidate median (ms) | Change |
| ---- | ------------------------------- | --------------------- | ------ |
| 1 | 37.459711 | 37.499962 | +0.107% |
| 2 | 37.504602 | 37.614993 | +0.294% |
| 3 | 37.314294 | 37.575807 | +0.701% |

Paired median change: +0.294%, so no net speedup established. Three process pairs do not reliably characterize a sub-percent regression. These are fixed-address L2 decode calls, not serving TPOT or a multi-run pipeline throughput test.

Without extra bind instrumentation, candidate image publication medians were 6.815–7.285 us, bind 158.066–172.057 us, and device execution 36.719–36.867 ms. The model is dominated by device execution; removing only a few microseconds of host staging cannot provide a large overall gain here. Stage medians are not additive critical-path attribution, and host publication timing is not asynchronous H2D completion timing.

The separate instrumented candidate run also passed golden and is retained in `profile-summary.json`; do not mix its timings into the above A/B. Local artifacts: `reports/issue-2883/qwen-validation-20261113/{summary.json,provenance.json,profile-summary.json}` and all raw samples/logs. Initial comparison task was cancelled while pending by an auxiliary waiter with a default 600-second queue timeout; retry task `task_20261113_203539_56790819629` completed successfully using one waiter. No PR was pushed by this validation.

## Follow-up: capacity and queue-layout controls

The pool-capacity experiment was also removed from the final candidate: fewer
allocations did not establish the required whole-stack latency result. The current
acceptance is measurable memory reduction and a representative latency regression
of at most 1%, with uncertainty resolved before passing; speedup is optional.

A controlled six-pair PR-boundary experiment first found PR2/PR1 latency +1.427%
(95% paired-bootstrap interval +1.063% to +2.092%). A subsequent balanced
three-arm experiment did not reproduce that signal: unchanged PR2/PR1 +0.233%
(-0.346% to +1.918%). Restoring only fixed-limit queue backing gave -0.254%
relative to PR2 (-1.661% to +0.536%), while raising committed memory from
35,645,685 to 42,698,485 bytes. This intervention is rejected: it does not establish
a performance improvement and loses the memory benefit. Neither experiment
proves the cause of the variable latency difference.

These experiments use the historical published stack. Local integration on
`cbeeb4682` removes the additional admission budget gate and adapts to upstream's
removal of the legacy A5 scheduler pools. Integrated performance acceptance is a separate gate; historical measurements
do not certify that tree. The copied-zone
alignment fix has a separate deterministic failing/passing unit regression and
is not claimed as a measured speedup.

Raw samples, scripts, patches and summaries are retained under
`reports/issue-2883/acceptance-20261113/{boundaries,queue-control,integrated}`.

## Integrated acceptance on current main

The measured stack is `e8db3b4c3`, based on `cbeeb4682`; later delivery edits
change this investigation document only. Runtime binaries stayed frozen through
the comparisons. The metadata-capacity, expanded result-cache and wide queue
backing experiments are absent. The copied-zone alignment fix is included.

The first 12 alternating process pairs per vector depth were inconclusive:
depth one median +0.382%, 95% paired-bootstrap interval [+0.130%, +1.206%];
depth four +0.672%, [+0.200%, +1.115%]. Before collecting further data, each
depth was assigned one independent confirmation of exactly 36 new pairs, with
no additional repetition if still inconclusive. The confirmations are evaluated
separately from the initial samples. Every pair, including a +9.207% depth-one
pair, is retained; no outliers are removed.

- Depth one confirmation: median +0.316%, 95% interval [-0.014%, +0.693%].
  Runtime committed memory is 42,685,509 to 35,632,709 bytes, down 16.5%.
- Depth four confirmation: median +0.397%, 95% interval [+0.026%, +0.623%].
  Runtime committed memory is 167,133,855 to 37,392,547 bytes, down 77.6%.
- Qwen, six alternating pairs: median +0.418%, 95% interval
  [-0.890%, +0.967%]. This is L2 `Worker.run` latency, not serving TPOT.

Each confidence interval uses 20,000 paired-median bootstrap resamples with
seed 2883. The upper bounds meet the agreed +1% regression limit. This is a
memory-saving result with bounded median latency regression, not a speedup
claim or a per-invocation tail-latency guarantee. Vector runs use 64 warmup and
512 measured batches, numeric threads one and CPU affinity; Qwen uses 64 calls
with the first 16 excluded. All 144 vector confirmation processes and all 12
Qwen processes passed output validation, including Qwen's KV caches.

One earlier depth-four attempt failed in the first unmodified-main process on
device 6, before any candidate process: a VEC UB out-of-bounds error preceded
scheduler error 100, AICPU exception, finalize 507018 and a cleanup segfault.
The complete 12-pair rerun and 36-pair confirmation on device 0 passed; this
does not explain or fix the original failure. Its host and device logs remain
preserved separately. No A5 hardware result is claimed; local A5 validation
covers simulation and CPU tests.

Final raw data and analyses are under
`reports/issue-2883/acceptance-20261113/`, in `integrated`,
`integrated-depth4-retry`, `integrated-depth1-confirmatory`,
`integrated-depth4-confirmatory` and `integrated-qwen`. Local delivery edits
and all four PR drafts remain unpublished.
