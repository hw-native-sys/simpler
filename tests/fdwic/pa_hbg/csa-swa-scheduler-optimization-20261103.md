# A5 HBG CSA/SWA scheduling optimization

## Scope and retained behavior

This change extracts ordinary resident-scheduler optimizations from
`nalinaly/simpler:feat/fully_distributed_within_core` at
`9f5c0d08b7e257b7e45a5b821c7b3a7c4ca8c73d` onto official main
`142ee15a7fa46925b9802984fa1ac75676e87d43`.
Composite Graph definition, child materialization, ENTER/EXIT nodes, Graph wire
extensions and Graph-specific tests are excluded. Graph calls retain main's
AICPU fallback. Business kernels, fixtures and golden tolerances are unchanged.
The existing `scheduler_graph.h` changes only ordinary MIX resource-demand
counting; its graph-view layout and Graph execution selection remain unchanged.

| Optimization | Preserved constraint |
| ------------ | -------------------- |
| MIX members follow `active_mask \| AIC`; assignments retain each lane's predecessor | Every required arrival/release and descriptor generation is checked; all workers join startup/shutdown |
| Final valid claimant publishes an exhaustion certificate | Invalid reservation tickets cannot seal; the certificate is atomic-published in an isolated region, and positive hints are atomically confirmed |
| Close seals occupy independent atomic regions | A slower member retains its task/cluster closure evidence across mailbox reuse |
| Preparation/commit, ACK prefilters and fixed-context checks share same-call validation | Retries discard the proof; error, epoch, phase and execution identity are checked live |
| Profiling specializes the resident loop and protocol helpers | L0 removes recording work; L1-L4, sampled timing and PMU retain their gates |
| Local NOP spin and immutable configuration/topology caching | Progress resets spin to 8; idle caps at 1024, startup/exit at 256; READY publishes the validated topology |

The extracted certificate uses the existing atomic-exchange primitive rather
than introducing another bypass-store publication. It has its own 128-byte
region, separate from main's existing admission signal. TaskControl grows from
512 to 640 bytes. Close seals grow from 8 to 128 bytes per MIX task per candidate
cluster. The additional GM requirement is `128 * task_count + 120 * mix_tasks *
cluster_count` bytes, before alignment. Per-worker admission state grows by
32 bytes. Public task, Tensor and Graph APIs do not change.

## Controlled comparison

- Date: 2026-11-03. One `task-submit` allocation on A5 device 7, 32 AIC / 64 AIV.
- CANN 9.3.0; Python 3.12.15; Torch 2.12.0+cpu; CPU affinity 0,1;
  at most two compiler workers and Torch intra-op threads.
- Both runtimes use main's PTO-ISA pin
  `f83d4b7e4be0a4ca9805944ae2d9ce8c84e581f7`.
- Same frozen compute binaries in all groups: 54 CSA and 28 SWA kernels.
  They originate from the previously validated `217bb989` PTO-ISA build;
  only Host orchestration is rebuilt against each runtime's headers.
- CSA: V4-Pro B4S2 seed 0, all seven original golden outputs.
- SWA generation used PyPTO-lib `26ef3e865733747a89c3356ec145e567851ad985`,
  PyPTO `be23606f791fa2251d7fccabdb91c75582268fad` and PTOAS v0.70.
- SWA: DS4.1, 72 tokens / 12 requests, six queries per request, start position
  8192, frozen seed-17 inputs, all five original output comparators.
- Each group has exactly ten calls and no extra workload warm-up. INOUT tensors
  are restored before every call; golden comparison runs after every call.
  The first sample is retained. Formal timings use swimlane level 0.
- The AICPU control is a separately frozen build with only the Host scheduler
  selection forced to legacy mode 4. Within each source version, only
  `libhost_runtime.so` differs between the two arms; the AICore/AICPU executors
  and dispatcher are byte-identical.
  The normal AICore arm publishes mode 1 on every call.
- Device time is `chip.run.runner_run.device_wall`; Host time is `chip.run`.
  These are separate metrics. All ten spans are present, with no dropped records.

The tested candidate runtime source patch has SHA256
`72566d6d4317127593476028234e2f00a4e3264c3f8bf9a2092744a6d2568cc2`
(relative to the main commit above, excluding documentation). No Graph runtime
implementation is required by this patch.

## Results

Medians in milliseconds. Device change is `(candidate / main - 1) * 100`.

| Case | Scheduler | Main Device | Candidate Device | Device change | Main Host | Candidate Host |
| ---- | --------- | ----------: | ---------------: | ------------: | --------: | -------------: |
| V4-Pro CSA | AICore | 3.132435 | 2.531305 | -19.19% | 27.998880 | 19.926590 |
| V4-Pro CSA | AICPU control | 2.039332 | 2.060380 | +1.03% | 27.475670 | 27.644360 |
| DS4.1 SWA | AICore | 1.076120 | 0.898628 | -16.49% | 13.529415 | 12.904500 |
| DS4.1 SWA | AICPU control | 0.732697 | 0.729719 | -0.41% | 14.528170 | 10.832385 |

AICore Device latency decreases by 19.19% for CSA and 16.49% for SWA.
AICPU Device differences are within 1.1%. Host/bind times fluctuate separately;
these observations do not establish a Host speedup or a per-optimization causal
breakdown. The two AICore paths remain slower than their AICPU controls.

All 80 timed calls and four separate L4 diagnostic calls pass their golden
checks. L4 timings are excluded from the performance table. Both captures pass
strict sequence/causality audits and have exactly the same task/block/subtask
execution identities: 2,308 for CSA and 549 for SWA. Loaded compute fingerprints
also match across all four arms for each case.

| L4 evidence | CSA main | CSA candidate | SWA main | SWA candidate |
| ----------- | -------: | ------------: | -------: | ------------: |
| Resident workers | 96 | 96 | 96 | 96 |
| Two-lane MIX tasks | 26 | 26 | 1 | 1 |
| Members per MIX task | 96 | 64 | 96 | 64 |
| Total member obligations | 4640 | 3808 | 1376 | 1344 |
| Compute executions | 2308 | 2308 | 549 | 549 |

The member count is task-level membership, not a chip-wide barrier for every
block. Each MIX block still coordinates only its local cluster. AICore and the
active AIV execute the same kernels; the unused AIV has no MIX obligation.

## Individual Device samples

Each row is one call. All values are milliseconds; no sample is discarded.

| Call | CSA main Core | CSA candidate Core | CSA main CPU | CSA candidate CPU | SWA main Core | SWA candidate Core | SWA main CPU | SWA candidate CPU |
| ---: | ------------: | -----------------: | -----------: | ----------------: | ------------: | -----------------: | -----------: | ----------------: |
| 1 | 3.174081 | 2.526631 | 2.289162 | 2.393048 | 1.091168 | 0.901398 | 1.056712 | 1.005194 |
| 2 | 3.119694 | 2.499422 | 2.108618 | 2.055962 | 1.097485 | 0.878105 | 0.731600 | 0.729733 |
| 3 | 3.140658 | 2.543238 | 2.053118 | 2.079882 | 1.048284 | 0.900963 | 0.727432 | 0.723351 |
| 4 | 3.074797 | 2.528306 | 2.042111 | 2.106580 | 1.083579 | 0.890592 | 0.790157 | 0.733766 |
| 5 | 3.134736 | 2.536668 | 2.036553 | 2.064798 | 1.069885 | 0.923534 | 0.753213 | 0.707712 |
| 6 | 3.056498 | 2.544352 | 2.029776 | 1.998048 | 1.089068 | 0.896293 | 0.739906 | 0.715810 |
| 7 | 3.130133 | 2.531359 | 2.008511 | 2.009325 | 1.037146 | 0.918707 | 0.719314 | 0.729706 |
| 8 | 3.106429 | 2.526419 | 2.019529 | 2.047833 | 1.082355 | 0.884378 | 0.732091 | 0.728194 |
| 9 | 3.152364 | 2.531251 | 2.099593 | 2.093325 | 1.048518 | 0.876138 | 0.733303 | 0.769764 |
| 10 | 3.167665 | 2.547596 | 2.011282 | 2.040541 | 1.020685 | 0.909383 | 0.730568 | 0.733207 |

## Validation

- A5 HBG C++ checks: 93 executables have passing results. Three test fixtures
  required layout/resource-scope/compact-identity adaptation and passed isolated
  reruns; the initial failing output is retained in the local run artifacts.
- Final Python HBG regression: 3,974 tests passed, including 651 failure-fixture
  cases covering the versioned certificate/stride/lock requirements.
- A5 HBG simulation: 36 L2 entries and 49 resource tests passed; two resource
  tests skipped. Results span the initial sweep and eight isolated fault-case
  reruns after adapting their v5-only expectations to strict v8 evidence.
  The parent of the initial sweep hit its 1,200-second session limit; its L2
  child completed all 36 entries. This is not one uninterrupted passing sweep.
- A5 single-chip onboard regression: 48 L2 entries and 57 resource tests
  passed; one L2 test skipped. This combines the 103-entry sweep, two isolated
  reruns adapting fixed three-lane expectations to exact resource sets, and
  three additional two-lane MIX fault/recovery cases. The main sweep used
  device 2; the targeted cases used device 1, each under `task-submit`.
- Source changes after the performance freeze are confined to tests and
  documentation. The runtime source patch matches the measured snapshot.
- The two initial broad onboard attempts stopped before executing cases because
  their selection included multi-device L3 tests; the final onboard scope is
  A5 HBG chip-level work. A queue-maintenance rejection also executed no cases.
- All applicable pre-commit hooks passed, including formatting, clang-tidy,
  lint and type checks; no hook was bypassed.

The local artifact directory is
`logs/csa-swa-scheduler-pr-20261103` under the development workspace. It retains
frozen runtime manifests, exact source patches, payload hashes, inputs' identity
records, all samples, L4 captures, strict audit results and failed-attempt logs.
Raw binaries and machine-specific paths are not part of this change.
