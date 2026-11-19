# A5 HBG completion reuse of the executor snapshot

Date: 2026-11-12. Decision: reverted; no performance benefit established.

## Experiment

The reference was main `46230662e` with Ready publication outlined separately.
The executor saved a `SchedulerLocalSlotState` before calling the kernel and
checked execution identity on return. The candidate passed that snapshot to
completion to avoid copying the live slot again. Callers without the snapshot
retained their binding checks and local copy.

Five related C++ targets passed. The SP audit found no stack realignment in
14 AIC/AIV objects, and policy-dependent functions remained LOCAL. The complete
AICore cold build and link took 141.980 seconds. The ordinary AIC main function
shrank by 624 bytes; the AIV function grew by 84 bytes. These static sizes do not
establish a reduction in executed instructions.

## Hardware result

Both variants used device 3 with the same SWA input, runner and compiler settings.
L3 had one warmup and three complete captures; L0 had five warmups and 100
measurements. Output validation passed. The measurements were sequential, not
interleaved, so shared-host and time-dependent variation remain limitations.

| Metric, microseconds | Outlined Ready only | Snapshot reuse |
| -------------------- | ------------------: | -------------: |
| Last producer kernel to completion return, three runs | 7.630 / 6.542 / 6.086 | 7.397 / 8.173 / 7.676 |
| RoPE to attention, three runs | 104.641 / 87.242 / 84.954 | 94.340 / 96.256 / 95.726 |
| L0 Device median | 928.160 | 928.244 |
| L0 Device p95 | 940.375 | 944.015 |

L0 remained effectively flat and completion did not become shorter. One fewer
source-level copy was not sufficient evidence to retain the change. The candidate
was reverted. Required-pointer invariants and redundant-check removal are separate
work. Revisit only with generated-code evidence of fewer hot-path copies or spills.

## Evidence

Local evidence under `outputs/swa-next-20261112/` is not distributed with the repo:
`snapshot-candidate.patch`, `snapshot-provenance.json`, `snapshot-comparison.json`,
`comparison.json`, `snapshot-plain-function-sizes.json`, `snapshot-stack-audit.json`
and both variants' raw L3 captures. This result does not establish that all snapshot
reuse strategies regress performance.
