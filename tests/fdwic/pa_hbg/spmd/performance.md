# CSA performance and retained optimizations

## Measurement boundary

Use the frozen HBG CSA B4S2 seed-0 workload: 54 registered kernels, 52 arguments,
original block counts and all seven golden outputs. `device_us` measures the
complete `chip.run.runner_run.device_wall` window. It is neither Host end-to-end
latency, the sum of kernels, nor the first-kernel-to-last-kernel trace span.

Performance runs disable swimlane capture. Each benchmark first validates three
rounds, then runs five warmups and 100 measured rounds in a separate process.
Every round restores the original INOUT state. L1-L4 runs diagnose behavior and
carry their own instrumentation overhead; do not substitute their spans for the
uninstrumented Device metric.

## Retained result

The last accepted optimization reused an already-negative ready probe within
the same scheduler fill call. Historical A/B/A/B measurements on 2026-10-28,
on one Ascend950PR_958b (32 AIC / 64 AIV), with Torch 2.12.0+cpu:

| Batch | Version | Samples after warmup | Device median, us |
| ----- | ------- | -------------------: | ----------------: |
| A1 | Before same-turn probe reuse | 100 | 2251.3110 |
| B1 | Retained implementation | 100 | 2084.9525 |
| A2 | Before same-turn probe reuse | 100 | 2239.7165 |
| B2 | Retained implementation | 100 | 2080.9475 |
| A combined | Before | 200 | 2245.2640 |
| B combined | Retained | 200 | 2083.5975 |

The combined Device reduction was 161.6665 us (7.20%). This is approximately
2.084 ms, not a claimed 1.5 ms result. It is historical evidence for the retained
code, not a fresh benchmark of the documentation-only upstream packaging.
Host latency was noisy; no corresponding Host end-to-end speedup is claimed.
The environment lacked a device reservation tool, so these are not guaranteed
exclusive-device CI baselines or performance thresholds.

Source provenance is the final implementation from
[the original PR](https://github.com/hw-native-sys/simpler/pull/2766), pinned at
[d4a0d17477faf18333220f37fcbe30622cd16e02](https://github.com/nalinaly/simpler/commit/d4a0d17477faf18333220f37fcbe30622cd16e02).
The source snapshot retains the historical A/B summary under
`tests/fdwic/pa_hbg/spmd/perf_data/2026-10-28-idle/ready-probe-summary.json`,
including per-batch runtime hashes and fixture identity. Those raw/process files
are deliberately not copied into the upstream implementation. The fixture
manifest SHA256 was `d1511befaae48a3b692a723320a076064e2ebb1bced193175866c692f9909b45`.

## Optimizations preserved in code

| Optimization | Preserved correctness boundary |
| ------------ | ------------------------------ |
| Ticket allocation for ordinary N>1 | Irreversible unique ownership, bounded invalid tail tickets, confirmed valid-claim exhaustion |
| Isolated admission/error/log bypass signals | Whole signal domain excludes atomic updates and stale cached writeback; ownership words remain atomic |
| Bounded admission-record observation reuse | Keep complete shape/member/epoch validation; do not manufacture entry permission |
| Phase-specific GM service and same-turn retire/next-publish | Still require all three resource releases and exact group identity |
| Execute first pure C/V READY before unnecessary second-slot work | Keep final entry gate and two-candidate ownership bounds |
| Early Mix-ready prefix publication during dependency resolution | Validate each successor, never rewrite a published prefix or let a later suffix overtake it |
| Same-fill negative ready-probe reuse | No cross-turn emptiness cache, ownership shortcut or skipped obligation service |
| Task-level Ready trace publication | Avoid per-instance instrumentation inflation without reducing task workload |

These are generic scheduler changes, not CSA-name or task-ID special cases.
Individual percentage gains are not summed: measurements at different stages
are not independent. Removed experiments such as speculative ACK preflight,
early Mix retirement and close-prefilter variants are not part of the final
runtime. Their applicable safety regression tests remain useful.

## Reproduce

Follow the [CSA setup guide](../../../../examples/a5/host_build_graph/deepseek_v4_pro_csa/README.zh-CN.md),
build the current checkout, import the exact fixture and reserve the device:

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
CSA_HBG=examples/a5/host_build_graph/deepseek_v4_pro_csa
python "$CSA_HBG/run.py" l4 -d 0 --require-aicore-scheduler
python "$CSA_HBG/run.py" bench -d 0
```

The runner records checkout, fixture and runtime hashes and saves unique output
directories. For a causal comparison, alternate independently built A/B worktrees
on the same idle device, fixed toolchain, input, kernels and capture level. Do not
compare an old AICPU run or a reduced-workload trace as if it were that A/B pair.
Keep bulky traces and samples outside Git; retain reproducible commands and
compact conclusions with their measurement boundary.
