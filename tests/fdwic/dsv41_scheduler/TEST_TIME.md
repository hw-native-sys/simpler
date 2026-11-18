# DSV4.1 scheduler test execution time

## Original 12 cases, HBG and percounter (2026-11-18)

Each value below sums the time spent executing HBG and percounter, then averages
two complete matrix runs per runner. The four dual-device rows use one automatic
device pair per case for both runtimes. All sweeps used the same prepared inputs,
the same runtime binaries (`5ccae103b`), and at most eight devices. The test
runner was built from `fa51bfdcb` with Release optimization in both arms.
The matrix includes the latest PyPTO-lib SWA. TMR retains its one-time baseline.

| Scope | Previous runner (s) | Optimized runner (s) |
| ----- | ------------------: | -------------------: |
| DSL MoE | 21.96 | 20.50 |
| Latest native MoE | 19.98 | 20.70 |
| DSL layers 19–21 | 32.09 | 27.88 |
| Latest layers 19–21 | 30.84 | 23.00 |
| Sum of four two-device cases | 104.88 | 92.07 |
| Sum of all 12 cases | 300.21 | 285.58 |

End-to-end matrix wall times including the shared queue were 64.09 and 124.94 s
before, and 70.73 and 78.95 s after. The longest submission-to-job-start delay
in each sweep was 1.93, 80.13, 18.20 and 29.35 s, respectively. These queue-inclusive
times vary independently of the summed case execution times.

Each layers 19–21 rank initializes about 30.7 GB of repeated tensor data. The
runner fills uniform bytes on the device and uploads raw inputs directly from
their source buffer. Prefix generation, the Device start rendezvous, one warmup,
ten timed rounds and all replay output comparisons run at the same boundaries.
The Release runner target enables compiler optimization even when the build cache
has empty Release flags.

## Standalone native runner (2026-11-16)

Measured on A5 on 2026-11-16 with the 12 selected cases, runtime source
`5abd1794980eaa493f618067b91aeff117191a63`, without PR #3108. Times below
include startup, input loading, device setup, warmup, measurement, validation
where enabled, and teardown. They are whole-test seconds, not AICore latency.

| Scope | Previous runner (s) | Standalone native runner (s) |
| ----- | ------------------: | ---------------------------: |
| Native MoE, same allocated device pair | 16.29 | 6.77 |
| Full 12-case percounter matrix, up to eight devices | 62.76 | 44.00 |

The MoE comparison ran both launchers sequentially on automatically allocated
devices 2 and 3, using the same original callable, input bytes and percounter
runtime. It is one observed pair; startup variability remains. The full matrices
used separate automatic allocations and include submission, queueing and result
collection, so their difference is not a controlled paired speedup estimate.
The standalone matrix passed all 12 cases with ten valid samples each.

## Execution and dependencies

- Prepared tests use standard Python plus the C++ runner and Simpler/CANN.
  The full matrix ran under isolated system Python (`python3 -I`), without
  loading PyPTO, PyPTO-lib, Torch or Simpler Python bindings.
- Native MoE retains the original generated C++ Host orchestration, actual
  `get_tensor_data<int32_t>(num_tokens)`, nonzero weights, forced routing and
  device communication. All 50 exported per-rank tensor inputs were checked
  byte-for-byte against the original model fixtures. Kernels were reused.
- All cases retain one warmup and ten measured rounds. Two-device runs retain
  the Device-side start rendezvous before timing. The other 11 cases retain
  numerical checks; native MoE remains performance-only for `72/72` tokens,
  explicitly recording `numerical_validation: false`.
- Initial native MoE bundle preparation requires CPU Torch and writes local
  raw inputs. Its 2.48 GB weights/cache stay in ignored build directories.
  Preparation is outside these timings; its cold duration was not remeasured.
  Prepared bundles need no model checkout or Python ML environment.

## Reproduction and evidence

Use `production/run.sh` with level `0`, warmup `1`, rounds `10` and prepared
bundles. Allocate devices using `task-submit --device auto --device-num N`
and use the returned `TASK_DEVICE`; do not pin device IDs. No more than eight
devices were occupied concurrently.

Local evidence in the standalone worktree is under `artifacts/native-moe/`:
`pair.json`, `percounter-state.json`, `summary.json`, `input-parity.json` and
per-case `wall.json`/timing logs record allocations, measured times, input
comparisons and rendezvous evidence. The previous matrix is recorded in the
base-refresh worktree's `artifacts/base-refresh/summary.json`.
