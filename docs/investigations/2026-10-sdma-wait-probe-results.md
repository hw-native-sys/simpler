# SDMA Wait polling under full AIV fan-out

**Date**: 2026-10-20
**Verdict**: pace PyPTO prefetch waits through the existing public
`AsyncEvent::Test` API. PTO-ISA changes are not required for this integration.

## Scope and environment

Simpler #2583 fixes resident AIV channel ownership. The companion patch
on PyPTO #3001 gives each prefetch context compiler-managed UB scratch.
Those changes address two independently reproduced defects.

The remaining failure was an intermittent false return from SDMA Wait.
Channel ownership and scratch allocation alone did not resolve it.

The local environment uses A3 hardware, CANN 9.0.0, torch 2.6.0, and
torch_npu 2.6.0.post2. Searches of the system toolkit, shared CANN
packages, shared Conda environments, and the task's workspace found no
CANN 9.2 / torch_npu 2.10 installation. Additional versions found were
CANN 9.0.0-beta.2 and torch_npu 2.8.x.

Tested revisions:

- Simpler: `1d9d16e50711baa756e22274a2efc56399a93d82`.
- PyPTO local patch: `1211c510cccf9549c2ef8d798af9194f5b25152f`.
- PTO-ISA: `327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d`.

## Unmodified regression result

The test uses a full 24-AIC/48-AIV MIX. Every AIV issues prefetch while
keeping a business tile live across submit/wait. Each iteration changes
the input and checks Cube output, Vector output, completion flags, and
guard elements.

The latest device 10 run used an exclusive `task-submit` lock:

| Runtime | Execution | Result |
| ------- | --------- | ------ |
| TMR | Eager | 100 rounds passed |
| HBG | Eager | 100 rounds passed |
| TMR | ACLGraph replay | Failed at iteration 42, counted from zero |
| HBG | ACLGraph replay | 100 rounds passed |

In the failure, worker 47 recorded completion value 1 (Wait returned
false), and its entire 1024-element output stayed at the -999 sentinel.
The same code had previously failed on device 10 at iteration 33 on
worker 31. Thus one successful run cannot establish stability.

Earlier failure-only diagnostics found valid events and sessions,
distinct channel IDs, and an unread completion flag. The first Wait
took approximately 17 ms. A second Wait sometimes succeeded and
sometimes failed. These observations do not establish whether the
cause is publication, completion visibility, or a lower-level runtime
problem.

## Controlled pacing result

A diagnostic selected the polling interval through input data. All six
runs used byte-identical AIV binaries (SHA-256
`723321652df6ac09e4fc6ea400266f979837ca07c66077f1984a75f5d50d3964`)
and the same 17 ms deadline:

| Polling interval | Three fresh-process runs, 300 rounds each |
| ---------------- | ----------------------------------------- |
| No delay | Failed at iterations 204, 2, and 69 |
| Approximately 1 microsecond | All passed |

A separate paced probe with only 1000 attempts also passed three
300-round runs. Thus the signal does not depend on increasing the
original 100000-attempt allowance. The result identifies polling rate
as a causal factor, but does not establish the hardware arbitration
mechanism beneath it.

The initial PTO-ISA proposal
`ab4604f36c1a39d335080bc9e82c9629033a9f90` spaces
unsuccessful blocking reads by approximately 1 microsecond, with a
20 ms deadline per queue. `Test` remains nonblocking; a timeout still
returns false and does not cancel the outstanding transfer. The
final implementation passed four fresh-process TMR replay runs of
300 rounds each. A native 48-worker test passed ready-record,
nonblocking-incomplete, and bounded-timeout checks.

The integration instead lowers `pl.prefetch.wait` to a PyPTO adapter
using public `AsyncEvent::Test` with the same interval and a 20 ms event
deadline. Prefetch contexts provision one queue. Test already reads the
completion record and updates completion state; the adapter does not copy
PTO-ISA queue or record internals. The original PTO-ISA pin
`327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d` is retained. Simpler exposes its
architecture platform include directory so the adapter uses the existing
system-counter frequency rather than duplicating architecture constants.

This scopes the policy to PyPTO prefetch instead of changing all native
SDMA Wait callers, including communication operations. It does not change
PTO-ISA's internal submission waits or claim a generic library fix.
The original-library prototype passed 300 rounds of full-AIV TMR replay
and 100 replay rounds on each runtime with 65 consecutive posts before
waiting for the last event.
The expanded regression also covers that flag-slot reuse boundary for
both runtimes and execution modes. Final integration results are recorded
in the companion PRs.

A separate device 10 framework-only eager/replay control passed. A
300-round mapping probe observed exactly channels 0 through 47 with
no aliases. A5 hardware and CANN 9.2 were unavailable; they are not
claimed as validated.

## Rejected probes

Each header experiment used a private copy; production headers and
the regression's expected results were unchanged.

| Candidate change | Observed result on device 10 |
| ---------------- | ---------------------------- |
| Volatile UB scalar load in `GetValue` | Warmup failed |
| Explicit MTE2-to-scalar synchronization in `GetValue` | Iteration 29 failed |
| Scalar-to-MTE3 sync in `SetValue` | Later process: iteration 49 failed |
| Compiler automatic scalar DCCI | Later process: warmup failed |
| Upstream persistent post-ID lifecycle fix | Warmup failed on worker 42 |
| Single publication of data and completion SQEs | Later process: warmup failed |
| Scalar completion polling with cache invalidation | Iteration 1 failed |

The upstream lifecycle probe applied only the SDMA lifecycle hunks from PTO-ISA
commit `e6334fbd6c5d6bfdf550201bbdb81a71831a99af`: retain completion
records and resume/persist post IDs across session reconstruction.
That is a separate upstream correctness change, but it did not remove
this failure and is not proposed as its fix.

Optimized LLVM IR also keeps the completion load inside the polling
loop. The available evidence does not support a hoisted-load fix.
None of the rejected probes was incorporated into the production patch.

## Evidence and limits

The local PyPTO checkout keeps reproduction artifacts under
`build/sdma-mix-review/`:

- `environment-search.json`: versions, revisions, and search scope.
- `zhangtao-confirm-device10.log` and `.xml`: four-case regression.
- `zhangtao-confirm-device10/test_kernel_mix_prefetch_prese2/`:
  failing input iteration, output differences, completion flags, and log.
- `lifecycle-device10.log`: isolated upstream lifecycle probe.
- `pacing-control-series.log`: same-binary, same-deadline A/B comparison.
- `formal-paced-series.log`: four final-implementation stress runs.
- `timeout-device10-timed-0/native-failure.json`: failure-only timing
  and session diagnostics.

Reproduction entry point:
`tests/st/runtime/kernel/test_mix_prefetch.py`, select `-k all`,
`--platform a2a3`, and the queue-assigned device, with `-n0`.

Device 0 is not a clean control: an earlier invalid diagnostic MMIO
read triggered a BIU error, and that shared device was not reset.
Its later pure-framework data errors cannot explain the independent
device 10 Wait failures.

Reconsider a candidate fix only with a distinguishing causal test
and repeated checks of the unmodified regression. Do not suppress
Wait failures, change expected values, or increase the polling limit
and call that a root-cause fix.

## References

- [Simpler #2583](https://github.com/hw-native-sys/simpler/pull/2583)
- [PyPTO #3001](https://github.com/hw-native-sys/pypto/pull/3001)
- [PTO-ISA lifecycle change](https://github.com/hw-native-sys/pto-isa/commit/e6334fbd6c5d6bfdf550201bbdb81a71831a99af)
- [PTO-ISA polling fix #363](https://github.com/hw-native-sys/pto-isa/pull/363)
