# A5 HBG legacy execution removal

The implementation removes A5 HBG's legacy execution route. Valid ordinary and
native Graph tasks execute through resident scheduling. Invalid input fails
prepare or enters failed-start retirement instead of selecting another executor.

The research and hardware comparison baseline is
`ca27b263d5b6576b9f5dbd9050fa7b7ab6c53ff0`.
The initial publication rebased onto `6c691bef8a1752aff34889b366204da36fe41701`,
including upstream's scalar-trace ownership fix; HBG/percounter production builds
and static checks passed at that base. Subsequent review updates rebase onto
upstream/main. Historical execution-removal results below precede those updates;
follow-up builds and device checks are recorded separately.
The [Chinese investigation and validation record](../zh-cn/investigations/2026-11-a5-hbg-legacy-removal.md)
contains the layout proofs, startup pseudocode, source references, individual
hardware task IDs and failure comparisons.

## Input and lifecycle contracts

- The native Definition planner requires canonical contiguous tensor/scalar
  spans covering their entire pools, with expanded arguments fitting the reserved
  execution tail. Overlap, gaps, unused tails and insufficient capacity fail prepare.
- AICores publish their physical identity and report epoch before bootstrap
  validation. Invalid modes, absent/misaligned context bases and overflowing
  context extents use EXIT/nonce/teardown retirement without accessing scheduler
  backing. READY publishes the context for accepted launches.
- AICPU initialization joins each phase owner before cleanup. The leader commits
  the final initialization outcome; partitions already released for execution
  still send EXIT and retire if a subsequent startup barrier fails.
- A register window has one retirement owner. Missing reports and missing ACKs
  do not prove retirement. Failed terminals preserve the first execution error
  and cleanup outcome. PMU settlement requires ownership established by this run.

The shared arena allocation is removed in the follow-up described below. The
kernel context cleanup below completes removal of the obsolete A5 scheduler stack. A2A3 HBG, A5 TMR and the active percounter extension remain
supported.

## Compact task-image arena follow-up

After #3064 merged as `e37461139e7e58dc57f3b8914d278494c76bddd7`, the A5 maker
stops reserving the common full arena. The task image occupies offset zero of the
pooled runtime allocation: request size, `sm_image_bytes` and publication length
are equal, and publication targets `gm_sm_ptr`. No SchedulerState, ready queue,
completion mailbox, SM wrapper or copied RuntimeContext header precedes it.
Host orchestration uses OrchContext and OrchestratorState directly; retained retries
store their callback result independently. Commit attaches the populated host mirror without
resetting its task count or execution header. Native boundary addresses, ordinary storage
addresses and protocol source mapping all use the compact image base.

The common full arena remains active for A2A3/WSE. Percounter compiles the A5 maker
and receives the same image-only allocation. No kernel context/async ABI change
or device execution/retirement change is part of this follow-up.

Regression assertions cover both A5 image-only and A2A3 full-arena contracts,
including ordinary and Graph prepares and publication. Local CPU/simulator tests
are intentionally not run. Follow-up build/device validation is recorded in the
Chinese investigation: both HBG/percounter production builds passed, A5 IR scanning
completed 16/16 TUs with zero findings, and normal scene tests passed 12 with one
simulator-only skip. Twenty distinct async/reuse nodes passed across two device
submissions: the first hit its whole-session 600-second budget after 13 collected
passes; the remaining seven passed in a separate submission, with assertions
unchanged. These checks use the `e37461139` production source base; rebasing onto
`60a2ba4cf` adds only CI/docs/test-environment changes. No benchmark or new fault
acceptance result is claimed. After fixing populated-mirror attachment, both production builds passed again,
with 12 normal tests passing (one simulator-only skip) and both cross-run epoch
reuse nodes passing in a separate submission. The admission trimming test injects
one explicit allocation refusal instead of depending on the arena's size.
Historical results below apply only to #3064.

## Obsolete scheduler and kernel context cleanup

After #3071 merged, the A5 build no longer includes the old AICPU SchedulerState,
mailbox or async wait-list implementation. The active A2A3/WSE full-arena initializer
and mailbox live under `src/common/host_build_graph/full_arena/`, selected by their
build configurations. Shared Host orchestration and queue-capacity POD types no
longer include a concrete device scheduler. A2A3/WSE makers explicitly carry the
inline-completion count into their device RuntimeContext.

A5 LocalContext contains only logical block geometry (8 bytes). DispatchPayload
has no source-payload gate or embedded legacy slab/token context. Its argument
array stays 64-byte aligned at offset 64; GlobalContext remains 48 bytes at offset
464, and the slot stride stays 512 bytes. All current producers and consumers,
including percounter, are rebuilt together. No historical-object reader is kept.
Percounter derives L1 task identity from the selected task when committing the
record, preserving global and Graph-child attribution without storing an async token.

The kernel async API uses the resident group exclusively. Its existing completion
aliases address real group storage. An absent group rejects registration without
touching GM; invalid epochs/layouts and unsafe output capture retain fail-closed
and first-error behavior. SDMA reservations and synchronous TPUT completion use
the same resident writer. Address-based kernels retain generic range publication;
context publication writes back the current resident group without sealing it. The 32-byte completion-entry format, 320-byte common
task storage, resident ready/control queues and retirement protocol remain active.
Shared Host tests retain both architectures; AICPU queue/wait-list tests target
A2A3 and the independent A5 TMR mailbox test targets TMR.

Validation uses the #3071 merge base `3bcaf8d4f` plus this cleanup, with
Ascend950DT_9581, CANN 9.3.0 and the repository-pinned PTO-ISA. HBG/percounter
and the independent A5 TMR production builds passed; A2A3 HBG and WSE production
compilation also passed. Static hooks and strict MkDocs passed. A5 IR scanning
completed 16/16 HBG/percounter device TUs with zero findings. Symbol inspection
found no old SchedulerState/full-arena/mailbox implementation in the built A5
HBG/percounter Host or AICPU libraries.

Five task-submit jobs passed 43 cases with one existing simulator-only skip:
normal/Graph/shape/trace 31, async capacity/last-condition/epoch reuse 7, peer SDMA
3, percounter L1 rotation/repeated-run 1 and deferred-wait data visibility 1.
Six DSV4.1 wait adapters also passed A5 kernel compilation. The Chinese record contains task IDs.
This is directed validation, not full hardware CI or a new fault-acceptance result.
No local CPU/simulator functional tests were run. Workload compilation was warmed
outside allocations; plain probes also invoke the compiler cache from their test
entry points. No size or latency gain is inferred from source deletion.

## AICPU dispatch scaffolding and build dependencies

After #3076, A5's scheduler types contain only the resident wire contracts and
shared watchdog budget. The unused AICPU CoreTracker/CoreExecState, slot events,
sync-start drain state, dispatch counters and profiling macros are removed.
The CoreTracker regression belongs to A2A3's active AICPU dispatcher; current A5
MIX/admission tests continue to cover resident placement.

The default A5 AICPU build excludes the common GraphExecution materializer:
Host builds the native Graph plan, and AICore workers bind its instance arguments.
Percounter still compiles that materializer for its AICPU Graph preparation;
A2A3/WSE retain their existing selection. Argument dumping still reads the common
task image. This cleanup does not change the 320-byte task storage, resident
ready/control queues, worker slots, async completion or shutdown contracts.

## Execution-removal validation on 2026-11-13

Compilation and workload preparation took place before device allocation. All
hardware tests ran through `task-submit`, with an in-job architecture precheck:
Ascend950DT_9581, CANN 9.3.0. CPU and simulator validation were omitted at the
user's request. No performance result is claimed.

- HBG and percounter production builds passed for Host, AICPU and AIC/AIV.
- A5 IR scanning completed for 12/12 device translation units with zero findings.
  The CANN `llvm-link -S` reader preserved the original target and data layout;
  the system LLVM reader could not read CANN's `block_local` extension.
- 55 distinct normal hardware cases passed at least once, covering ordinary and
  native Graph execution, AIC/AIV/MIX, predicates, sync-start normalization and
  capacity rejection, argument dumps, async completion and run reuse. One
  simulator-only case was skipped by its existing condition.
- Five bounded/native failure cases still failed: they returned the expected
  `507018` after about 30–40 seconds, exceeding the test's 20-second bound.
  All five failures reproduced on the unchanged baseline on the same device.
  Later containment/recovery assertions were not reached.
- The pure-AIC single-block sync-start case failed twice on device 2 with an
  AICore execution exception followed by watchdog/retirement errors. Both
  baseline and candidate passed that exact case on device 0. The original
  device was occupied before baseline comparison, so regression attribution is
  unresolved. An initial pure-AIC failure-test healthy pre-run also failed,
  then passed when repeated.

These are targeted device results, not a complete CI or failed-start fault
injection verdict. #3064 was subsequently merged by the maintainer. The unexplained healthy-run
exceptions and full failure acceptance remain unresolved; these historical results
do not validate the arena follow-up. Do not relax test assertions to turn these
runs green.
