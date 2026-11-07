# Native framework submission

Status: the core design is implemented; full qualification and independent implementation review remain pending. This document covers direct L2 eager delivery for issue #2774 and PR #2776. The test matrix below specifies acceptance targets, not a claim that every case has been implemented or passed. Validation must identify the tested source and native build; results from an earlier revision do not automatically qualify a later one.

## Background and goals

With the Torch NPU host task queue enabled, the caller enqueues work and a queue thread issues native device operations. A bridge that calls Python `Worker.submit()` from that thread needs the GIL:

```text
Python caller -> C++ enqueue -> queue thread acquires GIL
                            -> Python Worker.submit() -> native execution
```

If the caller holds the GIL while waiting for the queue to drain, the queue callback cannot proceed. Wrapping a Python closure in C++ does not remove this dependency. Draining before every call defeats asynchronous enqueueing.

Design inputs:

- [Issue #2774](https://github.com/hw-native-sys/simpler/issues/2774): problem and delivery scope.
- [Consumer requirements](https://github.com/hw-native-sys/simpler/issues/2774#issuecomment-6653614631): ticket management and safe retirement after failure.
- [Architecture requirements](https://github.com/hw-native-sys/simpler/pull/2776#issuecomment-6654551103) and [clarification](https://github.com/hw-native-sys/simpler/pull/2776#issuecomment-6655744756): one native execution path, a same-build interface and a real framework consumer.
- [Run completion contract](run-completion-fence.md): execution, result observation, lifetime and cleanup responsibilities.

The earlier consumer discussion accepted a versioned-handle proposal. Architecture review subsequently required a same-build interface without an independent version protocol. That change remains part of the consumer handoff; agreement on the overall direction does not approve every implementation detail.

| Key | Goal | Observable acceptance |
| --- | ---- | --------------------- |
| R1 | Queue execution does not depend on the GIL | Submission, error recording and native reference destruction progress while the caller holds the GIL |
| R2 | Preserve asynchronous enqueueing and ordering | Numerical Torch predecessor/Simpler/Torch successor correctness without per-call drain or synchronization |
| R3 | One execution authority | Reuse existing admission, preparation, launch, completion and retirement |
| R4 | Values and resources survive queue delay | Stable argument/config snapshots; callable, storage and stream owners survive their last use |
| R5 | One-shot and explicit failure semantics | Submit/cancel have one winner; errors remain observable; release eligibility is programmatically available |
| R6 | Preserve existing contracts | Ordinary L2 submit, unregister, close, context and process validation have documented regression coverage |
| R7 | Qualify a real consumer | Execute HBG/TMR with the host queue both enabled and disabled |

## Scope and responsibility

Delivery includes the native entry, Python handoff helpers, a runnable framework reference adapter, lifetime/error regressions, documentation and current-revision CI. Native execution remains independent of Python and Torch; torch_npu queue integration belongs in the adapter.

The scope is direct L2 eager. Program and Kernel use the common execution API; differences in environment and IO management do not require separate submit families. Capture/replay, model/graph ownership, L3 native submission, reusable prepared calls, automatic preparation caches and full-model/vLLM qualification are excluded. Existing capacity backpressure remains valid.

[PyPTO #3209](https://github.com/hw-native-sys/pypto/issues/3209) owns downstream adapter migration and qualification. That separate migration does not excuse an unused interface here, and a passing reference consumer does not prove that PyPTO has migrated.

```text
Python caller, holding GIL
  validate/snapshot values, retain owners, register pending ticket
                  |
Framework C++ adapter
  enqueue native action, or execute the same action if queue is disabled
                  |
Existing Simpler native execution, without Python/GIL
  ChipRunLane admission -> common prepare -> stream launch
                  |
Existing result observation and cleanup
  establish retirement proof
                  |
Python-safe boundary
  release Python owners and remove retired tickets
```

| Layer | Responsibility | Boundary |
| ----- | -------------- | -------- |
| Python Worker | Public entry, original identity checks, owner retention, wait/close integration | No independent device-completion decision |
| Python bindings | Convert and hand off to the common native entry | No binding-only execution engine |
| Submission envelope | One-shot consumption, cancellation, native references and association with the real run | No device queue, executor or reusable preparation cache |
| Existing run/lane | Admission, preparation, launch, completion, error observation and cleanup | No Python owner destruction on the queue thread |
| Framework adapter | Host emission ordering, native references and callback error channel | No private RTS bypass |
| PyPTO adapter | Pending-ticket tracking, periodic retirement and explicit drain/close | No per-operator wait/cancel burden for end users |

## Design contracts

### Preparation and native handoff

`Worker.prepare_submission(callable, args, config, caller_stream=..., keepalive=...)` prepares a queue submission, not necessarily device preparation. The queued native action may perform existing device preparation.

Preparation validates Worker state, process/context, original callable identity and call values. Value snapshots and source-use reservations share existing synchronization boundaries. Original-handle validation prevents unregister/slot reuse from redirecting a prepared invocation to a replacement callable.

Ordinary submission and native handoff use distinct entry paths over shared argument snapshotting, source-use reservation and materialization. Both register accepted runs through the same bookkeeping helper. Preparation failure releases reservations before any native execution; an ordinary native execution failure still requires an explicit safe-release verdict. Native retirement removes Python roots only after native retirement succeeds.

Snapshots copy descriptors, scalars, configuration and readonly HOST/NONE control contents before handoff returns. The shared `_prepare_l2_args` boundary invokes native `CallArgumentSnapshot` before either ordinary submission or delayed handoff. Shared argument preparation uses the callable signature to select these inputs, preserves view geometry and retains native snapshot storage through safe retirement. Unknown native entry failures retain snapshot storage through the existing safe-use retirement/owner-close boundary, even if no run handle is returned. HOST OUT/INOUT retains its existing external writeback destination and access validation; it is not redirected into an input snapshot. DEVICE contents remain borrowed and governed by queue/stream ordering. Borrowed storage must remain valid and must not be mutated before its ordered use ends.

The native interface contains context, retain, release and submit. It retains capsule type identity, non-null checks and lifetime validation, but has no V1 type, version field, version negotiation or capability-query replacement. Adapter and extension rebuild together; no independently versioned binary compatibility is promised.

Resolve the adapter include directory using `simpler_setup.environment.PROJECT_ROOT / "src/common/worker"`. This selects the source checkout for editable installs and the packaged `_assets/src/common/worker` directory for wheels.

The Python caller extracts the capsule and obtains a native reference before handing it to the queue. Final native reference destruction must not destroy Python objects or invoke a deleter that acquires the GIL. Python capsule destruction and native queue-reference destruction are separate lifetime boundaries.

### Execution and ordering

The native action uses existing admission, common preparation and stream launch. The envelope may wait on or forward the real run result; its consumed state is not evidence of device completion.

Both framework host emission order and device stream order matter. Sharing a device stream cannot order Simpler behind a Torch operation that has not yet left the host queue.

Preparation success, enqueue return, native callback success, device completion and safe retirement are distinct events. Normal submission adds no per-call queue drain or device synchronization. Existing capacity backpressure remains in force.

### Ownership and cancellation

| Resource | Lifetime responsibility | Release condition |
| -------- | ----------------------- | ----------------- |
| Native envelope references | Python ticket and adapter native references | Each reference exits; native destruction never accesses Python |
| Callable/source-buffer uses | Existing Worker registrations and use reservations | Safe pre-consumption cancellation or existing execution retirement proof |
| Tensor/stream Python owners | Worker retention and adapter pending collection | Safe retirement, released with the GIL |
| Device execution/completion resources | Existing run/lane | Existing completion, observation and cleanup contract |

Consumption and cancellation compete atomically. Once consumption starts, cancellation returns false; duplicate submission is rejected without changing the original invocation. Device state remains authoritative in the real run, not another envelope state machine.

Register a pending ticket before any step that could lose it. An enqueue exception does not prove that the callback was never queued: attempt cancellation, and keep tracking the ticket if it was consumed or cancellation is uncertain.

A true cancellation result proves cancellation, not successful cleanup. Check `safe_to_release` before dropping owners. Cleanup failure can leave a cancelled ticket retained until successful owner close. An exception does not prove that cancellation failed to take effect.

Dropping a Python ticket does not cancel it. There is no public lookup API to recover a lost ticket. An unconsumed, uncancelled ticket can prevent close from draining; close does not implicitly cancel it. Keep tickets accessible until retirement. A wait without a timeout can remain blocked until another thread submits or cancels the ticket.

Global roots, when used, retain Python owners only. They must have verified removal paths and must not become another device-resource authority.

### Error observation and safe retirement

The native callback reports success/failure and a message through the adapter. Python retains the original error and exposes a programmatic retirement predicate; error-string parsing is not a release protocol.

`safe_to_release` projects existing cleanup results. It starts false and becomes true only when this invocation's external-owner uses are discharged. It is not a new completion event, does not mean success, and does not release another invocation's uses of the same object.

| Outcome | Observable result | Retirement |
| ------- | ----------------- | ---------- |
| Safe cancellation or proven pre-launch refusal | Preserve cancellation/refusal | Allowed once cleanup is safe |
| Successful execution and observation | Success | Allowed after cleanup |
| Execution failure with proven cleanup | Preserve original execution error | Allowed; never convert failure to success |
| Timeout, unknown completion or unfinished cleanup | Preserve error/timeout | Retain owners and recovery obligations |
| Duplicate submission | Refuse duplicate | Leave original invocation unchanged |
| Retryable observation-context refusal | Preserve context refusal | Keep result unconsumed for retry in the correct context |

A native submit error without a returned run supplies no run cleanup proof and can require successful Worker close. Proven pre-launch refusal and safely cleaned execution failure must remain distinguishable from unknown completion. Neither releasing on every error nor retaining forever on every error is correct.

The adapter periodically considers completed tickets for bounded observation and retirement. `done` is only a candidate filter; observation can still block. An observation timeout leaves the ticket retained. Explicit drain/close handles remaining obligations, and errors remain observable after retirement.

### Unregister and integration

Both prepared and ordinary L2 invocations protect the original callable while it is in use. Unregister refuses the last registration while an unretired use remains; refusal leaves the registration valid for retry. Other references follow existing registration counts. Physical completion alone does not retire the Python use.

Preserve main's context checks, completion/observation separation (#2829), finalize boundaries (#2844), framework setup and CI scheduling when integrating. Shared-file conflicts require semantic reconciliation. Public CI timeout changes are outside this feature. A5 follows the current `/run-a5` policy.

## Test design and delivery criteria

CPU tests exercise states, references and error contracts. Real NPU tests qualify framework queue/stream behavior. Neither substitutes for the other. Hardware qualification targets A3/A5 x HBG/TMR x `TASK_QUEUE_ENABLE=0/1`; pending-callback GIL drains apply to the enabled queue.

Record head, base, CI merge/source, native build identity, Python/Torch/torch_npu/CANN versions, platform, queue mode and device-task outcome. Verify test plan and JUnit agree. Skipped/xfail required cases do not count as passes; unexecuted combinations remain explicit gaps.

| Key | Layer | Scenario | Required observation |
| --- | ----- | -------- | -------------------- |
| T01 | CPU/NPU | Prepare HOST input at 7, mutate backing to 19, delete wrappers and change scalars/config before consumption | Saved value is 7; offset/stride geometry and lifetime survive; writable HOST destinations remain external; no launch or lane-capacity wait during prepare |
| T02 | CPU | Concurrent submit/cancel | At most one execution; successful cancellation prevents execution |
| T03 | CPU | Copy/move/cross-thread release; invalid handles | Balanced references, no Python destruction; reject invalid type/null interface |
| T04 | CPU | Submission failure with multiple waiters | Original error remains observable; no false success |
| T05 | CPU | Exception after prepare but before enqueue | Safe cancellation returns uses and removes roots |
| T06 | CPU/NPU | Enqueue raises after callback may start | Failed cancellation preserves tracking and owners |
| T07 | CPU | Deterministic unregister/slot-reuse interleaving | Original identity cannot be redirected; no source-use leak |
| T08 | CPU | Last-registration removal for ordinary/prepared calls | Refuse while unretired, preserve handle, allow safe retry; preserve duplicate counts |
| T09 | CPU | Delete accessible ticket/tensor/stream wrappers | Owners survive until retirement; weakrefs demonstrate eventual release |
| T10 | CPU/NPU | Success, pre-launch refusal, safely cleaned failure | Correct retirement predicate, original errors, release once |
| T11 | CPU/NPU | Timeout, unknown completion, cleanup failure | No early release; later observation/close follows existing recovery |
| T12 | CPU | Duplicate while original executes | Reject duplicate without changing original result or lifetime |
| T13 | CPU/NPU | Close with pending callback/token | No premature teardown; cancellation allows retry after pending-token timeout |
| T14 | CPU/NPU | Change observation context, then restore | Preserve refusal and retry; no false completion or retirement |
| T15 | CPU | Access inherited token after fork | Reject before taking inherited locks; no device action |
| T16 | NPU | Torch input -> Simpler -> Torch output | Full numerical comparison |
| T17 | NPU | Three dependent calls with Torch predecessors/successors | Expected 8/22/50 results without per-call forced synchronization |
| T18 | NPU | Hold GIL during query/empty_cache with pending native callback | Controlled gate proves pending state; callback progresses and results are correct |
| T19 | Isolated NPU processes | Python callback negative controls, released-GIL and native controls | Classify GIL wait using stage evidence; bounded termination and cleanup |
| T20 | NPU | Background native-reference destruction while caller holds GIL | Background release completes; Python owners remain safely managed |
| T21 | NPU | Delayed consumption with changed/deleted descriptor/config wrappers | Stable snapshot and numerical result; no promise of tensor-data copying |
| T22 | Integration | Long submission sequence with periodic retirement | Pending/roots decrease after safe retirement; no premature release or exit-only accumulation |
| T23 | Packaging/CI | Installed same-build package and adapter | No accidental source-tree dependency; required cases selected/executed and plan/JUnit checked |

Fault injection distinguishes safely cleaned failure from unknown completion. Deterministic defect regressions demonstrate failure before the repair and success afterwards. GIL negative controls run in isolated processes with bounded termination and verified device-lock release.

Delivery requires evidence for R1-R7, the same-build common native entry, a real reference consumer, required current-revision CI, declared hardware qualification, and documented success/error cleanup. Design agreement, implementation review and validation are separate records. Downstream PyPTO migration remains separate from this PR's merge criteria.

Implementation review must confirm retirement projection semantics, entry/type placement, ordinary-unregister compatibility and consumer handoff. Pending qualification includes A5 and complete current-build GIL controls. Coverage gaps in the matrix must be resolved or explicitly reconciled with the agreed scope; a green CI status alone does not close them.
