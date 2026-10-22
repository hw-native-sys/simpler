# Run completion fences

How an onboard run's completion is established, who owns the resources that
establish it, and what runs queued on the same streams may rely on.

## Initial publication of the run-result region

Each slot lazily allocates one result region. Preparation snapshots a zero
`published` marker into the existing launch metadata batch, without copying to
device. Launch publishes the batch from slot-owned pinned memory on the AICPU
stream before the entry fork and either kernel. Only a successful publication
marks the allocation initialized; readback occurs after drain. The pinned source
stays live through retirement or quarantine, including failed enqueue attempts.

A failed or skipped initial publication leaves the allocation untrusted. Even
allocator bytes equal to the requested epoch cannot produce a certified read or
terminal result; the existing drain decision uses its conservative fallback.
A failure after any async enqueue attempt is Partial and prevents kernel launch.
Preparation abort discards its unpublished snapshot, and a later attempt stages
a fresh marker. Once initialized, ordinary slot reuse needs no marker copy:
per-run epochs distinguish the device's results. This adds eight bytes to an
uninitialized slot's pinned publication source, with no warm-run transfer or
latency claim. Cold allocation and other initialization remain outside capture.

## The question a stream cannot answer

An onboard run submits exactly two kernels: an AICore kernel and an AICPU
kernel, on two distinct streams. `rtStreamQuery` and
`aclrtSynchronizeStreamWithTimeout` answer *"is this queue drained"*. While a
queue only ever holds one run's work, that is the same question as *"is this run
finished"* — which is why the runner used to ask it.

It stops being the same question the moment a successor is queued behind a
predecessor. A stream query would then report the predecessor incomplete until
the successor had also finished, and a stream wait would block on the
successor's kernel to answer a question about the predecessor's. Both are wrong
in the direction that matters: a run would never be reported complete until the
whole pipeline drained.

So a run gets a boundary of its own.

## Stream binding

`PreparedExecution::streams` records the actual AICPU/AICore handles at launch,
under the execution claim and before Runtime publication. Publication, kernel
submission, boundary records, completion fallbacks and queued-wait discharge use
that binding. The handles borrow their lifetime from the existing owner: A2/A3's
`RunStreamPair` retains submitted run owners through retirement; A5 retains its
bootstrap streams for owned execution and its kernel-context streams for borrowed
execution until runner teardown. Preparing a successor does not select
or replace a stream.

On an execution failure, an owned Program context keeps its bounded device drain
and reset policy. A borrowed kernel context is poisoned without a device-wide
drain or reset; the caller owns device recovery. The native eager entry
`simpler_launch_run_on_stream` binds a borrowed caller stream to an already
prepared run. Public `Worker.init(borrowed_context_generation=...)` and
`Worker.submit(..., caller_stream=...)` use this same native path. Capture/replay
is rejected by the eager entry.

## Borrowed caller ordering

The caller orders its own device input writes before the native launch and may
queue output consumers immediately after that launch returns successfully.
`simpler_launch_run_on_stream(ctx, run, caller, predecessor)` uses the existing
prepare/launch/poll/wait/finalize transaction; `predecessor` is null for ordinary
launch, or names the same joined-launch predecessor as `simpler_launch_run_joined`.
The caller must serialize submissions and keep its stream, external buffers and
native run storage alive until finalization proves completion. Launch returns an
enqueue result; wait/finalize still report the run's device verdict.

```text
caller:  prior work -> record Start ------------------------ wait CpuDone -> record Tail -> later work
AICPU:                wait Start -> metadata/descriptor H2D -> record Fork -> kernel -> wait CoreDone -> record CpuDone
AICore:                                              wait Fork -> kernel -> record CoreDone
```

`Fork` precedes the AICPU kernel: that kernel can wait for AICore's handshake,
so putting the fork after it would deadlock. `CpuDone` is the existing run fence,
recorded after both the AICPU kernel and the wait for `CoreDone`; caller ordering
requires this whole-operator join even when cross-run early enqueue is disabled.

`RunCallerOrder` owns three reusable events per occupied slot, allocated lazily
for explicit caller launches. It holds a reference to the run's existing AICPU
boundary until `Tail` proves the caller consumed that wait. Poll therefore also
queries `Tail`, and drain discharges that reference before retirement. Ordinary
program launch borrows no caller stream. An asynchronous descriptor publication
uses the same AICPU-to-AICore fork even without a caller; a cached descriptor with
no caller queues neither a copy nor a fork. Timing marks are independent
and do not substitute for these ordering edges.

An attempted record or wait may be enqueued even when its API reports an error.
The launch transaction is therefore `Partial` from the first caller-edge or
asynchronous descriptor-copy attempt, with its resources retained. If no tail
was recorded, bounded synchronization of the bound streams is the fallback
proof (the two private streams, and the caller stream when present). A failed proof keeps references
quarantined; borrowed execution never resets or synchronizes the whole device.
Failure does not establish the caller's exit dependency, so the caller must not
consume output merely because launch returned an error.

This entry is eager-only. It validates the current borrowed RTS context and
rejects an active caller capture before launch metadata publication. Preparation
and callable registration take place outside capture. The common eager launch
publishes its Runtime descriptor asynchronously on the actual AICPU stream;
its entry fork orders the AICore reader behind that copy. The slot retains a
pinned immutable source through run retirement. A copy attempt marks Partial
before entering RTS, so enqueue failure cannot acquire a no-submission receipt.
The two kernel boundaries consume an ordinary publication fork; unproven partial
work requires bounded private-stream discharge or quarantine. Bulk HBG arena
publication and DFX writes retain their own synchronization. The compatibility
`simpler_kernel_mode_launch` stub and capability bit remain closed; public Worker
eager uses the shared native lane. Native completion events and their caller
references are not a captured-graph lifetime contract.

A direct L2 Python `Worker` owns the native `_ChipWorker` itself. Its callable
handles, argument imports, source uses and close journal belong to `Worker`;
there is no intervening public Python `ChipWorker` registry or lifecycle lock.
The public compatibility `ChipWorker` is the only route that retains its own
wrapper lifecycle; the forked chip-child path holds the native binding directly,
as direct L2 does. Every route initializes the native runtime through the same
internal binary-path/logging adapter and executes through the existing native
run lane. `Worker`'s operation lease and owner-thread close checks remain the
direct L2 lifecycle authority.

The direct `ChipWorker::submit_chip_run` / internal `_submit_chip_run_direct`
entry accepts an optional caller stream. The lane validates it before admitting
or preparing any new run. A successful return guarantees actual enqueue, not
merely a prepared handle: it tries the existing joined-launch route, and drains
a predecessor before launch if that route declines or the configured launch
depth is one. Launch errors are thrown before returning instead of being deferred
to the handle. Without a caller stream, the existing prepared-successor behavior
is unchanged.

The stream is recorded on that run, including a declined join's ordinary-launch
fallback. Poll/wait/finalize and argument borrows remain the lane's existing
lifecycle. Caller ordering covers device work; Program host output copy-back
still requires run completion/finalization. The public `Worker.submit` exposes
this argument on direct L2 and supports borrowed RTS initialization.

`ChipWorker::init` / internal `_ChipWorker.init` can select a borrowed,
already-current RTS context with `borrowed_context_generation`. The value must
be nonzero; omission selects owned-device initialization. Both routes load the
same runtime and create the same native storage and lane. Borrowed initialization
rejects SDMA/simulator options. Both onboard ownership policies manage internal
workspace and accept the same optional byte budget. Its config is the native
admission input; each submission still supplies its own config. Generation is retained in the native environment
binding; public Worker forwards it on the direct L2 onboard route.

ChipWorker retains a host-only environment identity from native context creation
through successful teardown: process, non-repeating incarnation, nonzero logical
context generation, device and owned/borrowed policy. Borrowed generation is
supplied by the caller; owned generation belongs to its native incarnation.
Failed rollback/teardown preserves the identity and runtime binding for cleanup.
Native context validation rejects a foreign process before entering the runtime.
The run lane also retains its immutable creating process, independently of the
worker pointer or environment lifetime. Inherited run handles and lane operations
reject before taking the lane mutex, including cached status queries: a parent
thread may have held that mutex at fork. Non-throwing stop/detach cleanup in the
child leaves the inherited lane untouched; worker destruction retains its native
storage and library without locking the lane or invoking device cleanup. This
is a refusal contract, not permission to execute inherited device work.
Buffer sources capture this identity and compare it on wrapping, descriptor
validation and attachment; it is independent of run, lease and Buffer generations.
The identity associates the source with the runner that validates the adopted
RTS context; it neither detects an external reset nor proves pointer ownership.

Before lane progress or native phase mutation, current-context validation rejects
a mismatched RTS context with `ExecutionContextRefused`, a `std::runtime_error`
subclass translated to the Python `RuntimeError` subclass of the same name.
The classification reaches `RunHandle.wait`/`result` without a separate Python
preflight callback; refusal preserves waiter retry and run references. Restoration
permits retry with the existing run handle. Other native wait errors remain
terminal; the C ABI validation status codes are unchanged.
Python operation leases check the same native context outside the lifecycle lock
before the operation body. This includes L2 submission, registry and memory/IO
operations and automatically covers new leased APIs. The source-only
`device_location` query explicitly opts out; its lifecycle lease still pins the
tree. Direct native allocation/free/copies, device-memory queries and diagnostic
flush enforce their own backend-entry boundary. Close checks before
claiming CLOSED or a teardown attempt; a rejected claim preserves the prior
lifecycle. An accepted close failure keeps CLOSED and retryable cleanup debt.
Native register/unregister and the legacy Python registry enforce their own
pre-mutation boundary through the same validator.
A failed borrowed initialization explicitly closes before destroying the runner;
if close fails, it retains the context, library and storage for `finalize`, and
refuses reinitialization. Successful rollback leaves an uninitialized worker that
can retry. Successful `finalize` is terminal. If the worker itself is destroyed
without completing close, it detaches surviving handles with an error and retains
native storage, lane snapshots, context and runtime library for the process
lifetime. It cannot prove safe reclamation at that point.

Public direct L2 `Worker` consumes this native lane for owned and borrowed
eager calls. Captured-graph ownership remains a separate requirement.

A failed preparation is a distinct terminal outcome. Only a rollback that
released its bindings, execution preparation and saved resource plan publishes
its host-slot release marker with the attempt's run_epoch. ChipWorker consumes
that exact epoch and reports `RunPreparationRefused`. The lane releases that
run's caller borrow, and Python releases that invocation's source reservation.
Unknown or stale release state retains borrows and poisons lane admission;
carrier destruction alone is not release evidence. Resource-wait flow control
keeps its existing live plan and is not a terminal release verdict.

A launch setup failure uses the same release verdict only after an explicit
`NotStarted` result and successful finalize. The platform clears this proof
before calling the launch backend: a backend exception cannot infer it from a
missing ActiveExecution. The lane preserves the original launch error while
retiring this invocation's references and allowing a valid retry. Partial,
ambiguous and failed-cleanup paths do not produce this verdict.

The common onboard and simulator `launch_execution` entry owns the prepared
state and allocates its `ActiveExecution` record before invoking the backend.
The backend receives a reference and cannot consume that owner. An explicit
`NotStarted` result returns the prepared state for rollback; a partial result
or escaping backend exception returns an active owner without acceptance.
The hand-off after submission only moves existing owners and allocates no
memory. Query, wait and finalize receive that owner; the existing drain and
failure-isolation rules decide retirement. Partial results and backend
exceptions close runner admission through the same policy, after the owner
is installed. A failing recovery policy cannot discard the owner or replace
the launch rc. An exception or missing receipt is never a no-submission proof.

## Callable registration and execution mode identity

The native registration and execution-mode contracts apply to both owned and
borrowed runners; the caller-ordering details for borrowed submissions are in
the preceding section.

Native `ChipWorker::register_callable(image, size)` (Python
`_ChipWorker.register_callable(callable)`) returns a runtime-assigned ID on an
initialized PROGRAM or KERNEL worker. Both modes use the optional
`simpler_register_callable_v2` symbol; a KERNEL context uses the kernel
preparation path internally. Explicit-ID `register_callable(id, callable)`
remains available and shares the registry; an ID that is already registered is
rejected with INVALID_ARGUMENT. A runtime without the optional entry can still
initialize and serve its existing APIs; requesting auto-ID registration raises
an explicit unsupported-entry error. Neither overload initializes the worker.
The public Python `ChipWorker` wrapper retains its existing CallableHandle API.

Automatic and explicit registration share one ID namespace, one image-buffer
cache and one 64-slot table (`MAX_REGISTERED_CALLABLE_IDS`, mirrored by the
AICPU executor's `orch_so_table_`); the slot number is the ID. Equal images
receive distinct IDs, with upload sharing governed by that cache. Capacity is a
slot count, not a memory budget: with all 64 slots occupied, registration fails
until an ID is unregistered. Failure leaves the output ID at -1 and preserves
the runtime's existing cleanup ownership; exhaustion returns INVALID_STATE.
IDs can be reused after explicit unregistration, so callers retire all users
first and discard stale IDs. The registration and unregistration entry points
serialize internally on the context, so an auto-ID scan and its claim cannot
race an explicit-ID claim of the same slot; finalize takes the same lock, so
teardown cannot clear the registry under a registration. Runs, initialization
and destroy still rely on the caller contract. The old C symbol's signature and
kernel-mode capability bit remain unchanged. Adapters must explicitly select
v2; an integration-branch binary expecting a different signature under the old
symbol cannot be loaded as if it were compatible.

Execution mode is latched per native device-context handle, not per process or
submission. This handle identifies the C++ runner; it is not the borrowed
ACL/RTS current-context handle. `ExecutionModeLatch::latched_mode()` returns an
optional identity snapshot: empty means unclaimed, not PROGRAM. Identity,
initialization phase and resource retirement are separate facts. The latch mutex
serializes identity claims only; callers serialize native init/finalize/destroy
against operations using that handle. Concurrent claims cannot choose different
identities, but this does not make concurrent runner initialization supported.
A successful mode claim is never undone by finalize or a later initialization
failure. The same handle cannot switch between PROGRAM and
KERNEL; a finalized PROGRAM handle can initialize as PROGRAM again only when
its cleanup proof permits it. This C ABI permission does not apply to
`ChipWorker`, which rejects reinitialization after finalize. Borrowed submissions
use the shared native run machinery, so sharing prepare/launch functions is not
a mode violation.

## The boundary

Each stream records an event immediately after its own kernel:

```text
AICore stream: kernel(N) -> record core_done(N)
AICPU  stream: kernel(N) -> record cpu_done(N)
```

Stream order is what makes this a proof: nothing else enters the stream between
the kernel and its record, so the event cannot complete before the kernel it
follows has exited. A run is device-complete when **both** of its recorded
boundaries complete.

Three things that are *not* proofs, each of which is a way to get this wrong:

- **One boundary.** The two kernels handshake — the AICPU Run kernel spins
  waiting for AICore workers to report in — so either kernel can still be
  running while the other has exited.
- **A device-side handshake or completion flag.** Those are published from
  inside the kernel, before it returns.
- **A timeout expiring.** Expiry says the host stopped waiting. It says nothing
  about what the device is still doing, so it can never license freeing memory
  the device may hold.

The host must not wait for the AICore boundary before submitting the AICPU
kernel, for the same handshake reason: the AICore kernel may be spinning for a
kernel that has not been submitted yet. The two submissions stay back to back,
AICore first (see the launch-order note in each arch's `launch_execution_impl`), and
only the boundaries are waited on.

## Completion is not the verdict — a measured constraint

A boundary proves the kernel ahead of it **exited**. It carries nothing about
whether the device was happy with it, and on this SDK the two facts travel on
different channels.

Measured on a2a3 / CANN 9.0, on a run whose AICPU kernel returns a fatal status
(`tests/st/runtime_fatal_codes`, `scope_deadlock` and `explicit_fatal`):

| Call | Result |
| ---- | ------ |
| `aclrtSynchronizeEventWithTimeout` on both boundaries | `0` — both complete, so both kernels did exit |
| `rtStreamQuery` on both streams | `0` — drained, no error |
| `aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL)` before any synchronize | `0` — nothing recorded |
| `aclrtSynchronizeStreamWithTimeout(stream, 0)` | `107000` — a zero timeout is rejected outright |
| `aclrtSynchronizeStreamWithTimeout` with the run's timeout | **`507018`** |
| `aclrtPeekAtLastError` *after* that synchronize | `507018` |

So the stream synchronize does not merely *report* the exception, it is what
**materializes** it: nothing observes the fault until that call runs, and there
is no non-blocking form of it.

That splits the drain path in two, and the split is deliberate:

- **completion** — the run's own two boundaries. Bounded, run-scoped, and
  unaffected by anything queued behind them.
- **the device's verdict** — `sync_stream_pair`, because no other call produces
  one.

The verdict read is why the drain path can still touch the whole pair. It is
not a hidden completion dependency — completion is already settled before it
runs. **On a normal success it no longer runs at all**: when both boundaries
complete, the run's own record transfer reports nothing and that record says
`Ok`, `wait_run_fence` returns on that evidence alone. Every other shape still
reaches `sync_stream_pair`, and on those shapes a change that queues a
successor makes it wait for the successor's kernels to answer a question about
the predecessor's. That cost is bounded by the stream-sync timeout and is
disclosed rather than solved.

The candidate that does not require a synchronize is
`aclrtSetExceptionInfoCallback` — the driver invokes it when a device exception
occurs, so the error arrives without anyone waiting. It is device-scoped rather
than run-scoped and brings its own threading and lifetime contract, so it is a
subsystem rather than a call to drop in.

That subsystem now exists, is consumed, and refuses admission. The process owns
the driver's single callback slot (`host/device_fault_monitor.h`), and each
runner records the notices naming streams its own runs submit on, against the
device's live generation (`host/device_health_state.h`, including the fence that
retires a generation at a confirmed reset so one recovered fault cannot
quarantine a card for the life of the process).

**A notice the runner matches to one of its own run streams refuses future
admission, and does nothing else.** `device_admits_new_run` composes that
suspicion with the arch's own quarantine, and the shared c_api asks it at
prepare and at launch. The run whose finalize recorded the notice keeps the
outcome its own channels gave it; no drain, reset or recovery starts from a
notice; no resource is reclaimed; and the teardown consumer records only.
Unattributed, undecided, lost and dropped notices refuse nothing, and neither
does the absence of a notice — silence is not evidence of health.

What the match is, precisely: the notice names this device in the logical id
space, and its stream id is one this runner's run boundaries were recorded on.
That is **membership, not provenance.** It does not identify the submission that
faulted — on a5 the same `stream_aicpu_` also carries binary load, AICPU init
and callable registration — and it carries no time, so a late or recycled
identity can refuse work the device would have served. That availability cost is
the deliberate trade: a notice carries no run identity and can arrive late (16 s
is the longest lag measured, not a bound the SDK promises), so no rule over this
channel can distinguish a fault that impaired a run from one that did not.

Two consequences follow from membership-not-provenance, and both are accepted
rather than worked around:

- **A control-plane failure the host already reported synchronously can still
  refuse later admission.** On a5 the registration, binary-load and AICPU-init
  submissions ride `stream_aicpu_`, which is also a run-boundary stream, so a
  device exception raised by one of them matches. The caller has already seen
  that failure as a return code; the notice refuses admission on top of it.
- **There is no guarantee that an ordinary close and re-init is recovery.**
  Admission returns only through the existing confirmed-generation-retirement
  path — the force reset that confirms, then `retire_after_confirmed_device_reset`.
  A reset that was not confirmed, or a notification delivered into the new
  generation, leaves the runner refusing.

A test that deliberately drives a device-side failure therefore must not share a
worker with cases that expect a healthy runner. In the `prepared_callable`
directories that case lives in its own class
(`TestPreparedCallableRegistrationFailure`), because the worker fixture is
class-scoped and `DevicePool.allocate` refuses rather than queues — a
finer-scoped worker requested while the class's is still cached would need a
second device. Class scope makes the two lifetimes sequential instead. That
confines shared mutable runner state between an intentional-failure case and the
positive cases around it; it is not a claim that the device is restored, since a
notification delivered later can still name it.

What a notice still cannot do is decide a run. The trigger is about the device
and stays on its own axis: **a decided run result neither causes nor vetoes a
device-health action.** A run that failed for its own reasons can leave a healthy
card, and a run that succeeded can sit on a card that faulted underneath it —
both were measured here. Result, health and resource retirement stay three
separate decisions with three separate inputs, which is what the design has said
since node A and what this channel must not quietly merge.

Two limits of the generation fence, stated because neither is fixable from inside
this channel. It skips notices **already in the ring** at the reset, so a fault
caused before the reset but *delivered* after it still reads as the new
generation's — the notice carries no timestamp, so the two are indistinguishable.
And the stream identities it retires are held with finite capacity, so a
sufficiently long generation reports attribution as undecided rather than
claiming a notice is not its own.

### What this leaves open

Run-scoped completion is delivered, and so is a run-scoped *normal* drain: a
run whose boundaries completed, whose record transfer reported nothing and
whose own record says `Ok` no longer waits on the pair. Issue #2267 stays open
for the shapes that still do. Concretely, what remains owed:

- ~~an error channel that reports a device exception without a stream
  synchronize~~ — **delivered**, recorded per device generation, and refusing
  future admission on a matched notice. It does **not** make removing the
  normal-path synchronize safe: the rows that keep synchronizing include the
  ordinary successful run that produces no notice at all, so this channel is a
  fallback for uncovered work rather than a replacement for the verdict read;
- every supported execution variant now has a terminal publication path. a5's
  `host_build_graph` legacy executor was the last without one, and it now folds
  and publishes on the same wire, epoch, selector and publisher as the other
  three. A publication path is not a record per run: a run that failed in init,
  one whose participants did not all claim the audited path, and an unmarked
  fallback the wrapper rejects publish **nothing and stay undecided — but only
  when no error is attributable to them**. `run_terminal_select` reports a
  header or participant failure on those paths too; what the missing claim and
  the unmarked mode withhold is the *success*, never the diagnosis. That is what
  keeps "no record" from reading as success. **Authority is now partial**: the
  fenced drain acts on the record to conclude a normal success, and every other
  shape still answers from the stream synchronize, with
  `report_terminal_disagreement` covering those shapes by logging rather than
  overriding;
- a decision on how a *successor's* fault is attributed, since a stream carries
  its error stickily and a predecessor's drain that falls back to the
  synchronize would otherwise report it.

`poll_execution` needs none of this. It reports completion from the boundaries
and additionally reads the streams' *sticky* error state, which is exactly what
the whole-pair query it replaced reported — so a poll that used to surface a
stream left in error still does, and a poll that never detected a device
exception (measured above: `rtStreamQuery` returns `0`) still does not.

### Measured: the verdict read does wait, and the record read does not

The paragraph above predicts that keeping the verdict read would make a
predecessor's drain wait for its successor. That is now measured rather than
argued, on a2a3, both runtimes — by `tests/st/run_retention`, which reaches a
state production admission refuses (a predecessor complete and read but not
finalized, while its successor executes on the same streams) through the fixture
in `src/common/platform/onboard/host/run_retention_probe.h`.

Two arms over one sequence, differing only in whether `sync_stream_pair` runs
before the predecessor's result is read:

| Arm | Successor's boundary, before → after the read | Successor's drain afterwards |
| --- | --------------------------------------------- | ---------------------------- |
| read the published record directly | Pending → **Pending** | 476 µs |
| `sync_stream_pair` first (the fallback shapes' path) | Pending → **Complete** | 17 µs |

**The Pending reading is the whole proof, and it is one-sided.** A read that
waited for the successor could not have produced it, so an observed Pending
settles the question. `Complete` does not settle the converse — the successor may
simply have finished on its own — and the drain does not separate those two
causes: a drain costs time on an already-finished run too, which is exactly what
the control arm's 17 µs measures. The drain column is a cross-arm comparison, not
a per-sample discriminator.

What does separate them is the read's own cost, measured twice on the same box.
A read that waits for the successor absorbs that successor's remaining device
work, so it cannot come out near the same read with nothing in flight — and the
successor's residency is the arm's to set, since the workload is a task count
passed as an argument rather than a constant compiled into the orchestration.
`tests/st/run_retention` therefore treats a `Complete` attempt as inconclusive,
escalates the successor's task count so the next attempt has more window, and
reports a `Complete` reached by a read that cost several times the unblocked
baseline as the read-ordering regression it is. It fails as "never observed"
only when the escalation is exhausted with every read at its ordinary cost.

The escalation's ceiling is half the default 16384-slot ring task window. A
single scope cannot fill that window — fanout references are released only at
scope_end, so a scope reaching the cap reports `SCOPE_DEADLOCK` instead of
running; measured on `tensormap_and_ringbuffer`, which admits 12288 tasks in one
scope and refuses 16384.

So:

- **the record read is successor-independent** — 11–13 µs, flat across runtimes
  and across successor size, and it yields a decided verdict rather than merely
  returning early;
- **the retained synchronize costs 283–429 µs with a successor in flight**, which
  on `host_build_graph` is 3.6× the entire candidate drain (≈77 µs). It scales
  with the successor's work where the candidate drain does not, which is the
  argument in one line.

The rule combining the two channels into one outcome is
`decide_run_execution` (`host/run_outcome_decision.h`), which is pure and covered
without a device. It is now what the fenced drain acts on for a normal success.
These numbers are what that path was expected to save; they were taken on the
fixture, not on the production drain, and nothing here measures the production
change.

## Ownership: the run owns the facts, the runner owns the handles

| Thing | Owner | Released by |
| ----- | ----- | ----------- |
| Stream handles | `RunStreamPair` (a2a3) / the persistent pair (a5) | `finalize()` |
| Boundary event handles | `RunCompletionFence`, one per pipeline slot | `finalize_common_impl` |
| Submitted / recorded / complete **facts** | the fence's *arming*, keyed on `NativeRunIdentity` | the run, at `retire` |

The events are runner-owned per pipeline slot rather than created per run.
`aclrtCreateEventExWithFlag` produces an event that re-records without an
`aclrtResetEvent`, so one pair can serve every run a slot hosts; creating and
destroying a pair per run would add four device calls to every dispatch and buy
nothing. This is the same arrangement as the slot's `SlotPersistentArgs` device
blocks and its retained graph-definition block.

What keeps that safe is that the *facts* are not slot-scoped. Every read and
mutation on the fence carries the run's `NativeRunIdentity`, so a poller holding
a stale identity and a later run reusing the slot both fail the check rather
than observing a boundary that is not theirs. `arm` drops the previous run's
facts and keeps the handles.

Events are created during the launch arming prologue, ahead of the first
device-visible submission. A creation failure therefore costs the run nothing:
it has submitted nothing and rolls back.

## API and timeout choices

| Operation | Call | Why this one |
| --------- | ---- | ------------ |
| create | `aclrtCreateEventExWithFlag(ACL_EVENT_SYNC)` | `Ex` events re-record without a reset, which is what makes per-slot reuse possible. `ACL_EVENT_SYNC` is completion-only — these boundaries are never read for a timestamp. |
| record | `aclrtRecordEvent` | |
| query | `aclrtQueryEventStatus` | Reports not-ready as a *status*, so a pending boundary is never confused with a failed query. |
| wait | `aclrtSynchronizeEventWithTimeout` | Bounded, at `timeout_config_.stream_sync_timeout_ms` — the same budget the whole-stream wait it replaces used, so existing timeout configuration keeps its meaning. Its timeout sentinel is `ACL_ERROR_RT_EVENT_SYNC_TIMEOUT` (507047) where the stream form returned `ACL_ERROR_RT_STREAM_SYNC_TIMEOUT` (507046). |
| destroy | `aclrtDestroyEvent` | |

`aclrtQueryEventWaitStatus` is not used: the question asked here is whether the
recorded boundary has been reached, which is the *recorded* status.

The events are created with `ACL_EVENT_SYNC`, the completion-only flag: a fence
boundary is never read for a timestamp, so it carries none of the timeline cost
a readable `ACL_EVENT_TIME_LINE` event would.

## Submitted work with no boundary

A record can fail *after* its kernel is already on the device. The fence keeps
those two facts apart, and reports such a run `Unfenced`: its own events cannot
decide it.

| Per-stream facts | Handling |
| ---------------- | -------- |
| No kernel, no queued wait | Nothing device-visible; resources roll back |
| Kernel submitted, boundary recorded | Query / wait this stream's boundary |
| Kernel submitted, boundary missing | `Unfenced` — retain resources until an independent bounded proof |
| No kernel, but a committed wait exists | Kernel parameters may be reclaimable; the wait's event reference is still live |

For an `Unfenced` run, `poll_run_fence` falls back to querying the streams and
`wait_run_fence` skips the boundary wait and goes straight to the bounded
`sync_stream_pair` — which on that path is the only evidence available, not just
the verdict read. A partial launch is by construction not a run with a successor
queued behind it, so the over-wait a stream carries costs nothing there.

Appending *another* event to an `Unfenced` run would prove nothing — the AICore
kernel may already be waiting on an AICPU kernel whose submission failed, so
nothing after it in that stream will ever run. When the fallback cannot
establish quiescence either, the existing policy takes over unchanged:
`recover_device_or_mark_unusable` marks the context unusable, `can_accept_run()`
then fails admission, and `finalize()` takes its force-reset teardown, which is
the only thing that actually invalidates the device's references. Nothing is
freed on an unproven path and nothing is re-executed.

## The cross-run join a later change will use

Opening admission to two launched runs needs the successor ordered after the
predecessor. With per-run boundaries that is two queued waits, crossed:

```text
successor AICore stream waits predecessor cpu_done
successor AICPU  stream waits predecessor core_done
```

Crossed rather than parallel, because each stream already orders itself: the
AICore edge puts `cpu_done` before the successor's AICore kernel, the stream's
own order puts `core_done` before `cpu_done`'s peer, and together the two edges
put **both** predecessor kernels ahead of **either** successor kernel. A
predecessor record must already be submitted before a successor wait on it is
queued, and no wait may point back at a successor, or the pair deadlocks.

`RunCompletionFence` carries the reference protocol this needs today, with no
production caller — inserting the waits and admitting a second launched run
belong to the change that opens admission. The normal success path no longer
synchronizes, but the fallback shapes still do, so that change also owes a
decision on what a fallback synchronize means once a successor is queued
behind the run it is draining.

1. `reserve_wait_reference(identity, boundary, waiter)` before queueing the
   wait. Reserving first is what makes the failure path decidable.
2. Queue the stream wait on `boundary_event(identity, boundary)`.
3. `commit_wait_reference` on success, `revoke_wait_reference` on failure.
4. `release_wait_reference(ref, proven_waiter)` exactly once, and only once a
   completion proof covers the successor stream `waiter` — the stream that
   actually holds the wait. Since the join crosses the streams, a proof about
   the *other* stream leaves the wait uncovered, and the call rejects it.
   `release_wait_reference_on_quiescence` is the alternative for a verified
   reset or quarantine, which invalidates every reference at once.

A reference *is* a count, and there is exactly one way to get a count wrong:
decrement one that is not yours. Every rule below closes one route to that.

- **The token moves, never copies, and a move empties the source.** Two tokens
  naming one reference would each be releasable, and the second release would
  consume some other live wait's count. Move assignment is deleted rather than
  allowed to overwrite — and so silently drop — a live destination token.
- **The token carries the fence and the arming it was minted against, and every
  mutation re-checks both.** The counters are per boundary role, so a token
  offered to the wrong fence, or to the right fence after it re-armed for a
  later run, would otherwise decrement whichever counter happened to share its
  two roles. Both are refused.
- **A fence holding any reservation or committed reference refuses `arm`,
  `record`, `retire` and `release`.** The first three keep it from re-recording
  an event a queued wait still names; `release` is the last place the event
  could be destroyed under one, so it is guarded rather than trusted, and leaves
  handles and counters intact for a caller that has since released.
- **Only a verified reset may drop a count without a per-stream proof, and it
  does so wholesale.** `abandon()` invalidates the generation and its counts
  together. Tokens still held against it then name counters that are gone, and
  `discard_stale_wait_reference` is how their holders drop them. Only the fence
  that minted a token may judge it stale, and it refuses two cases rather than
  one: a token it still recognises is current, so emptying it would leak the
  count it holds; and a token *another* fence minted is not stale here either —
  emptying it would strand the count over there, leaving that fence blocked from
  retiring with no token left to release it. "Not mine" and "no longer live" are
  different questions.
- **`NotStarted` kernels do not imply no reference.** A run whose own kernels
  never launched can still have queued a wait, and that reference has to be
  released rather than assumed away.

## Where this lives

| Piece | File |
| ----- | ---- |
| State machine | `src/common/platform/include/host/run_completion_fence.h` |
| ACL event operations | `make_acl_event_ops()` in `src/common/platform/onboard/host/device_runner_base.cpp` |
| Shared arm / record / poll / wait / retire helpers | `DeviceRunnerBase`, same file |
| Launch-transaction accounting | `LaunchProgressSink` in `src/common/worker/native_run_execution.h` |
| Per-arch wiring | `src/{a2a3,a5}/platform/onboard/host/device_runner.cpp` |
| Tests | `tests/ut/cpp/common/platform/test_run_completion_fence.cpp` |

`LaunchProgressSink` is part of this and not an aside. A submit callback now
submits a kernel *and then* records its boundary, so it has a step that can fail
after the device has already accepted work. Without the sink, that failure's
non-zero return would be read as a failure before any submission and the run
graded `NotStarted` — an already-submitted kernel reported as never launched,
with its resources free to roll back. The callback therefore marks the sink the
instant the submission is accepted, and the transaction grades a later failure
`Partial`.

Simulation records no events: its submit callbacks take the sink and ignore it,
and its grading is unchanged.

## Borrowed TMR eager invocation admission

Borrowed eager already enters the common native-run lane. The framed TMR
invocation adds registered callable occupancy and per-run binding checks to
that path; it does not introduce another executor, registration API or caller
ordering protocol.

Borrowed TMR eager calls use the existing native-run lane, caller ordering and
completion fence. Their launch package carries a 40-byte invocation header,
the device Runtime address and nonzero run epoch, followed by DEVICE Tensor
descriptors and scalar values. The callable occupancy generation is allocated
by the runner's registry and recorded independently by AICPU registration.
ID reuse gets a new generation.

The Host packet is a snapshot of the prepared Runtime image. It survives until
RTS copies the launch package. AICPU checks framing, registered counts and
generation, the run binding and Tensor geometry before materializing argument
storage owned by the executor. Orchestration borrows that storage through the
same completion and teardown path as a program call; cleanup drops the
orchestration view before clearing the storage.

The first descriptor publication still initializes its complete handshake
prefix. Program descriptor and launch-envelope routes retain their behavior;
kernel packets require permitted launch transport and cannot silently fall back
to unvalidated descriptor values. Active capture remains rejected by the eager
entry. These per-run resources do not provide captured-graph lifetime.

Registration and context finalization retain their common locking and admission
rules. A prepared ordinary native run prevents all registration entries and
unregistration from changing the registry. Refusal leaves automatic ID output
at -1 and performs no allocation or copy. A refused context finalize does not
retire the run or restore dispatch readiness; each of those states keeps its
own existing check.

Execution stream identities are recorded through the same runner helper used
by capture, independently of recording a completion fence. An unavailable
stream ID leaves fault attribution undecided under the existing health policy;
it does not establish that later unmatched notices belong to another runner.
