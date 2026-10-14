# Acceptance mapping — configurable run-resource depth

Per-obligation coverage for the configurable run-resource feature, mapped against the normative
acceptance table — rows 1–39 plus 17a, 17b, 19a and 20a, forty-three obligations in all.

Every row names the node that discharges it and the assertion inside that node which is the
discharge. A node named without an assertion is not coverage, and a row whose obligation no
assertion states is marked a gap rather than attached to a neighbouring one.

## How to read the evidence column

| Kind | Meaning |
| ---- | ------- |
| **native** | A deterministic in-process case. No device; the outcomes it needs are injected or driven synchronously, so it establishes ordering and refusal exactly. |
| **scene** | A real `Worker.submit` onboard case over one backend's public route. Establishes the production path end to end, observing a live device. |
| **source** | Established by reading the code — a compile-time guard, a single-owner argument, or a bound held in one place. No case executes it. |
| **gap** | Not covered. Stated here rather than implied by a neighbouring row. |

A **shared native** row covers an invariant that lives in code all four backends link, and the row
says which code. A row naming a backend-specific behaviour carries a scene, or a case against that
backend's own runner, instead.

Where a case is duplicated across backends it is because the behaviour is the backend's; a shared
invariant is proved once, in the place it lives, and the row says so.

## Execution status

Four distinct states, kept apart on purpose:

- **Executed and passing at head `931d59d40`** (a prior head of this branch, hardware): the
  capacity and turnover rows. A3 onboard job 115905104831 and A5 onboard job 115905105190 both
  show the HBG and TMR five-launched/accepted cases and the 8-held / 4-launched / 64-submission
  cases passing, and `TestA2a3TmrWideOutstandingClose::test_close_drains_a_wide_unwaited_prefix`
  passing in 8.4 s. That is evidence about `931d59d40`, not about any later head.
- **Executed and failing at head `fd0b0ce9d`**: Ubuntu Python UT, job 115950714059, reported
  3204 passed and 2 failed. Both failures were in
  `tests/ut/py/test_worker/test_wide_close_drain.py`, which called `RunHandle.done` — a property —
  as a method, raising `TypeError`. The C++ step was skipped behind the Python failure and the
  macOS sibling was cancelled mid-run. Those two call sites now read the property. This document
  does not describe that head's coverage as unexecuted.
- **Authored since, compiled, not executed**: every case this document marks *new*. They build
  clean and pass the repository's lint gates; none has been run.
- **Pre-existing**: everything else, which this branch did not change.

## Capacity, grant and admission — rows 1–9

| # | Obligation | Evidence | Node / assertion |
| - | ---------- | -------- | ---------------- |
| 1 | Default unchanged; `M = 4`; one set for a single run | native + scene | `tests/ut/py/test_worker/test_pipeline_depth_request.py` — the default request is two and `M = granted + K`; **new** `test_the_grant_is_the_depth_the_child_resolved` pins the grant at one for a request of one, and `test_a_grant_below_one_set_is_refused` that zero is not a state a child may publish; `Test*EarlyEnqueueDepthOneControl::test_no_overlap_without_the_opt_in` on all four routes asserts no two runs are ever launched together |
| 2 | `pipeline_depth=8, launch_depth=4`: ≥ 4 distinct runs launched and accepted at one coherent instant | scene ×4 | `Test*WorkedCapacityExample::test_sixty_four_runs_retire_through_eight_sets` — `observer.widest == _WORKED_LAUNCH_DEPTH`, over a two-pass coherent snapshot so the four are one instant and not a window |
| 3 | 64 sustained submissions through 8 reused sets; `M = 10`; construction stops at 8 | scene ×4 | same four cases — `len(observer.held_sets) == _WORKED_SETS`, `observer.widest_held == _WORKED_SETS`, all 64 handles drained; the entry-table width read from `mailbox_run_entry_count` in `_frame_views` |
| 4 | TMR compatible `N > 2`: 8 queued runs hit the prebuilt-arena cache; **no bind writes a device byte**; device order serial | scene (TMR) + native (**new**) | `TestA5TmrCapacityBeyondTheOldCeiling` / `TestA2a3TmrCapacityBeyondTheOldCeiling::test_five_runs_are_launched_and_accepted_at_once`, plus each suite's serial-boundary evidence reader asserting one whole operator at a time; the cache decision in `test_trb_runtime_temp_buffer.cpp::PreparedRuntimeEnvRequiresTheActiveArenaKey`; the cache *hit* in **new** `CompatibleRetainedBindsReuseTheArenaWithoutWritingIt`, built for both arches — see below |
| 5 | TMR incompatible sizing drains to depth one with a free set and **does not poison** | native | `test_chip_run_lane.cpp::IncompatibleSuccessorRetriesAfterPredecessorFence` — `PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE` from the prepare stub, successor relaunched at the front; `PrepareFailureIsTerminalWithoutPoisoningTheLane` — the lane admits the next run afterwards |
| 6 | Request above the grant fails startup; no budget required; sim still refuses one | native + source | `test_host_worker.py::test_a_chip_published_run_resource_count_the_parent_cannot_honour_fails_startup` (4 parameterisations) and `test_pipeline_depth_request.py::test_an_unsupported_grant_fails_startup_with_both_numbers`, which requires the shortfall to name both numbers; `test_device_memory_info.py::test_simulator_reports_device_memory_info_as_unsupported` for the sim refusal shape. "No budget required" is the absence of a second key: `_validated_pipeline_depth` takes one number and there is no budget parameter anywhere on the path |
| 7 | Sustained submit without waiting: `M` never exhausted; backpressure is `pending_run_depth` | scene + native | the four worked examples (64 unwaited submissions turning over); `tests/ut/py/test_worker/test_pending_run_depth.py` for the bound itself |
| 8 | FIFO launch order: a freed set goes to the next unprepared run | native | `test_chip_run_lane.cpp::OwnsFifoPreparationAndLaunch`, `EarlierActiveDispatchOrdersBeforeAnAlreadyStagedSuccessor`; upstream #2577's three publication-order regressions |
| 9 | Set withheld until copyback returned **and** zero-rc binding release **and** borrow discharged | native + scene | `test_native_run_operation_lease.cpp::WholeCloseRefusesToReclaimBehindAnUnprovenRun`, `NonDestroyingSuccessIsACleanupFailure`; the borrow half on a real route in **new** `TestA5WideDeviceResultChain::test_a_device_result_feeds_four_later_runs` — the registry witness, the refusal, and the clean release after the drain |

### Row 4: what "no bind writes a device byte" is about

Not every bind. The first one builds the arena image and uploads it, and
`ChildMemoryIsPassThroughAndPureOutSkipsStaging` pins that single `copy_to` by name. The clause is
about the binds that follow: `bind_callable_to_runtime_impl` probes the prebuilt-arena cache
first, `bind_cached_runtime_image` only wires three pointers on a hit, and
`build_and_cache_prebuilt_arena` — the one place a static arena is set up and the one
`copy_to_device` of the image — runs on a miss alone. That is in the runtime maker, which both
`test_a2a3_tmr_trb_runtime_temp_buffer` and `test_a5_tmr_trb_runtime_temp_buffer` already compile
for real.

The fixture's cache hooks recorded nothing and answered a hit with placeholder addresses, so every
bind in it took the miss path — which is why an earlier draft of this document mistook "every bind
uploads" for the production invariant and put the skip in the runner. Both were wrong. The hooks
now have a recording mode that stores exactly what the build published and hands the same
pointers, offset and image back on every later lookup.

`CompatibleRetainedBindsReuseTheArenaWithoutWritingIt` warms the cache with one bind, writes a
sentinel pattern over the whole arena, then binds **eight** further runs of compatible sizing
without releasing any of them. Each is required to hit the cache and to be wired to the recorded
arena *by address* — base, offset and shared-memory pointer — so a bind that quietly rebuilt would
answer with storage of its own. Afterwards: no further upload, no second static-arena setup, no
replacement cache entry, no new allocation, the arena pointer unmoved, and the sentinel bytes
unchanged. A ninth run with a host input then shows the distinction the clause needs — its bind
still copies nothing, its `stage_inputs` does, and the arena is still untouched. The claim is
about the shared arena, not about forbidding the H2D a run legitimately needs.

## Operation and finalization protocol — rows 10–25, 28

Fault-injection rows. They live in the native fixtures whose stubs can return each code, throw, and
leave each storage tag — the seam no scene can reach.

| # | Obligation | Evidence | Node / assertion |
| - | ---------- | -------- | ---------------- |
| 10 | Poll mapping: `0` ⇒ false, `1` ⇒ `COMPLETED` with nothing written to `pending_rc`, any other code ⇒ throws with no state change | native | `test_chip_run_lane.cpp::PollFailureReportsTheTerminalNativeError`, `ActivateStopsAtTheLaunchFenceBeforePolling`; `test_native_run_operation_lease.cpp::PollReportsNoProgressWhileAnotherOperationHoldsTheLease` |
| 11 | Completed fast paths: a second `poll`/`wait` returns without entering the runtime or taking the lease | native | `test_native_run_operation_lease.cpp::RefusedFinalizeKeepsItsWaitOutcomeAndDoesNotWaitTwice` — `g_wait_calls == 1` across two finalizations |
| 12 | A wait that returned −7 latches it and sets `COMPLETED`; the later finalize skips the wait and reports −7 | native | same case — `pending_rc == -7` with `pending_rc_set`, and `g_wait_calls` still 1 after the retry |
| 13 | Failed joined launch that completed: probe `COMPLETE` ⇒ `COMPLETED` + latched launch rc, then throw; finalize skips the wait | native | `test_native_run_operation_lease.cpp::FailedLaunchThatCompletedKeepsItsOwnError` |
| 14 | `UNSUPPORTED` decline changes no state and is not an error | native | `test_chip_run_lane_joined_launch.cpp::ADeclinedJoinLeavesTheRunToLaunchOrdinarily`; `test_native_run_operation_lease.cpp::JoinedLaunchDeclinesARetiredPredecessor` — `g_join_calls == 0` and the phase unchanged |
| 15 | Execution facts survive reservation; the finalizer reads post-drain state | native | `test_native_run_operation_lease.cpp::WaitThrowLeavesTheRunFinalizableAgain`, `CapabilityQueryHoldsBothCarriersAgainstConcurrentFinalize` |
| 16 | Exception plumbing: every throwing path rethrows the captured `exception_ptr`; no bare `throw;` outside a handler | source + native | the `std::exception_ptr failure` capture-and-rethrow shape in `chip_worker.cpp`'s native entry points; `TeardownClosesEveryRunAndSwallowsForeignExceptions` shows no terminate on a foreign exception |
| 17 | F1 — wait throw before the C finalize: object not destroyed, post-wait state restored, reservation cleared, rethrown, lane poisons | native | `test_native_run_operation_lease.cpp::WaitThrowLeavesTheRunFinalizableAgain`; the lane half in `test_chip_run_lane.cpp::FinalizeFailurePoisonsAdmissionAndCloseReportsIt` |
| 17a | `wait(-7)` → `Canonical` rc ≠ 0: −7 persisted before the C finalize, `finalize_rc` by precedence, restored state `REAPED`, no re-wait | native | `RefusedFinalizeKeepsItsWaitOutcomeAndDoesNotWaitTwice` + `LaunchFailureWhoseFinalizeFailsReportsThePoisonedLane` |
| 17b | Precedence: `finalize_rc` outranks `wait_rc`; one source per value; nothing written twice | native | same two cases — the reported message carries the finalize code, not the wait code |
| 18 | F2/F3 — tag `Zero` ⇒ entry erased and host storage recycled, resource set still withheld | native | `WholeCloseReleasesEverythingWhenEveryRunIsProvedDestroyed` (`native_runs_` empty, `runtime_bufs_` reclaimed) against `WholeCloseRefusesToReclaimBehindAnUnprovenRun` |
| 19 | F4 — `Canonical` rc ≠ 0 ⇒ run owned, low-level retryable, borrow retained, lane poisons | native | `test_chip_run_lane.cpp::EarlierFailedRunRetainsItsErrorAfterLaterRunPoisonsLane` |
| 19a | F5 — `Canonical` rc 0 ⇒ synthesized cleanup failure, quarantined, no retry | native | `NonDestroyingSuccessIsACleanupFailure` |
| 20 | F6 — `Noncanonical` ⇒ `destruction_unknown`, quarantined, counted against the grant, failure surfaced even with every rc 0 | native | `UnknownTeardownIsIsolatedAndNeverReportsSuccess` + `AnUnprovenRunsStorageOutlivesTheWorkerThatHeldIt` |
| 20a | F7–F9 — threw inside the C finalize, each tag ⇒ `destruction_unknown`, quarantined, no retry or further device call, original exception preserved | native | `TeardownClosesEveryRunAndSwallowsForeignExceptions`; `test_native_run_execution.cpp::AicoreExceptionPoisonsWithoutTryingAicpu`, `AicpuExceptionAfterAicorePoisonsWithoutReceipt` |
| 21 | Waiter + active op + finalizer: waiter hits `AlreadyComplete`, finalizer skips the wait, one destruction | native | `CapabilityQueryHoldsBothCarriersAgainstConcurrentFinalize`, `RefusedFinalizeKeepsItsWaitOutcomeAndDoesNotWaitTwice` |
| 22 | Two finalizers: exactly one enters `finalize_run_fn_`; the other throws | native | `SecondFinalizerIsRefusedAndNeverEntersTheRuntime` — `g_finalize_calls == 1` |
| 23 | Waiter wakes after erasure ⇒ `Absent` ⇒ stale throw, no dereference of recycled storage | native | `CapabilityQueryRefusesStaleIdentity`; the storage half in `AnUnprovenRunsStorageOutlivesTheWorkerThatHeldIt` |
| 24 | Another run's admission rehashes the run table while a waiter sleeps; the waiter is unaffected | native (**new**) | `test_native_run_operation_lease.cpp::AdmittingLaterRunsRehashesTheTableUnderASleepingWaiter` — a waiter is parked inside the backend `wait` for the whole window, 23 later runs are admitted, `native_runs_.bucket_count()` is asserted to have grown (so the rehash happened rather than being argued), the entry pointer the waiter holds is unchanged, its `ops_in_flight` is still 1, and its own `pending_rc == -7` lands on its own entry with `g_wait_calls == 1`. The admissions run on the case's own thread; the waiter is inside the runtime for every one of them, which is the concurrency the row is about. A `ParkedWaiter` guard owns the release and the join, so a failed premise reports its finding rather than terminating on a joinable thread |
| 25 | Two concurrent operations on one run are serialized — `poll` not-ready, `wait` blocks then proceeds | native | `PollReportsNoProgressWhileAnotherOperationHoldsTheLease`, `CapabilityQueryRefusesRatherThanWaiting` |
| 28 | Entry/frame recycle: stale `entry_generation` refused and poisons progress; `F` reclaimed exactly once; no child write after the terminal | native | `test_scheduler.cpp::StaleFrameIdentityWithholdsCompletionUntilChildQuiesces`, `ThirdDispatchCannotMutateTwoOccupiedFrames`, `TransportSaturationResumesOnTheAcknowledgedFrame`, and **new** `MailboxPipelineSchedulerFixture::AStagedRequestParkedByFullTransportResumesOnTheAcknowledgedFrame`; the child-side store order in `tests/ut/py/test_worker/test_request_frame_handback.py` |

### Row 28: the two transport cases are different claims

`TransportSaturationResumesOnTheAcknowledgedFrame` drives a `LocalMailboxEndpoint` directly. It
establishes the **endpoint's** behaviour: that `available_request_transport()` reports zero while
both frames are out, reports one again after an acknowledgement, and that a publication onto the
reclaimed frame overwrites neither the in-flight payload nor the entry of the run that
acknowledged it. Nothing in it is a scheduling decision — the test publishes by hand.

`AStagedRequestParkedByFullTransportResumesOnTheAcknowledgedFrame` is the **scheduler's**. Three
runs are submitted to a real mailbox endpoint whose child withholds both hand-backs. The scheduler
alone decides what to publish: the first run goes active, the second stages, and the third is left
in the ready queue. Under `scheduler.loop_mutex()` the case reads that only
`MAILBOX_REQUEST_FRAMES` payloads reached the child, that the third task is still `READY` rather
than claimed or failed, and that `can_stage()` is false — the gate a staged successor is refused
by, whose inflight and staged-lane terms both have headroom at `M = granted + K` with two
dispatches out.

One frame is then released, and the case waits for the **acknowledgement itself** before it reads
anything about a republication. That step is load-bearing rather than tidy. The parent never
clears a frame's state word, so the frame still carries the previous publication's `TASK_READY`
for as long as the child holds it: a poll for "the frame is ready" would accept the payload that
was already there and conclude nothing. The child's hand-back writes `IDLE` before the
acknowledgement it publishes, so once that acknowledgement is observed, any state other than
`IDLE` on that frame is a *new* publication — and the read is an acquire against the parent's
release store, which orders the entry and the frame header it wrote first.

What is then asserted is the third task's own identity, not merely that something arrived: the
entry the frame names carries `tasks[2].task_slot`, is neither live run's entry, and the frame
carries `PREPARE_READY` because the third run of the pipeline is a staged successor. The run that
made the acknowledgement has published no terminal at that point — deterministic, because this
child publishes one only when the test releases it — so no completion, retirement or promotion
can be the cause. The withheld frame is untouched, no payload is re-delivered, no failure is
reported, and all three runs then retire in order.

## Outcomes, failure and close — rows 26–34

| # | Obligation | Evidence | Node / assertion |
| - | ---------- | -------- | ---------------- |
| 26 | Repeated late `wait()`/`result()` after reuses and after `close()` return the same outcome; `None` on success | native | `tests/ut/py/test_worker/test_error_propagation.py` and `test_submit_guard.py`; at width, **new** `test_wide_close_drain.py::test_close_drains_a_wide_unwaited_prefix_and_leaves_no_run_registered` reads every handle's `result()` twice after the Worker is shut and requires `None` both times |
| 27 | A failed run's late `wait()`/`result()` returns the same error object every time | native | **new** `test_wide_close_drain.py::test_a_failure_in_a_wide_closed_prefix_stays_the_run_s_own` — `first.value is second.value` on two post-close reads; plus `test_chip_run_lane.cpp::EarlierFailedRunRetainsItsErrorAfterLaterRunPoisonsLane` for the lane-side retention |
| 29 | Execution rc ≠ 0 poisons the lane; each already-launched successor still reports its **own** outcome | native + scene | the lane half in `test_chip_run_lane.cpp::EarlierFailedRunRetainsItsErrorAfterLaterRunPoisonsLane`; the successor half in **new** `test_scheduler.cpp::AFailureAmongLiveRunsIsNotItsSuccessorsOutcome` — `M` runs live at once, one publishes a failure, and every other run's completion is asserted `SUCCESS` with an empty message, keyed by its own task slot. On a real route and at width, `TestA5TmrWideFailureAttribution::test_one_failure_amid_a_wide_prefix_is_its_own` and the narrow `Test*TmrEarlyEnqueueErrorAttribution::test_a_failing_successor_reports_its_own_status` |
| 30 | Partial: set quarantined, lane poisoned, no further launch, close reports it, no per-resource free on abandon | native | `test_native_run_execution.cpp::AicpuFailureAfterMarkedSubmissionStaysPartial`, `AicoreFailureAfterMarkedSubmissionPoisons`; `AnUnprovenRunsStorageOutlivesTheWorkerThatHeldIt` for the no-free half |
| 31 | A bind/allocation failure is a pre-launch refusal, not a poison; the next submission still succeeds | native | `test_native_run_execution.cpp::AicoreFailureIsSafePrelaunchFailure`, `ArmingFailureReportedAsRcStaysSafe`, `PermitIsOneShot`; `test_chip_run_lane.cpp::PrepareFailureIsTerminalWithoutPoisoningTheLane` |
| 32 | Error-text pressure: every one of `M` concurrent failures keeps its own full text and attribution | native (**new**) + scene | `test_scheduler.cpp::EveryConcurrentFailureKeepsItsOwnFullText` — one run per run-state entry, all live together, each failing with a distinct text filled to `MAILBOX_ERROR_MSG_SIZE - 1`; every completion is required to carry its own whole text and its own `code=`, no run reports twice, and a run admitted into a recycled entry afterwards is required to report its own text and not the previous occupant's. The scene half is `TestA5TmrWideFailureAttribution` |
| 33 | Caller DEVICE allocation refused in flight, accepted after retirement | scene | **new** `TestA5WideDeviceResultChain::test_a_device_result_feeds_four_later_runs` — see below; and the three-link `TestA5ThreeRunCapacity` / `TestDeviceResultChain` siblings at the old width |
| 34 | Close drains an unwaited prefix of > 3 runs | scene + native (**new**) | `TestA2a3TmrWideOutstandingClose::test_close_drains_a_wide_unwaited_prefix` — five submitted and unwaited at `pipeline_depth=5`, `outstanding > 3` read from the Worker's registry, the drain left to the fixture's `close()`. What happens *after* that close is `test_wide_close_drain.py`, which owns its own Worker: `_accepted_run_handles` and `_abandoned_run_handles` both empty, `_has_live_resources()` false, and every handle terminal with its own repeatable outcome |

### Row 33: what the device-chain case observes, and why

The obligation is that a DEVICE allocation a live run names cannot be released, and can be once
the run has retired. The case makes one producer's output the input of **four** separate later
runs, so the allocation is borrowed across more live consumers than the old per-set layout could
carry, and the last of them is the last submission the case makes. The four consumers each use a
different control scalar, so four correct and *different* results are the evidence that each of
them read that same allocation.

The borrow window is read from the Worker's own accepted-run registry — the same collection, under
the same lock, that `Worker.free` scans — and every one of the five runs is required to be in it,
with its cleanup unpublished, at the instant before the refusal. That is what makes the refusal
entailed rather than raced for: a run leaves that registry only inside `RunHandle.wait()`, on the
thread that calls it, so a caller that has waited nothing still holds every run it submitted. An
earlier draft asserted the refusal after sampling that *some* frame was launched, which does not
entail that a run naming *this* allocation is live; that premise is gone.

Width is asserted separately from chain length, and for what it is: more than three runs are live
*borrowers* of the allocation at that instant, with none still parked in admission. That is a
parent-side reading, which is the side the refusal is on. It is deliberately **not** a claim that
more than three native resource sets are occupied together — a parent handle outlives the native
run it names, since only its wait retires it, so the two counts are not interchangeable. Native
occupancy at width is the worked-capacity examples' claim, read from the entry table itself by a
two-pass coherent snapshot. If the free unexpectedly succeeds the case records it, drains, and
fails without releasing anything a second time.

## Endpoints, diagnostics, teardown and sim — rows 35–39

| # | Obligation | Evidence | Node / assertion |
| - | ---------- | -------- | ---------------- |
| 35 | Two endpoints: per-endpoint grants; each reaches its own overlap; outcomes aggregate | scene + native | `TestA5TmrWideTwoEndpoints::test_each_endpoint_carries_a_wide_grant` — both endpoints negotiate `_WIDE_SETS` from their own published counts, a wide prefix is admitted on each interleaved, every result is its own. The per-endpoint overlap at the old width is `Test*TmrEarlyEnqueueTwoEndpoints::test_each_endpoint_enqueues_early`. Differing grants between children: `test_host_worker.py::test_a_reconciled_chip_resource_grant_is_taken_as_published` takes `[2, 1]` from two children in their own order |
| 36 | Two endpoints: cross-endpoint DEVICE address refused with a clear error | native | `tests/ut/py/test_worker/test_child_addr_guard.py::TestSubmitDispatchGuard::test_child_arg_to_wrong_worker_rejected` — a Worker with two chip endpoints, an allocation live on worker 0 submitted to worker 1, `ValueError` matching `"not a live allocation on target worker 1"`, and `fake.submit_next_level.assert_not_called()` so nothing reached native. The group form is `test_group_member_child_arg_wrong_worker_rejected`; the check in isolation is `TestDispatchResolution::test_unique_target_but_wrong_worker_rejected`. This is the production path: `Orchestrator.submit_next_level` reaches the same `_child_prov_check_dispatch_locked` a scene submission does |
| 37 | Diagnostics: with swimlane on and 8 sets, a 4th diagnostic run waits for a bank and still collects; released only on report; mapped by epoch | native (partly **new**) + one outstanding item | **the wait itself**, in **new** `test_chip_run_lane_joined_launch.cpp::ADiagnosticSuccessorWithNoTerminalBankStaysPreparedUntilOneIsReleased` — run 372 behind run 371, every bank held, no joined launch reaching the backend, the successor left prepared, then the same run joining exactly once after a release. The bank table in `test_chip_swimlane_run_terminal.cpp::BanksAreHeldByEpochAndComeBackOnRelease`, `AWithdrawnRunReturnsItsBankUnused`, `AnUnprovenRunWithdrawsItsBankRatherThanReturningIt`, `FinalizeReturnsEveryBankIncludingWithdrawnOnes`, `ANonRetainingCollectorStillReturnsAProvenUnusedBank`, `ABankAcquiredForAnArmThatRefusedComesBack`. The front's separate arming path in **new** `ARunThatFoundNoBankStillCollectsAndIsServedOnTheNextRelease` and **new** `test_sim_callable_pool.cpp::TheSimRunnerHandsDiagnosticBanksOutByEpochAndRefusesBeyondTheBankCount`. The onboard copy of the arming helper is in no `cpput` target and is a twin of the sim definition asserted here |
| 38 | Child death holding an entry ⇒ `ENDPOINT_FAILURE`; entry reclaimed only at teardown | native | `test_scheduler.cpp::MailboxChildExitTest::PreservesOnlyMatchingTerminalResults` (4 parameterisations: exit 0/7 × stale identity), `PoisonedProgressWithholdsCompletionUntilTheChildExits` |
| 39 | Sim parity: same admission, retirement, refusals, destruction boundary and bank mapping; no joined launch | native (**new**) | `test_sim_callable_pool.cpp::TheSimRunnerHandsDiagnosticBanksOutByEpochAndRefusesBeyondTheBankCount` for the bank surface, against the real `SimDeviceRunnerBase`; **new** `test_pipeline_depth_request.py::test_the_grant_follows_the_route_that_can_order_two_submissions` for the grant itself, which is one set on both simulated platforms because sim answers no to joined launch. Sim's own C `prepare`/`launch`/`poll`/`wait`/`finalize` are executed by the existing `st-sim-*` suites over the public route; they are not transferred from the shared lane — see below |

### Row 37: where a diagnostic run actually waits

The row says a 4th diagnostic run waits for a bank and still collects. It does, and the wait is in
the **lane**, not in the arming helper. Getting that the wrong way round — as an earlier draft of
this document did — turns a run that was held back into a run that launched and lost its snapshot,
which is a different and weaker claim.

**The decision.** `DeviceRunnerBase::can_join_diagnostic_run` ends with
`can_take_run_terminal_bank(succ.identity.run_epoch)`. A false answer reaches
`ChipRunLane::permits_joined_diagnostic_launch`, and `launch_ready_prefix` **returns before
calling `launch_native_run_joined`** — so that successor is left `PREPARED`, with
`joined_launch_declined` deliberately not set, and its permission is re-asked on the next round.
Nothing is ordered onto the device without a bank to publish into, and no thread blocks: the run
waits in lane state until a retirement releases one. This is §12's shape.

**The assertion.** `test_chip_run_lane_joined_launch.cpp::ADiagnosticSuccessorWithNoTerminalBankStaysPreparedUntilOneIsReleased`
drives that path with the backend's answer *being* a bank table: run 372 is ordered behind run 371
at swimlane level one, every bank is held by a run it cannot see, and the case requires that no
joined launch reached the backend at all, that the refused run is 372 by name, and that it holds
no bank. One bank is then released and the **same** run — same id, same slot, asked as the same
pair — joins, exactly once. The output claim is the one the product makes: a join is permitted
only in a round where a bank was available for that epoch. It does not claim the successor has
acquired one, because the acquire happens later, at the arm.

**The bank table itself** — held by epoch, refused when full, handed on at release — is
`test_chip_swimlane_run_terminal.cpp`'s: `BanksAreHeldByEpochAndComeBackOnRelease` and its five
siblings. The two are layered, not repeated.

**The arming helper is a different path, and it is the front's.** A run at the front launches
ordinarily and consults no joined permission, so `arm_chip_swimlane_run_terminal_bank` can return
zero there and the run proceeds without a snapshot.
`test_chip_swimlane_run_terminal.cpp::ARunThatFoundNoBankStillCollectsAndIsServedOnTheNextRelease`
covers what the collector then does — the run keeps recording, publishes into nobody else's bank,
and is served on the next release — and
`test_sim_callable_pool.cpp::TheSimRunnerHandsDiagnosticBanksOutByEpochAndRefusesBeyondTheBankCount`
drives the production helper on a real `SimDeviceRunnerBase` through both of its zero-returning
branches: no shared region, where the acquire succeeds and the arm refuses and the credit must
come back; and a full table, where the acquire refuses. The **onboard** copy of that helper
compiles into no `cpput` target — nothing under `tests/ut/cpp/` builds
`onboard/host/device_runner_base.cpp` — so it is reached only by the onboard scene suites under
CI. It is a line-for-line twin of the sim one that is asserted here, which is why this is a
difference in where the assertion lives rather than an unexercised branch.

### Row 39: what is sim code, and what the grant now means

Two parts of this row are shared code and one is not, and a third part changed shape under the
run-resource refactor that landed on top of this work.

**Shared.** Admission, the operation lease and the retirement bookkeeping live in
`src/common/worker/chip_worker.cpp` and `src/common/worker/chip_run_lane.cpp`, which a sim build
links unchanged. A sim copy of rows 10–25 would re-execute the same translation units.

**Not shared.** The C entry points beneath that lane are sim's own — `prepare`, `launch`, `poll`,
`wait` and `finalize` all have separate definitions under `src/common/platform/sim/host/`. So the
lifecycle rows are *not* transferred to sim merely because the lane is shared, and this document
does not claim they are. What executes sim's own entry points is the existing `st-sim-*` scene
suites over the public route.

**Joining and granting are now separate questions, and sim answers them differently.** The grant
used to be decided by `_local_task_frame_count`, which conflated the run-resource sets a child
holds with the task frames a successor would occupy, and so handed a single set to any route whose
child could not order one native submission behind another — including both simulated platforms.
That conflation is gone: `_local_run_resource_set_count` returns the depth the child resolved, with
no platform, runtime or joined-launch term, because request frames are a fixed independent pool
that only carries hand-offs. A simulated child still answers *no* to joined launch —
`src/common/platform/sim/host/c_api_shared.cpp` defines `supports_joined_native_launch_ctx` as an
unconditional `return 0;` — and now still receives the grant it asked for. The two facts are
asserted separately: the grant in
`test_pipeline_depth_request.py::test_the_grant_is_the_depth_the_child_resolved` and
`test_host_worker.py::test_a_chip_child_grants_the_depth_it_resolved`, the joined-launch answer by
that definition. What a sim route therefore does *not* get is a joined launch, which is exactly
what this row says.

The bank surface sim owns is covered by the runner case above.

## Outstanding obligations

No contract row is left without an assertion. What remains is coverage that would add nothing, and
it is listed so that its absence is a decision rather than an oversight.

### Coverage that would add nothing

Each has layered evidence already, and none is required by a contract row.

- A wide-close scene on the three onboard routes other than a2a3 TMR. The close path is shared and
  is asserted twice — once on a real route, once at the Worker level including the post-close
  state.
- A two-endpoint scene for row 36. The refusal is already asserted on the production path with two
  endpoints registered; a live second chip child would re-prove the same guard.
- A hardware scene for the row-37 wait. The lane decision and the bank table are asserted
  natively, and the arming helper is asserted on the sim runner whose definition is a twin of the
  onboard one; a scene would observe the same three through a device.
- An onboard-runner `cpput` target. `onboard/host/device_runner_base.cpp` is in no such target, so
  its copy of `arm_chip_swimlane_run_terminal_bank` is reached only by the onboard scene suites.
  It is a line-for-line twin of the sim definition that is asserted here.

### Execution state of what is here

Everything marked **new** in this document is authored and compiles. Nothing in it has been
executed, and the one piece of CI evidence that exists for this branch is recorded under
"Execution status" above: it belongs to head `931d59d40`, and the later head `fd0b0ce9d` **failed**
Ubuntu Python UT with two errors in `test_wide_close_drain.py`, which called the `done` property as
a method. Those two call sites are fixed; that fix is itself unexecuted.

## Limitation

Everything marked **new** in this document is authored and compiles; none of it has been executed.
The passing hardware evidence recorded under "Execution status" belongs to head `931d59d40`; the
one later head that reached CI, `fd0b0ce9d`, failed two Python unit tests, and the fix for those
is itself unexecuted. Neither is evidence about the current head.
