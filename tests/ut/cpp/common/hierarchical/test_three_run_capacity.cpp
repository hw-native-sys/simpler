/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * A third run in flight: what admission, preparation, launch and authorization do at capacity
 * three, and what they still refuse there.
 *
 * The capacity a context grants is one number, and every layer reads it rather than a constant of
 * its own: the lane admits that many runs, the orchestrator leases and authorizes that many, and
 * the run behind the last one waits. Device execution stays serial — each launch is ordered behind
 * the run immediately ahead of it — and the scope the second run always had is unchanged for the
 * third: diagnostics stay exclusive to one launched run.
 */

#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "call_config.h"
#include "orchestrator.h"
#include "worker_manager.h"
#include "pipeline_slot_pool.h"
#include "ring.h"
#include "runtime_c_api.h"
#include "scope.h"
#include "task_args.h"
#include "tensor.h"
#include "tensormap.h"
#include "types.h"

#define private public
#include "chip_worker.h"
#undef private
#include "chip_run_lane.h"

#include <gtest/gtest.h>

namespace {

// ---------------------------------------------------------------------------
// Lane: three runs over three runtime slots
// ---------------------------------------------------------------------------

constexpr size_t kCapacity = 3;

std::unordered_map<void *, uint32_t> g_slots;
std::array<bool, kCapacity> g_complete{};
std::vector<std::string> g_events;
bool g_supports_joined{true};
/** The slot whose `finalize_run` reports a device error, or `kNoFailingSlot` for none. */
constexpr uint32_t kNoFailingSlot = ~0u;
uint32_t g_finalize_fails_slot{kNoFailingSlot};
/** Every caller-reference release a run made, as `(borrow id, kept)`. */
std::vector<std::pair<uint64_t, bool>> g_releases;

uint32_t slot_of(void *runtime) { return g_slots.at(runtime); }

int prepare_run(
    void *, void *runtime, int32_t, const void *, const CallConfig *, const NativeRunDescriptor *descriptor
) {
    EXPECT_EQ(slot_of(runtime), descriptor->pipeline_slot);
    g_complete[descriptor->pipeline_slot] = false;
    g_events.push_back("prepare" + std::to_string(descriptor->pipeline_slot));
    return 0;
}

int launch_run(void *, void *runtime) {
    g_events.push_back("launch" + std::to_string(slot_of(runtime)));
    return 0;
}

int launch_run_joined(void *, void *runtime, void *predecessor) {
    g_events.push_back("joined" + std::to_string(slot_of(runtime)) + "behind" + std::to_string(slot_of(predecessor)));
    return 0;
}

int poll_run(void *, void *runtime) {
    return g_complete[slot_of(runtime)] ? SIMPLER_NATIVE_RUN_POLL_COMPLETE : SIMPLER_NATIVE_RUN_POLL_NOT_READY;
}

int wait_run(void *, void *runtime) {
    const uint32_t slot = slot_of(runtime);
    g_events.push_back("wait" + std::to_string(slot));
    g_complete[slot] = true;
    return 0;
}

int finalize_run(void *, void *runtime) {
    const uint32_t slot = slot_of(runtime);
    g_events.push_back("finalize" + std::to_string(slot));
    return slot == g_finalize_fails_slot ? PTO_RUNTIME_ERR_INTERNAL : 0;
}

/** Records what each run asked its caller references to do when it ended. */
void release_caller_buffers(void *, uint64_t borrow_id, int keep) { g_releases.push_back({borrow_id, keep != 0}); }

int supports_successor(void *) { return 1; }
int supports_joined(void *) { return g_supports_joined ? 1 : 0; }

/** A worker whose granted capacity is `depth`, with `launch_depth` of those launchable at once. */
void prime_worker(ChipWorker &worker, uint32_t depth, unsigned launch_depth) {
    g_slots.clear();
    g_complete = {};
    g_events.clear();
    g_supports_joined = true;
    g_finalize_fails_slot = kNoFailingSlot;
    g_releases.clear();
    worker.launch_depth_ = launch_depth;
    worker.initialized_ = true;
    worker.pipeline_contract_ = {PTO_PIPELINE_CONTRACT_ABI_VERSION, 0, depth, {}};
    for (uint32_t slot = 0; slot < depth; ++slot) {
        worker.runtime_bufs_.emplace_back(64, alignof(std::max_align_t));
    }
    for (uint32_t slot = 0; slot < worker.runtime_bufs_.size(); ++slot) {
        g_slots.emplace(worker.runtime_bufs_[slot].data(), slot);
    }
    worker.prepare_run_fn_ = prepare_run;
    worker.launch_run_fn_ = launch_run;
    worker.launch_run_joined_fn_ = launch_run_joined;
    worker.poll_run_fn_ = poll_run;
    worker.wait_run_fn_ = wait_run;
    worker.finalize_run_fn_ = finalize_run;
    worker.supports_concurrent_native_prepare_fn_ = supports_successor;
    worker.supports_joined_native_launch_fn_ = supports_joined;
    worker.device_release_caller_buffers_ctx_fn_ = release_caller_buffers;
}

/** One host-space tensor, which is what keeps a run inside the joined scope. */
ChipStorageTaskArgs host_args() {
    ChipStorageTaskArgs args{};
    ChipTensor t{};
    const uint32_t shape = 1;
    t.init_external(reinterpret_cast<void *>(0x1000), 1, &shape, 1, DataType::UINT8, AddressSpace::HOST);
    args.add_tensor(t);
    return args;
}

ChipRun submit_at(
    ChipRunLane &lane, uint64_t run_id, uint32_t slot, const CallConfig &config = CallConfig{}, bool activate = true
) {
    return lane.submit(
        1, host_args(), config, PipelineSlotLease{slot, 0, run_id}, run_id, run_id, nullptr, 0, activate
    );
}

using Events = std::vector<std::string>;

TEST(ThreeRunCapacityLaneTest, AThirdRunIsPreparedAndLaunchedBehindTheSecond) {
    ChipWorker worker;
    prime_worker(worker, /*depth=*/3, /*launch_depth=*/3);
    ChipRunLane lane(worker);

    ChipRun first = submit_at(lane, 1, 0, CallConfig{}, /*activate=*/true);
    ChipRun second = submit_at(lane, 2, 1, CallConfig{}, /*activate=*/false);
    ChipRun third = submit_at(lane, 3, 2, CallConfig{}, /*activate=*/false);
    // The second prepares beside the launched front. The third does not yet: a run is prepared
    // beside the run it will be ordered behind, and that run has not reached the device.
    EXPECT_EQ(g_events, (Events{"prepare0", "launch0", "prepare1"}));

    // Activation is what each authorization reaches the child as, and the second's launch is what
    // makes the third preparable. Each run is ordered behind the one immediately ahead of it, so
    // the device still runs one whole operator at a time.
    second.activate();
    EXPECT_FALSE(g_complete[0]);
    EXPECT_TRUE(second.launched());
    EXPECT_EQ(g_events, (Events{"prepare0", "launch0", "prepare1", "joined1behind0", "prepare2"}));

    third.activate();
    EXPECT_FALSE(g_complete[0]);
    EXPECT_TRUE(third.launched());
    EXPECT_EQ(g_events, (Events{"prepare0", "launch0", "prepare1", "joined1behind0", "prepare2", "joined2behind1"}));

    g_complete = {true, true, true};
    EXPECT_TRUE(first.done());
    EXPECT_TRUE(second.done());
    EXPECT_TRUE(third.done());
    lane.close();
    worker.finalize();
}

TEST(ThreeRunCapacityLaneTest, TheRunPastTheGrantedCapacityIsRefusedRatherThanQueued) {
    ChipWorker worker;
    prime_worker(worker, /*depth=*/3, /*launch_depth=*/3);
    ChipRunLane lane(worker);

    ChipRun first = submit_at(lane, 1, 0, CallConfig{}, /*activate=*/true);
    ChipRun second = submit_at(lane, 2, 1, CallConfig{}, /*activate=*/false);
    second.activate();
    ChipRun third = submit_at(lane, 3, 2, CallConfig{}, /*activate=*/false);
    third.activate();

    // The dispatched path never queues past the capacity: the lease a fourth run would need names
    // a slot another run still holds, and admitting anyway would hand it that run's resources.
    EXPECT_THROW(submit_at(lane, 4, 0, CallConfig{}, /*activate=*/false), std::runtime_error);

    g_complete = {true, true, true};
    EXPECT_TRUE(first.done());
    EXPECT_TRUE(second.done());
    EXPECT_TRUE(third.done());
    lane.close();
    worker.finalize();
}

TEST(ThreeRunCapacityLaneTest, AMiddleRunsFailureKeepsItsOwnReferencesAndLeavesThePredecessorAlone) {
    // Failure attribution at three, where there is a run on each side of the failing one. The
    // device error is injected at the native boundary rather than produced on hardware: the
    // question here is which run owns the outcome and what its references do, and that is lane
    // state. What a real device failure then publishes to a *Worker* is a separate contract, and
    // the predecessor-outcome half of it is external PR #2532's — this case does not reach it.
    ChipWorker worker;
    prime_worker(worker, /*depth=*/3, /*launch_depth=*/3);
    g_finalize_fails_slot = 1;
    ChipRunLane lane(worker);

    ChipRun first = submit_at(lane, 1, 0, CallConfig{}, /*activate=*/true);
    ChipRun second = submit_at(lane, 2, 1, CallConfig{}, /*activate=*/false);
    second.activate();
    ChipRun third = submit_at(lane, 3, 2, CallConfig{}, /*activate=*/false);
    third.activate();

    g_complete = {true, true, true};
    // The predecessor completes on its own terms: its slot's finalize reported success, so its
    // references are discharged rather than kept.
    EXPECT_TRUE(first.done());
    ASSERT_EQ(g_releases.size(), 1u);
    EXPECT_FALSE(g_releases.front().second);

    // The failing run carries its own error, and nothing the predecessor did is restated. Bounded:
    // `wait_until` reports a deadline it did not meet as `false`, so a run that stopped reaching an
    // outcome fails this assertion instead of parking the suite.
    const ChipRun::Deadline deadline = ChipRun::Clock::now() + std::chrono::seconds(5);
    EXPECT_THROW(second.wait_until(deadline), std::runtime_error);
    ASSERT_EQ(g_releases.size(), 2u);
    // Kept, not released: an unproven finalize cannot establish that the device stopped reading
    // those bytes, so the caller's release stays refused from here on.
    EXPECT_TRUE(g_releases.back().second);
    EXPECT_NE(g_releases.back().first, g_releases.front().first);

    // The run behind it had already crossed its own launch fence, so it owns device work only its
    // own drain can retire: it reaches its own outcome and discharges its own references. What the
    // failure took is the lane's ability to admit or serve anything further.
    EXPECT_TRUE(second.lane_poisoned());
    EXPECT_TRUE(third.wait_until(ChipRun::Clock::now() + std::chrono::seconds(5)));
    ASSERT_EQ(g_releases.size(), 3u);
    EXPECT_FALSE(g_releases.back().second);

    // Closing a poisoned lane reports the failure rather than swallowing it, so this is the
    // expected outcome of the close rather than an escaped error.
    EXPECT_THROW(lane.close(), std::runtime_error);
    worker.finalize();
}

TEST(ThreeRunCapacityLaneTest, AGrantOfTwoStillAdmitsOnlyTwo) {
    ChipWorker worker;
    prime_worker(worker, /*depth=*/2, /*launch_depth=*/2);
    ChipRunLane lane(worker);

    ChipRun first = submit_at(lane, 1, 0, CallConfig{}, /*activate=*/true);
    ChipRun second = submit_at(lane, 2, 1, CallConfig{}, /*activate=*/false);
    second.activate();
    // A slot the grant does not cover is not a slot this lane has.
    EXPECT_THROW(submit_at(lane, 3, 2, CallConfig{}, /*activate=*/false), std::runtime_error);

    EXPECT_EQ(g_events, (Events{"prepare0", "launch0", "prepare1", "joined1behind0"}));
    g_complete = {true, true, false};
    EXPECT_TRUE(first.done());
    EXPECT_TRUE(second.done());
    lane.close();
    worker.finalize();
}

TEST(ThreeRunCapacityLaneTest, DiagnosticsStayExclusiveToOneLaunchedRunAtCapacityThree) {
    ChipWorker worker;
    prime_worker(worker, /*depth=*/3, /*launch_depth=*/3);
    ChipRunLane lane(worker);

    CallConfig diagnostic;
    diagnostic.enable_chip_swimlane = 1;
    // A diagnostic config without an output prefix is refused at prepare, so the run would fail
    // for a reason that has nothing to do with the scope under test.
    diagnostic.output_prefix[0] = 'x';
    ASSERT_TRUE(diagnostic.diagnostics_any());

    ChipRun first = submit_at(lane, 1, 0, diagnostic, /*activate=*/true);
    ChipRun second = submit_at(lane, 2, 1, CallConfig{}, /*activate=*/false);
    ChipRun third = submit_at(lane, 3, 2, CallConfig{}, /*activate=*/false);
    second.activate();
    third.activate();

    // Preparation is allowed beside a diagnostic run; launching is not. Neither successor may be
    // joined behind it, so nothing but the front reaches the device.
    for (const std::string &event : g_events) {
        EXPECT_NE(event.rfind("joined", 0), 0U) << "a diagnostic run was joined: " << event;
    }
    EXPECT_FALSE(second.launched());
    EXPECT_FALSE(third.launched());

    // One completion at a time, each set after its own run is prepared: the fake's prepare clears
    // its slot's completion, as a real preparation means a run that has not run yet. The third is
    // prepared only once the second launches, which is only once the diagnostic front retires —
    // so pre-setting all three would arm a slot that preparation then disarms.
    g_complete[0] = true;
    EXPECT_TRUE(first.done());
    EXPECT_TRUE(second.launched()) << "the successor did not launch once the diagnostic run retired";
    EXPECT_FALSE(third.launched());

    g_complete[1] = true;
    EXPECT_TRUE(second.done());
    EXPECT_TRUE(third.launched()) << "the third run did not launch once the second retired";

    g_complete[2] = true;
    EXPECT_TRUE(third.done());

    // Over the whole lifetime, not just the window above: nothing was ever ordered behind the
    // diagnostic run, and the two plain runs were free to be joined to each other.
    for (const std::string &event : g_events) {
        EXPECT_EQ(event.find("behind0"), std::string::npos) << "a run was joined behind the diagnostic run: " << event;
    }
    lane.close();
    worker.finalize();
}

// ---------------------------------------------------------------------------
// A run staged before the run ahead of it launches
// ---------------------------------------------------------------------------

char *capacity_task_frame(std::array<char, MAILBOX_SIZE> &mailbox, size_t index) {
    return mailbox.data() + (MAILBOX_FIRST_TASK_FRAME + index) * MAILBOX_FRAME_SIZE;
}

MailboxState capacity_frame_state(const char *frame) {
    const auto *state = reinterpret_cast<const int32_t *>(frame + MAILBOX_OFF_STATE);
    return static_cast<MailboxState>(__atomic_load_n(state, __ATOMIC_ACQUIRE));
}

void set_capacity_frame_state(char *frame, MailboxState state) {
    auto *wire_state = reinterpret_cast<int32_t *>(frame + MAILBOX_OFF_STATE);
    __atomic_store_n(wire_state, static_cast<int32_t>(state), __ATOMIC_RELEASE);
}

void set_capacity_frame_disposition(char *frame, MailboxPreparationDisposition disposition) {
    const auto value = static_cast<int32_t>(disposition);
    auto *wire = reinterpret_cast<int32_t *>(frame + MAILBOX_OFF_PREPARATION_DISPOSITION);
    __atomic_store_n(wire, value, __ATOMIC_RELEASE);
}

TaskSlot capacity_progress_slot(Ring &ring, RunId run_id, uint32_t pipeline_slot, uint64_t generation) {
    AllocResult allocation = ring.alloc(/*heap_bytes=*/0, /*depth=*/0);
    if (allocation.slot == INVALID_SLOT) return INVALID_SLOT;
    TaskSlotState *slot = ring.slot_state(allocation.slot);
    if (slot == nullptr) return INVALID_SLOT;
    slot->reset();
    slot->run_id = run_id;
    slot->pipeline_lease = PipelineSlotLease{pipeline_slot, 0, generation};
    return allocation.slot;
}

// The interleaving the scheduler is allowed to produce: C is staged while B is only prepared, so
// its own preparation cannot happen yet. It must stay eligible rather than fall back for good, and
// the preparation it gets once B launches must be the one the parent sees.
TEST(ThreeRunCapacityLaneTest, ARunStagedBeforeItsPredecessorLaunchesIsPreparedWhenItDoes) {
    ChipWorker worker;
    prime_worker(worker, /*depth=*/3, /*launch_depth=*/3);
    ChipRunLane lane(worker);

    ChipRun first = submit_at(lane, 1, 0, CallConfig{}, /*activate=*/true);
    ChipRun second = submit_at(lane, 2, 1, CallConfig{}, /*activate=*/false);
    ChipRun third = submit_at(lane, 3, 2, CallConfig{}, /*activate=*/false);

    // C arrived behind a predecessor that has not reached the device, so it holds no native
    // preparation — and no permanent fallback either: nothing about its own shape refused it.
    EXPECT_EQ(third.preparation_disposition(), ChipRunPreparationDisposition::VALIDATED_ONLY);
    EXPECT_EQ(second.preparation_disposition(), ChipRunPreparationDisposition::NATIVE_PREPARED);
    EXPECT_EQ(g_events, (Events{"prepare0", "launch0", "prepare1"}));

    // B launching is the event that makes C preparable, and the lane takes it then.
    second.activate();
    EXPECT_EQ(third.preparation_disposition(), ChipRunPreparationDisposition::NATIVE_PREPARED);
    EXPECT_EQ(g_events, (Events{"prepare0", "launch0", "prepare1", "joined1behind0", "prepare2"}));

    third.activate();
    EXPECT_TRUE(third.launched());
    EXPECT_EQ(g_events, (Events{"prepare0", "launch0", "prepare1", "joined1behind0", "prepare2", "joined2behind1"}));

    g_complete = {true, true, true};
    EXPECT_TRUE(first.done());
    EXPECT_TRUE(second.done());
    EXPECT_TRUE(third.done());
    lane.close();
    worker.finalize();
}

// The other half of that interleaving, at the endpoint: a staged frame reported validated-only and
// then promoted must reach the parent a second time, or the authorization it earned is never given.
TEST(ThreeRunCapacityEndpointTest, ALaterNativePreparationIsReportedOnceMore) {
    alignas(8) std::array<char, MAILBOX_SIZE> mailbox{};
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    const TaskSlot active_slot = capacity_progress_slot(allocator, /*run_id=*/71, /*pipeline_slot=*/0, 5);
    const TaskSlot staged_slot = capacity_progress_slot(allocator, /*run_id=*/72, /*pipeline_slot=*/1, 6);
    ASSERT_NE(active_slot, INVALID_SLOT);
    ASSERT_NE(staged_slot, INVALID_SLOT);

    LocalMailboxEndpoint endpoint(/*worker_id=*/0, mailbox.data(), /*child_pid=*/-1, /*task_frame_count=*/3);
    endpoint.submit_progress(&allocator, WorkerDispatch{active_slot, 0, /*dispatch_id=*/81, /*prepare_only=*/false});
    endpoint.submit_progress(&allocator, WorkerDispatch{staged_slot, 0, /*dispatch_id=*/82, /*prepare_only=*/true});

    char *staged_frame = capacity_task_frame(mailbox, 1);
    ASSERT_EQ(capacity_frame_state(staged_frame), MailboxState::PREPARE_READY);

    // Staged behind a predecessor that has not launched: the child can only say it validated the
    // frame.
    set_capacity_frame_disposition(staged_frame, MailboxPreparationDisposition::VALIDATED_ONLY);
    set_capacity_frame_state(staged_frame, MailboxState::FRAME_STAGED);
    WorkerEndpointProgress progress;
    ASSERT_TRUE(endpoint.poll_progress(progress));
    EXPECT_EQ(progress.kind, WorkerProgressKind::FRAME_STAGED);
    EXPECT_EQ(progress.dispatch.dispatch_id, 82u);
    EXPECT_EQ(progress.preparation_disposition, MailboxPreparationDisposition::VALIDATED_ONLY);

    // Nothing has changed yet, so nothing is reported again.
    EXPECT_FALSE(endpoint.poll_progress(progress));

    // The child prepares it natively once its predecessor reaches the device and republishes into
    // the same frame, under the same identity.
    set_capacity_frame_disposition(staged_frame, MailboxPreparationDisposition::NATIVE_PREPARED);
    set_capacity_frame_state(staged_frame, MailboxState::FRAME_STAGED);
    ASSERT_TRUE(endpoint.poll_progress(progress));
    EXPECT_EQ(progress.kind, WorkerProgressKind::FRAME_STAGED);
    EXPECT_EQ(progress.dispatch.dispatch_id, 82u);
    EXPECT_EQ(progress.preparation_disposition, MailboxPreparationDisposition::NATIVE_PREPARED);

    // Once, not on every poll: a promotion is a change, and the report follows the change.
    EXPECT_FALSE(endpoint.poll_progress(progress));
    EXPECT_EQ(capacity_frame_state(staged_frame), MailboxState::FRAME_STAGED);
    allocator.shutdown();
}

// An activation the parent has already published is the parent's word on that frame, and a
// promotion must not be read as a reason to revisit it.
TEST(ThreeRunCapacityEndpointTest, APromotionDoesNotDisturbAnActivationAlreadyPublished) {
    alignas(8) std::array<char, MAILBOX_SIZE> mailbox{};
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    const TaskSlot staged_slot = capacity_progress_slot(allocator, /*run_id=*/73, /*pipeline_slot=*/0, 7);
    ASSERT_NE(staged_slot, INVALID_SLOT);

    LocalMailboxEndpoint endpoint(/*worker_id=*/0, mailbox.data(), /*child_pid=*/-1, /*task_frame_count=*/3);
    endpoint.submit_progress(&allocator, WorkerDispatch{staged_slot, 0, /*dispatch_id=*/83, /*prepare_only=*/true});
    char *staged_frame = capacity_task_frame(mailbox, 0);

    set_capacity_frame_disposition(staged_frame, MailboxPreparationDisposition::NATIVE_PREPARED);
    set_capacity_frame_state(staged_frame, MailboxState::FRAME_STAGED);
    WorkerEndpointProgress progress;
    ASSERT_TRUE(endpoint.poll_progress(progress));
    EXPECT_EQ(progress.preparation_disposition, MailboxPreparationDisposition::NATIVE_PREPARED);

    // The parent authorizes it, which writes ACTIVATE into the frame.
    EXPECT_TRUE(endpoint.activate_progress(/*run_id=*/73));
    EXPECT_EQ(capacity_frame_state(staged_frame), MailboxState::ACTIVATE);

    // A republished disposition on an activated frame changes nothing: the state is the child's to
    // move from here, and the endpoint neither reports again nor rewrites it.
    set_capacity_frame_disposition(staged_frame, MailboxPreparationDisposition::NATIVE_PREPARED);
    EXPECT_FALSE(endpoint.poll_progress(progress));
    EXPECT_EQ(capacity_frame_state(staged_frame), MailboxState::ACTIVATE);
    allocator.shutdown();
}

// ---------------------------------------------------------------------------
// Orchestrator: leasing and authorizing more than one successor
// ---------------------------------------------------------------------------

struct CapacityHarness {
    TensorMap tm;
    Ring allocator;
    Scope scope;
    NextLevelReadyQueues rq_next_level;
    ReadyQueue rq_sub;
    Orchestrator orch;

    CapacityHarness(uint32_t depth, uint32_t launch_depth) {
        allocator.init(/*heap_bytes=*/1ULL << 20);
        rq_next_level.reset({0});
        orch.init(&tm, &allocator, &scope, &rq_sub, &rq_next_level, nullptr, [] {});
        orch.configure_pipeline_depth(depth, depth, launch_depth);
    }

    ~CapacityHarness() { allocator.shutdown(); }

    SubmitResult submit_one(uint64_t buffer_id) {
        CallConfig cfg;
        TaskArgs args;
        Tensor t{};
        t.buffer.backend_kind = static_cast<uint8_t>(BackendKind::POSIX_SHM);
        t.buffer.access = static_cast<uint8_t>(AccessMode::READWRITE);
        t.buffer.nbytes = 1;
        t.buffer.identity.buffer_id = buffer_id;
        t.ndims = 1;
        t.shapes[0] = 1;
        t.strides[0] = 1;
        t.dtype = DataType::UINT8;
        args.add_tensor(t, TensorArgType::OUTPUT);
        CallableIdentity callable;
        callable.digest.fill(53);
        return orch.submit_next_level(callable, args, cfg, 0);
    }

    struct Run {
        RunId id{INVALID_RUN_ID};
        SubmitResult task{};
    };

    Run build_closed_run(uint64_t buffer_id) {
        Run run;
        run.id = orch.begin_run();
        run.task = submit_one(buffer_id);
        orch.close_run_submission(run.id);
        return run;
    }

    void accept(const Run &run) { orch.mark_task_accepted(run.task.task_slot); }

    /**
     * Drive a run's only task to completion so the run retires and gives its FIFO entry back.
     *
     * Reports rather than asserts: its callers run in a child process, where a gtest failure
     * reaches nobody and only the exit status is carried back.
     */
    bool complete(const Run &run) {
        allocator.slot_state(run.task.task_slot)->state.store(TaskState::COMPLETED, std::memory_order_release);
        return orch.on_consumed(run.task.task_slot);
    }
};

/** Poll `ready` until it holds, or report the deadline as the failure it is. */
bool settled_within(const std::function<bool()> &ready, double seconds = 5.0) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (std::chrono::steady_clock::now() < deadline) {
        if (ready()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

/** What an admission scenario concluded, carried back through a child's exit status. */
enum class AdmissionOutcome : int {
    OK = 0,
    HEAD_NOT_ACTIVE = 11,
    SUBMISSION_RAISED = 12,
    NEVER_PARKED = 13,
    ADMITTED_WHILE_HELD = 14,
    NOT_RELEASED_BY_RETIREMENT = 15,
    WAITER_LEFT_BEHIND = 16,
    ADMITTED_AN_EXISTING_ID = 17,
    RETIREMENT_REFUSED = 18,
    SCENARIO_THREW = 19,
};

int outcome_code(AdmissionOutcome outcome) { return static_cast<int>(outcome); }

const char *admission_outcome_text(int code) {
    switch (code) {
    case 0:
        return "the scenario held";
    case 11:
        return "the head run never became active, so nothing held the first set";
    case 12:
        return "the parked submission failed instead of waiting";
    case 13:
        return "no submission was ever counted in admission, so the bound was never met";
    case 14:
        return "a further run was admitted while the granted sets were all held";
    case 15:
        return "retiring one run did not release the parked submission";
    case 16:
        return "a waiter was left in admission after the submission returned";
    case 17:
        return "admission handed back the id of a run that already held a set";
    case 18:
        return "a run would not retire, so the release this case tests never had its cause";
    case 19:
        return "the scenario raised before it could conclude";
    case -1:
        return "the child could not be started or collected";
    case -2:
        return "the child did not exit: either its alarm ended a submission that never returned, or it aborted";
    default:
        return "an unrecognized outcome";
    }
}

/**
 * Run one admission scenario in a child process, and report what it concluded.
 *
 * The parked caller is why this is isolated. `begin_run` waits on a condition variable with no
 * deadline of its own, so if the retirement that must release it stops doing so, the thread waits
 * forever and no join in the scenario can return. Here that leaves the *child* parked, `alarm`
 * ends it, and this process reports a bounded failure — without detaching a thread, without any
 * reference outliving its owner, and without a production knob that exists only for a test.
 */
int isolated_admission_outcome(const std::function<int()> &scenario, unsigned alarm_seconds = 30) {
    const pid_t child = fork();
    if (child < 0) return -1;
    if (child == 0) {
        alarm(alarm_seconds);
        int code = outcome_code(AdmissionOutcome::SCENARIO_THREW);
        try {
            code = scenario();
        } catch (...) {
            // Deliberate: an escaping exception here would abort the child, and the status that
            // produced would be indistinguishable from the alarm this case exists to detect.
        }
        _exit(code);
    }
    int status = 0;
    pid_t collected = 0;
    do {
        collected = waitpid(child, &status, 0);
    } while (collected < 0 && errno == EINTR);
    if (collected != child) return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -2;
}

/**
 * A fourth caller meets the capacity bound, and one retirement is what releases it.
 *
 * Nothing is concluded before the submitted thread has been released and joined, so the scenario
 * always has a thread to join and never leaves one behind.
 */
int parks_until_one_of_three_retires() {
    CapacityHarness h(/*depth=*/3, /*launch_depth=*/3);
    const auto first = h.build_closed_run(1);
    const auto second = h.build_closed_run(2);
    const auto third = h.build_closed_run(3);
    h.accept(first);
    if (h.orch.active_run_id() != first.id) return outcome_code(AdmissionOutcome::HEAD_NOT_ACTIVE);

    std::atomic<RunId> fourth{INVALID_RUN_ID};
    std::atomic<bool> raised{false};
    std::thread submitter([&] {
        try {
            fourth.store(h.orch.begin_run(), std::memory_order_release);
        } catch (...) {
            raised.store(true, std::memory_order_release);
        }
    });

    // A caller is counted only once the FIFO bound has stopped it, so being counted is having met
    // backpressure. Having started a thread would be evidence of nothing.
    const bool parked = settled_within([&] {
        return h.orch.begin_run_waiter_count_for_test() == 1;
    });
    const RunId while_three_held = fourth.load(std::memory_order_acquire);

    // The retirement is the release, and it is the only thing that happens between that reading
    // and the one below.
    const bool retired = h.complete(first);
    submitter.join();

    if (raised.load(std::memory_order_acquire)) return outcome_code(AdmissionOutcome::SUBMISSION_RAISED);
    if (!parked) return outcome_code(AdmissionOutcome::NEVER_PARKED);
    if (while_three_held != INVALID_RUN_ID) return outcome_code(AdmissionOutcome::ADMITTED_WHILE_HELD);
    if (!retired) return outcome_code(AdmissionOutcome::RETIREMENT_REFUSED);
    if (h.orch.begin_run_waiter_count_for_test() != 0) return outcome_code(AdmissionOutcome::WAITER_LEFT_BEHIND);
    const RunId admitted = fourth.load(std::memory_order_acquire);
    if (admitted == INVALID_RUN_ID) return outcome_code(AdmissionOutcome::NOT_RELEASED_BY_RETIREMENT);
    if (admitted == first.id || admitted == second.id || admitted == third.id) {
        return outcome_code(AdmissionOutcome::ADMITTED_AN_EXISTING_ID);
    }
    return outcome_code(AdmissionOutcome::OK);
}

/** The same boundary one set lower, where the run this admits third is the one that waits. */
int parks_until_one_of_two_retires() {
    CapacityHarness h(/*depth=*/2, /*launch_depth=*/2);
    const auto first = h.build_closed_run(1);
    const auto second = h.build_closed_run(2);
    h.accept(first);
    if (h.orch.active_run_id() != first.id) return outcome_code(AdmissionOutcome::HEAD_NOT_ACTIVE);

    std::atomic<RunId> third{INVALID_RUN_ID};
    std::atomic<bool> raised{false};
    std::thread submitter([&] {
        try {
            third.store(h.orch.begin_run(), std::memory_order_release);
        } catch (...) {
            raised.store(true, std::memory_order_release);
        }
    });

    const bool parked = settled_within([&] {
        return h.orch.begin_run_waiter_count_for_test() == 1;
    });
    const RunId while_two_held = third.load(std::memory_order_acquire);

    const bool retired = h.complete(first);
    submitter.join();

    if (raised.load(std::memory_order_acquire)) return outcome_code(AdmissionOutcome::SUBMISSION_RAISED);
    if (!parked) return outcome_code(AdmissionOutcome::NEVER_PARKED);
    if (while_two_held != INVALID_RUN_ID) return outcome_code(AdmissionOutcome::ADMITTED_WHILE_HELD);
    if (!retired) return outcome_code(AdmissionOutcome::RETIREMENT_REFUSED);
    const RunId admitted = third.load(std::memory_order_acquire);
    if (admitted == INVALID_RUN_ID) return outcome_code(AdmissionOutcome::NOT_RELEASED_BY_RETIREMENT);
    if (admitted == first.id || admitted == second.id) {
        return outcome_code(AdmissionOutcome::ADMITTED_AN_EXISTING_ID);
    }
    return outcome_code(AdmissionOutcome::OK);
}

// The causal half of the capacity claim, which no observation of running hardware can give: a
// fourth caller reaching admission while three runs hold the three granted sets is parked there,
// and the retirement of one of those runs is what releases it. Everything is driven synchronously,
// so the ordering is established rather than inferred from timing — the only wait is on the
// orchestrator's own count of parked callers, and exhausting it is a failure.
TEST(ThreeRunCapacityOrchestratorTest, AFourthRunParksInAdmissionUntilOneOfTheThreeRetires) {
    const int outcome = isolated_admission_outcome(parks_until_one_of_three_retires);
    EXPECT_EQ(outcome, 0) << admission_outcome_text(outcome);
}

// The park follows the granted capacity rather than a constant.
TEST(ThreeRunCapacityOrchestratorTest, AGrantOfTwoParksTheThirdRunInAdmission) {
    const int outcome = isolated_admission_outcome(parks_until_one_of_two_retires);
    EXPECT_EQ(outcome, 0) << admission_outcome_text(outcome);
}

TEST(ThreeRunCapacityOrchestratorTest, BothSuccessorsArePreparableAtCapacityThree) {
    CapacityHarness h(/*depth=*/3, /*launch_depth=*/3);
    const auto first = h.build_closed_run(1);
    const auto second = h.build_closed_run(2);
    const auto third = h.build_closed_run(3);
    h.accept(first);

    ASSERT_EQ(h.orch.active_run_id(), first.id);
    // The single-run query keeps answering with the first successor; the list is what lets the
    // scheduler stage the one behind it in the same round.
    EXPECT_EQ(h.orch.preparable_run_id(), second.id);
    EXPECT_EQ(h.orch.preparable_run_ids(), (std::vector<RunId>{second.id, third.id}));
}

TEST(ThreeRunCapacityOrchestratorTest, AuthorizationReachesTheThirdRunOnlyOnceTheSecondIsAccepted) {
    CapacityHarness h(/*depth=*/3, /*launch_depth=*/3);
    const auto first = h.build_closed_run(1);
    const auto second = h.build_closed_run(2);
    const auto third = h.build_closed_run(3);

    h.accept(first);
    // The second is authorized on the head's acceptance; the third is not, because the run it
    // would be ordered behind still has a dispatch that has not reached an outcome.
    EXPECT_EQ(h.orch.early_launch_run_ids(), (std::vector<RunId>{second.id}));

    h.accept(second);
    EXPECT_EQ(h.orch.early_launch_run_ids(), (std::vector<RunId>{second.id, third.id}));
}

TEST(ThreeRunCapacityOrchestratorTest, ALaunchDepthOfTwoAuthorizesOneSuccessorOutOfTwoPreparable) {
    CapacityHarness h(/*depth=*/3, /*launch_depth=*/2);
    const auto first = h.build_closed_run(1);
    const auto second = h.build_closed_run(2);
    const auto third = h.build_closed_run(3);
    h.accept(first);
    h.accept(second);

    // Capacity and launch depth are separate budgets: three runs hold resources, two reach the
    // device.
    EXPECT_EQ(h.orch.preparable_run_ids(), (std::vector<RunId>{second.id, third.id}));
    EXPECT_EQ(h.orch.early_launch_run_ids(), (std::vector<RunId>{second.id}));
}

TEST(ThreeRunCapacityOrchestratorTest, ADepthOfTwoStillLeasesAndAuthorizesOneSuccessor) {
    CapacityHarness h(/*depth=*/2, /*launch_depth=*/2);
    const auto first = h.build_closed_run(1);
    const auto second = h.build_closed_run(2);
    h.accept(first);

    EXPECT_EQ(h.orch.preparable_run_ids(), (std::vector<RunId>{second.id}));
    EXPECT_EQ(h.orch.early_launch_run_ids(), (std::vector<RunId>{second.id}));
    EXPECT_EQ(h.orch.early_launch_run_id(), second.id);
}

}  // namespace
