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

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "call_config.h"
#include "worker_manager.h"
#include "run_resource_arena.h"
#include "ring.h"
#include "runtime_c_api.h"
#include "scope.h"
#include "task_args.h"
#include "tensor.h"
#include "tensormap.h"
#include "types.h"

// The activation hook holds runs_mu_; the locking public capacity getter
// cannot be called from it. Only the existing protected value is read there.
#define private public
#include "orchestrator.h"
#undef private
#include "worker.h"

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
    g_slots[runtime] = descriptor->pipeline_slot;
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

int wait_run_completion(void *, void *runtime) {
    g_complete[slot_of(runtime)] = true;
    return 0;
}

int finalize_run(void *, void *runtime, int) {
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
    // One generation per granted set, as `init` sizes it: an unsized filter has no entry for
    // slot 0, and `is_admissible` refuses a lease outside it, so every preparation would be
    // refused as stale before reaching the stubs below.
    worker.pipeline_generations_.resize(depth);
    for (uint32_t slot = 0; slot < depth; ++slot) {
        worker.runtime_bufs_.emplace_back(64, alignof(std::max_align_t), nullptr);
    }
    worker.prepare_run_fn_ = prepare_run;
    worker.launch_run_fn_ = launch_run;
    worker.launch_run_joined_fn_ = launch_run_joined;
    worker.poll_run_fn_ = poll_run;
    worker.wait_run_fn_ = wait_run;
    worker.wait_run_completion_fn_ = wait_run_completion;
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

TEST(ThreeRunCapacityLaneTest, TheUnobservedCompletedBacklogIsCountedAndItsPeakRetained) {
    // The backlog the pipeline depth deliberately does not bound: each run's
    // execution has ended and its slot is refillable, while its result is still
    // the caller's to take. The count is what makes that retention visible, and
    // the peak survives the drain that empties the count.
    ChipWorker worker;
    prime_worker(worker, /*depth=*/3, /*launch_depth=*/3);
    ChipRunLane lane(worker);

    EXPECT_EQ(worker.unobserved_completed_run_count(), 0u);
    EXPECT_EQ(worker.unobserved_completed_run_high_water(), 0u);

    ChipRun first = submit_at(lane, 1, 0, CallConfig{}, /*activate=*/true);
    ChipRun second = submit_at(lane, 2, 1, CallConfig{}, /*activate=*/false);
    second.activate();
    ChipRun third = submit_at(lane, 3, 2, CallConfig{}, /*activate=*/false);
    third.activate();
    // Launched, not ended: execution still owns all three.
    EXPECT_EQ(worker.unobserved_completed_run_count(), 0u);

    g_complete = {true, true, true};
    EXPECT_TRUE(first.done());
    EXPECT_TRUE(second.done());
    EXPECT_TRUE(third.done());
    // Ended and unobserved: three results the callers have not taken.
    EXPECT_EQ(worker.unobserved_completed_run_count(), 3u);
    EXPECT_EQ(worker.unobserved_completed_run_high_water(), 3u);

    // Past the granted depth, which is the property being claimed: the first
    // run's slot refilled while its result is still untaken, so a backlog
    // capped at the pipeline depth would stop here. Submitting resets the
    // slot's completion flag, so this run needs its own.
    ChipRun fourth = submit_at(lane, 4, 0, CallConfig{}, /*activate=*/true);
    g_complete[0] = true;
    EXPECT_TRUE(fourth.done());
    EXPECT_EQ(worker.unobserved_completed_run_count(), 4u);
    EXPECT_EQ(worker.unobserved_completed_run_high_water(), 4u);

    lane.close();
    // Close delivers every accepted run, so the backlog empties while the peak
    // stays readable — which is what a test can assert after the race is over.
    EXPECT_EQ(worker.unobserved_completed_run_count(), 0u);
    EXPECT_EQ(worker.unobserved_completed_run_high_water(), 4u);
    worker.finalize();
}

TEST(ThreeRunCapacityLaneTest, AMiddleRunsResultFailurePreservesConfirmedResultsAndProvedLastUse) {
    // Three live runs separate confirmed results from later unobserved outcomes.
    // The native CPU leaf injects the result failure; each run still needs its
    // own physical last-use proof and cleanup.
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
    // Confirm the predecessor's result before observing the middle failure.
    // done() alone proves physical completion, not successful finalization.
    EXPECT_TRUE(first.done());
    EXPECT_TRUE(first.wait_until(ChipRun::Clock::now() + std::chrono::seconds(5)));
    ASSERT_EQ(g_releases.size(), 1u);
    EXPECT_FALSE(g_releases.front().second);

    // The failing run carries its own error, and nothing the predecessor did is restated. Bounded:
    // `wait_until` reports a deadline it did not meet as `false`, so a run that stopped reaching an
    // outcome fails this assertion instead of parking the suite.
    const ChipRun::Deadline deadline = ChipRun::Clock::now() + std::chrono::seconds(5);
    EXPECT_THROW(second.wait_until(deadline), std::runtime_error);
    ASSERT_EQ(g_releases.size(), 2u);
    // Its own boundary already proved last use before the result failure.
    EXPECT_FALSE(g_releases.back().second);
    EXPECT_NE(g_releases.back().first, g_releases.front().first);

    // The unobserved successor inherits the known lane failure. Its own drain
    // still proves last use and discharges its references.
    EXPECT_TRUE(second.lane_poisoned());
    try {
        (void)third.wait_until(ChipRun::Clock::now() + std::chrono::seconds(5));
        FAIL() << "unobserved successor lost the middle run's lane failure";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(std::string(error.what()).find("chip run lane is poisoned"), std::string::npos);
        EXPECT_NE(std::string(error.what()).find("finalize_native_run failed with code -1000"), std::string::npos);
        EXPECT_NE(std::string(error.what()).find("run_id=2"), std::string::npos);
    }
    ASSERT_EQ(g_releases.size(), 3u);
    EXPECT_FALSE(g_releases.back().second);
    const auto settled_events = g_events;
    EXPECT_TRUE(first.wait_until(ChipRun::Clock::now() + std::chrono::seconds(5)));
    EXPECT_EQ(g_events, settled_events) << "confirmed predecessor must remain cached";

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

// A run's state lives in its own entry, allocated in submission order, rather
// than in the request frame that delivered its payload.
MailboxRunEntry *capacity_entry(std::array<char, MAILBOX_SIZE> &mailbox, size_t index) {
    return mailbox_run_entry(mailbox.data(), index);
}

MailboxState capacity_entry_state(const MailboxRunEntry *entry) {
    return static_cast<MailboxState>(__atomic_load_n(&entry->state, __ATOMIC_ACQUIRE));
}

void set_capacity_entry_state(MailboxRunEntry *entry, MailboxState state) {
    __atomic_store_n(&entry->state, static_cast<int32_t>(state), __ATOMIC_RELEASE);
}

void set_capacity_entry_disposition(MailboxRunEntry *entry, MailboxPreparationDisposition disposition) {
    __atomic_store_n(&entry->preparation_disposition, static_cast<int32_t>(disposition), __ATOMIC_RELEASE);
}

/** The child's hand-off: echo the frame's generation, then stage the entry. */
void capacity_consume_frame(
    std::array<char, MAILBOX_SIZE> &mailbox, size_t frame_index, MailboxPreparationDisposition disposition
) {
    char *frame = capacity_task_frame(mailbox, frame_index);
    uint64_t entry_index = 0;
    uint64_t frame_generation = 0;
    std::memcpy(&entry_index, frame + MAILBOX_OFF_FRAME_ENTRY_INDEX, sizeof(entry_index));
    std::memcpy(&frame_generation, frame + MAILBOX_OFF_FRAME_GENERATION, sizeof(frame_generation));
    MailboxRunEntry *entry = capacity_entry(mailbox, static_cast<size_t>(entry_index));
    set_capacity_entry_disposition(entry, disposition);
    __atomic_store_n(&entry->frame_consumed_generation, frame_generation, __ATOMIC_RELEASE);
    set_capacity_entry_state(entry, MailboxState::FRAME_STAGED);
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

    LocalMailboxEndpoint endpoint(/*worker_id=*/0, mailbox.data(), mailbox.size(), /*child_pid=*/-1,
                                  /*granted_sets=*/3);
    endpoint.submit_progress(&allocator, WorkerDispatch{active_slot, 0, /*dispatch_id=*/81, /*prepare_only=*/false});
    endpoint.submit_progress(&allocator, WorkerDispatch{staged_slot, 0, /*dispatch_id=*/82, /*prepare_only=*/true});

    // Entry 1 is the second submission; its payload went out on request frame 1.
    MailboxRunEntry *staged_entry = capacity_entry(mailbox, 1);
    ASSERT_EQ(capacity_entry_state(staged_entry), MailboxState::IDLE);

    // Staged behind a predecessor that has not launched: the child can only say it validated the
    // payload it just took.
    capacity_consume_frame(mailbox, 1, MailboxPreparationDisposition::VALIDATED_ONLY);
    WorkerEndpointProgress progress;
    ASSERT_TRUE(endpoint.poll_progress(progress));
    EXPECT_EQ(progress.kind, WorkerProgressKind::FRAME_STAGED);
    EXPECT_EQ(progress.dispatch.dispatch_id, 82u);
    EXPECT_EQ(progress.preparation_disposition, MailboxPreparationDisposition::VALIDATED_ONLY);

    // Nothing has changed yet, so nothing is reported again.
    EXPECT_FALSE(endpoint.poll_progress(progress));

    // The child prepares it natively once its predecessor reaches the device and republishes into
    // the same entry, under the same identity. Only the disposition word moves: the state word is
    // the parent's exchange, and the frame that carried the payload is long since back.
    set_capacity_entry_disposition(staged_entry, MailboxPreparationDisposition::NATIVE_PREPARED);
    set_capacity_entry_state(staged_entry, MailboxState::FRAME_STAGED);
    ASSERT_TRUE(endpoint.poll_progress(progress));
    EXPECT_EQ(progress.kind, WorkerProgressKind::FRAME_STAGED);
    EXPECT_EQ(progress.dispatch.dispatch_id, 82u);
    EXPECT_EQ(progress.preparation_disposition, MailboxPreparationDisposition::NATIVE_PREPARED);

    // Once, not on every poll: a promotion is a change, and the report follows the change.
    EXPECT_FALSE(endpoint.poll_progress(progress));
    EXPECT_EQ(capacity_entry_state(staged_entry), MailboxState::FRAME_STAGED);
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

    LocalMailboxEndpoint endpoint(/*worker_id=*/0, mailbox.data(), mailbox.size(), /*child_pid=*/-1,
                                  /*granted_sets=*/3);
    endpoint.submit_progress(&allocator, WorkerDispatch{staged_slot, 0, /*dispatch_id=*/83, /*prepare_only=*/true});
    MailboxRunEntry *staged_entry = capacity_entry(mailbox, 0);

    capacity_consume_frame(mailbox, 0, MailboxPreparationDisposition::NATIVE_PREPARED);
    WorkerEndpointProgress progress;
    ASSERT_TRUE(endpoint.poll_progress(progress));
    EXPECT_EQ(progress.preparation_disposition, MailboxPreparationDisposition::NATIVE_PREPARED);

    // The parent authorizes it, which writes ACTIVATE into the frame.
    EXPECT_TRUE(endpoint.activate_progress(/*run_id=*/73));
    EXPECT_EQ(capacity_entry_state(staged_entry), MailboxState::ACTIVATE);

    // A republished disposition on an activated frame changes nothing: the state is the child's to
    // move from here, and the endpoint neither reports again nor rewrites it.
    set_capacity_entry_disposition(staged_entry, MailboxPreparationDisposition::NATIVE_PREPARED);
    EXPECT_FALSE(endpoint.poll_progress(progress));
    EXPECT_EQ(capacity_entry_state(staged_entry), MailboxState::ACTIVATE);
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

TEST(ThreeRunCapacityEndpointTest, PhysicalEndRetainsTheEntryUntilStrongTerminalDelivery) {
    alignas(8) std::array<char, MAILBOX_SIZE> mailbox{};
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    const TaskSlot first = capacity_progress_slot(allocator, 171, 0, 15);
    const TaskSlot second = capacity_progress_slot(allocator, 172, 0, 16);
    LocalMailboxEndpoint endpoint(0, mailbox.data(), mailbox.size(), -1, 3);
    endpoint.submit_progress(&allocator, WorkerDispatch{first, 0, 181, false});
    capacity_consume_frame(mailbox, 0, MailboxPreparationDisposition::NATIVE_PREPARED);
    auto *entry = capacity_entry(mailbox, 0);
    const auto old_generation = entry->entry_generation;
    __atomic_store_n(&entry->accepted, MAILBOX_TASK_ACCEPTED, __ATOMIC_RELEASE);
    __atomic_store_n(&entry->admissible_run_resource_sets, 3, __ATOMIC_RELEASE);
    set_capacity_entry_state(entry, MailboxState::TASK_ENDED);
    WorkerEndpointProgress progress;
    ASSERT_TRUE(endpoint.poll_progress(progress));
    EXPECT_EQ(progress.kind, WorkerProgressKind::ACCEPTED);
    ASSERT_TRUE(endpoint.poll_progress(progress));
    EXPECT_EQ(progress.kind, WorkerProgressKind::ENDED);
    EXPECT_EQ(progress.dispatch.dispatch_id, 181u);
    EXPECT_EQ(progress.completion.task_slot, first);
    EXPECT_EQ(progress.completion.group_index, 0);
    EXPECT_EQ(progress.completion.worker_id, 0);
    EXPECT_EQ(progress.completion.admissible_run_resource_sets, 3u);
    EXPECT_FALSE(endpoint.poll_progress(progress));
    EXPECT_EQ(entry->entry_generation, old_generation);
    endpoint.request_observation(171);
    EXPECT_EQ(__atomic_load_n(&entry->observation_requested, __ATOMIC_ACQUIRE), 1);
    EXPECT_EQ(capacity_entry_state(entry), MailboxState::TASK_ENDED);
    EXPECT_EQ(entry->accepted, MAILBOX_TASK_ACCEPTED);
    set_capacity_entry_state(entry, MailboxState::TASK_DONE);
    ASSERT_TRUE(endpoint.poll_progress(progress));
    EXPECT_EQ(progress.kind, WorkerProgressKind::COMPLETED);
    EXPECT_EQ(entry->entry_generation % 2, 0u);
    endpoint.submit_progress(&allocator, WorkerDispatch{second, 0, 182, false});
    ASSERT_NE(entry->entry_generation, old_generation);
    EXPECT_EQ(entry->observation_requested, 0);
    endpoint.request_observation(171);  // Old run identity cannot request its successor.
    EXPECT_EQ(entry->observation_requested, 0);
    allocator.shutdown();
}

TEST(ThreeRunCapacityEndpointTest, AWaitBeforePublicationPersistsWithoutPeriodicRequests) {
    class CountingEndpoint : public LocalMailboxEndpoint {
    public:
        CountingEndpoint(void *mailbox, size_t bytes) :
            LocalMailboxEndpoint(0, mailbox, bytes, -1, 1) {}
        void request_observation(RunId id) override {
            ++requests;
            LocalMailboxEndpoint::request_observation(id);
        }
        size_t requests{0};
    };
    alignas(8) std::array<char, MAILBOX_SIZE> mailbox{};
    Ring allocator;
    allocator.init(/*heap_bytes=*/1ULL << 20);
    TensorMap tm;
    Scope scope;
    ReadyQueue sub;
    NextLevelReadyQueues next;
    next.reset({0});
    WorkerManager manager;
    auto endpoint = std::make_unique<CountingEndpoint>(mailbox.data(), mailbox.size());
    auto *const counter = endpoint.get();
    manager.add_next_level_endpoint(std::move(endpoint));
    Orchestrator orch;
    manager.start(
        &allocator, [](WorkerCompletion) {}, [](WorkerDispatch) {}, {}, {},
        [&orch](RunId id) noexcept {
            return orch.run_observation_requested(id);
        }
    );
    orch.init(&tm, &allocator, &scope, &sub, &next, &manager);
    const RunId run = orch.begin_run();
    CallableIdentity callable;
    callable.digest.fill(192);
    const auto submitted = orch.submit_next_level(callable, TaskArgs{}, CallConfig{}, 0);
    orch.close_run_submission(run);

    // No entry exists yet. The caller deadline ends this wait, not its demand
    // for a strong result once the actual dispatch is published.
    EXPECT_FALSE(orch.wait_run_for(run, 0));
    EXPECT_EQ(counter->requests, 0u);
    EXPECT_FALSE(orch.wait_run_for(run, 0.01));
    EXPECT_EQ(counter->requests, 1u) << "one persistent demand must replace timer-driven retries";
    EXPECT_FALSE(orch.wait_run_for(run, 0.0001));
    EXPECT_EQ(counter->requests, 1u) << "a later waiter shares the run's existing demand";
    auto *const thread = manager.get_worker_by_id(WorkerType::NEXT_LEVEL, 0);
    ASSERT_NE(thread, nullptr);
    allocator.slot_state(submitted.task_slot)->state.store(TaskState::RUNNING);
    thread->dispatch(WorkerDispatch{submitted.task_slot, 0, 0, false});
    auto *const entry = capacity_entry(mailbox, 0);
    EXPECT_EQ(entry->observation_requested, 1) << "the wait preceded this entry's publication";

    capacity_consume_frame(mailbox, 0, MailboxPreparationDisposition::NATIVE_PREPARED);
    set_capacity_entry_state(entry, MailboxState::TASK_DONE);
    manager.progress();
    allocator.slot_state(submitted.task_slot)->state.store(TaskState::COMPLETED);
    EXPECT_TRUE(orch.on_consumed(submitted.task_slot));
    orch.release_run(run);
    EXPECT_FALSE(orch.run_observation_requested(run));
    manager.stop();
    allocator.shutdown();
}

namespace {

void complete_capacity_requests(
    std::array<char, MAILBOX_SIZE> &mailbox, uint32_t capacity,
    std::array<uint64_t, MAILBOX_REQUEST_FRAMES> &consumed_generations
) {
    for (size_t frame_index = 0; frame_index < MAILBOX_REQUEST_FRAMES; ++frame_index) {
        char *frame = capacity_task_frame(mailbox, frame_index);
        const auto state = static_cast<MailboxState>(
            __atomic_load_n(reinterpret_cast<int32_t *>(frame + MAILBOX_OFF_STATE), __ATOMIC_ACQUIRE)
        );
        if (state != MailboxState::TASK_READY && state != MailboxState::PREPARE_READY &&
            state != MailboxState::ACTIVATE)
            continue;
        uint64_t generation = 0;
        std::memcpy(&generation, frame + MAILBOX_OFF_FRAME_GENERATION, sizeof(generation));
        if (generation == consumed_generations[frame_index]) continue;
        consumed_generations[frame_index] = generation;
        uint64_t index = 0;
        std::memcpy(&index, frame + MAILBOX_OFF_FRAME_ENTRY_INDEX, sizeof(index));
        capacity_consume_frame(mailbox, frame_index, MailboxPreparationDisposition::NATIVE_PREPARED);
        auto *entry = capacity_entry(mailbox, index);
        __atomic_store_n(&entry->accepted, MAILBOX_TASK_ACCEPTED, __ATOMIC_RELEASE);
        __atomic_store_n(&entry->admissible_run_resource_sets, capacity, __ATOMIC_RELEASE);
        set_capacity_entry_state(entry, MailboxState::TASK_DONE);
    }
}

}  // namespace

TEST(ThreeRunCapacityEndpointTest, PhysicalEndFoldsMeasuredCapacityBeforeSemanticDelivery) {
    alignas(8) std::array<char, MAILBOX_SIZE> mailbox{};
    Worker worker(3, 1ULL << 20);
    worker.configure_pipeline_depth(3, 0, 3, 1);
    worker.add_next_level_worker(0, mailbox.data(), mailbox.size(), -1, 3);
    worker.init();
    Orchestrator &orch = worker.get_orchestrator();
    const RunId first = orch.begin_run();
    CallableIdentity callable;
    callable.digest.fill(219);
    orch.submit_next_level(callable, TaskArgs{}, CallConfig{}, 0);
    orch.close_run_submission(first);
    auto *const entry = capacity_entry(mailbox, 0);
    const bool dispatched = settled_within([&] {
        return __atomic_load_n(
                   reinterpret_cast<int32_t *>(capacity_task_frame(mailbox, 0) + MAILBOX_OFF_STATE), __ATOMIC_ACQUIRE
               ) == static_cast<int32_t>(MailboxState::TASK_READY);
    });
    EXPECT_TRUE(dispatched);
    if (dispatched) {
        capacity_consume_frame(mailbox, 0, MailboxPreparationDisposition::NATIVE_PREPARED);
        __atomic_store_n(&entry->accepted, MAILBOX_TASK_ACCEPTED, __ATOMIC_RELEASE);
        __atomic_store_n(&entry->admissible_run_resource_sets, 3, __ATOMIC_RELEASE);
        const uint64_t generation = __atomic_load_n(&entry->entry_generation, __ATOMIC_ACQUIRE);
        set_capacity_entry_state(entry, MailboxState::TASK_ENDED);
        const bool measured_before_delivery = settled_within([&] {
            return orch.admissible_depth() == 3;
        });
        EXPECT_TRUE(measured_before_delivery)
            << "the actual end callback returned execution credit without forwarding measured capacity";
        EXPECT_EQ(__atomic_load_n(&entry->entry_generation, __ATOMIC_ACQUIRE), generation);
        EXPECT_EQ(capacity_entry_state(entry), MailboxState::TASK_ENDED);
        EXPECT_FALSE(orch.run_done(first));

        std::vector<RunId> successors;
        if (measured_before_delivery) {
            for (int index = 0; index < 3; ++index) {
                const RunId next = orch.begin_run();
                orch.submit_next_level(callable, TaskArgs{}, CallConfig{}, 0);
                orch.close_run_submission(next);
                successors.push_back(next);
            }
            EXPECT_EQ(orch.active_run_id(), successors[0]);
            EXPECT_EQ(orch.preparable_run_ids(), (std::vector<RunId>{successors[1], successors[2]}));
            EXPECT_FALSE(orch.run_done(first));
            EXPECT_EQ(capacity_entry_state(entry), MailboxState::TASK_ENDED);
        }

        // The terminal route remains a positive control for the same actual mailbox figure.
        set_capacity_entry_state(entry, MailboxState::TASK_DONE);
        EXPECT_TRUE(orch.wait_run_for(first, 5.0));
        EXPECT_EQ(orch.admissible_depth(), 3u);
        orch.release_run(first);
        std::array<uint64_t, MAILBOX_REQUEST_FRAMES> consumed_generations{};
        EXPECT_TRUE(settled_within([&] {
            complete_capacity_requests(mailbox, 3, consumed_generations);
            return std::all_of(successors.begin(), successors.end(), [&](RunId id) {
                return orch.run_done(id);
            });
        }));
        for (RunId id : successors) {
            EXPECT_TRUE(orch.wait_run_for(id, 5.0));
            orch.release_run(id);
        }
    }
    worker.close();
}

TEST(ThreeRunCapacityEndpointTest, TerminalFallbackRegistersNarrowingBeforePromotingTheNextRun) {
    for (uint32_t reported : {0u, 1u}) {
        SCOPED_TRACE(reported);
        alignas(8) std::array<char, MAILBOX_SIZE> mailbox{};
        Worker worker(3, 1ULL << 20);
        worker.configure_pipeline_depth(3, 0, 3);
        worker.add_next_level_worker(0, mailbox.data(), mailbox.size(), -1, 3);
        worker.init();
        auto &orch = worker.get_orchestrator();
        CallableIdentity callable;
        callable.digest.fill(221);
        std::array<RunId, 2> runs{};
        for (auto &run : runs) {
            run = orch.begin_run();
            orch.submit_next_level(callable, TaskArgs{}, CallConfig{}, 0);
            orch.close_run_submission(run);
        }
        std::atomic<uint32_t> capacity_at_promotion{0};
        orch.set_test_hook([&](OrchestratorTestPoint point) {
            if (point == OrchestratorTestPoint::RUN_ACTIVATION_PUBLISHED)
                capacity_at_promotion.store(orch.admission_depth_, std::memory_order_release);
        });
        auto *entry = capacity_entry(mailbox, 0);
        EXPECT_TRUE(settled_within([&] {
            return __atomic_load_n(
                       reinterpret_cast<int32_t *>(capacity_task_frame(mailbox, 0) + MAILBOX_OFF_STATE),
                       __ATOMIC_ACQUIRE
                   ) == static_cast<int32_t>(MailboxState::TASK_READY);
        }));
        capacity_consume_frame(mailbox, 0, MailboxPreparationDisposition::NATIVE_PREPARED);
        __atomic_store_n(&entry->accepted, MAILBOX_TASK_ACCEPTED, __ATOMIC_RELEASE);
        __atomic_store_n(&entry->admissible_run_resource_sets, reported, __ATOMIC_RELEASE);
        set_capacity_entry_state(entry, MailboxState::TASK_DONE);
        EXPECT_TRUE(settled_within([&] {
            return capacity_at_promotion.load(std::memory_order_acquire) != 0;
        }));
        EXPECT_EQ(capacity_at_promotion.load(std::memory_order_acquire), reported == 0 ? 3u : 1u);
        EXPECT_TRUE(orch.wait_run_for(runs[0], 5.0));
        orch.release_run(runs[0]);
        std::array<uint64_t, MAILBOX_REQUEST_FRAMES> consumed_generations{};
        EXPECT_TRUE(settled_within([&] {
            complete_capacity_requests(mailbox, reported, consumed_generations);
            return orch.run_done(runs[1]);
        }));
        EXPECT_TRUE(orch.wait_run_for(runs[1], 5.0));
        orch.release_run(runs[1]);
        worker.close();
    }
}

TEST(ThreeRunCapacityEndpointTest, GroupCapacityWaitsForEveryPhysicalMemberAndClosedSubmission) {
    for (uint32_t second_report : {0u, 2u}) {
        SCOPED_TRACE(second_report);
        alignas(8) std::array<char, MAILBOX_SIZE> first_mailbox{};
        alignas(8) std::array<char, MAILBOX_SIZE> second_mailbox{};
        Worker worker(3, 1ULL << 20);
        worker.configure_pipeline_depth(3, 0, 3, 1);
        worker.add_next_level_worker(0, first_mailbox.data(), first_mailbox.size(), -1, 3);
        worker.add_next_level_worker(1, second_mailbox.data(), second_mailbox.size(), -1, 3);
        worker.init();
        auto &orch = worker.get_orchestrator();
        const RunId run = orch.begin_run();
        CallableIdentity callable;
        callable.digest.fill(223);
        orch.submit_next_level_group(callable, {TaskArgs{}, TaskArgs{}}, CallConfig{}, {0, 1});
        std::array<std::array<char, MAILBOX_SIZE> *, 2> mailboxes{&first_mailbox, &second_mailbox};
        for (size_t index = 0; index < mailboxes.size(); ++index) {
            auto &mailbox = *mailboxes[index];
            EXPECT_TRUE(settled_within([&] {
                return __atomic_load_n(
                           reinterpret_cast<int32_t *>(capacity_task_frame(mailbox, 0) + MAILBOX_OFF_STATE),
                           __ATOMIC_ACQUIRE
                       ) == static_cast<int32_t>(MailboxState::TASK_READY);
            }));
            capacity_consume_frame(mailbox, 0, MailboxPreparationDisposition::NATIVE_PREPARED);
            auto *entry = capacity_entry(mailbox, 0);
            __atomic_store_n(&entry->accepted, MAILBOX_TASK_ACCEPTED, __ATOMIC_RELEASE);
            __atomic_store_n(&entry->admissible_run_resource_sets, index == 0 ? 3u : second_report, __ATOMIC_RELEASE);
            set_capacity_entry_state(entry, MailboxState::TASK_ENDED);
        }
        EXPECT_TRUE(settled_within([&] {
            std::shared_ptr<RunState> state;
            {
                std::lock_guard<std::mutex> lock(orch.runs_mu_);
                state = orch.runs_.at(run);
                auto first = orch.endpoint_capacity_.find(0);
                if (first == orch.endpoint_capacity_.end() || first->second != 3) return false;
                auto second = orch.endpoint_capacity_.find(1);
                if (second_report != 0 && (second == orch.endpoint_capacity_.end() || second->second != second_report))
                    return false;
            }
            std::lock_guard<std::mutex> lock(state->completion_mu);
            return state->physical_tasks.size() == 1 &&
                   state->physical_tasks[0].ended_members == std::vector<uint8_t>{1, 1};
        }));
        EXPECT_EQ(orch.admissible_depth(), 1u);
        EXPECT_EQ(orch.active_run_id(), run);
        EXPECT_FALSE(orch.run_done(run));
        for (auto *mailbox : mailboxes) {
            auto *entry = capacity_entry(*mailbox, 0);
            EXPECT_EQ(capacity_entry_state(entry), MailboxState::TASK_ENDED);
            EXPECT_EQ(__atomic_load_n(&entry->entry_generation, __ATOMIC_ACQUIRE) % 2, 1u);
        }
        orch.close_run_submission(run);
        EXPECT_TRUE(settled_within([&] {
            return orch.admissible_depth() == (second_report == 0 ? 3u : 2u);
        }));
        EXPECT_FALSE(orch.run_done(run));
        for (auto *mailbox : mailboxes)
            set_capacity_entry_state(capacity_entry(*mailbox, 0), MailboxState::TASK_DONE);
        EXPECT_TRUE(orch.wait_run_for(run, 5.0));
        EXPECT_EQ(orch.admissible_depth(), second_report == 0 ? 3u : 2u);
        orch.release_run(run);
        worker.close();
    }
}

TEST(ThreeRunCapacityEndpointTest, PersistentDemandWaitsOnlyThroughAuthorizedActivationPublication) {
    CapacityHarness h(2, 2);
    const auto first = h.build_closed_run(301);
    const auto second = h.build_closed_run(302);
    h.accept(first);
    alignas(8) std::array<char, MAILBOX_SIZE> mailbox{};
    LocalMailboxEndpoint endpoint(0, mailbox.data(), mailbox.size(), -1, 2);
    endpoint.set_observation_demand([&](RunId id) {
        return h.orch.run_observation_requested(id);
    });
    endpoint.set_activation_permission([&](RunId id) {
        return h.orch.run_activation_authorized(id);
    });
    endpoint.submit_progress(&h.allocator, WorkerDispatch{first.task.task_slot, 0, 1, false});
    endpoint.submit_progress(&h.allocator, WorkerDispatch{second.task.task_slot, 0, 2, true});
    auto *a = capacity_entry(mailbox, 0);
    auto *b = capacity_entry(mailbox, 1);
    EXPECT_TRUE(h.orch.run_activation_authorized(second.id));
    EXPECT_FALSE(h.orch.wait_run_for(first.id, 0.0001));
    ASSERT_TRUE(h.orch.run_observation_requested(first.id));
    endpoint.progress_observations();
    EXPECT_EQ(a->observation_requested, 0) << "installed eligible B is still being classified";
    capacity_consume_frame(mailbox, 1, MailboxPreparationDisposition::NATIVE_PREPARED);
    endpoint.progress_observations();
    EXPECT_EQ(a->observation_requested, 0) << "native preparation is not parent ACTIVATE publication";
    ASSERT_TRUE(endpoint.activate_progress(second.id));
    endpoint.progress_observations();
    EXPECT_EQ(capacity_entry_state(b), MailboxState::ACTIVATE);
    EXPECT_EQ(a->observation_requested, 1) << "do not wait for B launch, admission or semantic result";
}

TEST(ThreeRunCapacityEndpointTest, ObservationRechecksInstallationsAcrossTheUnlockedPolicyQuery) {
    alignas(8) std::array<char, MAILBOX_SIZE> mailbox{};
    Ring ring;
    ring.init(0);
    const auto retired = capacity_progress_slot(ring, 400, 2, 40);
    const auto a = capacity_progress_slot(ring, 401, 0, 41);
    const auto b = capacity_progress_slot(ring, 402, 1, 42);
    const auto c = capacity_progress_slot(ring, 403, 2, 43);
    LocalMailboxEndpoint endpoint(0, mailbox.data(), mailbox.size(), -1, 3);
    endpoint.submit_progress(&ring, WorkerDispatch{retired, 0, 1, false});
    endpoint.submit_progress(&ring, WorkerDispatch{a, 0, 2, false});
    capacity_consume_frame(mailbox, 0, MailboxPreparationDisposition::NATIVE_PREPARED);
    endpoint.submit_progress(&ring, WorkerDispatch{b, 0, 3, true});
    set_capacity_entry_state(capacity_entry(mailbox, 0), MailboxState::TASK_DONE);
    WorkerEndpointProgress terminal;
    ASSERT_TRUE(endpoint.poll_progress(terminal));
    ASSERT_EQ(terminal.kind, WorkerProgressKind::COMPLETED);
    ASSERT_EQ(terminal.dispatch.dispatch_id, 1u);
    // A remains in entry1 and B in entry2. C reuses the now-free entry0,
    // which the observation scan has already passed before B's policy query.
    capacity_consume_frame(mailbox, 1, MailboxPreparationDisposition::NATIVE_PREPARED);
    bool installed = false;
    endpoint.set_activation_permission([&](RunId id) {
        if (!installed) {
            installed = true;
            endpoint.submit_progress(&ring, WorkerDispatch{c, 0, 4, true});
        }
        return id == 403;
    });
    endpoint.request_observation(401);
    ASSERT_TRUE(installed);
    EXPECT_EQ(capacity_entry(mailbox, 0)->run_id, 403u);
    EXPECT_EQ(capacity_entry(mailbox, 1)->observation_requested, 0);
    endpoint.request_observation(401);
    EXPECT_EQ(capacity_entry(mailbox, 1)->observation_requested, 0) << "newly installed eligible C owns precedence";
    capacity_consume_frame(mailbox, 1, MailboxPreparationDisposition::VALIDATED_ONLY);
    endpoint.request_observation(401);
    EXPECT_EQ(capacity_entry(mailbox, 1)->observation_requested, 1);
    ring.shutdown();
}

TEST(ThreeRunCapacityEndpointTest, ClassificationFailureAndIneligibleSuccessorsNeverStrandObservation) {
    for (int scenario = 0; scenario != 5; ++scenario) {
        alignas(8) std::array<char, MAILBOX_SIZE> mailbox{};
        Ring ring;
        ring.init(0);
        const auto a = capacity_progress_slot(ring, 501, 0, 51);
        const auto b = capacity_progress_slot(ring, 502, 1, 52);
        LocalMailboxEndpoint endpoint(0, mailbox.data(), mailbox.size(), -1, 2);
        endpoint.submit_progress(&ring, WorkerDispatch{a, 0, 1, false});
        endpoint.submit_progress(&ring, WorkerDispatch{b, 0, 2, true});
        endpoint.set_activation_permission([&](RunId) {
            if (scenario == 4) throw std::runtime_error("policy unavailable");
            return scenario != 0;
        });
        if (scenario == 1) capacity_consume_frame(mailbox, 1, MailboxPreparationDisposition::VALIDATED_ONLY);
        if (scenario == 2) set_capacity_entry_state(capacity_entry(mailbox, 1), MailboxState::TASK_FAILED);
        if (scenario == 3) set_capacity_entry_state(capacity_entry(mailbox, 1), MailboxState::TASK_DONE);
        endpoint.request_observation(501);
        EXPECT_EQ(capacity_entry(mailbox, 0)->observation_requested, 1) << scenario;
        ring.shutdown();
    }
}

TEST(ThreeRunCapacityEndpointTest, PublishedDemandIsNeverWithdrawnAndTerminalEntriesAreNeverWritten) {
    alignas(8) std::array<char, MAILBOX_SIZE> mailbox{};
    Ring ring;
    ring.init(0);
    const auto a = capacity_progress_slot(ring, 601, 0, 61);
    const auto b = capacity_progress_slot(ring, 602, 1, 62);
    LocalMailboxEndpoint endpoint(0, mailbox.data(), mailbox.size(), -1, 2);
    endpoint.set_activation_permission([](RunId) {
        return true;
    });
    endpoint.submit_progress(&ring, WorkerDispatch{a, 0, 1, false});
    endpoint.request_observation(601);
    EXPECT_EQ(capacity_entry(mailbox, 0)->observation_requested, 1);
    endpoint.submit_progress(&ring, WorkerDispatch{b, 0, 2, true});
    endpoint.request_observation(601);
    EXPECT_EQ(capacity_entry(mailbox, 0)->observation_requested, 1);
    set_capacity_entry_state(capacity_entry(mailbox, 1), MailboxState::TASK_DONE);
    endpoint.request_observation(602);
    EXPECT_EQ(capacity_entry(mailbox, 1)->observation_requested, 0);
    ring.shutdown();
}
