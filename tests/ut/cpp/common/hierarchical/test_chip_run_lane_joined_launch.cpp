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
 * The lane launching a *ready prefix* rather than only its front.
 *
 * The front is the oldest launched run, not the only one: everything behind it
 * that is activated, natively prepared and in scope follows it onto the device
 * while it is still executing. These tests pin the scope the lane enforces —
 * host-space tensors, exclusive diagnostics, no live communication or exported
 * device resource — the silent fallback when a backend declines, and that a
 * second launched run does not let a successor be reported finished ahead of
 * the run it was ordered behind.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "call_config.h"
#include "pipeline_slot_pool.h"
#include "runtime_c_api.h"
#include "task_args.h"
#include "tensor.h"
#include "types.h"

#define private public
#include "chip_worker.h"
#undef private
#include "chip_run_lane.h"

#include <gtest/gtest.h>

namespace {

std::unordered_map<void *, uint32_t> g_slots;
std::array<bool, 2> g_complete{};
std::array<int, 2> g_launch_rc{};
std::array<bool, 2> g_joinable_boundary_requested{};
int g_joined_rc{0};
bool g_joined_throws{false};
bool g_supports_joined{true};
std::vector<std::string> g_events;

uint32_t slot_of(void *runtime) { return g_slots.at(runtime); }

int prepare_run(
    void *, void *runtime, int32_t, const void *, const CallConfig *, const NativeRunDescriptor *descriptor
) {
    EXPECT_EQ(slot_of(runtime), descriptor->pipeline_slot);
    g_complete[descriptor->pipeline_slot] = false;
    g_joinable_boundary_requested[descriptor->pipeline_slot] = descriptor->joinable_boundary != 0;
    g_events.push_back("prepare" + std::to_string(descriptor->pipeline_slot));
    return 0;
}

int launch_run(void *, void *runtime) {
    const uint32_t slot = slot_of(runtime);
    g_events.push_back("launch" + std::to_string(slot));
    return g_launch_rc[slot];
}

int launch_run_joined(void *, void *runtime, void *predecessor) {
    const uint32_t slot = slot_of(runtime);
    g_events.push_back("joined" + std::to_string(slot) + "behind" + std::to_string(slot_of(predecessor)));
    if (g_joined_throws) throw std::runtime_error("joined launch threw");
    return g_joined_rc;
}

int poll_run(void *, void *runtime) {
    const uint32_t slot = slot_of(runtime);
    return g_complete[slot] ? SIMPLER_NATIVE_RUN_POLL_COMPLETE : SIMPLER_NATIVE_RUN_POLL_NOT_READY;
}

int wait_run(void *, void *runtime) {
    const uint32_t slot = slot_of(runtime);
    g_events.push_back("wait" + std::to_string(slot));
    g_complete[slot] = true;
    return 0;
}

int g_finalize_rc{0};

int finalize_run(void *, void *runtime) {
    g_events.push_back("finalize" + std::to_string(slot_of(runtime)));
    return g_finalize_rc;
}

int supports_successor(void *) { return 1; }
int supports_joined(void *) { return g_supports_joined ? 1 : 0; }

void prime_worker(ChipWorker &worker, unsigned launch_depth = 2) {
    g_slots.clear();
    g_complete = {};
    g_launch_rc = {};
    g_joinable_boundary_requested = {};
    g_joined_rc = 0;
    g_joined_throws = false;
    g_supports_joined = true;
    g_finalize_rc = 0;
    g_events.clear();
    worker.launch_depth_ = launch_depth;
    worker.initialized_ = true;
    worker.pipeline_contract_ = {PTO_PIPELINE_CONTRACT_ABI_VERSION, 0, 2, {}};
    worker.runtime_bufs_.emplace_back(64, alignof(std::max_align_t));
    worker.runtime_bufs_.emplace_back(64, alignof(std::max_align_t));
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

/** The same, in the caller's own device address space. */
ChipStorageTaskArgs device_args() {
    ChipStorageTaskArgs args{};
    ChipTensor t{};
    const uint32_t shape = 1;
    t.init_external(reinterpret_cast<void *>(0x2000), 1, &shape, 1, DataType::UINT8, AddressSpace::DEVICE);
    args.add_tensor(t);
    return args;
}

ChipRun submit(
    ChipRunLane &lane, uint64_t run_id, uint32_t slot, bool activate, const ChipStorageTaskArgs &args,
    const CallConfig &config = CallConfig{}
) {
    return lane.submit(1, args, config, PipelineSlotLease{slot, 0, run_id}, run_id, run_id, nullptr, 0, activate);
}

// The caller-device-buffer hooks, for the runs whose arguments name one. A lane only borrows when
// the runtime publishes them, which is why every case above leaves them unset and keeps its
// device-tensor runs off the joined path.
int g_borrow_rc{0};
bool g_borrow_throws{false};
std::vector<uint64_t> g_borrowed;
std::vector<std::pair<uint64_t, int>> g_released;

int borrow_caller_buffers(void *, const CallerBufferSpan *, uint32_t count, uint64_t borrow_id) {
    if (g_borrow_throws) throw std::bad_alloc();
    if (g_borrow_rc != 0) return g_borrow_rc;
    (void)count;
    g_borrowed.push_back(borrow_id);
    return 0;
}

void release_caller_buffers(void *, uint64_t borrow_id, int keep) { g_released.emplace_back(borrow_id, keep); }

void prime_caller_buffers(ChipWorker &worker) {
    g_borrow_rc = 0;
    g_borrow_throws = false;
    g_borrowed.clear();
    g_released.clear();
    worker.device_borrow_caller_buffers_ctx_fn_ = borrow_caller_buffers;
    worker.device_release_caller_buffers_ctx_fn_ = release_caller_buffers;
}

using Events = std::vector<std::string>;

TEST(ChipRunLaneJoinedLaunchTest, TheSuccessorReachesTheDeviceWhileTheFrontStillExecutes) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);

    ChipRun first = submit(lane, 101, 0, true, host_args());
    ChipRun second = submit(lane, 102, 1, false, host_args());
    // Every run launched at a depth above one builds the whole-operator
    // boundary, including the first: it is launched before any successor can be
    // authorized to join it.
    EXPECT_TRUE(g_joinable_boundary_requested[0]);
    EXPECT_TRUE(g_joinable_boundary_requested[1]);
    EXPECT_EQ(g_events, (Events{"prepare0", "launch0", "prepare1"}));

    // Activation is what the parent's authorization reaches the child as, and
    // the front has not finished.
    second.activate();
    EXPECT_FALSE(g_complete[0]);
    EXPECT_TRUE(second.launched());
    EXPECT_EQ(g_events, (Events{"prepare0", "launch0", "prepare1", "joined1behind0"}));

    g_complete[0] = true;
    EXPECT_TRUE(first.done());
    g_complete[1] = true;
    EXPECT_TRUE(second.done());
    EXPECT_EQ(g_events, (Events{"prepare0", "launch0", "prepare1", "joined1behind0", "finalize0", "finalize1"}));
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneJoinedLaunchTest, DepthOneNeitherJoinsNorAsksForAJoinableBoundary) {
    ChipWorker worker;
    prime_worker(worker, /*launch_depth=*/1);
    ChipRunLane lane(worker);

    ChipRun first = submit(lane, 101, 0, true, host_args());
    ChipRun second = submit(lane, 102, 1, false, host_args());
    EXPECT_FALSE(g_joinable_boundary_requested[0]);
    EXPECT_FALSE(g_joinable_boundary_requested[1]);

    second.activate();
    EXPECT_FALSE(second.launched());
    g_complete[0] = true;
    EXPECT_TRUE(first.done());
    EXPECT_TRUE(second.launched());
    EXPECT_EQ(g_events, (Events{"prepare0", "launch0", "prepare1", "finalize0", "launch1"}));
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneJoinedLaunchTest, ADeclinedJoinLeavesTheRunToLaunchOrdinarily) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    g_joined_rc = PTO_RUNTIME_ERR_UNSUPPORTED;

    ChipRun first = submit(lane, 101, 0, true, host_args());
    ChipRun second = submit(lane, 102, 1, false, host_args());
    second.activate();
    // The backend refused and changed nothing, so the run is still prepared.
    EXPECT_FALSE(second.launched());
    EXPECT_EQ(g_events, (Events{"prepare0", "launch0", "prepare1", "joined1behind0"}));

    // Asked once, not once per progress round. A backend declines only for
    // reasons that cannot change while the run ahead is still executing, so
    // re-asking would issue a device call per round to get the same answer.
    EXPECT_FALSE(second.done());
    EXPECT_FALSE(second.done());
    EXPECT_EQ(g_events, (Events{"prepare0", "launch0", "prepare1", "joined1behind0"}));

    g_complete[0] = true;
    EXPECT_TRUE(first.done());
    EXPECT_TRUE(second.launched());
    EXPECT_EQ(g_events, (Events{"prepare0", "launch0", "prepare1", "joined1behind0", "finalize0", "launch1"}));
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneJoinedLaunchTest, ADeviceSpaceTensorOnEitherRunFallsBackToTheSerialLaunch) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);

    // A device-space tensor is the caller's own address with the caller's
    // lifetime, where a host one is copied into staging this run retains.
    ChipRun first = submit(lane, 101, 0, true, host_args());
    ChipRun second = submit(lane, 102, 1, false, device_args());
    second.activate();
    EXPECT_FALSE(second.launched());
    // The backend was never asked, so no refusal was needed.
    EXPECT_EQ(g_events, (Events{"prepare0", "launch0", "prepare1"}));

    g_complete[0] = true;
    EXPECT_TRUE(first.done());
    EXPECT_TRUE(second.launched());
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneJoinedLaunchTest, DiagnosticsStayExclusiveToOneLaunchedRun) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    CallConfig diagnostic;
    diagnostic.enable_pmu = true;
    diagnostic.output_prefix[0] = 'x';

    // Preparation is not special-cased — the successor still prepares natively
    // — but the collector pools are armed for one run at launch, so a second
    // launched run would publish one run's records as the other's.
    ChipRun first = submit(lane, 101, 0, true, host_args());
    ChipRun second = submit(lane, 102, 1, false, host_args(), diagnostic);
    EXPECT_EQ(second.preparation_disposition(), ChipRunPreparationDisposition::NATIVE_PREPARED);
    second.activate();
    EXPECT_FALSE(second.launched());
    EXPECT_EQ(g_events, (Events{"prepare0", "launch0", "prepare1"}));

    g_complete[0] = true;
    EXPECT_TRUE(first.done());
    EXPECT_TRUE(second.launched());
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneJoinedLaunchTest, ALiveExportedDeviceRegionWithholdsEveryJoin) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);

    // Exported regions are released before the child's device reset, so a run
    // that could still be naming one must not be queued behind a peer whose
    // failure would end the generation.
    worker.set_exported_device_regions_live(true);
    ChipRun first = submit(lane, 101, 0, true, host_args());
    ChipRun second = submit(lane, 102, 1, false, host_args());
    second.activate();
    EXPECT_FALSE(second.launched());

    g_complete[0] = true;
    EXPECT_TRUE(first.done());
    EXPECT_TRUE(second.launched());
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneJoinedLaunchTest, AJoinedSuccessorIsNotReportedFinishedAheadOfItsPredecessor) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);

    ChipRun first = submit(lane, 101, 0, true, host_args());
    ChipRun second = submit(lane, 102, 1, false, host_args());
    second.activate();
    ASSERT_TRUE(second.launched());

    // The device cannot finish the successor first — it waits on the
    // predecessor's whole-operator boundary — so a poll that says otherwise is
    // answering about the wrong run. The lane drives the front first and
    // refuses to finalize out of order, which is what keeps one run's
    // copy-back and errors from being charged to the other.
    g_complete[1] = true;
    EXPECT_FALSE(second.done());
    EXPECT_EQ(g_events, (Events{"prepare0", "launch0", "prepare1", "joined1behind0"}));

    g_complete[0] = true;
    EXPECT_TRUE(second.done());
    // Front first, then the run behind it.
    EXPECT_EQ(g_events, (Events{"prepare0", "launch0", "prepare1", "joined1behind0", "finalize0", "finalize1"}));
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneJoinedLaunchTest, AThrowingJoinedLaunchFailsOnlyThatRunAndPoisonsTheLane) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    g_joined_throws = true;

    ChipRun first = submit(lane, 101, 0, true, host_args());
    ChipRun second = submit(lane, 102, 1, false, host_args());
    second.activate();

    // `done` reports progress and never raises; the error surfaces where a
    // caller asks for the outcome. The successor is terminal with its own
    // failure, which is the run that failed.
    EXPECT_TRUE(second.done());
    EXPECT_THROW(second.wait_until(ChipRunLane::Deadline::max()), std::runtime_error);

    // The lane stops admitting runs. A failure at this point may already have
    // reached the device, and what it may hold sits in the stream pair the run
    // ahead is still executing on, so a third run must not be queued behind an
    // uncertain one. This is a failure, not a decline: a decline reports
    // false, having changed nothing, and leaves the lane usable.
    EXPECT_TRUE(lane.poisoned());

    // The predecessor is retained rather than failed with it: it is still
    // executing, it owns its own outcome, and its drain is what discharges
    // whatever the failed successor may have queued against its boundary.
    g_complete[0] = true;
    EXPECT_TRUE(first.done());
    EXPECT_TRUE(first.wait_until(ChipRunLane::Deadline::max()));
    EXPECT_EQ(g_events, (Events{"prepare0", "launch0", "prepare1", "joined1behind0", "finalize1", "finalize0"}));

    EXPECT_THROW(lane.close(), std::runtime_error);
    worker.finalize();
}

// ---------------------------------------------------------------------------
// Caller device buffers: what admits a device-space run, and what admission
// does when taking the reference fails.
// ---------------------------------------------------------------------------

TEST(ChipRunLaneCallerBuffersTest, ABorrowedDeviceArgumentReachesTheDeviceEarly) {
    ChipWorker worker;
    prime_worker(worker);
    prime_caller_buffers(worker);
    ChipRunLane lane(worker);

    // The reference is what makes the caller's address admissible: with it the successor's device
    // arguments no longer keep it off the joined path.
    ChipRun first = submit(lane, 101, 0, true, device_args());
    ChipRun second = submit(lane, 102, 1, false, device_args());
    second.activate();
    EXPECT_TRUE(second.launched());
    EXPECT_EQ(g_events, (Events{"prepare0", "launch0", "prepare1", "joined1behind0"}));
    // One per slot, and both held while both runs are live.
    EXPECT_EQ(g_borrowed, (std::vector<uint64_t>{1, 2}));
    EXPECT_TRUE(g_released.empty());

    g_complete[0] = true;
    g_complete[1] = true;
    EXPECT_TRUE(first.done());
    EXPECT_TRUE(second.done());
    // Each reference is given back at its own run's finalize, and given back rather than kept:
    // the finalize succeeded, so the last consumer is proven done.
    EXPECT_EQ(g_released, (std::vector<std::pair<uint64_t, int>>{{1, 0}, {2, 0}}));
    lane.close();
    worker.finalize();
}

// A predecessor whose device span has no provable owner carries no successor *preparation*
// either, not just no joined launch. The successor's bind runs its own host graph build, which can
// read a device argument's bytes; the only thing that keeps it off bytes the run ahead has not
// produced is that run's declaration, and a run holding no borrow proved no span to declare.
TEST(ChipRunLaneCallerBuffersTest, AnUnprovableFrontCarriesNoConcurrentPreparation) {
    ChipWorker worker;
    prime_worker(worker);
    prime_caller_buffers(worker);
    g_borrow_rc = PTO_RUNTIME_ERR_INVALID_STATE;
    ChipRunLane lane(worker);

    ChipRun first = submit(lane, 101, 0, true, device_args());
    ASSERT_TRUE(first.launched());
    ASSERT_TRUE(g_borrowed.empty()) << "the front proved no owner for its device span";

    // Admitted and activated, and still not prepared: the front is launched and the backend
    // supports concurrent preparation, so the shape is the only thing refusing it.
    ChipRun second = submit(lane, 102, 1, true, host_args());
    EXPECT_EQ(g_events, (Events{"prepare0", "launch0"}));
    EXPECT_FALSE(second.launched());

    // It prepares at the front instead, once the run ahead has retired.
    g_complete[0] = true;
    EXPECT_TRUE(first.done());
    EXPECT_TRUE(second.launched());
    EXPECT_EQ(g_events, (Events{"prepare0", "launch0", "finalize0", "prepare1", "launch1"}));

    g_complete[1] = true;
    EXPECT_TRUE(second.done());
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneCallerBuffersTest, AnUnprovableDeviceArgumentStaysOnTheSerialPath) {
    ChipWorker worker;
    prime_worker(worker);
    prime_caller_buffers(worker);
    g_borrow_rc = PTO_RUNTIME_ERR_INVALID_STATE;
    ChipRunLane lane(worker);

    // A refused borrow is an address whose owner this context cannot prove. The run is admitted
    // and correct, it simply does not overlap — which is the behaviour it had before any of this.
    ChipRun first = submit(lane, 101, 0, true, host_args());
    ChipRun second = submit(lane, 102, 1, false, device_args());
    second.activate();
    EXPECT_FALSE(second.launched());
    EXPECT_TRUE(g_borrowed.empty());
    EXPECT_TRUE(g_released.empty()) << "no run has retired yet";

    g_complete[0] = true;
    EXPECT_TRUE(first.done());
    EXPECT_TRUE(second.launched());
    // Every retired run is discharged, borrow or no borrow: the release is also what drops the
    // declaration its bind may have made, and a run whose borrow was refused can still have made
    // one. So a release with nothing to give back is the ordinary case, not a leak.
    EXPECT_EQ(g_released, (std::vector<std::pair<uint64_t, int>>{{1, 0}}));
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneCallerBuffersTest, AFailedBorrowLeavesNoQueuedRunBehind) {
    ChipWorker worker;
    prime_worker(worker);
    prime_caller_buffers(worker);
    ChipRunLane lane(worker);

    ChipRun first = submit(lane, 101, 0, true, device_args());
    ASSERT_TRUE(first.launched());
    ASSERT_EQ(g_borrowed, (std::vector<uint64_t>{1}));

    // Taking the reference allocates, so it can fail. The run must then fail as a run — not leave
    // an entry queued behind the one still executing, which would make the lane's front a run
    // nothing will ever finish.
    g_borrow_throws = true;
    ChipRun second = submit(lane, 102, 1, false, device_args());
    // The handle carries the failure the borrow raised, unwrapped: admission stored this run's own
    // error, so the caller sees what actually went wrong rather than a lane-shaped substitute.
    EXPECT_ANY_THROW(second.activate());
    EXPECT_FALSE(second.launched());
    // A throw here acquired nothing and admitted nothing, so this run's identity never became
    // live in the table and there is nothing to discharge for it.
    EXPECT_EQ(g_borrowed, (std::vector<uint64_t>{1}));
    EXPECT_TRUE(g_released.empty());
    // The run ahead is untouched: still launched, still holding its own reference, and still the
    // front — so the slot the failed run had is free for the next admission.
    EXPECT_TRUE(first.launched());
    g_borrow_throws = false;
    ChipRun third = submit(lane, 103, 1, false, device_args());
    EXPECT_EQ(g_borrowed, (std::vector<uint64_t>{1, 2}));

    g_complete[0] = true;
    g_complete[1] = true;
    EXPECT_TRUE(first.done());
    third.activate();
    EXPECT_TRUE(third.done());
    EXPECT_EQ(g_released, (std::vector<std::pair<uint64_t, int>>{{1, 0}, {2, 0}}));
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneCallerBuffersTest, AnUnprovenLastConsumerKeepsTheReference) {
    ChipWorker worker;
    prime_worker(worker);
    prime_caller_buffers(worker);
    ChipRunLane lane(worker);

    ChipRun run = submit(lane, 101, 0, true, device_args());
    ASSERT_TRUE(run.launched());
    // A finalize that failed proves the opposite of what a release needs: the device may still
    // name those bytes, so the reference is kept rather than handed back.
    g_finalize_rc = -5;
    g_complete[0] = true;
    EXPECT_TRUE(run.done());
    EXPECT_EQ(g_released, (std::vector<std::pair<uint64_t, int>>{{1, 1}}));

    g_finalize_rc = 0;
    EXPECT_THROW(lane.close(), std::runtime_error);
    worker.finalize();
}

}  // namespace
