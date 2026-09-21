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
#include <string>
#include <unordered_map>
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

int finalize_run(void *, void *runtime) {
    g_events.push_back("finalize" + std::to_string(slot_of(runtime)));
    return 0;
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

}  // namespace
