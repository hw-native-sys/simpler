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
 * What a second diagnostic run may assume while the first still executes.
 *
 * Three things have to hold before a successor's submission may reach the
 * device beside its predecessor's, and each is a separate mechanism:
 *
 *  - the producer's record counters start this run at zero, cleared by the
 *    device itself rather than by a host write that would land mid-flight;
 *  - a run's metadata belongs to the run, taken at admission rather than read
 *    at close from a copy the successor has already replaced;
 *  - a hand-off this collector cannot charge is reported under one permanent
 *    policy, decided once, so a default run's loss never downgrades a retained
 *    artifact and a retained failure is never hidden by turning retention off.
 *
 * Host and device share process memory here, so nothing below is evidence
 * about device cache visibility or about real early submission.
 */

#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "common/chip_swimlane_profiling.h"
#include "host/chip_swimlane_collector.h"
#include "host/raii_scope_guard.h"

namespace fs = std::filesystem;

namespace {

void *joined_alloc(size_t size) { return std::calloc(1, size); }
int joined_free(void *p) {
    std::free(p);
    return 0;
}
std::thread joined_thread(std::function<void()> fn) { return std::thread(std::move(fn)); }

/**
 * A temp directory this process alone owns.
 *
 * This file is built PER_ARCH PER_RUNTIME, so four binaries carry the same case
 * names and the host lane runs them with `ctest -j4`. A root named from the case
 * alone would therefore be the same path in four concurrent processes, and each
 * one's setup or teardown would delete directories the others are publishing
 * into. The pid is what makes the root private, as `ArtifactRoot` in
 * `test_chip_swimlane_retained_runs.cpp` already does.
 */
fs::path private_artifact_root(const char *name) {
    return fs::temp_directory_path() / ("simpler-ut-" + std::string(name) + "-" + std::to_string(::getpid()));
}

profiling_common::RetiredHandoff<ChipSwimlaneModule> retired_handoff(uint64_t epoch, uint32_t records) {
    profiling_common::RetiredHandoff<ChipSwimlaneModule> r;
    r.identified = true;
    r.info.run_epoch = epoch;
    r.info.record_count = records;
    return r;
}

profiling_common::RetiredHandoff<ChipSwimlaneModule> unidentified_handoff() {
    return profiling_common::RetiredHandoff<ChipSwimlaneModule>{};
}

// ---------------------------------------------------------------------------
// The producer clears its own counters, over the whole grid
// ---------------------------------------------------------------------------

/**
 * Drives the real `chip_swimlane_aicpu_init` over a host-allocated region, the
 * way the collector lays one out, and reads the pool heads back afterwards.
 *
 * The point of each case is a head the device loop does *not* visit: a core
 * past `worker_count`, a phase pool that only exists above level 2, and a pool
 * whose free queue is empty so the acquisition branch is skipped. Those are the
 * heads the host loop used to cover, and they are what makes "move the reset to
 * the device" a full-grid question rather than a placement one.
 */
struct DeviceResetFixture {
    ChipSwimlaneCollector collector;

    // No artifact root: these cases drive the device initializer over a
    // collector that never admits a run, so nothing here publishes a file.
    explicit DeviceResetFixture(
        ChipSwimlaneLevel level = ChipSwimlaneLevel::SCHED_PHASES, int num_aicore = 2, int aicpu_thread_num = 1
    ) {
        EXPECT_EQ(
            collector.initialize(
                num_aicore, aicpu_thread_num, /*device_id=*/0, level, joined_alloc, nullptr, joined_free
            ),
            0
        );
        set_platform_chip_swimlane_base(reinterpret_cast<uint64_t>(collector.get_chip_swimlane_setup_device_ptr()));
        set_chip_swimlane_enabled(true);
    }
    ~DeviceResetFixture() {
        set_chip_swimlane_enabled(false);
        set_platform_chip_swimlane_base(0);
        collector.finalize(nullptr, joined_free);
    }

    void *shm() { return collector.get_chip_swimlane_setup_device_ptr(); }

    /** Dirty every head on the grid, as a previous run at a higher level would. */
    void dirty_every_head(uint32_t value) {
        for (int i = 0; i < PLATFORM_MAX_CORES; i++) {
            dirty(get_perf_buffer_state(shm(), i)->head, value);
            dirty(get_aicore_buffer_state(shm(), i)->head, value);
        }
        for (int t = 0; t < PLATFORM_MAX_AICPU_THREADS; t++) {
            dirty(get_sched_phase_buffer_state(shm(), t)->head, value);
            dirty(get_orch_phase_buffer_state(shm(), t)->head, value);
        }
    }

    /** Every head on the grid, after whatever the case just did to it. */
    ::testing::AssertionResult whole_grid_is_zero() {
        for (int i = 0; i < PLATFORM_MAX_CORES; i++) {
            if (auto r = all_zero(get_perf_buffer_state(shm(), i)->head, "task head"); !r) return r << " at core " << i;
            if (auto r = all_zero(get_aicore_buffer_state(shm(), i)->head, "aicore head"); !r)
                return r << " at core " << i;
        }
        for (int t = 0; t < PLATFORM_MAX_AICPU_THREADS; t++) {
            if (auto r = all_zero(get_sched_phase_buffer_state(shm(), t)->head, "sched head"); !r)
                return r << " at thread " << t;
            if (auto r = all_zero(get_orch_phase_buffer_state(shm(), t)->head, "orch head"); !r)
                return r << " at thread " << t;
        }
        return ::testing::AssertionSuccess();
    }

    static void dirty(ChipSwimlaneActiveHead &head, uint32_t value) {
        head.total_record_count = value;
        head.dropped_record_count = value;
        head.live_record_count = value;
        head.published_record_count = value;
        head.published_buffer_count = value;
    }

    static ::testing::AssertionResult all_zero(const ChipSwimlaneActiveHead &head, const char *what) {
        if (head.total_record_count != 0) return ::testing::AssertionFailure() << what << " total";
        if (head.dropped_record_count != 0) return ::testing::AssertionFailure() << what << " dropped";
        if (head.live_record_count != 0) return ::testing::AssertionFailure() << what << " live";
        if (head.published_record_count != 0) return ::testing::AssertionFailure() << what << " published_records";
        if (head.published_buffer_count != 0) return ::testing::AssertionFailure() << what << " published_buffers";
        return ::testing::AssertionSuccess();
    }
};

TEST(SwimlaneDeviceResetTest, EveryHeadOnTheGridStartsAtZero) {
    DeviceResetFixture fx;
    fx.dirty_every_head(7);

    // Two cores, which is fewer than the grid and fewer than the platform
    // maximum: the device acquisition loop visits only these.
    chip_swimlane_aicpu_init(/*worker_count=*/2);

    for (int i = 0; i < PLATFORM_MAX_CORES; i++) {
        EXPECT_TRUE(DeviceResetFixture::all_zero(get_perf_buffer_state(fx.shm(), i)->head, "task head"))
            << " at core " << i;
        EXPECT_TRUE(DeviceResetFixture::all_zero(get_aicore_buffer_state(fx.shm(), i)->head, "aicore head"))
            << " at core " << i;
    }
    for (int t = 0; t < PLATFORM_MAX_AICPU_THREADS; t++) {
        EXPECT_TRUE(DeviceResetFixture::all_zero(get_sched_phase_buffer_state(fx.shm(), t)->head, "sched head"))
            << " at thread " << t;
        EXPECT_TRUE(DeviceResetFixture::all_zero(get_orch_phase_buffer_state(fx.shm(), t)->head, "orch head"))
            << " at thread " << t;
    }
}

TEST(SwimlaneDeviceResetTest, EveryLevelClearsTheWholeGrid) {
    // The region is laid out at `PLATFORM_MAX_CORES` and
    // `PLATFORM_MAX_AICPU_THREADS` whatever level sized it, so the full-grid
    // clear is in bounds at every level -- including the two levels below
    // SCHED_PHASES, where the phase pools exist in the region but have no
    // device initializer of their own. A level-dependent layout would make
    // that clear write past the end instead, which is why each level is
    // driven rather than argued about.
    for (const ChipSwimlaneLevel level :
         {ChipSwimlaneLevel::TASK_TIMING, ChipSwimlaneLevel::SCHEDULE_TIMING, ChipSwimlaneLevel::SCHED_PHASES,
          ChipSwimlaneLevel::ORCH_PHASES}) {
        DeviceResetFixture fx(level);
        fx.dirty_every_head(13);
        chip_swimlane_aicpu_init(/*worker_count=*/2);
        EXPECT_TRUE(fx.whole_grid_is_zero()) << " at level " << static_cast<uint32_t>(level);
    }
}

TEST(SwimlaneDeviceResetTest, ANarrowerRunAtALowerLevelInheritsNothingFromAWiderOne) {
    // The 4-to-2 transition on one region, and the level has to actually move:
    // `begin_run` is the production arm path, and `publish_run_config` inside
    // it is what writes the device's single level word. Changing only the core
    // count would leave the device latched at the higher level and prove
    // nothing about a run that collects less than the one before it.
    DeviceResetFixture fx(ChipSwimlaneLevel::ORCH_PHASES, /*num_aicore=*/4, /*aicpu_thread_num=*/2);
    ChipSwimlaneDataHeader *header = get_chip_swimlane_header(fx.shm());

    fx.collector.begin_run(/*output_prefix=*/"", ChipSwimlaneLevel::ORCH_PHASES);
    ASSERT_EQ(header->chip_swimlane_level, static_cast<uint32_t>(ChipSwimlaneLevel::ORCH_PHASES));
    chip_swimlane_aicpu_init(/*worker_count=*/4);

    // What the wide, high-level run leaves behind: dirty heads on every core
    // and both phase pools, and a buffer each core still owns.
    uint64_t owned[PLATFORM_MAX_CORES];
    for (int i = 0; i < PLATFORM_MAX_CORES; i++) {
        owned[i] = get_perf_buffer_state(fx.shm(), i)->head.current_buf_ptr;
    }
    fx.dirty_every_head(17);

    fx.collector.begin_run(/*output_prefix=*/"", ChipSwimlaneLevel::SCHEDULE_TIMING);
    ASSERT_EQ(header->chip_swimlane_level, static_cast<uint32_t>(ChipSwimlaneLevel::SCHEDULE_TIMING))
        << "the device's level word did not follow the run that armed it";

    chip_swimlane_aicpu_init(/*worker_count=*/1);

    EXPECT_TRUE(fx.whole_grid_is_zero()) << " after narrowing from four cores at level 4 to one at level 2";
    for (int i = 0; i < PLATFORM_MAX_CORES; i++) {
        EXPECT_EQ(get_perf_buffer_state(fx.shm(), i)->head.current_buf_ptr, owned[i])
            << "the transition stranded or replaced a buffer core " << i << " still owned";
    }
}

TEST(SwimlaneDeviceResetTest, AHeadWithNoFreeBufferIsStillCleared) {
    DeviceResetFixture fx;
    fx.dirty_every_head(11);

    // Core 0's two pools have nothing to acquire, so init takes the branch that
    // leaves `current_buf_ptr` at zero -- the branch that never reached the old
    // per-acquisition clear.
    ChipSwimlaneAicpuTaskPool *task = get_perf_buffer_state(fx.shm(), 0);
    ChipSwimlaneAicoreTaskPool *aicore = get_aicore_buffer_state(fx.shm(), 0);
    task->head.current_buf_ptr = 0;
    task->free_queue.head = task->free_queue.tail;
    aicore->head.current_buf_ptr = 0;
    aicore->free_queue.head = aicore->free_queue.tail;

    chip_swimlane_aicpu_init(/*worker_count=*/2);

    EXPECT_TRUE(DeviceResetFixture::all_zero(task->head, "dry task head"));
    EXPECT_TRUE(DeviceResetFixture::all_zero(aicore->head, "dry aicore head"));
}

TEST(SwimlaneDeviceResetTest, RetainedBufferOwnershipSurvivesTheReset) {
    DeviceResetFixture fx;
    chip_swimlane_aicpu_init(/*worker_count=*/2);

    // What a run that could not hand its buffer over leaves behind: a head
    // still naming it, which only a successful enqueue ever releases.
    ChipSwimlaneAicpuTaskPool *task = get_perf_buffer_state(fx.shm(), 0);
    const uint64_t retained_ptr = task->head.current_buf_ptr;
    ASSERT_NE(retained_ptr, 0u);
    const uint32_t free_head = task->free_queue.head;
    DeviceResetFixture::dirty(task->head, 5);

    chip_swimlane_aicpu_init(/*worker_count=*/2);

    EXPECT_EQ(task->head.current_buf_ptr, retained_ptr) << "the reset must not strand a retained buffer";
    EXPECT_EQ(task->free_queue.head, free_head) << "a retained buffer is reused in place, not replaced";
    EXPECT_TRUE(DeviceResetFixture::all_zero(task->head, "retained task head"));
}

// ---------------------------------------------------------------------------
// A run's metadata is the run's
// ---------------------------------------------------------------------------

/** A retaining collector brought up the way the runner brings one up. */
struct AdmissionFixture {
    ChipSwimlaneCollector collector;
    fs::path root;
    std::string dir;

    explicit AdmissionFixture(const char *name) :
        root(private_artifact_root(name)),
        dir(root.string()) {
        std::error_code ec;
        fs::remove_all(root, ec);
        collector.configure_retained_runs(true, simpler::dfx::runs::kDefaultBudgetBytes);
        EXPECT_EQ(collector.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, joined_alloc, nullptr, joined_free), 0);
        collector.start(joined_thread);
    }
    ~AdmissionFixture() {
        collector.finish_retained_runs();
        collector.stop();
        collector.finalize(nullptr, joined_free);
        std::error_code ec;
        fs::remove_all(root, ec);
    }
};

TEST(SwimlaneRunLocalMetadataTest, EachRunKeepsTheCoreTypesItWasAdmittedWith) {
    AdmissionFixture fx("joined-metadata");
    const CoreType first[] = {CoreType::AIC, CoreType::AIV};
    const CoreType second[] = {CoreType::AIV, CoreType::AIC};

    ASSERT_TRUE(fx.collector.run_begin(1, fx.dir, ChipSwimlaneLevel::TASK_TIMING, first, 2, false));
    // The successor is admitted while the predecessor is still open, which is
    // the whole point: with two buckets live the resident copy a close would
    // have read is already the successor's.
    ASSERT_TRUE(fx.collector.run_begin(2, fx.dir, ChipSwimlaneLevel::TASK_TIMING, second, 2, true));

    const auto one = fx.collector.pending_metadata_for_test(1);
    const auto two = fx.collector.pending_metadata_for_test(2);
    ASSERT_EQ(one.core_types.size(), 2u);
    ASSERT_EQ(two.core_types.size(), 2u);
    EXPECT_EQ(one.core_types[0], CoreType::AIC);
    EXPECT_EQ(one.core_types[1], CoreType::AIV);
    EXPECT_EQ(two.core_types[0], CoreType::AIV) << "the successor's admission must not rewrite the predecessor's";
    EXPECT_EQ(two.core_types[1], CoreType::AIC);
    EXPECT_FALSE(one.host_orchestrated);
    EXPECT_TRUE(two.host_orchestrated);
}

TEST(SwimlaneRunLocalMetadataTest, AResidentSetterDoesNotReachAnAdmittedRun) {
    AdmissionFixture fx("joined-metadata-resident");
    const CoreType mine[] = {CoreType::AIC};
    ASSERT_TRUE(fx.collector.run_begin(3, fx.dir, ChipSwimlaneLevel::TASK_TIMING, mine, 1, false));

    // What a successor's launch does to the resident copy. The admitted run's
    // bucket must not follow it.
    const CoreType theirs[] = {CoreType::AIV};
    fx.collector.set_core_types(theirs, 1);

    const auto held = fx.collector.pending_metadata_for_test(3);
    ASSERT_EQ(held.core_types.size(), 1u);
    EXPECT_EQ(held.core_types[0], CoreType::AIC);
}

/**
 * A retaining collector whose host budget is the fixed reservation plus a
 * couple of bytes: readiness succeeds, and the first caller-sized copy does
 * not. That is the only way to reach the admission-time refusal without
 * stubbing the budget.
 */
/**
 * A retaining collector opened at the smallest budget the host budget accepts:
 * the fixed reservation plus the minimum working set, so every byte of
 * headroom is known. The core-type copy below is then sized past that
 * headroom, which is the only way to reach the admission-time refusal through
 * the production call rather than by stubbing the budget.
 */
struct TightBudgetFixture {
    ChipSwimlaneCollector collector;
    fs::path root;
    std::string dir;

    explicit TightBudgetFixture(const char *name) :
        root(private_artifact_root(name)),
        dir(root.string()) {
        std::error_code ec;
        fs::remove_all(root, ec);
        EXPECT_EQ(collector.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, joined_alloc, nullptr, joined_free), 0);
        // The reservation is known only once the collector is sized, so the
        // budget is chosen here rather than at construction.
        collector.configure_retained_runs(
            true, collector.retained_fixed_overhead_for_test() + simpler::dfx::runs::kMinWorkingSetBytes
        );
        collector.start(joined_thread);
    }
    ~TightBudgetFixture() {
        collector.finish_retained_runs();
        collector.stop();
        collector.finalize(nullptr, joined_free);
        std::error_code ec;
        fs::remove_all(root, ec);
    }
};

/**
 * The text of the artifact a run published, once the writer has been waited
 * for.
 *
 * `run_close` only hands the bucket over; from there it is the writer's, and
 * the slot returns to `Free` as soon as the file exists. So a closed run's
 * verdict is read from that file -- which is stable and is what a consumer
 * sees -- rather than from a bucket that may already have been retired.
 * `flush_retained_runs` is the production wait for exactly that publication.
 */
std::string published_artifact(ChipSwimlaneCollector &collector, const fs::path &root, uint64_t epoch) {
    std::string error;
    const bool flushed = collector.flush_retained_runs(/*timeout_ms=*/8000, &error);
    const std::string name = "records_e" + std::to_string(epoch) + ".json";
    std::error_code ec;
    for (const auto &entry : fs::directory_iterator(root, ec)) {
        if (!entry.is_directory()) continue;
        const fs::path candidate = entry.path() / name;
        if (!fs::exists(candidate, ec)) continue;
        std::ifstream in(candidate);
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    }
    ADD_FAILURE() << "run " << epoch << " published no artifact under " << root << "; flush returned " << flushed
                  << " (" << error << ")";
    return {};
}

TEST(SwimlaneRunLocalMetadataTest, ARefusedCoreTypeCopyIsStillIncompleteAtClose) {
    TightBudgetFixture fx("joined-metadata-budget");
    // Larger than the whole working set, so the charge is refused. The count is
    // chosen to defeat the budget rather than to model a real shape -- what is
    // under test is what a refusal leaves behind, which is the same whatever
    // made the charge fail.
    const std::vector<CoreType> types(simpler::dfx::runs::kMinWorkingSetBytes / sizeof(CoreType) + 1024, CoreType::AIC);

    ASSERT_TRUE(fx.collector.run_begin(
        31, fx.dir, ChipSwimlaneLevel::TASK_TIMING, types.data(), static_cast<int>(types.size()), false
    ));
    const auto held = fx.collector.pending_metadata_for_test(31);
    ASSERT_TRUE(held.found);
    EXPECT_TRUE(held.core_types.empty()) << "a refusal keeps no bytes the budget said it could not pay for";

    // The close's own charge is a different allocation, and the budget has room
    // for it by then -- which is exactly how the completeness flag used to come
    // out true for a run whose core types were dropped.
    fx.collector.run_close(31, /*bank_index=*/0, /*device_execution_complete=*/false);

    const std::string artifact = published_artifact(fx.collector, fx.root, 31);
    EXPECT_NE(artifact.find("\"metadata_complete\": false"), std::string::npos)
        << "a run that lost its core types at admission may not publish complete metadata: " << artifact;
}

TEST(SwimlaneRunLocalMetadataTest, AnAdmittedCoreTypeCopyPublishesCompleteMetadata) {
    AdmissionFixture fx("joined-metadata-complete");
    const CoreType types[] = {CoreType::AIC};
    ASSERT_TRUE(fx.collector.run_begin(33, fx.dir, ChipSwimlaneLevel::TASK_TIMING, types, 1, false));
    fx.collector.run_close(33, /*bank_index=*/0, /*device_execution_complete=*/false);

    const std::string artifact = published_artifact(fx.collector, fx.root, 33);
    EXPECT_NE(artifact.find("\"metadata_complete\": true"), std::string::npos)
        << "the ordinary path must not be made pessimistic by the new term: " << artifact;
    // The admitted copy is the one the file carries, so completeness is not the
    // only thing the close got right.
    EXPECT_NE(artifact.find("\"core_types\": [\"aic\"]"), std::string::npos) << artifact;
}

// ---------------------------------------------------------------------------
// One permanent reporting policy, decided once
// ---------------------------------------------------------------------------

/** A collector that has never retained: the default, serial shape. */
struct DefaultFixture {
    ChipSwimlaneCollector collector;

    DefaultFixture() {
        EXPECT_EQ(collector.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, joined_alloc, nullptr, joined_free), 0);
    }
    ~DefaultFixture() { collector.finalize(nullptr, joined_free); }
};

TEST(SwimlaneLossPolicyTest, ANeverRetainedCollectorReportsNothingSticky) {
    DefaultFixture fx;
    ASSERT_FALSE(fx.collector.ever_retained());

    // Every anonymous path: a descriptor that did not validate, one naming a
    // run this collector does not hold, and one with no identity at all.
    fx.collector.on_handoff_retired(unidentified_handoff());
    fx.collector.on_handoff_retired(retired_handoff(99, 4));
    fx.collector.on_handoff_retired(retired_handoff(0, 2));

    EXPECT_EQ(fx.collector.unattributable_handoffs_for_test(), 3u) << "the loss is still counted and logged";
    std::string error;
    EXPECT_TRUE(fx.collector.flush_retained_runs(0, &error))
        << "a collector that never retained answers for its losses through its own run's artifact: " << error;
}

TEST(SwimlaneLossPolicyTest, FirstRetentionDiscardsThePreRetentionCountOnce) {
    AdmissionFixture fx("joined-policy-boundary");
    // Retention setup is lazy: configuring and starting is not retaining, and
    // until the first admission completes it this collector is still answering
    // for its losses the default way.
    ASSERT_FALSE(fx.collector.ever_retained());
    fx.collector.on_handoff_retired(retired_handoff(101, 9));
    ASSERT_EQ(fx.collector.unattributable_handoffs_for_test(), 1u);
    {
        std::string error;
        ASSERT_TRUE(fx.collector.flush_retained_runs(0, &error)) << error;
    }

    // The first admission is what completes setup and moves the policy.
    ASSERT_TRUE(fx.collector.run_begin(6, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));
    ASSERT_TRUE(fx.collector.ever_retained());
    EXPECT_EQ(fx.collector.unattributable_handoffs_for_test(), 0u)
        << "a loss answered by a default run's own artifact must not downgrade a retained one";
    {
        std::string error;
        EXPECT_TRUE(fx.collector.flush_retained_runs(0, &error)) << error;
    }

    // Past the boundary the same receipt is permanent and reportable.
    fx.collector.on_handoff_retired(retired_handoff(404, 6));
    EXPECT_EQ(fx.collector.unattributable_handoffs_for_test(), 1u);

    // A second admission must not clear it again: one boundary, or a
    // disable-and-re-enable cycle would erase a retained failure.
    ASSERT_TRUE(fx.collector.run_begin(7, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));
    EXPECT_EQ(fx.collector.unattributable_handoffs_for_test(), 1u);
    std::string error;
    EXPECT_FALSE(fx.collector.flush_retained_runs(0, &error));
    EXPECT_NE(error.find("unattributed"), std::string::npos) << error;
}

TEST(SwimlaneLossPolicyTest, APostActivationLossSurvivesRetentionBeingTurnedOff) {
    AdmissionFixture fx("joined-policy-persists");
    ASSERT_TRUE(fx.collector.run_begin(9, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));
    ASSERT_TRUE(fx.collector.ever_retained());
    fx.collector.on_handoff_retired(retired_handoff(808, 3));

    // What a collector rebuild does: retention reconfigured off, then the
    // release the runner's `finalize_collectors()` performs. `retained_ready_`
    // goes false and the wait has nothing left to do -- the sticky report must
    // still be made.
    fx.collector.configure_retained_runs(false, simpler::dfx::runs::kDefaultBudgetBytes);
    EXPECT_EQ(fx.collector.finalize(nullptr, joined_free), 0);
    ASSERT_FALSE(fx.collector.retains_runs());
    ASSERT_TRUE(fx.collector.ever_retained());

    std::string error;
    EXPECT_FALSE(fx.collector.flush_retained_runs(0, &error)) << "a retained failure outlives the retention";
    EXPECT_FALSE(error.empty());

    // And a default-mode receipt after the boundary is still covered, which is
    // the disclosed cost of one permanent policy.
    fx.collector.on_handoff_retired(unidentified_handoff());
    EXPECT_EQ(fx.collector.unattributable_handoffs_for_test(), 2u);
}

TEST(SwimlaneLossPolicyTest, TheFlushEntryReportsAfterRetentionIsDisabled) {
    // The collector-side half of the public-boundary policy, which both runner
    // bases now gate on `ever_retained()`. Driven here rather than through a
    // runner because the entry point is what each of them calls.
    AdmissionFixture fx("joined-policy-entry");
    ASSERT_TRUE(fx.collector.run_begin(13, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));
    fx.collector.on_handoff_retired(retired_handoff(606, 2));

    fx.collector.configure_retained_runs(false, simpler::dfx::runs::kDefaultBudgetBytes);
    EXPECT_EQ(fx.collector.finalize(nullptr, joined_free), 0);
    ASSERT_FALSE(fx.collector.retains_runs());

    std::string error;
    EXPECT_FALSE(fx.collector.flush_retained_runs(0, &error))
        << "a runner that consults this entry after disabling retention must still see the failure";
    EXPECT_FALSE(error.empty());
}

TEST(SwimlaneLossPolicyTest, AMismatchedDescriptorIsChargedAndSticky) {
    AdmissionFixture fx("joined-policy-mismatch");
    ASSERT_TRUE(fx.collector.run_begin(11, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));
    ASSERT_TRUE(fx.collector.ever_retained());

    // The third anonymous-counter path. Unlike the two retirement paths it is
    // reached only while the collector is ready to retain, so it is tested
    // there; what it shares with them is the policy gate, not the branch.
    //
    // The descriptor and the payload are written by one producer while it owns
    // the buffer, so a disagreement is a corrupt hand-off and the records are
    // declined rather than placed from either side.
    ChipSwimlaneAicpuTaskBuffer payload{};
    payload.run_epoch = 11;
    payload.local_seq = 1;
    payload.count = 3;

    ReadyBufferInfo info{};
    info.type = ProfBufferType::AICPU_TASK;
    info.index = 0;
    info.run_epoch = 11;
    info.buffer_seq = 1;
    info.record_count = 99;  // disagrees with the payload
    info.host_buffer_ptr = &payload;
    info.dev_buffer_ptr = &payload;
    fx.collector.on_buffer_collected(info, 0);

    EXPECT_EQ(fx.collector.unattributable_handoffs_for_test(), 1u);
    std::string error;
    EXPECT_FALSE(fx.collector.flush_retained_runs(0, &error));
    EXPECT_NE(error.find("disagreed"), std::string::npos) << error;
}

TEST(SwimlaneLossPolicyTest, ABudgetRefusedAdmissionLeavesThePolicyWhereItWas) {
    // The budget is taken before the writer is prepared or a directory
    // reserved, so `run_begin` returns here without ever reaching
    // `start_run_writer`. That makes this an admission refusal, not a writer
    // that failed to start -- the case below is the latter -- and what it
    // covers is that a refusal this early leaves the permanent policy alone.
    AdmissionFixture fx("joined-policy-startup");
    fx.collector.configure_retained_runs(/*retain_across_runs=*/true, /*budget_bytes=*/1024);
    ASSERT_FALSE(fx.collector.run_begin(15, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));
    ASSERT_FALSE(fx.collector.ever_retained()) << "a refused admission must not move the permanent policy";

    // So a loss here is still the default kind: counted, logged, and answered
    // by the run's own artifact rather than made sticky.
    fx.collector.on_handoff_retired(retired_handoff(505, 2));
    EXPECT_EQ(fx.collector.unattributable_handoffs_for_test(), 1u);
    std::string error;
    EXPECT_TRUE(fx.collector.flush_retained_runs(0, &error)) << error;

    // The retry is what completes setup, and it discards that count once.
    fx.collector.configure_retained_runs(/*retain_across_runs=*/true, simpler::dfx::runs::kDefaultBudgetBytes);
    ASSERT_TRUE(fx.collector.run_begin(16, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));
    EXPECT_TRUE(fx.collector.ever_retained());
    EXPECT_EQ(fx.collector.unattributable_handoffs_for_test(), 0u);
}

TEST(SwimlaneLossPolicyTest, AWriterThatCannotBeSpawnedLeavesNothingRetained) {
    // The reason the latch sits *after* `start_run_writer`: a spawn that throws
    // leaves this collector having retained nothing, and a policy that had
    // already moved could not be taken back down with the rest of the
    // preparation. Nothing a caller controls can make a thread fail to spawn,
    // so the writer's one construction point takes a substitute here -- the
    // rollback that runs is the production one, reached through the production
    // call.
    AdmissionFixture fx("joined-policy-writer");
    fx.collector.set_writer_thread_factory_for_test([](std::function<void()>) -> std::thread {
        throw std::runtime_error("writer spawn refused");
    });

    EXPECT_THROW(
        (void)fx.collector.run_begin(19, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false), std::runtime_error
    );
    EXPECT_FALSE(fx.collector.ever_retained()) << "a writer that never started must not move the permanent policy";

    // And the loss policy is still the default one, which is what the latch
    // staying down has to mean rather than only what it says.
    fx.collector.on_handoff_retired(unidentified_handoff());
    EXPECT_EQ(fx.collector.unattributable_handoffs_for_test(), 1u);
    std::string error;
    EXPECT_TRUE(fx.collector.flush_retained_runs(0, &error)) << error;

    // The rollback was complete, not merely begun: with the substitute gone the
    // same collector prepares, and the count it carried is discarded once, as
    // it would have been had the first attempt succeeded.
    fx.collector.set_writer_thread_factory_for_test(nullptr);
    ASSERT_TRUE(fx.collector.run_begin(20, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));
    EXPECT_TRUE(fx.collector.ever_retained());
    EXPECT_EQ(fx.collector.unattributable_handoffs_for_test(), 0u);
}

TEST(SwimlaneLossPolicyTest, AChargeParkedAcrossTheFirstRetentionIsCountedOnItsOwnSide) {
    // Both sides of the transition, with the side each charge landed on fixed
    // rather than left to the scheduler -- so a charge lost in the window is a
    // failure here instead of an interleaving that happened not to occur.
    //
    // `set_pre_charge_hook_for_test` is the production charge path's own seam:
    // it runs just before a charge takes the accounting lock, holding no lock
    // itself, which is exactly the window the activation has to survive.
    AdmissionFixture fx("joined-policy-parked");

    // Side one: a charge that completes entirely before the boundary. It is
    // counted now and must be discarded by the activation.
    fx.collector.on_handoff_retired(unidentified_handoff());
    ASSERT_EQ(fx.collector.unattributable_handoffs_for_test(), 1u);
    ASSERT_FALSE(fx.collector.ever_retained());

    // Side two: a charge parked in the window, released only after the
    // activation has run. An identified receipt naming a run this collector
    // does not hold reaches the same anonymous counter, and it is the path the
    // hook sits on.
    std::promise<void> parked;
    std::promise<void> release;
    std::shared_future<void> released(release.get_future());
    std::atomic<bool> armed{true};
    std::once_flag release_once;
    auto let_go = [&release, &release_once] {
        std::call_once(release_once, [&release] {
            release.set_value();
        });
    };
    fx.collector.set_pre_charge_hook_for_test([&parked, &released, &armed] {
        if (!armed.exchange(false, std::memory_order_acq_rel)) return;
        parked.set_value();
        released.wait();
    });

    std::thread charger([&fx] {
        fx.collector.on_handoff_retired(retired_handoff(909, 4));
    });
    // Every exit from here -- a failing ASSERT, a throw, or the ordinary end of
    // the case -- has to let the parked thread go and join it. A `std::thread`
    // destroyed while joinable calls `std::terminate`, which would turn one
    // case's failure into the whole binary's, and a parked thread with no
    // releaser would hang the target instead of reporting.
    auto unpark = RAIIScopeGuard([&let_go, &charger, &fx] {
        let_go();
        if (charger.joinable()) charger.join();
        fx.collector.set_pre_charge_hook_for_test(nullptr);
    });

    ASSERT_EQ(parked.get_future().wait_for(std::chrono::seconds(10)), std::future_status::ready)
        << "the charge never reached the window the activation has to survive";

    ASSERT_TRUE(fx.collector.run_begin(17, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));
    ASSERT_TRUE(fx.collector.ever_retained());

    // The ordinary path takes the same route as every failing one, so the
    // assertions below read a settled collector either way.
    let_go();
    charger.join();
    fx.collector.set_pre_charge_hook_for_test(nullptr);
    unpark.dismiss();

    // Exactly one: the pre-boundary charge was discarded and the parked one was
    // kept. Zero would mean the parked charge was lost in the window; two would
    // mean the activation's clear missed the charge that preceded it.
    EXPECT_EQ(fx.collector.unattributable_handoffs_for_test(), 1u);
    std::string error;
    EXPECT_FALSE(fx.collector.flush_retained_runs(0, &error)) << "the surviving charge was not reportable";
    EXPECT_FALSE(error.empty());

    // One boundary, not one per run.
    ASSERT_TRUE(fx.collector.run_begin(18, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));
    EXPECT_EQ(fx.collector.unattributable_handoffs_for_test(), 1u)
        << "a later admission cleared a charge the first one made permanent";
}

// ---------------------------------------------------------------------------
// The capacity term the backend's pair gate ends in
// ---------------------------------------------------------------------------
//
// `DeviceRunnerBase::can_join_diagnostic_run` decides a pair on four terms --
// execution mode, matching swimlane level with no other channel, an unchanged
// collector shape, and capacity -- and returns the last of them,
// `can_admit_retained_run()`, as its answer. That call is production code and
// is driven directly here. The three terms above it are refused by the lane
// before the backend is consulted and are covered in
// `test_chip_run_lane_joined_launch.cpp`; the composition of all four has no
// unit-level coverage, because the predicate is a member of a runner that
// cpput does not construct.

TEST(SwimlanePairCapacityTest, ACollectorThatDoesNotRetainAdmitsNoPair) {
    DefaultFixture fx;
    EXPECT_FALSE(fx.collector.can_admit_retained_run())
        << "a serial collector has no second bucket to put a successor in";
}

TEST(SwimlanePairCapacityTest, TwoOpenRunsFillTheTableAndRefuseAThird) {
    AdmissionFixture fx("joined-capacity-full");
    // The table holds exactly the pair this feature admits, which is where the
    // depth-two restriction comes from rather than being a separate rule.
    ASSERT_EQ(simpler::dfx::runs::kMaxOpenEpochs, 2u);

    // Retention setup is lazy -- configuring and starting a collector is not
    // retaining, and the first admission is what completes it. So until a run
    // is open there is no pair for the gate to answer about, and the first run
    // of a sequence reaches the device through the ordinary front.
    EXPECT_FALSE(fx.collector.can_admit_retained_run());

    ASSERT_TRUE(fx.collector.run_begin(41, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));
    // ASSERT rather than EXPECT before every admission in this file: `run_begin`
    // has no refusal for a full table and waits on the writer instead, so a case
    // that called it after this failed would hang rather than report.
    ASSERT_TRUE(fx.collector.can_admit_retained_run()) << "one open run still leaves the successor's bucket";
    ASSERT_TRUE(fx.collector.run_begin(42, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));

    // A full table is why the pair is decided by this query rather than by
    // trying the admission: `run_begin` has no refusal for a full table, it
    // waits on the writer to free a slot. A third run must therefore be
    // declined here, before anything is submitted -- so the call is not made.
    EXPECT_FALSE(fx.collector.can_admit_retained_run())
        << "a full table must decline the next pair before it is admitted";

    fx.collector.run_close(41, /*bank_index=*/0, /*device_execution_complete=*/false);
    fx.collector.run_close(42, /*bank_index=*/1, /*device_execution_complete=*/false);
}

TEST(SwimlanePairCapacityTest, APublishedRunGivesItsSlotBackToTheNextPair) {
    AdmissionFixture fx("joined-capacity-refill");
    ASSERT_TRUE(fx.collector.run_begin(51, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));
    ASSERT_TRUE(fx.collector.run_begin(52, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));
    ASSERT_FALSE(fx.collector.can_admit_retained_run());

    // Sustained operation is what a depth-two queue does for a whole workload:
    // the predecessor publishes while its successor is still open, and the slot
    // it gives back is what the run after that is admitted into. A table that
    // only emptied when both runs had closed would serialize every third run.
    fx.collector.run_close(51, /*bank_index=*/0, /*device_execution_complete=*/false);
    EXPECT_FALSE(published_artifact(fx.collector, fx.root, 51).empty());
    ASSERT_TRUE(fx.collector.can_admit_retained_run()) << "the published run's bucket is free again";

    ASSERT_TRUE(fx.collector.run_begin(53, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false))
        << "the next successor is admitted beside the run that is still open";
    EXPECT_FALSE(fx.collector.can_admit_retained_run());

    fx.collector.run_close(52, /*bank_index=*/1, /*device_execution_complete=*/false);
    fx.collector.run_close(53, /*bank_index=*/0, /*device_execution_complete=*/false);
    EXPECT_FALSE(published_artifact(fx.collector, fx.root, 53).empty())
        << "each run of the sustained sequence publishes its own file";
}

}  // namespace
