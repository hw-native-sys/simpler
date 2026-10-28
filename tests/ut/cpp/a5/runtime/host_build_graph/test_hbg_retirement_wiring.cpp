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

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "common/args_dump.h"
#include "aicore_lifecycle.h"
#include "aicpu/platform_regs.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "runtime.h"
#include "scheduler/scheduler_context.h"

// The register-layer tests exercise the real ACK, close, read-back, drain and
// release sequence. This seam verifies that both HBG owners pass the real per-core
// gate to that protocol, including their failure and overlapping-exit paths.
namespace {
struct RetireCall {
    std::vector<uint64_t> addrs;
    std::vector<AicoreTeardownControl *> gates;
};

std::vector<RetireCall> retire_calls;
std::vector<uint64_t> exit_signals;
int32_t retirement_result = 0;
std::mutex retire_mutex;
}  // namespace

uint64_t platform_aicore_exit_deadline() { return std::numeric_limits<uint64_t>::max(); }

int32_t platform_retire_aicore_group(const AicoreExitTarget *targets, size_t count, uint64_t, bool *released) {
    RetireCall call;
    for (size_t i = 0; i < count; ++i) {
        call.addrs.push_back(targets[i].reg_addr);
        call.gates.push_back(targets[i].teardown);
        if (released != nullptr) released[i] = retirement_result == 0;
    }
    {
        std::lock_guard<std::mutex> lock(retire_mutex);
        retire_calls.push_back(std::move(call));
    }
    return retirement_result;
}

void platform_signal_aicore_exit(uint64_t reg_addr) { exit_signals.push_back(reg_addr); }
void platform_init_aicore_regs(uint64_t) {}
void write_reg(uint64_t, RegId, uint64_t) {}
// The PMU teardown reads a core's control word back before restoring it, to
// record whether this run's enable survived its tasks. No case here programs a
// PMU window, so there is nothing for it to observe.
uint64_t read_reg(uint64_t, RegId) { return 0; }
extern "C" bool is_dump_args_enabled() { return false; }
// The shutdown partition settles its dump lane; this target links no args-dump
// producer, and the stub above keeps either from being reached.
void dump_args_flush(int) {}
void dump_args_close_run(int) {}
void dump_args_init(int, DumpArgsLevel) {}

class SchedulerContextTestPeer {
public:
    static void seed(SchedulerContext &context, const std::array<uint64_t, 3> &addrs) {
        context.cores_total_num_ = 3;
        context.core_trackers_[0].init(1);
        context.core_trackers_[0].set_cluster(0, 0, 1, 2);
        for (int i = 0; i < 3; ++i) {
            context.core_exec_states_[i].reg_addr = addrs[i];
            context.core_retired_[i].store(false, std::memory_order_relaxed);
        }
    }
    static void emergency(SchedulerContext &context, Runtime *runtime) { context.emergency_shutdown(runtime); }
};

class AicoreLifecycleTestPeer {
public:
    static void seed(AicoreLifecycle &lifecycle, const std::array<uint64_t, 3> &addrs, int32_t threads = 1) {
        lifecycle.core_count_ = 3;
        lifecycle.aicpu_thread_num_ = threads;
        lifecycle.retired_core_count_.store(0, std::memory_order_release);
        for (int i = 0; i < 3; ++i)
            lifecycle.cores_[i].reg_addr = addrs[i];
    }

    static int32_t retired_cores(const AicoreLifecycle &lifecycle) {
        return lifecycle.retired_core_count_.load(std::memory_order_acquire);
    }
    static int32_t core_count(const AicoreLifecycle &lifecycle) { return lifecycle.core_count_; }

    static bool collect_pmu(AicoreLifecycle &lifecycle, Runtime *runtime, uint64_t *missing) {
        return lifecycle.collect_run_pmu_records(runtime, missing);
    }
};

/**
 * The smallest scheduler state `collect_run_pmu_records` can read: one
 * executable single-subtask task, its metadata, its PMU cell and the graph
 * storage its identity comes from. Laid out by hand because this target links
 * no runner to build one.
 */
struct MinimalSchedulerState {
    static constexpr uint64_t kMetadataOffset = sizeof(SchedulerWorkerContext);
    static constexpr uint64_t kCellsOffset = kMetadataOffset + 64;
    static constexpr uint64_t kTotalSize = kCellsOffset + sizeof(SchedulerTaskPmuCell);

    std::vector<uint8_t> state{std::vector<uint8_t>(kTotalSize, 0)};
    std::vector<uint8_t> storage{std::vector<uint8_t>(SCHEDULER_GRAPH_TASK_STORAGE_STRIDE, 0)};

    SchedulerWorkerContext *context() { return reinterpret_cast<SchedulerWorkerContext *>(state.data()); }
    SchedulerTaskMetadata *metadata() {
        return reinterpret_cast<SchedulerTaskMetadata *>(state.data() + kMetadataOffset);
    }
    SchedulerTaskPmuCell *cell() { return reinterpret_cast<SchedulerTaskPmuCell *>(state.data() + kCellsOffset); }

    void publish(Runtime *runtime) {
        runtime->set_worker_count(1);
        SchedulerWorkerContext *ctx = context();
        ctx->scheduler_state_base_address = reinterpret_cast<uint64_t>(state.data());
        ctx->task_metadata_offset = kMetadataOffset;
        ctx->pmu_cells_offset = kCellsOffset;
        ctx->graph_task_count = 1;
        ctx->graph_storage_address = reinterpret_cast<uint64_t>(storage.data());
        // One executable task of one block with subtask slot 0 active.
        metadata()[0].flags = SCHEDULER_TASK_EXECUTABLE;
        metadata()[0].active_mask = LaneMask::aic();
        metadata()[0].logical_block_num = 1;
        metadata()[0].trace_index_base = 0;
        metadata()[0].kernel_ids[0] = 5;
        runtime->publish_scheduler_bootstrap(SCHEDULER_RUNTIME_MODE_RESIDENT_READY, reinterpret_cast<uint64_t>(ctx));
    }
};

namespace {
constexpr std::array<uint64_t, 3> kAddrs = {0x1000, 0x2000, 0x3000};

class HbgRetirementWiring : public testing::Test {
protected:
    void SetUp() override {
        retire_calls.clear();
        exit_signals.clear();
        retirement_result = 0;
        set_pmu_enabled(false);
    }

    void TearDown() override { set_pmu_enabled(false); }

    static void expect_group(const Runtime &runtime) {
        ASSERT_EQ(retire_calls.size(), 1u);
        EXPECT_EQ(retire_calls[0].addrs, std::vector<uint64_t>(kAddrs.begin(), kAddrs.end()));
        for (size_t i = 0; i < kAddrs.size(); ++i)
            EXPECT_EQ(retire_calls[0].gates[i], &runtime.dev.teardown_gates[i]);
    }
};

TEST_F(HbgRetirementWiring, SchedulerNormalShutdownClaimsEachCoreBeforeEmergency) {
    auto runtime = std::make_unique<Runtime>();
    auto scheduler = std::make_unique<SchedulerContext>();
    SchedulerContextTestPeer::seed(*scheduler, kAddrs);

    EXPECT_EQ(scheduler->shutdown(runtime.get(), 0), 0);
    SchedulerContextTestPeer::emergency(*scheduler, runtime.get());
    expect_group(*runtime);
}

TEST_F(HbgRetirementWiring, SchedulerClearsStaleGatesBeforeHandshake) {
    auto runtime = std::make_unique<Runtime>();
    runtime->set_worker_count(3);
    for (auto &gate : runtime->dev.teardown_gates) {
        gate.post_close_release = AICORE_POST_CLOSE_RELEASE;
        gate.exit_request = AICORE_EXIT_NONCE_1;
    }
    auto scheduler = std::make_unique<SchedulerContext>();

    ASSERT_EQ(scheduler->pre_handshake_init(runtime.get(), 1, 0), 0);
    for (size_t i = 0; i < kAddrs.size(); ++i) {
        EXPECT_EQ(runtime->dev.teardown_gates[i].post_close_release, 0u);
        EXPECT_EQ(runtime->dev.teardown_gates[i].exit_request, 0u);
    }
}

TEST_F(HbgRetirementWiring, SchedulerEmergencyClaimsEachCoreBeforeNormalShutdown) {
    auto runtime = std::make_unique<Runtime>();
    auto scheduler = std::make_unique<SchedulerContext>();
    SchedulerContextTestPeer::seed(*scheduler, kAddrs);

    SchedulerContextTestPeer::emergency(*scheduler, runtime.get());
    EXPECT_EQ(scheduler->shutdown(runtime.get(), 0), 0);
    expect_group(*runtime);
}

TEST_F(HbgRetirementWiring, ConcurrentNormalAndEmergencyShutdownRetireEachCoreOnce) {
    auto runtime = std::make_unique<Runtime>();
    auto scheduler = std::make_unique<SchedulerContext>();
    SchedulerContextTestPeer::seed(*scheduler, kAddrs);
    std::atomic<bool> go{false};
    auto wait_for_start = [&] {
        while (!go.load(std::memory_order_acquire))
            std::this_thread::yield();
    };
    std::thread normal([&] {
        wait_for_start();
        EXPECT_EQ(scheduler->shutdown(runtime.get(), 0), 0);
    });
    std::thread emergency([&] {
        wait_for_start();
        SchedulerContextTestPeer::emergency(*scheduler, runtime.get());
    });
    go.store(true, std::memory_order_release);
    normal.join();
    emergency.join();
    std::array<int, 3> claims{};
    size_t target_count = 0;
    for (const RetireCall &call : retire_calls) {
        for (size_t target = 0; target < call.addrs.size(); ++target) {
            ++target_count;
            for (size_t core = 0; core < kAddrs.size(); ++core) {
                if (call.addrs[target] != kAddrs[core]) continue;
                ++claims[core];
                EXPECT_EQ(call.gates[target], &runtime->dev.teardown_gates[core]);
            }
        }
    }
    EXPECT_EQ(target_count, kAddrs.size());
    EXPECT_EQ(claims, (std::array<int, 3>{1, 1, 1}));
}

TEST_F(HbgRetirementWiring, ResidentShutdownSignalsAndRetiresTheSameGatedPartition) {
    auto runtime = std::make_unique<Runtime>();
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::seed(lifecycle, kAddrs);

    lifecycle.signal_shutdown_partition(0);
    EXPECT_EQ(exit_signals, std::vector<uint64_t>(kAddrs.begin(), kAddrs.end()));
    EXPECT_EQ(lifecycle.finish_shutdown_partition(0, runtime.get()), 0);
    expect_group(*runtime);
}

TEST_F(HbgRetirementWiring, ResidentClearsStaleRequestAndReleaseBeforeHandshake) {
    auto runtime = std::make_unique<Runtime>();
    runtime->set_worker_count(3);
    for (auto &gate : runtime->dev.teardown_gates) {
        gate.post_close_release = AICORE_POST_CLOSE_RELEASE;
        gate.exit_request = AICORE_EXIT_NONCE_1;
    }
    AicoreLifecycle lifecycle;

    ASSERT_EQ(lifecycle.pre_handshake_init(runtime.get(), 1, 0), 0);
    for (size_t i = 0; i < kAddrs.size(); ++i) {
        EXPECT_EQ(runtime->dev.teardown_gates[i].post_close_release, 0u);
        EXPECT_EQ(runtime->dev.teardown_gates[i].exit_request, 0u);
    }
}

TEST_F(HbgRetirementWiring, FailedResidentStartupStillRetiresThroughPerCoreGates) {
    auto runtime = std::make_unique<Runtime>();
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::seed(lifecycle, kAddrs);
    retirement_result = -1;

    EXPECT_EQ(lifecycle.release_partition(runtime.get(), 0, false), -1);
    expect_group(*runtime);
}

// --- What gates reading a core's data, and what the run owed ---------------

/**
 * A partition that could not retire its cores contributes no proof.
 *
 * `finish_shutdown_partition` returning failure is the partition's own report;
 * what the settlement needs is per-core, because a core that never
 * acknowledged exit may still be executing and is neither a safe source to
 * read nor a safe target to restore registers on.
 */
TEST_F(HbgRetirementWiring, AFailedPartitionRetirementProvesNoCoreStopped) {
    auto runtime = std::make_unique<Runtime>();
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::seed(lifecycle, kAddrs);
    retirement_result = -1;

    EXPECT_NE(lifecycle.finish_shutdown_partition(0, runtime.get()), 0);
    EXPECT_EQ(AicoreLifecycleTestPeer::retired_cores(lifecycle), 0)
        << "a partition whose cores were not released was counted as proof that they had stopped";
}

TEST_F(HbgRetirementWiring, ARetiredPartitionProvesEveryCoreItAcknowledged) {
    auto runtime = std::make_unique<Runtime>();
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::seed(lifecycle, kAddrs);

    EXPECT_EQ(lifecycle.finish_shutdown_partition(0, runtime.get()), 0);
    EXPECT_EQ(AicoreLifecycleTestPeer::retired_cores(lifecycle), AicoreLifecycleTestPeer::core_count(lifecycle));
}

/**
 * The shape the gate cannot see: one partition fails to retire and the *other*
 * is the last participant to arrive. Arrival order says nothing about whether
 * the cores stopped, so the settlement must refuse on the aggregate rather
 * than run because the last arrival got there.
 */
TEST_F(HbgRetirementWiring, ALastArrivingParticipantDoesNotSettleAnotherPartitionsLiveCores) {
    auto runtime = std::make_unique<Runtime>();
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::seed(lifecycle, kAddrs, /*threads=*/3);

    // Partition 0 cannot retire its core; partitions 1 and 2 can, and the last
    // of them is the thread that would close the completion gate.
    retirement_result = -1;
    EXPECT_NE(lifecycle.finish_shutdown_partition(0, runtime.get()), 0);
    retirement_result = 0;
    EXPECT_EQ(lifecycle.finish_shutdown_partition(1, runtime.get()), 0);
    EXPECT_EQ(lifecycle.finish_shutdown_partition(2, runtime.get()), 0);

    EXPECT_LT(AicoreLifecycleTestPeer::retired_cores(lifecycle), AicoreLifecycleTestPeer::core_count(lifecycle));
    EXPECT_FALSE(lifecycle.settle_run_diagnostics(runtime.get(), /*graph_completed=*/true))
        << "the run settled its diagnostics while a core had not acknowledged exit";
}

TEST_F(HbgRetirementWiring, AProvedRetirementWithNoOwnersEnabledSettles) {
    auto runtime = std::make_unique<Runtime>();
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::seed(lifecycle, kAddrs);

    ASSERT_EQ(lifecycle.finish_shutdown_partition(0, runtime.get()), 0);
    EXPECT_TRUE(lifecycle.settle_run_diagnostics(runtime.get(), /*graph_completed=*/true));
}

/**
 * A subtask the graph says executed and whose cell is absent is a lost
 * measurement, not a subtask that did not run.
 *
 * The coverage comes from the graph — every active subtask of every executable
 * task — precisely so the cells cannot vouch for themselves. A run reporting
 * the smaller set that did arrive would publish a complete-looking PMU result
 * for a run that lost one.
 */
TEST_F(HbgRetirementWiring, AnAbsentPmuCellIsAMissingMeasurementRatherThanAnUnexecutedSubtask) {
    auto runtime = std::make_unique<Runtime>();
    MinimalSchedulerState scheduler_state;
    scheduler_state.publish(runtime.get());
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::seed(lifecycle, kAddrs);

    uint64_t missing = 0;
    EXPECT_FALSE(AicoreLifecycleTestPeer::collect_pmu(lifecycle, runtime.get(), &missing));
    EXPECT_EQ(missing, 1u) << "an owed subtask with no staged cell was not counted as missing";
}

TEST_F(HbgRetirementWiring, AStagedPmuCellCoversTheSubtaskTheGraphOwed) {
    auto runtime = std::make_unique<Runtime>();
    MinimalSchedulerState scheduler_state;
    scheduler_state.publish(runtime.get());
    // The core that executed task 0 published its cell: the identity stamp is
    // the task id plus one, which is what tells it from a cell never written.
    scheduler_state.cell()->worker_id = 0;
    scheduler_state.cell()->published = 1;
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::seed(lifecycle, kAddrs);

    uint64_t missing = 0;
    EXPECT_TRUE(AicoreLifecycleTestPeer::collect_pmu(lifecycle, runtime.get(), &missing));
    EXPECT_EQ(missing, 0u);
}

/**
 * A proved stop with incomplete data still settles nothing — and is still the
 * run that may touch the registers.
 *
 * The two obligations are separate. Whether this run's measurements are all of
 * them is a question about the data; whether a core's counters may be put back
 * is a question about whether that core stopped. Tying the restore to the
 * first leaves a stopped core programmed for the next run, which inherits a PMU
 * it never asked for.
 */
TEST_F(HbgRetirementWiring, AProvedStopWithAMissingMeasurementSettlesNothingAndStillCleansUp) {
    auto runtime = std::make_unique<Runtime>();
    MinimalSchedulerState scheduler_state;
    scheduler_state.publish(runtime.get());
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::seed(lifecycle, kAddrs);
    set_pmu_enabled(true);

    ASSERT_EQ(lifecycle.finish_shutdown_partition(0, runtime.get()), 0);
    ASSERT_TRUE(lifecycle.cores_retired());
    // The graph completed, so every active subtask owed a cell; none published
    // one, so the accounting cannot be settled.
    EXPECT_FALSE(lifecycle.settle_run_diagnostics(runtime.get(), /*graph_completed=*/true))
        << "a run short of an owed measurement published its accounting as settled";
}

/**
 * The same run with its cell published settles.
 *
 * Pairs with the case above so the refusal is attributable to the missing
 * measurement rather than to anything else the settlement does.
 */
TEST_F(HbgRetirementWiring, AProvedStopWithEveryOwedMeasurementSettles) {
    auto runtime = std::make_unique<Runtime>();
    MinimalSchedulerState scheduler_state;
    scheduler_state.publish(runtime.get());
    scheduler_state.cell()->worker_id = 0;
    scheduler_state.cell()->published = 1;
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::seed(lifecycle, kAddrs);
    set_pmu_enabled(true);

    ASSERT_EQ(lifecycle.finish_shutdown_partition(0, runtime.get()), 0);
    EXPECT_TRUE(lifecycle.settle_run_diagnostics(runtime.get(), /*graph_completed=*/true));
}

/**
 * A graph that did not complete cannot say what it owed, so it settles nothing
 * even with every cell it did stage present.
 */
TEST_F(HbgRetirementWiring, AnIncompleteGraphSettlesNothingEvenWithItsStagedCells) {
    auto runtime = std::make_unique<Runtime>();
    MinimalSchedulerState scheduler_state;
    scheduler_state.publish(runtime.get());
    scheduler_state.cell()->worker_id = 0;
    scheduler_state.cell()->published = 1;
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::seed(lifecycle, kAddrs);
    set_pmu_enabled(true);

    ASSERT_EQ(lifecycle.finish_shutdown_partition(0, runtime.get()), 0);
    EXPECT_FALSE(lifecycle.settle_run_diagnostics(runtime.get(), /*graph_completed=*/false));
}

/**
 * An unproved stop is refused before anything reads a cell or touches a
 * register, whatever the data would have said.
 */
TEST_F(HbgRetirementWiring, AnUnprovedStopIsRefusedBeforeAnyCellIsRead) {
    auto runtime = std::make_unique<Runtime>();
    MinimalSchedulerState scheduler_state;
    scheduler_state.publish(runtime.get());
    scheduler_state.cell()->worker_id = 0;
    scheduler_state.cell()->published = 1;
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::seed(lifecycle, kAddrs);
    set_pmu_enabled(true);
    retirement_result = -1;

    EXPECT_NE(lifecycle.finish_shutdown_partition(0, runtime.get()), 0);
    ASSERT_FALSE(lifecycle.cores_retired());
    EXPECT_FALSE(lifecycle.settle_run_diagnostics(runtime.get(), /*graph_completed=*/true));
}
}  // namespace
