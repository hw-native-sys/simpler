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
#include "host_build_graph/runtime_status.h"
#include "host_build_graph/shared_memory.h"
#include "runtime.h"
#include "scheduler/scheduler_graph.h"
#include "scheduler/scheduler_types.h"

// The register-layer tests exercise the real ACK, close, read-back, drain and
// release sequence. This seam verifies that the HBG lifecycle passes the real
// per-core gate on normal shutdown and failed startup.
namespace {
struct RetireCall {
    std::vector<uint64_t> addrs;
    std::vector<AicoreTeardownControl *> gates;
};

std::vector<RetireCall> retire_calls;
std::vector<uint64_t> exit_signals;
int32_t retirement_result = 0;
uint64_t released_despite_failure = 0;
Runtime *observed_retirement_runtime = nullptr;
std::mutex retire_mutex;
int scheduler_timeout_ms = 0;
bool dump_enabled = false;
uint64_t dump_base = 0;
std::array<uint64_t, PLATFORM_MAX_CORES> pmu_registers{};
uint32_t register_reads = 0;
uint32_t register_writes = 0;
int32_t dump_initializations = 0;
std::vector<int> dump_flushes;
std::vector<int> dump_closes;
}  // namespace

uint32_t platform_get_physical_cores_count() { return 3; }

uint64_t platform_aicore_exit_deadline() { return std::numeric_limits<uint64_t>::max(); }

int32_t platform_retire_aicore_group(const AicoreExitTarget *targets, size_t count, uint64_t, bool *released) {
    RetireCall call;
    for (size_t i = 0; i < count; ++i) {
        call.addrs.push_back(targets[i].reg_addr);
        call.gates.push_back(targets[i].teardown);
        if (observed_retirement_runtime != nullptr) {
            const auto worker = targets[i].teardown - observed_retirement_runtime->get_teardown_gates();
            EXPECT_NE(observed_retirement_runtime->dev.workers[worker].aicore_done, 0u);
            EXPECT_NE(observed_retirement_runtime->dev.workers[worker].aicpu_ready, 0u);
        }
        if (released != nullptr)
            released[i] = retirement_result == 0 || targets[i].reg_addr == released_despite_failure;
    }
    {
        std::lock_guard<std::mutex> lock(retire_mutex);
        retire_calls.push_back(std::move(call));
    }
    return retirement_result;
}

void platform_signal_aicore_exit(uint64_t reg_addr) { exit_signals.push_back(reg_addr); }
void platform_init_aicore_regs(uint64_t) {}
void write_reg(uint64_t, RegId, uint64_t) { ++register_writes; }
uint64_t read_reg(uint64_t, RegId) {
    ++register_reads;
    return 0;
}
uint64_t get_platform_regs() { return reinterpret_cast<uint64_t>(pmu_registers.data()); }
extern "C" int get_scheduler_timeout_ms() { return scheduler_timeout_ms; }
extern "C" bool is_dump_args_enabled() { return dump_enabled; }
extern "C" uint64_t get_platform_dump_base() { return dump_base; }
void dump_args_flush(int thread) { dump_flushes.push_back(thread); }
void dump_args_close_run(int thread) { dump_closes.push_back(thread); }
void dump_args_init(int, DumpArgsLevel) { ++dump_initializations; }

// This retirement fixture owns no task-copy image; real copy initialization is
// exercised by test_hbg_args_dump_service and the end-to-end dump scenes.
bool initialize_resident_args_dump_outputs(void *image, uint64_t, uint64_t) {
    EXPECT_EQ(image, nullptr);
    return image == nullptr;
}

class AicoreLifecycleTestPeer {
public:
    static void seed(AicoreLifecycle &lifecycle, const std::array<uint64_t, 3> &addrs, int32_t threads = 1) {
        lifecycle.core_count_ = 3;
        lifecycle.aicpu_thread_num_ = threads;
        lifecycle.retired_core_count_.store(0, std::memory_order_release);
        for (int i = 0; i < 3; ++i)
            lifecycle.cores_[i].reg_addr = addrs[i];
    }

    static void prepare_handshake(AicoreLifecycle &lifecycle, const uint64_t *registers) {
        lifecycle.core_count_ = 3;
        lifecycle.aicpu_thread_num_ = 1;
        lifecycle.regs_base_ = reinterpret_cast<uint64_t>(registers);
        for (auto &core : lifecycle.cores_)
            core = {};
    }
    static bool handshake_failed(const AicoreLifecycle &lifecycle) { return lifecycle.handshake_failed_.load(); }

    static int32_t retired_cores(const AicoreLifecycle &lifecycle) {
        return lifecycle.retired_core_count_.load(std::memory_order_acquire);
    }
    static int32_t core_count(const AicoreLifecycle &lifecycle) { return lifecycle.core_count_; }

    static void seed_discovered_cluster(AicoreLifecycle &lifecycle, const std::array<uint64_t, 3> &addrs) {
        seed(lifecycle, addrs);
        for (int i = 0; i < 3; ++i) {
            lifecycle.cores_[i].core_type = i == 0 ? CoreType::AIC : CoreType::AIV;
            lifecycle.cores_[i].physical_core_id = static_cast<uint32_t>(3 + 17 * i);
        }
    }

    static void mark_pmu_initialized(AicoreLifecycle &lifecycle) { lifecycle.pmu_initialized_ = true; }

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

    alignas(SchedulerWorkerContext) std::array<uint8_t, kTotalSize> state{};
    alignas(64) std::array<uint8_t, SCHEDULER_GRAPH_TASK_STORAGE_STRIDE> storage{};

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
        released_despite_failure = 0;
        observed_retirement_runtime = nullptr;
        scheduler_timeout_ms = 0;
        dump_enabled = false;
        dump_base = 0;
        dump_initializations = 0;
        dump_flushes.clear();
        dump_closes.clear();
        pmu_registers.fill(0);
        register_reads = 0;
        register_writes = 0;
        set_platform_pmu_base(0);
        set_pmu_enabled(false);
    }

    void TearDown() override {
        set_pmu_enabled(false);
        set_platform_pmu_base(0);
    }

    static void expect_group(const Runtime &runtime) {
        ASSERT_EQ(retire_calls.size(), 1u);
        EXPECT_EQ(retire_calls[0].addrs, std::vector<uint64_t>(kAddrs.begin(), kAddrs.end()));
        for (size_t i = 0; i < kAddrs.size(); ++i)
            EXPECT_EQ(retire_calls[0].gates[i], &runtime.dev.teardown_gates[i]);
    }
};

TEST_F(HbgRetirementWiring, OnlyAcknowledgedReportsAreInvalidatedOnEveryExitRoute) {
    for (int route = 0; route < 2; ++route) {
        for (int failure = 0; failure < 3; ++failure) {
            const bool fail = failure != 0;
            auto runtime = std::make_unique<Runtime>();
            for (int i = 0; i < 3; ++i) {
                auto &report = runtime->dev.workers[i];
                report.aicore_done = i + 1;
                report.aicpu_ready = 2;
                report.report_epoch = 42;
                report.task = 0x1234;
            }
            observed_retirement_runtime = runtime.get();
            retirement_result = fail ? -1 : 0;
            released_despite_failure = failure == 1 ? kAddrs[1] : 0;
            AicoreLifecycle lifecycle;
            AicoreLifecycleTestPeer::seed(lifecycle, kAddrs);
            const int rc = route == 0 ? lifecycle.release_partition(runtime.get(), 0, false) :
                                        lifecycle.finish_shutdown_partition(0, runtime.get());
            EXPECT_EQ(rc, fail ? -1 : 0);
            for (int i = 0; i < 3; ++i) {
                const auto &report = runtime->dev.workers[i];
                const bool acknowledged = !fail || (failure == 1 && i == 1);
                EXPECT_EQ(report.aicore_done, acknowledged ? 0u : static_cast<uint32_t>(i + 1));
                EXPECT_EQ(report.aicpu_ready, acknowledged ? 0u : 2u);
                EXPECT_EQ(report.report_epoch, 42u);
                EXPECT_EQ(report.task, 0x1234u);
            }
            observed_retirement_runtime = nullptr;
        }
    }
}

TEST_F(HbgRetirementWiring, PartialHandshakeTimeoutRetiresEveryAcceptedReport) {
    scheduler_timeout_ms = 1;
    auto runtime = std::make_unique<Runtime>();
    SharedMemoryHeader header{};
    runtime->set_gm_sm_ptr(&header);
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::prepare_handshake(lifecycle, kAddrs.data());
    for (int i : {0, 2}) {
        runtime->dev.workers[i].physical_core_id = i;
        runtime->dev.workers[i].core_type = CoreType::AIV;
        runtime->dev.workers[i].aicore_done = i + 1;
    }
    lifecycle.handshake_partition(runtime.get(), 0, 1);
    ASSERT_TRUE(AicoreLifecycleTestPeer::handshake_failed(lifecycle));
    EXPECT_EQ(lifecycle.release_partition(runtime.get(), 0, false), -SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    EXPECT_EQ(AicoreLifecycleTestPeer::retired_cores(lifecycle), 2);
    EXPECT_FALSE(lifecycle.cores_retired());
    EXPECT_EQ(header.sched_error_code.load(), SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    ASSERT_EQ(retire_calls.size(), 1u);
    EXPECT_EQ(retire_calls[0].addrs, (std::vector<uint64_t>{kAddrs[0], kAddrs[2]}));
    EXPECT_EQ(retire_calls[0].gates[0], &runtime->dev.teardown_gates[0]);
    EXPECT_EQ(retire_calls[0].gates[1], &runtime->dev.teardown_gates[2]);
}

TEST_F(HbgRetirementWiring, MissingReportsLeaveRetirementUnknownWithoutOpeningAWindow) {
    auto runtime = std::make_unique<Runtime>();
    SharedMemoryHeader header{};
    runtime->set_gm_sm_ptr(&header);
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::prepare_handshake(lifecycle, kAddrs.data());
    EXPECT_EQ(lifecycle.release_partition(runtime.get(), 0, false), -SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    EXPECT_TRUE(retire_calls.empty());
    EXPECT_EQ(AicoreLifecycleTestPeer::retired_cores(lifecycle), 0);
    EXPECT_FALSE(lifecycle.cores_retired());
    EXPECT_FALSE(lifecycle.settle_run_diagnostics(runtime.get(), false));
    EXPECT_EQ(header.sched_error_code.load(), SIMPLER_ERROR_SCHEDULER_TIMEOUT);
}

TEST_F(HbgRetirementWiring, DuplicateReportsKeepOneWindowOwnerAndLeaveRetirementUnproved) {
    auto runtime = std::make_unique<Runtime>();
    SharedMemoryHeader header{};
    runtime->set_gm_sm_ptr(&header);
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::prepare_handshake(lifecycle, kAddrs.data());
    for (int32_t worker = 0; worker < 3; ++worker) {
        auto &report = runtime->dev.workers[worker];
        report.physical_core_id = 0;
        report.core_type = CoreType::AIC;
        report.aicore_done = worker + 1;
    }
    lifecycle.handshake_partition(runtime.get(), 0, 1);
    EXPECT_EQ(lifecycle.validate_discovery(), -SIMPLER_ERROR_INVALID_ARGS);
    EXPECT_TRUE(lifecycle.core_reported(0));
    EXPECT_FALSE(lifecycle.core_reported(1));
    EXPECT_FALSE(lifecycle.core_reported(2));
    EXPECT_EQ(lifecycle.release_partition(runtime.get(), 0, false), -SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    ASSERT_EQ(retire_calls.size(), 1u);
    EXPECT_EQ(retire_calls[0].addrs, (std::vector<uint64_t>{kAddrs[0]}));
    ASSERT_EQ(retire_calls[0].gates.size(), 1u);
    EXPECT_EQ(retire_calls[0].gates[0], &runtime->dev.teardown_gates[0]);
    EXPECT_EQ(runtime->dev.workers[0].aicore_done, 0u);
    EXPECT_EQ(runtime->dev.workers[1].aicore_done, 2u);
    EXPECT_EQ(runtime->dev.workers[2].aicore_done, 3u);
    EXPECT_EQ(AicoreLifecycleTestPeer::retired_cores(lifecycle), 1);
    EXPECT_FALSE(lifecycle.cores_retired());
    EXPECT_FALSE(lifecycle.settle_run_diagnostics(runtime.get(), false));
}

TEST_F(HbgRetirementWiring, AReportedCoreWithoutARegisterWindowCannotProveDiscoveryOrRetirement) {
    auto runtime = std::make_unique<Runtime>();
    SharedMemoryHeader header{};
    runtime->set_gm_sm_ptr(&header);
    AicoreLifecycle lifecycle;
    const std::array<uint64_t, 3> registers{kAddrs[0], 0, kAddrs[2]};
    AicoreLifecycleTestPeer::prepare_handshake(lifecycle, registers.data());
    for (int32_t worker = 0; worker < 3; ++worker) {
        auto &report = runtime->dev.workers[worker];
        report.physical_core_id = worker;
        report.core_type = CoreType::AIV;
        report.aicore_done = worker + 1;
    }
    lifecycle.handshake_partition(runtime.get(), 0, 1);
    EXPECT_EQ(lifecycle.validate_discovery(), -SIMPLER_ERROR_INVALID_ARGS);
    EXPECT_EQ(lifecycle.release_partition(runtime.get(), 0, false), -SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    ASSERT_EQ(retire_calls.size(), 1u);
    EXPECT_EQ(retire_calls[0].addrs, (std::vector<uint64_t>{kAddrs[0], kAddrs[2]}));
    EXPECT_EQ(runtime->dev.workers[1].aicore_done, 2u);
    EXPECT_EQ(AicoreLifecycleTestPeer::retired_cores(lifecycle), 2);
    EXPECT_FALSE(lifecycle.cores_retired());
}

TEST_F(HbgRetirementWiring, CompleteHandshakePreservesFailureRetirementError) {
    auto runtime = std::make_unique<Runtime>();
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::prepare_handshake(lifecycle, kAddrs.data());
    for (int i = 0; i < 3; ++i) {
        runtime->dev.workers[i].physical_core_id = i;
        runtime->dev.workers[i].core_type = CoreType::AIV;
        runtime->dev.workers[i].aicore_done = i + 1;
    }
    lifecycle.handshake_partition(runtime.get(), 0, 1);
    ASSERT_FALSE(AicoreLifecycleTestPeer::handshake_failed(lifecycle));
    retirement_result = -1;
    EXPECT_EQ(lifecycle.release_partition(runtime.get(), 0, false), -1);
    expect_group(*runtime);
}

struct DiscoveryProofState {
    SchedulerRunControl run{};
    SchedulerGraphSummary summary{};
    std::array<SchedulerWorkerContext, 3> contexts{};
    std::array<AicpuThreadLifecycleTrace, 1> traces{};

    void bind(Runtime &runtime) {
        runtime.set_worker_count(3);
        runtime.dev.host_total_tasks = 0;
        summary.worker_count = 3;
        summary.cluster_count = 1;
        for (uint64_t worker = 0; worker < contexts.size(); ++worker) {
            auto &context = contexts[worker];
            context.worker_index = worker;
            context.scheduler_state_base_address = reinterpret_cast<uint64_t>(this);
            context.run_control_offset = offsetof(DiscoveryProofState, run);
            context.graph_summary_offset = offsetof(DiscoveryProofState, summary);
            context.worker_contexts_offset = offsetof(DiscoveryProofState, contexts);
            context.aicpu_lifecycle_traces_offset = offsetof(DiscoveryProofState, traces);
            runtime.dev.workers[worker].aicpu_ready = 0;
            runtime.dev.workers[worker].task = 0;
        }
        runtime.publish_scheduler_bootstrap(
            SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING, reinterpret_cast<uint64_t>(contexts.data())
        );
    }
};

TEST_F(HbgRetirementWiring, DiscoveryProofPrecedesEveryReadyReply) {
    auto runtime = std::make_unique<Runtime>();
    DiscoveryProofState state;
    state.bind(*runtime);
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::seed_discovered_cluster(lifecycle, kAddrs);
    ASSERT_EQ(lifecycle.post_handshake_init(runtime.get()), 0);
    EXPECT_EQ(state.summary.topology_layout, SchedulerTopologyLayout::COMPACT_LAUNCH);
    for (uint64_t worker = 0; worker < state.contexts.size(); ++worker) {
        EXPECT_EQ(runtime->dev.workers[worker].aicpu_ready, 0u);
        EXPECT_EQ(state.contexts[worker].worker_index, worker);
        EXPECT_EQ(state.contexts[worker].cluster_worker_ids[0], 0u);
        EXPECT_EQ(state.contexts[worker].cluster_worker_ids[1], 1u);
        EXPECT_EQ(state.contexts[worker].cluster_worker_ids[2], 2u);
    }
    lifecycle.publish_context_partition(runtime.get(), 0);
    for (uint64_t worker = 0; worker < state.contexts.size(); ++worker) {
        EXPECT_EQ(runtime->dev.workers[worker].aicpu_ready, SCHEDULER_RUNTIME_MODE_RESIDENT_READY);
        EXPECT_EQ(runtime->dev.workers[worker].task, reinterpret_cast<uint64_t>(&state.contexts[worker]));
    }
}

TEST_F(HbgRetirementWiring, InvalidCompactIdentityCannotPublishATopologyProof) {
    for (uint64_t bad = 0; bad < 3; ++bad) {
        auto runtime = std::make_unique<Runtime>();
        DiscoveryProofState state;
        state.bind(*runtime);
        state.contexts[bad].worker_index = (bad + 1) % 3;
        AicoreLifecycle lifecycle;
        AicoreLifecycleTestPeer::seed_discovered_cluster(lifecycle, kAddrs);
        EXPECT_EQ(lifecycle.post_handshake_init(runtime.get()), -1);
        EXPECT_EQ(state.summary.topology_layout, SchedulerTopologyLayout::GENERAL);
        for (uint64_t worker = 0; worker < 3; ++worker)
            EXPECT_EQ(runtime->dev.workers[worker].aicpu_ready, 0u);
    }
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

TEST_F(HbgRetirementWiring, ARunWithoutPmuInitializationCannotRestoreThePreviousCollectorsWindows) {
    struct PmuImage {
        PmuDataHeader header{};
        std::array<PmuBufferState, 3> states{};
    };
    static auto previous = std::make_unique<PmuImage>();
    static auto buffers = std::make_unique<std::array<PmuBuffer, 3>>();
    for (size_t core = 0; core < kAddrs.size(); ++core) {
        pmu_registers[core] = kAddrs[core];
        previous->states[core].current_buf_ptr = reinterpret_cast<uint64_t>(&(*buffers)[core]);
    }
    const uint32_t physical[] = {0, 1, 2};
    set_platform_pmu_base(reinterpret_cast<uint64_t>(previous.get()));
    pmu_aicpu_init(physical, 3, PmuEventType::PIPE_UTILIZATION);
    ASSERT_GT(register_writes, 0u);
    register_reads = 0;
    register_writes = 0;

    auto runtime = std::make_unique<Runtime>();
    MinimalSchedulerState scheduler_state;
    scheduler_state.publish(runtime.get());
    scheduler_state.cell()->worker_id = 0;
    scheduler_state.cell()->published = 1;
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::seed(lifecycle, kAddrs);
    set_pmu_enabled(true);
    set_platform_pmu_base(0);
    ASSERT_EQ(lifecycle.finish_shutdown_partition(0, runtime.get()), 0);
    ASSERT_TRUE(lifecycle.cores_retired());
    EXPECT_FALSE(lifecycle.settle_run_diagnostics(runtime.get(), false));
    EXPECT_EQ(register_reads, 0u);
    EXPECT_EQ(register_writes, 0u);

    pmu_registers.fill(0);
    set_platform_pmu_base(reinterpret_cast<uint64_t>(previous.get()));
    pmu_aicpu_init(physical, 3, PmuEventType::PIPE_UTILIZATION);
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
    AicoreLifecycleTestPeer::mark_pmu_initialized(lifecycle);
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
    AicoreLifecycleTestPeer::mark_pmu_initialized(lifecycle);
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
    AicoreLifecycleTestPeer::mark_pmu_initialized(lifecycle);
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
    AicoreLifecycleTestPeer::mark_pmu_initialized(lifecycle);
    set_pmu_enabled(true);
    retirement_result = -1;

    EXPECT_NE(lifecycle.finish_shutdown_partition(0, runtime.get()), 0);
    ASSERT_FALSE(lifecycle.cores_retired());
    EXPECT_FALSE(lifecycle.settle_run_diagnostics(runtime.get(), /*graph_completed=*/true));
}
}  // namespace

TEST_F(HbgRetirementWiring, EnabledDumpWithoutItsTransportRefusesBeforeProducerInitialization) {
    auto runtime = std::make_unique<Runtime>();
    runtime->set_worker_count(3);
    dump_enabled = true;
    AicoreLifecycle lifecycle;
    EXPECT_NE(lifecycle.pre_handshake_init(runtime.get(), 1, 0), 0);
    EXPECT_EQ(dump_initializations, 0);
}

TEST_F(HbgRetirementWiring, ArgumentAccountingWaitsForRetirementAndDoesNotCertifyAnIncompleteGraph) {
    auto runtime = std::make_unique<Runtime>();
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::seed(lifecycle, kAddrs, 3);
    dump_enabled = true;
    EXPECT_FALSE(lifecycle.settle_run_diagnostics(runtime.get(), true));
    EXPECT_TRUE(dump_flushes.empty());
    EXPECT_TRUE(dump_closes.empty());
    for (int thread = 0; thread < 3; ++thread)
        ASSERT_EQ(lifecycle.finish_shutdown_partition(thread, runtime.get()), 0);
    ASSERT_TRUE(lifecycle.settle_run_diagnostics(runtime.get(), false));
    EXPECT_EQ(dump_flushes, (std::vector<int>{0, 1, 2}));
    EXPECT_TRUE(dump_closes.empty());
    dump_flushes.clear();
    ASSERT_TRUE(lifecycle.settle_run_diagnostics(runtime.get(), true));
    EXPECT_EQ(dump_flushes, (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(dump_closes, (std::vector<int>{0, 1, 2}));
}
