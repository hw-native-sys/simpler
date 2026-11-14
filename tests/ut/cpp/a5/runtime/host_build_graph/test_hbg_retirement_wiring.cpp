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
#include <sstream>
#include "host/scheduler_profiling_json.h"

#include "common/args_dump.h"
#include "aicore_lifecycle.h"
#include "aicpu/platform_regs.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "host_build_graph/runtime_status.h"
#include "host_build_graph/shared_memory.h"
#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "runtime.h"
#include "task_image_status.h"
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
    static void prove_retired(AicoreLifecycle &lifecycle) {
        lifecycle.retired_core_count_.store(lifecycle.core_count_);
    }

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
    TaskImageStatus header{};
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
    EXPECT_EQ(header.error_code.load(), SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    ASSERT_EQ(retire_calls.size(), 1u);
    EXPECT_EQ(retire_calls[0].addrs, (std::vector<uint64_t>{kAddrs[0], kAddrs[2]}));
    EXPECT_EQ(retire_calls[0].gates[0], &runtime->dev.teardown_gates[0]);
    EXPECT_EQ(retire_calls[0].gates[1], &runtime->dev.teardown_gates[2]);
}

TEST_F(HbgRetirementWiring, MissingReportsLeaveRetirementUnknownWithoutOpeningAWindow) {
    auto runtime = std::make_unique<Runtime>();
    TaskImageStatus header{};
    runtime->set_gm_sm_ptr(&header);
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::prepare_handshake(lifecycle, kAddrs.data());
    EXPECT_EQ(lifecycle.release_partition(runtime.get(), 0, false), -SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    EXPECT_TRUE(retire_calls.empty());
    EXPECT_EQ(AicoreLifecycleTestPeer::retired_cores(lifecycle), 0);
    EXPECT_FALSE(lifecycle.cores_retired());
    EXPECT_FALSE(lifecycle.settle_run_diagnostics(runtime.get(), false));
    EXPECT_EQ(header.error_code.load(), SIMPLER_ERROR_SCHEDULER_TIMEOUT);
}

TEST_F(HbgRetirementWiring, DuplicateReportsKeepOneWindowOwnerAndLeaveRetirementUnproved) {
    auto runtime = std::make_unique<Runtime>();
    TaskImageStatus header{};
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
    TaskImageStatus header{};
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
            context.scheduler_state_bytes = sizeof(*this);
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

namespace {
struct ResidentSwimlaneState {
    SchedulerWorkerContext context{};
    SchedulerTaskMetadata metadata{};
    SchedulerTaskTrace trace{};
    std::array<uint8_t, SCHEDULER_GRAPH_TASK_STORAGE_STRIDE> graph{};
    void bind(Runtime &runtime, uint64_t epoch) {
        runtime.set_worker_count(3);
        context.scheduler_state_base_address = reinterpret_cast<uint64_t>(this);
        context.scheduler_state_bytes = sizeof(*this);
        context.task_metadata_offset = offsetof(ResidentSwimlaneState, metadata);
        context.trace_cells_offset = offsetof(ResidentSwimlaneState, trace);
        context.graph_task_count = 1;
        context.graph_storage_address = reinterpret_cast<uint64_t>(graph.data());
        metadata.flags = SCHEDULER_TASK_EXECUTABLE;
        metadata.active_mask = LaneMask::aic();
        metadata.logical_block_num = 1;
        trace.valid = 1;
        trace.worker_id = 1;
        trace.task_ref = 0;
        trace.kernel_start_cycles = 200;
        trace.kernel_end_cycles = 300;
        trace.dispatch_start_cycles = 100;
        trace.dispatch_end_cycles = 150;
        trace.complete_start_cycles = 350;
        trace.complete_end_cycles = 400;
        trace.execution_run_epoch = epoch;
        runtime.publish_scheduler_bootstrap(
            SCHEDULER_RUNTIME_MODE_RESIDENT_READY, reinterpret_cast<uint64_t>(&context)
        );
    }
};
struct SwimlaneTransport {
    std::vector<uint64_t> storage{std::vector<uint64_t>((calc_perf_data_size_with_phases() + 7) / 8)};
    std::array<std::vector<ChipSwimlaneAicoreTaskBuffer>, 3> core;
    std::array<std::vector<ChipSwimlaneAicpuTaskBuffer>, 3> timing;
    std::vector<ChipSwimlaneAicpuSchedPhaseBuffer> phases =
        std::vector<ChipSwimlaneAicpuSchedPhaseBuffer>(PLATFORM_PROF_SLOT_COUNT, ChipSwimlaneAicpuSchedPhaseBuffer{});
    SwimlaneTransport() {
        for (int i = 0; i < 3; ++i) {
            core[i].resize(PLATFORM_PROF_SLOT_COUNT);
            timing[i].resize(PLATFORM_PROF_SLOT_COUNT);
            auto *a = get_aicore_buffer_state(storage.data(), i);
            auto *t = get_perf_buffer_state(storage.data(), i);
            for (int j = 0; j < PLATFORM_PROF_SLOT_COUNT; ++j) {
                a->free_queue.buffer_ptrs[j] = reinterpret_cast<uint64_t>(&core[i][j]);
                t->free_queue.buffer_ptrs[j] = reinterpret_cast<uint64_t>(&timing[i][j]);
            }
            a->free_queue.tail = t->free_queue.tail = PLATFORM_PROF_SLOT_COUNT;
        }
        auto *p = get_sched_phase_buffer_state(storage.data(), 0);
        for (int i = 0; i < PLATFORM_PROF_SLOT_COUNT; ++i)
            p->free_queue.buffer_ptrs[i] = reinterpret_cast<uint64_t>(&phases[i]);
        p->free_queue.tail = PLATFORM_PROF_SLOT_COUNT;
    }
    void arm(uint64_t epoch) {
        set_platform_run_result(0, epoch);
        set_chip_swimlane_enabled(true);
        set_platform_chip_swimlane_base(reinterpret_cast<uint64_t>(storage.data()));
        set_platform_chip_swimlane_aicore_rotation_table(0);
        chip_swimlane_aicpu_init(3, ChipSwimlaneLevel::ORCH_PHASES);
        chip_swimlane_aicpu_init_phase(3, 1, 0);
    }
    ~SwimlaneTransport() {
        set_chip_swimlane_enabled(false);
        set_platform_chip_swimlane_base(0);
    }
    uint64_t records() const {
        auto *h = get_chip_swimlane_header(const_cast<uint64_t *>(storage.data()));
        uint64_t n = 0;
        for (uint32_t i = h->queue_heads[0]; i < h->queue_tails[0]; ++i) {
            const auto &e = h->queues[0][i % PLATFORM_PROF_READYQUEUE_SIZE];
            if (e.kind == ChipSwimlaneBufferKind::AicoreTask) n += e.record_count;
        }
        return n;
    }
};
}  // namespace

TEST_F(HbgRetirementWiring, ResidentSwimlanePublishesEachRetiredReplayWithItsEpochAndWorker) {
    auto runtime = std::make_unique<Runtime>();
    ResidentSwimlaneState state;
    auto transport = std::make_unique<SwimlaneTransport>();
    state.bind(*runtime, 41);
    transport->arm(41);
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::seed_discovered_cluster(lifecycle, kAddrs);
    AicoreLifecycleTestPeer::prove_retired(lifecycle);
    ASSERT_TRUE(lifecycle.settle_run_diagnostics(runtime.get(), true));
    EXPECT_EQ(transport->records(), 1u);
    EXPECT_EQ(transport->core[1][0].run_epoch, 41u);
    EXPECT_EQ(transport->core[1][0].records[0].start_time, 200u);
    EXPECT_EQ(transport->timing[1][0].records[0].dispatch_time, 150u);
    EXPECT_EQ(transport->timing[1][0].records[0].finish_time, 350u);
    EXPECT_EQ(transport->core[0][0].count, 0u);
}

namespace {
std::string phase_json(SwimlaneTransport &transport, const std::string *snapshot = nullptr, bool unknown = true) {
    std::vector<std::vector<CollectedRecord<ChipSwimlaneAicpuSchedPhaseRecord>>> streams(1);
    auto *header = get_chip_swimlane_header(transport.storage.data());
    for (uint32_t i = header->queue_heads[0]; i < header->queue_tails[0]; ++i) {
        const auto &entry = header->queues[0][i % PLATFORM_PROF_READYQUEUE_SIZE];
        if (entry.kind != ChipSwimlaneBufferKind::AicpuSchedPhase) continue;
        auto *buffer = reinterpret_cast<ChipSwimlaneAicpuSchedPhaseBuffer *>(entry.buffer_ptr);
        for (uint32_t j = 0; j < entry.record_count; ++j)
            streams[0].push_back({buffer->records[j], entry.run_epoch});
    }
    std::ostringstream out;
    chip_swimlane_write_scheduler_records(
        out, streams, {get_sched_phase_buffer_state(transport.storage.data(), 0)->head.dropped_record_count}, unknown,
        snapshot
    );
    return out.str();
}
}  // namespace

TEST_F(HbgRetirementWiring, ResidentSwimlaneAccumulatesInterleavedEpochsAndKeepsTruePhaseParticipants) {
    auto runtime = std::make_unique<Runtime>();
    auto a = std::make_unique<SwimlaneTransport>();
    auto b = std::make_unique<SwimlaneTransport>();
    ResidentSwimlaneState state;
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::seed_discovered_cluster(lifecycle, kAddrs);
    AicoreLifecycleTestPeer::prove_retired(lifecycle);
    runtime->dev.workers[2].core_type = CoreType::AIV;
    runtime->dev.workers[2].physical_core_id = 37;
    for (bool first : {false, true, false, true, false, false}) {
        const uint64_t epoch = first ? 41 : 42;
        state.bind(*runtime, epoch);
        state.trace.dispatch_scheduler_worker_id = state.trace.complete_scheduler_worker_id = 2;
        auto &transport = first ? *a : *b;
        transport.arm(epoch);
        ASSERT_TRUE(lifecycle.settle_run_diagnostics(runtime.get(), true));
    }
    EXPECT_EQ(a->records(), 2u);
    EXPECT_EQ(b->records(), 4u);
    const auto json = phase_json(*a);
    EXPECT_NE(json.find("\"producer\": \"aicore\""), std::string::npos);
    EXPECT_NE(json.find("\"worker_id\": 2"), std::string::npos);
    EXPECT_NE(json.find("\"physical_core_id\": 37"), std::string::npos);
    EXPECT_NE(json.find("\"core_type\": \"aiv\""), std::string::npos);
    EXPECT_NE(json.find("\"kind\": \"dispatch\""), std::string::npos);
    EXPECT_NE(json.find("\"kind\": \"complete\""), std::string::npos);
    EXPECT_EQ(json.find("\"producer\": \"aicpu\""), std::string::npos);
    a->arm(41);
    // The real ring export is also a valid runtime snapshot. It stays canonical
    // for its AICore participant while a separate ordinary AICPU event survives.
    chip_swimlane_aicpu_record_task_phase(0, SchedPhaseKind::Complete, 501, 502, 1, TaskId::make_global(0));
    chip_swimlane_aicpu_flush_sched_phase_buffer(0);
    auto *phase_pool = get_sched_phase_buffer_state(a->storage.data(), 0);
    ASSERT_EQ(phase_pool->head.current_buf_ptr, 0u);
    phase_pool->free_queue.head = phase_pool->free_queue.tail;
    chip_swimlane_aicpu_record_staged_phase(0, a->phases[2].records[0]);
    ASSERT_EQ(get_sched_phase_buffer_state(a->storage.data(), 0)->head.dropped_record_count, 1u);
    const auto merged = phase_json(*a, &json, false);
    const auto aicpu_stream = merged.find("\"producer\": \"aicpu\"");
    ASSERT_NE(aicpu_stream, std::string::npos);
    const auto aicpu_loss = merged.find("\"dropped\": null", aicpu_stream);
    EXPECT_LT(aicpu_loss, merged.find("\"records\":", aicpu_stream));
    EXPECT_NE(merged.find("\"producer\": \"aicpu\""), std::string::npos);
    const auto first = merged.find("\"producer\": \"aicore\"");
    ASSERT_NE(first, std::string::npos);
    EXPECT_EQ(merged.find("\"producer\": \"aicore\"", first + 1), std::string::npos);
    const auto body_start = json.find('[') + 1;
    EXPECT_NE(merged.find(json.substr(body_start, json.rfind(']') - body_start)), std::string::npos);
    const std::string empty_snapshot = R"({"streams": []})";
    const auto ordinary = phase_json(*a, &empty_snapshot, false);
    const auto ordinary_start = ordinary.find('[') + 1;
    const auto ordinary_body = ordinary.substr(ordinary_start, ordinary.rfind(']') - ordinary_start);
    const std::string prefix = R"({"nested":{"streams":[]},"streams": [)";
    const std::string suffix =
        R"(],"prepared_calls":{"schema":"hbg-prepared-calls-v4","run_epoch":41,"source":"host",)"
        R"("columns":["execution_index","node_ref","task_id","block_idx","subtask_slot","owner","phase","generation"],)"
        R"("records":[[0,0,0,0,0,2,3,1]]}})";
    for (const bool populated : {false, true}) {
        SCOPED_TRACE(populated);
        auto body = populated ? json.substr(body_start, json.rfind(']') - body_start) : " \n ";
        if (populated) body.insert(body.find('{') + 1, R"("note":"escaped \" ] [ \\","nested":[[],[1,2]],)");
        const auto snapshot = prefix + body + suffix;
        EXPECT_EQ(phase_json(*a, &snapshot, false), prefix + body + (populated ? "," : "") + ordinary_body + suffix);
    }
    for (size_t i = 0; i < 2; ++i)
        EXPECT_EQ(a->core[1][i].run_epoch, 41u);
    for (size_t i = 0; i < 4; ++i)
        EXPECT_EQ(b->core[1][i].run_epoch, 42u);
}

TEST_F(HbgRetirementWiring, ResidentSwimlaneDoesNotReadUnretiredOrStaleTraceAndZeroReplayStaysEmpty) {
    auto runtime = std::make_unique<Runtime>();
    auto transport = std::make_unique<SwimlaneTransport>();
    ResidentSwimlaneState state;
    state.bind(*runtime, 41);
    transport->arm(42);
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::seed(lifecycle, kAddrs);
    EXPECT_EQ(transport->records(), 0u);
    EXPECT_FALSE(lifecycle.settle_run_diagnostics(runtime.get(), true));
    EXPECT_EQ(transport->records(), 0u);
    AicoreLifecycleTestPeer::prove_retired(lifecycle);
    EXPECT_TRUE(lifecycle.settle_run_diagnostics(runtime.get(), true));
    EXPECT_EQ(transport->records(), 0u);
}

TEST_F(HbgRetirementWiring, OrdinaryAicpuPhaseKeepsItsParticipantAndStagedCapacityLossIsCounted) {
    auto transport = std::make_unique<SwimlaneTransport>();
    transport->arm(9);
    chip_swimlane_aicpu_record_task_phase(0, SchedPhaseKind::Complete, 11, 12, 1, TaskId::make_global(0));
    chip_swimlane_aicpu_flush_sched_phase_buffer(0);
    const auto json = phase_json(*transport);
    EXPECT_NE(json.find("\"producer\": \"aicpu\""), std::string::npos);
    EXPECT_NE(json.find("\"worker_id\": 0"), std::string::npos);
    EXPECT_NE(json.find("\"physical_core_id\": null"), std::string::npos);
    chip_swimlane_aicpu_init(3, ChipSwimlaneLevel::TASK_TIMING);
    auto *pool = get_aicore_buffer_state(transport->storage.data(), 1);
    pool->free_queue.head = pool->free_queue.tail;
    ChipSwimlaneAicoreTaskRecord record{100, 200, TaskId::make_global(0), 1, 0};
    for (uint32_t i = 0; i < PLATFORM_AICORE_BUFFER_SIZE + 1; ++i) {
        record.reg_task_id = i;
        chip_swimlane_aicpu_record_staged_task(1, 0, record, 90, 210);
    }
    int core = 1;
    chip_swimlane_aicpu_flush(0, &core, 1);
    EXPECT_EQ(pool->head.total_record_count, PLATFORM_AICORE_BUFFER_SIZE + 1);
    EXPECT_EQ(pool->head.dropped_record_count, 1u);
    EXPECT_EQ(pool->head.published_record_count, PLATFORM_AICORE_BUFFER_SIZE);
    EXPECT_EQ(transport->core[1][0].records[0].reg_task_id, 0u);
    EXPECT_EQ(
        transport->core[1][0].records[PLATFORM_AICORE_BUFFER_SIZE - 1].reg_task_id, PLATFORM_AICORE_BUFFER_SIZE - 1
    );
    EXPECT_EQ(transport->core[1][0].records[0].end_time, 200u);
}

TEST_F(HbgRetirementWiring, CanonicalSnapshotCannotAttributeAnEntirelyDroppedParticipantToAicpu) {
    auto runtime = std::make_unique<Runtime>();
    auto snapshot_transport = std::make_unique<SwimlaneTransport>();
    ResidentSwimlaneState state;
    state.bind(*runtime, 41);
    state.trace.dispatch_scheduler_worker_id = state.trace.complete_scheduler_worker_id = 2;
    runtime->dev.workers[2].core_type = CoreType::AIV;
    runtime->dev.workers[2].physical_core_id = 37;
    snapshot_transport->arm(41);
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::seed(lifecycle, kAddrs);
    AicoreLifecycleTestPeer::prove_retired(lifecycle);
    ASSERT_TRUE(lifecycle.settle_run_diagnostics(runtime.get(), true));
    const auto snapshot = phase_json(*snapshot_transport);
    auto transport = std::make_unique<SwimlaneTransport>();
    transport->arm(42);
    chip_swimlane_aicpu_record_task_phase(0, SchedPhaseKind::Complete, 501, 502, 1, TaskId::make_global(0));
    chip_swimlane_aicpu_flush_sched_phase_buffer(0);
    auto *pool = get_sched_phase_buffer_state(transport->storage.data(), 0);
    ASSERT_EQ(pool->head.current_buf_ptr, 0u);
    pool->free_queue.head = pool->free_queue.tail;
    chip_swimlane_aicpu_record_staged_phase(0, snapshot_transport->phases[0].records[0]);
    ASSERT_EQ(pool->head.dropped_record_count, 1u);
    ASSERT_EQ(phase_json(*transport).find("\"producer\": \"aicore\""), std::string::npos);
    const auto merged = phase_json(*transport, &snapshot, false);
    const auto stream = merged.find("\"producer\": \"aicpu\"");
    ASSERT_NE(stream, std::string::npos);
    EXPECT_LT(merged.find("\"dropped\": null", stream), merged.find("\"records\":", stream));
}

TEST_F(HbgRetirementWiring, ResidentSwimlaneFinalFlushNeverWaitsOrPublishesAnUnpairedTask) {
    auto runtime = std::make_unique<Runtime>();
    auto transport = std::make_unique<SwimlaneTransport>();
    ResidentSwimlaneState state;
    state.bind(*runtime, 41);
    state.trace.dispatch_scheduler_worker_id = state.trace.complete_scheduler_worker_id = 2;
    transport->arm(41);
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::seed(lifecycle, kAddrs);
    AicoreLifecycleTestPeer::prove_retired(lifecycle);
    ASSERT_TRUE(lifecycle.settle_run_diagnostics(runtime.get(), true));
    const auto saved_core = transport->core[1][0].records[0];
    const auto saved_timing = transport->timing[1][0].records[0];
    transport->arm(41);
    auto *header = get_chip_swimlane_header(transport->storage.data());
    header->queue_tails[0] = PLATFORM_PROF_READYQUEUE_SIZE - 1;
    const auto sentinel = header->queues[0][PLATFORM_PROF_READYQUEUE_SIZE - 1];
    ASSERT_TRUE(lifecycle.settle_run_diagnostics(runtime.get(), true));
    auto *core = get_aicore_buffer_state(transport->storage.data(), 1);
    auto *timing = get_perf_buffer_state(transport->storage.data(), 1);
    EXPECT_EQ(header->queue_tails[0], PLATFORM_PROF_READYQUEUE_SIZE - 1);
    EXPECT_EQ(std::memcmp(&sentinel, &header->queues[0][PLATFORM_PROF_READYQUEUE_SIZE - 1], sizeof(sentinel)), 0);
    for (auto *head : {&core->head, &timing->head}) {
        EXPECT_EQ(head->total_record_count, 1u);
        EXPECT_EQ(head->dropped_record_count, 1u);
        EXPECT_EQ(head->published_record_count, 0u);
        EXPECT_NE(head->current_buf_ptr, 0u);
    }
    EXPECT_EQ(transport->core[1][1].count, 0u);
    EXPECT_EQ(transport->timing[1][1].count, 0u);
    EXPECT_EQ(std::memcmp(&saved_core, &transport->core[1][0].records[0], sizeof(saved_core)), 0);
    EXPECT_EQ(std::memcmp(&saved_timing, &transport->timing[1][0].records[0], sizeof(saved_timing)), 0);
    auto *phase = get_sched_phase_buffer_state(transport->storage.data(), 0);
    EXPECT_EQ(phase->head.total_record_count, phase->head.dropped_record_count);
    EXPECT_EQ(phase->head.published_record_count, 0u);
    EXPECT_NE(phase->head.current_buf_ptr, 0u);
}

TEST_F(HbgRetirementWiring, StagedTaskRotationPublishesJoinedPairsAtEitherBufferCapacity) {
    auto transport = std::make_unique<SwimlaneTransport>();
    transport->arm(51);
    const uint32_t count = std::min(PLATFORM_AICORE_BUFFER_SIZE, PLATFORM_PROF_BUFFER_SIZE);
    ChipSwimlaneAicoreTaskRecord record{100, 200, TaskId::make_global(0), 0, 0};
    for (uint32_t i = 0; i <= count; ++i) {
        record.reg_task_id = i;
        ASSERT_TRUE(chip_swimlane_aicpu_record_staged_task(1, 0, record, 90, 210));
    }
    auto *header = get_chip_swimlane_header(transport->storage.data());
    ASSERT_EQ(header->queue_tails[0], 2u);
    EXPECT_EQ(header->queues[0][0].kind, ChipSwimlaneBufferKind::AicoreTask);
    EXPECT_EQ(header->queues[0][1].kind, ChipSwimlaneBufferKind::AicpuTask);
    EXPECT_EQ(header->queues[0][0].record_count, count);
    EXPECT_EQ(header->queues[0][1].record_count, count);
    const auto saved = transport->core[1][0].records[count - 1];
    header->queue_tails[0] = PLATFORM_PROF_READYQUEUE_SIZE - 2;
    int core = 1;
    chip_swimlane_aicpu_flush_staged(0, &core, 1);
    EXPECT_EQ(header->queue_tails[0], PLATFORM_PROF_READYQUEUE_SIZE - 2);
    auto *a = get_aicore_buffer_state(transport->storage.data(), 1);
    auto *b = get_perf_buffer_state(transport->storage.data(), 1);
    for (auto *head : {&a->head, &b->head}) {
        EXPECT_EQ(head->total_record_count, count + 1);
        EXPECT_EQ(head->published_record_count, count);
        EXPECT_EQ(head->dropped_record_count, 1u);
    }
    EXPECT_EQ(std::memcmp(&saved, &transport->core[1][0].records[count - 1], sizeof(saved)), 0);
}
