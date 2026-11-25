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

// The register-layer tests exercise the real ACK, close, read-back, drain and
// release sequence. This seam verifies that the HBG lifecycle passes the real
// per-core gate on normal shutdown and failed startup.
namespace {
struct RetireCall {
    bool uses_dispatch_register{false};
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
        call.uses_dispatch_register |= targets[i].uses_dispatch_register;
        call.addrs.push_back(targets[i].reg_addr);
        call.gates.push_back(targets[i].teardown);
        if (observed_retirement_runtime != nullptr) {
            const auto worker = targets[i].teardown - observed_retirement_runtime->get_teardown_gates();
            EXPECT_NE(observed_retirement_runtime->get_workers()[worker].aicore_done, 0u);
            EXPECT_NE(observed_retirement_runtime->get_workers()[worker].aicpu_ready, 0u);
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
void publish_aicore_lifecycle_word(volatile uint32_t *word, uint32_t value) {
    __atomic_store_n(word, value, __ATOMIC_RELEASE);
}
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

    static void expect_group(Runtime &runtime) {
        ASSERT_EQ(retire_calls.size(), 1u);
        EXPECT_EQ(retire_calls[0].addrs, std::vector<uint64_t>(kAddrs.begin(), kAddrs.end()));
        for (size_t i = 0; i < kAddrs.size(); ++i)
            EXPECT_EQ(retire_calls[0].gates[i], &runtime.get_teardown_gates()[i]);
    }
};

TEST_F(HbgRetirementWiring, OnlyAcknowledgedReportsAreInvalidatedOnEveryExitRoute) {
    for (int route = 0; route < 2; ++route) {
        for (int failure = 0; failure < 3; ++failure) {
            const bool fail = failure != 0;
            auto runtime = std::make_unique<Runtime>();
            for (int i = 0; i < 3; ++i) {
                auto &report = runtime->get_workers()[i];
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
                const auto &report = runtime->get_workers()[i];
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

TEST_F(HbgRetirementWiring, WorkWindowIsCapturedByRetiringPartitionsAndNotReadAgain) {
    auto runtime = std::make_unique<Runtime>();
    runtime->set_worker_count(3);
    runtime->get_workers()[0].work_start_cycle = 100;
    runtime->get_workers()[0].work_end_cycle = 200;
    runtime->get_workers()[2].work_start_cycle = 150;
    runtime->get_workers()[2].work_end_cycle = 300;
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::seed(lifecycle, kAddrs, 2);
    AicoreWorkWindow result{};
    ASSERT_EQ(lifecycle.finish_shutdown_partition(0, runtime.get()), 0);
    EXPECT_FALSE(lifecycle.retired_work_window(result, true));
    ASSERT_EQ(lifecycle.finish_shutdown_partition(1, runtime.get()), 0);
    // Final aggregation consumes the retired snapshots, not the report lines.
    for (int i = 0; i < 3; ++i) {
        runtime->get_workers()[i].work_start_cycle = 0;
        runtime->get_workers()[i].work_end_cycle = 0;
    }
    ASSERT_TRUE(lifecycle.retired_work_window(result, true));
    EXPECT_EQ(result.start_cycle, 100u);
    EXPECT_EQ(result.end_cycle, 300u);
    EXPECT_FALSE(lifecycle.retired_work_window(result, false));
    EXPECT_EQ(result.start_cycle, 0u);
    EXPECT_EQ(result.end_cycle, 0u);
    ASSERT_EQ(lifecycle.pre_handshake_init(runtime.get(), 2, 0), 0);
    EXPECT_FALSE(lifecycle.retired_work_window(result, true));
}

TEST_F(HbgRetirementWiring, WorkWindowRejectsIncompleteEndpointsAndEntirelyIdleRuns) {
    for (bool incomplete : {false, true}) {
        auto runtime = std::make_unique<Runtime>();
        if (incomplete) {
            runtime->get_workers()[0].work_start_cycle = 100;
            runtime->get_workers()[0].work_end_cycle = 200;
            runtime->get_workers()[2].work_end_cycle = 300;
        }
        AicoreLifecycle lifecycle;
        AicoreLifecycleTestPeer::seed(lifecycle, kAddrs, 2);
        ASSERT_EQ(lifecycle.finish_shutdown_partition(0, runtime.get()), 0);
        ASSERT_EQ(lifecycle.finish_shutdown_partition(1, runtime.get()), 0);
        AicoreWorkWindow result{1, 2};
        EXPECT_FALSE(lifecycle.retired_work_window(result, true));
        EXPECT_EQ(result.start_cycle, 0u);
        EXPECT_EQ(result.end_cycle, 0u);
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
        runtime->get_workers()[i].physical_core_id = i;
        runtime->get_workers()[i].core_type = CoreType::AIV;
        runtime->get_workers()[i].aicore_done = i + 1;
    }
    lifecycle.handshake_partition(runtime.get(), 0, 1);
    ASSERT_TRUE(AicoreLifecycleTestPeer::handshake_failed(lifecycle));
    EXPECT_EQ(lifecycle.release_partition(runtime.get(), 0, false), -SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    EXPECT_EQ(AicoreLifecycleTestPeer::retired_cores(lifecycle), 2);
    EXPECT_FALSE(lifecycle.cores_retired());
    EXPECT_EQ(header.sched_error_code.load(), SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    ASSERT_EQ(retire_calls.size(), 1u);
    EXPECT_EQ(retire_calls[0].addrs, (std::vector<uint64_t>{kAddrs[0], kAddrs[2]}));
    EXPECT_EQ(retire_calls[0].gates[0], &runtime->get_teardown_gates()[0]);
    EXPECT_EQ(retire_calls[0].gates[1], &runtime->get_teardown_gates()[2]);
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
        auto &report = runtime->get_workers()[worker];
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
    EXPECT_EQ(retire_calls[0].gates[0], &runtime->get_teardown_gates()[0]);
    EXPECT_EQ(runtime->get_workers()[0].aicore_done, 0u);
    EXPECT_EQ(runtime->get_workers()[1].aicore_done, 2u);
    EXPECT_EQ(runtime->get_workers()[2].aicore_done, 3u);
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
        auto &report = runtime->get_workers()[worker];
        report.physical_core_id = worker;
        report.core_type = CoreType::AIV;
        report.aicore_done = worker + 1;
    }
    lifecycle.handshake_partition(runtime.get(), 0, 1);
    EXPECT_EQ(lifecycle.validate_discovery(), -SIMPLER_ERROR_INVALID_ARGS);
    EXPECT_EQ(lifecycle.release_partition(runtime.get(), 0, false), -SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    ASSERT_EQ(retire_calls.size(), 1u);
    EXPECT_EQ(retire_calls[0].addrs, (std::vector<uint64_t>{kAddrs[0], kAddrs[2]}));
    EXPECT_EQ(runtime->get_workers()[1].aicore_done, 2u);
    EXPECT_EQ(AicoreLifecycleTestPeer::retired_cores(lifecycle), 2);
    EXPECT_FALSE(lifecycle.cores_retired());
}

TEST_F(HbgRetirementWiring, CompleteHandshakePreservesFailureRetirementError) {
    auto runtime = std::make_unique<Runtime>();
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::prepare_handshake(lifecycle, kAddrs.data());
    for (int i = 0; i < 3; ++i) {
        runtime->get_workers()[i].physical_core_id = i;
        runtime->get_workers()[i].core_type = CoreType::AIV;
        runtime->get_workers()[i].aicore_done = i + 1;
    }
    lifecycle.handshake_partition(runtime.get(), 0, 1);
    ASSERT_FALSE(AicoreLifecycleTestPeer::handshake_failed(lifecycle));
    retirement_result = -1;
    EXPECT_EQ(lifecycle.release_partition(runtime.get(), 0, false), -1);
    expect_group(*runtime);
}

TEST_F(HbgRetirementWiring, ShutdownSignalsAndRetiresTheSameGatedPartition) {
    auto runtime = std::make_unique<Runtime>();
    AicoreLifecycle lifecycle;
    AicoreLifecycleTestPeer::seed(lifecycle, kAddrs);

    lifecycle.signal_shutdown_partition(0);
    EXPECT_EQ(exit_signals, std::vector<uint64_t>(kAddrs.begin(), kAddrs.end()));
    EXPECT_EQ(lifecycle.finish_shutdown_partition(0, runtime.get()), 0);
    expect_group(*runtime);
}

TEST_F(HbgRetirementWiring, ClearsStaleRequestAndReleaseBeforeHandshake) {
    auto runtime = std::make_unique<Runtime>();
    runtime->set_worker_count(3);
    for (size_t i = 0; i < RUNTIME_MAX_WORKER; ++i) {
        auto &gate = runtime->get_teardown_gates()[i];
        gate.post_close_release = AICORE_POST_CLOSE_RELEASE;
        gate.exit_request = AICORE_EXIT_NONCE_1;
    }
    AicoreLifecycle lifecycle;

    ASSERT_EQ(lifecycle.pre_handshake_init(runtime.get(), 1, 0), 0);
    for (size_t i = 0; i < kAddrs.size(); ++i) {
        EXPECT_EQ(runtime->get_teardown_gates()[i].post_close_release, 0u);
        EXPECT_EQ(runtime->get_teardown_gates()[i].exit_request, 0u);
    }
}

TEST_F(HbgRetirementWiring, ReusedLifecycleWordSeparatesStartExitAndRejectedLaunchAcrossWrap) {
    auto runtime = std::make_unique<Runtime>();
    runtime->set_worker_count(3);
    alignas(32) uint32_t notification = UINT32_MAX;
    runtime->dev.scheduler_bootstrap.lifecycle_notify_address = reinterpret_cast<uint64_t>(&notification);
    AicoreLifecycle lifecycle;
    for (bool start : {true, false, true}) {
        const uint32_t previous = notification;
        ASSERT_EQ(lifecycle.pre_handshake_init(runtime.get(), 1, 0), 0);
        AicoreLifecycleTestPeer::seed(lifecycle, kAddrs);
        ASSERT_EQ(notification, previous);
        if (start) {
            lifecycle.publish_lifecycle_start();
            const uint32_t running = notification;
            EXPECT_NE(running, previous);
            EXPECT_EQ(running & AICORE_LIFECYCLE_EXIT_BIT, 0u);
            EXPECT_EQ(lifecycle.release_partition(runtime.get(), 0, true), 0);
            lifecycle.signal_shutdown_partition(0);
            EXPECT_EQ(notification, running | AICORE_LIFECYCLE_EXIT_BIT);
            EXPECT_EQ(lifecycle.finish_shutdown_partition(0, runtime.get()), 0);
        } else {
            EXPECT_EQ(lifecycle.release_partition(runtime.get(), 0, false), 0);
        }
        EXPECT_NE(notification, previous);
        EXPECT_NE(notification & AICORE_LIFECYCLE_EXIT_BIT, 0u);
        EXPECT_TRUE(lifecycle.cores_retired());
        EXPECT_TRUE(exit_signals.empty());
        ASSERT_FALSE(retire_calls.empty());
        EXPECT_FALSE(retire_calls.back().uses_dispatch_register);
        lifecycle.deinit();
    }
}

TEST_F(HbgRetirementWiring, RejectedDiscoveryOnlyMarksWorkersWithoutRetirementTargets) {
    auto runtime = std::make_unique<Runtime>();
    runtime->set_worker_count(3);
    alignas(32) uint32_t notification = UINT32_MAX;
    alignas(64) std::array<uint32_t, 3> requests{};
    runtime->dev.scheduler_bootstrap.lifecycle_notify_address = reinterpret_cast<uint64_t>(&notification);
    runtime->dev.scheduler_bootstrap.rejected_exit_address = reinterpret_cast<uint64_t>(requests.data());
    AicoreLifecycle lifecycle;
    ASSERT_EQ(lifecycle.pre_handshake_init(runtime.get(), 1, 0), 0);
    AicoreLifecycleTestPeer::seed(lifecycle, {kAddrs[0], 0, kAddrs[0]});
    ASSERT_NE(lifecycle.validate_discovery(), 0);
    EXPECT_EQ(notification, UINT32_MAX);
    EXPECT_EQ(requests, (std::array<uint32_t, 3>{0, 1, 1}));
    EXPECT_NE(lifecycle.release_partition(runtime.get(), 0, false), 0);
    EXPECT_EQ(notification, requests[1]);
    ASSERT_EQ(retire_calls.size(), 1u);
    EXPECT_EQ(retire_calls[0].addrs, std::vector<uint64_t>{kAddrs[0]});
    EXPECT_FALSE(lifecycle.cores_retired());
}

TEST_F(HbgRetirementWiring, FailedStartupStillRetiresThroughPerCoreGates) {
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
