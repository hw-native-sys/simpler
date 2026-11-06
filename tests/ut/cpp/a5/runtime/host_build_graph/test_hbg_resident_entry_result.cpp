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

#include "aicpu_executor.cpp"  // NOLINT(bugprone-suspicious-include): exercises the production TU's entry owner

#include <gtest/gtest.h>

#include <array>
#include <condition_variable>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>

#include "aicpu/pmu_collector_aicpu.h"
#include "common/args_dump.h"
#include "common/native_invocation_evidence.h"
#include "async_transport_result.h"

namespace {

constexpr int32_t kParticipants = 3;
constexpr uint64_t kEpoch = 0x5100'0000'0000'0011ULL;
constexpr std::array<uint64_t, kParticipants> kRegisters{0x1000, 0x2000, 0x3000};

struct SchedulerImage {
    std::array<SchedulerWorkerContext, kParticipants> contexts{};
    SchedulerRunControl control{};
    SchedulerGraphSummary summary{};
    std::array<AicpuThreadLifecycleTrace, kParticipants> traces{};
    SchedulerTaskMetadata metadata{};
    alignas(128) SchedulerAsyncLayout async_layout{};
};

struct RegisterTransport {
    Runtime *runtime = nullptr;
    SharedMemoryHeader *header = nullptr;
    SchedulerImage *image = nullptr;
    DeviceRunResultRegion *result = nullptr;
    std::mutex mutex;
    std::condition_variable changed;
    std::array<bool, kParticipants> returned{};
    std::array<int, kParticipants> opened{};
    std::array<int, kParticipants> signaled{};
    std::array<int, kParticipants> retired{};
    int held_participant = -1;
    int failed_participant = -1;
    int32_t header_before_late = SIMPLER_ERROR_NONE;
    bool late_error = false;
    bool invalidate_graph_on_release = false;
    bool invalidated = false;
    bool saw_held_owner = false;

    bool peers_returned(int32_t held) const {
        for (int32_t i = 0; i < kParticipants; ++i)
            if (i != held && !returned[i]) return false;
        return true;
    }
};

RegisterTransport *transport = nullptr;

int register_index(uint64_t address) {
    for (int i = 0; i < kParticipants; ++i)
        if (kRegisters[i] == address) return i;
    ADD_FAILURE() << "unexpected register address " << address;
    return 0;
}

}  // namespace

extern "C" uint64_t get_platform_regs() { return reinterpret_cast<uint64_t>(kRegisters.data()); }
uint32_t platform_get_physical_cores_count() { return kParticipants; }
uint64_t platform_aicore_exit_deadline() { return std::numeric_limits<uint64_t>::max(); }
void write_reg(uint64_t, RegId, uint64_t) {}
uint64_t read_reg(uint64_t, RegId) { return 0; }

void platform_init_aicore_regs(uint64_t address) {
    std::lock_guard<std::mutex> lock(transport->mutex);
    ++transport->opened[register_index(address)];
    if (transport->invalidate_graph_on_release && !transport->invalidated) {
        transport->runtime->dev.host_total_tasks = -1;
        transport->invalidated = true;
    }
}

void platform_signal_aicore_exit(uint64_t address) {
    std::lock_guard<std::mutex> lock(transport->mutex);
    ++transport->signaled[register_index(address)];
}

int32_t platform_retire_aicore_group(const AicoreExitTarget *targets, size_t count, uint64_t, bool *released) {
    const int32_t participant = platform_aicpu_affinity_thread_idx();
    std::unique_lock<std::mutex> lock(transport->mutex);
    if (participant == transport->held_participant) {
        transport->changed.wait(lock, [participant] {
            return transport->peers_returned(participant);
        });
        EXPECT_TRUE(g_aicpu_executor.init_done_.load());
        EXPECT_EQ(transport->result->published, 0u);
        EXPECT_EQ(g_aicpu_executor.normal_path_claims_.load(), kParticipants - 1);
        transport->saw_held_owner = true;
        if (transport->late_error) {
            EXPECT_EQ(g_aicpu_executor.run_status_.load(), 0);
            EXPECT_EQ(transport->header->sched_error_code.load(), transport->header_before_late);
            EXPECT_EQ(transport->image->control.scheduler_error, 0u);
            EXPECT_TRUE(record_aicore_scheduler_runtime_error(
                &transport->image->control, SchedulerGraphResult::TIMEOUT, SchedulerErrorSite::EXIT_WAIT_TIMEOUT
            ));
        }
    }
    const bool success = participant != transport->failed_participant;
    for (size_t i = 0; i < count; ++i) {
        const int core = register_index(targets[i].reg_addr);
        EXPECT_EQ(targets[i].teardown, &transport->runtime->dev.teardown_gates[core]);
        ++transport->retired[core];
        if (released != nullptr) released[i] = success;
    }
    return success ? 0 : -1;
}

// This fixture owns no enabled argument-dump producer.
extern "C" bool is_dump_args_enabled() { return false; }
void dump_args_flush(int) {}
void dump_args_close_run(int) {}
void dump_args_init(int, DumpArgsLevel) {}

extern "C" void aicpu_reset_legacy_execution_entry() {}

extern "C" int32_t legacy_aicpu_execute(Runtime *) {
    ADD_FAILURE() << "resident entry reached legacy execution";
    return -1;
}

namespace {

class ResidentEntryResult : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_FALSE(g_aicpu_executor.init_done_.load());
        set_chip_swimlane_enabled(false);
        set_pmu_enabled(false);
        set_scheduler_timeout_ms(5000);
        configure(SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING);
    }

    void TearDown() override {
        EXPECT_FALSE(g_aicpu_executor.init_done_.load());
        EXPECT_EQ(aicpu_release_callables(), 0);
        set_platform_run_result(0, 0);
        set_scheduler_timeout_ms(0);
        transport = nullptr;
    }

    void configure(uint32_t mode) {
        runtime = std::make_unique<Runtime>();
        image = std::make_unique<SchedulerImage>();
        state = std::make_unique<RegisterTransport>();
        header.sched_error_code.store(SIMPLER_ERROR_NONE);
        result_storage = {};
        private_result = true;
        expect_transport_normalized = false;
        runtime->set_worker_count(kParticipants);
        runtime->set_gm_sm_ptr(&header);
        runtime->dev.aicpu_thread_num = kParticipants;
        runtime->dev.host_total_tasks = 1;
        for (int32_t i = 0; i < kParticipants; ++i) {
            auto &context = image->contexts[i];
            context.worker_index = static_cast<uint64_t>(i);
            context.scheduler_state_base_address = reinterpret_cast<uint64_t>(image.get());
            context.worker_contexts_offset = offsetof(SchedulerImage, contexts);
            context.run_control_offset = offsetof(SchedulerImage, control);
            context.graph_summary_offset = offsetof(SchedulerImage, summary);
            context.aicpu_lifecycle_traces_offset = offsetof(SchedulerImage, traces);
            context.task_metadata_offset = offsetof(SchedulerImage, metadata);
            context.graph_task_count = 1;
            context.scheduler_state_bytes = sizeof(SchedulerImage);
            context.runtime_worker_count = kParticipants;
            auto &report = runtime->dev.workers[i];
            report.physical_core_id = i;
            report.core_type = i == 0 ? CoreType::AIC : CoreType::AIV;
            report.aicore_done = i + 1;
            report.report_epoch = kEpoch;
        }
        image->control.bootstrap_complete = 1;
        image->control.expected_task_count = 1;
        image->control.resolved_task_count = 1;
        image->control.aic_worker_demand = 1;
        image->control.aiv_worker_demand = 2;
        image->summary.run_epoch = kEpoch;
        image->summary.worker_count = kParticipants;
        image->summary.cluster_count = 1;
        image->metadata.flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_MIX;
        image->metadata.active_mask = LaneMask::all();
        image->metadata.logical_block_num = 1;
        image->metadata.timing_slot = -1;
        runtime->publish_scheduler_bootstrap(mode, reinterpret_cast<uint64_t>(image->contexts.data()));
        state->runtime = runtime.get();
        state->header = &header;
        state->image = image.get();
        state->result = &result;
        transport = state.get();
        set_platform_run_result(reinterpret_cast<uint64_t>(&result), kEpoch);
    }

    void enable_async_transport() {
        image->summary.async_layout_offset = offsetof(SchedulerImage, async_layout);
        image->async_layout.run_epoch = kEpoch;
        image->async_layout.total_bytes = sizeof(SchedulerAsyncLayout);
        image->async_layout.task_count = 1;
        image->async_layout.group_count = 1;
        image->async_layout.finish_count = 1;
    }

    std::array<int32_t, kParticipants> execute() {
        const uint64_t result_base = private_result ? reinterpret_cast<uint64_t>(&result_storage) : 0;
        const uint64_t result_epoch = private_result ? kEpoch : 0;
        const bool async_transport = aicpu_async_transport_enabled(runtime.get(), result_epoch);
        std::array<int32_t, kParticipants> transport_statuses{};
        std::array<int32_t, kParticipants> statuses{};
        std::array<int32_t, kParticipants> native_statuses{};
        EXPECT_EQ(
            aicpu_begin_native_invocation(
                kParticipants, private_result ? reinterpret_cast<uint64_t>(&result_storage) : 0,
                private_result ? kEpoch : 0
            ),
            0
        );
        aicpu_reset_execution_entry();
        std::array<std::thread, kParticipants> threads;
        for (int32_t i = 0; i < kParticipants; ++i) {
            threads[i] = std::thread([&, i] {
                platform_aicpu_affinity_set_thread_idx(i);
                statuses[i] = aicpu_execute(runtime.get());
                // Keep the runtime's own status assertions and additionally
                // exercise the production owner that publishes after teardown.
                native_statuses[i] = aicpu_finish_native_invocation(i, statuses[i]);
                transport_statuses[i] =
                    a5_async_transport_result(async_transport, result_base, result_epoch, native_statuses[i]);
                {
                    std::lock_guard<std::mutex> lock(state->mutex);
                    state->returned[i] = true;
                }
                state->changed.notify_all();
            });
        }
        for (auto &thread : threads)
            thread.join();
        // Each case independently asserts these runtime statuses. Do not
        // derive the expected native return from the record being verified.
        int expected_error = 0;
        for (int status : statuses)
            if (status != 0) {
                expected_error = status;
                break;
            }
        if (private_result) {
            const auto terminal = device_run_result_terminal(result, kEpoch);
            EXPECT_EQ(
                terminal.state, expected_error != 0 ? DeviceRunTerminalState::Failed : DeviceRunTerminalState::Succeeded
            );
            EXPECT_EQ(terminal.code, expected_error);
        }
        int failed_callbacks = 0;
        for (int status : native_statuses) {
            EXPECT_TRUE(status == 0 || status == expected_error);
            failed_callbacks += status != 0;
        }
        EXPECT_EQ(failed_callbacks, expected_error != 0 ? 1 : 0);
        if (state->held_participant >= 0) EXPECT_EQ(native_statuses[state->held_participant], expected_error);
        for (int32_t i = 0; i < kParticipants; ++i)
            EXPECT_EQ(transport_statuses[i], expect_transport_normalized ? 0 : native_statuses[i]);
        return statuses;
    }

    void expect_normal_lifecycle() {
        EXPECT_EQ(state->opened, (std::array<int, kParticipants>{1, 1, 1}));
        EXPECT_EQ(state->signaled, (std::array<int, kParticipants>{1, 1, 1}));
        EXPECT_EQ(state->retired, (std::array<int, kParticipants>{1, 1, 1}));
        EXPECT_EQ(image->control.active_worker_count, kParticipants);
        EXPECT_EQ(image->control.aic_active_worker_count, 1u);
        EXPECT_EQ(image->control.aiv_active_worker_count, 2u);
        EXPECT_FALSE(g_aicpu_executor.init_done_.load());
        EXPECT_EQ(aicpu_release_callables(), 0);
    }

    std::unique_ptr<Runtime> runtime;
    std::unique_ptr<SchedulerImage> image;
    std::unique_ptr<RegisterTransport> state;
    SharedMemoryHeader header{};
    NativeInvocationResultStorage result_storage{};
    DeviceRunResultRegion &result{result_storage.result};
    bool private_result{true};
    bool expect_transport_normalized{false};
};

TEST_F(ResidentEntryResult, PublishedAsyncHeaderFailuresKeepSemanticStatusAndUseSuccessfulTransport) {
    for (const int32_t code :
         {SIMPLER_ERROR_SCHEDULER_TIMEOUT, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW,
          SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED}) {
        for (int32_t last = 0; last < kParticipants; ++last) {
            SCOPED_TRACE(code);
            SCOPED_TRACE(last);
            configure(SCHEDULER_RUNTIME_MODE_RESIDENT_READY);
            enable_async_transport();
            ASSERT_TRUE(aicpu_async_transport_enabled(runtime.get(), kEpoch));
            state->held_participant = last;
            header.sched_error_code.store(code);
            expect_transport_normalized = true;
            EXPECT_EQ(execute(), (std::array<int32_t, kParticipants>{-code, -code, -code}));
            expect_normal_lifecycle();
            EXPECT_TRUE(state->saw_held_owner);
            EXPECT_TRUE(hbg_async_cores_retired(result.payload, result.payload_bytes, kEpoch));
            EXPECT_EQ(device_run_result_terminal(result, kEpoch).source, DeviceRunCodeSource::Header);
        }
    }
}

TEST_F(ResidentEntryResult, AsyncTransportStillRefusesUnretiredUnknownAndMissingPrivateResult) {
    enable_async_transport();
    state->failed_participant = 0;
    header.sched_error_code.store(SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
    EXPECT_EQ(execute(), (std::array<int32_t, kParticipants>{-102, -102, -102}));
    EXPECT_FALSE(hbg_async_cores_retired(result.payload, result.payload_bytes, kEpoch));

    configure(SCHEDULER_RUNTIME_MODE_RESIDENT_READY);
    enable_async_transport();
    header.sched_error_code.store(999);
    EXPECT_EQ(execute(), (std::array<int32_t, kParticipants>{-999, -999, -999}));

    configure(SCHEDULER_RUNTIME_MODE_RESIDENT_READY);
    enable_async_transport();
    private_result = false;
    header.sched_error_code.store(SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
    EXPECT_EQ(execute(), (std::array<int32_t, kParticipants>{-102, -102, -102}));
    EXPECT_EQ(result.published, 0u);
}

TEST_F(ResidentEntryResult, AsyncQualificationUsesOnlyThisInvocationsBoundedHostImage) {
    EXPECT_FALSE(aicpu_async_transport_enabled(runtime.get(), kEpoch));
    enable_async_transport();
    ASSERT_TRUE(aicpu_async_transport_enabled(runtime.get(), kEpoch));
    EXPECT_FALSE(aicpu_async_transport_enabled(nullptr, kEpoch));
    EXPECT_FALSE(aicpu_async_transport_enabled(runtime.get(), 0));
    EXPECT_FALSE(aicpu_async_transport_enabled(runtime.get(), kEpoch + 1));
    const auto saved_summary = image->summary;
    const auto saved_layout = image->async_layout;
    auto &context = image->contexts[0];
    const uint64_t saved_bytes = context.scheduler_state_bytes;
    context.scheduler_state_bytes = offsetof(SchedulerImage, async_layout) + sizeof(SchedulerAsyncLayout) - 1;
    EXPECT_FALSE(aicpu_async_transport_enabled(runtime.get(), kEpoch));
    context.scheduler_state_bytes = saved_bytes;
    image->summary.async_layout_offset += 8;
    EXPECT_FALSE(aicpu_async_transport_enabled(runtime.get(), kEpoch));
    image->summary = saved_summary;
    ++image->summary.run_epoch;
    EXPECT_FALSE(aicpu_async_transport_enabled(runtime.get(), kEpoch));
    image->summary = saved_summary;
    for (int field = 0; field < 5; ++field) {
        image->async_layout = saved_layout;
        if (field == 0) ++image->async_layout.version;
        if (field == 1) ++image->async_layout.run_epoch;
        if (field == 2) image->async_layout.group_count = 0;
        if (field == 3) image->async_layout.total_bytes = UINT64_MAX;
        if (field == 4) ++image->async_layout.task_count;
        EXPECT_FALSE(aicpu_async_transport_enabled(runtime.get(), kEpoch));
    }
    image->async_layout = saved_layout;
    runtime->dev.scheduler_bootstrap.runtime_mode = SCHEDULER_RUNTIME_MODE_LEGACY_GRAPH;
    EXPECT_FALSE(aicpu_async_transport_enabled(runtime.get(), kEpoch));
}

TEST_F(ResidentEntryResult, BothResidentModesReturnSuccessThroughTheMixedTopologyLifecycle) {
    for (uint32_t mode : {SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING, SCHEDULER_RUNTIME_MODE_RESIDENT_READY}) {
        SCOPED_TRACE(mode);
        configure(mode);
        EXPECT_EQ(execute(), (std::array<int32_t, kParticipants>{0, 0, 0}));
        expect_normal_lifecycle();
        EXPECT_EQ(device_run_result_terminal(result, kEpoch).state, DeviceRunTerminalState::Succeeded);
    }
}

TEST_F(ResidentEntryResult, TheActualLastEntryReturnsAnErrorClaimedAfterItsPeersReturned) {
    for (int32_t last = 0; last < kParticipants; ++last) {
        SCOPED_TRACE(last);
        configure(SCHEDULER_RUNTIME_MODE_RESIDENT_READY);
        state->held_participant = last;
        state->late_error = true;
        const auto statuses = execute();
        EXPECT_TRUE(state->saw_held_owner);
        for (int32_t i = 0; i < kParticipants; ++i)
            EXPECT_EQ(statuses[i], i == last ? -SIMPLER_ERROR_SCHEDULER_TIMEOUT : 0);
        EXPECT_EQ(header.sched_error_code.load(), SIMPLER_ERROR_SCHEDULER_TIMEOUT);
        expect_normal_lifecycle();
        configure(SCHEDULER_RUNTIME_MODE_RESIDENT_READY);
        EXPECT_EQ(execute(), (std::array<int32_t, kParticipants>{0, 0, 0}));
        EXPECT_EQ(device_run_result_terminal(result, kEpoch).state, DeviceRunTerminalState::Succeeded);
    }
}

TEST_F(ResidentEntryResult, AnEarlierHeaderErrorStillOwnsTheActualLateEntryStatus) {
    state->held_participant = 0;
    state->late_error = true;
    state->header_before_late = SIMPLER_ERROR_INVALID_ARGS;
    header.sched_error_code.store(SIMPLER_ERROR_INVALID_ARGS);
    EXPECT_EQ(
        execute(), (std::array<int32_t, kParticipants>{
                       -SIMPLER_ERROR_INVALID_ARGS, -SIMPLER_ERROR_INVALID_ARGS, -SIMPLER_ERROR_INVALID_ARGS
                   })
    );
    EXPECT_EQ(header.sched_error_code.load(), SIMPLER_ERROR_INVALID_ARGS);
    expect_normal_lifecycle();
}

TEST_F(ResidentEntryResult, TheLateEntryStatusDoesNotDependOnPublishingAPrivateResult) {
    state->held_participant = 2;
    state->late_error = true;
    private_result = false;
    EXPECT_EQ(execute(), (std::array<int32_t, kParticipants>{0, 0, -SIMPLER_ERROR_SCHEDULER_TIMEOUT}));
    EXPECT_EQ(result.published, 0u);
    expect_normal_lifecycle();
}

TEST_F(ResidentEntryResult, AShutdownFailureReachesItsEntryEvenWhenAnotherParticipantFinishesLast) {
    state->held_participant = 1;
    state->failed_participant = 0;
    EXPECT_EQ(execute(), (std::array<int32_t, kParticipants>{-1, 0, 0}));
    EXPECT_EQ(header.sched_error_code.load(), SIMPLER_ERROR_NONE);
    expect_normal_lifecycle();
    const auto terminal = device_run_result_terminal(result, kEpoch);
    EXPECT_EQ(terminal.state, DeviceRunTerminalState::Failed);
    EXPECT_EQ(terminal.source, DeviceRunCodeSource::ShutdownRc);
}

TEST_F(ResidentEntryResult, AnInvalidGraphAtRunEntryReturnsTheSupervisorsOwnFailure) {
    state->invalidate_graph_on_release = true;
    EXPECT_EQ(execute(), (std::array<int32_t, kParticipants>{-1, -1, -1}));
    EXPECT_EQ(header.sched_error_code.load(), SIMPLER_ERROR_NONE);
    expect_normal_lifecycle();
}

TEST_F(ResidentEntryResult, FailedTopologyInitializationReturnsThroughTheActualAbortCleanup) {
    runtime->dev.workers[2].core_type = CoreType::AIC;
    EXPECT_EQ(execute(), (std::array<int32_t, kParticipants>{-1, -1, -1}));
    EXPECT_EQ(state->opened, (std::array<int, kParticipants>{0, 0, 0}));
    EXPECT_EQ(state->signaled, (std::array<int, kParticipants>{0, 0, 0}));
    EXPECT_EQ(state->retired, (std::array<int, kParticipants>{1, 1, 1}));
    const auto terminal = device_run_result_terminal(result, kEpoch);
    EXPECT_EQ(terminal.state, DeviceRunTerminalState::Failed);
    EXPECT_EQ(terminal.code, -1);
    EXPECT_EQ(terminal.source, DeviceRunCodeSource::ThreadRc);
    EXPECT_EQ(aicpu_release_callables(), 0);
    configure(SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING);
    EXPECT_EQ(execute(), (std::array<int32_t, kParticipants>{0, 0, 0}));
    expect_normal_lifecycle();
    EXPECT_EQ(device_run_result_terminal(result, kEpoch).state, DeviceRunTerminalState::Succeeded);
}

}  // namespace

TEST(HbgResidentEntryInitialization, ClearsCoordinationBeforeRuntimeInitialization) {
    auto &executor = g_aicpu_executor;
    executor.init_done_ = true;
    executor.init_failed_ = true;
    executor.hs_setup_done_ = true;
    executor.hs_config_done_ = true;
    executor.hs_bootstrap_done_ = true;
    executor.shutdown_ready_ = true;
    executor.thread_idx_ = 4;
    executor.hs_thread_seq_ = 4;
    executor.hs_arrived_ = 4;
    executor.hs_release_arrived_ = 4;
    executor.init_failure_arrived_ = 4;
    executor.shutdown_signaled_ = 4;
    executor.run_status_ = -1;
    executor.normal_path_claims_ = 4;
    executor.terminal_publisher_.take({DeviceRunVerdict::Ok, 0, DeviceRunCodeSource::None}, nullptr, 0);
    aicpu_reset_execution_entry();
    DeviceRunResultRegion result{};
    EXPECT_TRUE(executor.terminal_publisher_.publish(reinterpret_cast<uint64_t>(&result), 7));
    EXPECT_EQ(result.published, 7u);
    EXPECT_FALSE(executor.init_done_.load());
    EXPECT_FALSE(executor.init_failed_.load());
    EXPECT_FALSE(executor.hs_setup_done_.load());
    EXPECT_FALSE(executor.hs_config_done_.load());
    EXPECT_FALSE(executor.hs_bootstrap_done_.load());
    EXPECT_FALSE(executor.shutdown_ready_.load());
    EXPECT_EQ(executor.thread_idx_.load(), 0);
    EXPECT_EQ(executor.hs_thread_seq_.load(), 0);
    EXPECT_EQ(executor.hs_arrived_.load(), 0);
    EXPECT_EQ(executor.hs_release_arrived_.load(), 0);
    EXPECT_EQ(executor.init_failure_arrived_.load(), 0);
    EXPECT_EQ(executor.shutdown_signaled_.load(), 0);
    EXPECT_EQ(executor.run_status_.load(), 0);
    EXPECT_EQ(executor.normal_path_claims_.load(), 0);
}

TEST(HbgAsyncTransportResult, NativeFinalOwnerPublicationDoesNotAuthorizeOtherErrorsOrMalformedEvidence) {
    enum class Case {
        VALID,
        NOT_ASYNC,
        NO_RESULT,
        ZERO_EPOCH,
        STALE_EPOCH,
        UNPUBLISHED,
        CODE_MISMATCH,
        THREAD_SOURCE,
        SHUTDOWN_SOURCE,
        UNKNOWN_CODE,
        SDK_CODE,
        NO_RETIREMENT,
        BAD_MAGIC,
        BAD_VERSION,
        OLD_PAYLOAD_EPOCH,
        BAD_FLAGS,
        SHORT_PAYLOAD,
        LONG_PAYLOAD
    };
    for (Case test :
         {Case::VALID, Case::NOT_ASYNC, Case::NO_RESULT, Case::ZERO_EPOCH, Case::STALE_EPOCH, Case::UNPUBLISHED,
          Case::CODE_MISMATCH, Case::THREAD_SOURCE, Case::SHUTDOWN_SOURCE, Case::UNKNOWN_CODE, Case::SDK_CODE,
          Case::NO_RETIREMENT, Case::BAD_MAGIC, Case::BAD_VERSION, Case::OLD_PAYLOAD_EPOCH, Case::BAD_FLAGS,
          Case::SHORT_PAYLOAD, Case::LONG_PAYLOAD}) {
        SCOPED_TRACE(static_cast<int>(test));
        NativeInvocationResultStorage storage{};
        const uint64_t base = test == Case::NO_RESULT ? 0 : reinterpret_cast<uint64_t>(&storage);
        const uint64_t epoch = base == 0 ? 0 : kEpoch;
        ASSERT_EQ(aicpu_begin_native_invocation(2, base, epoch), 0);
        HbgAsyncRetirementPayload retirement{
            HBG_ASYNC_RETIREMENT_MAGIC, kEpoch, HBG_ASYNC_RETIREMENT_VERSION, HBG_ASYNC_CORES_RETIRED
        };
        if (test == Case::NO_RETIREMENT) retirement.flags = 0;
        if (test == Case::BAD_MAGIC) retirement.magic = 0;
        if (test == Case::BAD_VERSION) ++retirement.version;
        if (test == Case::OLD_PAYLOAD_EPOCH) --retirement.run_epoch;
        if (test == Case::BAD_FLAGS) retirement.flags |= 2;
        int32_t code = -SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW;
        if (test == Case::UNKNOWN_CODE) code = -999;
        if (test == Case::SDK_CODE) code = 507018;
        const auto source = test == Case::THREAD_SOURCE   ? DeviceRunCodeSource::ThreadRc :
                            test == Case::SHUTDOWN_SOURCE ? DeviceRunCodeSource::ShutdownRc :
                                                            DeviceRunCodeSource::Header;
        ASSERT_TRUE(
            aicpu_accept_runtime_terminal({DeviceRunVerdict::Error, code, source}, &retirement, sizeof(retirement))
        );
        ASSERT_EQ(aicpu_finish_native_invocation(0, code), 0);
        EXPECT_EQ(storage.result.published, 0u);
        // A non-final callback must not inspect an untrusted/unpublished address.
        EXPECT_EQ(a5_async_transport_result(true, 1, epoch, 0), 0);
        const int32_t raw = aicpu_finish_native_invocation(1, code);
        ASSERT_EQ(raw, code);
        if (test == Case::UNPUBLISHED) storage.result.published = 0;
        if (test == Case::CODE_MISMATCH) storage.result.completion_code = -SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
        if (test == Case::SHORT_PAYLOAD) --storage.result.payload_bytes;
        if (test == Case::LONG_PAYLOAD) ++storage.result.payload_bytes;
        const uint64_t observed_epoch = test == Case::ZERO_EPOCH ? 0 : test == Case::STALE_EPOCH ? epoch + 1 : epoch;
        EXPECT_EQ(
            a5_async_transport_result(test != Case::NOT_ASYNC, base, observed_epoch, raw), test == Case::VALID ? 0 : raw
        );
        if (test == Case::VALID) {
            const auto terminal = device_run_result_terminal(storage.result, kEpoch);
            EXPECT_EQ(terminal.state, DeviceRunTerminalState::Failed);
            EXPECT_EQ(terminal.code, -SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
        }
    }
    set_platform_run_result(0, 0);
}
