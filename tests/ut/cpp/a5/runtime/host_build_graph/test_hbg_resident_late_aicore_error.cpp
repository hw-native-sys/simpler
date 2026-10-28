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
// What a5 host_build_graph's resident executor publishes when the AICore claims
// its error late.
//
// The AICore's exit-wait watchdog claims `scheduler_error` after the
// supervisor's last look at it, then acknowledges exit normally — so the
// AICPU's shutdown succeeds, every participant's rc is zero, every participant
// claims the audited path, and the shared header is the only place that error
// can still land.
//
// These cases call `AicpuExecutor::snapshot_run_terminal` and
// `AicpuExecutor::publish_run_terminal` **themselves**, on a real executor, and
// read the record out of a host region. So they cover the call site, not only
// the rule it calls: reverting the finalizer to a bare header read fails the
// first case.
//
// The executor's struct is defined in its own translation unit, so this test
// compiles that unit rather than duplicating anything from it. Nothing here
// runs `run()`, touches a register, or starts a thread: the finalizer is
// entered directly with the state a finished run leaves behind. That the
// production call sits inside the completion gate's finalizer — the
// all-participant boundary — is reviewed statically and is not what these cases
// establish.
#include "aicpu_executor.cpp"  // NOLINT(bugprone-suspicious-include): drives the executor in its own TU

#include <gtest/gtest.h>

#include <cstdint>

namespace {

constexpr uint64_t kEpoch = 0x5100'0000'0000'0007ULL;

/** A finished resident run: one participant, the audited path claimed, nothing failed. */
class ResidentFinalizer : public ::testing::Test {
protected:
    void SetUp() override {
        context.scheduler_state_base_address = reinterpret_cast<uint64_t>(&run_control);
        context.run_control_offset = 0;
        runtime.set_worker_count(8);
        runtime.set_gm_sm_ptr(&header_storage);
        runtime.publish_scheduler_bootstrap(
            SCHEDULER_RUNTIME_MODE_RESIDENT_READY, reinterpret_cast<uint64_t>(&context)
        );
        executor.aicpu_thread_num_ = 1;
        executor.normal_path_claims_.store(1);
        set_platform_run_result(reinterpret_cast<uint64_t>(&region), kEpoch);
    }

    std::atomic<int32_t> &header() { return header_storage.sched_error_code; }

    /** What the AICore's exit-wait watchdog does after the supervisor stopped looking. */
    bool exit_wait_watchdog_fires() {
        return record_aicore_scheduler_runtime_error(
            &run_control, SchedulerGraphResult::TIMEOUT, SchedulerErrorSite::EXIT_WAIT_TIMEOUT
        );
    }

    /** The finalizer, then the cleanup owner's commit — both the executor's own. */
    DeviceRunTerminal finalize_and_publish(bool cores_retired = true) {
        executor.snapshot_run_terminal(&runtime, cores_retired);
        executor.publish_run_terminal();
        return device_run_result_terminal(region, kEpoch);
    }

    SharedMemoryHeader header_storage{};
    SchedulerWorkerContext context{};
    SchedulerRunControl run_control{};
    DeviceRunResultRegion region{};
    Runtime runtime{};
    AicpuExecutor executor{};
};

// The case the fold exists for. Without it the finalizer reads a clean header
// and publishes Ok for a run the AICore recorded a timeout against.
TEST_F(ResidentFinalizer, AnErrorClaimedAfterTheSupervisorsReadStillDecidesTheRun) {
    ASSERT_TRUE(exit_wait_watchdog_fires());
    ASSERT_EQ(header().load(), SIMPLER_ERROR_NONE) << "nothing has folded it yet";

    const DeviceRunTerminal terminal = finalize_and_publish();

    EXPECT_EQ(terminal.state, DeviceRunTerminalState::Failed);
    EXPECT_EQ(terminal.code, -SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    EXPECT_EQ(terminal.source, DeviceRunCodeSource::Header);
}

// Settling is not a licence to invent: a control the AICore left clean still
// publishes the success every participant claimed.
TEST_F(ResidentFinalizer, ACleanControlPublishesTheSuccessItClaimed) {
    const DeviceRunTerminal terminal = finalize_and_publish();

    EXPECT_EQ(terminal.state, DeviceRunTerminalState::Succeeded);
    EXPECT_EQ(terminal.code, 0);
    EXPECT_EQ(header().load(), SIMPLER_ERROR_NONE) << "no write on a clean path";
}

TEST_F(ResidentFinalizer, FailedInitializationReleasesOnlyItsUnusedLocalServiceAllocation) {
    ASSERT_TRUE(executor.begin_args_dump_initialization());
    executor.args_dump_services_ = new ResidentArgsDumpService[1];
    executor.args_dump_service_count_ = 1;
    executor.args_dump_services_allocated_this_init_ = true;
    executor.init_failed_.store(true);
    EXPECT_EQ(executor.finish_failed_init(&runtime), -1);
    EXPECT_EQ(executor.args_dump_services_, nullptr);
    EXPECT_EQ(executor.args_dump_service_count_, 0);
    EXPECT_FALSE(executor.args_dump_services_allocated_this_init_);
    // The actual next-init admission guard no longer rejects a healthy/OFF
    // attempt because of this stopped local allocation. No device poison or
    // shared allocation is discharged by this gate.
    EXPECT_TRUE(executor.begin_args_dump_initialization());
}

TEST_F(ResidentFinalizer, FailedInitializationPreservesAPriorUnknownServiceAllocation) {
    executor.args_dump_services_ = new ResidentArgsDumpService[1];
    executor.args_dump_service_count_ = 1;
    auto *retained = executor.args_dump_services_;
    ASSERT_FALSE(executor.begin_args_dump_initialization());
    executor.init_failed_.store(true);
    EXPECT_EQ(executor.finish_failed_init(&runtime), -1);
    EXPECT_EQ(executor.args_dump_services_, retained);
    EXPECT_EQ(executor.args_dump_service_count_, 1);
    EXPECT_FALSE(executor.begin_args_dump_initialization());
    delete[] retained;
    executor.args_dump_services_ = nullptr;
    executor.args_dump_service_count_ = 0;
}

TEST_F(ResidentFinalizer, FailedInitializationCannotAbandonAnOlderOpenProducerReservation) {
    const size_t bytes = calc_dump_data_size(1);
    std::unique_ptr<void, decltype(&std::free)> shared(std::aligned_alloc(64, (bytes + 63) / 64 * 64), &std::free);
    ASSERT_NE(shared, nullptr);
    std::memset(shared.get(), 0, bytes);
    auto *dump_header = get_dump_header(shared.get());
    dump_header->magic = ARGS_DUMP_MAGIC;
    dump_header->num_dump_threads = 1;
    auto metadata = std::make_unique<DumpMetaBuffer>();
    auto *state = get_dump_buffer_state(shared.get(), 0);
    std::array<uint8_t, 64> arena{};
    state->arena_base = reinterpret_cast<uint64_t>(arena.data());
    state->arena_size = arena.size();
    state->free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(metadata.get());
    state->free_queue.tail = 1;
    set_dump_args_enabled(true);
    set_platform_dump_base(reinterpret_cast<uint64_t>(shared.get()));
    dump_args_init(1, DumpArgsLevel::FULL);
    uint32_t source = 17;
    ArgsDumpInfo info{};
    info.task_id = TaskId::make_global(0);
    info.buffer_addr = reinterpret_cast<uint64_t>(&source);
    info.dtype = static_cast<uint8_t>(DataType::UINT32);
    info.ndims = 1;
    info.shapes[0] = 1;
    info.strides[0] = 1;
    info.kind = static_cast<uint8_t>(ArgsDumpKind::TENSOR);
    info.role = ArgsDumpRole::INPUT;
    info.stage = ArgsDumpStage::BEFORE_DISPATCH;
    DumpArgReservation owner;
    ASSERT_EQ(dump_arg_try_begin(0, info, &owner), DumpArgBeginResult::Ready);
    ASSERT_TRUE(owner.open());
    executor.args_dump_services_ = new ResidentArgsDumpService[1];
    executor.args_dump_service_count_ = 1;
    auto *retained = executor.args_dump_services_;
    ASSERT_FALSE(executor.begin_args_dump_initialization());
    executor.init_failed_.store(true);
    EXPECT_EQ(executor.finish_failed_init(&runtime), -1);
    EXPECT_EQ(executor.args_dump_services_, retained);
    EXPECT_EQ(executor.args_dump_service_count_, 1);
    EXPECT_TRUE(owner.open());
    EXPECT_TRUE(dump_arg_has_open_record(0));
    EXPECT_EQ(metadata->count, 0);
    EXPECT_EQ(state->run_dropped_record_count, 0);
    EXPECT_FALSE(executor.begin_args_dump_initialization());
    EXPECT_TRUE(dump_arg_abandon(&owner));
    delete[] retained;
    executor.args_dump_services_ = nullptr;
    executor.args_dump_service_count_ = 0;
    set_platform_dump_base(0);
    set_dump_args_enabled(false);
}

// First-wins: an error already latched keeps its own code, so a late timeout
// cannot relabel the failure the run actually had.
TEST_F(ResidentFinalizer, AnEarlierHeaderErrorKeepsPrecedence) {
    header().store(SIMPLER_ERROR_INVALID_ARGS);
    ASSERT_TRUE(exit_wait_watchdog_fires());

    const DeviceRunTerminal terminal = finalize_and_publish();

    EXPECT_EQ(header().load(), SIMPLER_ERROR_INVALID_ARGS) << "the first error owns the slot";
    EXPECT_EQ(terminal.state, DeviceRunTerminalState::Failed);
    EXPECT_EQ(terminal.code, -SIMPLER_ERROR_INVALID_ARGS);
    EXPECT_EQ(terminal.source, DeviceRunCodeSource::Header);
}

// A participant's own failure stays the run's business result; settling a clean
// control gives the header no new precedence over it.
TEST_F(ResidentFinalizer, AParticipantFailureIsNotReplacedBySettling) {
    executor.terminal_.record_participant(/*run_rc=*/-1, /*shutdown_rc=*/0);

    const DeviceRunTerminal terminal = finalize_and_publish();

    EXPECT_EQ(terminal.state, DeviceRunTerminalState::Failed);
    EXPECT_EQ(terminal.code, -1);
    EXPECT_EQ(terminal.source, DeviceRunCodeSource::ThreadRc);
}

// A participant that never claimed the audited path withholds the success, and
// settling does not supply one.
TEST_F(ResidentFinalizer, AnUnclaimedAuditedPathPublishesNothing) {
    executor.normal_path_claims_.store(0);

    executor.snapshot_run_terminal(&runtime, /*cores_retired=*/true);
    executor.publish_run_terminal();

    EXPECT_EQ(region.published, 0u) << "no record to publish";
    EXPECT_EQ(device_run_result_terminal(region, kEpoch).state, DeviceRunTerminalState::Undecided);
}

// A run whose cores did not prove they stopped did not finish the audited path,
// whatever every participant returned. The record is still published — the host
// reading no verdict at all is worse than reading a failure — but it cannot be
// the success a proved run gets.
TEST_F(ResidentFinalizer, AnUnprovedStopCannotPublishTheSuccessItsParticipantsClaimed) {
    const DeviceRunTerminal terminal = finalize_and_publish(/*cores_retired=*/false);

    EXPECT_NE(terminal.state, DeviceRunTerminalState::Succeeded)
        << "a run that could not prove its cores stopped published a success";
    EXPECT_EQ(header().load(), SIMPLER_ERROR_NONE) << "no write on a clean path";
}

}  // namespace
