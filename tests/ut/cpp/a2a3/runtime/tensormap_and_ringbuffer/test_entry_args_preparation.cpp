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

#include "aicpu_executor.cpp"  // NOLINT(bugprone-suspicious-include): exercises the production init path

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

constexpr int kOrchestrator = 3;
thread_local int execution_index = kOrchestrator;
PlatformEntryArgs launch_view{};
std::atomic<int> view_reads{0};
std::mutex view_mutex;
std::condition_variable view_changed;

void orchestration(const ChipTaskArgs &) {}

class EntryArgsPreparationTest : public ::testing::Test {
protected:
    void SetUp() override {
        runtime = std::make_unique<Runtime>();
        executor = std::make_unique<AicpuExecutor>();
        // A distinct leader and stable top-index Orch are the early route's prerequisites.
        runtime->dev.aicpu_thread_num = kOrchestrator + 1;
        runtime->set_active_callable_id(0);
        runtime->dev.entry_args_source_ = static_cast<uint32_t>(EntryArgsSource::LaunchEnvelope);
        runtime->dev.entry_tensor_count_ = 1;
        runtime->dev.entry_scalar_count_ = 1;
        auto &entry = executor->orch_so_table_[0];
        entry.handle = reinterpret_cast<void *>(uintptr_t{1});
        entry.func = orchestration;

        simpler::tmr::Tensor tensor{};
        tensor.buffer.addr = 0x12340000;
        tensor.ndims = 1;
        tensor.shapes[0] = 17;
        simpler::tmr::EntryArgsStorage args{};
        args.add_tensor(tensor);
        args.add_scalar(0x5678);
        payload.resize(LAUNCH_ENVELOPE_HEADER_BYTES + args.wire_bytes(1, 1));
        std::memcpy(payload.data() + LAUNCH_ENVELOPE_HEADER_BYTES, args.tensors_, sizeof(tensor));
        std::memcpy(payload.data() + LAUNCH_ENVELOPE_HEADER_BYTES + sizeof(tensor), args.scalars_, sizeof(uint64_t));
        launch_view = {
            payload.data(), LAUNCH_ENVELOPE_HEADER_BYTES, 1, 1, static_cast<uint32_t>(EntryArgsSource::LaunchEnvelope)
        };
        view_reads.store(0);
        execution_index = kOrchestrator;
    }

    void TearDown() override {
        set_scheduler_timeout_ms(0);
        // The fixture's registration has no loader-owned handle to close.
        executor->orch_so_table_[0].handle = nullptr;
        executor->orch_so_table_[0].func = nullptr;
        launch_view = {};
    }

    bool prepare_while_leader_is_pending(bool fail_leader, int expected_reads = 2, int index = kOrchestrator) {
        auto pending = std::async(std::launch::async, [this, index] {
            execution_index = index;
            return executor->init(runtime.get(), preparation);
        });
        bool observed = false;
        {
            std::unique_lock<std::mutex> lock(view_mutex);
            // A timeout only bounds a broken protocol; this test has no latency assertion.
            observed = view_changed.wait_for(lock, std::chrono::seconds(2), [expected_reads] {
                return view_reads.load(std::memory_order_acquire) >= expected_reads;
            });
        }
        EXPECT_FALSE(executor->hs_setup_done_.load());
        EXPECT_FALSE(executor->runtime_init_ready_.load());
        EXPECT_EQ(pending.wait_for(std::chrono::milliseconds(10)), std::future_status::timeout);
        // Release even after a failed observation so a regression cannot strand the worker.
        executor->init_failed_.store(fail_leader, std::memory_order_release);
        executor->hs_setup_done_.store(true, std::memory_order_release);
        EXPECT_EQ(pending.get(), fail_leader ? -1 : 0);
        return observed;
    }

    std::unique_ptr<Runtime> runtime;
    std::unique_ptr<AicpuExecutor> executor;
    std::vector<unsigned char> payload;
    EntryArgsPreparation preparation{EntryArgsPreparation::NotPrepared};
};

}  // namespace

int32_t platform_aicpu_affinity_thread_idx() { return execution_index; }
PlatformEntryArgs get_platform_entry_args(int32_t) {
    {
        std::lock_guard<std::mutex> lock(view_mutex);
        view_reads.fetch_add(1, std::memory_order_release);
    }
    view_changed.notify_all();
    return launch_view;
}
extern "C" uint64_t get_platform_regs() { return 0; }
extern "C" uint64_t get_platform_run_result_epoch() { return 0; }

// Orch init consumes no AICore registers. These boundaries fail if it does.
uint32_t platform_get_physical_cores_count() {
    ADD_FAILURE() << "Orch init queried physical cores";
    return 0;
}
uint64_t platform_aicore_exit_deadline() { return 0; }
uint64_t read_reg(uint64_t, RegId) {
    ADD_FAILURE() << "Orch init read an AICore register";
    return 0;
}
void platform_init_aicore_regs(uint64_t) { ADD_FAILURE() << "Orch init handshook an AICore"; }
void platform_signal_aicore_exit(uint64_t) { ADD_FAILURE() << "Orch init signaled an AICore"; }
int32_t platform_retire_aicore_group(const AicoreExitTarget *, size_t, uint64_t, bool *) {
    ADD_FAILURE() << "Orch init retired an AICore";
    return -1;
}

TEST_F(EntryArgsPreparationTest, LaunchEnvelopePreparationStartsBeforeLeaderReadiness) {
    ASSERT_TRUE(prepare_while_leader_is_pending(false));
    EXPECT_EQ(preparation, EntryArgsPreparation::Ready);
    EXPECT_FALSE(executor->runtime_init_ready_.load());
    const auto &args = runtime->dev.orch_args_storage_;
    ASSERT_EQ(args.tensor_count(), 1);
    ASSERT_EQ(args.scalar_count(), 1);
    EXPECT_EQ(args.tensor(0).buffer.addr, 0x12340000U);
    EXPECT_EQ(args.tensor(0).shapes[0], 17U);
    EXPECT_EQ(args.scalar(0), 0x5678U);
}

TEST_F(EntryArgsPreparationTest, RejectedEnvelopeStillWaitsForLeaderAndDoesNotOverwriteStorage) {
    runtime->dev.orch_args_storage_.add_scalar(0x9999);
    launch_view.tensor_count = CHIP_MAX_TENSOR_ARGS + 1;
    ASSERT_TRUE(prepare_while_leader_is_pending(false));
    EXPECT_EQ(preparation, EntryArgsPreparation::Rejected);
    EXPECT_FALSE(executor->runtime_init_ready_.load());
    ASSERT_EQ(runtime->dev.orch_args_storage_.scalar_count(), 1);
    EXPECT_EQ(runtime->dev.orch_args_storage_.scalar(0), 0x9999U);
}

TEST_F(EntryArgsPreparationTest, LeaderFailureRemainsAuthoritativeAfterEarlyRejection) {
    launch_view.tensor_count = CHIP_MAX_TENSOR_ARGS + 1;
    ASSERT_TRUE(prepare_while_leader_is_pending(true));
    EXPECT_EQ(preparation, EntryArgsPreparation::Rejected);
    EXPECT_FALSE(executor->runtime_init_ready_.load());
}

TEST_F(EntryArgsPreparationTest, LeaderFailureRemainsAuthoritativeAfterSuccessfulAdoption) {
    ASSERT_TRUE(prepare_while_leader_is_pending(true));
    EXPECT_EQ(preparation, EntryArgsPreparation::Ready);
    EXPECT_FALSE(executor->runtime_init_ready_.load());
}

TEST_F(EntryArgsPreparationTest, DescriptorAndKernelRoutesRemainUnprepared) {
    for (auto source : {EntryArgsSource::Descriptor, EntryArgsSource::KernelInvocation}) {
        launch_view.source = static_cast<uint32_t>(source);
        view_reads.store(0);
        executor->hs_setup_done_.store(true);
        executor->init_failed_.store(false);
        preparation = EntryArgsPreparation::NotPrepared;
        ASSERT_EQ(executor->init(runtime.get(), preparation), 0);
        EXPECT_EQ(preparation, EntryArgsPreparation::NotPrepared);
        EXPECT_EQ(view_reads.load(), 1);
        EXPECT_EQ(runtime->dev.orch_args_storage_.tensor_count(), 0);
    }
}

TEST_F(EntryArgsPreparationTest, UnregisteredCallableIsLeftForTheOriginalLateRejection) {
    executor->orch_so_table_[0].handle = nullptr;
    ASSERT_TRUE(prepare_while_leader_is_pending(true, 1));
    EXPECT_EQ(preparation, EntryArgsPreparation::NotPrepared);
    EXPECT_EQ(runtime->dev.orch_args_storage_.tensor_count(), 0);
}

TEST_F(EntryArgsPreparationTest, SerialAndUnstableRolesDoNotPrepareEarly) {
    runtime->dev.serial_orch_sched = true;
    executor->hs_setup_done_.store(true);
    executor->init_failed_.store(true);
    EXPECT_EQ(executor->init(runtime.get(), preparation), -1);
    EXPECT_EQ(preparation, EntryArgsPreparation::NotPrepared);
    EXPECT_EQ(view_reads.load(), 0);

    runtime->dev.serial_orch_sched = false;
    execution_index = -1;
    executor->hs_thread_seq_.store(kOrchestrator);
    EXPECT_EQ(executor->init(runtime.get(), preparation), -1);
    EXPECT_EQ(preparation, EntryArgsPreparation::NotPrepared);
    EXPECT_EQ(view_reads.load(), 0);
}

TEST_F(EntryArgsPreparationTest, GateOwnerClearsBeforeSharedReleaseAndOnlyOnce) {
    runtime->dev.worker_count = 3;
    auto *gates = runtime->get_teardown_gates();
    gates[0].post_close_release = 1;
    gates[0].exit_request = 23;
    gates[3].post_close_release = 1;

    auto worker = std::async(std::launch::async, [this] {
        execution_index = 1;
        auto args = EntryArgsPreparation::NotPrepared;
        return executor->init(runtime.get(), args);
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (executor->gate_reset_.load(std::memory_order_acquire) != GateReset::Complete &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    const bool ready = executor->gate_reset_.load(std::memory_order_acquire) == GateReset::Complete;
    EXPECT_TRUE(ready);
    EXPECT_FALSE(executor->hs_setup_done_.load(std::memory_order_acquire));
    if (ready) {
        EXPECT_EQ(gates[0].post_close_release, 0u);
        EXPECT_EQ(gates[0].exit_request, 0u);
        EXPECT_EQ(gates[3].post_close_release, 1u);
        gates[0].post_close_release = 1;
        executor->finish_teardown_gate_reset(runtime.get(), 3);
        EXPECT_EQ(gates[0].post_close_release, 1u);
    }
    executor->init_failed_.store(true, std::memory_order_release);
    executor->hs_setup_done_.store(true, std::memory_order_release);
    EXPECT_EQ(worker.get(), -1);
}

TEST_F(EntryArgsPreparationTest, LeaderClaimsUnstartedGateResetAndRearmsForNextRun) {
    auto *gates = runtime->get_teardown_gates();
    for (int32_t workers : {3, 6, 3}) {
        SCOPED_TRACE(workers);
        gates[workers - 1].post_close_release = 1;
        gates[workers - 1].exit_request = 23;
        executor->finish_teardown_gate_reset(runtime.get(), workers);
        EXPECT_EQ(executor->gate_reset_.load(), GateReset::Complete);
        EXPECT_EQ(gates[workers - 1].post_close_release, 0u);
        EXPECT_EQ(gates[workers - 1].exit_request, 0u);
        EXPECT_FALSE(executor->try_reset_teardown_gates(runtime.get(), workers));
        executor->reset_entry_state();
        EXPECT_EQ(executor->gate_reset_.load(), GateReset::Unclaimed);
    }
}

TEST_F(EntryArgsPreparationTest, StalledGateOwnerCannotReleaseTheFailedInvocation) {
    runtime->dev.worker_count = 3;
    auto *gates = runtime->get_teardown_gates();
    gates[0].post_close_release = 1;
    gates[0].exit_request = 23;
    executor->gate_reset_.store(GateReset::Claimed, std::memory_order_relaxed);
    set_scheduler_timeout_ms(1);

    execution_index = 0;
    ASSERT_EQ(executor->init(runtime.get(), preparation), -1);
    EXPECT_TRUE(executor->init_failed_.load(std::memory_order_acquire));
    EXPECT_FALSE(executor->runtime_init_ready_.load(std::memory_order_acquire));
    EXPECT_EQ(executor->gate_reset_.load(), GateReset::Claimed);
    EXPECT_EQ(gates[0].post_close_release, 1u);
    EXPECT_EQ(gates[0].exit_request, 23u);

    SchedulerContext::reset_teardown_gates(runtime.get(), runtime->dev.worker_count);
    executor->gate_reset_.store(GateReset::Complete, std::memory_order_release);
    for (int idx : {1, 2, kOrchestrator}) {
        execution_index = idx;
        EXPECT_EQ(executor->init(runtime.get(), preparation), -1);
    }
    EXPECT_TRUE(executor->init_failed_.load(std::memory_order_acquire));
    EXPECT_FALSE(executor->runtime_init_ready_.load(std::memory_order_acquire));
    EXPECT_EQ(gates[0].post_close_release, 0u);
    EXPECT_EQ(gates[0].exit_request, 0u);
}
