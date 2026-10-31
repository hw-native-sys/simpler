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
#include <cinttypes>
#include <cstdio>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "callable.h"
#include "orchestrator.h"
#include "scheduler/scheduler_context.h"

class MixAdmissionTestPeer {
public:
    static void bind(
        SchedulerContext &context, SchedulerState &scheduler, uint64_t *callables, uint32_t *conditions,
        uint64_t *registers
    ) {
        context.sched_ = &scheduler;
        context.active_sched_threads_ = 1;
        context.aicpu_thread_num_ = 1;
        context.cores_total_num_ = 3;
        context.func_id_to_addr_ = callables;
        context.func_id_to_addr_count_ = 1;
        for (int owner = 0; owner < 2; ++owner) {
            auto &tracker = context.core_trackers_[owner];
            tracker.init(1);
            tracker.set_cluster(0, owner * 3, owner * 3 + 1, owner * 3 + 2);
        }
        for (int core = 0; core < 6; ++core) {
            auto &state = context.core_exec_states_[core];
            state = {};
            state.reg_addr = reinterpret_cast<uint64_t>(registers + core * (0x6000 / sizeof(uint64_t)));
            state.cond_ptr = conditions + core;
            state.running_reg_task_id = AICPU_TASK_INVALID;
            state.pending_reg_task_id = AICPU_TASK_INVALID;
        }
    }

    static void dispatch(SchedulerContext &context) {
        bool progress = false, pushed = false;
        context.dispatch_ready_tasks(0, context.core_trackers_[0], false, progress, pushed);
    }

    static int poll(SchedulerContext &context) {
        int completed = 0, thread_completed = 0, release_count = 0;
        bool progress = false;
        ChipTaskSlotState *releases[256]{};
        context.check_running_cores_for_completion(
            0, nullptr, completed, thread_completed, progress, releases, release_count
        );
        for (int i = 0; i < release_count; ++i)
            context.sched_->on_task_release(*releases[i]);
        return completed;
    }

    static CoreExecState &core(SchedulerContext &context) { return context.core_exec_states_[0]; }

    static std::string dump(SchedulerContext &context, StallDumpReport report) {
        testing::internal::CaptureStderr();
        context.log_stall_diagnostics(0, 42, report);
        return testing::internal::GetCapturedStderr();
    }

    static std::string owned_dump(SchedulerContext &context, int owner, StallDumpReport report) {
        testing::internal::CaptureStderr();
        context.log_owned_dispatch_records(owner, report);
        return testing::internal::GetCapturedStderr();
    }

    static std::string shutdown_dump(SchedulerContext &context) {
        testing::internal::CaptureStderr();
        context.log_shutdown_stall_snapshot(0, 42, 0, "scheduler_timeout");
        return testing::internal::GetCapturedStderr();
    }

    static std::string timeout_dump(SchedulerContext &context, SharedMemoryHeader *header) {
        // This CPU fixture has no published retirement targets. The real shutdown
        // path records REQUESTED without attempting hardware retirement or a join.
        testing::internal::CaptureStderr();
        EXPECT_EQ(context.handle_timeout_exit(0, header, nullptr, 42, 0), -SIMPLER_ERROR_SCHEDULER_TIMEOUT);
        EXPECT_TRUE(context.fatal_shutdown_started_.load());
        EXPECT_TRUE(context.is_completed());
        for (int core = 0; core < context.cores_total_num_; ++core)
            EXPECT_EQ(context.retirement_state_[core], SchedulerContext::RETIREMENT_REQUESTED);
        return testing::internal::GetCapturedStderr();
    }

    template <typename Context>
    static LoopAction idle_fatal(Context &context, SharedMemoryHeader *header) {
        if constexpr (std::is_invocable_v<
                          decltype(&Context::check_idle_fatal_error), Context &, int32_t, SharedMemoryHeader *,
                          Runtime *>) {
            return context.check_idle_fatal_error(0, header, nullptr);
        } else {
            return context.check_idle_fatal_error(0, header, nullptr, 42, 0);
        }
    }

    static std::string fatal_dump(SchedulerContext &context, SharedMemoryHeader *header, bool idle) {
        testing::internal::CaptureStderr();
        int32_t task_count = 0;
        const auto action =
            idle ? idle_fatal(context, header) : context.handle_orchestrator_exit(0, header, nullptr, task_count);
        EXPECT_EQ(action, LoopAction::BREAK_LOOP);
        return testing::internal::GetCapturedStderr();
    }

    static void set_completed(SchedulerContext &context) { context.completed_.store(true); }

    static void clear_owner(SchedulerContext &context) { context.core_trackers_[0].init(0); }

    static void expect_fatal_retained(SchedulerContext &context) {
        EXPECT_TRUE(context.fatal_shutdown_started_.load());
        EXPECT_TRUE(context.is_completed());
        for (int core = 0; core < context.cores_total_num_; ++core)
            EXPECT_EQ(context.retirement_state_[core], SchedulerContext::RETIREMENT_REQUESTED);
    }

    static void seed_other_owner(SchedulerContext &context) {
        context.active_sched_threads_ = 2;
        context.aicpu_thread_num_ = 2;
        context.cores_total_num_ = 6;
        auto &state = context.core_exec_states_[3];
        state.pending_reg_task_id = 42;
        state.pending_slot_state = reinterpret_cast<ChipTaskSlotState *>(uintptr_t{1});
        auto &payload = context.payload_per_core_[3][0];
        payload.local_context.async_ctx.task_token = TaskId::make(2, 99);
        payload.local_context.block_idx = 5;
        payload.local_context.block_num = 7;
        payload.src_payload = 1;
    }
};

namespace {

std::string hex_id(TaskId id) {
    char text[32];
    snprintf(text, sizeof(text), "0x%" PRIx64, TaskId::to_uint64(id));
    return text;
}

void expect_slot_record(
    const std::string &records, const char *slot, int32_t token, TaskId task, int32_t block, int32_t blocks
) {
    const std::string expected = "SLOT core=0 slot=" + std::string(slot) + " token=" + std::to_string(token) +
                                 " bank=" + std::to_string(token & 1) + " task_id=" + hex_id(task) +
                                 " block=" + std::to_string(block) + "/" + std::to_string(blocks);
    EXPECT_NE(records.find(expected), std::string::npos) << expected << "\n" << records;
}

// Register ACK/FIN is controlled by the test; no AICore kernel or TWAIT runs here.
class BlockingWaitContract : public testing::Test {
protected:
    DeviceArena shared_arena;
    DeviceArena runtime_arena;
    SharedMemoryHandle *shared = nullptr;
    std::unique_ptr<OrchestratorState> orch = std::make_unique<OrchestratorState>();
    std::unique_ptr<SchedulerState> scheduler = std::make_unique<SchedulerState>();
    std::unique_ptr<SchedulerContext> context = std::make_unique<SchedulerContext>();
    std::vector<char> heap;
    std::vector<uint8_t> callable = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, nullptr, 0);
    uint64_t callables[1]{};
    std::array<uint32_t, 6> conditions{};
    std::vector<uint64_t> registers = std::vector<uint64_t>(6 * (0x6000 / sizeof(uint64_t)));

    void SetUp() override {
        shared = SharedMemoryHandle::create_and_init_default(shared_arena);
        ASSERT_NE(shared, nullptr);
        heap.resize(4096 * CHIP_MAX_RING_DEPTH);
        int32_t windows[CHIP_MAX_RING_DEPTH];
        for (auto &window : windows)
            window = CHIP_TASK_WINDOW_SIZE;
        const auto orch_layout = OrchestratorState::reserve_layout(runtime_arena, windows);
        const auto sched_layout = SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);
        ASSERT_TRUE(orch->init_data_from_layout(
            orch_layout, runtime_arena, shared->sm_base, heap.data(), 4096, CHIP_TASK_WINDOW_SIZE
        ));
        ASSERT_TRUE(scheduler->init_data_from_layout(sched_layout, runtime_arena, shared->sm_base));
        scheduler->wire_arena_pointers(sched_layout, runtime_arena);
        orch->wire_arena_pointers(orch_layout, runtime_arena, scheduler.get());
        reinterpret_cast<CoreCallable *>(callable.data())->set_resolved_addr(1);
        callables[0] = reinterpret_cast<uint64_t>(callable.data());
        MixAdmissionTestPeer::bind(*context, *scheduler, callables, conditions.data(), registers.data());
    }

    void TearDown() override {
        orch->destroy();
        scheduler->destroy();
    }

    TaskId submit(const TaskId *dependency = nullptr, int16_t blocks = 1) {
        CoreTaskArgs args;
        args.launch_spec.set_block_num(blocks);
        const DepFlags kind = DEP_WAIT;
        if (dependency != nullptr) args.set_dependencies_with_kinds(dependency, &kind, 1);
        MixedKernels kernels;
        kernels.aic_kernel_id = 0;
        const TaskId id = orch->submit_task(kernels, args).task_id();
        EXPECT_TRUE(id.is_valid());
        EXPECT_FALSE(orch->fatal);
        return id;
    }

    ChipTaskSlotState *slot(TaskId id) {
        return &shared->header->rings[id.ring()].get_slot_state_by_task_id(id.local_id());
    }

    CoreExecState &core() { return MixAdmissionTestPeer::core(*context); }

    void ack() {
        conditions[0] = MAKE_ACK_VALUE(core().running_reg_task_id);
        EXPECT_EQ(MixAdmissionTestPeer::poll(*context), 0);
    }

    void finish() {
        if (core().pending_reg_task_id != AICPU_TASK_INVALID) {
            auto *pending = core().pending_slot_state;
            const bool same_task = core().running_slot_state == pending;
            conditions[0] = MAKE_FIN_VALUE(core().running_reg_task_id);
            EXPECT_EQ(MixAdmissionTestPeer::poll(*context), 0);
            conditions[0] = MAKE_ACK_VALUE(core().pending_reg_task_id);
            EXPECT_EQ(MixAdmissionTestPeer::poll(*context), same_task ? 0 : 1);
            EXPECT_EQ(core().running_slot_state, pending);
            EXPECT_EQ(core().pending_slot_state, nullptr);
        }
        conditions[0] = MAKE_FIN_VALUE(core().running_reg_task_id);
        EXPECT_GT(MixAdmissionTestPeer::poll(*context), 0);
    }

    void make_counterexample(TaskId &waiter, TaskId &producer) {
        waiter = submit();
        MixAdmissionTestPeer::dispatch(*context);
        ASSERT_EQ(core().running_slot_state, slot(waiter));
        ack();
        producer = submit();
        MixAdmissionTestPeer::dispatch(*context);
        ASSERT_EQ(core().running_slot_state, slot(waiter));
        ASSERT_EQ(core().pending_slot_state, slot(producer));
    }
};

TEST_F(BlockingWaitContract, IndependentProducerRemainsPendingWhileWaiterHasNoCompletion) {
    for (int epoch = 0; epoch < 3; ++epoch) {
        SCOPED_TRACE(epoch);
        orch->begin_scope();
        TaskId waiter, producer;
        make_counterexample(waiter, producer);
        const int waiter_token = core().running_reg_task_id;
        const int producer_token = core().pending_reg_task_id;
        for (int poll = 0; poll < 16; ++poll) {
            EXPECT_EQ(MixAdmissionTestPeer::poll(*context), 0);
            EXPECT_EQ(core().running_slot_state, slot(waiter));
            EXPECT_EQ(core().pending_slot_state, slot(producer));
            EXPECT_EQ(core().running_reg_task_id, waiter_token);
            EXPECT_EQ(core().pending_reg_task_id, producer_token);
        }
        finish();
        EXPECT_EQ(core().running_slot_state, nullptr);
        EXPECT_EQ(core().pending_slot_state, nullptr);
        orch->end_scope();
    }
}

TEST_F(BlockingWaitContract, ExplicitWaitDependencyDelaysWaiterUntilProducerCompletion) {
    for (int epoch = 0; epoch < 3; ++epoch) {
        SCOPED_TRACE(epoch);
        orch->begin_scope();
        const TaskId producer = submit();
        const TaskId waiter = submit(&producer);
        MixAdmissionTestPeer::dispatch(*context);
        ASSERT_EQ(core().running_slot_state, slot(producer));
        ack();
        MixAdmissionTestPeer::dispatch(*context);
        EXPECT_EQ(core().pending_slot_state, nullptr);
        EXPECT_LT(slot(waiter)->fanin_refcount.load(), slot(waiter)->fanin_count);
        finish();
        EXPECT_EQ(slot(waiter)->fanin_refcount.load(), slot(waiter)->fanin_count);
        MixAdmissionTestPeer::dispatch(*context);
        ASSERT_EQ(core().running_slot_state, slot(waiter));
        finish();
        EXPECT_EQ(core().running_slot_state, nullptr);
        orch->end_scope();
    }
}

TEST_F(BlockingWaitContract, OwnerPublicationRecordsIdentifyBothTaskGenerations) {
    std::string previous_producer;
    int previous_token = -1;
    for (int epoch = 0; epoch < 3; ++epoch) {
        orch->begin_scope();
        TaskId waiter, producer;
        make_counterexample(waiter, producer);
        const std::string records = MixAdmissionTestPeer::dump(*context, StallDumpReport::Periodic);
        EXPECT_NE(
            records.find("publication_records=owner_local implicit_wait_metadata=unavailable"), std::string::npos
        );
        expect_slot_record(records, "running", core().running_reg_task_id, waiter, 0, 1);
        expect_slot_record(records, "pending", core().pending_reg_task_id, producer, 0, 1);
        if (epoch != 0) {
            EXPECT_EQ(records.find(previous_producer), std::string::npos);
            EXPECT_GT(core().pending_reg_task_id, previous_token);
        }
        previous_producer = "task_id=" + hex_id(producer) + " block=";
        previous_token = core().pending_reg_task_id;
        finish();
        const std::string idle = MixAdmissionTestPeer::dump(*context, StallDumpReport::Periodic);
        EXPECT_EQ(idle.find("SLOT core="), std::string::npos);
        orch->end_scope();
    }
}

TEST_F(BlockingWaitContract, SpmdRecordsKeepSeparateLogicalBlocksInBothBanks) {
    orch->begin_scope();
    const TaskId task = submit(nullptr, 2);
    MixAdmissionTestPeer::dispatch(*context);
    ack();
    MixAdmissionTestPeer::dispatch(*context);
    ASSERT_EQ(core().running_slot_state, slot(task));
    ASSERT_EQ(core().pending_slot_state, slot(task));
    const std::string records = MixAdmissionTestPeer::dump(*context, StallDumpReport::Periodic);
    expect_slot_record(records, "running", core().running_reg_task_id, task, 0, 2);
    expect_slot_record(records, "pending", core().pending_reg_task_id, task, 1, 2);
    finish();
    orch->end_scope();
}

TEST_F(BlockingWaitContract, OwnerRecordsDoNotReadOtherOwnersOrTaskPointers) {
    MixAdmissionTestPeer::seed_other_owner(*context);
    const std::string own = MixAdmissionTestPeer::owned_dump(*context, 0, StallDumpReport::Shutdown);
    EXPECT_EQ(own.find("SLOT core="), std::string::npos);
    EXPECT_EQ(own.find("task_id=0x200000063"), std::string::npos);
    const std::string other = MixAdmissionTestPeer::owned_dump(*context, 1, StallDumpReport::Shutdown);
    EXPECT_NE(other.find("SLOT core=3 slot=pending token=42 bank=0 task_id=0x200000063 block=5/7"), std::string::npos);
    EXPECT_NE(other.find("src_payload=0x1"), std::string::npos);
    EXPECT_EQ(MixAdmissionTestPeer::owned_dump(*context, -1, StallDumpReport::Shutdown), "");
    EXPECT_EQ(MixAdmissionTestPeer::owned_dump(*context, 2, StallDumpReport::Shutdown), "");
}

TEST_F(BlockingWaitContract, CrossOwnerShutdownScanDoesNotEmitOwnerLocalRecords) {
    MixAdmissionTestPeer::seed_other_owner(*context);
    const std::string records = MixAdmissionTestPeer::shutdown_dump(*context);
    EXPECT_NE(records.find("thread=1"), std::string::npos);
    EXPECT_EQ(records.find("publication_records=owner_local"), std::string::npos);
    EXPECT_EQ(records.find("SLOT core="), std::string::npos);
}

TEST_F(BlockingWaitContract, TimeoutWinnerPublishesOwnerRecordsAndLatchesErrorOnce) {
    orch->begin_scope();
    TaskId waiter, producer;
    make_counterexample(waiter, producer);
    const std::string records = MixAdmissionTestPeer::timeout_dump(*context, shared->header);
    expect_slot_record(records, "running", core().running_reg_task_id, waiter, 0, 1);
    expect_slot_record(records, "pending", core().pending_reg_task_id, producer, 0, 1);
    EXPECT_NE(records.find("publication_records=owner_local implicit_wait_metadata=unavailable"), std::string::npos);
    EXPECT_EQ(shared->header->sched_error_code.load(), SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    EXPECT_EQ(shared->header->sched_error_thread.load(), 0);
    EXPECT_EQ(shared->header->sched_error_bitmap.load(), 1U);
    const std::string repeated = MixAdmissionTestPeer::timeout_dump(*context, shared->header);
    EXPECT_EQ(repeated.find("publication_records=owner_local"), std::string::npos);
    EXPECT_EQ(repeated.find("SLOT core="), std::string::npos);
    EXPECT_EQ(core().running_slot_state, slot(waiter));
    EXPECT_EQ(core().pending_slot_state, slot(producer));
    orch->end_scope();
}

class TensorWaitFatalContract : public BlockingWaitContract,
                                public testing::WithParamInterface<bool> {};

TEST_P(TensorWaitFatalContract, ReachedOwnerReportsSlotsBeforeRetirement) {
    orch->begin_scope();
    TaskId waiter, producer;
    make_counterexample(waiter, producer);
    MixAdmissionTestPeer::seed_other_owner(*context);
    shared->header->orch_error_code.store(SIMPLER_ERROR_TENSOR_WAIT_TIMEOUT);
    const std::string records = MixAdmissionTestPeer::fatal_dump(*context, shared->header, GetParam());
    expect_slot_record(records, "running", core().running_reg_task_id, waiter, 0, 1);
    expect_slot_record(records, "pending", core().pending_reg_task_id, producer, 0, 1);
    EXPECT_NE(records.find("publication_records=owner_local implicit_wait_metadata=unavailable"), std::string::npos);
    EXPECT_EQ(records.find("SLOT core=3"), std::string::npos);
    const auto retirement = records.find("Emergency shutdown:");
    ASSERT_NE(retirement, std::string::npos) << records;
    EXPECT_LT(records.find("SLOT core=0 slot=running"), retirement);
    EXPECT_LT(records.find("SLOT core=0 slot=pending"), retirement);
    EXPECT_EQ(shared->header->orch_error_code.load(), SIMPLER_ERROR_TENSOR_WAIT_TIMEOUT);
    EXPECT_EQ(shared->header->sched_error_code.load(), SIMPLER_ERROR_NONE);
    EXPECT_EQ(shared->header->sched_error_bitmap.load(), 0U);
    EXPECT_EQ(core().running_slot_state, slot(waiter));
    EXPECT_EQ(core().pending_slot_state, slot(producer));
    MixAdmissionTestPeer::expect_fatal_retained(*context);
    const std::string repeated = MixAdmissionTestPeer::fatal_dump(*context, shared->header, GetParam());
    EXPECT_EQ(repeated.find("SLOT core="), std::string::npos);
    orch->end_scope();
}

TEST_P(TensorWaitFatalContract, UnrelatedFatalKeepsExistingDiagnosticScope) {
    shared->header->orch_error_code.store(SIMPLER_ERROR_INVALID_ARGS);
    const std::string records = MixAdmissionTestPeer::fatal_dump(*context, shared->header, GetParam());
    EXPECT_EQ(records.find("publication_records=owner_local"), std::string::npos);
    EXPECT_EQ(records.find("SLOT core="), std::string::npos);
    EXPECT_EQ(shared->header->orch_error_code.load(), SIMPLER_ERROR_INVALID_ARGS);
    EXPECT_EQ(shared->header->sched_error_code.load(), SIMPLER_ERROR_NONE);
    MixAdmissionTestPeer::expect_fatal_retained(*context);
}

TEST_P(TensorWaitFatalContract, CompletedOwnerDoesNotReachFatalDiagnostic) {
    shared->header->orch_error_code.store(SIMPLER_ERROR_TENSOR_WAIT_TIMEOUT);
    MixAdmissionTestPeer::set_completed(*context);
    EXPECT_EQ(MixAdmissionTestPeer::fatal_dump(*context, shared->header, GetParam()), "");
}

TEST_P(TensorWaitFatalContract, EmptyTrackerDoesNotInventSlotRecords) {
    MixAdmissionTestPeer::clear_owner(*context);
    shared->header->orch_error_code.store(SIMPLER_ERROR_TENSOR_WAIT_TIMEOUT);
    const std::string records = MixAdmissionTestPeer::fatal_dump(*context, shared->header, GetParam());
    EXPECT_EQ(records.find("SLOT core="), std::string::npos);
    EXPECT_EQ(shared->header->orch_error_code.load(), SIMPLER_ERROR_TENSOR_WAIT_TIMEOUT);
    MixAdmissionTestPeer::expect_fatal_retained(*context);
}

INSTANTIATE_TEST_SUITE_P(FatalEntries, TensorWaitFatalContract, testing::Values(false, true));

}  // namespace
