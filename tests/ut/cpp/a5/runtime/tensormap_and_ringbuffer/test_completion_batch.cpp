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
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/device_phase_aicpu.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "runtime.h"
#include "runtime_core.h"
#include "scheduler/scheduler_context.h"

namespace {
// Only hardware/profiling boundaries are replaced. State classification, slab
// handling, mailbox publication, counting and fanout callbacks are production.
struct Observers {
    std::function<void(const volatile uint32_t *)> read;
    std::function<void(const void *)> invalidate;
    std::function<void()> doorbell;
    std::vector<uint64_t> finishes;
    uint64_t clock{100};
};
thread_local Observers observers;

struct Task {
    TaskDescriptor descriptor{};
    TaskPayload payload{};
    ChipTaskSlotState slot{};

    explicit Task(int total, int id = 0, uint8_t ring = 0) {
        descriptor.task_id = TaskId::make(ring, id);
        slot.task = &descriptor;
        slot.payload = &payload;
        slot.ring_id = ring;
        slot.total_required_subtasks = static_cast<int16_t>(total);
        slot.task_state.store(CHIP_TASK_PENDING);
    }
};

struct PollResult {
    int32_t tasks{0};
    int32_t physical{0};
    bool progress{false};
    std::array<ChipTaskSlotState *, 256> released{};
    int32_t release_count{0};
};
}  // namespace

uint32_t reg_load_acquire(const volatile uint32_t *ptr) {
    if (observers.read) observers.read(ptr);
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

volatile uint32_t *get_reg_ptr(uint64_t addr, RegId reg) {
    EXPECT_EQ(reg, RegId::DATA_MAIN_BASE);
    if (observers.doorbell) observers.doorbell();
    return reinterpret_cast<volatile uint32_t *>(addr);
}

uint64_t get_sys_cnt_aicpu() { return ++observers.clock; }

namespace aicpu_cache_maintenance {
void invalidate_range_impl(const void *addr, size_t) {
    if (observers.invalidate) observers.invalidate(addr);
}
void flush_range_impl(const void *, size_t) {}
}  // namespace aicpu_cache_maintenance

#if SIMPLER_DFX
void chip_swimlane_aicpu_on_aicore_ack(int, int, uint32_t) {}
int chip_swimlane_aicpu_complete_task(int, int, uint32_t, uint64_t, uint64_t finish) {
    observers.finishes.push_back(finish);
    return 0;
}
void chip_swimlane_aicpu_record_sched_phase(
    int, SchedPhaseKind, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, const int16_t *, const int16_t *
) {}
extern "C" bool is_pmu_enabled() { return false; }
void pmu_aicpu_complete_record(int, int, uint32_t, TaskId, uint32_t, CoreType) { ADD_FAILURE() << "PMU is disabled"; }
#endif

class CompletionBatchTestPeer {
public:
    static constexpr int CORES_PER_OWNER = 9;

    static void bind(SchedulerContext &context, RuntimeContext &runtime, SharedMemoryHeader &header) {
        context.sched_ = &runtime.scheduler;
        context.rt_ = &runtime;
        runtime.scheduler.sm_header = &header;
        context.active_sched_threads_ = 2;
        ASSERT_TRUE(context.prepare_completion_scratch(runtime.prebuilt_layout.sizing.task_window_sizes));
        for (int owner = 0; owner < 2; ++owner) {
            auto &tracker = context.core_trackers_[owner];
            tracker.init(CORES_PER_OWNER / 3);
            for (int cluster = 0; cluster < CORES_PER_OWNER / 3; ++cluster) {
                int core = owner * CORES_PER_OWNER + cluster * 3;
                tracker.set_cluster(cluster, core, core + 1, core + 2);
            }
#if SIMPLER_DFX
            context.sched_chip_swimlane_[owner].chip_swimlane_enabled = true;
#endif
        }
#if SIMPLER_DFX
        context.chip_swimlane_level_ = ChipSwimlaneLevel::SCHEDULE_TIMING;
#endif
    }

    static CoreExecState &core(SchedulerContext &context, int id) { return context.core_exec_states_[id]; }
    static DeferredCompletionSlab &slab(SchedulerContext &context, int core, int token) {
        return context.deferred_slab_per_core_[core][token & 1];
    }
    static void running(SchedulerContext &context, int id, bool groupable) {
        auto &tracker = context.core_trackers_[id / CORES_PER_OWNER];
        tracker.change_core_state(id % CORES_PER_OWNER);
        tracker.set_running_groupable(id % CORES_PER_OWNER, groupable);
        tracker.set_pending_occupied(id % CORES_PER_OWNER);
    }
    static void pending(SchedulerContext &context, int id, bool groupable) {
        auto &tracker = context.core_trackers_[id / CORES_PER_OWNER];
        tracker.set_pending_occupied(id % CORES_PER_OWNER);
        tracker.set_pending_groupable(id % CORES_PER_OWNER, groupable);
    }
    static bool running_groupable(SchedulerContext &context, int id) {
        return context.core_trackers_[id / CORES_PER_OWNER].is_running_groupable(id % CORES_PER_OWNER);
    }
    static bool pending_groupable(SchedulerContext &context, int id) {
        return context.core_trackers_[id / CORES_PER_OWNER].is_pending_groupable(id % CORES_PER_OWNER);
    }
    static bool free_slot(SchedulerContext &context, int id) {
        auto bits = context.core_trackers_[id / CORES_PER_OWNER].get_free_slot_states();
        while (bits.has_value()) {
            if (bits.pop_first() == id % CORES_PER_OWNER) return true;
        }
        return false;
    }
    static void poll(SchedulerContext &context, int owner, Handshake *handshake, PollResult &result) {
        context.check_running_cores_for_completion(
            owner, handshake, result.tasks, result.physical, result.progress, result.released.data(),
            result.release_count
        );
    }
};

class CompletionBatch : public testing::Test {
protected:
    std::unique_ptr<SchedulerContext> context = std::make_unique<SchedulerContext>();
    std::unique_ptr<RuntimeContext> runtime = std::make_unique<RuntimeContext>();
    std::unique_ptr<SharedMemoryHeader> header = std::make_unique<SharedMemoryHeader>();
    std::unique_ptr<AICoreCompletionMailbox> mailbox = std::make_unique<AICoreCompletionMailbox>();
    DevicePhaseBufferStorage<PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH> phases{};
    std::array<uint32_t, 18> cond{};
    std::array<Handshake, 18> handshake{};

    void SetUp() override {
        observers = {};
        runtime->aicore_mailbox = mailbox.get();
        for (auto &ring : header->rings) {
            ring.task_window_size = 32;
            ring.task_window_mask = 31;
        }
        for (auto &size : runtime->prebuilt_layout.sizing.task_window_sizes)
            size = 32;
        CompletionBatchTestPeer::bind(*context, *runtime, *header);
        set_platform_phase_base(reinterpret_cast<uint64_t>(&phases));
    }
    void TearDown() override {
        set_platform_phase_base(0);
        observers = {};
    }
    void running(int id, Task &task, int token = 2) {
        auto &core = CompletionBatchTestPeer::core(*context, id);
        core.running_slot_state = &task.slot;
        core.running_reg_task_id = token;
        core.running_subslot = SubtaskSlot::AIC;
        core.pending_reg_task_id = AICPU_TASK_INVALID;
        core.cond_ptr = &cond[id];
        cond[id] = static_cast<uint32_t>(MAKE_FIN_VALUE(token));
        CompletionBatchTestPeer::running(*context, id, task.slot.total_required_subtasks > 1);
    }
    void pending(int id, Task &task, int token = 3) {
        auto &core = CompletionBatchTestPeer::core(*context, id);
        core.pending_slot_state = &task.slot;
        core.pending_reg_task_id = token;
        core.pending_subslot = SubtaskSlot::AIC;
        CompletionBatchTestPeer::pending(*context, id, task.slot.total_required_subtasks > 1);
    }
    DeferredCompletionSlab &slab(int id, int token = 2) { return CompletionBatchTestPeer::slab(*context, id, token); }
    void deferred(int id, int token = 2) {
        auto &s = slab(id, token);
        s.count = 1;
        s.entries[0].addr = 1000 + id * 2 + (token & 1);
        s.entries[0].expected_value = 7;
        s.entries[0].completion_type = COMPLETION_TYPE_COUNTER;
    }
    PollResult poll(int owner = 0) {
        PollResult result;
        CompletionBatchTestPeer::poll(*context, owner, handshake.data(), result);
        return result;
    }
    std::vector<AICoreCompletionMsgView> messages() {
        std::vector<AICoreCompletionMsgView> result;
        AICoreCompletionMsgView message{};
        while (mailbox->try_pop(message))
            result.push_back(message);
        return result;
    }
};

TEST_F(CompletionBatch, NonAdjacentEventsKeepCreditsUntilTheLastRegistration) {
    Task grouped(2), independent(1, 1);
    running(0, grouped);
    running(1, independent);
    running(2, grouped);
    deferred(0);
    deferred(2);
    int registrations = 0;
    observers.invalidate = [&](const void *) {
        EXPECT_EQ(grouped.slot.completed_subtasks.load(), 0);
        EXPECT_FALSE(grouped.slot.is_completion_flag_set());
        ++registrations;
    };
    auto result = poll();
    EXPECT_EQ(registrations, 2);
    EXPECT_EQ(grouped.slot.completed_subtasks.load(), 2);
    EXPECT_EQ(result.tasks, 1);
    ASSERT_EQ(result.release_count, 1);
    EXPECT_EQ(result.released[0], &independent.slot);
    auto received = messages();
    ASSERT_EQ(received.size(), 3U);
    EXPECT_EQ(received[0].addr, 1000U);
    EXPECT_EQ(received[1].addr, 1004U);
    EXPECT_EQ(received[2].kind, MSG_KIND_TASK_NORMAL_DONE);
    EXPECT_EQ(received[2].addr, reinterpret_cast<uint64_t>(&grouped.slot));
    EXPECT_FALSE(grouped.slot.is_completion_flag_set());
}

TEST_F(CompletionBatch, PendingFinCompletesBothSlotsInPendingThenRunningOrder) {
    Task task(2);
    running(0, task);
    pending(0, task);
    cond[0] = static_cast<uint32_t>(MAKE_FIN_VALUE(3));
    deferred(0, 2);
    deferred(0, 3);
    auto result = poll();
    EXPECT_EQ(result.physical, 2);
    EXPECT_EQ(task.slot.completed_subtasks.load(), 2);
    auto received = messages();
    ASSERT_EQ(received.size(), 3U);
    EXPECT_EQ(received[0].addr, 1001U);
    EXPECT_EQ(received[1].addr, 1000U);
    EXPECT_EQ(received[2].kind, MSG_KIND_TASK_NORMAL_DONE);
    EXPECT_EQ(CompletionBatchTestPeer::core(*context, 0).running_slot_state, nullptr);
    EXPECT_EQ(CompletionBatchTestPeer::core(*context, 0).pending_slot_state, nullptr);
    EXPECT_TRUE(CompletionBatchTestPeer::free_slot(*context, 0));
}

TEST_F(CompletionBatch, NinthObservationStartsAfterTheFirstEightCreditsPublish) {
    Task task(9);
    for (int i = 0; i < 9; ++i)
        running(i, task);
    int reads = 0;
    observers.read = [&](const volatile uint32_t *) {
        EXPECT_EQ(task.slot.completed_subtasks.load(), reads < 8 ? 0 : 8);
        EXPECT_FALSE(task.slot.is_completion_flag_set());
        ++reads;
    };
    auto result = poll();
    EXPECT_EQ(reads, 9);
    EXPECT_EQ(result.physical, 9);
    EXPECT_EQ(result.tasks, 1);
    EXPECT_EQ(result.release_count, 1);
    EXPECT_EQ(task.slot.completed_subtasks.load(), 9);
#if SIMPLER_DFX
    ASSERT_EQ(observers.finishes.size(), 9U);
    for (int i = 1; i < 8; ++i)
        EXPECT_EQ(observers.finishes[i], observers.finishes[0]);
    EXPECT_GT(observers.finishes[8], observers.finishes[0]);
#endif
}

TEST_F(CompletionBatch, FullBatchCanRetireSixteenPhysicalCompletions) {
    Task task(16);
    for (int i = 0; i < 8; ++i) {
        running(i, task);
        pending(i, task);
        cond[i] = static_cast<uint32_t>(MAKE_FIN_VALUE(3));
    }
    auto result = poll();
    EXPECT_EQ(result.physical, 16);
    EXPECT_EQ(result.tasks, 1);
    EXPECT_EQ(result.release_count, 1);
    EXPECT_EQ(task.slot.completed_subtasks.load(), 16);
    for (int i = 0; i < 8; ++i)
        EXPECT_TRUE(CompletionBatchTestPeer::free_slot(*context, i));
}

TEST_F(CompletionBatch, FullBatchRetiresSixteenDistinctSingletonTasksInOrder) {
    std::vector<std::unique_ptr<Task>> tasks;
    for (int i = 0; i < 16; ++i)
        tasks.push_back(std::make_unique<Task>(1, i));
    for (int i = 0; i < 8; ++i) {
        running(i, *tasks[2 * i]);
        pending(i, *tasks[2 * i + 1]);
        cond[i] = static_cast<uint32_t>(MAKE_FIN_VALUE(3));
    }
    int reads = 0;
    observers.read = [&](const volatile uint32_t *) {
        for (int i = 0; i < 2 * reads; ++i)
            EXPECT_TRUE(tasks[i]->slot.is_completion_flag_set());
        ++reads;
    };
    auto result = poll();
    EXPECT_EQ(reads, 8);
    EXPECT_EQ(result.physical, 16);
    EXPECT_EQ(result.tasks, 16);
    ASSERT_EQ(result.release_count, 16);
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(result.released[2 * i], &tasks[2 * i + 1]->slot);
        EXPECT_EQ(result.released[2 * i + 1], &tasks[2 * i]->slot);
        EXPECT_TRUE(CompletionBatchTestPeer::free_slot(*context, i));
    }
    for (const auto &task : tasks) {
        EXPECT_EQ(task->slot.completed_subtasks.load(), 1);
        EXPECT_TRUE(task->slot.is_completion_flag_set());
    }
#if SIMPLER_DFX
    ASSERT_EQ(observers.finishes.size(), 16U);
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(observers.finishes[2 * i], observers.finishes[2 * i + 1]);
        if (i > 0) EXPECT_GT(observers.finishes[2 * i], observers.finishes[2 * i - 1]);
    }
#endif
}

TEST_F(CompletionBatch, FullBatchTracksSixteenDistinctGroupableTasks) {
    std::vector<std::unique_ptr<Task>> tasks;
    for (int i = 0; i < 16; ++i)
        tasks.push_back(std::make_unique<Task>(2, i));
    for (int i = 0; i < 8; ++i) {
        running(i, *tasks[2 * i]);
        pending(i, *tasks[2 * i + 1]);
        cond[i] = static_cast<uint32_t>(MAKE_FIN_VALUE(3));
    }
    auto result = poll();
    EXPECT_EQ(result.physical, 16);
    EXPECT_EQ(result.tasks, 0);
    EXPECT_EQ(result.release_count, 0);
    for (const auto &task : tasks) {
        EXPECT_EQ(task->slot.completed_subtasks.load(), 1);
        EXPECT_FALSE(task->slot.is_completion_flag_set());
    }
#if SIMPLER_DFX
    ASSERT_EQ(observers.finishes.size(), 16U);
    for (uint64_t finish : observers.finishes)
        EXPECT_EQ(finish, observers.finishes[0]);
#endif
}

TEST_F(CompletionBatch, GroupablePendingSlotsSelectBatchingWithSingletonRunningSlots) {
    Task grouped(2), first(1, 1), second(1, 2);
    running(0, first);
    pending(0, grouped);
    running(1, second);
    pending(1, grouped);
    cond[0] = static_cast<uint32_t>(MAKE_FIN_VALUE(3));
    cond[1] = static_cast<uint32_t>(MAKE_FIN_VALUE(3));
    deferred(0, 3);
    deferred(1, 3);
    int registrations = 0;
    observers.invalidate = [&](const void *) {
        EXPECT_EQ(grouped.slot.completed_subtasks.load(), 0);
        ++registrations;
    };
    auto result = poll();
    EXPECT_EQ(registrations, 2);
    EXPECT_EQ(grouped.slot.completed_subtasks.load(), 2);
    EXPECT_EQ(result.physical, 4);
    EXPECT_EQ(result.tasks, 2);
    EXPECT_EQ(result.release_count, 2);
}

TEST_F(CompletionBatch, PendingFinGroupsRunningAndPendingSlotsIndependently) {
    Task grouped(2), first(1, 1), second(1, 2);
    running(0, first);
    pending(0, grouped);
    running(1, grouped);
    pending(1, second);
    cond[0] = static_cast<uint32_t>(MAKE_FIN_VALUE(3));
    cond[1] = static_cast<uint32_t>(MAKE_FIN_VALUE(3));
    deferred(0, 3);
    deferred(1, 2);
    int registrations = 0;
    observers.invalidate = [&](const void *) {
        EXPECT_EQ(grouped.slot.completed_subtasks.load(), 0);
        ++registrations;
    };
    auto result = poll();
    EXPECT_EQ(registrations, 2);
    EXPECT_EQ(grouped.slot.completed_subtasks.load(), 2);
    EXPECT_EQ(result.tasks, 2);
    EXPECT_EQ(result.physical, 4);
    ASSERT_EQ(result.release_count, 2);
    EXPECT_EQ(result.released[0], &first.slot);
    EXPECT_EQ(result.released[1], &second.slot);
    auto received = messages();
    ASSERT_EQ(received.size(), 3U);
    EXPECT_EQ(received[0].kind, MSG_KIND_CONDITION);
    EXPECT_EQ(received[1].kind, MSG_KIND_CONDITION);
    EXPECT_EQ(received[2].kind, MSG_KIND_TASK_NORMAL_DONE);
    for (int i = 0; i < 2; ++i) {
        EXPECT_FALSE(CompletionBatchTestPeer::running_groupable(*context, i));
        EXPECT_FALSE(CompletionBatchTestPeer::pending_groupable(*context, i));
    }
}

TEST_F(CompletionBatch, PendingAckTransfersClassificationAndRunningAckPreservesIt) {
    Task grouped(2), first(1, 1), second(1, 2);
    running(0, first);
    pending(0, grouped);
    running(1, grouped);
    pending(1, second);
    cond[0] = 3;
    cond[1] = 3;
    auto promoted = poll();
    EXPECT_EQ(promoted.tasks, 1);
    EXPECT_EQ(promoted.physical, 2);
    EXPECT_EQ(grouped.slot.completed_subtasks.load(), 1);
    EXPECT_TRUE(CompletionBatchTestPeer::running_groupable(*context, 0));
    EXPECT_FALSE(CompletionBatchTestPeer::pending_groupable(*context, 0));
    EXPECT_FALSE(CompletionBatchTestPeer::running_groupable(*context, 1));
    EXPECT_FALSE(CompletionBatchTestPeer::pending_groupable(*context, 1));
    EXPECT_TRUE(CompletionBatchTestPeer::free_slot(*context, 0));
    cond[1] = static_cast<uint32_t>(MAKE_FIN_VALUE(3));
    auto acknowledged = poll();
    EXPECT_EQ(acknowledged.tasks, 1);
    EXPECT_TRUE(CompletionBatchTestPeer::running_groupable(*context, 0));
    EXPECT_EQ(grouped.slot.completed_subtasks.load(), 1);
    cond[0] = static_cast<uint32_t>(MAKE_FIN_VALUE(3));
    auto finished = poll();
    EXPECT_EQ(finished.tasks, 1);
    EXPECT_EQ(grouped.slot.completed_subtasks.load(), 2);
    EXPECT_FALSE(CompletionBatchTestPeer::running_groupable(*context, 0));
}

TEST_F(CompletionBatch, EqualSlotIndicesInDifferentRingsRemainIndependent) {
    Task first(2, 0, 0), second(2, 0, 1);
    running(0, first);
    running(1, second);
    running(2, first);
    running(3, second);
    auto result = poll();
    EXPECT_EQ(result.tasks, 2);
    EXPECT_EQ(first.slot.completed_subtasks.load(), 2);
    EXPECT_EQ(second.slot.completed_subtasks.load(), 2);
    ASSERT_EQ(result.release_count, 2);
    EXPECT_EQ(result.released[0], &first.slot);
    EXPECT_EQ(result.released[1], &second.slot);
}

TEST_F(CompletionBatch, ColdPreparationUsesValidatedLayoutBeforeHeaderInitialization) {
    for (auto &ring : header->rings) {
        ring.task_window_size = 0;
        ring.task_window_mask = 0;
    }
    for (auto &size : runtime->prebuilt_layout.sizing.task_window_sizes)
        size = 64;
    // Fresh storage must use the validated layout, not the unwritten GM header.
    context = std::make_unique<SchedulerContext>();
    CompletionBatchTestPeer::bind(*context, *runtime, *header);
    Task first(2, 1), second(2, 33);
    running(0, first);
    running(1, second);
    running(2, first);
    running(3, second);
    auto result = poll();
    EXPECT_EQ(result.tasks, 2);
    EXPECT_EQ(first.slot.completed_subtasks.load(), 2);
    EXPECT_EQ(second.slot.completed_subtasks.load(), 2);
    ASSERT_EQ(result.release_count, 2);
    EXPECT_EQ(result.released[0], &first.slot);
    EXPECT_EQ(result.released[1], &second.slot);
}

TEST_F(CompletionBatch, ReusedRingSlotDoesNotKeepThePreviousBatchGroup) {
    Task reused(2), other(2, 1);
    running(0, reused);
    running(1, reused);
    ASSERT_EQ(poll().tasks, 1);
    ASSERT_EQ(reused.slot.completed_subtasks.load(), 2);
    reused.slot.reset_for_reuse();
    reused.slot.task_state.store(CHIP_TASK_PENDING);
    reused.descriptor.task_id = TaskId::make(0, 32);
    // Same slot, next generation, and a different first group in this batch.
    running(0, other);
    running(1, reused);
    running(2, other);
    running(3, reused);
    auto result = poll();
    EXPECT_EQ(result.tasks, 2);
    EXPECT_EQ(other.slot.completed_subtasks.load(), 2);
    EXPECT_EQ(reused.slot.completed_subtasks.load(), 2);
    ASSERT_EQ(result.release_count, 2);
    EXPECT_EQ(result.released[0], &other.slot);
    EXPECT_EQ(result.released[1], &reused.slot);
}

class CompletionBatchError : public CompletionBatch,
                             public testing::WithParamInterface<int32_t> {};

TEST_P(CompletionBatchError, ErrorFlushesEarlierCreditsAndDoesNotCountTheFailedEvent) {
    Task task(3), independent(1, 1);
    for (int i = 0; i < 3; ++i)
        running(i, task);
    running(3, independent);
    if (GetParam() == SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED) {
        slab(1).count = MAX_COMPLETIONS_PER_TASK + 1;
    } else {
        slab(1).error_code = GetParam();
    }
    deferred(2);
    observers.invalidate = [&](const void *) {
        EXPECT_EQ(task.slot.completed_subtasks.load(), 1);
        EXPECT_EQ(header->sched_error_code.load(), GetParam());
    };
    auto result = poll();
    EXPECT_TRUE(context->is_completed());
    EXPECT_EQ(header->sched_error_code.load(), GetParam());
    EXPECT_EQ(task.slot.completed_subtasks.load(), 2);
    EXPECT_FALSE(task.slot.is_completion_flag_set());
    EXPECT_TRUE(independent.slot.is_completion_flag_set());
    EXPECT_EQ(result.tasks, 1);
    auto received = messages();
    ASSERT_EQ(received.size(), 1U);
    EXPECT_EQ(received[0].kind, MSG_KIND_CONDITION);
}

INSTANTIATE_TEST_SUITE_P(
    SlabFailures, CompletionBatchError,
    testing::Values(
        SIMPLER_ERROR_ASYNC_COMPLETION_INVALID, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW,
        SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED
    )
);

TEST_F(CompletionBatch, LastEventErrorPublishesHealthyCreditsWithoutRetiringTheTask) {
    Task task(3);
    for (int i = 0; i < 3; ++i)
        running(i, task);
    slab(2).error_code = SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
    auto result = poll();
    EXPECT_EQ(task.slot.completed_subtasks.load(), 2);
    EXPECT_FALSE(task.slot.is_completion_flag_set());
    EXPECT_EQ(result.tasks, 0);
    EXPECT_EQ(result.release_count, 0);
    EXPECT_EQ(header->sched_error_code.load(), SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST_F(CompletionBatch, OtherOwnerCannotRetireWhileRegistrationIsInFlight) {
    Task task(3);
    running(0, task);
    running(1, task);
    running(9, task);
    deferred(0);
    deferred(1);
    std::atomic<bool> registering{false}, release{false};
    PollResult first;
    std::thread owner([&] {
        int calls = 0;
        observers.invalidate = [&](const void *) {
            if (++calls == 2) {
                registering.store(true, std::memory_order_release);
                while (!release.load(std::memory_order_acquire)) {}
            }
        };
        CompletionBatchTestPeer::poll(*context, 0, handshake.data(), first);
    });
    while (!registering.load(std::memory_order_acquire)) {}
    EXPECT_EQ(task.slot.completed_subtasks.load(), 0);
    auto second = poll(1);
    EXPECT_EQ(task.slot.completed_subtasks.load(), 1);
    EXPECT_EQ(second.tasks, 0);
    EXPECT_FALSE(task.slot.is_completion_flag_set());
    release.store(true, std::memory_order_release);
    owner.join();
    EXPECT_EQ(task.slot.completed_subtasks.load(), 3);
    EXPECT_EQ(first.tasks, 0);
    auto received = messages();
    ASSERT_EQ(received.size(), 3U);
    EXPECT_EQ(received[0].kind, MSG_KIND_CONDITION);
    EXPECT_EQ(received[1].kind, MSG_KIND_CONDITION);
    EXPECT_EQ(received[2].kind, MSG_KIND_TASK_NORMAL_DONE);
}

TEST_F(CompletionBatch, ConcurrentOwnersHaveExactlyOneTerminalCallback) {
    Task task(18);
    for (int i = 0; i < 18; ++i)
        running(i, task);
    std::atomic<bool> go{false};
    std::array<PollResult, 2> results;
    std::array<std::thread, 2> owners;
    for (int i = 0; i < 2; ++i) {
        owners[i] = std::thread([&, i] {
            while (!go.load(std::memory_order_acquire)) {}
            CompletionBatchTestPeer::poll(*context, i, handshake.data(), results[i]);
        });
    }
    go.store(true, std::memory_order_release);
    for (auto &owner : owners)
        owner.join();
    EXPECT_EQ(task.slot.completed_subtasks.load(), 18);
    EXPECT_EQ(results[0].tasks + results[1].tasks, 1);
    EXPECT_EQ(results[0].release_count + results[1].release_count, 1);
    EXPECT_TRUE(task.slot.is_completion_flag_set());
}

TEST_F(CompletionBatch, CallbackUngatingKeepsAcquiredTransitionsAndFinishTimes) {
    Task producer(1), running_task(2, 1), gated(1, 2);
    running(0, producer);
    running(1, running_task);
    pending(1, gated);
    running(2, running_task);
    producer.slot.task_attrs.set_timing_slot(0);
    running_task.slot.task_attrs.set_timing_slot(1);
    gated.payload.early_dispatch_state.store(EARLY_DISPATCH_STAGING);
    gated.slot.fanin_count = 1;
    gated.slot.next_block_idx.store(1);
    DepListEntry edge{};
    edge.slot_state = &gated.slot;
    producer.slot.fanout_head = &edge;
    uint64_t doorbell = 0;
    gated.payload.staged_core_mask[0].store(1ULL << 1);
    runtime->scheduler.early_dispatch_doorbell_table[1].addr = reinterpret_cast<uint64_t>(&doorbell);
    runtime->scheduler.early_dispatch_doorbell_table[1].token = 3;
    int callbacks = 0;
    uint64_t callback_clock = 0;
    observers.doorbell = [&] {
        ++callbacks;
        EXPECT_GT(phases.tail[0].finish_cycle, 0U);
        EXPECT_GT(phases.tail[1].finish_cycle, 0U);
        callback_clock = ++observers.clock;
        observers.clock += 1000;  // Make callback work visible to a late timestamp.
    };
    auto result = poll();
    EXPECT_EQ(callbacks, 1);
    EXPECT_EQ(doorbell, (uint64_t{3} << 32) | 3);
    EXPECT_EQ(result.tasks, 2);
    EXPECT_EQ(result.physical, 3);
    EXPECT_EQ(gated.payload.early_dispatch_state.load(), EARLY_DISPATCH_DISPATCHED);
    EXPECT_EQ(CompletionBatchTestPeer::core(*context, 1).running_slot_state, &gated.slot);
    EXPECT_EQ(CompletionBatchTestPeer::core(*context, 1).pending_slot_state, nullptr);
    EXPECT_FALSE(CompletionBatchTestPeer::free_slot(*context, 1));
    EXPECT_FALSE(CompletionBatchTestPeer::running_groupable(*context, 1));
    EXPECT_FALSE(CompletionBatchTestPeer::pending_groupable(*context, 1));
    EXPECT_LT(phases.tail[1].finish_cycle, callback_clock);
#if SIMPLER_DFX
    ASSERT_EQ(observers.finishes.size(), 3U);
    EXPECT_EQ(observers.finishes[0], observers.finishes[1]);
    EXPECT_EQ(observers.finishes[1], observers.finishes[2]);
    EXPECT_LT(observers.finishes[1], callback_clock);
#endif
    cond[1] = 3;  // ACK releases occupancy; the promoted task has not finished.
    auto ack = poll();
    EXPECT_EQ(ack.physical, 0);
    EXPECT_EQ(gated.slot.completed_subtasks.load(), 0);
    EXPECT_TRUE(CompletionBatchTestPeer::free_slot(*context, 1));
}
