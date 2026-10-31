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
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "callable.h"
#include "orchestrator.h"
#include "runtime_core.h"
#include "scheduler/scheduler_context.h"
#include "tensormap_and_ringbuffer/blocking_wait.h"

class MixAdmissionTestPeer {
public:
    static void bind(
        SchedulerContext &context, SchedulerState &scheduler, RuntimeContext &runtime, AICoreCompletionMailbox &mailbox,
        uint64_t *callables, uint32_t *conditions, uint64_t *registers, int clusters
    ) {
        runtime.aicore_mailbox = &mailbox;
        context.rt_ = &runtime;
        context.sched_ = &scheduler;
        context.active_sched_threads_ = 1;
        context.aicpu_thread_num_ = 1;
        context.cores_total_num_ = clusters * 3;
        context.aic_count_ = clusters;
        context.aiv_count_ = clusters * 2;
        context.func_id_to_addr_ = callables;
        context.func_id_to_addr_count_ = 1;
        auto &tracker = context.core_trackers_[0];
        tracker.init(clusters);
        for (int cluster = 0; cluster < clusters; ++cluster)
            tracker.set_cluster(cluster, cluster * 3, cluster * 3 + 1, cluster * 3 + 2);
        for (int core = 0; core < 6; ++core) {
            auto &state = context.core_exec_states_[core];
            state = {};
            state.reg_addr = reinterpret_cast<uint64_t>(registers + core * (0x6000 / sizeof(uint64_t)));
            state.cond_ptr = conditions + core;
            state.running_reg_task_id = AICPU_TASK_INVALID;
            state.pending_reg_task_id = AICPU_TASK_INVALID;
            for (auto &slab : context.deferred_slab_per_core_[core]) {
                slab.count = 0;
                slab.error_code = SIMPLER_ERROR_NONE;
            }
        }
        mailbox.head.store(0, std::memory_order_relaxed);
        mailbox.tail.store(0, std::memory_order_relaxed);
        for (auto &entry : mailbox.entries)
            entry.seq.store(0, std::memory_order_relaxed);
    }

    static void dispatch(SchedulerContext &context) {
        if (context.drain_state_.sync_start_pending.load(std::memory_order_acquire) != 0) {
            context.handle_drain_mode(0);
            return;
        }
        bool progress = false, pushed = false;
        auto &tracker = context.core_trackers_[0];
        context.dispatch_ready_tasks(0, tracker, false, progress, pushed);
        (void)context.try_early_dispatch(0, tracker, false, progress, pushed);
    }

    static int poll(SchedulerContext &context) {
        int completed = 0, thread_completed = 0, release_count = 0;
        bool progress = false;
        ChipTaskSlotState *releases[256]{};
        context.check_running_cores_for_completion(
            0, nullptr, completed, thread_completed, progress, releases, release_count
        );
        release(context, releases, release_count);
        return completed;
    }

    static AsyncPollResult poll_async(SchedulerContext &context) {
        ChipTaskSlotState *releases[256]{};
        int release_count = 0;
        const auto result = context.sched_->async_wait_list.poll_and_complete<false>(
            context.rt_->aicore_mailbox, context.sched_, releases, release_count, 256
#if SIMPLER_SCHED_PROFILING
            ,
            0
#endif
        );
        release(context, releases, release_count);
        return result;
    }

    static CoreExecState &core(SchedulerContext &context, int core) { return context.core_exec_states_[core]; }

    static uint64_t running_gate(SchedulerContext &context, int core) {
        const auto token = context.core_exec_states_[core].running_reg_task_id;
        return token == AICPU_TASK_INVALID ? 0 : context.payload_per_core_[core][token & 1].src_payload;
    }

    static void register_counter(SchedulerContext &context, int core, volatile uint32_t *counter) {
        const int token = context.core_exec_states_[core].running_reg_task_id;
        ASSERT_NE(token, AICPU_TASK_INVALID);
        auto &slab = context.deferred_slab_per_core_[core][token & 1];
        slab.entries[0].addr = reinterpret_cast<uint64_t>(counter);
        slab.entries[0].backend_cookie = 0;
        slab.entries[0].expected_value = 1;
        slab.entries[0].engine = COMPLETION_ENGINE_CCU;
        slab.entries[0].completion_type = COMPLETION_TYPE_COUNTER;
        slab.count = 1;
    }

    static int free_aiv_slots(SchedulerContext &context) {
        const auto &tracker = context.core_trackers_[0];
        return tracker.get_dispatchable_cores(ResourceShape::AIV, CoreTracker::DispatchPhase::IDLE).count() +
               tracker.get_dispatchable_cores(ResourceShape::AIV, CoreTracker::DispatchPhase::PENDING).count();
    }

    static int drain_pending(SchedulerContext &context) {
        return context.drain_state_.sync_start_pending.load(std::memory_order_acquire);
    }

    static ChipTaskSlotState *drain_task(SchedulerContext &context) {
        return context.drain_state_.pending_task.load(std::memory_order_acquire);
    }

private:
    static void release(SchedulerContext &context, ChipTaskSlotState **releases, int count) {
        for (int i = 0; i < count; ++i) {
#if SIMPLER_SCHED_PROFILING
            (void)context.sched_->on_task_release(*releases[i], 0);
#else
            context.sched_->on_task_release(*releases[i]);
#endif
        }
    }
};

namespace {

enum class KernelKind { Aic, Aiv, Mix };
enum class WaitAdmissionPath { LegacyEarly, Declared, LegacyConservative };

MixedKernels kernels(KernelKind kind) {
    MixedKernels result;
    if (kind != KernelKind::Aiv) result.aic_kernel_id = 0;
    if (kind != KernelKind::Aic) result.aiv0_kernel_id = 0;
    if (kind == KernelKind::Mix) result.aiv1_kernel_id = 0;
    return result;
}

// ACK/FIN, kernel-side registration and notifications are controlled by the CPU test.
class BlockingWaitAdmission : public testing::Test {
protected:
    static constexpr int kWindow = 16;
    static constexpr uint64_t kHeapBytes = 4096;
    DeviceArena shared_arena;
    DeviceArena runtime_arena;
    SharedMemoryHandle shared{};
    std::unique_ptr<OrchestratorState> orch = std::make_unique<OrchestratorState>();
    std::unique_ptr<SchedulerState> scheduler = std::make_unique<SchedulerState>();
    std::unique_ptr<SchedulerContext> context = std::make_unique<SchedulerContext>();
    std::unique_ptr<RuntimeContext> runtime = std::make_unique<RuntimeContext>();
    std::unique_ptr<AICoreCompletionMailbox> mailbox = std::make_unique<AICoreCompletionMailbox>();
    std::vector<char> heap = std::vector<char>(kHeapBytes * CHIP_MAX_RING_DEPTH);
    std::vector<uint8_t> callable = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, nullptr, 0);
    uint64_t callables[1]{};
    std::array<uint32_t, 6> conditions{};
    std::vector<uint64_t> registers = std::vector<uint64_t>(6 * (0x6000 / sizeof(uint64_t)));
    int core_count = 3;

    void SetUp() override {
        const auto shared_bytes = SharedMemoryHandle::calculate_size(kWindow);
        const auto shared_offset = shared_arena.reserve(shared_bytes, CHIP_ALIGN_SIZE);
        ASSERT_NE(shared_arena.commit(), nullptr);
        void *base = shared_arena.region_ptr(shared_offset);
        std::memset(base, 0, static_cast<size_t>(shared_bytes));
        ASSERT_TRUE(shared.init(base, shared_bytes, kWindow, kHeapBytes));
        int32_t windows[CHIP_MAX_RING_DEPTH];
        for (auto &window : windows)
            window = kWindow;
        const auto orch_layout = OrchestratorState::reserve_layout(runtime_arena, windows, 128);
        const auto sched_layout = SchedulerState::reserve_layout(runtime_arena, 128);
        ASSERT_NE(runtime_arena.commit(), nullptr);
        ASSERT_TRUE(
            orch->init_data_from_layout(orch_layout, runtime_arena, shared.sm_base, heap.data(), kHeapBytes, kWindow)
        );
        ASSERT_TRUE(scheduler->init_data_from_layout(sched_layout, runtime_arena, shared.sm_base));
        scheduler->wire_arena_pointers(sched_layout, runtime_arena);
        orch->wire_arena_pointers(orch_layout, runtime_arena, scheduler.get());
        reinterpret_cast<CoreCallable *>(callable.data())->set_resolved_addr(1);
        callables[0] = reinterpret_cast<uint64_t>(callable.data());
        configure_clusters(1);
        orch->begin_scope();
    }

    void TearDown() override {
        orch->destroy();
        scheduler->destroy();
    }

    void configure_clusters(int clusters) {
        core_count = clusters * 3;
        orch->total_cluster_count = clusters;
        orch->total_aiv_count = clusters * 2;
        MixAdmissionTestPeer::bind(
            *context, *scheduler, *runtime, *mailbox, callables, conditions.data(), registers.data(), clusters
        );
    }

    TaskId submit(
        KernelKind kind, int16_t blocks = 1, bool early = false, const TaskId *dependency = nullptr,
        const BlockingWait *wait = nullptr, bool sync_start = false
    ) {
        CoreTaskArgs args;
        args.launch_spec.set_block_num(blocks);
        args.launch_spec.set_require_sync_start(sync_start);
        args.set_allow_early_resolve(early);
        const DepFlags flags = DEP_WAIT;
        if (dependency != nullptr) args.set_dependencies_with_kinds(dependency, &flags, 1);
        const auto result = wait == nullptr ? orch->submit_task(kernels(kind), args) :
                                              orch->submit_task_with_wait(kernels(kind), args, *wait);
        EXPECT_FALSE(orch->fatal);
        EXPECT_TRUE(result.task_id().is_valid());
        return result.task_id();
    }

    ChipTaskSlotState *slot(TaskId id) {
        return &shared.header->rings[id.ring()].get_slot_state_by_task_id(id.local_id());
    }

    CoreExecState &core(int id) { return MixAdmissionTestPeer::core(*context, id); }

    void dispatch() { MixAdmissionTestPeer::dispatch(*context); }

    int poll() { return MixAdmissionTestPeer::poll(*context); }

    int poll_async() {
        const auto result = MixAdmissionTestPeer::poll_async(*context);
        EXPECT_EQ(result.error_code, SIMPLER_ERROR_NONE);
        return result.completed;
    }

    void ack(int core_id) {
        ASSERT_NE(core(core_id).running_reg_task_id, AICPU_TASK_INVALID);
        conditions[core_id] = MAKE_ACK_VALUE(core(core_id).running_reg_task_id);
        EXPECT_EQ(poll(), 0);
    }

    void finish(TaskId task) {
        int occupied = 0;
        for (int id = 0; id < core_count; ++id) {
            if (core(id).running_slot_state != slot(task)) continue;
            ASSERT_NE(core(id).running_reg_task_id, AICPU_TASK_INVALID);
            ASSERT_EQ(core(id).pending_slot_state, nullptr);
            conditions[id] = MAKE_ACK_VALUE(core(id).running_reg_task_id);
            ++occupied;
        }
        ASSERT_GT(occupied, 0);
        ASSERT_EQ(poll(), 0);
        for (int id = 0; id < core_count; ++id) {
            if (core(id).running_slot_state == slot(task))
                conditions[id] = MAKE_FIN_VALUE(core(id).running_reg_task_id);
        }
        ASSERT_EQ(poll(), 1);
        EXPECT_GE(slot(task)->task_state.load(), CHIP_TASK_COMPLETED);
    }

    void expect_no_occupancy(TaskId task) {
        auto *state = slot(task);
        for (int id = 0; id < core_count; ++id) {
            EXPECT_NE(core(id).running_slot_state, state);
            EXPECT_NE(core(id).pending_slot_state, state);
        }
        EXPECT_EQ(state->next_block_idx.load(), 0);
        EXPECT_EQ(state->payload->published_block_count.load(), 0);
        EXPECT_EQ(state->payload->early_dispatch_state.load(), EARLY_DISPATCH_NONE);
        EXPECT_EQ(state->payload->early_sync_drain_state.load(), EARLY_SYNC_DRAIN_NONE);
        for (auto &mask : state->payload->staged_core_mask)
            EXPECT_EQ(mask.load(), 0U);
        for (int shape = 0; shape < NUM_RESOURCE_SHAPES; ++shape)
            EXPECT_EQ(scheduler->early_dispatch_queues[shape].size(), 0U);
        EXPECT_EQ(scheduler->early_sync_start_queue.size(), 0U);
        EXPECT_EQ(MixAdmissionTestPeer::drain_pending(*context), 0);
        EXPECT_EQ(MixAdmissionTestPeer::drain_task(*context), nullptr);
    }

    std::vector<uint64_t> allocation_snapshot() {
        std::vector<uint64_t> values{
            orch->submit_seq, static_cast<uint64_t>(orch->scope_tasks_size),
            static_cast<uint64_t>(orch->tensor_map.current_used())
        };
        for (int ring = 0; ring < CHIP_MAX_RING_DEPTH; ++ring) {
            const auto &resources = orch->rings[ring];
            values.push_back(resources.task_allocator.task_head());
            values.push_back(resources.task_allocator.heap_top());
            values.push_back(resources.task_allocator.heap_tail());
            values.push_back(resources.fanin_pool.top);
            values.push_back(scheduler->ring_sched_states[ring].dep_pool.top);
            values.push_back(shared.header->rings[ring].fc.current_task_index.load());
        }
        for (int shape = 0; shape < NUM_RESOURCE_SHAPES; ++shape) {
            values.push_back(scheduler->ready_queues[shape].size());
            values.push_back(scheduler->ready_sync_queues[shape].size());
            values.push_back(scheduler->early_dispatch_queues[shape].size());
        }
        values.push_back(scheduler->dummy_ready_queue.size());
        values.push_back(scheduler->early_sync_start_queue.size());
        return values;
    }
};

class DeferredWaitAdmission : public BlockingWaitAdmission,
                              public testing::WithParamInterface<WaitAdmissionPath> {};

TEST_P(DeferredWaitAdmission, CompletionFirstLeavesTheLateNotifierAnExecutionSlot) {
    const auto path = GetParam();
    const bool completion_first = path == WaitAdmissionPath::Declared;
    const bool producer_early = path != WaitAdmissionPath::LegacyConservative;
    const bool progress_expected = path != WaitAdmissionPath::LegacyEarly;
    alignas(64) volatile uint32_t notification = 0;
    const TaskId producer = submit(KernelKind::Aic, 1, producer_early);
    ASSERT_TRUE(producer.is_valid());
    dispatch();
    ASSERT_EQ(core(0).running_slot_state, slot(producer));
    ack(0);
    MixAdmissionTestPeer::register_counter(*context, 0, &notification);
    conditions[0] = MAKE_FIN_VALUE(core(0).running_reg_task_id);
    ASSERT_EQ(poll(), 0);
    ASSERT_EQ(core(0).running_slot_state, nullptr);
    ASSERT_EQ(poll_async(), 0);
    const auto *deferred = scheduler->async_wait_list.find_entry_by_token(producer);
    ASSERT_NE(deferred, nullptr);
    ASSERT_TRUE(deferred->normal_done);
    ASSERT_EQ(deferred->waiting_completion_count, 1);
    ASSERT_EQ(slot(producer)->completed_subtasks.load(), 1);
    ASSERT_EQ(slot(producer)->task_state.load(), CHIP_TASK_PENDING);
    ASSERT_EQ(slot(producer)->has_dispatch_propagated(), producer_early);

    const BlockingWait declaration = BlockingWait::after_producers(&producer, 1);
    const TaskId waiter = submit(KernelKind::Aiv, 2, true, &producer, completion_first ? &declaration : nullptr);
    ASSERT_TRUE(waiter.is_valid());
    ASSERT_EQ(slot(waiter)->fanin_count, 2);
    ASSERT_EQ(slot(waiter)->fanin_refcount.load(), 1);
    ASSERT_TRUE(slot(waiter)->task_attrs.allow_early_resolve());
    dispatch();

    const TaskId enabling_task = submit(KernelKind::Aic);
    ASSERT_TRUE(enabling_task.is_valid());
    const TaskId notifier = submit(KernelKind::Aiv, 1, false, &enabling_task);
    ASSERT_TRUE(notifier.is_valid());
    dispatch();
    ASSERT_EQ(core(0).running_slot_state, slot(enabling_task));
    finish(enabling_task);
    ASSERT_EQ(slot(notifier)->fanin_refcount.load(), slot(notifier)->fanin_count);
    ASSERT_EQ(slot(notifier)->next_block_idx.load(), 0);
    ASSERT_EQ(poll_async(), 0);

    if (progress_expected) {
        expect_no_occupancy(waiter);
        ASSERT_EQ(MixAdmissionTestPeer::free_aiv_slots(*context), 2);
        dispatch();
        ASSERT_EQ(core(1).running_slot_state, slot(notifier));
        ack(1);
        notification = 1;
        conditions[1] = MAKE_FIN_VALUE(core(1).running_reg_task_id);
        ASSERT_EQ(poll(), 1);
        ASSERT_EQ(poll_async(), 1);
        ASSERT_EQ(slot(notifier)->next_block_idx.load(), 1);
        ASSERT_GE(slot(producer)->task_state.load(), CHIP_TASK_COMPLETED);
        ASSERT_EQ(slot(waiter)->next_block_idx.load(), 0);
        dispatch();
    } else {
        ASSERT_EQ(slot(waiter)->payload->staged_core_mask[0].load(), 6U);
        ASSERT_EQ(slot(waiter)->next_block_idx.load(), 2);
        ASSERT_EQ(MixAdmissionTestPeer::free_aiv_slots(*context), 0);
        for (int turn = 0; turn < 8; ++turn) {
            ASSERT_EQ(poll(), 0);
            ASSERT_EQ(poll_async(), 0);
            dispatch();
            ASSERT_EQ(slot(notifier)->next_block_idx.load(), 0);
            ASSERT_EQ(notification, 0U);
            ASSERT_EQ(slot(producer)->task_state.load(), CHIP_TASK_PENDING);
            ASSERT_EQ(scheduler->ready_queues[static_cast<int>(ResourceShape::AIV)].size(), 1U);
            for (int id = 1; id <= 2; ++id) {
                ASSERT_EQ(core(id).running_slot_state, slot(waiter));
                ASSERT_NE(MixAdmissionTestPeer::running_gate(*context, id), 0U);
            }
        }
        // External notification is containment, not progress by the blocked notifier.
        notification = 1;
        ASSERT_EQ(poll_async(), 1);
        ASSERT_EQ(slot(notifier)->next_block_idx.load(), 0);
    }

    finish(waiter);
    if (!progress_expected) {
        dispatch();
        ASSERT_EQ(core(1).running_slot_state, slot(notifier));
        finish(notifier);
    }
    EXPECT_EQ(scheduler->async_wait_list.count, 0);
    EXPECT_FALSE(mailbox->has_pending());
    EXPECT_EQ(MixAdmissionTestPeer::free_aiv_slots(*context), 2);
    orch->end_scope();
}

INSTANTIATE_TEST_SUITE_P(
    LegacyAndDeclared, DeferredWaitAdmission,
    testing::Values(WaitAdmissionPath::LegacyEarly, WaitAdmissionPath::Declared, WaitAdmissionPath::LegacyConservative)
);

TEST_F(BlockingWaitAdmission, AdmittedWaiterKeepsItsSourceEarlyResolveHint) {
    const TaskId producer = submit(KernelKind::Aic, 1, true);
    ASSERT_TRUE(producer.is_valid());
    dispatch();
    ack(0);
    const BlockingWait wait = BlockingWait::after_producers(&producer, 1);
    const TaskId waiter = submit(KernelKind::Aiv, 1, true, nullptr, &wait);
    ASSERT_TRUE(waiter.is_valid());
    const TaskId successor = submit(KernelKind::Aiv, 1, false, &waiter);
    ASSERT_TRUE(successor.is_valid());
    dispatch();
    expect_no_occupancy(waiter);
    finish(producer);
    dispatch();
    ASSERT_TRUE(slot(waiter)->task_attrs.allow_early_resolve());
    ASSERT_TRUE(slot(waiter)->has_dispatch_propagated());
    ASSERT_EQ(core(1).running_slot_state, slot(waiter));
    ASSERT_EQ(core(2).running_slot_state, slot(successor));
    ASSERT_NE(MixAdmissionTestPeer::running_gate(*context, 2), 0U);
    finish(waiter);
    finish(successor);
    orch->end_scope();
}

TEST_F(BlockingWaitAdmission, NamedWaitMergesWithRetentionWithoutDuplicatePins) {
    const TaskId producer = submit(KernelKind::Aic);
    ASSERT_TRUE(producer.is_valid());
    const uint32_t references = slot(producer)->fanout_count;
    const TaskId named[] = {producer, producer};
    CoreTaskArgs args;
    const DepFlags retain = DEP_RETAIN;
    args.set_dependencies_with_kinds(&producer, &retain, 1);
    const TaskId waiter =
        orch->submit_task_with_wait(kernels(KernelKind::Aiv), args, BlockingWait::after_producers(named, 2)).task_id();
    ASSERT_TRUE(waiter.is_valid());
    ASSERT_FALSE(orch->fatal);
    EXPECT_EQ(slot(waiter)->payload->fanin_actual_count, 1);
    EXPECT_EQ(slot(waiter)->payload->fanin_wait_count, 1);
    EXPECT_EQ(slot(waiter)->fanin_count, 2);
    EXPECT_EQ(slot(producer)->fanout_count, references + 1);
    int entries = 0;
    for_each_fanin_slot_state(*slot(waiter)->payload, [&](ChipTaskSlotState *input, DepFlags flags) {
        EXPECT_EQ(input, slot(producer));
        EXPECT_TRUE(dep_has_wait(flags));
        EXPECT_TRUE(dep_has_retain(flags));
        ++entries;
    });
    EXPECT_EQ(entries, 1);
    dispatch();
    finish(producer);
    dispatch();
    finish(waiter);
    orch->end_scope();
}

TEST_F(BlockingWaitAdmission, TransitiveReductionKeepsTheTargetBlockerBoolean) {
    const TaskId producer = submit(KernelKind::Aic);
    ASSERT_TRUE(producer.is_valid());
    const TaskId intermediate = submit(KernelKind::Aic, 1, true, &producer);
    ASSERT_TRUE(intermediate.is_valid());
    const BlockingWait wait = BlockingWait::after_producers(&producer, 1);
    const TaskId waiter = submit(KernelKind::Aiv, 1, true, &intermediate, &wait);
    ASSERT_TRUE(waiter.is_valid());
    EXPECT_EQ(slot(waiter)->payload->fanin_actual_count, 2);
    EXPECT_EQ(slot(waiter)->payload->fanin_wait_count, 1);
    EXPECT_EQ(slot(waiter)->payload->early_dispatch_blocked, 1);
    EXPECT_EQ(slot(waiter)->payload->early_dispatch_target(), 2);
    bool direct_wait = false;
    for_each_fanin_slot_state(*slot(waiter)->payload, [&](ChipTaskSlotState *input, DepFlags flags) {
        if (input == slot(producer)) direct_wait = dep_has_wait(flags);
    });
    EXPECT_FALSE(direct_wait);
    dispatch();
    expect_no_occupancy(waiter);
    finish(producer);
    dispatch();
    expect_no_occupancy(waiter);
    finish(intermediate);
    dispatch();
    finish(waiter);
    orch->end_scope();
}

TEST_F(BlockingWaitAdmission, RetiredWitnessAndReusedPayloadKeepTheirDistinctContracts) {
    const TaskId producer = submit(KernelKind::Aic);
    ASSERT_TRUE(producer.is_valid());
    dispatch();
    finish(producer);
    const BlockingWait wait = BlockingWait::after_producers(&producer, 1);
    const TaskId waiter = submit(KernelKind::Aiv, 1, false, nullptr, &wait);
    ASSERT_TRUE(waiter.is_valid());
    auto *reused_slot = slot(waiter);
    ASSERT_EQ(reused_slot->payload->early_dispatch_blocked, 1);
    dispatch();
    finish(waiter);
    orch->end_scope();
    for (int task = 2; task < kWindow; ++task) {
        orch->begin_scope();
        const TaskId ordinary = submit(KernelKind::Aic);
        ASSERT_TRUE(ordinary.is_valid());
        dispatch();
        finish(ordinary);
        orch->end_scope();
    }
    ASSERT_EQ(orch->rings[0].task_allocator.task_head(), kWindow);
    ASSERT_EQ(orch->rings[0].task_allocator.task_tail(), kWindow);
    orch->begin_scope();
    const TaskId next_producer = submit(KernelKind::Aic, 1, true);
    ASSERT_TRUE(next_producer.is_valid());
    dispatch();
    ack(0);
    const TaskId ordinary = submit(KernelKind::Aiv, 1, false, &next_producer);
    ASSERT_TRUE(ordinary.is_valid());
    ASSERT_EQ(slot(ordinary), reused_slot);
    ASSERT_EQ(slot(ordinary)->payload->early_dispatch_blocked, 0);
    dispatch();
    ASSERT_EQ(core(1).running_slot_state, slot(ordinary));
    ASSERT_NE(MixAdmissionTestPeer::running_gate(*context, 1), 0U);
    finish(next_producer);
    finish(ordinary);
    orch->end_scope();

    // The old local id is below the reclaim watermark; its physical slot is reused.
    orch->begin_scope();
    const TaskId retired_waiter = submit(KernelKind::Aiv, 1, false, nullptr, &wait);
    ASSERT_TRUE(retired_waiter.is_valid());
    EXPECT_EQ(slot(retired_waiter)->payload->fanin_wait_count, 0);
    dispatch();
    finish(retired_waiter);
    orch->end_scope();
}

enum class InvalidDeclaration {
    Empty,
    Null,
    HugeCount,
    InvalidId,
    BadRing,
    NegativeId,
    Self,
    Future,
    Resident,
    Unknown
};

class InvalidWaitAdmission : public BlockingWaitAdmission,
                             public testing::WithParamInterface<InvalidDeclaration> {};

TEST_P(InvalidWaitAdmission, RefusesBeforeAllocatingOrPublishingATask) {
    TaskId named = TaskId::make(0, 0);
    BlockingWait wait = BlockingWait::after_producers(&named, 1);
    switch (GetParam()) {
    case InvalidDeclaration::Empty:
        wait.producer_count = 0;
        break;
    case InvalidDeclaration::Null:
        wait.producers = nullptr;
        break;
    case InvalidDeclaration::HugeCount:
        wait.producer_count = std::numeric_limits<uint32_t>::max();
        break;
    case InvalidDeclaration::InvalidId:
        named = TaskId::invalid();
        break;
    case InvalidDeclaration::BadRing:
        named = TaskId::make(CHIP_MAX_RING_DEPTH, 0);
        break;
    case InvalidDeclaration::NegativeId:
        named = TaskId::make(0, -1);
        break;
    case InvalidDeclaration::Self:
        break;
    case InvalidDeclaration::Future:
        named = TaskId::make(0, 7);
        break;
    case InvalidDeclaration::Resident:
        wait = BlockingWait::resident();
        break;
    case InvalidDeclaration::Unknown:
        wait.mode = static_cast<BlockingWaitMode>(99);
        break;
    }
    const uint32_t shape[] = {16};
    TensorCreateInfo output(shape, 1, DataType::UINT32);
    CoreTaskArgs args;
    args.add_output(output);
    const auto before = allocation_snapshot();
    for (int attempt = 0; attempt < 2; ++attempt) {
        const auto result = orch->submit_task_with_wait(kernels(KernelKind::Aiv), args, wait);
        EXPECT_FALSE(result.task_id().is_valid());
        EXPECT_TRUE(orch->fatal);
        EXPECT_EQ(shared.header->orch_error_code.load(), SIMPLER_ERROR_INVALID_ARGS);
        EXPECT_EQ(allocation_snapshot(), before);
        EXPECT_EQ(MixAdmissionTestPeer::free_aiv_slots(*context), 2);
    }
}

INSTANTIATE_TEST_SUITE_P(
    UnsupportedOrMalformed, InvalidWaitAdmission,
    testing::Values(
        InvalidDeclaration::Empty, InvalidDeclaration::Null, InvalidDeclaration::HugeCount,
        InvalidDeclaration::InvalidId, InvalidDeclaration::BadRing, InvalidDeclaration::NegativeId,
        InvalidDeclaration::Self, InvalidDeclaration::Future, InvalidDeclaration::Resident, InvalidDeclaration::Unknown
    )
);

struct WaitShape {
    KernelKind kind;
    bool sync_start;
};

class ShapeWaitAdmission : public BlockingWaitAdmission,
                           public testing::WithParamInterface<WaitShape> {};

TEST_P(ShapeWaitAdmission, WaitsForLogicalReadinessBeforeAnyCoreOrDrainClaim) {
    configure_clusters(2);
    const TaskId producer = submit(KernelKind::Aic, 1, true);
    ASSERT_TRUE(producer.is_valid());
    dispatch();
    ack(0);
    const BlockingWait wait = BlockingWait::after_producers(&producer, 1);
    const auto shape = GetParam();
    const TaskId waiter = submit(shape.kind, 2, true, nullptr, &wait, shape.sync_start);
    ASSERT_TRUE(waiter.is_valid());
    for (int turn = 0; turn < 4; ++turn) {
        dispatch();
        expect_no_occupancy(waiter);
    }
    finish(producer);
    ASSERT_EQ(slot(waiter)->fanin_refcount.load(), slot(waiter)->fanin_count);
    for (int turn = 0; turn < 3 && slot(waiter)->next_block_idx.load() != 2; ++turn)
        dispatch();
    ASSERT_EQ(slot(waiter)->next_block_idx.load(), 2);
    finish(waiter);
    orch->end_scope();
}

INSTANTIATE_TEST_SUITE_P(
    AllCoreShapes, ShapeWaitAdmission,
    testing::Values(
        WaitShape{KernelKind::Aic, false}, WaitShape{KernelKind::Aiv, false}, WaitShape{KernelKind::Mix, false},
        WaitShape{KernelKind::Aic, true}, WaitShape{KernelKind::Aiv, true}, WaitShape{KernelKind::Mix, true}
    )
);

}  // namespace
